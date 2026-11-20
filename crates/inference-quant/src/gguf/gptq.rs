//! GPTQ-style error feedback for GGUF types whose blocks stand alone: a matrix quantized a block of columns at a time
//! by the type's own quantizer (weighed by the Hessian's diagonal, as an imatrix would), each block's rounding error
//! carried onto the columns not yet quantized through the inverse Hessian's upper Cholesky factor `U` (`H^-1 = U^T U`):
//! the block's error `E` moves them by `-E U_bb^-1 U_b,rest`, the compensation that keeps the layer's output error
//! least in the Hessian's metric. The bytes stay the type's.

use inference_tensor::quantized::Quantizer;
use inference_tensor::{DType, Device, Result, Tensor};
use rayon::prelude::*;

use super::kernel::GgufType;

// the importance a column no calibration token reached gets
const DEAD_IMPORTANCE: f32 = 1e-9;
const QK_K: usize = 256;

/// Whether [`quantize`] feeds errors forward for `ty`: its blocks carry their own scales, or it is one of the row-scaled
/// types it encodes against a row scale fixed first.
pub fn supports(ty: GgufType) -> bool {
    super::iq_quantize::row_scaled(ty).is_some()
        || matches!(
            ty,
            GgufType::Q4_0
                | GgufType::Q4_1
                | GgufType::Q5_0
                | GgufType::Q5_1
                | GgufType::Q8_0
                | GgufType::Q2K
                | GgufType::Q3K
                | GgufType::Q4K
                | GgufType::Q5K
                | GgufType::Q6K
                | GgufType::Iq4Nl
                | GgufType::Iq4Xs
        )
}

// the inverse of an upper triangular `n x n` matrix, in f64
fn upper_inverse(u: &[f32], n: usize) -> Vec<f32> {
    let mut inv = vec![0f64; n * n];
    for c in 0..n {
        inv[c * n + c] = 1.0 / f64::from(u[c * n + c]);
        for r in (0..c).rev() {
            let s: f64 = (r + 1..=c)
                .map(|k| f64::from(u[r * n + k]) * inv[k * n + c])
                .sum();
            inv[r * n + c] = -s / f64::from(u[r * n + r]);
        }
    }
    inv.into_iter().map(|v| v as f32).collect()
}

/// A Hessian made ready for the feedback, shareable by matrices with the same input (a stack's experts): the
/// quantizer's importances (its diagonal), the columns no token reached, and `U` when feeding back.
pub struct Prepared {
    importance: Vec<f32>,
    dead: Vec<usize>,
    u: Option<Tensor>,
}

/// `hessian` (`cols x cols`, the inputs' `X^T X / n`) damped by `damp` of its mean diagonal, a column no token reached
/// given an identity row, and factored on `device` when `feedback` (without it only the diagonal weighs the
/// quantizer, the baseline the feedback is measured against).
pub fn prepare(
    hessian: &[f32],
    cols: usize,
    feedback: bool,
    damp: f64,
    device: &Device,
) -> Result<Prepared> {
    if hessian.len() != cols * cols {
        inference_tensor::bail!("a Hessian of {} for rows of {cols}", hessian.len());
    }
    let mut h = hessian.to_vec();
    let diag: Vec<f32> = (0..cols).map(|i| h[i * cols + i]).collect();
    let dead: Vec<usize> = (0..cols).filter(|&i| diag[i] <= 0.0).collect();
    for &i in &dead {
        h[i * cols + i] = 1.0;
    }
    let damp = damp * diag.iter().map(|&d| f64::from(d)).sum::<f64>() / cols as f64;
    for i in 0..cols {
        h[i * cols + i] += damp as f32;
    }
    Ok(Prepared {
        importance: diag
            .iter()
            .map(|&d| if d > 0.0 { d } else { DEAD_IMPORTANCE })
            .collect(),
        dead,
        u: feedback
            .then(|| inverse_factor(&h, cols, device))
            .transpose()?,
    })
}

// `U`, the upper Cholesky factor of the damped Hessian's inverse (`H^-1 = U^T U`)
fn inverse_factor(h: &[f32], cols: usize, device: &Device) -> Result<Tensor> {
    let hinv =
        super::super::exl3::linalg::inverse_spd(Tensor::from_slice(h, (cols, cols), device)?)?
            .ok_or_else(|| {
                inference_tensor::Error::Msg("the damped Hessian is not positive definite".into())
            })?;
    let lower = super::super::exl3::linalg::cholesky(hinv)?.ok_or_else(|| {
        inference_tensor::Error::Msg("the inverse Hessian is not positive definite".into())
    })?;
    // cholesky leaves its input in the other triangle
    let index = Tensor::arange(0u32, cols as u32, device)?;
    let mask = index
        .unsqueeze(1)?
        .broadcast_ge(&index.unsqueeze(0)?)?
        .to_dtype(DType::F32)?;
    lower.mul(&mask)?.t()?.contiguous()
}

// the columns after block `b` (of `block`) moved by its error: `rest - E U_bb^-1 U_b,rest`
fn carry(rest: &Tensor, error: &Tensor, u: &Tensor, b: usize, block: usize) -> Result<Tensor> {
    let width = rest.dim(1)?;
    let ubb = u
        .narrow(0, b * block, block)?
        .narrow(1, b * block, block)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let ubb_inv = Tensor::from_slice(&upper_inverse(&ubb, block), (block, block), rest.device())?;
    let u_rest = u
        .narrow(0, b * block, block)?
        .narrow(1, (b + 1) * block, width - block)?
        .contiguous()?;
    rest.narrow(1, block, width - block)? - error.matmul(&ubb_inv)?.matmul(&u_rest)?
}

/// `values` (rows of `cols`) as `ty` with the `prepared` Hessian, on `device`. A row-scaled type (IQ4_KS, IQ4_KSS,
/// IQ3_KS) takes its row scale from a first pass over the row and encodes a super-block at a time against it, the
/// scale refit to the row at the end.
#[allow(clippy::too_many_arguments)]
pub fn quantize(
    ty: GgufType,
    cols: usize,
    values: &[f32],
    prepared: &Prepared,
    quantizer: Quantizer,
    device: &Device,
    encode: impl Fn(GgufType, usize, &[f32], Option<&[f32]>, Quantizer) -> Result<Vec<u8>>,
    decode: impl Fn(GgufType, usize, &[u8]) -> Result<Vec<f32>>,
) -> Result<Vec<u8>> {
    quantize_on(
        ty, cols, values, prepared, quantizer, device, encode, decode, true,
    )
}

// `quantize`, its row-scaled encoding on the GPU (when `gpu_encode` and the type has kernels) or the CPU
#[allow(clippy::too_many_arguments)]
fn quantize_on(
    ty: GgufType,
    cols: usize,
    values: &[f32],
    prepared: &Prepared,
    quantizer: Quantizer,
    device: &Device,
    encode: impl Fn(GgufType, usize, &[f32], Option<&[f32]>, Quantizer) -> Result<Vec<u8>>,
    decode: impl Fn(GgufType, usize, &[u8]) -> Result<Vec<f32>>,
    gpu_encode: bool,
) -> Result<Vec<u8>> {
    let rows = values.len() / cols;
    let row_scaled = super::iq_quantize::row_scaled(ty);
    let block = if row_scaled.is_some() {
        QK_K
    } else {
        ty.block_size()
    };
    if !cols.is_multiple_of(block) || prepared.importance.len() != cols {
        inference_tensor::bail!(
            "{ty:?}: a Hessian of {} columns for rows of {cols}",
            prepared.importance.len()
        );
    }
    // a column no token reached does nothing, so its weights are zeroed
    let mut w = values.to_vec();
    for &i in &prepared.dead {
        for r in 0..rows {
            w[r * cols + i] = 0.0;
        }
    }
    let importance = &prepared.importance;
    let u = &prepared.u;
    let blocks = cols / block;
    let mut rest = Tensor::from_slice(&w, (rows, cols), device)?;
    if let (Some((header, super_bytes)), Device::Cuda(dev), true) = (row_scaled, device, gpu_encode)
        && super::quantize_cuda::fixed_supported(ty)
    {
        // the whole loop on the GPU: the row scales, each super-block's encoding and its error carried on
        let row_bytes = header + blocks * super_bytes;
        let mut out = vec![0u8; rows * row_bytes];
        let row_d = super::quantize_cuda::row_scales(dev, ty, &rest, importance)?;
        let mut sums = dev.alloc_zeros::<f32>(2 * rows)?;
        for b in 0..blocks {
            let current = rest.narrow(1, 0, block)?.contiguous()?;
            let imp = &importance[b * block..(b + 1) * block];
            let (bytes, levels) =
                super::quantize_cuda::super_block_fixed(dev, ty, &current, imp, &row_d, &mut sums)?;
            for (o, sb) in out
                .chunks_exact_mut(row_bytes)
                .zip(bytes.chunks_exact(super_bytes))
            {
                o[header + b * super_bytes..][..super_bytes].copy_from_slice(sb);
            }
            if b + 1 == blocks {
                break;
            }
            match u {
                Some(u) => {
                    let decoded = levels.broadcast_mul(&row_d.unsqueeze(1)?)?;
                    rest = carry(&rest, &(current - decoded)?, u, b, block)?;
                }
                None => rest = rest.narrow(1, block, cols - (b + 1) * block)?,
            }
        }
        let d = row_d.to_vec1::<f32>()?;
        let s = dev.clone_dtoh(&sums)?;
        for (r, o) in out.chunks_exact_mut(row_bytes).enumerate() {
            super::iq_quantize::row_scaled_header(
                ty,
                d[r],
                (s[2 * r], s[2 * r + 1]),
                &mut o[..header],
            );
        }
        return Ok(out);
    }
    if let Some((header, super_bytes)) = row_scaled {
        let row_bytes = header + blocks * super_bytes;
        let mut out = vec![0u8; rows * row_bytes];
        let scales: Vec<f32> = w
            .par_chunks(cols)
            .map(|row| super::iq_quantize::row_scale(ty, row, Some(importance)))
            .collect();
        let mut sums = vec![(0f32, 0f32); rows];
        for b in 0..blocks {
            let current = rest.narrow(1, 0, block)?.contiguous()?;
            let wb = current.flatten_all()?.to_vec1::<f32>()?;
            let mut decoded = vec![0f32; rows * block];
            out.par_chunks_mut(row_bytes)
                .zip(wb.par_chunks(block))
                .zip(decoded.par_chunks_mut(block))
                .zip(sums.par_iter_mut())
                .zip(scales.par_iter())
                .for_each(|((((o, xbl), deq), s), &d)| {
                    let sb = &mut o[header + b * super_bytes..][..super_bytes];
                    let imp = &importance[b * block..(b + 1) * block];
                    super::iq_quantize::row_scaled_super_block(ty, xbl, Some(imp), d, sb, s, deq);
                    for v in deq.iter_mut() {
                        *v *= d;
                    }
                });
            if let Some(u) = &u
                && b + 1 < blocks
            {
                let error = (current - Tensor::from_slice(&decoded, (rows, block), device)?)?;
                rest = carry(&rest, &error, u, b, block)?;
            } else if b + 1 < blocks {
                rest = rest.narrow(1, block, cols - (b + 1) * block)?;
            }
        }
        out.par_chunks_mut(row_bytes)
            .zip(sums.par_iter())
            .zip(scales.par_iter())
            .for_each(|((o, &s), &d)| {
                super::iq_quantize::row_scaled_header(ty, d, s, &mut o[..header])
            });
        return Ok(out);
    }
    let row_block_bytes = encode(
        ty,
        block,
        &w[..block],
        Some(&importance[..block]),
        quantizer,
    )?
    .len();
    let mut out = vec![0u8; rows * blocks * row_block_bytes];
    for b in 0..blocks {
        let current = rest.narrow(1, 0, block)?.contiguous()?;
        let wb = current.flatten_all()?.to_vec1::<f32>()?;
        let bytes = encode(
            ty,
            block,
            &wb,
            Some(&importance[b * block..(b + 1) * block]),
            quantizer,
        )?;
        for r in 0..rows {
            out[(r * blocks + b) * row_block_bytes..][..row_block_bytes]
                .copy_from_slice(&bytes[r * row_block_bytes..][..row_block_bytes]);
        }
        if b + 1 == blocks {
            break;
        }
        match &u {
            Some(u) => {
                let q = decode(ty, block, &bytes)?;
                let error = (current - Tensor::from_slice(&q, (rows, block), device)?)?;
                rest = carry(&rest, &error, u, b, block)?;
            }
            None => rest = rest.narrow(1, block, cols - (b + 1) * block)?,
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // U^-1 U is the identity
    #[test]
    fn upper_inverse_inverts() {
        let n = 5;
        let u: Vec<f32> = (0..n * n)
            .map(|i| {
                let (r, c) = (i / n, i % n);
                if c < r {
                    0.0
                } else if c == r {
                    2.0 + r as f32
                } else {
                    0.3 * (r + c) as f32
                }
            })
            .collect();
        let inv = upper_inverse(&u, n);
        for r in 0..n {
            for c in 0..n {
                let v: f32 = (0..n).map(|k| inv[r * n + k] * u[k * n + c]).sum();
                assert!(
                    (v - if r == c { 1.0 } else { 0.0 }).abs() < 1e-5,
                    "{r},{c}: {v}"
                );
            }
        }
    }

    // the GPU loop writes the CPU loop's bytes for each row-scaled type (random rows with outliers, a dead column,
    // a random positive definite Hessian, two super-blocks so the feedback runs)
    fn gpu_and_cpu_encodings_agree() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let (rows, cols, samples) = (24, 512, 96);
        let mut state = 7u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        let values: Vec<f32> = (0..rows * cols)
            .map(|i| next() * if i % 37 == 0 { 0.5 } else { 0.05 })
            .collect();
        let x: Vec<f32> = (0..samples * cols)
            .map(|i| if i % cols == 3 { 0.0 } else { next() })
            .collect();
        let x = Tensor::from_slice(&x, (samples, cols), &Device::Cpu)?;
        let h = (x.t()?.matmul(&x)? / samples as f64)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let prepared = prepare(&h, cols, true, 0.1, &device)?;
        let unused_encode =
            |_: GgufType, _: usize, _: &[f32], _: Option<&[f32]>, _: Quantizer| Ok(Vec::new());
        let unused_decode = |_: GgufType, _: usize, _: &[u8]| Ok(Vec::new());
        for ty in [GgufType::Iq4Ks, GgufType::Iq4Kss, GgufType::Iq3Ks] {
            let run = |gpu| {
                quantize_on(
                    ty,
                    cols,
                    &values,
                    &prepared,
                    Quantizer::IkLlama,
                    &device,
                    unused_encode,
                    unused_decode,
                    gpu,
                )
            };
            let (gpu, cpu) = (run(true)?, run(false)?);
            let differing = gpu.iter().zip(&cpu).filter(|(a, b)| a != b).count();
            assert_eq!((gpu.len(), differing), (cpu.len(), 0), "{ty:?}");
        }
        Ok(())
    }

    inference_tensor::gpu_cases!(cuda_cases, [gpu_and_cpu_encodings_agree]);
}
