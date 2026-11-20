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

// the inverse of an upper triangular `n x n` matrix, in f64: back substitution a row at a time (row r from the rows
// below it), so the inner loop runs along contiguous rows
fn upper_inverse(u: &[f32], n: usize) -> Vec<f32> {
    let mut inv = vec![0f64; n * n];
    let mut acc = vec![0f64; n];
    for r in (0..n).rev() {
        acc[r..].fill(0.0);
        for k in r + 1..n {
            let urk = f64::from(u[r * n + k]);
            for (a, &v) in acc[k..].iter_mut().zip(&inv[k * n + k..(k + 1) * n]) {
                *a += urk * v;
            }
        }
        let d = 1.0 / f64::from(u[r * n + r]);
        inv[r * n + r] = d;
        for c in r + 1..n {
            inv[r * n + c] = -acc[c] * d;
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
    // GPTAQ's residual term `P = (dX X^T U^T) o M_U U`: each quantized column's input drift carried onto the
    // columns after it, `W_rest += W_b P_b,rest`
    p: Option<Tensor>,
}

impl Prepared {
    /// GPTAQ (Li et al., 2025): with `cross` (`cols x cols`, `(X_fp - X_q)^T X_q / n`, the quantized model's inputs
    /// `X_q` against the full-precision model's `X_fp`, and this Hessian over `X_q`), the feedback fits the layer's
    /// output to the full-precision model's instead of to its own input.
    pub fn with_cross(mut self, cross: &[f32], device: &Device) -> Result<Self> {
        let Some(u) = &self.u else {
            return Ok(self);
        };
        let cols = self.importance.len();
        if cross.len() != cols * cols {
            inference_tensor::bail!("a cross moment of {} for rows of {cols}", cross.len());
        }
        let c = Tensor::from_slice(cross, (cols, cols), device)?;
        let index = Tensor::arange(
            0u32,
            u32::try_from(cols).map_err(inference_tensor::Error::wrap)?,
            device,
        )?;
        let strictly_upper = index
            .unsqueeze(1)?
            .broadcast_lt(&index.unsqueeze(0)?)?
            .to_dtype(DType::F32)?;
        let p = c.matmul(&u.t()?)?.mul(&strictly_upper)?.matmul(u)?;
        self.p = Some(p);
        Ok(self)
    }
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
        p: None,
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
// `carry` for equal groups of rows each with its own factor (`u`: `[groups, cols, cols]`)
fn carry_groups(
    rest: &Tensor,
    error: &Tensor,
    u: &Tensor,
    b: usize,
    block: usize,
) -> Result<Tensor> {
    let (rows, width) = rest.dims2()?;
    let groups = u.dim(0)?;
    let ubb = u
        .narrow(1, b * block, block)?
        .narrow(2, b * block, block)?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let inverses = ubb
        .par_chunks(block * block)
        .map(|m| upper_inverse(m, block))
        .collect::<Vec<_>>()
        .concat();
    let ubb_inv = Tensor::from_vec(inverses, (groups, block, block), rest.device())?;
    let u_rest = u
        .narrow(1, b * block, block)?
        .narrow(2, (b + 1) * block, width - block)?
        .contiguous()?;
    let moved = error
        .reshape((groups, rows / groups, block))?
        .matmul(&ubb_inv)?
        .matmul(&u_rest)?
        .reshape((rows, width - block))?;
    rest.narrow(1, block, width - block)? - moved
}

// with GPTAQ's `p`, also the block's pre-quantization values `current` times its rows of `p`
fn carry(
    rest: &Tensor,
    error: &Tensor,
    u: &Tensor,
    b: usize,
    block: usize,
    asym: Option<(&Tensor, &Tensor)>,
) -> Result<Tensor> {
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
    let moved =
        (rest.narrow(1, block, width - block)? - error.matmul(&ubb_inv)?.matmul(&u_rest)?)?;
    match asym {
        Some((p, current)) => {
            let p_rest = p
                .narrow(0, b * block, block)?
                .narrow(1, (b + 1) * block, width - block)?
                .contiguous()?;
            moved + current.matmul(&p_rest)?
        }
        None => Ok(moved),
    }
}

/// A stack's experts (`values`: `prepared.len()` equal runs of rows of `cols`, each with its own prepared Hessian) as
/// a row-scaled `ty`, all in one pass on the GPU; None where that pass does not cover `ty` or `device`.
pub fn quantize_experts(
    ty: GgufType,
    cols: usize,
    values: &[f32],
    prepared: &[&Prepared],
    device: &Device,
) -> Result<Option<Vec<u8>>> {
    let Device::Cuda(dev) = device else {
        return Ok(None);
    };
    if super::iq_quantize::row_scaled(ty).is_none() || !super::quantize_cuda::fixed_supported(ty) {
        return Ok(None);
    }
    if prepared
        .iter()
        .any(|p| p.importance.len() != cols || p.u.is_some() != prepared[0].u.is_some())
    {
        inference_tensor::bail!("{ty:?}: experts' Hessians differ in width or feedback");
    }
    let per = values.len() / prepared.len();
    let mut w = std::borrow::Cow::Borrowed(values);
    if prepared.iter().any(|p| !p.dead.is_empty()) {
        let w = w.to_mut();
        for (e, p) in prepared.iter().enumerate() {
            for &i in &p.dead {
                for r in 0..per / cols {
                    w[e * per + r * cols + i] = 0.0;
                }
            }
        }
    }
    let rest = Tensor::from_slice(&w, (values.len() / cols, cols), device)?;
    gpu_row_scaled(ty, cols, rest, prepared, dev).map(Some)
}

// the row-scaled loop on the GPU over `rest` (dead columns zeroed), `prepared.len()` equal groups of its rows each
// with its own Hessian: the row scales, each super-block's encoding against them, its error carried on through each
// group's `U`, and the scales refit
fn gpu_row_scaled(
    ty: GgufType,
    cols: usize,
    mut rest: Tensor,
    prepared: &[&Prepared],
    dev: &inference_tensor::CudaDevice,
) -> Result<Vec<u8>> {
    let Some((header, super_bytes)) = super::iq_quantize::row_scaled(ty) else {
        inference_tensor::bail!("{ty:?} is not row-scaled");
    };
    let rows = rest.dim(0)?;
    let groups = prepared.len();
    let per = rows / groups;
    let blocks = cols / QK_K;
    let row_bytes = header + blocks * super_bytes;
    let importance: Vec<f32> = prepared
        .iter()
        .flat_map(|p| p.importance.iter().copied())
        .collect();
    // a group's factor, or the groups' stacked `[groups, cols, cols]`
    let u = match prepared[0].u.as_ref() {
        Some(u) if groups == 1 => Some(u.clone()),
        Some(_) => Some(Tensor::stack(
            &prepared
                .iter()
                .filter_map(|p| p.u.as_ref())
                .collect::<Vec<_>>(),
            0,
        )?),
        None => None,
    };
    let mut out = vec![0u8; rows * row_bytes];
    let row_d = super::quantize_cuda::row_scales(dev, ty, &rest, &importance, per)?;
    let mut sums = dev.alloc_zeros::<f32>(2 * rows)?;
    for b in 0..blocks {
        let current = rest.narrow(1, 0, QK_K)?.contiguous()?;
        let imp: Vec<f32> = prepared
            .iter()
            .flat_map(|p| p.importance[b * QK_K..(b + 1) * QK_K].iter().copied())
            .collect();
        let (bytes, levels) = super::quantize_cuda::super_block_fixed(
            dev, ty, &current, &imp, per, &row_d, &mut sums,
        )?;
        out.par_chunks_exact_mut(row_bytes)
            .zip(bytes.par_chunks_exact(header + super_bytes))
            .for_each(|(o, sb)| {
                o[header + b * super_bytes..][..super_bytes].copy_from_slice(&sb[header..]);
            });
        if b + 1 == blocks {
            break;
        }
        rest = match &u {
            Some(u) => {
                let error = (&current - levels.broadcast_mul(&row_d.unsqueeze(1)?)?)?;
                if groups == 1 {
                    let asym = prepared[0].p.as_ref().map(|p| (p, &current));
                    carry(&rest, &error, u, b, QK_K, asym)?
                } else {
                    carry_groups(&rest, &error, u, b, QK_K)?
                }
            }
            None => rest.narrow(1, QK_K, cols - (b + 1) * QK_K)?,
        };
    }
    let d = row_d.to_vec1::<f32>()?;
    let s = dev.clone_dtoh(&sums)?;
    out.par_chunks_exact_mut(row_bytes)
        .enumerate()
        .for_each(|(r, o)| {
            super::iq_quantize::row_scaled_header(
                ty,
                d[r],
                (s[2 * r], s[2 * r + 1]),
                &mut o[..header],
            );
        });
    Ok(out)
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
    let mut w = std::borrow::Cow::Borrowed(values);
    if !prepared.dead.is_empty() {
        let w = w.to_mut();
        for &i in &prepared.dead {
            for r in 0..rows {
                w[r * cols + i] = 0.0;
            }
        }
    }
    let importance = &prepared.importance;
    let u = &prepared.u;
    let blocks = cols / block;
    let mut rest = Tensor::from_slice(&w, (rows, cols), device)?;
    if let (Some(_), Device::Cuda(dev), true) = (row_scaled, device, gpu_encode)
        && super::quantize_cuda::fixed_supported(ty)
    {
        return gpu_row_scaled(ty, cols, rest, &[prepared], dev);
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
                let error = (&current - Tensor::from_slice(&decoded, (rows, block), device)?)?;
                let asym = prepared.p.as_ref().map(|p| (p, &current));
                rest = carry(&rest, &error, u, b, block, asym)?;
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
                let error = (&current - Tensor::from_slice(&q, (rows, block), device)?)?;
                let asym = prepared.p.as_ref().map(|p| (p, &current));
                rest = carry(&rest, &error, u, b, block, asym)?;
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

    // a stack's experts in one batched pass write what each written alone does (three experts, each its own random
    // Hessian, one with a dead column, so the batched carry runs)
    fn experts_batched_match_alone() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let (experts, rows, cols, samples) = (3, 16, 512, 96);
        let mut state = 11u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        let values: Vec<f32> = (0..experts * rows * cols)
            .map(|i| next() * if i % 37 == 0 { 0.5 } else { 0.05 })
            .collect();
        let mut prepared = Vec::new();
        for e in 0..experts {
            let x: Vec<f32> = (0..samples * cols)
                .map(|i| if e == 1 && i % cols == 5 { 0.0 } else { next() })
                .collect();
            let x = Tensor::from_slice(&x, (samples, cols), &Device::Cpu)?;
            let h = (x.t()?.matmul(&x)? / samples as f64)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            prepared.push(prepare(&h, cols, true, 0.1, &device)?);
        }
        let refs = prepared.iter().collect::<Vec<_>>();
        let unused_encode =
            |_: GgufType, _: usize, _: &[f32], _: Option<&[f32]>, _: Quantizer| Ok(Vec::new());
        let unused_decode = |_: GgufType, _: usize, _: &[u8]| Ok(Vec::new());
        let per = rows * cols;
        for ty in [GgufType::Iq4Ks, GgufType::Iq4Kss, GgufType::Iq3Ks] {
            let batched = quantize_experts(ty, cols, &values, &refs, &device)?.expect("a GPU pass");
            let mut alone = Vec::new();
            for (e, p) in prepared.iter().enumerate() {
                alone.extend(quantize(
                    ty,
                    cols,
                    &values[e * per..(e + 1) * per],
                    p,
                    Quantizer::IkLlama,
                    &device,
                    unused_encode,
                    unused_decode,
                )?);
            }
            let differing = batched.iter().zip(&alone).filter(|(a, b)| a != b).count();
            assert_eq!((batched.len(), differing), (alone.len(), 0), "{ty:?}");
        }
        Ok(())
    }

    // GPTAQ fits the quantized layer to the full-precision output: with inputs drifted from the full-precision
    // ones (a systematic map plus noise, as earlier quantized layers leave them), its `||W X_fp - Q X_q||` is below
    // plain GPTQ's on the drifted inputs, and a zero cross moment changes nothing
    fn asymmetric_feedback_fits_the_full_precision_output() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let cpu = Device::Cpu;
        let (rows, cols, samples) = (32, 512, 1024);
        let mut state = 23u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        let mut random = |n: usize| (0..n).map(|_| next()).collect::<Vec<f32>>();
        let w = Tensor::from_vec(random(rows * cols), (rows, cols), &cpu)?;
        let mix = (Tensor::from_vec(random(cols * cols), (cols, cols), &cpu)? * 0.1)?;
        let x_fp = Tensor::from_vec(random(samples * cols), (samples, cols), &cpu)?.matmul(&mix)?;
        let drift = (Tensor::from_vec(random(cols * cols), (cols, cols), &cpu)? * 0.02)?;
        let noise = (Tensor::from_vec(random(samples * cols), (samples, cols), &cpu)? * 0.005)?;
        let x_q = ((&x_fp + x_fp.matmul(&drift)?)? + noise)?;
        let n = samples as f64;
        let h = (x_q.t()?.matmul(&x_q)? / n)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let cross = ((&x_fp - &x_q)?.t()?.matmul(&x_q)? / n)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let unused_encode =
            |_: GgufType, _: usize, _: &[f32], _: Option<&[f32]>, _: Quantizer| Ok(Vec::new());
        let unused_decode = |_: GgufType, _: usize, _: &[u8]| Ok(Vec::new());
        let ty = GgufType::Iq4Ks;
        let values = w.flatten_all()?.to_vec1::<f32>()?;
        let error = |prepared: &Prepared| -> Result<f32> {
            let bytes = quantize(
                ty,
                cols,
                &values,
                prepared,
                Quantizer::IkLlama,
                &device,
                unused_encode,
                unused_decode,
            )?;
            let q = Tensor::from_vec(
                super::super::raw::dequantize_rows(ty, cols, &bytes)?,
                (rows, cols),
                &cpu,
            )?;
            (w.matmul(&x_fp.t()?)? - q.matmul(&x_q.t()?)?)?
                .sqr()?
                .sum_all()?
                .sqrt()?
                .to_scalar::<f32>()
        };
        let gptq = error(&prepare(&h, cols, true, 0.1, &device)?)?;
        let gptaq = error(&prepare(&h, cols, true, 0.1, &device)?.with_cross(&cross, &device)?)?;
        let zero = error(
            &prepare(&h, cols, true, 0.1, &device)?.with_cross(&vec![0.0; cols * cols], &device)?,
        )?;
        assert!(gptaq < 0.9 * gptq, "GPTAQ {gptaq} against GPTQ {gptq}");
        assert_eq!(zero, gptq, "a zero cross moment is plain GPTQ");
        Ok(())
    }

    inference_tensor::gpu_cases!(
        cuda_cases,
        [
            gpu_and_cpu_encodings_agree,
            experts_batched_match_alone,
            asymmetric_feedback_fits_the_full_precision_output
        ]
    );
}
