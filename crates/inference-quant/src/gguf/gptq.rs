//! GPTQ-style error feedback for GGUF types whose blocks stand alone: a matrix quantized a block of columns at a time
//! by the type's own quantizer (weighed by the Hessian's diagonal, as an imatrix would), each block's rounding error
//! carried onto the columns not yet quantized through the inverse Hessian's upper Cholesky factor `U` (`H^-1 = U^T U`):
//! the block's error `E` moves them by `-E U_bb^-1 U_b,rest`, the compensation that keeps the layer's output error
//! least in the Hessian's metric. The bytes stay the type's.

use inference_tensor::quantized::Quantizer;
use inference_tensor::{DType, Device, Result, Tensor, profile};
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

/// A Hessian made ready for the feedback, shareable by matrices with the same input (a stack's experts): the
/// quantizer's importances (its diagonal), the columns no token reached, and `U` when feeding back.
pub struct Prepared {
    importance: Vec<f32>,
    dead: Vec<usize>,
    u: Option<Tensor>,
    // the factorizations' info words, read with the others of a pass in one go ([`ensure_factored`])
    infos: Vec<Tensor>,
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
    Ok(prepare_all(&[hessian], cols, feedback, damp, device)?
        .pop()
        .expect("one prepared"))
}

/// [`prepare`] for each of `hessians`, the factors computed in one batched pass.
pub fn prepare_all(
    hessians: &[&[f32]],
    cols: usize,
    feedback: bool,
    damp: f64,
    device: &Device,
) -> Result<Vec<Prepared>> {
    if let Some(h) = hessians.iter().find(|h| h.len() != cols * cols) {
        inference_tensor::bail!("a Hessian of {} for rows of {cols}", h.len());
    }
    // each Hessian damped straight into its slot of the stack the factors read
    let mut stacked = vec![0f32; hessians.len() * cols * cols];
    let columns = stacked
        .par_chunks_mut(cols * cols)
        .zip(hessians.par_iter())
        .map(|(h, hessian)| {
            h.copy_from_slice(hessian);
            let diag: Vec<f32> = (0..cols).map(|i| h[i * cols + i]).collect();
            let dead: Vec<usize> = (0..cols).filter(|&i| diag[i] <= 0.0).collect();
            for &i in &dead {
                h[i * cols + i] = 1.0;
            }
            let damp = damp * diag.iter().map(|&d| f64::from(d)).sum::<f64>() / cols as f64;
            for i in 0..cols {
                h[i * cols + i] += damp as f32;
            }
            (diag, dead)
        })
        .collect::<Vec<_>>();
    let _span = profile::span("gptq.factor", Some(device));
    let factors = match feedback {
        true => {
            let stacked = Tensor::from_vec(stacked, (hessians.len(), cols, cols), device)?;
            Some(super::super::exl3::linalg::inverse_upper_factors(&stacked)?)
        }
        false => None,
    };
    columns
        .into_iter()
        .enumerate()
        .map(|(e, (diag, dead))| {
            let (u, infos) = match &factors {
                Some((u, infos)) => (Some(u.get(e)?), vec![infos.narrow(0, e, 1)?]),
                None => (None, Vec::new()),
            };
            Ok(Prepared {
                importance: diag
                    .iter()
                    .map(|&d| if d > 0.0 { d } else { DEAD_IMPORTANCE })
                    .collect(),
                dead,
                u,
                infos,
                p: None,
            })
        })
        .collect()
}

// every factorization of `prepared` succeeded: their info words read back together, once
fn ensure_factored(prepared: &[&Prepared]) -> Result<()> {
    let infos = prepared
        .iter()
        .flat_map(|p| p.infos.iter())
        .collect::<Vec<_>>();
    if infos.is_empty() {
        return Ok(());
    }
    let words = Tensor::cat(&infos, 0)?
        .to_device(&Device::Cpu)?
        .to_vec1::<i32>()?;
    if words.iter().any(|&w| w != 0) {
        inference_tensor::bail!("a damped Hessian is not positive definite");
    }
    Ok(())
}

// the columns after block `b` (of `block`) moved by its error: `rest - E U_bb^-1 U_b,rest`
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
    let ubb_inv =
        super::super::exl3::linalg::upper_block_inverses(&u.unsqueeze(0)?, b * block, block)?
            .squeeze(0)?;
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
    values: &Tensor,
    prepared: &[&Prepared],
    device: &Device,
) -> Result<Option<Vec<u8>>> {
    let Device::Cuda(dev) = device else {
        return Ok(None);
    };
    ensure_factored(prepared)?;
    if super::iq_quantize::row_scaled(ty).is_none() || !super::quantize_cuda::fixed_supported(ty) {
        return Ok(None);
    }
    if prepared
        .iter()
        .any(|p| p.importance.len() != cols || p.u.is_some() != prepared[0].u.is_some())
    {
        inference_tensor::bail!("{ty:?}: experts' Hessians differ in width or feedback");
    }
    let dead = prepared
        .iter()
        .map(|p| p.dead.as_slice())
        .collect::<Vec<_>>();
    let rest = zero_dead(
        &values.to_device(device)?.to_dtype(DType::F32)?,
        &dead,
        cols,
    )?;
    gpu_row_scaled(ty, cols, rest, prepared, dev).map(Some)
}

// `x` (rows of `cols`, in `dead.len()` equal groups) with each group's dead columns zeroed, on its device: a column
// no token reached does nothing, so its weights go
fn zero_dead(x: &Tensor, dead: &[&[usize]], cols: usize) -> Result<Tensor> {
    if dead.iter().all(|d| d.is_empty()) {
        return x.contiguous();
    }
    let groups = dead.len();
    let mut mask = vec![1f32; groups * cols];
    for (g, d) in dead.iter().enumerate() {
        for &i in d.iter() {
            mask[g * cols + i] = 0.0;
        }
    }
    let rows = x.dim(0)?;
    let mask = Tensor::from_vec(mask, (groups, 1, cols), x.device())?;
    // + 0 turns the -0 of a negative weight times 0 into the 0 the host wrote
    (x.reshape((groups, rows / groups, cols))?
        .broadcast_mul(&mask)?
        + 0.0)?
        .reshape((rows, cols))
}

// `w`'s columns after block `b` (of `block`) moved by the block's error in place, `w[:, after] -= E U_bb^-1 U_b,after`:
// `M = U_bb^-1 U_b,after` first, then one GEMM (alpha -1, beta 1) writing into `w` itself, a group of rows each
// against its own factor (`u` `[groups, cols, cols]`) as one strided batch
fn carry_in_place(w: &Tensor, error: &Tensor, u: &Tensor, b: usize, block: usize) -> Result<()> {
    use inference_tensor::Storage;
    use inference_tensor::cuda_backend::cudarc::cublas::{result, sys};
    let (rows, cols) = w.dims2()?;
    let groups = u.dim(0)?;
    let per = rows / groups;
    let start = (b + 1) * block;
    let width = cols - start;
    let ubb_inv = super::super::exl3::linalg::upper_block_inverses(u, b * block, block)?;
    let u_rest = u
        .narrow(1, b * block, block)?
        .narrow(2, start, width)?
        .contiguous()?;
    let m = ubb_inv.matmul(&u_rest)?.contiguous()?;
    let error = error.contiguous()?;
    let Device::Cuda(dev) = w.device() else {
        inference_tensor::bail!("the carry runs on CUDA");
    };
    let (w_s, w_l) = w.storage_and_layout();
    let (e_s, e_l) = error.storage_and_layout();
    let (m_s, m_l) = m.storage_and_layout();
    let (Storage::Cuda(w_c), Storage::Cuda(e_c), Storage::Cuda(m_c)) = (&*w_s, &*e_s, &*m_s) else {
        inference_tensor::bail!("the carry's tensors live on CUDA");
    };
    if !w_l.is_contiguous() {
        inference_tensor::bail!("the carry writes into a contiguous matrix");
    }
    let (w_ptr, _w) =
        crate::utils::slice_ptr(w_c.as_cuda_slice::<f32>()?, w_l.start_offset() + start);
    let (e_ptr, _e) = crate::utils::slice_ptr(e_c.as_cuda_slice::<f32>()?, e_l.start_offset());
    let (m_ptr, _m) = crate::utils::slice_ptr(m_c.as_cuda_slice::<f32>()?, m_l.start_offset());
    let (alpha, beta) = (-1.0f32, 1.0f32);
    let as_int = |v: usize| i32::try_from(v).map_err(inference_tensor::Error::wrap);
    let as_long = |v: usize| i64::try_from(v).map_err(inference_tensor::Error::wrap);
    // column-major: w's rows are columns, so C^T (width x per) = M^T (width x block) E^T (block x per)
    // SAFETY: the pointers span their tensors (w from `start` with its row stride), written on the blas stream
    unsafe {
        result::gemm_strided_batched_ex(
            *dev.cublas_handle().handle(),
            sys::cublasOperation_t::CUBLAS_OP_N,
            sys::cublasOperation_t::CUBLAS_OP_N,
            as_int(width)?,
            as_int(per)?,
            as_int(block)?,
            (&raw const alpha).cast(),
            m_ptr as *const _,
            sys::cudaDataType_t::CUDA_R_32F,
            as_int(width)?,
            as_long(block * width)?,
            e_ptr as *const _,
            sys::cudaDataType_t::CUDA_R_32F,
            as_int(block)?,
            as_long(per * block)?,
            (&raw const beta).cast(),
            w_ptr as *mut _,
            sys::cudaDataType_t::CUDA_R_32F,
            as_int(cols)?,
            as_long(per * cols)?,
            as_int(groups)?,
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
    }
    .map_err(|e| inference_tensor::Error::Cuda(Box::new(e)))
}

// the matrix a GPTQ pass encodes a block at a time: its columns moved in place by each block's error, or (GPTAQ's
// residual term, which reads each block's values before quantization, or no feedback at all) narrowed past each block
enum Columns {
    InPlace(Tensor),
    Narrowing(Tensor),
}

impl Columns {
    fn new(rest: &Tensor, in_place: bool) -> Result<Self> {
        Ok(match in_place {
            // its own buffer: the carry writes into it, and `rest` may be the caller's
            true => Self::InPlace(rest.contiguous()?.copy()?),
            false => Self::Narrowing(rest.clone()),
        })
    }

    fn block(&self, b: usize, block: usize) -> Result<Tensor> {
        match self {
            Self::InPlace(w) => w.narrow(1, b * block, block)?.contiguous(),
            Self::Narrowing(rest) => rest.narrow(1, 0, block)?.contiguous(),
        }
    }

    // past block `b`: its `error` carried onto the columns after it through `u` (`[groups, cols, cols]`), with
    // GPTAQ's `asym` term where given (one group)
    fn advance(
        &mut self,
        b: usize,
        block: usize,
        error: Option<&Tensor>,
        u: Option<&Tensor>,
        asym: Option<(&Tensor, &Tensor)>,
    ) -> Result<()> {
        match self {
            Self::InPlace(w) => {
                if let (Some(error), Some(u)) = (error, u) {
                    carry_in_place(w, error, u, b, block)?;
                }
            }
            Self::Narrowing(rest) => {
                let width = rest.dim(1)?;
                *rest = match (error, u) {
                    (Some(error), Some(u)) => carry(rest, error, &u.squeeze(0)?, b, block, asym)?,
                    _ => rest.narrow(1, block, width - block)?,
                };
            }
        }
        Ok(())
    }
}

// the row-scaled loop on the GPU over `rest` (dead columns zeroed), `prepared.len()` equal groups of its rows each
// with its own Hessian: the row scales, each super-block's encoding against them, its error carried on through each
// group's `U`, and the scales refit
fn gpu_row_scaled(
    ty: GgufType,
    cols: usize,
    rest: Tensor,
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
    let span = profile::span("gptq.row_scales", Some(&rest.device().clone()));
    let row_d = super::quantize_cuda::row_scales(dev, ty, &rest, &importance, per)?;
    drop(span);
    let mut sums = dev.alloc_zeros::<f32>(2 * rows)?;
    let u = match &u {
        Some(u) if groups == 1 => Some(u.unsqueeze(0)?),
        other => other.clone(),
    };
    let residual = prepared.iter().any(|p| p.p.is_some());
    let mut columns = Columns::new(&rest, u.is_some() && !residual)?;
    for b in 0..blocks {
        let current = columns.block(b, QK_K)?;
        let imp: Vec<f32> = prepared
            .iter()
            .flat_map(|p| p.importance[b * QK_K..(b + 1) * QK_K].iter().copied())
            .collect();
        let span = profile::span("gptq.encode", Some(current.device()));
        let (bytes, levels) = super::quantize_cuda::super_block_fixed(
            dev, ty, &current, &imp, per, &row_d, &mut sums,
        )?;
        drop(span);
        out.par_chunks_exact_mut(row_bytes)
            .zip(bytes.par_chunks_exact(header + super_bytes))
            .for_each(|(o, sb)| {
                o[header + b * super_bytes..][..super_bytes].copy_from_slice(&sb[header..]);
            });
        if b + 1 == blocks {
            break;
        }
        let _span = profile::span("gptq.carry", Some(current.device()));
        let error = u
            .as_ref()
            .map(|_| &current - levels.broadcast_mul(&row_d.unsqueeze(1)?)?)
            .transpose()?;
        let asym = prepared[0].p.as_ref().map(|p| (p, &current));
        columns.advance(b, QK_K, error.as_ref(), u.as_ref(), asym)?;
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
    values: &Tensor,
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
    values: &Tensor,
    prepared: &Prepared,
    quantizer: Quantizer,
    device: &Device,
    encode: impl Fn(GgufType, usize, &[f32], Option<&[f32]>, Quantizer) -> Result<Vec<u8>>,
    decode: impl Fn(GgufType, usize, &[u8]) -> Result<Vec<f32>>,
    gpu_encode: bool,
) -> Result<Vec<u8>> {
    ensure_factored(&[prepared])?;
    let rows = values.dim(0)?;
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
    let importance = &prepared.importance;
    let u = &prepared.u;
    let blocks = cols / block;
    let rest = zero_dead(
        &values.to_device(device)?.to_dtype(DType::F32)?,
        &[&prepared.dead],
        cols,
    )?;
    if let (Some(_), Device::Cuda(dev), true) = (row_scaled, device, gpu_encode)
        && super::quantize_cuda::fixed_supported(ty)
    {
        return gpu_row_scaled(ty, cols, rest, &[prepared], dev);
    }
    let u3 = u.as_ref().map(|u| u.unsqueeze(0)).transpose()?;
    let mut columns = Columns::new(&rest, u.is_some() && prepared.p.is_none())?;
    if let Some((header, super_bytes)) = row_scaled {
        let row_bytes = header + blocks * super_bytes;
        let mut out = vec![0u8; rows * row_bytes];
        let w = rest.flatten_all()?.to_vec1::<f32>()?;
        let scales: Vec<f32> = w
            .par_chunks(cols)
            .map(|row| super::iq_quantize::row_scale(ty, row, Some(importance)))
            .collect();
        let mut sums = vec![(0f32, 0f32); rows];
        for b in 0..blocks {
            let current = columns.block(b, block)?;
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
            if b + 1 < blocks {
                let error = u3
                    .as_ref()
                    .map(|_| &current - Tensor::from_slice(&decoded, (rows, block), device)?)
                    .transpose()?;
                let asym = prepared.p.as_ref().map(|p| (p, &current));
                columns.advance(b, block, error.as_ref(), u3.as_ref(), asym)?;
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
    // the GPU encodes the blocks in place where it writes the CPU's bytes (ik_llama.cpp's rounding; Q8_0 rounds alike)
    let gpu_blocks = match device {
        Device::Cuda(dev)
            if gpu_encode
                && super::quantize_cuda::blocks_supported(ty)
                && (quantizer == Quantizer::IkLlama || ty == GgufType::Q8_0) =>
        {
            Some(dev)
        }
        _ => None,
    };
    if let Some(dev) = gpu_blocks {
        // each block's bytes stay on the device, read back once for the whole tensor
        let mut encoded_blocks = Vec::with_capacity(blocks);
        for b in 0..blocks {
            let current = columns.block(b, block)?;
            let span = profile::span("gptq.encode", Some(current.device()));
            let (bytes, decoded) = super::quantize_cuda::encode_blocks_on(
                dev,
                ty,
                &current,
                Some(&importance[b * block..(b + 1) * block]),
                rows,
            )?;
            drop(span);
            encoded_blocks.push(bytes);
            if b + 1 == blocks {
                break;
            }
            let error = u3.as_ref().map(|_| &current - decoded).transpose()?;
            let asym = prepared.p.as_ref().map(|p| (p, &current));
            columns.advance(b, block, error.as_ref(), u3.as_ref(), asym)?;
        }
        return Tensor::cat(&encoded_blocks, 1)?
            .flatten_all()?
            .to_vec1::<u8>();
    }
    let first = columns.block(0, block)?;
    let first = first.flatten_all()?.to_vec1::<f32>()?;
    let row_block_bytes = encode(
        ty,
        block,
        &first[..block],
        Some(&importance[..block]),
        quantizer,
    )?
    .len();
    let mut out = vec![0u8; rows * blocks * row_block_bytes];
    for b in 0..blocks {
        let current = columns.block(b, block)?;
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
        let error = match &u3 {
            Some(_) => {
                let q = decode(ty, block, &bytes)?;
                Some((&current - Tensor::from_slice(&q, (rows, block), device)?)?)
            }
            None => None,
        };
        let asym = prepared.p.as_ref().map(|p| (p, &current));
        columns.advance(b, block, error.as_ref(), u3.as_ref(), asym)?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // at most one byte in this many may differ between a batched pass and the experts run alone
    const BATCHED_TIE_SHARE: usize = 1000;

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
        let values = Tensor::from_slice(&values, (rows, cols), &device)?;
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
        // the block types, against the CPU's block quantizer and decoder
        let encode = |ty: GgufType, cols: usize, v: &[f32], w: Option<&[f32]>, q: Quantizer| {
            let dtype = ty.candle().expect("a block type");
            inference_tensor::quantized::quantize_blocks(dtype, v, w, cols, q)
        };
        let decode = |ty: GgufType, cols: usize, bytes: &[u8]| {
            use inference_tensor::quantized::{QStorage, QTensor};
            let dtype = ty.candle().expect("a block type");
            let rows = bytes.len() / ty.row_bytes(cols).expect("a block type");
            let storage =
                QStorage::from_data(std::borrow::Cow::Borrowed(bytes), &Device::Cpu, dtype)?;
            QTensor::new(storage, (rows, cols))?
                .dequantize(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()
        };
        for ty in [GgufType::Q6K, GgufType::Q8_0] {
            let run = |gpu| {
                quantize_on(
                    ty,
                    cols,
                    &values,
                    &prepared,
                    Quantizer::IkLlama,
                    &device,
                    encode,
                    decode,
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
    // Hessian, one with a dead column, so the batched carry runs), but for the odd value at a rounding tie: cuBLAS
    // picks its kernels by batch size, while an expert reading another's rows or factor would change most bytes
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
        let values = Tensor::from_slice(&values, (experts * rows, cols), &device)?;
        let unused_encode =
            |_: GgufType, _: usize, _: &[f32], _: Option<&[f32]>, _: Quantizer| Ok(Vec::new());
        let unused_decode = |_: GgufType, _: usize, _: &[u8]| Ok(Vec::new());
        for ty in [GgufType::Iq4Ks, GgufType::Iq4Kss, GgufType::Iq3Ks] {
            let batched = quantize_experts(ty, cols, &values, &refs, &device)?.expect("a GPU pass");
            let mut alone = Vec::new();
            for (e, p) in prepared.iter().enumerate() {
                alone.extend(quantize(
                    ty,
                    cols,
                    &values.narrow(0, e * rows, rows)?,
                    p,
                    Quantizer::IkLlama,
                    &device,
                    unused_encode,
                    unused_decode,
                )?);
            }
            let differing = batched.iter().zip(&alone).filter(|(a, b)| a != b).count();
            assert_eq!(batched.len(), alone.len(), "{ty:?}");
            assert!(
                differing * BATCHED_TIE_SHARE <= batched.len(),
                "{ty:?}: {differing} of {} bytes differ",
                batched.len()
            );
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
        let values = w.clone();
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
