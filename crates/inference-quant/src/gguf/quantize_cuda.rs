//! GGUF quantizers on the GPU (`kernels/cuda/gguf_quantize/*_instance_ggufq.cu`), writing the bytes the CPU ports in
//! `iq_quantize` write: every product, sum and fused multiply-add there is an explicit rounding intrinsic here.

use inference_tensor::cuda::cudarc::driver::CudaSlice;
use inference_tensor::{CudaDevice, DType, Result, Tensor};

use super::kernel::GgufType;
use crate::utils::slice_ptr;

const QK_K: usize = 256;
const BLOCK: usize = 32;
const IQ4_KS_ROW_HEADER: usize = 4;
const BLOCK_IQ4_KS: usize = QK_K / BLOCK + QK_K / 2;
const BLOCK_IQ4_KSS: usize = QK_K / 2;
const BLOCK_IQ4_XS: usize = 2 + 2 + QK_K / 64 + QK_K / 2;
const F16_ROW_HEADER: usize = 2;
const BLOCK_IQ3_KS: usize = 102;
const BLOCK_Q6K: usize = 210;
const Q6K_SUB: usize = 16;
const Q8_0_BLOCK: usize = 32;
const BLOCK_Q8_0: usize = 34;
const GGML_F32: u32 = 0;
const GGML_F16: u32 = 1;
const GGML_BF16: u32 = 30;

inference_tensor::kernel_library! {
    library = "inferencequant_ggufq";
    path = option_env!("INFERENCE_KERNEL_LIB_INFERENCEQUANT_GGUFQ").unwrap_or("");
    id = option_env!("INFERENCE_KERNEL_ID_INFERENCEQUANT_GGUFQ").unwrap_or("");
    unsafe extern "C" {
        /// `rows` rows of `cols` values of ggml type `dtype` (f32, f16 or bf16; `x` holds them as f32) to IQ4_KS
        /// rows in `out`, `imatrix` holding `cols` importances per `rows_per_imatrix` rows, or null. Scratch:
        /// `weights` and `terms` hold `rows * cols` floats, `scales` and `shifted` one entry per 32 values, `row_d`
        /// one per row. Returns 0, or a CUDA error code.
        fn launch_iq4_ks_quantize(
            source: *const u8,
            dtype: i32,
            x: *mut f32,
            imatrix: *const f32,
            rows_per_imatrix: i64,
            rows: i64,
            cols: i32,
            weights: *mut f32,
            terms: *mut f32,
            scales: *mut f32,
            shifted: *mut u8,
            row_d: *mut f32,
            out: *mut u8,
            stream: inference_tensor::cuda::cudarc::driver::sys::CUstream,
        ) -> i32;

        /// As `launch_iq4_ks_quantize`, for IQ4_KSS (`terms` its per-value levels; `shifted` unused).
        fn launch_iq4_kss_quantize(
            source: *const u8,
            dtype: i32,
            x: *mut f32,
            imatrix: *const f32,
            rows_per_imatrix: i64,
            rows: i64,
            cols: i32,
            weights: *mut f32,
            terms: *mut f32,
            scales: *mut f32,
            shifted: *mut u8,
            row_d: *mut f32,
            out: *mut u8,
            stream: inference_tensor::cuda::cudarc::driver::sys::CUstream,
        ) -> i32;

        /// As `launch_iq4_ks_quantize`, for IQ4_XS (no row scale; only `x`, `scales` and `out` are used).
        fn launch_iq4_xs_quantize(
            source: *const u8,
            dtype: i32,
            x: *mut f32,
            imatrix: *const f32,
            rows_per_imatrix: i64,
            rows: i64,
            cols: i32,
            weights: *mut f32,
            terms: *mut f32,
            scales: *mut f32,
            shifted: *mut u8,
            row_d: *mut f32,
            out: *mut u8,
            stream: inference_tensor::cuda::cudarc::driver::sys::CUstream,
        ) -> i32;

        /// As `launch_iq4_ks_quantize`, for IQ3_KS (an f16 row scale; `terms` its per-value levels).
        fn launch_iq3_ks_quantize(
            source: *const u8,
            dtype: i32,
            x: *mut f32,
            imatrix: *const f32,
            rows_per_imatrix: i64,
            rows: i64,
            cols: i32,
            weights: *mut f32,
            terms: *mut f32,
            scales: *mut f32,
            shifted: *mut u8,
            row_d: *mut f32,
            out: *mut u8,
            stream: inference_tensor::cuda::cudarc::driver::sys::CUstream,
        ) -> i32;

        /// GPTQ's first pass: each row's scale (`row_d`) of a row-scaled type from `rows` rows of `cols` f32 on the
        /// device, `importance` their `cols` weights; `weights`, `scales` and `shifted` scratch as above.
        fn launch_iq4_ks_row_scale(x: *const f32, importance: *const f32, rows: i64, rows_per_importance: i64, cols: i32, weights: *mut f32,
            scales: *mut f32, shifted: *mut u8, row_d: *mut f32,
            stream: inference_tensor::cuda::cudarc::driver::sys::CUstream) -> i32;
        fn launch_iq4_kss_row_scale(x: *const f32, importance: *const f32, rows: i64, rows_per_importance: i64, cols: i32, weights: *mut f32,
            scales: *mut f32, shifted: *mut u8, row_d: *mut f32,
            stream: inference_tensor::cuda::cudarc::driver::sys::CUstream) -> i32;
        fn launch_iq3_ks_row_scale(x: *const f32, importance: *const f32, rows: i64, rows_per_importance: i64, cols: i32, weights: *mut f32,
            scales: *mut f32, shifted: *mut u8, row_d: *mut f32,
            stream: inference_tensor::cuda::cudarc::driver::sys::CUstream) -> i32;

        /// One super-block (`x`: rows of 256 f32) encoded against the fixed `row_d`: `out` a row of the header's width
        /// then the super-block's bytes each, `levels` each value's level, `sums` (2 a row) continued in the CPU's order.
        fn launch_iq4_ks_super_fixed(x: *const f32, importance: *const f32, rows: i64, rows_per_importance: i64, row_d: *const f32,
            weights: *mut f32, scales: *mut f32, shifted: *mut u8, out: *mut u8, terms: *mut f32, levels: *mut f32,
            sums: *mut f32, stream: inference_tensor::cuda::cudarc::driver::sys::CUstream) -> i32;
        fn launch_iq4_kss_super_fixed(x: *const f32, importance: *const f32, rows: i64, rows_per_importance: i64, row_d: *const f32,
            weights: *mut f32, scales: *mut f32, shifted: *mut u8, out: *mut u8, terms: *mut f32, levels: *mut f32,
            sums: *mut f32, stream: inference_tensor::cuda::cudarc::driver::sys::CUstream) -> i32;
        fn launch_iq3_ks_super_fixed(x: *const f32, importance: *const f32, rows: i64, rows_per_importance: i64, row_d: *const f32,
            weights: *mut f32, scales: *mut f32, shifted: *mut u8, out: *mut u8, terms: *mut f32, levels: *mut f32,
            sums: *mut f32, stream: inference_tensor::cuda::cudarc::driver::sys::CUstream) -> i32;

        /// `rows` rows of `cols` f32 (a multiple of 256) as Q6_K into `out`, each `rows_per_importance` rows weighed by
        /// the next `cols` of `importance`, the decoded values into `decoded`; `ls` (a byte a value) and `scales` (a
        /// float per 16) scratch.
        fn launch_q6_k_encode(x: *const f32, importance: *const f32, rows: i64, rows_per_importance: i64, cols: i32,
            ls: *mut i8, scales: *mut f32, out: *mut u8, decoded: *mut f32,
            stream: inference_tensor::cuda::cudarc::driver::sys::CUstream) -> i32;
        /// As `launch_q6_k_encode`, for Q8_0 (`cols` a multiple of 32; no importance).
        fn launch_q8_0_encode(x: *const f32, rows: i64, cols: i32, out: *mut u8, decoded: *mut f32,
            stream: inference_tensor::cuda::cudarc::driver::sys::CUstream) -> i32;
    }
}

type Launch = unsafe extern "C" fn(
    *const u8,
    i32,
    *mut f32,
    *const f32,
    i64,
    i64,
    i32,
    *mut f32,
    *mut f32,
    *mut f32,
    *mut u8,
    *mut f32,
    *mut u8,
    inference_tensor::cuda::cudarc::driver::sys::CUstream,
) -> i32;

// a type's launcher and the bytes a row of `cols` takes
fn kernels(ty: GgufType, cols: usize) -> Option<(Launch, usize)> {
    match ty {
        GgufType::Iq4Ks => Some((
            launch_iq4_ks_quantize,
            IQ4_KS_ROW_HEADER + cols / QK_K * BLOCK_IQ4_KS,
        )),
        GgufType::Iq4Kss => Some((
            launch_iq4_kss_quantize,
            IQ4_KS_ROW_HEADER + cols / QK_K * BLOCK_IQ4_KSS,
        )),
        GgufType::Iq4Xs => Some((launch_iq4_xs_quantize, cols / QK_K * BLOCK_IQ4_XS)),
        GgufType::Iq3Ks => Some((
            launch_iq3_ks_quantize,
            F16_ROW_HEADER + cols / QK_K * BLOCK_IQ3_KS,
        )),
        _ => None,
    }
}

type RowScaleLaunch = unsafe extern "C" fn(
    *const f32,
    *const f32,
    i64,
    i64,
    i32,
    *mut f32,
    *mut f32,
    *mut u8,
    *mut f32,
    inference_tensor::cuda::cudarc::driver::sys::CUstream,
) -> i32;
type SuperFixedLaunch = unsafe extern "C" fn(
    *const f32,
    *const f32,
    i64,
    i64,
    *const f32,
    *mut f32,
    *mut f32,
    *mut u8,
    *mut u8,
    *mut f32,
    *mut f32,
    *mut f32,
    inference_tensor::cuda::cudarc::driver::sys::CUstream,
) -> i32;

// a row-scaled type's GPTQ launchers and its header and super-block widths
fn fixed_kernels(ty: GgufType) -> Option<(RowScaleLaunch, SuperFixedLaunch, usize, usize)> {
    match ty {
        GgufType::Iq4Ks => Some((
            launch_iq4_ks_row_scale,
            launch_iq4_ks_super_fixed,
            IQ4_KS_ROW_HEADER,
            BLOCK_IQ4_KS,
        )),
        GgufType::Iq4Kss => Some((
            launch_iq4_kss_row_scale,
            launch_iq4_kss_super_fixed,
            IQ4_KS_ROW_HEADER,
            BLOCK_IQ4_KSS,
        )),
        GgufType::Iq3Ks => Some((
            launch_iq3_ks_row_scale,
            launch_iq3_ks_super_fixed,
            F16_ROW_HEADER,
            BLOCK_IQ3_KS,
        )),
        _ => None,
    }
}

/// Whether GPTQ's row-scaled steps ([`row_scales`], [`super_block_fixed`]) run on the GPU for `ty`.
pub fn fixed_supported(ty: GgufType) -> bool {
    fixed_kernels(ty).is_some()
}

// a device tensor's f32 pointer: the tensor outlives each launch here, on its own stream
fn tensor_ptr(t: &Tensor) -> Result<u64> {
    let (st, layout) = t.storage_and_layout();
    let inference_tensor::Storage::Cuda(c) = &*st else {
        inference_tensor::bail!("a GPU quantizer's input lives on CUDA");
    };
    Ok(slice_ptr(c.as_cuda_slice::<f32>()?, layout.start_offset()).0)
}

/// Each row's scale for `ty` from an ordinary first pass over `x` (rows of `cols` f32 on `dev`), each
/// `rows_per_importance` rows weighed by the next `cols` of `importance` (an expert stack's experts in one call): the
/// row scale GPTQ encodes every super-block against.
pub fn row_scales(
    dev: &CudaDevice,
    ty: GgufType,
    x: &Tensor,
    importance: &[f32],
    rows_per_importance: usize,
) -> Result<Tensor> {
    let Some((launch, ..)) = fixed_kernels(ty) else {
        inference_tensor::bail!("{ty:?} has no GPU row scale");
    };
    ensure_loaded().map_err(inference_tensor::Error::msg)?;
    let x = x.to_dtype(DType::F32)?.contiguous()?;
    let (rows, cols) = x.dims2()?;
    let w = dev.clone_htod(importance)?;
    // SAFETY: each scratch buffer is written by one kernel before a later one reads it
    let weights = unsafe { dev.alloc::<f32>(rows * cols)? };
    // SAFETY: as above
    let scales = unsafe { dev.alloc::<f32>(rows * cols / BLOCK)? };
    // SAFETY: as above
    let shifted = unsafe { dev.alloc::<u8>(rows * cols / BLOCK)? };
    // SAFETY: the row-scale kernel writes every row's
    let row_d = unsafe { dev.alloc::<f32>(rows)? };
    {
        let xp = tensor_ptr(&x)?;
        let (wp, _w) = slice_ptr(&w, 0);
        let (gp, _g) = slice_ptr(&weights, 0);
        let (sp, _s) = slice_ptr(&scales, 0);
        let (fp, _f) = slice_ptr(&shifted, 0);
        let (dp, _d) = slice_ptr(&row_d, 0);
        // SAFETY: live device buffers sized as the kernels index them, on this device's stream
        let status = unsafe {
            launch(
                xp as *const f32,
                wp as *const f32,
                rows as i64,
                rows_per_importance as i64,
                cols as i32,
                gp as *mut f32,
                sp as *mut f32,
                fp as *mut u8,
                dp as *mut f32,
                dev.cuda_stream().cu_stream(),
            )
        };
        if status != 0 {
            inference_tensor::bail!(
                "the {ty:?} row-scale kernels failed to launch (CUDA error {status})"
            );
        }
    }
    Ok(Tensor::from((
        inference_tensor::Storage::Cuda(inference_tensor::CudaStorage::wrap_cuda_slice(
            row_d,
            dev.clone(),
        )),
        inference_tensor::Shape::from(rows),
    )))
}

/// One super-block of each row (`x`: rows of 256 f32 on `dev`, each `rows_per_importance` rows weighed by the next 256
/// of `importance`) as `ty` against the fixed `row_d`: each row's
/// super-block bytes after a header-wide gap (rows of the type's header plus super-block bytes), and each value's level (the value decodes to `row_d * level`); `sums` (f32, 2 a row) carries
/// each row's fit on for the refit of its scale.
pub fn super_block_fixed(
    dev: &CudaDevice,
    ty: GgufType,
    x: &Tensor,
    importance: &[f32],
    rows_per_importance: usize,
    row_d: &Tensor,
    sums: &mut CudaSlice<f32>,
) -> Result<(Vec<u8>, Tensor)> {
    let Some((_, launch, header, super_bytes)) = fixed_kernels(ty) else {
        inference_tensor::bail!("{ty:?} has no GPU super-block encoder");
    };
    ensure_loaded().map_err(inference_tensor::Error::msg)?;
    let x = x.to_dtype(DType::F32)?.contiguous()?;
    let (rows, cols) = x.dims2()?;
    if cols != QK_K || importance.len() * rows_per_importance != QK_K * rows {
        inference_tensor::bail!("a super-block is {QK_K} values, not {cols}");
    }
    let row_d = row_d.contiguous()?;
    let w = dev.clone_htod(importance)?;
    // SAFETY: each scratch buffer is written by one kernel before a later one reads it
    let weights = unsafe { dev.alloc::<f32>(rows * QK_K)? };
    // SAFETY: as above
    let terms = unsafe { dev.alloc::<f32>(rows * QK_K)? };
    // SAFETY: as above
    let scales = unsafe { dev.alloc::<f32>(rows * QK_K / BLOCK)? };
    // SAFETY: as above
    let shifted = unsafe { dev.alloc::<u8>(rows * QK_K / BLOCK)? };
    // SAFETY: the encoders write each row's super-block (the header's width before it is left as is)
    let out = dev.alloc_zeros::<u8>(rows * (header + super_bytes))?;
    // SAFETY: the encoders write every level
    let levels = unsafe { dev.alloc::<f32>(rows * QK_K)? };
    {
        let xp = tensor_ptr(&x)?;
        let rp = tensor_ptr(&row_d)?;
        let (wp, _w) = slice_ptr(&w, 0);
        let (gp, _g) = slice_ptr(&weights, 0);
        let (tp, _t) = slice_ptr(&terms, 0);
        let (sp, _s) = slice_ptr(&scales, 0);
        let (fp, _f) = slice_ptr(&shifted, 0);
        let (op, _o) = slice_ptr(&out, 0);
        let (lp, _l) = slice_ptr(&levels, 0);
        let (mp, _m) = slice_ptr(&*sums, 0);
        // SAFETY: live device buffers sized as the kernels index them, on this device's stream
        let status = unsafe {
            launch(
                xp as *const f32,
                wp as *const f32,
                rows as i64,
                rows_per_importance as i64,
                rp as *const f32,
                gp as *mut f32,
                sp as *mut f32,
                fp as *mut u8,
                op as *mut u8,
                tp as *mut f32,
                lp as *mut f32,
                mp as *mut f32,
                dev.cuda_stream().cu_stream(),
            )
        };
        if status != 0 {
            inference_tensor::bail!(
                "the {ty:?} super-block kernels failed to launch (CUDA error {status})"
            );
        }
    }
    let bytes = dev.clone_dtoh(&out)?;
    let levels = Tensor::from((
        inference_tensor::Storage::Cuda(inference_tensor::CudaStorage::wrap_cuda_slice(
            levels,
            dev.clone(),
        )),
        inference_tensor::Shape::from((rows, QK_K)),
    ));
    Ok((bytes, levels))
}

/// Whether [`encode_blocks`] writes `ty` (with an importance where it takes one) on the GPU.
pub fn blocks_supported(ty: GgufType) -> bool {
    matches!(ty, GgufType::Q6K | GgufType::Q8_0)
}

/// `x` (rows of `cols` f32 on `dev`) as `ty`'s blocks, the bytes the CPU's `quantize_blocks` writes, each
/// `rows_per_importance` rows weighed by the next `cols` of `importance` (Q6_K; Q8_0 takes none), with the values they
/// decode to (as the CPU decodes them), left on the device.
pub fn encode_blocks(
    dev: &CudaDevice,
    ty: GgufType,
    x: &Tensor,
    importance: Option<&[f32]>,
    rows_per_importance: usize,
) -> Result<(Vec<u8>, Tensor)> {
    ensure_loaded().map_err(inference_tensor::Error::msg)?;
    let x = x.to_dtype(DType::F32)?.contiguous()?;
    let (rows, cols) = x.dims2()?;
    let (block, block_bytes) = match ty {
        GgufType::Q6K => (QK_K, BLOCK_Q6K),
        GgufType::Q8_0 => (Q8_0_BLOCK, BLOCK_Q8_0),
        _ => inference_tensor::bail!("{ty:?} has no GPU block encoder"),
    };
    if !cols.is_multiple_of(block)
        || importance.is_some_and(|w| {
            rows_per_importance == 0 || w.len() != rows.div_ceil(rows_per_importance) * cols
        })
    {
        inference_tensor::bail!(
            "{ty:?}: rows of {cols} with an importance of {:?}",
            importance.map(<[f32]>::len)
        );
    }
    // SAFETY: the encoders write every byte and every decoded value
    let out = unsafe { dev.alloc::<u8>(rows * cols / block * block_bytes)? };
    // SAFETY: as above
    let decoded = unsafe { dev.alloc::<f32>(rows * cols)? };
    {
        let xp = tensor_ptr(&x)?;
        let (op, _o) = slice_ptr(&out, 0);
        let (dp, _d) = slice_ptr(&decoded, 0);
        let stream = dev.cuda_stream().cu_stream();
        // SAFETY: live device buffers sized as the kernels index them, on this device's stream
        let status = match ty {
            GgufType::Q6K => {
                let w = importance.map(|w| dev.clone_htod(w)).transpose()?;
                // SAFETY: the search kernel writes each before the super-block kernel reads it
                let ls = unsafe { dev.alloc::<i8>(rows * cols)? };
                // SAFETY: as above
                let scales = unsafe { dev.alloc::<f32>(rows * cols / Q6K_SUB)? };
                let w_ptr = w.as_ref().map(|w| slice_ptr(w, 0));
                let (lp, _l) = slice_ptr(&ls, 0);
                let (sp, _s) = slice_ptr(&scales, 0);
                unsafe {
                    launch_q6_k_encode(
                        xp as *const f32,
                        w_ptr
                            .as_ref()
                            .map_or(std::ptr::null(), |(p, _)| *p as *const f32),
                        rows as i64,
                        rows_per_importance as i64,
                        cols as i32,
                        lp as *mut i8,
                        sp as *mut f32,
                        op as *mut u8,
                        dp as *mut f32,
                        stream,
                    )
                }
            }
            _ => unsafe {
                launch_q8_0_encode(
                    xp as *const f32,
                    rows as i64,
                    cols as i32,
                    op as *mut u8,
                    dp as *mut f32,
                    stream,
                )
            },
        };
        if status != 0 {
            inference_tensor::bail!(
                "the {ty:?} block encoder failed to launch (CUDA error {status})"
            );
        }
    }
    let bytes = dev.clone_dtoh(&out)?;
    let decoded = Tensor::from((
        inference_tensor::Storage::Cuda(inference_tensor::CudaStorage::wrap_cuda_slice(
            decoded,
            dev.clone(),
        )),
        inference_tensor::Shape::from((rows, cols)),
    ));
    Ok((bytes, decoded))
}

/// Whether [`quantize_rows`] writes `ty` on the GPU.
pub fn can_quantize(ty: GgufType) -> bool {
    kernels(ty, QK_K).is_some() || blocks_supported(ty)
}

/// Rows of `cols` values (`source`, little-endian of ggml type `dtype`: f32, f16 or bf16) as `ty` on `dev`, the bytes
/// `iq_quantize::quantize_rows` writes with ik_llama.cpp's search; each `rows_per_imatrix` rows weighed by the next
/// `cols` of `imatrix` (an expert stack's experts in one call).
pub fn quantize_rows(
    dev: &CudaDevice,
    ty: GgufType,
    cols: usize,
    dtype: u32,
    source: &[u8],
    imatrix: Option<&[f32]>,
    rows_per_imatrix: usize,
) -> Result<Vec<u8>> {
    let width = match dtype {
        GGML_F32 => 4,
        GGML_F16 | GGML_BF16 => 2,
        other => inference_tensor::bail!(
            "the GPU quantizers read f32, f16 or bf16, not ggml type {other}"
        ),
    };
    let values = source.len() / width;
    if blocks_supported(ty) {
        let dtype = match dtype {
            GGML_F32 => DType::F32,
            GGML_F16 => DType::F16,
            _ => DType::BF16,
        };
        let shape = [values / cols.max(1), cols];
        let device = inference_tensor::Device::Cuda(dev.clone());
        let x = Tensor::from_raw_buffer(source, dtype, &shape, &device)?;
        return Ok(encode_blocks(dev, ty, &x, imatrix, rows_per_imatrix)?.0);
    }
    let Some((launch, row_bytes)) = kernels(ty, cols) else {
        inference_tensor::bail!("{ty:?} has no GPU quantizer");
    };
    if !cols.is_multiple_of(QK_K)
        || !values.is_multiple_of(cols)
        || !source.len().is_multiple_of(width)
    {
        inference_tensor::bail!("{ty:?} rows of {cols} values are not whole blocks of {QK_K}");
    }
    ensure_loaded().map_err(inference_tensor::Error::msg)?;
    let rows = values / cols;
    if let Some(w) = imatrix
        && (rows_per_imatrix == 0 || w.len() != rows.div_ceil(rows_per_imatrix) * cols)
    {
        inference_tensor::bail!(
            "an imatrix of {} for {rows} rows of {cols}, {rows_per_imatrix} a slice",
            w.len()
        );
    }
    let src = dev.clone_htod(source)?;
    // SAFETY: the conversion kernel writes every value before the others read them
    let x = unsafe { dev.alloc::<f32>(values)? };
    let w = imatrix.map(|w| dev.clone_htod(w)).transpose()?;
    // SAFETY: each scratch buffer is written by one kernel before a later one reads it
    let weights = unsafe { dev.alloc::<f32>(values)? };
    // SAFETY: as above
    let terms = unsafe { dev.alloc::<f32>(values)? };
    // SAFETY: as above
    let row_d = unsafe { dev.alloc::<f32>(rows)? };
    // SAFETY: as above
    let scales = unsafe { dev.alloc::<f32>(values / BLOCK)? };
    // SAFETY: as above
    let shifted = unsafe { dev.alloc::<u8>(values / BLOCK)? };
    // SAFETY: the row kernel writes each row's every byte
    let out = unsafe { dev.alloc::<u8>(rows * row_bytes)? };
    let (srcp, _src) = slice_ptr(&src, 0);
    let (xp, _x) = slice_ptr(&x, 0);
    let w_ptr = w.as_ref().map(|w| slice_ptr(w, 0));
    let (wp, _w) = slice_ptr(&weights, 0);
    let (tp, _t) = slice_ptr(&terms, 0);
    let (dp, _d) = slice_ptr(&row_d, 0);
    let (sp, _s) = slice_ptr(&scales, 0);
    let (fp, _f) = slice_ptr(&shifted, 0);
    let (op, _o) = slice_ptr(&out, 0);
    // SAFETY: the pointers are the live device buffers above, sized as the kernels index them, on this stream
    let status = unsafe {
        launch(
            srcp as *const u8,
            dtype as i32,
            xp as *mut f32,
            w_ptr
                .as_ref()
                .map_or(std::ptr::null(), |(p, _)| *p as *const f32),
            rows_per_imatrix as i64,
            rows as i64,
            cols as i32,
            wp as *mut f32,
            tp as *mut f32,
            sp as *mut f32,
            fp as *mut u8,
            dp as *mut f32,
            op as *mut u8,
            dev.cuda_stream().cu_stream(),
        )
    };
    if status != 0 {
        inference_tensor::bail!(
            "the {ty:?} quantize kernels failed to launch (CUDA error {status})"
        );
    }
    drop((_src, _x, w_ptr, _w, _t, _d, _s, _f, _o));
    dev.clone_dtoh(&out)
}

#[cfg(test)]
mod tests {
    use inference_tensor::quantized::Quantizer;
    use inference_tensor::{Device, Result};

    use super::*;
    use crate::gguf::iq_quantize;

    const GPU_TYPES: [GgufType; 4] = [
        GgufType::Iq4Ks,
        GgufType::Iq4Kss,
        GgufType::Iq4Xs,
        GgufType::Iq3Ks,
    ];

    // small values with sparse outliers, a zeroed block and a zeroed row, and a skewed imatrix
    fn layer(rows: usize, cols: usize, seed: u32) -> (Vec<f32>, Vec<f32>) {
        let mut state = seed;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / (1u32 << 24) as f32, state)
        };
        let mut values: Vec<f32> = (0..rows * cols)
            .map(|_| {
                let (u, s) = next();
                (u - 0.5) * if (s >> 4) & 31 == 0 { 0.4 } else { 0.04 }
            })
            .collect();
        values[BLOCK..2 * BLOCK].fill(0.0);
        if rows > 1 {
            values[cols..2 * cols].fill(0.0);
        }
        let imatrix = (0..cols).map(|_| 0.05 + 4.0 * next().0.powi(3)).collect();
        (values, imatrix)
    }

    fn bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn matches_the_cpu_port() -> Result<()> {
        let Device::Cuda(dev) = Device::new_cuda(0)? else {
            unreachable!("a CUDA device")
        };
        for ty in GPU_TYPES {
            for (rows, cols, seed) in [(64, 4096, 1), (3, 256, 2), (17, 2048, 3)] {
                let (values, imatrix) = layer(rows, cols, seed);
                for w in [None, Some(imatrix.as_slice())] {
                    let cpu = iq_quantize::quantize_rows(ty, cols, &values, w, Quantizer::IkLlama)?;
                    let gpu = quantize_rows(&dev, ty, cols, GGML_F32, &bytes(&values), w, rows)?;
                    let differing = cpu.iter().zip(&gpu).filter(|(a, b)| a != b).count();
                    assert_eq!(
                        (gpu.len(), differing),
                        (cpu.len(), 0),
                        "{ty:?} {rows}x{cols}, imatrix {}",
                        w.is_some()
                    );
                }
            }
        }
        Ok(())
    }

    // three experts of 8 rows in one call, each weighed by its own imatrix slice, as the CPU writes them one by one
    fn weighs_each_expert_by_its_slice() -> Result<()> {
        let Device::Cuda(dev) = Device::new_cuda(0)? else {
            unreachable!("a CUDA device")
        };
        for ty in GPU_TYPES {
            let (rows, cols, experts) = (8, 512, 3);
            let (values, _) = layer(rows * experts, cols, 4);
            let imatrix: Vec<f32> = (0..experts)
                .flat_map(|e| layer(1, cols, 10 + e as u32).1)
                .collect();
            let mut cpu = Vec::new();
            for e in 0..experts {
                cpu.extend(iq_quantize::quantize_rows(
                    ty,
                    cols,
                    &values[e * rows * cols..(e + 1) * rows * cols],
                    Some(&imatrix[e * cols..(e + 1) * cols]),
                    Quantizer::IkLlama,
                )?);
            }
            let gpu = quantize_rows(
                &dev,
                ty,
                cols,
                GGML_F32,
                &bytes(&values),
                Some(&imatrix),
                rows,
            )?;
            assert!(cpu == gpu, "{ty:?}");
        }
        Ok(())
    }

    // bf16 rows widen on the GPU to the f32 the CPU quantizes
    fn reads_bf16_rows() -> Result<()> {
        let Device::Cuda(dev) = Device::new_cuda(0)? else {
            unreachable!("a CUDA device")
        };
        for ty in GPU_TYPES {
            let (rows, cols) = (5, 512);
            let (values, imatrix) = layer(rows, cols, 5);
            let halves: Vec<half::bf16> = values.iter().map(|&v| half::bf16::from_f32(v)).collect();
            let widened: Vec<f32> = halves.iter().map(|h| h.to_f32()).collect();
            let source: Vec<u8> = halves.iter().flat_map(|h| h.to_le_bytes()).collect();
            let cpu =
                iq_quantize::quantize_rows(ty, cols, &widened, Some(&imatrix), Quantizer::IkLlama)?;
            let gpu = quantize_rows(&dev, ty, cols, GGML_BF16, &source, Some(&imatrix), rows)?;
            assert!(cpu == gpu, "{ty:?}");
        }
        Ok(())
    }

    // Q6_K and Q8_0 blocks as the CPU's `quantize_blocks` writes them, and the values as its `dequantize` reads them
    fn blocks_match_the_cpu() -> Result<()> {
        use inference_tensor::quantized::{GgmlDType, QStorage, QTensor, quantize_blocks};
        let device = Device::new_cuda(0)?;
        let Device::Cuda(dev) = &device else {
            unreachable!("a CUDA device")
        };
        for (ty, dtype) in [
            (GgufType::Q6K, GgmlDType::Q6K),
            (GgufType::Q8_0, GgmlDType::Q8_0),
        ] {
            for (rows, cols, seed) in [(64, 4096, 1), (3, 256, 2), (17, 2048, 3)] {
                let (values, imatrix) = layer(rows, cols, seed);
                let cpu =
                    quantize_blocks(dtype, &values, Some(&imatrix), cols, Quantizer::IkLlama)?;
                let x = Tensor::from_slice(&values, (rows, cols), &device)?;
                let (gpu, decoded) = encode_blocks(dev, ty, &x, Some(&imatrix), rows)?;
                let differing = cpu.iter().zip(&gpu).filter(|(a, b)| a != b).count();
                assert_eq!(
                    (gpu.len(), differing),
                    (cpu.len(), 0),
                    "{ty:?} {rows}x{cols}"
                );
                let storage =
                    QStorage::from_data(std::borrow::Cow::Borrowed(&cpu), &Device::Cpu, dtype)?;
                let expected = QTensor::new(storage, (rows, cols))?
                    .dequantize(&Device::Cpu)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let got = decoded.flatten_all()?.to_vec1::<f32>()?;
                let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
                assert!(
                    bits(&expected) == bits(&got),
                    "{ty:?} {rows}x{cols}: decoded values"
                );
                // without an importance, from bf16 rows, as a tensor without a Hessian is written
                let halves: Vec<half::bf16> =
                    values.iter().map(|&v| half::bf16::from_f32(v)).collect();
                let widened: Vec<f32> = halves.iter().map(|h| h.to_f32()).collect();
                let source: Vec<u8> = halves.iter().flat_map(|h| h.to_le_bytes()).collect();
                let cpu = quantize_blocks(dtype, &widened, None, cols, Quantizer::IkLlama)?;
                let gpu = quantize_rows(dev, ty, cols, GGML_BF16, &source, None, rows)?;
                assert!(cpu == gpu, "{ty:?} {rows}x{cols}: bf16 rows, no importance");
            }
        }
        Ok(())
    }

    inference_tensor::gpu_cases!(
        cuda_cases,
        [
            matches_the_cpu_port,
            weighs_each_expert_by_its_slice,
            reads_bf16_rows,
            blocks_match_the_cpu
        ]
    );
}
