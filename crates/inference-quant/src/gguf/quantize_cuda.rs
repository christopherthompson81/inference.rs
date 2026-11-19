//! GGUF quantizers on the GPU (`kernels/cuda/gguf_quantize/*_instance_ggufq.cu`), writing the bytes the CPU ports in
//! `iq_quantize` write: every product, sum and fused multiply-add there is an explicit rounding intrinsic here.

use inference_tensor::{CudaDevice, Result};

use super::kernel::GgufType;
use crate::utils::slice_ptr;

const QK_K: usize = 256;
const BLOCK: usize = 32;
const IQ4_KS_ROW_HEADER: usize = 4;
const BLOCK_IQ4_KS: usize = QK_K / BLOCK + QK_K / 2;
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
    }
}

/// Whether [`quantize_rows`] writes `ty` on the GPU.
pub fn can_quantize(ty: GgufType) -> bool {
    ty == GgufType::Iq4Ks
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
    if !can_quantize(ty) {
        inference_tensor::bail!("{ty:?} has no GPU quantizer");
    }
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
    let row_bytes = IQ4_KS_ROW_HEADER + cols / QK_K * BLOCK_IQ4_KS;
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
        launch_iq4_ks_quantize(
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
            "the IQ4_KS quantize kernels failed to launch (CUDA error {status})"
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

    fn iq4_ks_matches_the_cpu_port() -> Result<()> {
        let Device::Cuda(dev) = Device::new_cuda(0)? else {
            unreachable!("a CUDA device")
        };
        for (rows, cols, seed) in [(64, 4096, 1), (3, 256, 2), (17, 2048, 3)] {
            let (values, imatrix) = layer(rows, cols, seed);
            for w in [None, Some(imatrix.as_slice())] {
                let cpu = iq_quantize::quantize_rows(
                    GgufType::Iq4Ks,
                    cols,
                    &values,
                    w,
                    Quantizer::IkLlama,
                )?;
                let gpu = quantize_rows(
                    &dev,
                    GgufType::Iq4Ks,
                    cols,
                    GGML_F32,
                    &bytes(&values),
                    w,
                    rows,
                )?;
                let differing = cpu.iter().zip(&gpu).filter(|(a, b)| a != b).count();
                assert_eq!(
                    (gpu.len(), differing),
                    (cpu.len(), 0),
                    "{rows}x{cols}, imatrix {}",
                    w.is_some()
                );
            }
        }
        Ok(())
    }

    // three experts of 8 rows in one call, each weighed by its own imatrix slice, as the CPU writes them one by one
    fn iq4_ks_weighs_each_expert_by_its_slice() -> Result<()> {
        let Device::Cuda(dev) = Device::new_cuda(0)? else {
            unreachable!("a CUDA device")
        };
        let (rows, cols, experts) = (8, 512, 3);
        let (values, _) = layer(rows * experts, cols, 4);
        let imatrix: Vec<f32> = (0..experts)
            .flat_map(|e| layer(1, cols, 10 + e as u32).1)
            .collect();
        let mut cpu = Vec::new();
        for e in 0..experts {
            cpu.extend(iq_quantize::quantize_rows(
                GgufType::Iq4Ks,
                cols,
                &values[e * rows * cols..(e + 1) * rows * cols],
                Some(&imatrix[e * cols..(e + 1) * cols]),
                Quantizer::IkLlama,
            )?);
        }
        let gpu = quantize_rows(
            &dev,
            GgufType::Iq4Ks,
            cols,
            GGML_F32,
            &bytes(&values),
            Some(&imatrix),
            rows,
        )?;
        assert!(cpu == gpu);
        Ok(())
    }

    // bf16 rows widen on the GPU to the f32 the CPU quantizes
    fn iq4_ks_reads_bf16_rows() -> Result<()> {
        let Device::Cuda(dev) = Device::new_cuda(0)? else {
            unreachable!("a CUDA device")
        };
        let (rows, cols) = (5, 512);
        let (values, imatrix) = layer(rows, cols, 5);
        let halves: Vec<half::bf16> = values.iter().map(|&v| half::bf16::from_f32(v)).collect();
        let widened: Vec<f32> = halves.iter().map(|h| h.to_f32()).collect();
        let source: Vec<u8> = halves.iter().flat_map(|h| h.to_le_bytes()).collect();
        let cpu = iq_quantize::quantize_rows(
            GgufType::Iq4Ks,
            cols,
            &widened,
            Some(&imatrix),
            Quantizer::IkLlama,
        )?;
        let gpu = quantize_rows(
            &dev,
            GgufType::Iq4Ks,
            cols,
            GGML_BF16,
            &source,
            Some(&imatrix),
            rows,
        )?;
        assert!(cpu == gpu);
        Ok(())
    }

    inference_tensor::gpu_cases!(
        cuda_cases,
        [
            iq4_ks_matches_the_cpu_port,
            iq4_ks_weighs_each_expert_by_its_slice,
            iq4_ks_reads_bf16_rows
        ]
    );
}
