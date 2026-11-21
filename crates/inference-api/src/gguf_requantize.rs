//! A float GGUF rewritten with per-tensor types and an imatrix, a slice at a time.

pub use inference_core::{
    HessianDir, ImatrixMapping, MmprojSummary, cimatrix_for_gguf, hessians_for_gguf,
    mmproj_architectures, write_mmproj, write_nemotron_transcribe_audio_gguf,
};
pub use inference_quant::requantize::{
    DEFAULT_GPTQ_DAMP, HessianFile, HessianSource, Quantizer, RequantizePlan, RequantizeSummary,
    load_imatrix, save_legacy_imatrix, writable_type,
};

/// `src` rewritten to `dst` as `plan` says ([`inference_quant::requantize::requantize_gguf`]), its GPU's memory pool
/// kept from returning freed buffers to the driver meanwhile.
pub fn requantize_gguf(
    src: &std::path::Path,
    dst: &std::path::Path,
    plan: &RequantizePlan,
    progress: impl FnMut(usize, usize, &str),
) -> inference_tensor::Result<RequantizeSummary> {
    let _pool = plan.gpu.as_ref().map(inference_core::retain_memory_pool);
    inference_quant::requantize::requantize_gguf(src, dst, plan, progress)
}
