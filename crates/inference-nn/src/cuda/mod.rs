#[cfg(feature = "cuda")]
pub mod adain;
#[cfg(feature = "cuda")]
pub mod dflash_context;
#[cfg(feature = "cuda")]
pub mod dynamic_conv;
pub mod ffi;
#[cfg(feature = "cuda")]
pub mod hyper;
#[cfg(feature = "cuda")]
pub mod lstm;
#[cfg(feature = "cuda")]
pub mod mapped;
pub mod qsa;

/// Opens every CUDA kernel library a model can launch, so a missing one fails at model load rather than mid-run.
#[cfg(feature = "cuda")]
pub fn ensure_kernels_loaded() -> std::result::Result<(), String> {
    ffi::ensure_loaded()?;
    inference_quant::ensure_kernels_loaded()?;
    inference_fattn::ensure_kernels_loaded()?;
    #[cfg(target_family = "unix")]
    inference_paged_attn::ensure_kernels_loaded()?;
    Ok(())
}
pub mod gdn;
#[cfg(feature = "cuda")]
pub mod graph;
#[cfg(feature = "cuda")]
pub mod graph_capture;
#[cfg(feature = "cuda")]
pub mod indexed_copy;
#[cfg(feature = "cuda")]
pub mod input_packing;
pub mod moe;
#[cfg(feature = "cuda")]
pub mod phase_timer;
#[cfg(feature = "cuda")]
pub mod preload;
#[cfg(all(test, feature = "cuda"))]
mod rng_tests;
#[cfg(feature = "cuda")]
pub mod speculative_rejection;
pub mod ssm;
