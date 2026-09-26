//! Model-facing building blocks of `inference.rs`: layers, attention, KV and paged caches, GDN, MoE, and kernels.
#![deny(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

// GPU tests share one device and process-global CUDA state (memory pools, graph scopes), so they run one at a time.
#[cfg(all(test, feature = "cuda"))]
static CUDA_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

// GPU tests run by default under `--features cuda`, skip without a device, and hold the lock for the test's lifetime.
#[cfg(all(test, feature = "cuda"))]
macro_rules! skip_without_cuda {
    () => {
        let _cuda_test_guard = crate::CUDA_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if candle_core::Device::new_cuda(0).is_err() {
            eprintln!("SKIP {}: no CUDA device", module_path!());
            return Ok(());
        }
    };
}

pub mod amoe;
pub mod attention;
pub mod cuda;
pub mod device_map;
pub mod flashinfer;
pub mod gdn;
pub mod kv_cache;
pub mod layers;
pub mod lora;
pub mod matformer;
pub mod metal;
pub mod mla;
pub mod moe;
pub mod ops;
pub mod paged_attention;
pub mod perf_flags;
pub mod sampler;
pub mod speculative;
pub mod topology;
pub mod utils;

/// CUDA toolkit this crate's kernels were built with, as `major.minor`.
pub const BUILD_CUDA_VERSION: Option<&str> = option_env!("INFERENCE_RS_BUILD_CUDA_VERSION");
/// Same as [`BUILD_CUDA_VERSION`], as `major * 100 + minor`.
pub const BUILD_CUDA_VERSION_CODE: Option<&str> =
    option_env!("INFERENCE_RS_BUILD_CUDA_VERSION_CODE");
