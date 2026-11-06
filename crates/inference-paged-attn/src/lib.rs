#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KvCacheScales {
    pub k: f32,
    pub v: f32,
}

pub const DEFAULT_FP8_KV_CACHE_SCALES: KvCacheScales = KvCacheScales { k: 1.0, v: 1.0 };

/// Head sizes Metal's paged decode kernel is instantiated for.
pub const METAL_PAGED_HEAD_SIZES: [usize; 8] = [64, 80, 96, 112, 128, 192, 256, 512];

pub mod cpu;
pub mod kv_quant;

#[cfg(not(any(all(feature = "cuda", target_family = "unix"), feature = "metal")))]
pub use cpu::{copy_blocks, gather_kv_cache, kv_scale_update, paged_attention, reshape_and_cache};

#[cfg(all(feature = "cuda", target_family = "unix"))]
mod cuda;
#[cfg(all(feature = "cuda", target_family = "unix"))]
pub use cuda::*;
// a CUDA build's Vulkan device runs the shared module's kernels, which hand Vulkan tensors to its backend
#[cfg(all(feature = "cuda", target_family = "unix"))]
pub use vulkan_or_cuda::{copy_blocks, gather_kv_cache, paged_attention, reshape_and_cache};

#[cfg(all(feature = "cuda", target_family = "unix"))]
mod vulkan_or_cuda {
    use std::collections::HashMap;

    use inference_tensor::{DType, Result, Tensor};

    pub fn reshape_and_cache(
        key: &Tensor,
        value: &Tensor,
        k_scale: Option<&Tensor>,
        v_scale: Option<&Tensor>,
        key_cache: &Tensor,
        value_cache: &Tensor,
        slot_mapping: &Tensor,
    ) -> Result<()> {
        let write = if key_cache.device().is_vulkan() {
            crate::cpu::reshape_and_cache
        } else {
            crate::cuda::reshape_and_cache
        };
        write(
            key,
            value,
            k_scale,
            v_scale,
            key_cache,
            value_cache,
            slot_mapping,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gather_kv_cache(
        key_cache: &Tensor,
        value_cache: &Tensor,
        k_scale: Option<&Tensor>,
        v_scale: Option<&Tensor>,
        block_table: &Tensor,
        cu_seq_lens: &Tensor,
        num_tokens: usize,
        out_dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        let gather = if key_cache.device().is_vulkan() {
            crate::cpu::gather_kv_cache
        } else {
            crate::cuda::gather_kv_cache
        };
        gather(
            key_cache,
            value_cache,
            k_scale,
            v_scale,
            block_table,
            cu_seq_lens,
            num_tokens,
            out_dtype,
        )
    }

    pub fn copy_blocks(
        key_caches: Vec<&mut Tensor>,
        value_caches: Vec<&mut Tensor>,
        block_mapping: &HashMap<usize, Vec<usize>>,
    ) -> Result<()> {
        let vulkan = key_caches.first().is_some_and(|c| c.device().is_vulkan());
        let copy = if vulkan {
            crate::cpu::copy_blocks
        } else {
            crate::cuda::copy_blocks
        };
        copy(key_caches, value_caches, block_mapping)
    }

    /// The Standard layout's paged decode, which on a CUDA build only a Vulkan device runs.
    pub use crate::cpu::paged_attention;
}

#[cfg(feature = "metal")]
mod metal;
#[cfg(feature = "metal")]
pub use metal::*;
