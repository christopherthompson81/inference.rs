use std::sync::Arc;

use candle_core::{shape::Dim, DType, Result, Tensor, D};

#[cfg(feature = "cuda")]
use crate::cuda::ffi;

use crate::layers::Activation;

#[cfg(feature = "cuda")]
use candle_core::Shape;

#[cfg(feature = "cuda")]
const CUDA_TOPK_CHUNK_SIZE: usize = 2048;

#[cfg(feature = "cuda")]
pub(crate) const CUDA_TOPK_MAX_EXACT_PACKED_VOCAB: usize = (1 << 24) + 1;

#[cfg(feature = "cuda")]
const CUDA_TOPK_MAX_GRID_Y: usize = 65_535;

#[cfg(feature = "cuda")]
pub(crate) const CUDA_TOPK_MAX_K: usize = 128;

#[cfg(feature = "cuda")]
const CUDA_TOPK_MAX_STAGE2_CANDIDATES: usize = 47 * 1024;

#[cfg(feature = "cuda")]
pub(crate) const CUDA_CATEGORICAL_PACKED_WIDTH: usize = 2;

#[cfg(feature = "cuda")]
pub(crate) const CUDA_TOP1_PACKED_WIDTH: usize = 2;

#[cfg(feature = "cuda")]
pub(crate) const CUDA_TOP1_INVALID_TOKEN: u32 = u32::MAX;

#[cfg(feature = "cuda")]
const CUDA_ASYNC_TOKEN_RING_SLOTS: usize = 2;

#[cfg(feature = "cuda")]
const CUDA_TOPK_SAMPLING_PARAM_WIDTH: usize = 5;

#[cfg(feature = "cuda")]
pub(crate) const CUDA_DFLASH_SELECTOR_MAX_K: usize = 128;

#[cfg(all(feature = "cuda", test))]
const CUDA_DFLASH_SELECTOR_INVALID_TOKEN: u32 = u32::MAX;

#[cfg(feature = "cuda")]
const CUDA_DFLASH_SELECTOR_F32: i32 = 0;

#[cfg(feature = "cuda")]
const CUDA_DFLASH_SELECTOR_BF16: i32 = 1;

#[cfg(feature = "cuda")]
pub(crate) fn cuda_topk_ranked_packed_max_k(vocab: usize) -> Option<usize> {
    if vocab == 0 || vocab > CUDA_TOPK_MAX_EXACT_PACKED_VOCAB {
        return None;
    }
    let chunks = vocab.div_ceil(CUDA_TOPK_CHUNK_SIZE);
    let workspace_bound = CUDA_TOPK_MAX_STAGE2_CANDIDATES.checked_div(chunks)?;
    let max_k = vocab.min(CUDA_TOPK_MAX_K).min(workspace_bound);
    (max_k > 0).then_some(max_k)
}

mod topk;
pub use topk::*;
mod moe_router;
pub use moe_router::*;
#[cfg(any(feature = "cuda", feature = "metal"))]
mod topk_logits;
#[cfg(any(feature = "cuda", feature = "metal"))]
pub use topk_logits::*;
#[cfg(feature = "cuda")]
mod dflash_select;
#[cfg(feature = "cuda")]
pub(crate) use dflash_select::*;
#[cfg(feature = "cuda")]
mod top1;
#[cfg(feature = "cuda")]
pub use top1::*;
#[cfg(feature = "cuda")]
mod topk_sampling;
#[cfg(feature = "cuda")]
pub use topk_sampling::*;
#[cfg(any(feature = "cuda", feature = "metal"))]
mod logit_processing;
#[cfg(any(feature = "cuda", feature = "metal"))]
pub use logit_processing::*;
#[cfg(any(feature = "cuda", feature = "metal"))]
mod norm;
#[cfg(any(feature = "cuda", feature = "metal"))]
pub use norm::*;
#[cfg(feature = "cuda")]
mod rope;
#[cfg(feature = "cuda")]
pub(crate) use rope::*;
mod tensor_traits;
pub use tensor_traits::*;
mod projection;
pub use projection::*;

#[cfg(test)]
mod tests;
