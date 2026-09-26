//! Gemma family: the Gemma and Gemma 2 text models, Gemma 3, 3n and 4, and DiffusionGemma.
#![deny(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use inference_nn::serde_default_fn;
use inference_nn::{
    amoe, attention, device_map, kv_cache, layers, matformer, model, moe, ops, paged_attention,
    perf_flags, speculative, utils, vision,
};

pub mod diffusion_gemma;
pub mod gemma;
pub mod gemma2;
pub mod gemma3;
pub mod gemma3n;
pub mod gemma4;
