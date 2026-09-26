//! Qwen text models: Qwen2, Qwen3, Qwen3-MoE and Qwen3-Next.
#![deny(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use inference_nn::serde_default_fn;
use inference_nn::{
    amoe, attention, device_map, gdn, kv_cache, layers, model, moe, ops, paged_attention,
    speculative, utils,
};

pub mod qwen2;
pub mod qwen3;
pub mod qwen3_moe;
pub mod qwen3_next;
