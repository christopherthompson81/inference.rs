//! Qwen models: the Qwen2, Qwen3, Qwen3-MoE and Qwen3-Next text models, and Qwen2-VL, Qwen2.5-VL, Qwen3-VL (dense and MoE), MiniCPM-o and Muse-Glimmer.
#![deny(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use inference_nn::serde_default_fn;
use inference_nn::{
    amoe, attention, device_map, gdn, kv_cache, layers, model, moe, ops, paged_attention,
    speculative, utils, vision,
};

pub mod minicpmo;
pub mod muse_glimmer;
pub mod qwen2;
pub mod qwen2_5_vl;
pub mod qwen2vl;
pub mod qwen3;
pub mod qwen3_moe;
pub mod qwen3_next;
pub mod qwen3_vl;
pub mod qwen3_vl_moe;

inference_nn::json_config!(
    qwen2::Config,
    qwen3::Config,
    qwen3_moe::Config,
    qwen3_next::Config,
);
