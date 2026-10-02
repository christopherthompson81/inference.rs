//! Qwen models: the Qwen2, Qwen3, Qwen3-MoE and Qwen3-Next text models, Qwen2-VL, Qwen2.5-VL, Qwen3-VL (dense and MoE),
//! Qwen3.5 (dense and MoE) with its DFlash drafter, MiniCPM-o and Muse-Glimmer.
#![deny(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use inference_nn::{
    amoe, attention, cuda, device_map, gdn, kv_cache, layers, media_inputs, model, moe, ops,
    paged_attention, speculative, utils, vision,
};
use inference_nn::{get_mut_arcmutex, serde_default_fn};

pub mod dflash;
pub mod loaders;
pub mod minicpmo;
pub mod muse_glimmer;
pub mod qwen2;
pub mod qwen2_5_vl;
pub mod qwen2vl;
pub mod qwen3;
pub mod qwen3_5;
pub mod qwen3_5_moe;
pub mod qwen3_embedding;
pub mod qwen3_moe;
#[cfg(test)]
mod qwen3_moe_tests;
pub mod qwen3_next;
pub mod qwen3_vl;
pub mod qwen3_vl_moe;
#[cfg(test)]
mod qwen_vl_tests;

inference_nn::json_config!(
    minicpmo::config::MiniCpmOConfig,
    muse_glimmer::config::Config,
    qwen2::Config,
    qwen2_5_vl::config::Config,
    qwen2vl::config::Config,
    qwen3::Config,
    qwen3_5::config::Config,
    qwen3_5::config::TextConfig,
    qwen3_5_moe::config::Config,
    qwen3_embedding::Config,
    qwen3_moe::Config,
    qwen3_next::Config,
    qwen3_vl::config::Config,
);
