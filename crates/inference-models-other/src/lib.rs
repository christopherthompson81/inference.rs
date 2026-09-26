//! Text models from other families: DeepSeek 2/3, GLM-4 and its MoE variants, GPT-OSS, Granite, Hunyuan, LFM2 and StarCoder2.
#![deny(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use inference_nn::serde_default_fn;
use inference_nn::{
    amoe, attention, cuda, device_map, gdn, kv_cache, layers, metal, mla, model, moe, ops,
    paged_attention, speculative, utils,
};

pub mod deepseek2;
pub mod deepseek3;
pub mod glm4;
pub mod glm4_moe;
pub mod glm4_moe_lite;
pub mod gpt_oss;
pub mod granite;
mod hunyuan_rope;
pub mod hunyuan_v1_dense;
pub mod hunyuan_v1_moe;
pub mod lfm2;
pub mod starcoder2;
