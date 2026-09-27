//! Models from other families: DeepSeek 2/3, GLM-4 and its MoE variants, GPT-OSS, Granite, Hunyuan, LFM2 and StarCoder2 text models, and LFM2-VL and PaddleOCR-VL.
#![deny(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use inference_nn::serde_default_fn;
use inference_nn::{
    amoe, attention, cuda, device_map, gdn, kv_cache, layers, lora, metal, mla, model, moe, ops,
    paged_attention, speculative, utils, vision,
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
pub mod lfm2_vl;
pub mod paddleocr_vl;
pub mod starcoder2;
pub mod xlora;

inference_nn::json_config!(
    deepseek2::DeepSeekV2Config,
    deepseek3::DeepSeekV3Config,
    glm4::Config,
    glm4_moe::Glm4MoeConfig,
    glm4_moe_lite::Glm4MoeLiteConfig,
    gpt_oss::Config,
    granite::Config,
    hunyuan_v1_dense::Config,
    hunyuan_v1_moe::Config,
    lfm2::Config,
    starcoder2::Config,
);
