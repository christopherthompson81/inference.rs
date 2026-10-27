//! Models from other families: BERT-family, Nomic BERT, ModernBERT and NeoBERT encoders, DeepSeek 2/3, GLM-4 and its MoE variants, GPT-OSS, Granite, Hunyuan, LFM2 and StarCoder2 text models, and LFM2-VL and PaddleOCR-VL.
#![deny(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use inference_nn::serde_default_fn;
use inference_nn::{
    amoe, attention, cuda, decoder, device_map, embedding, gdn, kv_cache, layers, media_inputs,
    metal, mla, model, moe, ops, paged_attention, speculative, utils,
};

pub mod bert;
pub mod deepseek2;
pub mod deepseek3;
pub mod deepseek_family;
#[cfg(test)]
mod deepseek_family_tests;
pub mod glm4;
pub mod glm4_moe;
pub mod glm4_moe_lite;
#[cfg(test)]
mod glm4_tests;
pub mod gpt_oss;
pub mod granite;
#[cfg(test)]
mod hunyuan_dense_tests;
mod hunyuan_rope;
pub mod hunyuan_v1_dense;
pub mod hunyuan_v1_moe;
pub mod jina_v3;
pub mod lfm2;
pub mod lfm2_vl;
pub mod loaders;
pub mod modern_bert;
pub mod neo_bert;
pub mod nomic_bert;
pub mod paddleocr_vl;
#[cfg(test)]
mod paddleocr_vl_tests;
mod rope_encoder;
pub mod starcoder2;
#[cfg(test)]
mod starcoder2_tests;

inference_nn::json_config!(
    bert::Config,
    jina_v3::Config,
    modern_bert::Config,
    neo_bert::Config,
    nomic_bert::Config,
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
    lfm2_vl::config::Config,
    paddleocr_vl::config::Config,
    starcoder2::Config,
);
