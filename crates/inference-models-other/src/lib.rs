//! Models from other families: BERT-family, Nomic BERT, ModernBERT and NeoBERT encoders, DeepSeek 2/3, ERNIE 4.5 (dense and MoE), EXAONE 3.x and 4.0, K-EXAONE, GLM-4 and its MoE variants, GPT-OSS, Granite, Hunyuan, Inkling 2, LFM2, Seed-OSS and StarCoder2 text models, and LFM2-VL and PaddleOCR-VL.
#![deny(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use inference_nn::serde_default_fn;
use inference_nn::{
    amoe, attention, cuda, decision, decoder, device_map, embedding, gdn, kv_cache, layers,
    media_inputs, metal, mla, model, moe, ops, paged_attention, speculative, utils,
};

pub mod bert;
pub mod blip2_qformer;
pub mod classifier;
pub mod deepseek;
pub mod deepseek_family;
#[cfg(test)]
mod deepseek_family_tests;
pub mod deepseek_ocr;
pub mod ernie4_5;
#[cfg(test)]
mod ernie4_5_tests;
pub mod exaone;
pub mod exaone4;
pub mod exaone_moe;
pub mod falcon_h1;
pub mod glm4;
pub mod glm4_moe;
pub mod glm4_moe_lite;
#[cfg(test)]
mod glm4_tests;
pub mod gpt_oss;
pub mod granite;
pub mod granite4_vision;
pub mod granite_speech;
#[cfg(test)]
mod hunyuan_dense_tests;
mod hunyuan_rope;
pub mod hunyuan_v1_dense;
pub mod hunyuan_v1_moe;
pub mod inkling2;
pub mod jamba;
pub mod jina_v3;
pub mod kimi_linear;
pub mod lfm2;
pub mod lfm2_decision;
pub mod lfm2_vl;
pub mod loaders;
pub mod mamba;
pub mod mamba2;
pub mod modern_bert;
pub mod modern_bert_decision;
pub mod nemotron_h;
pub mod nemotron_omni;
pub mod nemotron_transcribe;
pub mod neo_bert;
pub mod nomic_bert;
pub mod olmo;
pub mod paddleocr_vl;
#[cfg(test)]
mod paddleocr_vl_tests;
mod rope_encoder;
pub mod rwkv;
pub mod rwkv6;
pub mod rwkv7;
pub mod rwkv_hybrid;
pub mod seed_oss;
pub mod starcoder2;
#[cfg(test)]
mod starcoder2_tests;

inference_nn::json_config!(
    bert::Config,
    jamba::Config,
    kimi_linear::Config,
    inkling2::Config,
    jina_v3::Config,
    modern_bert::Config,
    falcon_h1::Config,
    nemotron_h::Config,
    nemotron_omni::Config,
    nemotron_transcribe::Config,
    neo_bert::Config,
    nomic_bert::Config,
    deepseek::DeepSeekConfig,
    deepseek_ocr::config::Config,
    ernie4_5::Config,
    ernie4_5::MoeConfig,
    glm4::Config,
    glm4_moe::Glm4MoeConfig,
    glm4_moe_lite::Glm4MoeLiteConfig,
    gpt_oss::Config,
    granite::Config,
    granite4_vision::config::Config,
    granite_speech::config::Config,
    hunyuan_v1_dense::Config,
    hunyuan_v1_moe::Config,
    lfm2::Config,
    mamba::Config,
    mamba2::Config,
    olmo::OlmoConfig,
    olmo::Olmo2Config,
    olmo::OlmoeConfig,
    lfm2_vl::config::Config,
    paddleocr_vl::config::Config,
    seed_oss::Config,
    exaone::Config,
    exaone_moe::Config,
    rwkv6::Config,
    rwkv7::Config,
    rwkv_hybrid::ArwkvConfig,
    rwkv_hybrid::Qrwkv6Config,
    starcoder2::Config,
);
