//! Llama-family models: the Llama, Mistral, Mixtral and SmolLM3 text models, and LLaVA, Idefics 2/3, Mistral 3, Mllama, Llama 4 and Voxtral.
#![deny(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use inference_nn::serde_default_fn;
use inference_nn::{
    amoe, attention, decoder, device_map, gdn, gguf, kv_cache, layers, media_inputs, model, moe,
    ops, paged_attention, speculative, utils, vision,
};

#[cfg(test)]
mod dense_decoder_tests;
pub mod idefics2;
pub mod idefics3;
#[cfg(test)]
mod idefics_vision_tests;
pub mod llama;
pub mod llama4;
pub mod llava;
#[cfg(test)]
mod llava_tests;
pub mod loaders;
pub mod mistral;
pub mod mistral3;
pub mod mixtral;
pub mod mllama;
pub mod quantized_llama;
pub mod smollm3;
pub mod voxtral;

inference_nn::json_config!(
    idefics2::Config,
    idefics3::config::Idefics3Config,
    llama4::config::Llama4Config,
    llama::Config,
    llava::config::Config,
    mistral3::config::Mistral3Config,
    mistral::Config,
    mixtral::Config,
    mllama::config::MLlamaConfig,
    smollm3::Config,
    voxtral::config::VoxtralConfig,
);
