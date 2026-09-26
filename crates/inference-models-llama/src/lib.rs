//! Llama-family text models: Llama, Mistral, Mixtral and SmolLM3.
#![deny(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use inference_nn::serde_default_fn;
use inference_nn::{
    amoe, attention, device_map, kv_cache, layers, model, moe, ops, paged_attention, speculative,
    utils,
};

pub mod llama;
pub mod mistral;
pub mod mixtral;
pub mod smollm3;
