//! Phi models: Phi-2, Phi-3 and Phi-3.5-MoE text, Phi-3V, and Phi-4 multimodal with its conformer audio encoder.
#![deny(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use inference_nn::serde_default_fn;
use inference_nn::{
    amoe, attention, decoder, device_map, gdn, kv_cache, layers, media_inputs, model, moe, ops,
    paged_attention, speculative, utils, vision,
};

pub mod conformer;
pub mod loaders;
pub mod phi2;
#[cfg(test)]
mod phi2_tests;
pub mod phi3;
pub mod phi3_5_moe;
pub mod phi3_vision;
#[cfg(test)]
mod phi3v_tests;
pub mod phi4;

inference_nn::json_config!(
    phi2::Config,
    phi3::Config,
    phi3_5_moe::Config,
    phi3_vision::Config,
    phi4::config::Phi4MMConfig,
);
