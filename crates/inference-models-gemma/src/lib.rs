//! Gemma family: the Gemma and Gemma 2 text models, Gemma 3, 3n and 4, and DiffusionGemma.
#![deny(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use inference_nn::serde_default_fn;
use inference_nn::{
    amoe, attention, decoder, device_map, flashinfer, gdn, kv_cache, layers, matformer,
    media_inputs, model, moe, ops, paged_attention, speculative, utils, vision,
};

pub mod diffusion_gemma;
pub mod embedding_gemma;
pub mod gemma;
pub mod gemma2;
pub mod gemma3;
pub mod gemma3n;
pub mod gemma4;
#[cfg(test)]
mod gemma_dense_tests;
pub mod loaders;

inference_nn::json_config!(
    gemma::Config,
    gemma2::Config,
    gemma3::config::Gemma3Config,
    gemma3n::config::Gemma3nConfig,
    gemma4::config::Gemma4Config,
    diffusion_gemma::config::DiffusionGemmaConfig,
    gemma3::config::Gemma3TextConfig,
);
