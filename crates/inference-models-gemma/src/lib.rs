//! Gemma family: the Gemma and Gemma 2 text models, Gemma 3, 3n and 4, DiffusionGemma, and MERaLiON-2 on Gemma 2.
#![deny(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use inference_nn::serde_default_fn;
use inference_nn::{
    amoe, attention, audio, decoder, device_map, embedding, flashinfer, gdn, kv_cache, layers,
    matformer, media_inputs, model, moe, ops, paged_attention, speculative, utils, vision,
};

mod audio_attention;
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
pub mod meralion2;

inference_nn::json_config!(
    gemma::Config,
    gemma2::Config,
    gemma3::config::Gemma3Config,
    gemma3n::config::Gemma3nConfig,
    gemma4::config::Gemma4Config,
    diffusion_gemma::config::DiffusionGemmaConfig,
    gemma3::config::Gemma3TextConfig,
    meralion2::MERaLiON2Config,
);
