//! Gemma text models: Gemma and Gemma 2.
#![deny(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use inference_nn::serde_default_fn;
use inference_nn::{
    amoe, attention, device_map, kv_cache, layers, model, ops, paged_attention, speculative, utils,
};

pub mod gemma;
pub mod gemma2;
