//! Phi text models: Phi-2, Phi-3 and Phi-3.5-MoE.
#![deny(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use inference_nn::serde_default_fn;
use inference_nn::{
    amoe, attention, device_map, kv_cache, layers, model, moe, ops, paged_attention, speculative,
    utils,
};

pub mod phi2;
pub mod phi3;
pub mod phi3_5_moe;
