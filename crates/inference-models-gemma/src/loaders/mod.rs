//! The gemma family's loaders: config parsing, weight sizing, ISQ patterns and prompt prefixes.

use std::borrow::Cow;
use std::fmt::Debug;
use std::sync::Arc;

use anyhow::Result;
use inference_nn::attention::ATTENTION_CHUNK_SIZE;
use inference_nn::bias_if;
use inference_nn::device_map::AutoDeviceMapParams;
use inference_nn::loaders::*;
use inference_nn::matformer::MatformerSliceConfig;
use inference_nn::model::{EmbeddingModel, MultimodalModel, NormalLoadingMetadata, NormalModel};
use inference_nn::paged_attention::{
    AttentionImplementation, ModelConfigLike, ModelConfigMetadata,
};
use inference_quant::ShardedVarBuilder;
use inference_tensor::DType;
use inference_tensor::nn::Conv2dConfig;
use regex::Regex;

use crate::diffusion_gemma::{DiffusionGemmaConfig, DiffusionGemmaModel};
use crate::gemma3::Gemma3Model;
use crate::gemma3::config::Gemma3Config;
use crate::gemma3n::Gemma3nModel;
use crate::gemma3n::config::{Gemma3nConfig, IntermediateSize};
use crate::gemma4::Gemma4Model;
use crate::gemma4::config::Gemma4Config;

mod diffusion_gemma;
mod embedding_gemma;
mod gemma;
mod gemma2;
mod gemma3;
mod gemma3n;
mod gemma4;
#[cfg(test)]
mod sizing_tests;

pub use diffusion_gemma::*;
pub use embedding_gemma::*;
pub use gemma::*;
pub use gemma2::*;
pub use gemma3::*;
pub use gemma3n::*;
pub use gemma4::*;

fn supports_gemma4_incremental_cache(config: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(config)
        .ok()
        .and_then(|config| {
            config
                .pointer("/text_config/use_bidirectional_attention")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned)
        })
        .as_deref()
        != Some("all")
}

inference_nn::boxed_loaders!(NormalModelLoader: Gemma2Loader, GemmaLoader);
inference_nn::boxed_loaders!(
    MultimodalModelLoader:
    DiffusionGemmaLoader,
    Gemma3Loader,
    Gemma3nLoader,
    Gemma4Loader,
);
inference_nn::boxed_loaders!(EmbeddingModelLoader: EmbeddingGemmaLoader);
