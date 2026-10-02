//! The llama family's loaders: config parsing, weight sizing, ISQ patterns and prompt prefixes.

use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;

use anyhow::Result;
use candle_core::DType;
use candle_nn::Conv2dConfig;
use inference_nn::attention::ATTENTION_CHUNK_SIZE;
use inference_nn::bias_if;
use inference_nn::device_map::AutoDeviceMapParams;
use inference_nn::loaders::*;
use inference_nn::lora::{LoraConfig, Ordering};
use inference_nn::matformer::MatformerSliceConfig;
use inference_nn::media_inputs::image_processor::ImagePreProcessor;
use inference_nn::media_inputs::preprocessor_config::PreProcessorConfig;
use inference_nn::model::{MultimodalModel, NormalLoadingMetadata, NormalModel};
use inference_nn::paged_attention::{
    AttentionImplementation, ModelConfigLike, ModelConfigMetadata,
};
use inference_nn::vision::clip::get_clip_vit_num_elems;
use inference_nn::xlora::XLoraConfig;
use inference_quant::ShardedVarBuilder;
use regex::Regex;

use crate::idefics2::{Config as Idefics2Config, Idefics2};
use crate::idefics3::{Idefics3Config, Idefics3Model};
use crate::llama4::inputs_processor::Llama4ImageProcessor;
use crate::llama4::{self, Llama4Config, Llama4Model};
use crate::llava::config::Config as LLaVAConfig;
use crate::llava::llava_next::Model as LLaVANext;
use crate::llava::llava15::Model as LLaVA;
use crate::llava::{llava_inputs_processor, llava_next_inputs_processor};
use crate::mistral3::{Mistral3Config, Mistral3Model};
use crate::mllama::{MLlamaConfig, MLlamaModel};
use crate::voxtral::VoxtralModel;
use crate::voxtral::config::VoxtralConfig;

mod idefics2;
mod idefics3;
mod llama;
mod llava;
mod llava_next;
mod mistral;
mod mistral3;
mod mixtral;
#[cfg(test)]
mod sizing_tests;
mod smollm3;
mod vllama;
mod vllama4;
mod voxtral;

pub use idefics2::*;
pub use idefics3::*;
pub use llama::*;
pub use llava::*;
pub use llava_next::*;
pub use mistral::*;
pub use mistral3::*;
pub use mixtral::*;
pub use smollm3::*;
pub use vllama::*;
pub use vllama4::*;
pub use voxtral::*;

inference_nn::boxed_loaders!(
    NormalModelLoader:
    LlamaLoader,
    MistralLoader,
    MixtralLoader,
    SmolLm3Loader,
);
inference_nn::boxed_loaders!(
    MultimodalModelLoader:
    Idefics2Loader,
    Idefics3Loader,
    LLaVALoader,
    LLaVANextLoader,
    Mistral3Loader,
    VLlama4Loader,
    VLlamaLoader,
    VoxtralLoader,
);
