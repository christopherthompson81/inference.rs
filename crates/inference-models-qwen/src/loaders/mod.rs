//! The qwen family's loaders: config parsing, weight sizing, ISQ patterns and prompt prefixes.

use std::borrow::Cow;
use std::fmt::Debug;
use std::sync::Arc;

use anyhow::Result;
use inference_nn::attention::ATTENTION_CHUNK_SIZE;
use inference_nn::bias_if;
use inference_nn::device_map::{AutoDeviceMapParams, DeviceMapper};
use inference_nn::layers::Conv3dConfig;
use inference_nn::loaders::*;
use inference_nn::matformer::MatformerSliceConfig;
use inference_nn::media_inputs::video::VideoFrameSampling;
use inference_nn::model::{EmbeddingModel, MultimodalModel, NormalLoadingMetadata, NormalModel};
use inference_nn::paged_attention::{
    AttentionImplementation, HybridPagedKvCacheConfig, ModelConfigLike, ModelConfigMetadata,
};
use inference_nn::utils::varbuilder_utils::DeviceForLoadTensor;
use inference_quant::ShardedVarBuilder;
use inference_tensor::DType;
use inference_tensor::nn::Conv2dConfig;
use regex::Regex;

use crate::minicpmo::{MiniCpmOConfig, MiniCpmOModel};
use crate::muse_glimmer::{Config as MuseGlimmerConfig, MuseGlimmerModel};
use crate::qwen2_5_vl::{Config as Qwen2_5VLConfig, Qwen2_5VLModel};
use crate::qwen2vl::{Config as Qwen2VLConfig, Qwen2VLModel};
use crate::qwen3_5::{Config as Qwen3_5Config, Qwen3_5Model};
use crate::qwen3_vl::{Config as Qwen3VLConfig, Qwen3VLModel};
use crate::qwen3_vl_moe::{Config as Qwen3VLMoEConfig, Qwen3VLMoEModel};

// HF Qwen3VLVideoProcessor sampling defaults, shared by the Qwen3-VL/3.5 family.
const QWEN3_VIDEO_SAMPLING: VideoFrameSampling = VideoFrameSampling::Fps {
    fps: 2.0,
    min_frames: 4,
    max_frames: 768,
};

mod minicpm_o;
mod muse_glimmer;
mod qwen2;
mod qwen2_5vl;
mod qwen2vl;
mod qwen3;
mod qwen3_5;
mod qwen3_5_text;
mod qwen3_embedding;
mod qwen3_moe;
mod qwen3_next;
mod qwen3vl;
mod qwen3vl_moe;
#[cfg(test)]
mod sizing_tests;

pub use minicpm_o::*;
pub use muse_glimmer::*;
pub use qwen2::*;
pub use qwen2_5vl::*;
pub use qwen2vl::*;
pub use qwen3::*;
pub use qwen3_5::*;
pub use qwen3_5_text::*;
pub use qwen3_embedding::*;
pub use qwen3_moe::*;
pub use qwen3_next::*;
pub use qwen3vl::*;
pub use qwen3vl_moe::*;

inference_nn::boxed_loaders!(
    NormalModelLoader:
    Qwen2Loader,
    Qwen3Loader,
    Qwen3MoELoader,
    Qwen3NextLoader,
    Qwen3_5TextLoader,
    Qwen3_5MoeTextLoader,
);
inference_nn::boxed_loaders!(
    MultimodalModelLoader:
    MiniCpmOLoader,
    MuseGlimmerLoader,
    Qwen2VLLoader,
    Qwen2_5VLLoader,
    Qwen3VLLoader,
    Qwen3VLMoELoader,
    Qwen3_5Loader,
    Qwen3_5MoeLoader,
);
inference_nn::boxed_loaders!(EmbeddingModelLoader: Qwen3EmbeddingLoader);
