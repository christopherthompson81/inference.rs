//! The other family's loaders: config parsing, weight sizing, ISQ patterns and prompt prefixes.

use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;

use anyhow::Result;
use candle_core::DType;
use inference_nn::attention::ATTENTION_CHUNK_SIZE;
use inference_nn::bias_if;
use inference_nn::device_map::{AutoDeviceMapParams, DeviceMapper};
use inference_nn::loaders::*;
use inference_nn::lora::{LoraConfig, Ordering};
use inference_nn::matformer::MatformerSliceConfig;
use inference_nn::model::{MultimodalModel, NormalLoadingMetadata, NormalModel};
use inference_nn::paged_attention::{
    AttentionImplementation, ModelConfigLike, ModelConfigMetadata,
};
use inference_nn::utils::varbuilder_utils::DeviceForLoadTensor;
use inference_nn::xlora::XLoraConfig;
use inference_quant::ShardedVarBuilder;
use regex::Regex;

use crate::lfm2_vl::{Lfm2VlModel, config::Config as Lfm2VlConfig};
use crate::paddleocr_vl::{PaddleOcrVlModel, config::Config as PaddleOcrVlConfig};

mod deepseek2;
mod deepseek3;
mod deepseek_family;
mod glm4;
mod glm4_moe;
mod glm4_moe_lite;
mod gpt_oss;
mod granite;
mod hunyuan_v1_dense;
mod hunyuan_v1_moe;
mod lfm2;
mod lfm2vl;
mod paddleocr_vl;
mod starcoder2;

#[cfg(test)]
mod sizing_tests;

pub use deepseek2::*;
pub use deepseek3::*;
pub use glm4::*;
pub use glm4_moe::*;
pub use glm4_moe_lite::*;
pub use gpt_oss::*;
pub use granite::*;
pub use hunyuan_v1_dense::*;
pub use hunyuan_v1_moe::*;
pub use lfm2::*;
pub use lfm2vl::*;
pub use paddleocr_vl::*;
pub use starcoder2::*;

inference_nn::boxed_loaders!(
    NormalModelLoader:
    DeepSeekV2Loader,
    DeepSeekV3Loader,
    GLM4Loader,
    GLM4MoeLiteLoader,
    GLM4MoeLoader,
    GptOssLoader,
    GraniteMoeHybridLoader,
    HunYuanDenseV1Loader,
    HunYuanMoEV1Loader,
    Lfm2Loader,
    Starcoder2Loader,
);
inference_nn::boxed_loaders!(MultimodalModelLoader: Lfm2VlLoader, PaddleOcrVlLoader);
