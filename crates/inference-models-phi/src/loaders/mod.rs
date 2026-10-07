//! The phi family's loaders: config parsing, weight sizing, ISQ patterns and prompt prefixes.

use std::fmt::Debug;
use std::sync::Arc;

use anyhow::Result;
use inference_nn::attention::ATTENTION_CHUNK_SIZE;
use inference_nn::bias_if;
use inference_nn::device_map::AutoDeviceMapParams;
use inference_nn::loaders::*;
use inference_nn::matformer::MatformerSliceConfig;
use inference_nn::model::{MultimodalModel, NormalLoadingMetadata, NormalModel};
use inference_nn::paged_attention::{
    AttentionImplementation, ModelConfigLike, ModelConfigMetadata,
};
use inference_nn::vision::clip::get_clip_vit_num_elems;
use inference_quant::ShardedVarBuilder;
use inference_tensor::DType;
use inference_tensor::nn::Conv2dConfig;
use regex::Regex;

use crate::phi3_vision::{Config as Phi3Config, Model as Phi3, PHI3V_CLIP_CONFIG};
use crate::phi4::{self, PHI4_MM_VISION_CFG, Phi4MMConfig, Phi4MMModel};

mod phi2;
mod phi3;
mod phi3_5_moe;
mod phi3v;
mod phi4mm;
#[cfg(test)]
mod sizing_tests;

pub use phi2::*;
pub use phi3::*;
pub use phi3_5_moe::*;
pub use phi3v::*;
pub use phi4mm::*;

inference_nn::boxed_loaders!(NormalModelLoader: Phi2Loader, Phi3Loader, Phi3_5MoELoader);
inference_nn::boxed_loaders!(MultimodalModelLoader: Phi3VLoader, Phi4MMLoader);
