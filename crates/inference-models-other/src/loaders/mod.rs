//! The other family's loaders: config parsing, weight sizing, ISQ patterns and prompt prefixes.

use std::fmt::Debug;
use std::sync::Arc;

use anyhow::Result;
use inference_nn::attention::ATTENTION_CHUNK_SIZE;
use inference_nn::bias_if;
use inference_nn::device_map::{AutoDeviceMapParams, DeviceMapper};
use inference_nn::loaders::*;
use inference_nn::matformer::MatformerSliceConfig;
use inference_nn::model::{MultimodalModel, NormalLoadingMetadata, NormalModel};
use inference_nn::paged_attention::{
    AttentionImplementation, ModelConfigLike, ModelConfigMetadata,
};
use inference_nn::utils::varbuilder_utils::DeviceForLoadTensor;
use inference_quant::ShardedVarBuilder;
use inference_tensor::DType;
use regex::Regex;

use crate::lfm2_vl::{Lfm2VlModel, config::Config as Lfm2VlConfig};
use crate::paddleocr_vl::{PaddleOcrVlModel, config::Config as PaddleOcrVlConfig};

mod bert;
mod deepseek;
mod deepseek_family;
mod ernie4_5;
mod glm4;
mod glm4_moe;
mod glm4_moe_lite;
mod gpt_oss;
mod granite;
mod granite4_vision;
mod granite_speech;
mod hunyuan_v1_dense;
mod hunyuan_v1_moe;
mod lfm2;
mod lfm2_decision;
mod lfm2vl;
mod modern_bert;
mod modern_bert_decision;
mod neo_bert;
mod nomic_bert;
mod olmo;
mod paddleocr_vl;
mod starcoder2;

#[cfg(test)]
mod sizing_tests;

pub use bert::*;
pub use deepseek::*;
pub use ernie4_5::*;
pub use glm4::*;
pub use glm4_moe::*;
pub use glm4_moe_lite::*;
pub use gpt_oss::*;
pub use granite::*;
pub use granite_speech::*;
pub use granite4_vision::*;
pub use hunyuan_v1_dense::*;
pub use hunyuan_v1_moe::*;
pub use lfm2::*;
pub use lfm2_decision::*;
pub use lfm2vl::*;
pub use modern_bert::*;
pub use modern_bert_decision::*;
pub use neo_bert::*;
pub use nomic_bert::*;
pub use olmo::*;
pub use paddleocr_vl::*;
pub use starcoder2::*;

inference_nn::boxed_loaders!(
    NormalModelLoader:
    DeepSeekV2Loader,
    DeepSeekV3Loader,
    Ernie4_5Loader,
    Ernie4_5MoeLoader,
    GLM4Loader,
    GLM4MoeLiteLoader,
    GLM4MoeLoader,
    GptOssLoader,
    GraniteMoeHybridLoader,
    HunYuanDenseV1Loader,
    HunYuanMoEV1Loader,
    Lfm2Loader,
    Olmo2Loader,
    OlmoLoader,
    OlmoeLoader,
    Starcoder2Loader,
);
inference_nn::boxed_loaders!(
    MultimodalModelLoader: Granite4VisionLoader,
    GraniteSpeechLoader,
    Lfm2VlLoader,
    PaddleOcrVlLoader
);
inference_nn::boxed_loaders!(
    EmbeddingModelLoader: BertLoader,
    Lfm2DecisionLoader,
    ModernBertLoader,
    ModernBertDecisionLoader,
    NeoBertLoader,
    NomicBertLoader,
);
