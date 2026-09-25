use std::{
    borrow::Cow,
    collections::HashMap,
    fmt::{Debug, Display},
    str::FromStr,
    sync::Arc,
};

use crate::{attention::ATTENTION_CHUNK_SIZE, matformer::MatformerSliceConfig};

use crate::speculative::SpeculativeTargetMixin;

use crate::{
    amoe::AnyMoeBaseModelMixin,
    device_map::DeviceMapper,
    lora::{LoraConfig, Ordering},
    paged_attention::{AttentionImplementation, ModelConfigLike, ModelConfigMetadata},
    pipeline::{
        isq::IsqModelLoader, text_models_inputs_processor::FlashParams, EitherCache, IsqModel,
        ModelForwardContext,
    },
    utils::varbuilder_utils::DeviceForLoadTensor,
    xlora_models::NonGranularState,
};

use anyhow::Result;

use candle_core::{DType, Device, Tensor};

use inference_quant::log::once_log_debug;

use indicatif::MultiProgress;

use inference_quant::ShardedVarBuilder;

#[cfg(feature = "pyo3_macros")]
use pyo3::pyclass;

use regex::Regex;

use serde::Deserialize;

use crate::{
    models,
    xlora_models::{self, XLoraConfig},
};

use super::{AutoDeviceMapParams, DeviceMappedModelLoader};
// Loaders call these as `super::X`; they live one level up, in `loaders`.
use super::{
    language_model_pack_factors, language_model_pack_factors_with_aliases,
    tied_promoted_tensor_pack_factor, AutoDeviceMapQuantization,
};

use crate::gguf::normal_registry::RopePairing;

pub trait NormalModel: IsqModel + AnyMoeBaseModelMixin + SpeculativeTargetMixin {
    fn forward(
        &self,
        input_ids: &Tensor,
        ctx: &mut ModelForwardContext<'_>,
    ) -> candle_core::Result<Tensor>;
    #[allow(clippy::too_many_arguments)]
    fn xlora_forward(
        &self,
        input_ids: &Tensor,
        input_ids_full: &Tensor,
        seqlen_offsets: &[usize],
        seqlen_offsets_full: &[usize],
        no_kv_cache: bool,
        non_granular_state: &Option<NonGranularState>,
        context_lens: Vec<(usize, usize)>,
        position_ids: Vec<usize>,
        flash_params: &FlashParams,
        flash_params_full: &FlashParams,
    ) -> candle_core::Result<Tensor>;
    fn is_xlora(&self) -> bool;
    fn device(&self) -> &Device;
    fn cache(&self) -> &EitherCache;
    fn max_seq_len(&self) -> usize;
    fn config(&self) -> &ModelConfigMetadata;
    /// True only when the full forward handles packed prompts and never treats physical rows as logical requests.
    fn supports_packed_prefill(&self) -> bool {
        false
    }
    #[cfg(feature = "cuda")]
    fn supports_cuda_decode_graphs(&self) -> bool {
        false
    }
    fn model_config(&self) -> Arc<dyn ModelConfigLike + Send + Sync> {
        Arc::new(self.config().clone())
    }
}

/// Metadata for loading a model with ISQ or device mapping.
pub struct NormalLoadingMetadata {
    // Device mapping metadata which can be used to construct a concrete device mapper
    pub mapper: Box<dyn DeviceMapper + Send + Sync>,
    // Flag to check if loading in ISQ
    pub loading_isq: bool,
    // Device mapping target device (the one that is not the cpu)
    pub real_device: Device,
    // MultiProgress support for parallelized loading
    pub multi_progress: Arc<MultiProgress>,
    // Optional Matryoshka Transformer slicing configuration
    pub matformer_slicing_config: Option<MatformerSliceConfig>,
    pub(crate) rope_pairing: Option<RopePairing>,
}

pub trait NormalModelLoader: IsqModelLoader + Send + Sync + DeviceMappedModelLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>>;
    #[allow(clippy::too_many_arguments)]
    fn load_xlora(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        normal_loading_metadata: NormalLoadingMetadata,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>>;
    fn runtime_config<'a>(
        &self,
        config: &'a str,
        max_model_len: Option<usize>,
    ) -> Result<Cow<'a, str>> {
        if let Some(max_model_len) = max_model_len {
            anyhow::bail!("max_model_len={max_model_len} is not supported by this model loader");
        }
        Ok(Cow::Borrowed(config))
    }
    fn is_gptx(&self, config: &str) -> Result<bool>;
    fn is_gptx_for(
        &self,
        config: &str,
        normal_loading_metadata: &NormalLoadingMetadata,
    ) -> Result<bool> {
        match normal_loading_metadata.rope_pairing {
            Some(RopePairing::Adjacent) => Ok(false),
            Some(RopePairing::HalfSplit) => Ok(true),
            None => match super::qk_rope_layout_from_config(config)? {
                Some(RopePairing::Adjacent) => Ok(false),
                Some(RopePairing::HalfSplit) => Ok(true),
                None => self.is_gptx(config),
            },
        }
    }
    fn supports_paged_attention(&self, _config: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>>;
    fn get_device_for_tensor(
        &self,
        config: &str,
        _mapper: &dyn DeviceMapper,
        loading_isq: bool,
    ) -> Result<Arc<dyn Fn(String) -> DeviceForLoadTensor + Send + Sync + 'static>> {
        if loading_isq {
            Ok(Arc::new(|_| DeviceForLoadTensor::Base))
        } else {
            let re = Regex::new(r"\.layers\.(\d+)\.").unwrap();
            let num_layers = self.model_config(config)?.num_layers();
            let closure = move |name: String| {
                if let Some(captures) = re.captures(&name) {
                    captures
                        .get(1)
                        .and_then(|m| m.as_str().parse::<usize>().ok())
                        .map(|l| l.min(num_layers))
                        .map(DeviceForLoadTensor::Idx)
                        .unwrap_or(DeviceForLoadTensor::Base)
                } else {
                    DeviceForLoadTensor::Base
                }
            };

            Ok(Arc::new(closure))
        }
    }
}

#[cfg_attr(feature = "pyo3_macros", pyclass(eq, eq_int))]
#[derive(Clone, Debug, Deserialize, serde::Serialize, PartialEq, strum::EnumIter)]
/// The architecture to load the normal model as.
pub enum NormalLoaderType {
    #[serde(rename = "mistral")]
    Mistral,
    #[serde(rename = "gemma")]
    Gemma,
    #[serde(rename = "mixtral")]
    Mixtral,
    #[serde(rename = "llama")]
    Llama,
    #[serde(rename = "phi2")]
    Phi2,
    #[serde(rename = "phi3")]
    Phi3,
    #[serde(rename = "qwen2")]
    Qwen2,
    #[serde(rename = "gemma2")]
    Gemma2,
    #[serde(rename = "starcoder2")]
    Starcoder2,
    #[serde(rename = "phi3.5moe")]
    Phi3_5MoE,
    #[serde(rename = "deepseekv2")]
    DeepSeekV2,
    #[serde(rename = "deepseekv3")]
    DeepSeekV3,
    #[serde(rename = "qwen3")]
    Qwen3,
    #[serde(rename = "glm4")]
    GLM4,
    #[serde(rename = "glm4moelite")]
    GLM4MoeLite,
    #[serde(rename = "glm4moe")]
    GLM4Moe,
    #[serde(rename = "qwen3moe")]
    Qwen3Moe,
    #[serde(rename = "smollm3")]
    SmolLm3,
    #[serde(rename = "granitemoehybrid")]
    GraniteMoeHybrid,
    #[serde(rename = "gpt_oss")]
    GptOss,
    #[serde(rename = "hunyuanv1dense")]
    HunYuanDenseV1,
    #[serde(rename = "hunyuanv1moe")]
    HunYuanMoEV1,
    #[serde(rename = "qwen3next")]
    Qwen3Next,
    #[serde(rename = "qwen3_5")]
    Qwen3_5,
    #[serde(rename = "lfm2")]
    Lfm2,
    #[serde(rename = "lfm2_moe")]
    Lfm2Moe,
}

// https://github.com/huggingface/transformers/blob/cff06aac6fad28019930be03f5d467055bf62177/src/transformers/models/auto/modeling_auto.py#L448
impl NormalLoaderType {
    pub(crate) fn causal_lm_name(&self) -> &'static str {
        match self {
            Self::Mistral => "MistralForCausalLM",
            Self::Gemma => "GemmaForCausalLM",
            Self::Mixtral => "MixtralForCausalLM",
            Self::Llama => "LlamaForCausalLM",
            Self::Phi2 => "PhiForCausalLM",
            Self::Phi3 => "Phi3ForCausalLM",
            Self::Qwen2 => "Qwen2ForCausalLM",
            Self::Gemma2 => "Gemma2ForCausalLM",
            Self::Starcoder2 => "Starcoder2ForCausalLM",
            Self::Phi3_5MoE => "PhiMoEForCausalLM",
            Self::DeepSeekV2 => "DeepseekV2ForCausalLM",
            Self::DeepSeekV3 => "DeepseekV3ForCausalLM",
            Self::Qwen3 => "Qwen3ForCausalLM",
            Self::GLM4 => "Glm4ForCausalLM",
            Self::GLM4MoeLite => "Glm4MoeLiteForCausalLM",
            Self::GLM4Moe => "Glm4MoeForCausalLM",
            Self::Qwen3Moe => "Qwen3MoeForCausalLM",
            Self::SmolLm3 => "SmolLM3ForCausalLM",
            Self::GraniteMoeHybrid => "GraniteMoeHybridForCausalLM",
            Self::GptOss => "GptOssForCausalLM",
            Self::HunYuanDenseV1 => "HunYuanDenseV1ForCausalLM",
            Self::HunYuanMoEV1 => "HunYuanMoEV1ForCausalLM",
            Self::Qwen3Next => "Qwen3NextForCausalLM",
            Self::Qwen3_5 => "Qwen3_5ForCausalLM",
            Self::Lfm2 => "Lfm2ForCausalLM",
            Self::Lfm2Moe => "Lfm2MoeForCausalLM",
        }
    }

    pub(crate) fn model_type_name(&self) -> &'static str {
        match self {
            Self::Mistral => "mistral",
            Self::Gemma => "gemma",
            Self::Mixtral => "mixtral",
            Self::Llama => "llama",
            Self::Phi2 => "phi",
            Self::Phi3 => "phi3",
            Self::Qwen2 => "qwen2",
            Self::Gemma2 => "gemma2",
            Self::Starcoder2 => "starcoder2",
            Self::Phi3_5MoE => "phimoe",
            Self::DeepSeekV2 => "deepseek_v2",
            Self::DeepSeekV3 => "deepseek_v3",
            Self::Qwen3 => "qwen3",
            Self::GLM4 => "glm4",
            Self::GLM4MoeLite => "glm4_moe_lite",
            Self::GLM4Moe => "glm4_moe",
            Self::Qwen3Moe => "qwen3_moe",
            Self::SmolLm3 => "smollm3",
            Self::GraniteMoeHybrid => "granitemoehybrid",
            Self::GptOss => "gpt_oss",
            Self::HunYuanDenseV1 => "hunyuan_v1_dense",
            Self::HunYuanMoEV1 => "hunyuan_v1_moe",
            Self::Qwen3Next => "qwen3_next",
            Self::Qwen3_5 => "qwen3_5_text",
            Self::Lfm2 => "lfm2",
            Self::Lfm2Moe => "lfm2_moe",
        }
    }

    pub fn from_causal_lm_name(name: &str) -> Result<Self> {
        match name {
            "MistralForCausalLM" => Ok(Self::Mistral),
            "MixtralForCausalLM" => Ok(Self::Mixtral),
            "GemmaForCausalLM" => Ok(Self::Gemma),
            "Gemma2ForCausalLM" => Ok(Self::Gemma2),
            "PhiForCausalLM" => Ok(Self::Phi2),
            "Phi3ForCausalLM" => Ok(Self::Phi3),
            "LlamaForCausalLM" => Ok(Self::Llama),
            "Qwen2ForCausalLM" => Ok(Self::Qwen2),
            "Starcoder2ForCausalLM" => Ok(Self::Starcoder2),
            "PhiMoEForCausalLM" => Ok(Self::Phi3_5MoE),
            "DeepseekV2ForCausalLM" => Ok(Self::DeepSeekV2),
            "DeepseekV3ForCausalLM" => Ok(Self::DeepSeekV3),
            "Qwen3ForCausalLM" => Ok(Self::Qwen3),
            "Glm4ForCausalLM" => Ok(Self::GLM4),
            "Glm4MoeLiteForCausalLM" => Ok(Self::GLM4MoeLite),
            "Glm4MoeForCausalLM" => Ok(Self::GLM4Moe),
            "Qwen3MoeForCausalLM" => Ok(Self::Qwen3Moe),
            "SmolLM3ForCausalLM" => Ok(Self::SmolLm3),
            "GraniteMoeHybridForCausalLM" => Ok(Self::GraniteMoeHybrid),
            "GptOssForCausalLM" => Ok(Self::GptOss),
            "HunYuanDenseV1ForCausalLM" => Ok(Self::HunYuanDenseV1),
            "HunYuanMoEV1ForCausalLM" => Ok(Self::HunYuanMoEV1),
            "Qwen3NextForCausalLM" => Ok(Self::Qwen3Next),
            "Qwen3_5ForCausalLM" => Ok(Self::Qwen3_5),
            "Lfm2ForCausalLM" => Ok(Self::Lfm2),
            "Lfm2MoeForCausalLM" => Ok(Self::Lfm2Moe),
            other => anyhow::bail!(
                "Unsupported Hugging Face Transformers -CausalLM model class `{other}`. Please raise an issue."
            ),
        }
    }
}

impl FromStr for NormalLoaderType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "mistral" => Ok(Self::Mistral),
            "gemma" => Ok(Self::Gemma),
            "mixtral" => Ok(Self::Mixtral),
            "llama" => Ok(Self::Llama),
            "phi2" => Ok(Self::Phi2),
            "phi3" => Ok(Self::Phi3),
            "qwen2" => Ok(Self::Qwen2),
            "gemma2" => Ok(Self::Gemma2),
            "starcoder2" => Ok(Self::Starcoder2),
            "phi3.5moe" => Ok(Self::Phi3_5MoE),
            "deepseekv2" => Ok(Self::DeepSeekV2),
            "deepseekv3" => Ok(Self::DeepSeekV3),
            "qwen3" => Ok(Self::Qwen3),
            "glm4" => Ok(Self::GLM4),
            "glm4moelite" => Ok(Self::GLM4MoeLite),
            "glm4moe" => Ok(Self::GLM4Moe),
            "qwen3moe" => Ok(Self::Qwen3Moe),
            "smollm3" => Ok(Self::SmolLm3),
            "granitemoehybrid" => Ok(Self::GraniteMoeHybrid),
            "gpt_oss" => Ok(Self::GptOss),
            "hunyuanv1dense" => Ok(Self::HunYuanDenseV1),
            "hunyuanv1moe" => Ok(Self::HunYuanMoEV1),
            "qwen3next" => Ok(Self::Qwen3Next),
            "qwen3_5" => Ok(Self::Qwen3_5),
            "lfm2" => Ok(Self::Lfm2),
            "lfm2_moe" => Ok(Self::Lfm2Moe),
            a => Err(format!("Unknown architecture `{a}`. Possible architectures: `mistral`, `gemma`, `mixtral`, `llama`, `phi2`, `phi3`, `qwen2`, `gemma2`, `starcoder2`, `phi3.5moe`, `deepseekv2`, `deepseekv3`, `qwen3`, `glm4`, `glm4moelite`, `glm4moe`, `qwen3moe`, `smollm3`, `granitemoehybrid`, `gpt_oss`, `hunyuanv1dense`, `hunyuanv1moe`, `qwen3next`, `qwen3_5`, `lfm2`, `lfm2_moe`.")),
        }
    }
}

impl Display for NormalLoaderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gemma => write!(f, "gemma"),
            Self::Gemma2 => write!(f, "gemma2"),
            Self::Llama => write!(f, "llama"),
            Self::Mistral => write!(f, "mistral"),
            Self::Mixtral => write!(f, "mixtral"),
            Self::Phi2 => write!(f, "phi2"),
            Self::Phi3 => write!(f, "phi3"),
            Self::Phi3_5MoE => write!(f, "phi3.5moe"),
            Self::Qwen2 => write!(f, "qwen2"),
            Self::Starcoder2 => write!(f, "starcoder2"),
            Self::DeepSeekV2 => write!(f, "deepseekv2"),
            Self::DeepSeekV3 => write!(f, "deepseekv3"),
            Self::Qwen3 => write!(f, "qwen3"),
            Self::GLM4 => write!(f, "glm4"),
            Self::GLM4MoeLite => write!(f, "glm4moelite"),
            Self::GLM4Moe => write!(f, "glm4moe"),
            Self::Qwen3Moe => write!(f, "qwen3moe"),
            Self::SmolLm3 => write!(f, "smollm3"),
            Self::GraniteMoeHybrid => write!(f, "granitemoehybrid"),
            Self::GptOss => write!(f, "gpt_oss"),
            Self::HunYuanDenseV1 => write!(f, "hunyuanv1dense"),
            Self::HunYuanMoEV1 => write!(f, "hunyuanv1moe"),
            Self::Qwen3Next => write!(f, "qwen3next"),
            Self::Qwen3_5 => write!(f, "qwen3_5"),
            Self::Lfm2 => write!(f, "lfm2"),
            Self::Lfm2Moe => write!(f, "lfm2_moe"),
        }
    }
}

macro_rules! bias_if {
    ($cond:expr, $size:expr) => {
        if $cond {
            $size
        } else {
            0
        }
    };
}

mod auto;
pub use auto::*;
mod mistral;
pub use mistral::*;
mod gemma;
pub use gemma::*;
mod llama;
pub use llama::*;
mod mixtral;
pub use mixtral::*;
mod phi2;
pub use phi2::*;
mod phi3;
pub use phi3::*;
mod qwen2;
pub use qwen2::*;
mod gemma2;
pub use gemma2::*;
mod starcoder2;
pub use starcoder2::*;
mod phi3_5_moe;
pub use phi3_5_moe::*;
mod deepseek2;
pub use deepseek2::*;
mod deepseek3;
pub use deepseek3::*;
mod qwen3;
pub use qwen3::*;
mod hunyuan_v1_dense;
pub use hunyuan_v1_dense::*;
mod hunyuan_v1_moe;
pub use hunyuan_v1_moe::*;
mod glm4;
pub use glm4::*;
mod glm4_moe_lite;
pub use glm4_moe_lite::*;
mod glm4_moe;
pub use glm4_moe::*;
mod qwen3_moe;
pub use qwen3_moe::*;
mod smollm3;
pub use smollm3::*;
mod granite;
pub use granite::*;
mod gpt_oss;
pub use gpt_oss::*;
mod qwen3_next;
pub use qwen3_next::*;
mod qwen3_5_text;
pub use qwen3_5_text::*;
mod lfm2;
pub use lfm2::*;

#[cfg(test)]
mod tests;
