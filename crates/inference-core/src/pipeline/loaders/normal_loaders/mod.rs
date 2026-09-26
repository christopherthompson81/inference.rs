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

// One row per text architecture; everything that names an architecture is generated from it.
macro_rules! normal_loader_types {
    ($($variant:ident {
        cli: $cli:literal,
        hf: $hf:literal,
        model_type: $model_type:literal,
        loader: $loader:ident $(,)?
    }),* $(,)?) => {
        #[cfg_attr(feature = "pyo3_macros", pyclass(eq, eq_int))]
        #[derive(Clone, Debug, Deserialize, serde::Serialize, PartialEq, strum::EnumIter)]
        /// The architecture to load the normal model as.
        pub enum NormalLoaderType {
            $(#[serde(rename = $cli)] $variant,)*
        }

        // https://github.com/huggingface/transformers/blob/cff06aac6fad28019930be03f5d467055bf62177/src/transformers/models/auto/modeling_auto.py#L448
        impl NormalLoaderType {
            const CLI_NAMES: &'static [&'static str] = &[$($cli),*];

            pub(crate) fn causal_lm_name(&self) -> &'static str {
                match self {
                    $(Self::$variant => $hf,)*
                }
            }

            pub(crate) fn model_type_name(&self) -> &'static str {
                match self {
                    $(Self::$variant => $model_type,)*
                }
            }

            pub fn from_causal_lm_name(name: &str) -> Result<Self> {
                match name {
                    $($hf => Ok(Self::$variant),)*
                    other => anyhow::bail!(
                        "Unsupported Hugging Face Transformers -CausalLM model class `{other}`. Please raise an issue."
                    ),
                }
            }

            pub(crate) fn loader(&self) -> Box<dyn NormalModelLoader> {
                match self {
                    $(Self::$variant => Box::new($loader),)*
                }
            }
        }

        impl FromStr for NormalLoaderType {
            type Err = String;
            fn from_str(s: &str) -> Result<Self, Self::Err> {
                match s {
                    $($cli => Ok(Self::$variant),)*
                    a => Err(format!(
                        "Unknown architecture `{a}`. Possible architectures: {}.",
                        Self::CLI_NAMES.iter().map(|n| format!("`{n}`")).collect::<Vec<_>>().join(", ")
                    )),
                }
            }
        }

        impl Display for NormalLoaderType {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    $(Self::$variant => f.write_str($cli),)*
                }
            }
        }
    };
}

normal_loader_types! {
    Mistral { cli: "mistral", hf: "MistralForCausalLM", model_type: "mistral", loader: MistralLoader },
    Gemma { cli: "gemma", hf: "GemmaForCausalLM", model_type: "gemma", loader: GemmaLoader },
    Mixtral { cli: "mixtral", hf: "MixtralForCausalLM", model_type: "mixtral", loader: MixtralLoader },
    Llama { cli: "llama", hf: "LlamaForCausalLM", model_type: "llama", loader: LlamaLoader },
    Phi2 { cli: "phi2", hf: "PhiForCausalLM", model_type: "phi", loader: Phi2Loader },
    Phi3 { cli: "phi3", hf: "Phi3ForCausalLM", model_type: "phi3", loader: Phi3Loader },
    Qwen2 { cli: "qwen2", hf: "Qwen2ForCausalLM", model_type: "qwen2", loader: Qwen2Loader },
    Gemma2 { cli: "gemma2", hf: "Gemma2ForCausalLM", model_type: "gemma2", loader: Gemma2Loader },
    Starcoder2 { cli: "starcoder2", hf: "Starcoder2ForCausalLM", model_type: "starcoder2", loader: Starcoder2Loader },
    Phi3_5MoE { cli: "phi3.5moe", hf: "PhiMoEForCausalLM", model_type: "phimoe", loader: Phi3_5MoELoader },
    DeepSeekV2 {
        cli: "deepseekv2",
        hf: "DeepseekV2ForCausalLM",
        model_type: "deepseek_v2",
        loader: DeepSeekV2Loader,
    },
    DeepSeekV3 {
        cli: "deepseekv3",
        hf: "DeepseekV3ForCausalLM",
        model_type: "deepseek_v3",
        loader: DeepSeekV3Loader,
    },
    Qwen3 { cli: "qwen3", hf: "Qwen3ForCausalLM", model_type: "qwen3", loader: Qwen3Loader },
    GLM4 { cli: "glm4", hf: "Glm4ForCausalLM", model_type: "glm4", loader: GLM4Loader },
    GLM4MoeLite {
        cli: "glm4moelite",
        hf: "Glm4MoeLiteForCausalLM",
        model_type: "glm4_moe_lite",
        loader: GLM4MoeLiteLoader,
    },
    GLM4Moe { cli: "glm4moe", hf: "Glm4MoeForCausalLM", model_type: "glm4_moe", loader: GLM4MoeLoader },
    Qwen3Moe { cli: "qwen3moe", hf: "Qwen3MoeForCausalLM", model_type: "qwen3_moe", loader: Qwen3MoELoader },
    SmolLm3 { cli: "smollm3", hf: "SmolLM3ForCausalLM", model_type: "smollm3", loader: SmolLm3Loader },
    GraniteMoeHybrid {
        cli: "granitemoehybrid",
        hf: "GraniteMoeHybridForCausalLM",
        model_type: "granitemoehybrid",
        loader: GraniteMoeHybridLoader,
    },
    GptOss { cli: "gpt_oss", hf: "GptOssForCausalLM", model_type: "gpt_oss", loader: GptOssLoader },
    HunYuanDenseV1 {
        cli: "hunyuanv1dense",
        hf: "HunYuanDenseV1ForCausalLM",
        model_type: "hunyuan_v1_dense",
        loader: HunYuanDenseV1Loader,
    },
    HunYuanMoEV1 {
        cli: "hunyuanv1moe",
        hf: "HunYuanMoEV1ForCausalLM",
        model_type: "hunyuan_v1_moe",
        loader: HunYuanMoEV1Loader,
    },
    Qwen3Next { cli: "qwen3next", hf: "Qwen3NextForCausalLM", model_type: "qwen3_next", loader: Qwen3NextLoader },
    Qwen3_5 { cli: "qwen3_5", hf: "Qwen3_5ForCausalLM", model_type: "qwen3_5_text", loader: Qwen3_5TextLoader },
    Lfm2 { cli: "lfm2", hf: "Lfm2ForCausalLM", model_type: "lfm2", loader: Lfm2Loader },
    Lfm2Moe { cli: "lfm2_moe", hf: "Lfm2MoeForCausalLM", model_type: "lfm2_moe", loader: Lfm2Loader },
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
