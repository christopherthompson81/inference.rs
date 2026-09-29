pub use crate::model::{NormalLoadingMetadata, NormalModel};
use std::{
    borrow::Cow,
    collections::HashMap,
    fmt::{Debug, Display},
    str::FromStr,
};

use crate::matformer::MatformerSliceConfig;

use crate::{
    lora::{LoraConfig, Ordering},
    paged_attention::{AttentionImplementation, ModelConfigLike, ModelConfigMetadata},
    pipeline::isq::IsqModelLoader,
};
use anyhow::Result;
use candle_core::DType;
use inference_nn::bias_if;
use inference_quant::log::once_log_debug;

use inference_quant::ShardedVarBuilder;

use crate::pipeline::isq::isq_regexes;
use regex::Regex;
use serde::Deserialize;

#[cfg(any(feature = "models-gemma", feature = "models-llama"))]
use crate::models;
#[cfg(any(feature = "models-gemma", feature = "models-llama"))]
use crate::xlora_models;
use crate::xlora_models::XLoraConfig;

use super::{AutoDeviceMapParams, DeviceMappedModelLoader};
// Loaders reach these through `use super::*` (or as `super::X`); they live one level up, in `loaders`.
#[cfg(feature = "models-gemma")]
use super::tied_promoted_tensor_pack_factor;
use super::AutoDeviceMapQuantization;
#[cfg(feature = "models-llama")]
use super::{standard_non_mapped_size_in_bytes, LanguageModelEnds};

pub use inference_nn::loaders::NormalModelLoader;

// One row per text architecture; everything that names an architecture is generated from it.
macro_rules! normal_loader_types {
    ($($variant:ident {
        cli: $cli:tt,
        hf: $hf:literal,
        model_type: $model_type:literal,
        loader: $loader:ident
        $(, feature: $feature:literal)? $(,)?
    }),* $(,)?) => {
        #[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
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

            pub(crate) fn loader(&self) -> Result<Box<dyn NormalModelLoader>> {
                match self {
                    $(
                        $(#[cfg(feature = $feature)])?
                        Self::$variant => Ok($loader::boxed()),
                        $(
                            #[cfg(not(feature = $feature))]
                            Self::$variant => anyhow::bail!(
                                "architecture `{}` is not built in; enable the `{}` feature",
                                $cli,
                                $feature
                            ),
                        )?
                    )*
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
    Mistral { cli: "mistral", hf: "MistralForCausalLM", model_type: "mistral", loader: MistralLoader, feature: "models-llama" },
    Gemma { cli: "gemma", hf: "GemmaForCausalLM", model_type: "gemma", loader: GemmaLoader, feature: "models-gemma" },
    Mixtral { cli: "mixtral", hf: "MixtralForCausalLM", model_type: "mixtral", loader: MixtralLoader, feature: "models-llama" },
    Llama { cli: "llama", hf: "LlamaForCausalLM", model_type: "llama", loader: LlamaLoader, feature: "models-llama" },
    Phi2 { cli: "phi2", hf: "PhiForCausalLM", model_type: "phi", loader: Phi2Loader, feature: "models-phi" },
    Phi3 { cli: "phi3", hf: "Phi3ForCausalLM", model_type: "phi3", loader: Phi3Loader, feature: "models-phi" },
    Qwen2 { cli: "qwen2", hf: "Qwen2ForCausalLM", model_type: "qwen2", loader: Qwen2Loader, feature: "models-qwen" },
    Gemma2 { cli: "gemma2", hf: "Gemma2ForCausalLM", model_type: "gemma2", loader: Gemma2Loader, feature: "models-gemma" },
    Starcoder2 { cli: "starcoder2", hf: "Starcoder2ForCausalLM", model_type: "starcoder2", loader: Starcoder2Loader, feature: "models-other" },
    Phi3_5MoE { cli: "phi3.5moe", hf: "PhiMoEForCausalLM", model_type: "phimoe", loader: Phi3_5MoELoader, feature: "models-phi" },
    DeepSeekV2 {
        cli: "deepseekv2",
        hf: "DeepseekV2ForCausalLM",
        model_type: "deepseek_v2",
        loader: DeepSeekV2Loader, feature: "models-other",
    },
    DeepSeekV3 {
        cli: "deepseekv3",
        hf: "DeepseekV3ForCausalLM",
        model_type: "deepseek_v3",
        loader: DeepSeekV3Loader, feature: "models-other",
    },
    Qwen3 { cli: "qwen3", hf: "Qwen3ForCausalLM", model_type: "qwen3", loader: Qwen3Loader, feature: "models-qwen" },
    GLM4 { cli: "glm4", hf: "Glm4ForCausalLM", model_type: "glm4", loader: GLM4Loader, feature: "models-other" },
    GLM4MoeLite {
        cli: "glm4moelite",
        hf: "Glm4MoeLiteForCausalLM",
        model_type: "glm4_moe_lite",
        loader: GLM4MoeLiteLoader, feature: "models-other",
    },
    GLM4Moe { cli: "glm4moe", hf: "Glm4MoeForCausalLM", model_type: "glm4_moe", loader: GLM4MoeLoader, feature: "models-other" },
    Qwen3Moe { cli: "qwen3moe", hf: "Qwen3MoeForCausalLM", model_type: "qwen3_moe", loader: Qwen3MoELoader, feature: "models-qwen" },
    SmolLm3 { cli: "smollm3", hf: "SmolLM3ForCausalLM", model_type: "smollm3", loader: SmolLm3Loader, feature: "models-llama" },
    GraniteMoeHybrid {
        cli: "granitemoehybrid",
        hf: "GraniteMoeHybridForCausalLM",
        model_type: "granitemoehybrid",
        loader: GraniteMoeHybridLoader, feature: "models-other",
    },
    GptOss { cli: "gpt_oss", hf: "GptOssForCausalLM", model_type: "gpt_oss", loader: GptOssLoader, feature: "models-other" },
    HunYuanDenseV1 {
        cli: "hunyuanv1dense",
        hf: "HunYuanDenseV1ForCausalLM",
        model_type: "hunyuan_v1_dense",
        loader: HunYuanDenseV1Loader, feature: "models-other",
    },
    HunYuanMoEV1 {
        cli: "hunyuanv1moe",
        hf: "HunYuanMoEV1ForCausalLM",
        model_type: "hunyuan_v1_moe",
        loader: HunYuanMoEV1Loader, feature: "models-other",
    },
    Qwen3Next { cli: "qwen3next", hf: "Qwen3NextForCausalLM", model_type: "qwen3_next", loader: Qwen3NextLoader, feature: "models-qwen" },
    Qwen3_5 { cli: "qwen3_5", hf: "Qwen3_5ForCausalLM", model_type: "qwen3_5_text", loader: Qwen3_5TextLoader, feature: "models-qwen" },
    Lfm2 { cli: "lfm2", hf: "Lfm2ForCausalLM", model_type: "lfm2", loader: Lfm2Loader, feature: "models-other" },
    Lfm2Moe { cli: "lfm2_moe", hf: "Lfm2MoeForCausalLM", model_type: "lfm2_moe", loader: Lfm2Loader, feature: "models-other" },
}

mod auto;
pub use auto::*;
#[cfg(feature = "models-gemma")]
inference_nn::boxed_loaders!(NormalModelLoader: Gemma2Loader, GemmaLoader);
#[cfg(feature = "models-llama")]
inference_nn::boxed_loaders!(
    NormalModelLoader:
    LlamaLoader,
    MistralLoader,
    MixtralLoader,
    SmolLm3Loader,
);
#[cfg(feature = "models-other")]
pub use inference_models_other::loaders::{
    DeepSeekV2Loader, DeepSeekV3Loader, GLM4Loader, GLM4MoeLiteLoader, GLM4MoeLoader, GptOssLoader,
    GraniteMoeHybridLoader, HunYuanDenseV1Loader, HunYuanMoEV1Loader, Lfm2Loader, Starcoder2Loader,
};
#[cfg(feature = "models-phi")]
pub use inference_models_phi::loaders::{Phi2Loader, Phi3Loader, Phi3_5MoELoader};
#[cfg(feature = "models-qwen")]
pub use inference_models_qwen::loaders::{
    Qwen2Loader, Qwen3Loader, Qwen3MoELoader, Qwen3NextLoader, Qwen3_5TextLoader,
};
#[cfg(feature = "models-llama")]
mod mistral;
#[cfg(feature = "models-llama")]
pub use mistral::*;
#[cfg(feature = "models-gemma")]
mod gemma;
#[cfg(feature = "models-gemma")]
pub use gemma::*;
#[cfg(feature = "models-llama")]
mod llama;
#[cfg(feature = "models-llama")]
pub use llama::*;
#[cfg(feature = "models-llama")]
mod mixtral;
#[cfg(feature = "models-llama")]
pub use mixtral::*;
#[cfg(feature = "models-gemma")]
mod gemma2;
#[cfg(feature = "models-gemma")]
pub use gemma2::*;
#[cfg(feature = "models-llama")]
mod smollm3;
#[cfg(feature = "models-llama")]
pub use smollm3::*;

#[cfg(all(
    test,
    feature = "models-gemma",
    feature = "models-llama",
    feature = "models-other",
    feature = "models-phi",
    feature = "models-qwen"
))]
mod tests;
