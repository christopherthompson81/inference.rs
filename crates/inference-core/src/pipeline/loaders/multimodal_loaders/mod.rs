pub use crate::model::MultimodalModel;
use std::borrow::Cow;
use std::sync::Arc;
use std::{fmt::Debug, str::FromStr};

use anyhow::Result;
use candle_core::DType;
use candle_nn::Conv2dConfig;
use inference_nn::bias_if;
use inference_quant::ShardedVarBuilder;
use inference_quant::log::once_log_debug;

use crate::pipeline::isq::isq_regexes;
use regex::Regex;
use serde::Deserialize;

use super::{DeviceMappedModelLoader, NonMappedSubModel, NormalLoadingMetadata};
// Loaders call these as `super::X`; they live one level up, in `loaders`.
use super::language_model_pack_factors;
#[cfg(feature = "models-gemma")]
use super::promoted_tensor_pack_factor;
use super::{AutoDeviceMapQuantization, language_model_pack_factors_with_aliases};

use crate::attention::ATTENTION_CHUNK_SIZE;
use crate::device_map::DeviceMapper;
use crate::matformer::MatformerSliceConfig;
use crate::paged_attention::{AttentionImplementation, ModelConfigLike, ModelConfigMetadata};
use crate::pipeline::isq::IsqModelLoader;
use crate::pipeline::loaders::AutoDeviceMapParams;
use crate::pipeline::{Modalities, MultimodalPromptPrefixer, Processor, SupportedModality};
use crate::utils::varbuilder_utils::DeviceForLoadTensor;
#[cfg(feature = "models-gemma")]
use crate::vision_models::diffusion_gemma::{DiffusionGemmaConfig, DiffusionGemmaModel};
#[cfg(feature = "models-gemma")]
use crate::vision_models::gemma3::config::Gemma3Config;
#[cfg(feature = "models-gemma")]
use crate::vision_models::gemma3::{Gemma3Model, Gemma3Processor};
#[cfg(feature = "models-gemma")]
use crate::vision_models::gemma3n::config::{Gemma3nConfig, IntermediateSize};
#[cfg(feature = "models-gemma")]
use crate::vision_models::gemma3n::{Gemma3nModel, Gemma3nProcessor};
#[cfg(feature = "models-gemma")]
use crate::vision_models::gemma4::config::Gemma4Config;
#[cfg(feature = "models-gemma")]
use crate::vision_models::gemma4::{Gemma4Model, Gemma4Processor, Gemma4ProcessorSettings};
use crate::vision_models::preprocessor_config::PreProcessorConfig;
use crate::vision_models::processor_config::ProcessorConfig;

pub use inference_nn::loaders::MultimodalModelLoader;

/// The chat-template half of a multimodal loader, which stays with the engine's `Processor`.
pub(crate) trait MultimodalProcessorFactory {
    fn get_processor(
        &self,
        model_config: &str,
        processor_config: Option<ProcessorConfig>,
        preprocessor_config: PreProcessorConfig,
        max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync>;
}

// One row per multimodal architecture; the first `cli` name is canonical, the rest are accepted aliases.
macro_rules! multimodal_loader_types {
    ($($variant:ident {
        cli: $cli:tt $(| $cli_alias:tt)*,
        hf: $hf:literal $(| $hf_alias:literal)*,
        loader: $loader:ident
        $(, feature: $feature:literal)? $(,)?
    }),* $(,)?) => {
        #[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
        #[derive(Clone, Debug, Deserialize, serde::Serialize, PartialEq, strum::EnumIter)]
        /// The architecture to load the multimodal model as.
        pub enum MultimodalLoaderType {
            $(#[serde(rename = $cli)] $variant,)*
        }

        // https://github.com/huggingface/transformers/blob/cff06aac6fad28019930be03f5d467055bf62177/src/transformers/models/auto/modeling_auto.py#L448
        impl MultimodalLoaderType {
            const CLI_NAMES: &'static [&'static str] = &[$($cli),*];

            pub(crate) fn causal_lm_name(&self) -> &'static str {
                match self {
                    $(Self::$variant => $hf,)*
                }
            }

            pub fn from_causal_lm_name(name: &str) -> Result<Self> {
                match name {
                    $($hf $(| $hf_alias)* => Ok(Self::$variant),)*
                    other => anyhow::bail!(
                        "Unsupported Hugging Face Transformers -CausalLM model class `{other}`. Please raise an issue."
                    ),
                }
            }

            pub(crate) fn loader(&self) -> Result<Box<dyn MultimodalModelLoader>> {
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

            #[cfg_attr(
                not(any(
                    feature = "models-gemma",
                    feature = "models-llama",
                    feature = "models-other",
                    feature = "models-phi",
                    feature = "models-qwen"
                )),
                allow(unused_variables)
            )]
            pub(crate) fn get_processor(
                &self,
                model_config: &str,
                processor_config: Option<ProcessorConfig>,
                preprocessor_config: PreProcessorConfig,
                max_edge: Option<u32>,
            ) -> Result<Arc<dyn Processor + Send + Sync>> {
                match self {
                    $(
                        $(#[cfg(feature = $feature)])?
                        Self::$variant => Ok($loader.get_processor(
                            model_config,
                            processor_config,
                            preprocessor_config,
                            max_edge,
                        )),
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

        impl FromStr for MultimodalLoaderType {
            type Err = String;
            fn from_str(s: &str) -> Result<Self, Self::Err> {
                match s {
                    $($cli $(| $cli_alias)* => Ok(Self::$variant),)*
                    a => Err(format!(
                        "Unknown architecture `{a}`. Possible architectures: {}.",
                        Self::CLI_NAMES.iter().map(|n| format!("`{n}`")).collect::<Vec<_>>().join(", ")
                    )),
                }
            }
        }

        impl std::fmt::Display for MultimodalLoaderType {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    $(Self::$variant => f.write_str($cli),)*
                }
            }
        }
    };
}

multimodal_loader_types! {
    Phi3V { cli: "phi3v", hf: "Phi3VForCausalLM", loader: Phi3VLoader, feature: "models-phi" },
    Idefics2 { cli: "idefics2", hf: "Idefics2ForConditionalGeneration", loader: Idefics2Loader, feature: "models-llama" },
    LLaVANext { cli: "llava_next", hf: "LlavaNextForConditionalGeneration", loader: LLaVANextLoader, feature: "models-llama" },
    LLaVA { cli: "llava", hf: "LlavaForConditionalGeneration", loader: LLaVALoader, feature: "models-llama" },
    Lfm2Vl { cli: "lfm2vl" | "lfm2_vl", hf: "Lfm2VlForConditionalGeneration", loader: Lfm2VlLoader, feature: "models-other" },
    VLlama { cli: "vllama", hf: "MllamaForConditionalGeneration", loader: VLlamaLoader, feature: "models-llama" },
    Qwen2VL { cli: "qwen2vl", hf: "Qwen2VLForConditionalGeneration", loader: Qwen2VLLoader, feature: "models-qwen" },
    Idefics3 { cli: "idefics3", hf: "Idefics3ForConditionalGeneration", loader: Idefics3Loader, feature: "models-llama" },
    MiniCpmO { cli: "minicpmo", hf: "MiniCPMO", loader: MiniCpmOLoader, feature: "models-qwen" },
    Phi4MM { cli: "phi4mm", hf: "Phi4MMForCausalLM", loader: Phi4MMLoader, feature: "models-phi" },
    Qwen2_5VL { cli: "qwen2_5vl", hf: "Qwen2_5_VLForConditionalGeneration", loader: Qwen2_5VLLoader, feature: "models-qwen" },
    Gemma3 { cli: "gemma3", hf: "Gemma3ForConditionalGeneration" | "Gemma3ForCausalLM", loader: Gemma3Loader, feature: "models-gemma" },
    Mistral3 { cli: "mistral3", hf: "Mistral3ForConditionalGeneration", loader: Mistral3Loader, feature: "models-llama" },
    Llama4 { cli: "llama4", hf: "Llama4ForConditionalGeneration", loader: VLlama4Loader, feature: "models-llama" },
    Gemma3n { cli: "gemma3n", hf: "Gemma3nForConditionalGeneration", loader: Gemma3nLoader, feature: "models-gemma" },
    Qwen3VL { cli: "qwen3vl", hf: "Qwen3VLForConditionalGeneration", loader: Qwen3VLLoader, feature: "models-qwen" },
    Qwen3VLMoE { cli: "qwen3vlmoe", hf: "Qwen3VLMoeForConditionalGeneration", loader: Qwen3VLMoELoader, feature: "models-qwen" },
    Qwen3_5 { cli: "qwen3_5", hf: "Qwen3_5ForConditionalGeneration", loader: Qwen3_5Loader, feature: "models-qwen" },
    Qwen3_5Moe { cli: "qwen3_5moe", hf: "Qwen3_5MoeForConditionalGeneration", loader: Qwen3_5MoeLoader, feature: "models-qwen" },
    Voxtral { cli: "voxtral", hf: "VoxtralRealtimeForConditionalGeneration", loader: VoxtralLoader, feature: "models-llama" },
    Gemma4 {
        cli: "gemma4",
        hf: "Gemma4ForConditionalGeneration"
            | "Gemma4ForCausalLM"
            | "Gemma4UnifiedForConditionalGeneration"
            | "Gemma4UnifiedForCausalLM",
        loader: Gemma4Loader, feature: "models-gemma",
    },
    MuseGlimmer {
        cli: "muse_glimmer" | "museglimmer",
        hf: "MuseGlimmerForConditionalGeneration",
        loader: MuseGlimmerLoader,
        feature: "models-qwen",
    },
    DiffusionGemma { cli: "diffusiongemma", hf: "DiffusionGemmaForBlockDiffusion", loader: DiffusionGemmaLoader, feature: "models-gemma" },
    PaddleOcrVl { cli: "paddleocr_vl", hf: "PaddleOCRVLForConditionalGeneration", loader: PaddleOcrVlLoader, feature: "models-other" },
}

#[cfg(feature = "models-gemma")]
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

mod auto;
pub use auto::*;
#[cfg(feature = "models-gemma")]
inference_nn::boxed_loaders!(
    MultimodalModelLoader:
    DiffusionGemmaLoader,
    Gemma3Loader,
    Gemma3nLoader,
    Gemma4Loader,
);
#[cfg(feature = "models-llama")]
pub use inference_models_llama::loaders::{
    Idefics2Loader, Idefics3Loader, LLaVALoader, LLaVANextLoader, Mistral3Loader, VLlama4Loader,
    VLlamaLoader, VoxtralLoader,
};
#[cfg(feature = "models-other")]
pub use inference_models_other::loaders::{Lfm2VlLoader, PaddleOcrVlLoader};
#[cfg(feature = "models-phi")]
pub use inference_models_phi::loaders::{Phi3VLoader, Phi4MMLoader};
#[cfg(feature = "models-qwen")]
pub use inference_models_qwen::loaders::{
    MiniCpmOLoader, MuseGlimmerLoader, Qwen2_5VLLoader, Qwen2VLLoader, Qwen3_5Loader,
    Qwen3_5MoeLoader, Qwen3VLLoader, Qwen3VLMoELoader,
};
#[cfg(feature = "models-gemma")]
mod gemma3;
#[cfg(feature = "models-gemma")]
pub use gemma3::*;
#[cfg(feature = "models-gemma")]
mod gemma3n;
#[cfg(feature = "models-gemma")]
pub use gemma3n::*;
#[cfg(feature = "models-gemma")]
mod gemma4;
#[cfg(feature = "models-gemma")]
pub use gemma4::*;
#[cfg(feature = "models-gemma")]
mod diffusion_gemma;
#[cfg(feature = "models-gemma")]
pub use diffusion_gemma::*;

#[cfg(all(
    test,
    feature = "models-gemma",
    feature = "models-llama",
    feature = "models-other",
    feature = "models-phi",
    feature = "models-qwen"
))]
mod tests;
