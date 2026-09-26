pub use crate::model::MultimodalModel;
use std::borrow::Cow;
use std::sync::Arc;
use std::{fmt::Debug, str::FromStr};

use anyhow::Result;
use candle_core::{DType, Device, D};
use candle_nn::Conv2dConfig;
use image::{ColorType, DynamicImage};
use inference_quant::log::once_log_debug;
use inference_quant::ShardedVarBuilder;
use itertools::Itertools;

#[cfg(feature = "pyo3_macros")]
use pyo3::pyclass;

use regex::Regex;
use serde::Deserialize;

use self::minicpmo::{MiniCpmOConfig, MiniCpmOModel, MiniCpmOProcessor};

use super::{DeviceMappedModelLoader, NonMappedSubModel, NormalLoadingMetadata};
// Loaders call these as `super::X`; they live one level up, in `loaders`.
use super::{
    language_model_pack_factors, language_model_pack_factors_with_aliases,
    promoted_tensor_pack_factor, AutoDeviceMapQuantization,
};

use crate::attention::ATTENTION_CHUNK_SIZE;
use crate::device_map::DeviceMapper;
use crate::gguf::normal_registry::RopePairing;
use crate::layers::Conv3dConfig;
use crate::matformer::MatformerSliceConfig;
use crate::paged_attention::{
    AttentionImplementation, HybridPagedKvCacheConfig, ModelConfigLike, ModelConfigMetadata,
};
use crate::pipeline::isq::IsqModelLoader;
use crate::pipeline::loaders::AutoDeviceMapParams;
use crate::pipeline::{
    Modalities, MultimodalPromptPrefixer, Processor, ProcessorCreator, SupportedModality,
};
use crate::utils::varbuilder_utils::DeviceForLoadTensor;
use crate::vision_models::clip::ClipConfig;
use crate::vision_models::diffusion_gemma::{DiffusionGemmaConfig, DiffusionGemmaModel};
use crate::vision_models::gemma3::config::Gemma3Config;
use crate::vision_models::gemma3::{Gemma3Model, Gemma3Processor};
use crate::vision_models::gemma3n::config::{Gemma3nConfig, IntermediateSize};
use crate::vision_models::gemma3n::{Gemma3nModel, Gemma3nProcessor};
use crate::vision_models::gemma4::config::Gemma4Config;
use crate::vision_models::gemma4::{Gemma4Model, Gemma4Processor, Gemma4ProcessorSettings};
use crate::vision_models::idefics2::{Config as Idefics2Config, Idefics2};
use crate::vision_models::idefics2_input_processor::Idefics2Processor;
use crate::vision_models::idefics3::{Idefics3Config, Idefics3Model, Idefics3Processor};
use crate::vision_models::image_processor::ImagePreProcessor;
use crate::vision_models::inputs_processor::Phi4MMProcessor;
use crate::vision_models::lfm2_vl::{Config as Lfm2VlConfig, Lfm2VlModel, Lfm2VlProcessor};
use crate::vision_models::llama4::{
    self, Llama4Config, Llama4ImageProcessor, Llama4Model, Llama4Processor,
};
use crate::vision_models::llava::config::Config as LLaVAConfig;
use crate::vision_models::llava15::Model as LLaVA;
use crate::vision_models::llava_inputs_processor::{self, LLaVAProcessor};
use crate::vision_models::llava_next::Model as LLaVANext;
use crate::vision_models::llava_next_inputs_processor::{self, LLaVANextProcessor};
use crate::vision_models::mistral3::{Mistral3Config, Mistral3Model, Mistral3Processor};
use crate::vision_models::mllama::{MLlamaConfig, MLlamaModel, MLlamaProcessor};
use crate::vision_models::muse_glimmer::{
    Config as MuseGlimmerConfig, MuseGlimmerModel, MuseGlimmerProcessor,
};
use crate::vision_models::paddleocr_vl::config::Config as PaddleOcrVlConfig;
use crate::vision_models::paddleocr_vl::{
    inputs_processor::PaddleOcrVlProcessor, PaddleOcrVlModel,
};
use crate::vision_models::phi3::{Config as Phi3Config, Model as Phi3, PHI3V_CLIP_CONFIG};
use crate::vision_models::phi3_inputs_processor::Phi3Processor;
use crate::vision_models::phi4::{Phi4MMConfig, Phi4MMModel, PHI4_MM_VISION_CFG};
use crate::vision_models::preprocessor_config::PreProcessorConfig;
use crate::vision_models::processor_config::ProcessorConfig;
use crate::vision_models::qwen2_5_vl::{
    Config as Qwen2_5VLConfig, Qwen2_5VLModel, Qwen2_5VLProcessor,
};
use crate::vision_models::qwen2vl::{Config as Qwen2VLConfig, Qwen2VLModel, Qwen2VLProcessor};
use crate::vision_models::qwen3_5::{Config as Qwen3_5Config, Qwen3_5Model, Qwen3_5Processor};
use crate::vision_models::qwen3_5_moe::{
    Config as Qwen3_5MoeConfig, Qwen3_5MoeModel, Qwen3_5MoeProcessor,
};
use crate::vision_models::qwen3_vl::{Config as Qwen3VLConfig, Qwen3VLModel, Qwen3VLProcessor};
use crate::vision_models::qwen3_vl_moe::{
    Config as Qwen3VLMoEConfig, Qwen3VLMoEModel, Qwen3VLMoEProcessor,
};
use crate::vision_models::voxtral::config::VoxtralConfig;
use crate::vision_models::voxtral::{VoxtralModel, VoxtralProcessor};
use crate::vision_models::{minicpmo, phi4};

// HF Qwen3VLVideoProcessor sampling defaults, shared by the Qwen3-VL/3.5 family.
const QWEN3_VIDEO_SAMPLING: crate::VideoFrameSampling = crate::VideoFrameSampling::Fps {
    fps: 2.0,
    min_frames: 4,
    max_frames: 768,
};

pub trait MultimodalModelLoader: IsqModelLoader + Send + Sync + DeviceMappedModelLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn MultimodalModel + Send + Sync>>;
    fn runtime_config<'a>(
        &self,
        config: &'a str,
        max_model_len: Option<usize>,
    ) -> Result<Cow<'a, str>> {
        if let Some(max_model_len) = max_model_len {
            anyhow::bail!(
                "max_model_len={max_model_len} is not supported by this multimodal loader"
            );
        }
        Ok(Cow::Borrowed(config))
    }
    fn is_gptx(&self, config: &str) -> bool;
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
                None => Ok(self.is_gptx(config)),
            },
        }
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>>;
    fn get_processor(
        &self,
        model_config: &str,
        processor_config: Option<ProcessorConfig>,
        preprocessor_config: PreProcessorConfig,
        max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync>;
    fn supports_paged_attention(&self, config: &str) -> bool;
    fn supports_encoder_cache(&self, _config: &str) -> bool {
        false
    }
    fn supports_prefix_cacher(&self, _config: &str) -> bool {
        // Default is false, specific model must override.
        false
    }
    fn auto_device_map_params(
        &self,
        _config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<AutoDeviceMapParams> {
        Ok(params.maybe_promote_to_multimodal())
    }
    fn modalities(&self, config: &str) -> Result<Modalities>;
    fn prefixer(&self, config: &str) -> Arc<dyn MultimodalPromptPrefixer>;
    /// How to sample frames when decoding video inputs for this model.
    fn video_frame_sampling(&self, _config: &str) -> crate::VideoFrameSampling {
        crate::VideoFrameSampling::default()
    }
    /// Return a default chat template (Jinja string) for models that don't ship a
    /// `tokenizer_config.json` or `chat_template.jinja`. Returns `None` by default.
    /// The `config` parameter is the raw model config JSON, used by `AutoMultimodalLoader`
    /// to delegate to the correct concrete loader.
    fn default_chat_template(&self, _config: &str) -> Option<String> {
        None
    }
    /// Return default (bos_token, eos_token) strings for models that don't ship a
    /// `tokenizer_config.json`. Used to populate the chat template context and
    /// EOS token detection. Returns `None` by default.
    fn default_bos_eos(&self, _config: &str) -> Option<(String, String)> {
        None
    }
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

// One row per multimodal architecture; the first `cli` name is canonical, the rest are accepted aliases.
macro_rules! multimodal_loader_types {
    ($($variant:ident {
        cli: $cli:literal $(| $cli_alias:literal)*,
        hf: $hf:literal $(| $hf_alias:literal)*,
        loader: $loader:ident $(,)?
    }),* $(,)?) => {
        #[cfg_attr(feature = "pyo3_macros", pyclass(eq, eq_int))]
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

            pub(crate) fn loader(&self) -> Box<dyn MultimodalModelLoader> {
                match self {
                    $(Self::$variant => Box::new($loader),)*
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
    Phi3V { cli: "phi3v", hf: "Phi3VForCausalLM", loader: Phi3VLoader },
    Idefics2 { cli: "idefics2", hf: "Idefics2ForConditionalGeneration", loader: Idefics2Loader },
    LLaVANext { cli: "llava_next", hf: "LlavaNextForConditionalGeneration", loader: LLaVANextLoader },
    LLaVA { cli: "llava", hf: "LlavaForConditionalGeneration", loader: LLaVALoader },
    Lfm2Vl { cli: "lfm2vl" | "lfm2_vl", hf: "Lfm2VlForConditionalGeneration", loader: Lfm2VlLoader },
    VLlama { cli: "vllama", hf: "MllamaForConditionalGeneration", loader: VLlamaLoader },
    Qwen2VL { cli: "qwen2vl", hf: "Qwen2VLForConditionalGeneration", loader: Qwen2VLLoader },
    Idefics3 { cli: "idefics3", hf: "Idefics3ForConditionalGeneration", loader: Idefics3Loader },
    MiniCpmO { cli: "minicpmo", hf: "MiniCPMO", loader: MiniCpmOLoader },
    Phi4MM { cli: "phi4mm", hf: "Phi4MMForCausalLM", loader: Phi4MMLoader },
    Qwen2_5VL { cli: "qwen2_5vl", hf: "Qwen2_5_VLForConditionalGeneration", loader: Qwen2_5VLLoader },
    Gemma3 { cli: "gemma3", hf: "Gemma3ForConditionalGeneration" | "Gemma3ForCausalLM", loader: Gemma3Loader },
    Mistral3 { cli: "mistral3", hf: "Mistral3ForConditionalGeneration", loader: Mistral3Loader },
    Llama4 { cli: "llama4", hf: "Llama4ForConditionalGeneration", loader: VLlama4Loader },
    Gemma3n { cli: "gemma3n", hf: "Gemma3nForConditionalGeneration", loader: Gemma3nLoader },
    Qwen3VL { cli: "qwen3vl", hf: "Qwen3VLForConditionalGeneration", loader: Qwen3VLLoader },
    Qwen3VLMoE { cli: "qwen3vlmoe", hf: "Qwen3VLMoeForConditionalGeneration", loader: Qwen3VLMoELoader },
    Qwen3_5 { cli: "qwen3_5", hf: "Qwen3_5ForConditionalGeneration", loader: Qwen3_5Loader },
    Qwen3_5Moe { cli: "qwen3_5moe", hf: "Qwen3_5MoeForConditionalGeneration", loader: Qwen3_5MoeLoader },
    Voxtral { cli: "voxtral", hf: "VoxtralRealtimeForConditionalGeneration", loader: VoxtralLoader },
    Gemma4 {
        cli: "gemma4",
        hf: "Gemma4ForConditionalGeneration"
            | "Gemma4ForCausalLM"
            | "Gemma4UnifiedForConditionalGeneration"
            | "Gemma4UnifiedForCausalLM",
        loader: Gemma4Loader,
    },
    MuseGlimmer {
        cli: "muse_glimmer" | "museglimmer",
        hf: "MuseGlimmerForConditionalGeneration",
        loader: MuseGlimmerLoader,
    },
    DiffusionGemma { cli: "diffusiongemma", hf: "DiffusionGemmaForBlockDiffusion", loader: DiffusionGemmaLoader },
    PaddleOcrVl { cli: "paddleocr_vl", hf: "PaddleOCRVLForConditionalGeneration", loader: PaddleOcrVlLoader },
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

fn get_clip_vit_num_elems(cfg: &ClipConfig) -> usize {
    let pre_layer_norm = cfg.hidden_size;
    let final_layer_norm = cfg.hidden_size;

    let num_patches = (cfg.image_size / cfg.patch_size).pow(2);
    let num_positions = num_patches + 1;

    let class_embedding = cfg.hidden_size;

    let position_ids = num_positions;
    let position_embedding = num_positions * cfg.hidden_size;

    let conv2dconfig = Conv2dConfig {
        stride: cfg.patch_size,
        ..Default::default()
    };
    let patch_embedding =
        cfg.num_channels * cfg.hidden_size / conv2dconfig.groups * cfg.patch_size * cfg.patch_size;

    let encoder_layer_elems = {
        let layer_norm1 = cfg.hidden_size;
        let layer_norm2 = cfg.hidden_size;

        let q_proj = cfg.hidden_size * cfg.hidden_size + cfg.hidden_size;
        let k_proj = cfg.hidden_size * cfg.hidden_size + cfg.hidden_size;
        let v_proj = cfg.hidden_size * cfg.hidden_size + cfg.hidden_size;
        let o_proj = cfg.hidden_size * cfg.hidden_size + cfg.hidden_size;

        let fc1 = cfg.hidden_size * cfg.intermediate_size + cfg.intermediate_size;
        let fc2 = cfg.intermediate_size * cfg.hidden_size + cfg.hidden_size;

        layer_norm1 + layer_norm2 + q_proj + k_proj + v_proj + o_proj + fc1 + fc2
    };

    pre_layer_norm
        + final_layer_norm
        + class_embedding
        + position_ids
        + position_embedding
        + patch_embedding
        + cfg.num_hidden_layers * encoder_layer_elems
}

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
mod phi3v;
pub use phi3v::*;
mod idefics2;
pub use idefics2::*;
mod llava_next;
pub use llava_next::*;
mod llava;
pub use llava::*;
mod vllama;
pub use vllama::*;
mod qwen2vl;
pub use qwen2vl::*;
mod idefics3;
pub use idefics3::*;
mod minicpm_o;
pub use minicpm_o::*;
mod phi4mm;
pub use phi4mm::*;
mod qwen2_5vl;
pub use qwen2_5vl::*;
mod gemma3;
pub use gemma3::*;
mod mistral3;
pub use mistral3::*;
mod vllama4;
pub use vllama4::*;
mod gemma3n;
pub use gemma3n::*;
mod paddleocr_vl;
pub use paddleocr_vl::*;
mod qwen3vl;
pub use qwen3vl::*;
mod qwen3vl_moe;
pub use qwen3vl_moe::*;
mod qwen3_5;
pub use qwen3_5::*;
mod qwen3_5_moe;
pub use qwen3_5_moe::*;
mod voxtral;
pub use voxtral::*;
mod gemma4;
pub use gemma4::*;
mod muse_glimmer;
pub use muse_glimmer::*;
mod lfm2vl;
pub use lfm2vl::*;
mod diffusion_gemma;
pub use diffusion_gemma::*;

#[cfg(test)]
mod tests;
