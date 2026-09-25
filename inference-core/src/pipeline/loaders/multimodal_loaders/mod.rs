use std::any::Any;
use std::borrow::Cow;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};
use std::{fmt::Debug, str::FromStr};

use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
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

use crate::amoe::AnyMoeBaseModelMixin;
use crate::attention::ATTENTION_CHUNK_SIZE;
use crate::block_diffusion::BlockDiffusionMixin;
use crate::device_map::DeviceMapper;
use crate::gguf::normal_registry::RopePairing;
use crate::layers::Conv3dConfig;
use crate::matformer::MatformerSliceConfig;
use crate::paged_attention::{
    encoder_cache::EncoderCacheManager, AttentionImplementation, HybridPagedKvCacheConfig,
    ModelConfigLike, ModelConfigMetadata,
};
use crate::pipeline::isq::IsqModelLoader;
use crate::pipeline::loaders::AutoDeviceMapParams;
use crate::pipeline::{
    EitherCache, IsqModel, Modalities, ModelForwardContext, MultimodalPromptPrefixer, Processor,
    ProcessorCreator, SupportedModality,
};
use crate::speculative::SpeculativeTargetMixin;
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

pub trait MultimodalModel:
    IsqModel + AnyMoeBaseModelMixin + SpeculativeTargetMixin + BlockDiffusionMixin
{
    // pixel_values and pixel_attention_mask only specified for prompt seqs
    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        model_specific_args: Box<dyn Any>, // pixel attention mask, or image sizes, or anything else
        ctx: &mut ModelForwardContext<'_>,
    ) -> candle_core::Result<Tensor>;
    #[cfg(feature = "cuda")]
    fn supports_cuda_decode_graphs(&self) -> bool {
        false
    }
    #[cfg(feature = "cuda")]
    fn supports_cuda_decode_graphs_for_args(&self, _model_specific_args: &dyn Any) -> bool {
        self.supports_cuda_decode_graphs()
    }
    fn requires_uniform_completion_batch(&self) -> bool {
        self.is_block_diffusion()
    }
    fn supports_packed_prefill(&self) -> bool {
        false
    }
    fn supports_mixed_media_batches(&self) -> bool {
        false
    }
    fn device(&self) -> &Device;
    fn cache(&self) -> &EitherCache;
    fn max_seq_len(&self) -> usize;
    fn config(&self) -> &ModelConfigMetadata;
    fn model_config(&self) -> Arc<dyn ModelConfigLike + Send + Sync> {
        Arc::new(self.config().clone())
    }
    /// For a prompt without images. Requires batch size of 1!
    fn default_model_specific_args(&self, input_ids: &Tensor) -> Box<dyn Any>;
    fn encoder_cache(&self) -> Option<&Mutex<EncoderCacheManager>> {
        None
    }
    fn configure_encoder_cache_memory_bytes(&self, max_bytes: usize) -> bool {
        let Some(cache) = self.encoder_cache() else {
            return false;
        };
        cache
            .lock()
            .expect("encoder cache poisoned")
            .set_max_logical_bytes(max_bytes);
        true
    }
    fn encoder_cache_counters(&self) -> Option<(Arc<AtomicUsize>, Arc<AtomicUsize>)> {
        self.encoder_cache()
            .map(|cache| cache.lock().expect("encoder cache poisoned").counters())
    }
    fn reset_model_specific_state(&self) {}
    fn reset_model_specific_state_for_sequences(&self, _sequence_ids: &[usize]) {
        self.reset_model_specific_state();
    }
}

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

#[cfg_attr(feature = "pyo3_macros", pyclass(eq, eq_int))]
#[derive(Clone, Debug, Deserialize, serde::Serialize, PartialEq, strum::EnumIter)]
/// The architecture to load the multimodal model as.
pub enum MultimodalLoaderType {
    #[serde(rename = "phi3v")]
    Phi3V,
    #[serde(rename = "idefics2")]
    Idefics2,
    #[serde(rename = "llava_next")]
    LLaVANext,
    #[serde(rename = "llava")]
    LLaVA,
    #[serde(rename = "lfm2vl")]
    Lfm2Vl,
    #[serde(rename = "vllama")]
    VLlama,
    #[serde(rename = "qwen2vl")]
    Qwen2VL,
    #[serde(rename = "idefics3")]
    Idefics3,
    #[serde(rename = "minicpmo")]
    MiniCpmO,
    #[serde(rename = "phi4mm")]
    Phi4MM,
    #[serde(rename = "qwen2_5vl")]
    Qwen2_5VL,
    #[serde(rename = "gemma3")]
    Gemma3,
    #[serde(rename = "mistral3")]
    Mistral3,
    #[serde(rename = "llama4")]
    Llama4,
    #[serde(rename = "gemma3n")]
    Gemma3n,
    #[serde(rename = "qwen3vl")]
    Qwen3VL,
    #[serde(rename = "qwen3vlmoe")]
    Qwen3VLMoE,
    #[serde(rename = "qwen3_5")]
    Qwen3_5,
    #[serde(rename = "qwen3_5moe")]
    Qwen3_5Moe,
    #[serde(rename = "voxtral")]
    Voxtral,
    #[serde(rename = "gemma4")]
    Gemma4,
    #[serde(rename = "muse_glimmer")]
    MuseGlimmer,
    #[serde(rename = "diffusiongemma")]
    DiffusionGemma,
    #[serde(rename = "paddleocr_vl")]
    PaddleOcrVl,
}

// https://github.com/huggingface/transformers/blob/cff06aac6fad28019930be03f5d467055bf62177/src/transformers/models/auto/modeling_auto.py#L448
impl MultimodalLoaderType {
    pub fn from_causal_lm_name(name: &str) -> Result<Self> {
        match name {
            "Phi3VForCausalLM" => Ok(Self::Phi3V),
            "Idefics2ForConditionalGeneration" => Ok(Self::Idefics2),
            "LlavaNextForConditionalGeneration" => Ok(Self::LLaVANext),
            "LlavaForConditionalGeneration" => Ok(Self::LLaVA),
            "Lfm2VlForConditionalGeneration" => Ok(Self::Lfm2Vl),
            "MllamaForConditionalGeneration" => Ok(Self::VLlama),
            "Qwen2VLForConditionalGeneration" => Ok(Self::Qwen2VL),
            "Idefics3ForConditionalGeneration" => Ok(Self::Idefics3),
            "MiniCPMO" => Ok(Self::MiniCpmO),
            "Phi4MMForCausalLM" => Ok(Self::Phi4MM),
            "Qwen2_5_VLForConditionalGeneration" => Ok(Self::Qwen2_5VL),
            "Gemma3ForConditionalGeneration" | "Gemma3ForCausalLM" => Ok(Self::Gemma3),
            "Mistral3ForConditionalGeneration" => Ok(Self::Mistral3),
            "Llama4ForConditionalGeneration" => Ok(Self::Llama4),
            "Gemma3nForConditionalGeneration" => Ok(Self::Gemma3n),
            "Gemma4ForConditionalGeneration"
            | "Gemma4ForCausalLM"
            | "Gemma4UnifiedForConditionalGeneration"
            | "Gemma4UnifiedForCausalLM" => Ok(Self::Gemma4),
            "MuseGlimmerForConditionalGeneration" => Ok(Self::MuseGlimmer),
            "DiffusionGemmaForBlockDiffusion" => Ok(Self::DiffusionGemma),
            "PaddleOCRVLForConditionalGeneration" => Ok(Self::PaddleOcrVl),
            "Qwen3VLForConditionalGeneration" => Ok(Self::Qwen3VL),
            "Qwen3VLMoeForConditionalGeneration" => Ok(Self::Qwen3VLMoE),
            "Qwen3_5ForConditionalGeneration" => Ok(Self::Qwen3_5),
            "Qwen3_5MoeForConditionalGeneration" => Ok(Self::Qwen3_5Moe),
            "VoxtralRealtimeForConditionalGeneration" => Ok(Self::Voxtral),
            other => anyhow::bail!(
                "Unsupported Hugging Face Transformers -CausalLM model class `{other}`. Please raise an issue."
            ),
        }
    }
}

impl FromStr for MultimodalLoaderType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "phi3v" => Ok(Self::Phi3V),
            "idefics2" => Ok(Self::Idefics2),
            "llava_next" => Ok(Self::LLaVANext),
            "llava" => Ok(Self::LLaVA),
            "lfm2vl" | "lfm2_vl" => Ok(Self::Lfm2Vl),
            "vllama" => Ok(Self::VLlama),
            "qwen2vl" => Ok(Self::Qwen2VL),
            "idefics3" => Ok(Self::Idefics3),
            "minicpmo" => Ok(Self::MiniCpmO),
            "phi4mm" => Ok(Self::Phi4MM),
            "qwen2_5vl" => Ok(Self::Qwen2_5VL),
            "gemma3" => Ok(Self::Gemma3),
            "mistral3" => Ok(Self::Mistral3),
            "llama4" => Ok(Self::Llama4),
            "gemma3n" => Ok(Self::Gemma3n),
            "gemma4" => Ok(Self::Gemma4),
            "muse_glimmer" | "museglimmer" => Ok(Self::MuseGlimmer),
            "diffusiongemma" => Ok(Self::DiffusionGemma),
            "paddleocr_vl" => Ok(Self::PaddleOcrVl),
            "qwen3vl" => Ok(Self::Qwen3VL),
            "qwen3vlmoe" => Ok(Self::Qwen3VLMoE),
            "qwen3_5" => Ok(Self::Qwen3_5),
            "qwen3_5moe" => Ok(Self::Qwen3_5Moe),
            "voxtral" => Ok(Self::Voxtral),
            a => Err(format!("Unknown architecture `{a}`. Possible architectures: `phi3v`, `idefics2`, `llava_next`, `llava`, `lfm2vl`, `vllama`, `qwen2vl`, `idefics3`, `minicpmo`, `phi4mm`, `qwen2_5vl`, `gemma3`, `mistral3`, `llama4`, `gemma3n`, `gemma4`, `muse_glimmer`, `qwen3vl`, `qwen3vlmoe`, `qwen3_5`, `qwen3_5moe`, `voxtral`, `diffusiongemma`, `paddleocr_vl`.")),
        }
    }
}

impl std::fmt::Display for MultimodalLoaderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            MultimodalLoaderType::Phi3V => "phi3v",
            MultimodalLoaderType::Idefics2 => "idefics2",
            MultimodalLoaderType::LLaVANext => "llava_next",
            MultimodalLoaderType::LLaVA => "llava",
            MultimodalLoaderType::Lfm2Vl => "lfm2vl",
            MultimodalLoaderType::VLlama => "vllama",
            MultimodalLoaderType::Qwen2VL => "qwen2vl",
            MultimodalLoaderType::Idefics3 => "idefics3",
            MultimodalLoaderType::MiniCpmO => "minicpmo",
            MultimodalLoaderType::Phi4MM => "phi4mm",
            MultimodalLoaderType::Qwen2_5VL => "qwen2_5vl",
            MultimodalLoaderType::Gemma3 => "gemma3",
            MultimodalLoaderType::Mistral3 => "mistral3",
            MultimodalLoaderType::Llama4 => "llama4",
            MultimodalLoaderType::Gemma3n => "gemma3n",
            MultimodalLoaderType::Qwen3VL => "qwen3vl",
            MultimodalLoaderType::Qwen3VLMoE => "qwen3vlmoe",
            MultimodalLoaderType::Qwen3_5 => "qwen3_5",
            MultimodalLoaderType::Qwen3_5Moe => "qwen3_5moe",
            MultimodalLoaderType::Voxtral => "voxtral",
            MultimodalLoaderType::Gemma4 => "gemma4",
            MultimodalLoaderType::MuseGlimmer => "muse_glimmer",
            MultimodalLoaderType::DiffusionGemma => "diffusiongemma",
            MultimodalLoaderType::PaddleOcrVl => "paddleocr_vl",
        };
        write!(f, "{name}")
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
