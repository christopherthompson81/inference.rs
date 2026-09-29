//! What a model family knows about loading its models: the loader traits and their sizing and placement helpers.

mod isq;
mod placement;
mod rope;
mod sizing;

pub use isq::*;
pub use placement::*;
pub use rope::*;
pub use sizing::*;

use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;

use anyhow::Result;
use candle_core::DType;
use inference_quant::ShardedVarBuilder;

use crate::attention::ATTENTION_CHUNK_SIZE;
use crate::device_map::{AutoDeviceMapParams, DeviceMapper};
use crate::lora::{LoraConfig, Ordering};
use crate::matformer::MatformerSliceConfig;
use crate::media_inputs::video::VideoFrameSampling;
use crate::model::{
    EmbeddingModel, MultimodalModel, NormalLoadingMetadata, NormalModel, RopePairing,
};
use crate::paged_attention::{AttentionImplementation, ModelConfigLike};
use crate::utils::varbuilder_utils::DeviceForLoadTensor;
use crate::xlora::XLoraConfig;

#[derive(Clone, PartialEq, Eq)]
pub enum SupportedModality {
    Text,
    Audio,
    Vision,
    Video,
    Embedding,
}

impl Debug for SupportedModality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Text => write!(f, "📝 Text"),
            Self::Audio => write!(f, "🔊 Audio"),
            Self::Vision => write!(f, "🖼️ Vision"),
            Self::Video => write!(f, "🎬 Video"),
            Self::Embedding => write!(f, "🔢 Embedding"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Modalities {
    pub input: Vec<SupportedModality>,
    pub output: Vec<SupportedModality>,
}

/// Prepend a vision tag appropriate for the model to the prompt. Image indexing is assumed that start at 0.
pub trait MultimodalPromptPrefixer: Send + Sync {
    /// Prefix for inclusion in messages (may do nothing if the chat template handles it).
    fn prefix_image(&self, _image_indices: Vec<usize>, prompt: &str) -> String {
        prompt.to_string()
    }
    /// Prefix for inclusion in messages (may do nothing if the chat template handles it).
    fn prefix_audio(&self, _audio_indexes: Vec<usize>, prompt: &str) -> String {
        prompt.to_string()
    }
    /// Prefix for inclusion in messages (may do nothing if the chat template handles it).
    fn prefix_video(&self, _video_indexes: Vec<usize>, prompt: &str) -> String {
        prompt.to_string()
    }
}

pub trait DeviceMappedModelLoader {
    /// Maximum activation size of non-mapped parts of this model.
    /// Useful for the multimodal models which may prefer to keep the vison components on the GPU.
    fn non_mapped_max_act_size_elems(
        &self,
        _config: &str,
        _params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Ok(0)
    }
    /// Maximum activation size of mapped parts of the model; the default is a text decoder's attention scores.
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };
        Ok(max_batch_size
            * self.model_config(config)?.num_attn_heads()
            * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2))
    }
    /// weight_pack_factor only applies to quantized weights.
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        quantization: Option<&AutoDeviceMapQuantization<'_>>,
        matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize>;
    /// weight_pack_factor only applies to quantized weights.
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>>;
    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        None
    }
    fn non_mapped_sub_models_for_config(
        &self,
        _config: &str,
    ) -> Result<Option<Vec<NonMappedSubModel>>> {
        Ok(self.non_mapped_sub_models())
    }
    fn num_layers(&self, config: &str) -> Result<usize> {
        Ok(self.model_config(config)?.num_layers())
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>>;

    fn checkpoint_layer_index(&self, _config: &str, tensor_name: &str) -> Option<usize> {
        standard_layer_index(tensor_name)
    }
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
    fn is_gptx(&self, _config: &str) -> Result<bool> {
        Ok(true)
    }
    fn is_gptx_for(
        &self,
        config: &str,
        normal_loading_metadata: &NormalLoadingMetadata,
    ) -> Result<bool> {
        match normal_loading_metadata.rope_pairing {
            Some(RopePairing::Adjacent) => Ok(false),
            Some(RopePairing::HalfSplit) => Ok(true),
            None => match qk_rope_layout_from_config(config)? {
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
        layer_indexed_device(
            LAYER_INDEX_PATTERN,
            self.model_config(config)?.num_layers(),
            loading_isq,
        )
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
    fn is_gptx(&self, _config: &str) -> bool {
        true
    }
    fn is_gptx_for(
        &self,
        config: &str,
        normal_loading_metadata: &NormalLoadingMetadata,
    ) -> Result<bool> {
        match normal_loading_metadata.rope_pairing {
            Some(RopePairing::Adjacent) => Ok(false),
            Some(RopePairing::HalfSplit) => Ok(true),
            None => match qk_rope_layout_from_config(config)? {
                Some(RopePairing::Adjacent) => Ok(false),
                Some(RopePairing::HalfSplit) => Ok(true),
                None => Ok(self.is_gptx(config)),
            },
        }
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>>;
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
    fn video_frame_sampling(&self, _config: &str) -> VideoFrameSampling {
        VideoFrameSampling::default()
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
        layer_indexed_device(
            LAYER_INDEX_PATTERN,
            self.model_config(config)?.num_layers(),
            loading_isq,
        )
    }
}

pub trait EmbeddingModelLoader: IsqModelLoader + Send + Sync + DeviceMappedModelLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn EmbeddingModel + Send + Sync>>;
    fn is_gptx(&self, _config: &str) -> Result<bool> {
        Ok(true)
    }
    fn has_causal_attention(&self, config: &str) -> Result<bool>;
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>>;
    fn get_device_for_tensor(
        &self,
        config: &str,
        _mapper: &dyn DeviceMapper,
        loading_isq: bool,
    ) -> Result<Arc<dyn Fn(String) -> DeviceForLoadTensor + Send + Sync + 'static>> {
        layer_indexed_device(
            LAYER_INDEX_PATTERN,
            self.model_config(config)?.num_layers(),
            loading_isq,
        )
    }
}
