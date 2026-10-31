//! What a model family knows about loading its models: the loader traits and their sizing and placement helpers.

mod isq;
mod placement;
mod rope;
mod sizing;
mod types;

pub use isq::*;
pub use placement::*;
pub use rope::*;
pub use sizing::*;
pub use types::*;

// A loader's vtable carries its trait defaults, so building the box where the loader is defined compiles them there.
#[macro_export]
macro_rules! boxed_loaders {
    ($loader_trait:ident: $($loader:ident),+ $(,)?) => {$(
        impl $loader {
            pub fn boxed() -> Box<dyn $crate::loaders::$loader_trait> {
                Box::new(Self)
            }
        }
    )+};
}

use std::borrow::Cow;
use std::fmt::Debug;
use std::sync::Arc;

use anyhow::Result;
use inference_quant::ShardedVarBuilder;
use inference_tensor::DType;

use crate::attention::ATTENTION_CHUNK_SIZE;
use crate::device_map::{AutoDeviceMapParams, DeviceMapper};
use crate::matformer::MatformerSliceConfig;
use crate::media_inputs::video::VideoFrameSampling;
use crate::model::{
    EmbeddingModel, MultimodalModel, NormalLoadingMetadata, NormalModel, RopePairing,
};
use crate::paged_attention::{AttentionImplementation, ModelConfigLike};
use crate::utils::varbuilder_utils::DeviceForLoadTensor;

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
        Ok(params.max_batch_size()
            * self.model_config(config)?.num_attn_heads()
            * params.max_seq_len().min(ATTENTION_CHUNK_SIZE).pow(2))
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
    /// The bytes of a token embedding no head shares, part of `non_mapped_size_in_bytes`, which can stay in host
    /// memory; `None` when the loader doesn't size it apart or the head is tied.
    fn untied_embedding_size_in_bytes(
        &self,
        _config: &str,
        _dtype: DType,
        _weight_pack_factor: usize,
        _quantization: Option<&AutoDeviceMapQuantization<'_>>,
    ) -> Result<Option<usize>> {
        Ok(None)
    }
    /// Decoder layers that load with the non-mapped weights (an MTP head) and are counted in `non_mapped_size_in_bytes`,
    /// in order after the mapped layers.
    fn non_mapped_layer_sizes_in_bytes(
        &self,
        _config: &str,
        _dtype: DType,
        _weight_pack_factor: usize,
    ) -> Result<Vec<usize>> {
        Ok(Vec::new())
    }
    /// Activation bytes each prompt token adds to a prefill chunk, where the loader knows them: an MoE model whose
    /// experts stay in host memory sizes its prompt chunk by them.
    fn prompt_activation_bytes_per_token(&self, _config: &str) -> Result<Option<usize>> {
        Ok(None)
    }
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

/// A config whose context length, at `key`, is cut to `max_model_len`, for a loader whose RoPE and sizing read only
/// that length (not one, like LongRoPE, that scales attention by it); longer than the trained context is an error.
pub fn cap_context_length<'a>(
    config: &'a str,
    key: &str,
    max_model_len: Option<usize>,
) -> Result<Cow<'a, str>> {
    let Some(max_model_len) = max_model_len else {
        return Ok(Cow::Borrowed(config));
    };
    let mut value: serde_json::Value = serde_json::from_str(config)?;
    let trained = value
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| anyhow::anyhow!("the config has no `{key}` to cut to max_model_len"))?;
    if max_model_len == 0 || max_model_len as u64 > trained {
        anyhow::bail!(
            "max_model_len={max_model_len} is outside the model's context of {trained} tokens"
        );
    }
    value[key] = serde_json::Value::from(max_model_len);
    Ok(Cow::Owned(serde_json::to_string(&value)?))
}

pub trait NormalModelLoader: IsqModelLoader + Send + Sync + DeviceMappedModelLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
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
    /// The chat template of a family whose checkpoints ship none, used when the checkpoint and request give none.
    fn default_chat_template(&self, _config: &str) -> Option<String> {
        None
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

/// One clip's transcript as an audio-in decoder writes it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TranscriptText {
    pub language: Option<String>,
    pub text: String,
}

/// How an audio-in decoder serves a transcription request: the whole prompt around one clip's placeholder, with the
/// answer prefix that forces an ISO-639-1 `language` when one is given, and the parse of its answer.
#[derive(Clone, Copy)]
pub struct AudioTranscriber {
    pub prompt: fn(language: Option<&str>) -> anyhow::Result<String>,
    pub parse: fn(&str) -> TranscriptText,
}

impl std::fmt::Debug for AudioTranscriber {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("AudioTranscriber")
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
    /// For an audio-in decoder that transcribes, how a transcription request becomes its prompt and back.
    fn audio_transcriber(&self, _config: &str) -> Option<AudioTranscriber> {
        None
    }
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
    /// Half-split RoPE, unless a GGUF translation stamped the adjacent pairs llama.cpp's Llama converter writes.
    fn is_gptx(&self, config: &str) -> Result<bool> {
        Ok(!matches!(
            qk_rope_layout_from_config(config)?,
            Some(RopePairing::Adjacent)
        ))
    }
    fn has_causal_attention(&self, config: &str) -> Result<bool>;
    /// A classifier outputs its logits per input, with no sentence-transformers modules after it.
    fn is_classifier(&self, _config: &str) -> Result<bool> {
        Ok(false)
    }
    /// The tasks a request may name, each picking one of the model's adapters; `None` for a model without them.
    fn embedding_tasks(&self, _config: &str) -> Result<Option<crate::embedding::EmbeddingTasks>> {
        Ok(None)
    }
    /// Whether the model's own code pools the last token, for a `modules.json` that lists no pooling.
    fn pools_last_token(&self, _config: &str) -> Result<bool> {
        Ok(false)
    }
    /// A decision model's budgets and calibration; `None` for an embedder or a reranker.
    fn decision_spec(&self, _config: &str) -> Result<Option<crate::decision::DecisionSpec>> {
        Ok(None)
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

#[cfg(test)]
mod tests {
    use super::cap_context_length;

    const KEY: &str = "max_position_embeddings";
    const CONFIG: &str = r#"{"max_position_embeddings":131072,"hidden_size":64}"#;

    #[test]
    fn a_context_cap_cuts_only_within_the_trained_length() -> anyhow::Result<()> {
        assert_eq!(cap_context_length(CONFIG, KEY, None)?, CONFIG);
        let capped: serde_json::Value =
            serde_json::from_str(&cap_context_length(CONFIG, KEY, Some(4096))?)?;
        assert_eq!(capped[KEY], 4096);
        assert_eq!(capped["hidden_size"], 64);
        assert!(cap_context_length(CONFIG, KEY, Some(262_144)).is_err());
        assert!(cap_context_length(CONFIG, KEY, Some(0)).is_err());
        assert!(cap_context_length(CONFIG, "seq_length", Some(4096)).is_err());
        Ok(())
    }
}
