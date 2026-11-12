use super::*;

use crate::vibevoice_asr::{
    ENCODE_SECONDS, VibeVoiceAsrConfig, VibeVoiceAsrModel, config::EncoderConfig,
    inputs_processor::clip_tokens, transcription::VIBEVOICE_ASR,
};

const TEXT_LAYER_INDEX_PATTERN: &str = r"language_model\.model\.layers\.(\d+)\.";
// a minute of 24 kHz audio, what one encoder chunk and a clip's attention budget are sized by
const MINUTE_SAMPLES: usize = 1_440_000;

/// `MultimodalLoader` for VibeVoice-ASR (`VibeVoiceAsrForConditionalGeneration`).
pub struct VibeVoiceAsrLoader;

// the repo's template writes each clip's placeholders and duration itself
pub struct VibeVoiceAsrPrefixer;

impl MultimodalPromptPrefixer for VibeVoiceAsrPrefixer {}

impl VibeVoiceAsrLoader {
    fn config(config: &str) -> Result<VibeVoiceAsrConfig> {
        Ok(VibeVoiceAsrConfig::from_json(config)?)
    }
}

// one encoder's weights: its convs, blocks and head
fn encoder_elems(cfg: &EncoderConfig) -> usize {
    let block = |w: usize| {
        let inner = cfg.ffn_expansion * w;
        w * cfg.kernel_size + w + 2 * w + 2 * w + (w * inner + inner) + (inner * w + w)
    };
    let mut total = cfg.channels * cfg.num_filters * cfg.kernel_size + cfg.num_filters;
    total += cfg.depths[0] * block(cfg.num_filters);
    for (i, &ratio) in cfg.downsampling_ratios.iter().enumerate() {
        let (from, to) = (cfg.num_filters << i, cfg.num_filters << (i + 1));
        total += from * to * 2 * ratio + to + cfg.depths[i + 1] * block(to);
    }
    let last = cfg.num_filters << cfg.downsampling_ratios.len();
    total + last * cfg.hidden_size * cfg.kernel_size + cfg.hidden_size
}

impl MultimodalModelLoader for VibeVoiceAsrLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn MultimodalModel + Send + Sync>> {
        Ok(Box::new(VibeVoiceAsrModel::new(
            &Self::config(config)?,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        Ok(Box::new(Self::config(config)?))
    }
    fn supports_paged_attention(&self, _config: &str) -> bool {
        true
    }
    fn supports_prefix_cacher(&self, _config: &str) -> bool {
        true
    }
    fn modalities(&self, _config: &str) -> Result<Modalities> {
        Ok(Modalities {
            input: vec![SupportedModality::Text, SupportedModality::Audio],
            output: vec![SupportedModality::Text],
        })
    }
    fn prefixer(&self, _config: &str) -> Arc<dyn MultimodalPromptPrefixer> {
        Arc::new(VibeVoiceAsrPrefixer)
    }
    fn audio_transcriber(&self, _config: &str) -> Option<AudioTranscriber> {
        Some(VIBEVOICE_ASR)
    }
    fn get_device_for_tensor(
        &self,
        config: &str,
        _mapper: &dyn DeviceMapper,
        loading_isq: bool,
    ) -> Result<Arc<dyn Fn(String) -> DeviceForLoadTensor + Send + Sync + 'static>> {
        layer_indexed_device(
            TEXT_LAYER_INDEX_PATTERN,
            Self::config(config)?.text().num_hidden_layers,
            loading_isq,
        )
    }
}

impl IsqModelLoader for VibeVoiceAsrLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"^language_model\.model\.embed_tokens\.weight$",
            r"^language_model\.lm_head\.(weight|bias)$",
        ])
    }

    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"^language_model\.lm_head\.(weight|bias)$",
            r"^language_model\.model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.(weight|bias)$",
            r"^language_model\.model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.(weight|bias)$",
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for VibeVoiceAsrLoader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Multimodal {
            max_seq_len,
            max_batch_size,
            max_num_images,
            ..
        } = params
        else {
            anyhow::bail!("Expected multimodal AutoDeviceMapParams for this model!")
        };
        let cfg = Self::config(config)?;
        let clip = clip_tokens(MINUTE_SAMPLES, cfg.hop_length());
        let seq_len = max_seq_len.min(&ATTENTION_CHUNK_SIZE) + clip * max_num_images.max(&1);
        Ok(max_batch_size * cfg.text().num_attention_heads * seq_len * seq_len)
    }
    fn non_mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Multimodal { max_batch_size, .. } = params else {
            anyhow::bail!("Expected multimodal AutoDeviceMapParams for this model!")
        };
        // the widest encoder activation: a chunk's samples through the stem's FFN
        let cfg = Self::config(config)?;
        let enc = &cfg.semantic_tokenizer_encoder_config;
        let chunk = ENCODE_SECONDS * crate::vibevoice_asr::inputs_processor::SAMPLE_RATE as usize;
        Ok(max_batch_size * chunk * enc.num_filters * enc.ffn_expansion)
    }
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg = Self::config(config)?;
        let tc = cfg.text();
        let (embed_pack, head_pack) = super::language_model_pack_factors(
            quantization,
            "language_model.model.embed_tokens.weight",
            "language_model.lm_head.weight",
            tc.tie_word_embeddings,
            dtype,
            weight_pack_factor,
        )?;
        let h = tc.hidden_size;
        let text = h * tc.vocab_size / embed_pack
            + if tc.tie_word_embeddings {
                0
            } else {
                h * tc.vocab_size / head_pack
            }
            + h;
        let (acoustic, semantic) = (
            &cfg.acoustic_tokenizer_encoder_config,
            &cfg.semantic_tokenizer_encoder_config,
        );
        let branch = |latent: usize| latent * h + h + h + h * h + h;
        let audio = encoder_elems(acoustic)
            + encoder_elems(semantic)
            + branch(acoustic.hidden_size)
            + branch(semantic.hidden_size);
        Ok((text + audio) * dtype.size_in_bytes())
    }
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        Qwen2Loader.layer_sizes_in_bytes(
            &serde_json::to_string(&Self::config(config)?.text())?,
            dtype,
            weight_pack_factor,
            matformer_config,
        )
    }
    fn num_layers(&self, config: &str) -> Result<usize> {
        Ok(Self::config(config)?.text().num_hidden_layers)
    }
    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Audio])
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        Qwen2Loader.model_config(&serde_json::to_string(&Self::config(config)?.text())?)
    }
}
