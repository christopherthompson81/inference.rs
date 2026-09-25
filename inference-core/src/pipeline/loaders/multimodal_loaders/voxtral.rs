use super::*;

// ─── Voxtral ────────────────────────────────────────────────────────────────

/// [`MultimodalLoader`] for a Voxtral model.
///
/// [`MultimodalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.MultimodalLoader.html
pub struct VoxtralLoader;

pub struct VoxtralPrefixer;

impl MultimodalPromptPrefixer for VoxtralPrefixer {
    fn prefix_image(&self, _image_indexes: Vec<usize>, prompt: &str) -> String {
        prompt.to_string()
    }
}

impl MultimodalModelLoader for VoxtralLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn MultimodalModel + Send + Sync>> {
        let cfg: VoxtralConfig = serde_json::from_str(config)?;
        Ok(Box::new(VoxtralModel::new(
            &cfg,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn is_gptx(&self, _config: &str) -> bool {
        true
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: VoxtralConfig = serde_json::from_str(config)?;
        Ok(Box::new(cfg))
    }
    fn get_processor(
        &self,
        model_config: &str,
        _processor_config: Option<ProcessorConfig>,
        _preprocessor_config: PreProcessorConfig,
        _max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        let cfg: VoxtralConfig =
            serde_json::from_str(model_config).expect("Failed to parse VoxtralConfig");
        Arc::new(VoxtralProcessor::new(&cfg))
    }
    fn supports_paged_attention(&self, _config: &str) -> bool {
        true
    }
    fn supports_prefix_cacher(&self, _config: &str) -> bool {
        true
    }
    fn prefixer(&self, _config: &str) -> Arc<dyn MultimodalPromptPrefixer> {
        Arc::new(VoxtralPrefixer)
    }
    fn modalities(&self, _config: &str) -> Result<Modalities> {
        Ok(Modalities {
            input: vec![SupportedModality::Text, SupportedModality::Audio],
            output: vec![SupportedModality::Text],
        })
    }
    fn default_chat_template(&self, _config: &str) -> Option<String> {
        // Mistral v7 instruct format using [INST]/[/INST] tokens
        Some("{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token + ' ' }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}".to_string())
    }
    fn default_bos_eos(&self, _config: &str) -> Option<(String, String)> {
        // Mistral tekken tokenizer: <s> = ID 1, </s> = ID 2
        Some(("<s>".to_string(), "</s>".to_string()))
    }
}

impl IsqModelLoader for VoxtralLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"^mm_streams_embeddings\.embedding_module\.tok_embeddings\.weight$")?,
            Regex::new(r"^output\.(weight|bias)$")?,
        ])
    }

    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            // Output / lm_head (tied with tok_embeddings)
            Regex::new(r"lm_head\.(weight|bias)$")?,
            Regex::new(r"^output\.(weight|bias)$")?,
            // Decoder attention (Mistral-native naming)
            Regex::new(r"layers\.(\d+)\.attention\.wq\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.attention\.wk\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.attention\.wv\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.attention\.wo\.(weight|bias)$")?,
            // Decoder MLP (Mistral-native naming)
            Regex::new(r"layers\.(\d+)\.feed_forward\.w1\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.feed_forward\.w3\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.feed_forward\.w2\.(weight|bias)$")?,
        ])
    }
    fn immediate_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"tok_embeddings\.(weight|bias)$")?,
            Regex::new(r"^output\.(weight|bias)$")?,
            // Decoder attention
            Regex::new(r"layers\.(\d+)\.attention\.wq\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.attention\.wk\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.attention\.wv\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.attention\.wo\.(weight|bias)$")?,
            // Decoder MLP
            Regex::new(r"layers\.(\d+)\.feed_forward\.w1\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.feed_forward\.w3\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.feed_forward\.w2\.(weight|bias)$")?,
        ])
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
impl DeviceMappedModelLoader for VoxtralLoader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Multimodal {
            max_seq_len,
            max_batch_size,
            ..
        } = params
        else {
            anyhow::bail!("Expected multimodal AutoDeviceMapParams for this model!")
        };

        let cfg: VoxtralConfig = serde_json::from_str(config)?;

        // Audio tokens are prepended: max audio len + text seq len
        // Audio: ~30s at 16kHz = 480k samples, /160 hop = 3000 frames, /2 conv stride = 1500, /4 adapter = 375 tokens
        let max_audio_tokens = 375;
        let total_seq = max_audio_tokens + *max_seq_len.min(&ATTENTION_CHUNK_SIZE);
        Ok(max_batch_size * cfg.n_heads * total_seq * total_seq)
    }

    fn non_mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Multimodal { max_batch_size, .. } = params else {
            anyhow::bail!("Expected multimodal AutoDeviceMapParams for this model!")
        };

        let cfg: VoxtralConfig = serde_json::from_str(config)?;
        let enc = &cfg.multimodal.whisper_model_args.encoder_args;
        // Encoder max activation: attention matrix
        // ~3000 mel frames, encoder has 32 heads, seq_len^2
        let max_enc_seq = 3000usize;
        Ok(max_batch_size * enc.n_heads * max_enc_seq * max_enc_seq)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: VoxtralConfig = serde_json::from_str(config)?;
        let enc = &cfg.multimodal.whisper_model_args.encoder_args;
        let ds = &cfg.multimodal.whisper_model_args.downsample_args;

        let elem = dtype.size_in_bytes();

        // Encoder conv layers
        let conv1 = enc.dim * enc.audio_encoding_args.num_mel_bins * 3 + enc.dim; // weight + bias
        let conv2 = enc.dim * enc.dim * 3 + enc.dim;

        // Encoder layers
        let enc_attn_per_layer = 4 * enc.dim * enc.dim; // wq, wk, wv, wo (full heads)
        let enc_mlp_per_layer = 3 * enc.dim * enc.hidden_dim; // w1, w2, w3
        let enc_norm_per_layer = 2 * enc.dim; // attention_norm, ffn_norm
        let enc_layers =
            enc.n_layers * (enc_attn_per_layer + enc_mlp_per_layer + enc_norm_per_layer);
        let enc_final_norm = enc.dim;

        // Adapter
        let adapter_in_features = enc.dim * ds.downsample_factor;
        let adapter = adapter_in_features * cfg.dim + cfg.dim + cfg.dim * cfg.dim + cfg.dim;

        let total_encoder = conv1 + conv2 + enc_layers + enc_final_norm + adapter;

        let (embedding_pack_factor, output_pack_factor) = super::language_model_pack_factors(
            _quantization,
            "mm_streams_embeddings.embedding_module.tok_embeddings.weight",
            "output.weight",
            cfg.tied_embeddings,
            dtype,
            weight_pack_factor,
        )?;
        let embeddings = cfg.vocab_size * cfg.dim / embedding_pack_factor;
        let output = if cfg.tied_embeddings {
            0
        } else {
            cfg.vocab_size * cfg.dim / output_pack_factor
        };

        Ok((total_encoder + embeddings + output) * elem)
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: VoxtralConfig = serde_json::from_str(config)?;
        let elem = dtype.size_in_bytes();

        let attn = (cfg.dim * cfg.n_heads * cfg.head_dim
            + cfg.dim * cfg.n_kv_heads * cfg.head_dim
            + cfg.dim * cfg.n_kv_heads * cfg.head_dim
            + cfg.n_heads * cfg.head_dim * cfg.dim)
            / weight_pack_factor;
        let mlp = (cfg.dim * cfg.hidden_dim + cfg.hidden_dim * cfg.dim + cfg.dim * cfg.hidden_dim)
            / weight_pack_factor;
        let norms = 2 * cfg.dim; // attention_norm + ffn_norm

        let per_layer = (attn + mlp + norms) * elem;

        Ok(vec![per_layer; cfg.n_layers])
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: VoxtralConfig = serde_json::from_str(config)?;
        Ok(cfg.n_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: VoxtralConfig = serde_json::from_str(config)?;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.model_max_length,
            num_layers: cfg.n_layers,
            hidden_size: cfg.dim,
            num_kv_heads: cfg.n_kv_heads,
            num_attn_heads: cfg.n_heads,
            sliding_window: cfg.sliding_window,
            k_head_dim: cfg.head_dim,
            v_head_dim: cfg.head_dim,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }
}
