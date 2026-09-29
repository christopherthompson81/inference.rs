use super::*;

/// `NormalLoader` for an LFM2 hybrid attention/short-conv model.
pub struct Lfm2Loader;

impl NormalModelLoader for Lfm2Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg = crate::lfm2::Config::from_json(config)?;

        Ok(Box::new(crate::lfm2::Model::new(
            &cfg,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        _config: &str,
        _vb: ShardedVarBuilder,
        _lora_config: &[((String, String), LoraConfig)],
        _xlora_config: Option<XLoraConfig>,
        _xlora_ordering: Ordering,
        _normal_loading_metadata: NormalLoadingMetadata,
        _preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        anyhow::bail!("LFM2 does not support X-LoRA")
    }

    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg = crate::lfm2::Config::from_json(config)?;
        Ok(Box::new(cfg))
    }
    fn supports_paged_attention(&self, _config: &str) -> Result<bool> {
        Ok(true)
    }
}

impl IsqModelLoader for Lfm2Loader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"^model\.embed_tokens\.weight$",
            r"^lm_head\.(weight|bias)$",
        ])
    }

    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"lm_head\.(weight|bias)$",
            r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$",
            r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$",
            r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$",
            r"layers\.(\d+)\.self_attn\.out_proj\.(weight|bias)$",
            r"layers\.(\d+)\.conv\.in_proj\.(weight|bias)$",
            r"layers\.(\d+)\.conv\.out_proj\.(weight|bias)$",
            r"layers\.(\d+)\.feed_forward\.w1\.(weight|bias)$",
            r"layers\.(\d+)\.feed_forward\.w2\.(weight|bias)$",
            r"layers\.(\d+)\.feed_forward\.w3\.(weight|bias)$",
            r"layers\.(\d+)\.feed_forward\.experts\.(\d+)\.w1\.(weight|bias)$",
            r"layers\.(\d+)\.feed_forward\.experts\.(\d+)\.w2\.(weight|bias)$",
            r"layers\.(\d+)\.feed_forward\.experts\.(\d+)\.w3\.(weight|bias)$",
            r"layers\.(\d+)\.feed_forward\.experts\.(gate_proj|up_proj|down_proj)\.weight$",
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
    fn isq_layer_regexes_moqe(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"layers\.(\d+)\.feed_forward\.experts\.(\d+)\.w1\.(weight|bias)$",
            r"layers\.(\d+)\.feed_forward\.experts\.(\d+)\.w2\.(weight|bias)$",
            r"layers\.(\d+)\.feed_forward\.experts\.(\d+)\.w3\.(weight|bias)$",
            r"layers\.(\d+)\.feed_forward\.experts\.(gate_proj|up_proj|down_proj)\.weight$",
        ])
    }
    fn immediate_isq_predicates_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes_moqe(config)
    }
}

impl DeviceMappedModelLoader for Lfm2Loader {
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg = crate::lfm2::Config::from_json(config)?;
        standard_non_mapped_size_in_bytes(
            LanguageModelEnds {
                hidden_size: cfg.hidden_size,
                vocab_size: cfg.vocab_size,
                tie_word_embeddings: cfg.tie_word_embeddings(),
            },
            quantization,
            dtype,
            weight_pack_factor,
        )
    }
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg = crate::lfm2::Config::from_json(config)?;
        let head_dim = cfg.head_dim();
        let hidden = cfg.hidden_size;
        let intermediate = cfg.intermediate_size();
        let mut sizes = Vec::with_capacity(cfg.num_hidden_layers);

        for (layer_idx, layer_type) in cfg.layer_types().into_iter().enumerate() {
            let operator_norm = hidden;
            let ffn_norm = hidden;
            let feed_forward = match cfg.feed_forward_type(layer_idx) {
                crate::lfm2::FeedForwardType::Dense => {
                    3 * hidden * intermediate / weight_pack_factor
                }
                crate::lfm2::FeedForwardType::Moe => {
                    let gate = hidden * cfg.num_experts;
                    let expert_bias = if cfg.use_expert_bias {
                        cfg.num_experts
                    } else {
                        0
                    };
                    let experts = 3 * cfg.num_experts * hidden * cfg.moe_intermediate_size
                        / weight_pack_factor;
                    gate + expert_bias + experts
                }
            };
            let operator = match layer_type {
                crate::lfm2::LayerType::Attention => {
                    let q_dim = cfg.num_attention_heads * head_dim;
                    let kv_dim = cfg.num_key_value_heads * head_dim;
                    let projections = (hidden * q_dim + hidden * kv_dim * 2 + q_dim * hidden)
                        / weight_pack_factor;
                    projections + 2 * head_dim
                }
                crate::lfm2::LayerType::Conv => {
                    let projections = (hidden * 3 * hidden + hidden * hidden) / weight_pack_factor;
                    let conv = hidden * cfg.conv_l_cache;
                    let bias = if cfg.conv_bias { 5 * hidden } else { 0 };
                    projections + conv + bias
                }
            };

            sizes
                .push((operator_norm + ffn_norm + operator + feed_forward) * dtype.size_in_bytes());
        }
        Ok(sizes)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = crate::lfm2::Config::from_json(config)?;
        let head_dim = cfg.head_dim();
        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: None,
            k_head_dim: head_dim,
            v_head_dim: head_dim,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }
}
