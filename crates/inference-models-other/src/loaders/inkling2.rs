use super::*;

/// `NormalLoader` for Inkling 2's looped text model.
pub struct Inkling2Loader;

impl NormalModelLoader for Inkling2Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg = crate::inkling2::Config::from_json(config)?;
        Ok(Box::new(crate::inkling2::Model::new(
            &cfg,
            vb,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }

    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        Ok(Box::new(crate::inkling2::Config::from_json(config)?))
    }

    // the paged cache is read by the fused CUDA attention only
    fn supports_paged_attention(&self, _config: &str) -> Result<bool> {
        Ok(cfg!(feature = "cuda"))
    }
}

impl IsqModelLoader for Inkling2Loader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[r"^model\.embed_tokens\.weight$", r"^lm_head\.weight$"])
    }
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"lm_head\.(weight|bias)$",
            r"layers\.(\d+)\.self_attn\.(q|k|v|o|r|gate)_proj\.weight$",
            r"layers\.(\d+)\.mlp\.experts\.(gate_proj|up_proj|down_proj)\.weight$",
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for Inkling2Loader {
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg = crate::inkling2::Config::from_json(config)?;
        standard_non_mapped_size_in_bytes(
            LanguageModelEnds {
                hidden_size: cfg.hidden_size,
                vocab_size: cfg.vocab_size,
                tie_word_embeddings: false,
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
        let cfg = crate::inkling2::Config::from_json(config)?;
        let hidden = cfg.hidden_size;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let attention = (hidden * (3 * q_dim + 2 * kv_dim) + q_dim * hidden) / weight_pack_factor;
        let experts =
            3 * cfg.n_routed_experts * hidden * cfg.intermediate_size / weight_pack_factor;
        let router = hidden * cfg.n_routed_experts;
        Ok(vec![
            (attention + experts + router) * dtype.size_in_bytes();
            cfg.num_hidden_layers
        ])
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = crate::inkling2::Config::from_json(config)?;
        let slots = cfg.cache_slots();
        let base = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: slots,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: None,
            k_head_dim: cfg.head_dim,
            v_head_dim: cfg.head_dim,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };
        // every slot attends through the paged cache; each keeps an f32 convolution state beside it
        Ok(Box::new(
            inference_nn::paged_attention::HybridPagedKvCacheConfig::new(base, vec![true; slots])
                .with_recurrent_state_bytes(cfg.conv_state_bytes()),
        ))
    }
}
