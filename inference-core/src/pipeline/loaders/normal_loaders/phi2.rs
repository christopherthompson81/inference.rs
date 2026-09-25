use super::*;

/// [`NormalLoader`] for a Phi 2 model.
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct Phi2Loader;

impl NormalModelLoader for Phi2Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::phi2::Config = serde_json::from_str(config)?;

        Ok(Box::new(models::phi2::Model::new(
            &cfg,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        normal_loading_metadata: NormalLoadingMetadata,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::phi2::Config = serde_json::from_str(config)?;

        Ok(Box::new(xlora_models::XLoraPhi2::new(
            &cfg,
            vb,
            lora_config,
            xlora_config,
            xlora_ordering,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            preload_adapters,
        )?))
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::phi2::Config = serde_json::from_str(config)?;

        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for Phi2Loader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"^model\.embed_tokens\.weight$")?,
            Regex::new(r"^lm_head\.(weight|bias)$")?,
        ])
    }

    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.dense\.(weight|bias)$")?,
            // MLP
            Regex::new(r"layers\.(\d+)\.mlp\.fc1\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.fc2\.(weight|bias)$")?,
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for Phi2Loader {
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

        let cfg: crate::models::phi2::Config = serde_json::from_str(config)?;

        Ok(
            max_batch_size
                * cfg.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2),
        )
    }
    fn non_mapped_max_act_size_elems(
        &self,
        _config: &str,
        _params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Ok(0)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: crate::models::phi2::Config = serde_json::from_str(config)?;

        let elems = {
            let (embed_tokens_pack_factor, lm_head_pack_factor) =
                super::language_model_pack_factors(
                    _quantization,
                    "model.embed_tokens.weight",
                    "lm_head.weight",
                    cfg.tie_word_embeddings,
                    dtype,
                    weight_pack_factor,
                )?;
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / embed_tokens_pack_factor;
            let lm_head = if !cfg.tie_word_embeddings {
                cfg.hidden_size * cfg.vocab_size / lm_head_pack_factor
            } else {
                0
            };
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };
        Ok(elems * dtype.size_in_bytes())
    }
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: crate::models::phi2::Config = serde_json::from_str(config)?;

        let per_layer_elems = {
            let input_layernorm = cfg.hidden_size + cfg.hidden_size;

            let size_in = cfg.hidden_size;
            let size_q = cfg.head_dim() * cfg.num_attention_heads;
            let size_kv = cfg.head_dim() * cfg.num_key_value_heads();
            let q_proj = size_in * size_q / weight_pack_factor + size_q;
            let k_proj = size_in * size_kv / weight_pack_factor + size_kv;
            let v_proj = size_in * size_kv / weight_pack_factor + size_kv;
            let o_proj = size_q * size_in / weight_pack_factor + size_in;
            let (q_norm, k_norm) = if cfg.qk_layernorm {
                (cfg.head_dim(), cfg.head_dim())
            } else {
                (0, 0)
            };

            let h_size = cfg.hidden_size;
            let i_size = cfg.intermediate_size;
            let fc1 = h_size * i_size / weight_pack_factor;
            let fc2 = h_size * i_size / weight_pack_factor;

            input_layernorm + q_proj + k_proj + v_proj + o_proj + q_norm + k_norm + fc1 + fc2
        };
        Ok(vec![
            per_layer_elems * dtype.size_in_bytes();
            cfg.num_hidden_layers
        ])
    }
    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: crate::models::phi2::Config = serde_json::from_str(config)?;

        Ok(cfg.num_hidden_layers)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: crate::models::phi2::Config = serde_json::from_str(config)?;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads(),
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: None,
            k_head_dim: cfg.head_dim(),
            v_head_dim: cfg.head_dim(),
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }
}
