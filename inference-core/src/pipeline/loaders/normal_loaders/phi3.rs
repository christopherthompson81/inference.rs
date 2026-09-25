use super::*;

/// [`NormalLoader`] for a Phi 3 model.
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct Phi3Loader;

impl NormalModelLoader for Phi3Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::phi3::Config = serde_json::from_str(config)?;

        Ok(Box::new(models::phi3::Model::new(
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
        let cfg: crate::models::phi3::Config = serde_json::from_str(config)?;

        Ok(Box::new(xlora_models::XLoraPhi3::new(
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
        let cfg: crate::models::phi3::Config = serde_json::from_str(config)?;

        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for Phi3Loader {
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
            Regex::new(r"layers\.(\d+)\.self_attn\.qkv_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // MLP
            Regex::new(r"layers\.(\d+)\.mlp\.gate_up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for Phi3Loader {
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

        let cfg: crate::models::phi3::Config = serde_json::from_str(config)?;

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
        let cfg: crate::models::phi3::Config = serde_json::from_str(config)?;

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
        let cfg: crate::models::phi3::Config = serde_json::from_str(config)?;

        let per_layer_elems = {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let size_in = cfg.hidden_size;
            let head_dim = cfg.head_dim();
            let op_size =
                cfg.num_attention_heads * head_dim + 2 * cfg.num_key_value_heads * head_dim;
            let qkv_proj = size_in * op_size / weight_pack_factor;
            let o_proj =
                (cfg.num_attention_heads * head_dim) * size_in / weight_pack_factor + size_in;

            let h_size = cfg.hidden_size;
            let i_size = cfg.intermediate_size;
            let gate_up_proj = h_size * (2 * i_size) / weight_pack_factor;
            let down_proj = h_size * i_size / weight_pack_factor;

            input_layernorm
                + post_attention_layernorm
                + qkv_proj
                + o_proj
                + gate_up_proj
                + down_proj
        };
        Ok(vec![
            per_layer_elems * dtype.size_in_bytes();
            cfg.num_hidden_layers
        ])
    }
    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: crate::models::phi3::Config = serde_json::from_str(config)?;

        Ok(cfg.num_hidden_layers)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: crate::models::phi3::Config = serde_json::from_str(config)?;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: cfg.sliding_window,
            k_head_dim: cfg.head_dim(),
            v_head_dim: cfg.head_dim(),
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }
}
