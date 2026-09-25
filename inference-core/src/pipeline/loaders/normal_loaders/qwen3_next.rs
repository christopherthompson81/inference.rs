use super::*;

// ======================== Qwen3Next loader

/// [`NormalLoader`] for a Qwen3Next (Qwen3-Coder-Next) model.
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct Qwen3NextLoader;

impl NormalModelLoader for Qwen3NextLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::qwen3_next::Config = serde_json::from_str(config)?;

        Ok(Box::new(models::qwen3_next::Model::new(
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
        anyhow::bail!("Qwen3Next does not support X-LoRA")
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::qwen3_next::Config = serde_json::from_str(config)?;
        Ok(Box::new(cfg))
    }
    fn supports_paged_attention(&self, _config: &str) -> Result<bool> {
        Ok(true)
    }
}

impl IsqModelLoader for Qwen3NextLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"^model\.embed_tokens\.weight$")?,
            Regex::new(r"^lm_head\.(weight|bias)$")?,
        ])
    }

    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            Regex::new(
                r"layers\.(\d+)\.linear_attn\.(in_proj_qkvz|in_proj_qkv|in_proj_z|in_proj_ba|in_proj_b|in_proj_a)\.(weight|bias)$",
            )?,
            Regex::new(r"layers\.(\d+)\.linear_attn\.out_proj\.(weight|bias)$")?,
            Regex::new(
                r"layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|bias)$",
            )?,
            Regex::new(r"layers\.(\d+)\.mlp\.experts\.(gate_proj|up_proj|down_proj)\.weight$")?,
            Regex::new(
                r"layers\.(\d+)\.mlp\.shared_expert\.(gate_proj|up_proj|down_proj)\.(weight|bias)$",
            )?,
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
    fn isq_layer_regexes_moqe(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(
                r"layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|bias)$",
            )?,
            Regex::new(r"layers\.(\d+)\.mlp\.experts\.(gate_proj|up_proj|down_proj)\.weight$")?,
        ])
    }
    fn immediate_isq_predicates_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes_moqe(config)
    }
}

impl DeviceMappedModelLoader for Qwen3NextLoader {
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

        let cfg: crate::models::qwen3_next::Config = serde_json::from_str(config)?;

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
        let cfg: crate::models::qwen3_next::Config = serde_json::from_str(config)?;

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
        let cfg: crate::models::qwen3_next::Config = serde_json::from_str(config)?;
        let layer_types = cfg.layer_types();
        let mut layer_sizes = Vec::with_capacity(cfg.num_hidden_layers);

        for layer_type in &layer_types {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let attn_elems = match layer_type {
                crate::models::qwen3_next::LayerType::FullAttention => {
                    let hidden = cfg.hidden_size;
                    let q_dim = cfg.head_dim * cfg.num_attention_heads;
                    let kv_dim = cfg.head_dim * cfg.num_key_value_heads;
                    let q_proj = hidden * q_dim * 2 / weight_pack_factor;
                    let k_proj = hidden * kv_dim / weight_pack_factor;
                    let v_proj = hidden * kv_dim / weight_pack_factor;
                    let o_proj = q_dim * hidden / weight_pack_factor;
                    let q_norm = cfg.head_dim;
                    let k_norm = cfg.head_dim;
                    q_proj + k_proj + v_proj + o_proj + q_norm + k_norm
                }
                crate::models::qwen3_next::LayerType::LinearAttention => {
                    let hidden = cfg.hidden_size;
                    let key_dim = cfg.linear_key_dim();
                    let value_dim = cfg.linear_value_dim();
                    let conv_dim = cfg.linear_conv_dim();
                    // in_proj_qkvz: (2 * key_dim + 2 * value_dim, hidden)
                    let in_proj_qkvz = hidden * (key_dim * 2 + value_dim * 2) / weight_pack_factor;
                    // in_proj_ba: (2 * num_v_heads, hidden)
                    let in_proj_ba = hidden * (cfg.linear_num_value_heads * 2) / weight_pack_factor;
                    let out_proj = value_dim * hidden / weight_pack_factor;
                    let conv1d = conv_dim * cfg.linear_conv_kernel_dim;
                    let dt_bias = cfg.linear_num_value_heads;
                    let a_log = cfg.linear_num_value_heads;
                    let norm = cfg.linear_value_head_dim;
                    in_proj_qkvz + in_proj_ba + out_proj + conv1d + dt_bias + a_log + norm
                }
            };

            let moe_gate = cfg.hidden_size * cfg.num_experts;
            let shared_expert =
                3 * cfg.hidden_size * cfg.shared_expert_intermediate_size / weight_pack_factor;
            let routed_experts = cfg.num_experts * 3 * cfg.hidden_size * cfg.moe_intermediate_size
                / weight_pack_factor;

            let per_layer_elems = input_layernorm
                + post_attention_layernorm
                + attn_elems
                + moe_gate
                + shared_expert
                + routed_experts;

            layer_sizes.push(per_layer_elems * dtype.size_in_bytes());
        }

        Ok(layer_sizes)
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: crate::models::qwen3_next::Config = serde_json::from_str(config)?;
        Ok(cfg.num_hidden_layers)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: crate::models::qwen3_next::Config = serde_json::from_str(config)?;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: None,
            k_head_dim: cfg.head_dim,
            v_head_dim: cfg.head_dim,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }
}
