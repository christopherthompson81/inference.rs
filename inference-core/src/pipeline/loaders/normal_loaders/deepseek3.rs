use super::*;

/// [`NormalLoader`] for a DeepSeekV3 model.
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct DeepSeekV3Loader;

impl NormalModelLoader for DeepSeekV3Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::deepseek3::DeepSeekV3Config = serde_json::from_str(config)?;
        Ok(Box::new(models::deepseek3::DeepSeekV3::new(
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
        todo!()
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::deepseek3::DeepSeekV3Config = serde_json::from_str(config)?;
        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for DeepSeekV3Loader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"^model\.embed_tokens\.weight$")?,
            Regex::new(r"^lm_head\.(weight|bias)$")?,
        ])
    }
    fn isq_layer_regexes(&self, config: &str) -> Result<Vec<Regex>> {
        let mut data = vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"layers\.(\d+)\.self_attn\.kv_a_proj_with_mqa\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.(kv_b|k_b|v_b)_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.experts\.(gate_proj|up_proj|down_proj)\.weight$")?,
        ];
        let cfg: crate::models::deepseek3::DeepSeekV3Config = serde_json::from_str(config)?;
        if cfg.q_lora_rank.is_some() {
            data.extend(vec![
                Regex::new(r"layers\.(\d+)\.self_attn\.q_a_proj\.(weight|bias)$")?,
                Regex::new(r"layers\.(\d+)\.self_attn\.q_b_proj\.(weight|bias)$")?,
            ]);
        } else {
            data.push(Regex::new(
                r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$",
            )?);
        }
        for layer_idx in 0..cfg.num_hidden_layers {
            if let Some(n_routed_experts) = cfg.n_routed_experts.filter(|_| {
                layer_idx >= cfg.first_k_dense_replace && layer_idx % cfg.moe_layer_freq == 0
            }) {
                for i in 0..n_routed_experts {
                    data.extend(vec![
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.experts\.{i}\.gate_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.experts\.{i}\.up_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.experts\.{i}\.down_proj\.(weight|bias)$"
                        ))?,
                    ]);
                }
                if cfg.n_shared_experts.is_some() {
                    data.extend(vec![
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.shared_experts\.gate_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.shared_experts\.up_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.shared_experts\.down_proj\.(weight|bias)$"
                        ))?,
                    ]);
                }
            } else {
                data.extend(vec![
                    Regex::new(&format!(
                        r"layers\.{layer_idx}\.mlp\.gate_proj\.(weight|bias)$"
                    ))?,
                    Regex::new(&format!(r"layers.{layer_idx}.mlp\.up_proj\.(weight|bias)$"))?,
                    Regex::new(&format!(
                        r"layers\.{layer_idx}\.mlp\.down_proj\.(weight|bias)$"
                    ))?,
                ]);
            };
        }
        Ok(data)
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

impl DeviceMappedModelLoader for DeepSeekV3Loader {
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

        let cfg: crate::models::deepseek3::DeepSeekV3Config = serde_json::from_str(config)?;

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
        let cfg: crate::models::deepseek3::DeepSeekV3Config = serde_json::from_str(config)?;
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
        let cfg: crate::models::deepseek3::DeepSeekV3Config = serde_json::from_str(config)?;
        let mut per_layer_elems = Vec::new();

        for layer_idx in 0..cfg.num_hidden_layers {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let q_proj = match cfg.q_lora_rank {
                Some(lora_rank) => {
                    let a = cfg.hidden_size * lora_rank;
                    let norm = lora_rank;
                    let b = (cfg.num_attention_heads * cfg.q_head_dim()) * lora_rank;
                    a + norm + b
                }
                None => (cfg.num_attention_heads * cfg.q_head_dim()) * cfg.hidden_size,
            };
            let kv_a_proj_with_mqa = cfg.hidden_size * (cfg.kv_lora_rank + cfg.qk_rope_head_dim)
                / weight_pack_factor
                + bias_if!(cfg.attention_bias, cfg.kv_lora_rank + cfg.qk_rope_head_dim);
            let kv_a_layernorm = cfg.kv_lora_rank;
            let kv_b_proj = cfg.kv_lora_rank
                * cfg.num_attention_heads
                * (cfg.q_head_dim() - cfg.qk_rope_head_dim + cfg.v_head_dim)
                / weight_pack_factor;
            let o_proj = cfg.num_attention_heads * cfg.v_head_dim * cfg.hidden_size
                / weight_pack_factor
                + bias_if!(cfg.attention_bias, cfg.hidden_size);

            let moe_block = {
                let mut sum = 0;
                if let Some(n_routed_experts) = cfg.n_routed_experts.filter(|_| {
                    layer_idx >= cfg.first_k_dense_replace && layer_idx % cfg.moe_layer_freq == 0
                }) {
                    let h_size = cfg.hidden_size;
                    let gate_proj =
                        h_size * cfg.moe_intermediate_size / weight_pack_factor * n_routed_experts;
                    let up_proj =
                        h_size * cfg.moe_intermediate_size / weight_pack_factor * n_routed_experts;
                    let down_proj =
                        cfg.moe_intermediate_size * h_size / weight_pack_factor * n_routed_experts;
                    let shared_experts = if let Some(n_shared_experts) = cfg.n_shared_experts {
                        let gate_proj = h_size * (cfg.intermediate_size * n_shared_experts)
                            / weight_pack_factor;
                        let up_proj = h_size * (cfg.intermediate_size * n_shared_experts)
                            / weight_pack_factor;
                        let down_proj = (cfg.intermediate_size * n_shared_experts) * h_size
                            / weight_pack_factor;
                        gate_proj + up_proj + down_proj
                    } else {
                        0
                    };
                    let gate_weight = n_routed_experts * cfg.hidden_size;
                    sum += gate_proj + up_proj + down_proj + shared_experts + gate_weight;
                } else {
                    let h_size = cfg.hidden_size;
                    let i_size = cfg.intermediate_size;
                    let gate_proj = h_size * i_size / weight_pack_factor;
                    let up_proj = h_size * i_size / weight_pack_factor;
                    let down_proj = i_size * h_size / weight_pack_factor;
                    sum += gate_proj + up_proj + down_proj;
                }
                sum
            };

            per_layer_elems.push(
                input_layernorm
                    + post_attention_layernorm
                    + q_proj
                    + kv_a_layernorm
                    + kv_a_proj_with_mqa
                    + kv_b_proj
                    + o_proj
                    + moe_block,
            );
        }

        Ok(per_layer_elems
            .into_iter()
            .map(|x| x * dtype.size_in_bytes())
            .collect())
    }
    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: crate::models::deepseek3::DeepSeekV3Config = serde_json::from_str(config)?;
        Ok(cfg.num_hidden_layers)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: crate::models::deepseek3::DeepSeekV3Config = serde_json::from_str(config)?;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_attention_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: None,
            k_head_dim: cfg.qk_rope_head_dim + cfg.qk_nope_head_dim,
            v_head_dim: cfg.v_head_dim,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }
}
