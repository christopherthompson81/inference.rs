use super::*;

/// `NormalLoader` for a GLM 4 MoE model (GLM-4.5).
pub struct GLM4MoeLoader;

impl NormalModelLoader for GLM4MoeLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg = crate::glm4_moe::Glm4MoeConfig::from_json(config)?;
        Ok(Box::new(crate::glm4_moe::Glm4Moe::new(
            &cfg.family(),
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
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg = crate::glm4_moe::Glm4MoeConfig::from_json(config)?;
        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for GLM4MoeLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"^model\.embed_tokens\.weight$",
            r"^lm_head\.(weight|bias)$",
        ])
    }
    fn isq_layer_regexes(&self, config: &str) -> Result<Vec<Regex>> {
        let mut data = isq_regexes(&[
            r"lm_head\.(weight|bias)$",
            // Attention (standard GQA)
            r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$",
            r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$",
            r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$",
            r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$",
            r"layers\.(\d+)\.mlp\.experts\.(gate_proj|up_proj|down_proj)\.weight$",
        ])?;
        let cfg = crate::glm4_moe::Glm4MoeConfig::from_json(config)?;
        for layer_idx in 0..cfg.num_hidden_layers {
            if layer_idx >= cfg.first_k_dense_replace {
                // MoE layer
                for i in 0..cfg.n_routed_experts {
                    data.extend(isq_regexes(&[
                        format!(
                            r"layers\.{layer_idx}\.mlp\.experts\.{i}\.gate_proj\.(weight|bias)$"
                        ),
                        format!(r"layers\.{layer_idx}\.mlp\.experts\.{i}\.up_proj\.(weight|bias)$"),
                        format!(
                            r"layers\.{layer_idx}\.mlp\.experts\.{i}\.down_proj\.(weight|bias)$"
                        ),
                    ])?);
                }
                if cfg.n_shared_experts > 0 {
                    data.extend(isq_regexes(&[
                        format!(
                            r"layers\.{layer_idx}\.mlp\.shared_experts\.gate_proj\.(weight|bias)$"
                        ),
                        format!(
                            r"layers\.{layer_idx}\.mlp\.shared_experts\.up_proj\.(weight|bias)$"
                        ),
                        format!(
                            r"layers\.{layer_idx}\.mlp\.shared_experts\.down_proj\.(weight|bias)$"
                        ),
                    ])?);
                }
            } else {
                // Dense MLP layer
                data.extend(isq_regexes(&[
                    format!(r"layers\.{layer_idx}\.mlp\.gate_proj\.(weight|bias)$"),
                    format!(r"layers\.{layer_idx}\.mlp\.up_proj\.(weight|bias)$"),
                    format!(r"layers\.{layer_idx}\.mlp\.down_proj\.(weight|bias)$"),
                ])?);
            };
        }
        Ok(data)
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
    fn isq_layer_regexes_moqe(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|bias)$",
            r"layers\.(\d+)\.mlp\.experts\.(gate_proj|up_proj|down_proj)\.weight$",
        ])
    }
    fn immediate_isq_predicates_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes_moqe(config)
    }
}

impl DeviceMappedModelLoader for GLM4MoeLoader {
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg = crate::glm4_moe::Glm4MoeConfig::from_json(config)?;
        standard_non_mapped_size_in_bytes(
            LanguageModelEnds {
                hidden_size: cfg.hidden_size,
                vocab_size: cfg.vocab_size,
                tie_word_embeddings: cfg.tie_word_embeddings,
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
        let cfg = crate::glm4_moe::Glm4MoeConfig::from_json(config)?;
        let mut per_layer_elems = Vec::new();

        let head_dim = cfg.head_dim();
        for layer_idx in 0..cfg.num_hidden_layers {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            // Standard GQA attention
            let q_proj = cfg.hidden_size * cfg.num_attention_heads * head_dim / weight_pack_factor
                + bias_if!(cfg.attention_bias, cfg.num_attention_heads * head_dim);
            let k_proj = cfg.hidden_size * cfg.num_key_value_heads * head_dim / weight_pack_factor
                + bias_if!(cfg.attention_bias, cfg.num_key_value_heads * head_dim);
            let v_proj = cfg.hidden_size * cfg.num_key_value_heads * head_dim / weight_pack_factor
                + bias_if!(cfg.attention_bias, cfg.num_key_value_heads * head_dim);
            let o_proj = cfg.num_attention_heads * head_dim * cfg.hidden_size / weight_pack_factor;

            // QK norm if enabled
            let qk_norm = if cfg.use_qk_norm {
                head_dim * 2 // q_norm + k_norm
            } else {
                0
            };

            let moe_block = {
                let mut sum = 0;
                if layer_idx >= cfg.first_k_dense_replace {
                    // MoE layer
                    let h_size = cfg.hidden_size;
                    let gate_proj = h_size * cfg.moe_intermediate_size / weight_pack_factor
                        * cfg.n_routed_experts;
                    let up_proj = h_size * cfg.moe_intermediate_size / weight_pack_factor
                        * cfg.n_routed_experts;
                    let down_proj = cfg.moe_intermediate_size * h_size / weight_pack_factor
                        * cfg.n_routed_experts;
                    let shared_experts = if cfg.n_shared_experts > 0 {
                        let gate_proj = h_size * cfg.moe_intermediate_size / weight_pack_factor;
                        let up_proj = h_size * cfg.moe_intermediate_size / weight_pack_factor;
                        let down_proj = cfg.moe_intermediate_size * h_size / weight_pack_factor;
                        gate_proj + up_proj + down_proj
                    } else {
                        0
                    };
                    let gate_weight = cfg.n_routed_experts * cfg.hidden_size;
                    let e_score_correction_bias = cfg.n_routed_experts;
                    sum += gate_proj
                        + up_proj
                        + down_proj
                        + shared_experts
                        + gate_weight
                        + e_score_correction_bias;
                } else {
                    // Dense MLP layer
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
                    + k_proj
                    + v_proj
                    + o_proj
                    + qk_norm
                    + moe_block,
            );
        }

        Ok(per_layer_elems
            .into_iter()
            .map(|x| x * dtype.size_in_bytes())
            .collect())
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = crate::glm4_moe::Glm4MoeConfig::from_json(config)?;

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
