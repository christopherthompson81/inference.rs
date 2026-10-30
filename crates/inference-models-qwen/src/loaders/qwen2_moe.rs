use super::*;

/// `NormalLoader` for a Qwen2-MoE model.
pub struct Qwen2MoeLoader;

impl NormalModelLoader for Qwen2MoeLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg = crate::qwen2_moe::Config::from_json(config)?;

        Ok(Box::new(crate::qwen2_moe::Model::new(
            &cfg.decoder_spec()?,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg = crate::qwen2_moe::Config::from_json(config)?;

        Ok(Box::new(cfg))
    }
}

const MOE_EXPERT_REGEXES: [&str; 4] = [
    r"layers\.(\d+)\.mlp\.experts\.(\d+)\.gate_proj\.(weight|bias)$",
    r"layers\.(\d+)\.mlp\.experts\.(\d+)\.up_proj\.(weight|bias)$",
    r"layers\.(\d+)\.mlp\.experts\.(\d+)\.down_proj\.(weight|bias)$",
    r"layers\.(\d+)\.mlp\.experts\.(gate_up_proj|gate_proj|up_proj|down_proj)(\.weight)?$",
];

impl IsqModelLoader for Qwen2MoeLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"^model\.embed_tokens\.weight$",
            r"^lm_head\.(weight|bias)$",
        ])
    }

    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        let mut regexes = vec![
            r"lm_head\.(weight|bias)$",
            // Attention
            r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$",
            r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$",
            r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$",
            r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$",
            // Dense MLP and the shared expert
            r"layers\.(\d+)\.mlp\.(shared_expert\.)?gate_proj\.(weight|bias)$",
            r"layers\.(\d+)\.mlp\.(shared_expert\.)?up_proj\.(weight|bias)$",
            r"layers\.(\d+)\.mlp\.(shared_expert\.)?down_proj\.(weight|bias)$",
        ];
        regexes.extend(MOE_EXPERT_REGEXES);
        isq_regexes(&regexes)
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
    fn immediate_isq_predicates_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes_moqe(config)
    }
    fn isq_layer_regexes_moqe(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&MOE_EXPERT_REGEXES)
    }
}

impl DeviceMappedModelLoader for Qwen2MoeLoader {
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg = crate::qwen2_moe::Config::from_json(config)?;
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
        let cfg = crate::qwen2_moe::Config::from_json(config)?;
        let h = cfg.hidden_size;
        let gated = |intermediate_size: usize| 3 * h * intermediate_size / weight_pack_factor;
        Ok((0..cfg.num_hidden_layers)
            .map(|layer_idx| {
                let mlp = if cfg.is_moe_layer(layer_idx) {
                    // the router and the shared expert's gate stay unquantized
                    cfg.num_experts * (gated(cfg.moe_intermediate_size) + h)
                        + gated(cfg.shared_expert_intermediate_size)
                        + h
                } else {
                    gated(cfg.intermediate_size)
                };
                (decoder_shape(&cfg).elems(weight_pack_factor) + mlp) * dtype.size_in_bytes()
            })
            .collect())
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = crate::qwen2_moe::Config::from_json(config)?;
        Ok(Box::new(decoder_shape(&cfg).model_config(
            cfg.num_hidden_layers,
            cfg.max_position_embeddings,
            cfg.decoder_spec()?.sliding_window(),
        )))
    }
}

// the feed-forward is sized per layer, dense or MoE
fn decoder_shape(cfg: &crate::qwen2_moe::Config) -> DecoderLayerShape {
    DecoderLayerShape {
        hidden_size: cfg.hidden_size,
        num_attention_heads: cfg.num_attention_heads,
        num_key_value_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim(),
        qkv_bias: cfg.qkv_bias,
        o_bias: false,
        qk_norm: false,
        norms: 2,
        mlp: MlpShape::Gated {
            intermediate_size: 0,
        },
    }
}
