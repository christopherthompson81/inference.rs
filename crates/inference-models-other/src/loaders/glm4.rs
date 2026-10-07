use super::*;

/// `NormalLoader` for a GLM 4 model.
pub struct GLM4Loader;

impl NormalModelLoader for GLM4Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg = crate::glm4::Config::from_json(config)?;

        Ok(Box::new(crate::glm4::Model::new(
            &cfg.decoder_spec(),
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg = crate::glm4::Config::from_json(config)?;

        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for GLM4Loader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"^model\.embed_tokens\.weight$",
            r"^lm_head\.(weight|bias)$",
        ])
    }

    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"lm_head\.(weight|bias)$",
            // Attention
            r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$",
            r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$",
            r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$",
            r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$",
            // MLP
            r"layers\.(\d+)\.mlp\.gate_up_proj\.(weight|bias)$",
            r"layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$",
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for GLM4Loader {
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg = crate::glm4::Config::from_json(config)?;
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
        let cfg = crate::glm4::Config::from_json(config)?;
        Ok(decoder_shape(&cfg).layer_sizes_in_bytes(
            cfg.num_hidden_layers,
            dtype,
            weight_pack_factor,
        ))
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = crate::glm4::Config::from_json(config)?;
        Ok(Box::new(decoder_shape(&cfg).model_config(
            cfg.num_hidden_layers,
            cfg.max_position_embeddings,
            cfg.sliding_window,
        )))
    }
}

fn decoder_shape(cfg: &crate::glm4::Config) -> DecoderLayerShape {
    DecoderLayerShape {
        hidden_size: cfg.hidden_size,
        num_attention_heads: cfg.num_attention_heads,
        num_key_value_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim(),
        qkv_bias: cfg.attention_bias.unwrap_or(false),
        o_bias: false,
        qk_norm: false,
        // input, post_self_attn, post_attention and post_mlp
        norms: 4,
        mlp: MlpShape::Gated {
            intermediate_size: cfg.intermediate_size,
        },
    }
}
