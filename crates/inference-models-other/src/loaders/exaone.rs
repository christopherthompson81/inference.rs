use super::*;

/// `NormalLoader` for an EXAONE 3.0 or 3.5 model.
pub struct ExaoneLoader;

impl NormalModelLoader for ExaoneLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg = crate::exaone::Config::from_json(config)?;

        Ok(Box::new(crate::exaone::Model::new(
            &cfg.decoder_spec()?,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg = crate::exaone::Config::from_json(config)?;

        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for ExaoneLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[r"^transformer\.wte\.weight$", r"^lm_head\.(weight|bias)$"])
    }

    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"lm_head\.(weight|bias)$",
            r"h\.(\d+)\.attn\.attention\.(q_proj|k_proj|v_proj|out_proj)\.(weight|bias)$",
            r"h\.(\d+)\.mlp\.(c_fc_0|c_fc_1|c_proj)\.(weight|bias)$",
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for ExaoneLoader {
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg = crate::exaone::Config::from_json(config)?;
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
        let cfg = crate::exaone::Config::from_json(config)?;
        Ok(decoder_shape(&cfg).layer_sizes_in_bytes(cfg.num_layers, dtype, weight_pack_factor))
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = crate::exaone::Config::from_json(config)?;
        Ok(Box::new(decoder_shape(&cfg).model_config(
            cfg.num_layers,
            cfg.max_position_embeddings,
            None,
        )))
    }
}

fn decoder_shape(cfg: &crate::exaone::Config) -> DecoderLayerShape {
    DecoderLayerShape {
        hidden_size: cfg.hidden_size,
        num_attention_heads: cfg.num_attention_heads,
        num_key_value_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim(),
        qkv_bias: false,
        o_bias: false,
        qk_norm: false,
        norms: 2,
        mlp: MlpShape::Gated {
            intermediate_size: cfg.intermediate_size,
        },
    }
}
