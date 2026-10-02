use super::*;

/// `NormalLoader` for a Phi 2 model.
pub struct Phi2Loader;

impl NormalModelLoader for Phi2Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg = crate::phi2::Config::from_json(config)?;

        Ok(Box::new(crate::phi2::Model::new(
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
        let cfg = crate::phi2::Config::from_json(config)?;

        Ok(Box::new(crate::xlora::phi2::Model::new(
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
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg = crate::phi2::Config::from_json(config)?;

        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for Phi2Loader {
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
            r"layers\.(\d+)\.self_attn\.dense\.(weight|bias)$",
            // MLP
            r"layers\.(\d+)\.mlp\.fc1\.(weight|bias)$",
            r"layers\.(\d+)\.mlp\.fc2\.(weight|bias)$",
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for Phi2Loader {
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg = crate::phi2::Config::from_json(config)?;
        let ends = standard_non_mapped_size_in_bytes(
            LanguageModelEnds {
                hidden_size: cfg.hidden_size,
                vocab_size: cfg.vocab_size,
                tie_word_embeddings: cfg.tie_word_embeddings,
            },
            quantization,
            dtype,
            weight_pack_factor,
        )?;
        // the lm_head bias and the affine final LayerNorm's bias
        Ok(ends + (cfg.vocab_size + cfg.hidden_size) * dtype.size_in_bytes())
    }
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg = crate::phi2::Config::from_json(config)?;
        let shape = decoder_shape(&cfg);
        // q/k layernorms carry a bias the shape's qk norm does not count
        let qk_norm_bias = bias_if!(cfg.qk_layernorm, 2 * shape.head_dim) * dtype.size_in_bytes();
        Ok(shape
            .layer_sizes_in_bytes(cfg.num_hidden_layers, dtype, weight_pack_factor)
            .into_iter()
            .map(|size| size + qk_norm_bias)
            .collect())
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = crate::phi2::Config::from_json(config)?;
        Ok(Box::new(decoder_shape(&cfg).model_config(
            cfg.num_hidden_layers,
            cfg.max_position_embeddings,
            None,
        )))
    }
}

fn decoder_shape(cfg: &crate::phi2::Config) -> DecoderLayerShape {
    DecoderLayerShape {
        hidden_size: cfg.hidden_size,
        num_attention_heads: cfg.num_attention_heads,
        num_key_value_heads: cfg.num_key_value_heads(),
        head_dim: cfg.head_dim(),
        qkv_bias: true,
        o_bias: true,
        qk_norm: cfg.qk_layernorm,
        // one LayerNorm shared by the parallel attention and MLP, weight and bias
        norms: 2,
        mlp: MlpShape::Plain {
            intermediate_size: cfg.intermediate_size,
            bias: true,
        },
    }
}
