use super::*;

/// `NormalLoader` for a Starcoder2 model.
pub struct Starcoder2Loader;

impl NormalModelLoader for Starcoder2Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg = crate::starcoder2::Config::from_json(config)?;

        Ok(Box::new(crate::starcoder2::Model::new(
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
        let cfg = crate::starcoder2::Config::from_json(config)?;

        Ok(Box::new(crate::xlora::starcoder2::Model::new(
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
        let cfg = crate::starcoder2::Config::from_json(config)?;

        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for Starcoder2Loader {
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
            r"layers\.(\d+)\.mlp\.c_fc\.(weight|bias)$",
            r"layers\.(\d+)\.mlp\.c_proj\.(weight|bias)$",
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for Starcoder2Loader {
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg = crate::starcoder2::Config::from_json(config)?;

        let elems = {
            let embed_tokens_pack_factor = super::tied_promoted_tensor_pack_factor(
                _quantization,
                "model.embed_tokens.weight",
                "lm_head.weight",
                dtype,
                weight_pack_factor,
            )?;
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / embed_tokens_pack_factor;
            let lm_head = 0;
            let norm = cfg.hidden_size + cfg.hidden_size;
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
        let cfg = crate::starcoder2::Config::from_json(config)?;
        Ok(decoder_shape(&cfg).layer_sizes_in_bytes(
            cfg.num_hidden_layers,
            dtype,
            weight_pack_factor,
        ))
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = crate::starcoder2::Config::from_json(config)?;
        Ok(Box::new(decoder_shape(&cfg).model_config(
            cfg.num_hidden_layers,
            cfg.max_position_embeddings,
            cfg.sliding_window,
        )))
    }
}

fn decoder_shape(cfg: &crate::starcoder2::Config) -> DecoderLayerShape {
    DecoderLayerShape {
        hidden_size: cfg.hidden_size,
        num_attention_heads: cfg.num_attention_heads,
        num_key_value_heads: cfg.num_key_value_heads,
        head_dim: cfg.hidden_size / cfg.num_attention_heads,
        qkv_bias: cfg.use_bias,
        o_bias: cfg.use_bias,
        qk_norm: false,
        // two LayerNorms, each a weight and a bias of hidden size
        norms: 4,
        mlp: MlpShape::Plain {
            intermediate_size: cfg.intermediate_size,
            bias: cfg.use_bias,
        },
    }
}
