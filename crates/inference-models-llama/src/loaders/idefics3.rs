use super::*;

/// `MultimodalLoader` for an Idefics 3 Vision model.
pub struct Idefics3Loader;

pub struct Idefics3Prefixer;

impl MultimodalPromptPrefixer for Idefics3Prefixer {
    fn prefix_image(&self, _image_indexes: Vec<usize>, prompt: &str) -> String {
        // Chat template does it
        prompt.to_string()
    }
}

impl MultimodalModelLoader for Idefics3Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn MultimodalModel + Send + Sync>> {
        let cfg = crate::idefics3::Idefics3Config::from_json(config)?;
        Ok(Box::new(Idefics3Model::new(
            &cfg,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg = crate::idefics3::Idefics3Config::from_json(config)?;
        Ok(Box::new(cfg))
    }
    fn supports_paged_attention(&self, _config: &str) -> bool {
        true
    }
    fn supports_encoder_cache(&self, _config: &str) -> bool {
        true
    }
    fn supports_prefix_cacher(&self, _config: &str) -> bool {
        true
    }
    fn prefixer(&self, _config: &str) -> Arc<dyn MultimodalPromptPrefixer> {
        Arc::new(Idefics3Prefixer)
    }
    fn modalities(&self, _config: &str) -> Result<Modalities> {
        Ok(Modalities {
            input: vec![SupportedModality::Text, SupportedModality::Vision],
            output: vec![SupportedModality::Text],
        })
    }
}

impl IsqModelLoader for Idefics3Loader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"^model\.text_model\.embed_tokens\.weight$",
            r"^lm_head\.(weight|bias)$",
        ])
    }

    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"lm_head\.(weight|bias)$",
            // Attention
            r"model.text_model.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$",
            r"model.text_model.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$",
            r"model.text_model.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$",
            r"model.text_model.layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$",
            // MLP
            r"model.text_model.layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$",
            r"model.text_model.layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$",
            r"model.text_model.layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$",
        ])
    }
    fn immediate_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"lm_head\.(weight|bias)$",
            // Attention
            r"model\.text_model\.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$",
            r"model\.text_model\.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$",
            r"model\.text_model\.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$",
            r"model\.text_model\.layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$",
            // MLP
            r"model\.text_model\.layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$",
            r"model\.text_model\.layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$",
            r"model\.text_model\.layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$",
            // // Attention (vision)
            // Regex::new(
            //     r"model\.vision_model\.encoder\.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$",
            // )?,
            // Regex::new(
            //     r"model\.vision_model\.encoder\.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$",
            // )?,
            // Regex::new(
            //     r"model\.vision_model\.encoder\.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$",
            // )?,
            // Regex::new(
            //     r"model\.vision_model\.encoder\.layers\.(\d+)\.self_attn\.out_proj\.(weight|bias)$",
            // )?,
            // MLP (vision)
            // Regex::new(r"model\.vision_model\.encoder\.layers\.(\d+)\.mlp\.fc1\.(weight|bias)$")?,
            // Regex::new(r"model\.vision_model\.encoder\.layers\.(\d+)\.mlp\.fc2\.(weight|bias)$")?,
        ])
    }
}

impl DeviceMappedModelLoader for Idefics3Loader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        act_shape(config)?.mapped_max_act_size_elems(params)
    }
    fn non_mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        act_shape(config)?.non_mapped_max_act_size_elems(params)
    }
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg = Idefics3Config::from_json(config)?;
        let text_elems = {
            let cfg = &cfg.text_config;
            let (embed_tokens_pack_factor, lm_head_pack_factor) =
                super::language_model_pack_factors(
                    _quantization,
                    "model.text_model.embed_tokens.weight",
                    "lm_head.weight",
                    cfg.tie_word_embeddings,
                    dtype,
                    weight_pack_factor,
                )?;
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / embed_tokens_pack_factor;
            let lm_head = if cfg.tie_word_embeddings {
                0
            } else {
                cfg.hidden_size * cfg.vocab_size / lm_head_pack_factor
            };
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };

        let connector_elems = {
            let in_dim = cfg.vision_config.hidden_size * cfg.scale_factor.pow(2);
            let out_dim = cfg.text_config.hidden_size;

            in_dim * out_dim
        };

        let vision_transformer = {
            let cfg = &cfg.vision_config;

            cfg.siglip().weight_elems()
        };

        let elems = text_elems + connector_elems + vision_transformer;

        Ok(elems * dtype.size_in_bytes())
    }
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg = Idefics3Config::from_json(config)?.text_config;
        Ok(decoder_shape(&cfg).layer_sizes_in_bytes(
            cfg.num_hidden_layers,
            dtype,
            weight_pack_factor,
        ))
    }
    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg = Idefics3Config::from_json(config)?;
        Ok(cfg.text_config.num_hidden_layers)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = Idefics3Config::from_json(config)?.text_config;
        Ok(Box::new(decoder_shape(&cfg).model_config(
            cfg.num_hidden_layers,
            cfg.max_position_embeddings,
            None,
        )))
    }

    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Vision])
    }
}

fn decoder_shape(cfg: &crate::llama::Config) -> DecoderLayerShape {
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

fn act_shape(config: &str) -> Result<super::idefics2::IdeficsActShape> {
    let cfg = Idefics3Config::from_json(config)?;
    Ok(super::idefics2::IdeficsActShape {
        image_size: cfg.vision_config.image_size,
        patch_size: cfg.vision_config.patch_size,
        vision_heads: cfg.vision_config.num_attention_heads,
        text_heads: cfg.text_config.num_attention_heads,
    })
}
