use super::*;

/// `MultimodalLoader` for an Idefics 2 Vision model.
pub struct Idefics2Loader;

pub struct Idefics2Prefixer;

impl MultimodalPromptPrefixer for Idefics2Prefixer {
    fn prefix_image(&self, _image_indexes: Vec<usize>, prompt: &str) -> String {
        // Chat template does it
        prompt.to_string()
    }
}

impl MultimodalModelLoader for Idefics2Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn MultimodalModel + Send + Sync>> {
        let cfg = crate::idefics2::Config::from_json(config)?;
        Ok(Box::new(Idefics2::new(
            &cfg,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg = crate::idefics2::Config::from_json(config)?;
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
        Arc::new(Idefics2Prefixer)
    }
    fn modalities(&self, _config: &str) -> Result<Modalities> {
        Ok(Modalities {
            input: vec![SupportedModality::Text, SupportedModality::Vision],
            output: vec![SupportedModality::Text],
        })
    }
}

impl IsqModelLoader for Idefics2Loader {
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
            r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$",
            r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$",
            r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$",
            r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$",
            // MLP
            r"layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$",
            r"layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$",
            r"layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$",
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
        ])
    }
}

impl DeviceMappedModelLoader for Idefics2Loader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Multimodal {
            max_seq_len,
            max_batch_size,
            max_image_shape: _,
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected multimodal AutoDeviceMapParams for this model!")
        };

        let cfg = Idefics2Config::from_json(config)?;

        let num_patches = (cfg.vision_config.image_size / cfg.vision_config.patch_size).pow(2);
        let img_seq_len = (num_patches + 1) * max_num_images;

        let max_text_attn = {
            // This model injects the vision information directly into the input embeddings
            let max_seq_len = img_seq_len + max_seq_len.min(&ATTENTION_CHUNK_SIZE);
            max_batch_size * cfg.text_config.num_attention_heads * max_seq_len * max_seq_len
        };

        Ok(max_text_attn)
    }
    fn non_mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Multimodal {
            max_seq_len: _,
            max_batch_size,
            max_image_shape: _,
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected multimodal AutoDeviceMapParams for this model!")
        };

        let cfg = Idefics2Config::from_json(config)?;

        let num_patches = (cfg.vision_config.image_size / cfg.vision_config.patch_size).pow(2);
        let img_seq_len = num_patches + 1;

        let max_vision_attn = {
            // do_image_splitting = true
            let images_factor = 5;

            (max_batch_size * images_factor * max_num_images)
                * cfg.vision_config.num_attention_heads
                * img_seq_len
                * img_seq_len
        };

        Ok(max_vision_attn)
    }
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg = Idefics2Config::from_json(config)?;
        let text_elems = {
            let tie_word_embeddings = cfg.tie_word_embeddings;
            let cfg = &cfg.text_config;
            let (embed_tokens_pack_factor, lm_head_pack_factor) =
                super::language_model_pack_factors(
                    _quantization,
                    "model.text_model.embed_tokens.weight",
                    "lm_head.weight",
                    tie_word_embeddings,
                    dtype,
                    weight_pack_factor,
                )?;
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / embed_tokens_pack_factor;
            let lm_head = if !tie_word_embeddings {
                cfg.hidden_size * cfg.vocab_size / lm_head_pack_factor
            } else {
                0
            };
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };

        let connector_elems = {
            let tcfg = &cfg.text_config;
            let vcfg = &cfg.vision_config;
            let gate_proj = vcfg.hidden_size * tcfg.intermediate_size;
            let up_proj = vcfg.hidden_size * tcfg.intermediate_size;
            let down_proj = tcfg.intermediate_size * tcfg.hidden_size;

            let perceiver_elems = {
                let tcfg = &cfg.text_config;
                let pcfg = &cfg.perceiver_config;

                let n_latents = pcfg.resampler_n_latents;
                let hidden_size = tcfg.hidden_size;
                let depth = pcfg.resampler_depth;

                let norm = tcfg.hidden_size;
                let latents = n_latents * hidden_size;

                let layer_elems = {
                    let input_latents_norm = hidden_size;
                    let input_context_norm = hidden_size;
                    let post_attn_norm = hidden_size;

                    let num_heads = pcfg.resampler_n_heads;
                    let head_dim = pcfg.resampler_head_dim;
                    let num_key_value_heads = pcfg.num_key_value_heads;

                    let q_proj = hidden_size * num_heads * head_dim;
                    let k_proj = hidden_size * num_key_value_heads * head_dim;
                    let v_proj = hidden_size * num_key_value_heads * head_dim;
                    let o_proj = num_heads * head_dim * hidden_size;

                    let gate_proj = hidden_size * hidden_size * 4;
                    let up_proj = hidden_size * hidden_size * 4;
                    let down_proj = hidden_size * 4 * hidden_size;

                    input_latents_norm
                        + input_context_norm
                        + post_attn_norm
                        + q_proj
                        + k_proj
                        + v_proj
                        + o_proj
                        + gate_proj
                        + up_proj
                        + down_proj
                };

                norm + latents + layer_elems * depth
            };

            gate_proj + up_proj + down_proj + perceiver_elems
        };

        let vision_transformer = {
            let cfg = &cfg.vision_config;

            let post_layernorm = cfg.hidden_size;

            let conv_config = Conv2dConfig {
                stride: cfg.patch_size,
                ..Default::default()
            };
            let patch_embedding = cfg.num_channels * cfg.hidden_size / conv_config.groups
                * cfg.patch_size
                * cfg.patch_size;

            let num_patches_per_side = cfg.image_size / cfg.patch_size;
            let num_patches = num_patches_per_side.pow(2);
            let position_embedding = num_patches * cfg.hidden_size;

            let layer_elems = {
                let layer_norm_1 = cfg.hidden_size + bias_if!(true, cfg.hidden_size);
                let layer_norm_2 = cfg.hidden_size + bias_if!(true, cfg.hidden_size);

                let fc1 = cfg.hidden_size * cfg.intermediate_size + cfg.intermediate_size;
                let fc2 = cfg.intermediate_size * cfg.hidden_size + cfg.hidden_size;

                let q_proj = cfg.hidden_size * cfg.hidden_size + cfg.hidden_size;
                let k_proj = cfg.hidden_size * cfg.hidden_size + cfg.hidden_size;
                let v_proj = cfg.hidden_size * cfg.hidden_size + cfg.hidden_size;
                let o_proj = cfg.hidden_size * cfg.hidden_size + cfg.hidden_size;

                layer_norm_1 + layer_norm_2 + fc1 + fc2 + q_proj + k_proj + v_proj + o_proj
            };

            post_layernorm
                + patch_embedding
                + position_embedding
                + layer_elems * cfg.num_hidden_layers
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
        let cfg = crate::mistral::Config::from(Idefics2Config::from_json(config)?.text_config);
        Ok(decoder_shape(&cfg).layer_sizes_in_bytes(
            cfg.num_hidden_layers,
            dtype,
            weight_pack_factor,
        ))
    }
    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg = Idefics2Config::from_json(config)?;
        Ok(cfg.text_config.num_hidden_layers)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = crate::mistral::Config::from(Idefics2Config::from_json(config)?.text_config);
        Ok(Box::new(decoder_shape(&cfg).model_config(
            cfg.num_hidden_layers,
            cfg.max_position_embeddings,
            cfg.decoder_spec()?.sliding_window(),
        )))
    }

    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Vision])
    }
}

fn decoder_shape(cfg: &crate::mistral::Config) -> DecoderLayerShape {
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
