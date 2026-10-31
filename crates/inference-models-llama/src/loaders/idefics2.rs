use super::*;

// sub-images per image with do_image_splitting
const IMAGE_SPLITS: usize = 5;

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

// Both Idefics versions put one SigLIP sequence per sub-image into the text input
pub(super) struct IdeficsActShape {
    pub image_size: usize,
    pub patch_size: usize,
    pub vision_heads: usize,
    pub text_heads: usize,
}

impl IdeficsActShape {
    fn image_seq_len(&self) -> usize {
        (self.image_size / self.patch_size).pow(2) + 1
    }

    pub(super) fn mapped_max_act_size_elems(&self, params: &AutoDeviceMapParams) -> Result<usize> {
        let AutoDeviceMapParams::Multimodal {
            max_seq_len,
            max_batch_size,
            max_image_shape: _,
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected multimodal AutoDeviceMapParams for this model!")
        };
        let max_seq_len =
            self.image_seq_len() * max_num_images + max_seq_len.min(&ATTENTION_CHUNK_SIZE);
        Ok(max_batch_size * self.text_heads * max_seq_len * max_seq_len)
    }

    pub(super) fn non_mapped_max_act_size_elems(
        &self,
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
        let img_seq_len = self.image_seq_len();
        Ok((max_batch_size * IMAGE_SPLITS * max_num_images)
            * self.vision_heads
            * img_seq_len
            * img_seq_len)
    }
}

fn act_shape(config: &str) -> Result<IdeficsActShape> {
    let cfg = Idefics2Config::from_json(config)?;
    Ok(IdeficsActShape {
        image_size: cfg.vision_config.image_size,
        patch_size: cfg.vision_config.patch_size,
        vision_heads: cfg.vision_config.num_attention_heads,
        text_heads: cfg.text_config.num_attention_heads,
    })
}
