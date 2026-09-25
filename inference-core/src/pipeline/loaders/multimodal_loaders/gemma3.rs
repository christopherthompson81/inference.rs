use super::*;

/// [`MultimodalLoader`] for an Gemma 3 model.
///
/// [`MultimodalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.MultimodalLoader.html
pub struct Gemma3Loader;

impl MultimodalModelLoader for Gemma3Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn MultimodalModel + Send + Sync>> {
        let cfg: Gemma3Config = serde_json::from_str(config)?;
        Ok(Box::new(Gemma3Model::new(
            &cfg,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn is_gptx(&self, _config: &str) -> bool {
        true
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let config: Gemma3Config = serde_json::from_str(config)?;
        Ok(Box::new(config))
    }
    fn get_processor(
        &self,
        config: &str,
        processor_config: Option<ProcessorConfig>,
        _preprocessor_config: PreProcessorConfig,
        _max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        let config: Gemma3Config = serde_json::from_str(config).unwrap();
        // Handle the Gemma 3 1b case here
        Arc::new(Gemma3Processor::new(
            processor_config.unwrap_or_default(),
            matches!(config, Gemma3Config::WithVision { .. }),
        ))
    }
    fn supports_paged_attention(&self, _config: &str) -> bool {
        true
    }
    fn supports_encoder_cache(&self, config: &str) -> bool {
        serde_json::from_str::<serde_json::Value>(config).is_ok_and(|config| {
            config
                .get("vision_config")
                .is_some_and(|vision_config| !vision_config.is_null())
        })
    }
    fn supports_prefix_cacher(&self, _config: &str) -> bool {
        true
    }
    fn prefixer(&self, _config: &str) -> Arc<dyn MultimodalPromptPrefixer> {
        Arc::new(Gemma3Prefixer)
    }
    fn auto_device_map_params(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<AutoDeviceMapParams> {
        Ok(match serde_json::from_str::<Gemma3Config>(config)? {
            Gemma3Config::Text(_) => AutoDeviceMapParams::Text {
                max_seq_len: params.max_seq_len(),
                max_batch_size: params.max_batch_size(),
            },
            Gemma3Config::WithVision { .. } => params.maybe_promote_to_multimodal(),
        })
    }
    fn modalities(&self, config: &str) -> Result<Modalities> {
        let config: Gemma3Config = serde_json::from_str(config)?;
        Ok(Modalities {
            input: match config {
                Gemma3Config::Text(_) => vec![SupportedModality::Text],
                Gemma3Config::WithVision { .. } => {
                    vec![SupportedModality::Text, SupportedModality::Vision]
                }
            },
            output: vec![SupportedModality::Text],
        })
    }
}

impl IsqModelLoader for Gemma3Loader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"^(model|language_model\.model)\.embed_tokens\.weight$")?,
            Regex::new(r"^(lm_head|language_model\.lm_head)\.(weight|bias)$")?,
        ])
    }

    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // MLP
            Regex::new(r"layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
        ])
    }
    fn immediate_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"^(?:language_model\.)?lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(
                r"^(?:language_model\.)?model\.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"^(?:language_model\.)?model\.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"^(?:language_model\.)?model\.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"^(?:language_model\.)?model\.layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$",
            )?,
            // MLP
            Regex::new(
                r"^(?:language_model\.)?model\.layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"^(?:language_model\.)?model\.layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"^(?:language_model\.)?model\.layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$",
            )?,
        ])
    }
}

impl DeviceMappedModelLoader for Gemma3Loader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let cfg: Gemma3Config = serde_json::from_str(config)?;

        match cfg {
            Gemma3Config::Text(text_config) => {
                let (max_seq_len, max_batch_size) = match params {
                    AutoDeviceMapParams::Text {
                        max_seq_len,
                        max_batch_size,
                    }
                    | AutoDeviceMapParams::Multimodal {
                        max_seq_len,
                        max_batch_size,
                        ..
                    } => (*max_seq_len, *max_batch_size),
                };
                Ok(max_batch_size
                    * text_config.num_attention_heads
                    * max_seq_len.min(ATTENTION_CHUNK_SIZE).pow(2))
            }
            Gemma3Config::WithVision {
                text_config,
                vision_config,
                ..
            } => {
                let AutoDeviceMapParams::Multimodal {
                    max_seq_len,
                    max_batch_size,
                    max_num_images,
                    ..
                } = params
                else {
                    anyhow::bail!("Expected multimodal AutoDeviceMapParams for this model!")
                };
                let num_patches = (vision_config.image_size / vision_config.patch_size).pow(2);
                let img_seq_len = (num_patches + 1) * max_num_images;

                let max_text_attn = {
                    // This model injects the vision information directly into the input embeddings
                    let max_seq_len = img_seq_len + max_seq_len.min(&ATTENTION_CHUNK_SIZE);
                    max_batch_size * text_config.num_attention_heads * max_seq_len * max_seq_len
                };
                Ok(max_text_attn)
            }
        }
    }
    fn non_mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let cfg: Gemma3Config = serde_json::from_str(config)?;
        let Gemma3Config::WithVision { vision_config, .. } = cfg else {
            return Ok(0);
        };
        let AutoDeviceMapParams::Multimodal {
            max_seq_len: _,
            max_batch_size,
            max_image_shape: _,
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected multimodal AutoDeviceMapParams for this model!")
        };
        let num_patches = (vision_config.image_size / vision_config.patch_size).pow(2);
        let img_seq_len = num_patches + 1;
        Ok((max_batch_size * max_num_images)
            * vision_config.num_attention_heads
            * img_seq_len
            * img_seq_len)
    }
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: Gemma3Config = serde_json::from_str(config)?;

        let text_elems = {
            let cfg = match &cfg {
                Gemma3Config::Text(cfg) => cfg,
                Gemma3Config::WithVision { text_config, .. } => text_config,
            };
            let (embed_tokens_pack_factor, lm_head_pack_factor) =
                super::language_model_pack_factors_with_aliases(
                    _quantization,
                    &[
                        "model.embed_tokens.weight",
                        "language_model.model.embed_tokens.weight",
                    ],
                    &["lm_head.weight", "language_model.lm_head.weight"],
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

        let vision_transformer = if let Gemma3Config::WithVision {
            vision_config: cfg, ..
        } = &cfg
        {
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
        } else {
            0
        };

        let elems = text_elems + vision_transformer;

        Ok(elems * dtype.size_in_bytes())
    }
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: Gemma3Config = serde_json::from_str(config)?;

        let txt_cfg = match &cfg {
            Gemma3Config::Text(cfg) => cfg,
            Gemma3Config::WithVision { text_config, .. } => text_config,
        };
        let per_layer_elems = {
            let cfg = txt_cfg;

            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let size_in = cfg.hidden_size;
            let size_q = cfg.head_dim * cfg.num_attention_heads;
            let size_kv = cfg.head_dim * cfg.num_key_value_heads;
            let q_proj =
                size_in * size_q / weight_pack_factor + bias_if!(cfg.attention_bias, size_q);
            let k_proj =
                size_in * size_kv / weight_pack_factor + bias_if!(cfg.attention_bias, size_kv);
            let v_proj =
                size_in * size_kv / weight_pack_factor + bias_if!(cfg.attention_bias, size_kv);
            let o_proj =
                size_q * size_in / weight_pack_factor + bias_if!(cfg.attention_bias, size_in);

            let h_size = cfg.hidden_size;
            let i_size = cfg.intermediate_size;
            let gate_proj = h_size * i_size / weight_pack_factor;
            let up_proj = h_size * i_size / weight_pack_factor;
            let down_proj = i_size * h_size / weight_pack_factor;

            input_layernorm
                + post_attention_layernorm
                + q_proj
                + k_proj
                + v_proj
                + o_proj
                + gate_proj
                + up_proj
                + down_proj
        };
        Ok(vec![
            per_layer_elems * dtype.size_in_bytes();
            txt_cfg.num_hidden_layers
        ])
    }
    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: Gemma3Config = serde_json::from_str(config)?;

        let txt_cfg = match &cfg {
            Gemma3Config::Text(cfg) => cfg,
            Gemma3Config::WithVision { text_config, .. } => text_config,
        };

        Ok(txt_cfg.num_hidden_layers)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: Gemma3Config = serde_json::from_str(config)?;

        let cfg = match &cfg {
            Gemma3Config::Text(cfg) => cfg,
            Gemma3Config::WithVision { text_config, .. } => text_config,
        };

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: None, // None to be more forgiving, some do not
            k_head_dim: cfg.head_dim,
            v_head_dim: cfg.head_dim,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }

    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Vision])
    }
    fn non_mapped_sub_models_for_config(
        &self,
        config: &str,
    ) -> Result<Option<Vec<NonMappedSubModel>>> {
        let config: Gemma3Config = serde_json::from_str(config)?;
        Ok(match config {
            Gemma3Config::Text(_) => None,
            Gemma3Config::WithVision { .. } => self.non_mapped_sub_models(),
        })
    }
}

pub struct Gemma3Prefixer;

impl MultimodalPromptPrefixer for Gemma3Prefixer {
    fn prefix_image(&self, _image_indexes: Vec<usize>, prompt: &str) -> String {
        prompt.to_string()
    }
}
