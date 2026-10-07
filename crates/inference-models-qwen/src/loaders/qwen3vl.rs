use super::*;

/// `MultimodalLoader` for an Qwen3VL model.
pub struct Qwen3VLLoader;

pub struct Qwen3VLPrefixer;

impl MultimodalPromptPrefixer for Qwen3VLPrefixer {
    // No-op: With MessagesAction::Keep, the chat template handles image tokens
    // when it sees {"type": "image"} entries in the content.
}

impl MultimodalModelLoader for Qwen3VLLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn MultimodalModel + Send + Sync>> {
        let cfg = parse_config(config)?;
        Ok(Box::new(Qwen3VLModel::new(
            &cfg,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let config = parse_config(config)?;
        Ok(Box::new(config))
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
        Arc::new(Qwen3VLPrefixer)
    }
    fn video_frame_sampling(&self, _config: &str) -> VideoFrameSampling {
        QWEN3_VIDEO_SAMPLING
    }
    fn modalities(&self, _config: &str) -> Result<Modalities> {
        Ok(Modalities {
            input: vec![
                SupportedModality::Text,
                SupportedModality::Vision,
                SupportedModality::Video,
            ],
            output: vec![SupportedModality::Text],
        })
    }
}

impl IsqModelLoader for Qwen3VLLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"^(language_model\.model|model\.language_model)\.embed_tokens\.weight$",
            r"^lm_head\.(weight|bias)$",
        ])
    }

    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"lm_head\.(weight|bias)$",
            // Attention
            r"^(language_model\.model|model\.language_model)\.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$",
            r"^(language_model\.model|model\.language_model)\.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$",
            r"^(language_model\.model|model\.language_model)\.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$",
            r"^(language_model\.model|model\.language_model)\.layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$",
            // MLP
            r"^(language_model\.model|model\.language_model)\.layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$",
            r"^(language_model\.model|model\.language_model)\.layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$",
            r"^(language_model\.model|model\.language_model)\.layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$",
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for Qwen3VLLoader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Multimodal {
            max_seq_len,
            max_batch_size,
            max_image_shape,
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected multimodal AutoDeviceMapParams for this model!")
        };

        let cfg = parse_config(config)?;

        // For images, grid_t=1. After spatial merging, grid_h and grid_w are reduced.
        let img_seq_len = {
            let cfg = &cfg.vision_config;
            // grid_t is 1 for images (temporal dimension is for video only)
            let grid_t = 1;
            // After patch embedding and spatial merge, the effective grid dimensions are reduced
            let grid_h = (max_image_shape.0 / cfg.patch_size) / cfg.spatial_merge_size;
            let grid_w = (max_image_shape.1 / cfg.patch_size) / cfg.spatial_merge_size;
            grid_t * grid_h * grid_w * max_num_images
        };

        let max_text_attn = {
            let cfg = &cfg.text_config;
            // This model injects the vision information directly into the input embeddings
            let max_seq_len = img_seq_len + max_seq_len.min(&ATTENTION_CHUNK_SIZE);
            max_batch_size * cfg.num_attention_heads * max_seq_len * max_seq_len
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
            max_image_shape,
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected multimodal AutoDeviceMapParams for this model!")
        };

        let cfg = parse_config(config)?;

        // For the vision encoder, before spatial merging
        let img_seq_len = {
            let cfg = &cfg.vision_config;
            // grid_t is 1 for images
            let grid_t = 1;
            let grid_h = max_image_shape.0 / cfg.patch_size;
            let grid_w = max_image_shape.1 / cfg.patch_size;
            grid_t * grid_h * grid_w
        };
        let max_vision_attn = {
            let cfg = &cfg.vision_config;
            (max_batch_size * max_num_images) * cfg.num_heads * img_seq_len * img_seq_len
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
        let cfg = parse_config(config)?;
        let tie = cfg.tie_word_embeddings;
        let text_elems = {
            let cfg = &cfg.text_config;
            let (embed_tokens_pack_factor, lm_head_pack_factor) =
                super::language_model_pack_factors_with_aliases(
                    _quantization,
                    &[
                        "language_model.model.embed_tokens.weight",
                        "model.language_model.embed_tokens.weight",
                    ],
                    &["lm_head.weight"],
                    tie,
                    dtype,
                    weight_pack_factor,
                )?;
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / embed_tokens_pack_factor;
            let lm_head = if !tie {
                cfg.hidden_size * cfg.vocab_size / lm_head_pack_factor
            } else {
                0
            };
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };

        let (patch_merger, deepstack_mergers) = {
            let cfg = &cfg.vision_config;
            let hidden_size = cfg.hidden_size * cfg.spatial_merge_size.pow(2);

            let mlp0 = hidden_size * hidden_size + hidden_size;
            let mlp2 = hidden_size * cfg.out_hidden_size + cfg.out_hidden_size;

            // Main merger: norm uses cfg.hidden_size
            let ln_q = cfg.hidden_size + bias_if!(true, cfg.hidden_size);
            let merger = mlp0 + mlp2 + ln_q;

            // Deepstack mergers: norm uses merged hidden_size
            let ds_ln = hidden_size + bias_if!(true, hidden_size);
            let ds_merger = mlp0 + mlp2 + ds_ln;
            let deepstack = cfg.deepstack_visual_indexes.len() * ds_merger;

            (merger, deepstack)
        };

        let patch_embed = {
            let cfg = &cfg.vision_config;
            let conv_cfg = Conv3dConfig {
                stride: cfg.patch_size,
                ..Default::default()
            };
            let kernel_sizes = [cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size];
            let weight = cfg.in_chans * cfg.hidden_size / conv_cfg.groups
                * kernel_sizes[0]
                * kernel_sizes[1]
                * kernel_sizes[2];
            let bias = cfg.hidden_size;
            weight + bias
        };

        let pos_embed = {
            let cfg = &cfg.vision_config;
            cfg.num_position_embeddings * cfg.hidden_size
        };

        let encoder_layer = {
            let cfg = &cfg.vision_config;
            let norm1 = cfg.hidden_size + bias_if!(true, cfg.hidden_size);
            let norm2 = cfg.hidden_size + bias_if!(true, cfg.hidden_size);

            #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
            let fc1 = cfg.hidden_size * cfg.intermediate_size + cfg.intermediate_size;
            let fc2 = cfg.hidden_size * cfg.intermediate_size + cfg.hidden_size;

            let qkv = cfg.hidden_size * cfg.hidden_size * 3 + cfg.hidden_size * 3;
            let out = cfg.hidden_size * cfg.hidden_size + cfg.hidden_size;

            norm1 + norm2 + fc1 + fc2 + qkv + out
        };

        let elems = text_elems
            + patch_merger
            + deepstack_mergers
            + patch_embed
            + pos_embed
            + encoder_layer * cfg.vision_config.depth;

        Ok(elems * dtype.size_in_bytes())
    }
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg = parse_config(config)?.text_config;
        Ok(decoder_shape(&cfg).layer_sizes_in_bytes(
            cfg.num_hidden_layers,
            dtype,
            weight_pack_factor,
        ))
    }
    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg = parse_config(config)?;
        let cfg = &cfg.text_config;
        Ok(cfg.num_hidden_layers)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = parse_config(config)?.text_config;
        Ok(Box::new(decoder_shape(&cfg).model_config(
            cfg.num_hidden_layers,
            cfg.max_position_embeddings,
            cfg.decoder_spec(false).sliding_window(),
        )))
    }

    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Vision])
    }
}

fn parse_config(config: &str) -> Result<Qwen3VLConfig> {
    let cfg = Qwen3VLConfig::from_json(config)?;
    cfg.text_config.check_experts(false)?;
    Ok(cfg)
}

fn decoder_shape(cfg: &crate::qwen3_vl::config::TextConfig) -> DecoderLayerShape {
    DecoderLayerShape {
        hidden_size: cfg.hidden_size,
        num_attention_heads: cfg.num_attention_heads,
        num_key_value_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim,
        qkv_bias: false,
        o_bias: false,
        qk_norm: true,
        norms: 2,
        mlp: MlpShape::Gated {
            intermediate_size: cfg.intermediate_size,
        },
    }
}
