use super::*;

/// [`MultimodalLoader`] for a Qwen3.5 MoE (hybrid GDN + full attention) model.
///
/// [`MultimodalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.MultimodalLoader.html
pub struct Qwen3_5MoeLoader;

pub struct Qwen3_5MoePrefixer;

impl MultimodalPromptPrefixer for Qwen3_5MoePrefixer {
    // No-op: With MessagesAction::Keep, the chat template handles image tokens
    // when it sees {"type": "image"} entries in the content.
}

impl MultimodalModelLoader for Qwen3_5MoeLoader {
    fn runtime_config<'a>(
        &self,
        config: &'a str,
        max_model_len: Option<usize>,
    ) -> Result<Cow<'a, str>> {
        match max_model_len {
            Some(max_model_len) => Ok(Cow::Owned(
                crate::vision_models::qwen3_5::config::apply_max_model_len(config, max_model_len)?,
            )),
            None => Ok(Cow::Borrowed(config)),
        }
    }

    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn MultimodalModel + Send + Sync>> {
        let cfg: Qwen3_5MoeConfig = serde_json::from_str(config)?;
        Ok(Box::new(Qwen3_5MoeModel::new(
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
        let config: Qwen3_5MoeConfig = serde_json::from_str(config)?;
        Ok(Box::new(config))
    }
    fn get_processor(
        &self,
        _model_config: &str,
        _processor_config: Option<ProcessorConfig>,
        _preprocessor_config: PreProcessorConfig,
        max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        Arc::new(Qwen3_5MoeProcessor::new(max_edge))
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
        Arc::new(Qwen3_5MoePrefixer)
    }
    fn video_frame_sampling(&self, _config: &str) -> crate::VideoFrameSampling {
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

impl IsqModelLoader for Qwen3_5MoeLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"^(language_model\.model|model\.language_model)\.embed_tokens\.weight$")?,
            Regex::new(r"^lm_head\.(weight|bias)$")?,
        ])
    }

    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Full attention projections
            Regex::new(
                r"^(language_model\.model|model\.language_model)\.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"^(language_model\.model|model\.language_model)\.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"^(language_model\.model|model\.language_model)\.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"^(language_model\.model|model\.language_model)\.layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$",
            )?,
            // GDN linear attention projections
            Regex::new(
                r"^(language_model\.model|model\.language_model)\.layers\.(\d+)\.linear_attn\.in_proj_qkv\.(weight|bias)$",
            )?,
            Regex::new(
                r"^(language_model\.model|model\.language_model)\.layers\.(\d+)\.linear_attn\.in_proj_z\.(weight|bias)$",
            )?,
            Regex::new(
                r"^(language_model\.model|model\.language_model)\.layers\.(\d+)\.linear_attn\.in_proj_b\.(weight|bias)$",
            )?,
            Regex::new(
                r"^(language_model\.model|model\.language_model)\.layers\.(\d+)\.linear_attn\.in_proj_a\.(weight|bias)$",
            )?,
            Regex::new(
                r"^(language_model\.model|model\.language_model)\.layers\.(\d+)\.linear_attn\.out_proj\.(weight|bias)$",
            )?,
            // MoE experts
            Regex::new(
                r"^(language_model\.model|model\.language_model)\.layers\.(\d+)\.mlp\.experts\.(\d+)\.gate_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"^(language_model\.model|model\.language_model)\.layers\.(\d+)\.mlp\.experts\.(\d+)\.up_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"^(language_model\.model|model\.language_model)\.layers\.(\d+)\.mlp\.experts\.(\d+)\.down_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"^(language_model\.model|model\.language_model)\.layers\.(\d+)\.mlp\.experts\.(gate_proj|up_proj|down_proj)\.weight$",
            )?,
            Regex::new(
                r"^(language_model\.model|model\.language_model)\.layers\.(\d+)\.mlp\.experts\.gate_up_proj\.weight$",
            )?,
            Regex::new(
                r"^(language_model\.model|model\.language_model)\.layers\.(\d+)\.mlp\.experts\.down_proj\.weight$",
            )?,
            // Shared expert
            Regex::new(
                r"^(language_model\.model|model\.language_model)\.layers\.(\d+)\.mlp\.shared_expert\.gate_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"^(language_model\.model|model\.language_model)\.layers\.(\d+)\.mlp\.shared_expert\.up_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"^(language_model\.model|model\.language_model)\.layers\.(\d+)\.mlp\.shared_expert\.down_proj\.(weight|bias)$",
            )?,
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
    fn isq_layer_regexes_moqe(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            // MoE experts
            Regex::new(
                r"^(language_model\.model|model\.language_model)\.layers\.(\d+)\.mlp\.experts\.(\d+)\.gate_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"^(language_model\.model|model\.language_model)\.layers\.(\d+)\.mlp\.experts\.(\d+)\.up_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"^(language_model\.model|model\.language_model)\.layers\.(\d+)\.mlp\.experts\.(\d+)\.down_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"^(language_model\.model|model\.language_model)\.layers\.(\d+)\.mlp\.experts\.(gate_proj|up_proj|down_proj)\.weight$",
            )?,
            Regex::new(
                r"^(language_model\.model|model\.language_model)\.layers\.(\d+)\.mlp\.experts\.gate_up_proj\.weight$",
            )?,
            Regex::new(
                r"^(language_model\.model|model\.language_model)\.layers\.(\d+)\.mlp\.experts\.down_proj\.weight$",
            )?,
        ])
    }
    fn immediate_isq_predicates_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes_moqe(config)
    }
}

impl DeviceMappedModelLoader for Qwen3_5MoeLoader {
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

        let cfg: Qwen3_5MoeConfig = serde_json::from_str(config)?;

        let img_seq_len = {
            let cfg = &cfg.vision_config;
            let grid_t = 1;
            let grid_h = (max_image_shape.0 / cfg.patch_size) / cfg.spatial_merge_size;
            let grid_w = (max_image_shape.1 / cfg.patch_size) / cfg.spatial_merge_size;
            grid_t * grid_h * grid_w * max_num_images
        };

        let max_text_attn = {
            let cfg = &cfg.text_config;
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

        let cfg: Qwen3_5MoeConfig = serde_json::from_str(config)?;

        let img_seq_len = {
            let cfg = &cfg.vision_config;
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
        let cfg: Qwen3_5MoeConfig = serde_json::from_str(config)?;
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

            let ln_q = cfg.hidden_size + bias_if!(true, cfg.hidden_size);
            let merger = mlp0 + mlp2 + ln_q;

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
        let cfg: Qwen3_5MoeConfig = serde_json::from_str(config)?;
        let text_cfg = &cfg.text_config;
        let layer_types = text_cfg.layer_types();

        let mut layer_sizes = Vec::with_capacity(text_cfg.num_hidden_layers);

        for layer_type in &layer_types {
            let input_layernorm = text_cfg.hidden_size;
            let post_attention_layernorm = text_cfg.hidden_size;

            let attn_elems = match layer_type {
                crate::vision_models::qwen3_5_moe::config::LayerType::FullAttention => {
                    let size_in = text_cfg.hidden_size;
                    let size_q = text_cfg.head_dim * text_cfg.num_attention_heads;
                    let size_kv = text_cfg.head_dim * text_cfg.num_key_value_heads;
                    let q_proj = size_in * size_q * 2 / weight_pack_factor;
                    let k_proj = size_in * size_kv / weight_pack_factor;
                    let v_proj = size_in * size_kv / weight_pack_factor;
                    let o_proj = size_q * size_in / weight_pack_factor;
                    let q_norm = text_cfg.head_dim;
                    let k_norm = text_cfg.head_dim;
                    q_proj + k_proj + v_proj + o_proj + q_norm + k_norm
                }
                crate::vision_models::qwen3_5_moe::config::LayerType::LinearAttention => {
                    let hidden = text_cfg.hidden_size;
                    let value_dim = text_cfg.linear_value_dim();
                    let conv_dim = text_cfg.linear_conv_dim();
                    let in_proj_qkv = hidden * conv_dim / weight_pack_factor;
                    let in_proj_z = hidden * value_dim / weight_pack_factor;
                    let in_proj_ba =
                        hidden * (text_cfg.linear_num_value_heads * 2) / weight_pack_factor;
                    // out_proj: value_dim -> hidden
                    let out_proj = value_dim * hidden / weight_pack_factor;
                    // conv1d weight
                    let conv1d = conv_dim * text_cfg.linear_conv_kernel_dim;
                    // dt_bias, A_log, norm weight
                    let dt_bias = text_cfg.linear_num_value_heads;
                    let a_log = text_cfg.linear_num_value_heads;
                    // RmsNormGated over per-head value dim
                    let norm = text_cfg.linear_value_head_dim;
                    in_proj_qkv
                        + in_proj_z
                        + in_proj_ba
                        + out_proj
                        + conv1d
                        + dt_bias
                        + a_log
                        + norm
                }
            };

            // All layers have MoE
            let moe_elems = {
                let gate = text_cfg.hidden_size * text_cfg.num_experts;
                let per_expert = {
                    let h_size = text_cfg.hidden_size;
                    let i_size = text_cfg.moe_intermediate_size;
                    let gate_proj = h_size * i_size / weight_pack_factor;
                    let up_proj = h_size * i_size / weight_pack_factor;
                    let down_proj = i_size * h_size / weight_pack_factor;
                    gate_proj + up_proj + down_proj
                };
                let shared_expert = {
                    let h_size = text_cfg.hidden_size;
                    let i_size = text_cfg.shared_expert_intermediate_size;
                    let gate_proj = h_size * i_size / weight_pack_factor;
                    let up_proj = h_size * i_size / weight_pack_factor;
                    let down_proj = i_size * h_size / weight_pack_factor;
                    gate_proj + up_proj + down_proj
                };
                let shared_expert_gate = text_cfg.hidden_size;
                gate + per_expert * text_cfg.num_experts + shared_expert + shared_expert_gate
            };

            let per_layer_elems =
                input_layernorm + post_attention_layernorm + attn_elems + moe_elems;

            layer_sizes.push(per_layer_elems * dtype.size_in_bytes());
        }

        Ok(layer_sizes)
    }
    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: Qwen3_5MoeConfig = serde_json::from_str(config)?;
        Ok(cfg.text_config.num_hidden_layers)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: Qwen3_5MoeConfig = serde_json::from_str(config)?;
        let cfg = &cfg.text_config;

        let base = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: None,
            k_head_dim: cfg.head_dim,
            v_head_dim: cfg.head_dim,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        let paged_layers = cfg
            .layer_types()
            .into_iter()
            .map(|ty| {
                matches!(
                    ty,
                    crate::vision_models::qwen3_5_moe::config::LayerType::FullAttention
                )
            })
            .collect();

        Ok(Box::new(
            HybridPagedKvCacheConfig::new(base, paged_layers)
                .with_uniform_prefix_prefill_attention_features(Default::default()),
        ))
    }

    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Vision])
    }
}
