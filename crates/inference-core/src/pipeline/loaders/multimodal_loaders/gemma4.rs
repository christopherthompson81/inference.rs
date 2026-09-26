use super::*;

// ── Gemma4 ─────────────────────────────────────────────────────────────────

pub struct Gemma4Loader;

fn gemma4_runtime_config(config: &str, max_model_len: Option<usize>) -> Result<Cow<'_, str>> {
    let Some(max_model_len) = max_model_len else {
        return Ok(Cow::Borrowed(config));
    };
    anyhow::ensure!(max_model_len > 0, "max_model_len must be greater than zero");

    let parsed: Gemma4Config = serde_json::from_str(config)?;
    if parsed.text_config.max_position_embeddings <= max_model_len {
        return Ok(Cow::Borrowed(config));
    }

    let mut value: serde_json::Value = serde_json::from_str(config)?;
    let root = value
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("Gemma4 config must be a JSON object"))?;
    if root
        .get("text_config")
        .is_some_and(serde_json::Value::is_object)
    {
        root.get_mut("text_config")
            .and_then(serde_json::Value::as_object_mut)
            .expect("text_config was checked as an object")
            .insert(
                "max_position_embeddings".to_string(),
                serde_json::Value::from(max_model_len),
            );
    } else {
        root.insert(
            "max_position_embeddings".to_string(),
            serde_json::Value::from(max_model_len),
        );
    }

    Ok(Cow::Owned(serde_json::to_string(&value)?))
}

impl MultimodalModelLoader for Gemma4Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn MultimodalModel + Send + Sync>> {
        let cfg: Gemma4Config = serde_json::from_str(config)?;
        Ok(Box::new(Gemma4Model::new(
            &cfg,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn runtime_config<'a>(
        &self,
        config: &'a str,
        max_model_len: Option<usize>,
    ) -> Result<Cow<'a, str>> {
        gemma4_runtime_config(config, max_model_len)
    }
    fn is_gptx(&self, _config: &str) -> bool {
        true
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let config: Gemma4Config = serde_json::from_str(config)?;
        Ok(Box::new(config))
    }
    fn get_processor(
        &self,
        config: &str,
        processor_config: Option<ProcessorConfig>,
        _preprocessor_config: PreProcessorConfig,
        _max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        let cfg: Gemma4Config = serde_json::from_str(config).expect("Failed to parse Gemma4Config");
        let (patch_size, pooling_kernel_size, default_output_length, supports_images) = cfg
            .vision_config
            .as_ref()
            .map_or((16, 1, 0, false), |vision_cfg| {
                (
                    vision_cfg.patch_size,
                    vision_cfg.pooling_kernel_size,
                    vision_cfg.default_output_length,
                    true,
                )
            });
        let raw_audio_frame_size = cfg
            .audio_config
            .as_ref()
            .and_then(|audio_cfg| cfg.is_unified().then_some(audio_cfg.input_feat_size()));
        Arc::new(Gemma4Processor::new(Gemma4ProcessorSettings {
            processor_config: processor_config.unwrap_or_default(),
            patch_size,
            pooling_kernel_size,
            default_output_length,
            supports_images,
            supports_audio: cfg.audio_config.is_some(),
            raw_audio_frame_size,
            is_unified: cfg.is_unified(),
            decode_window: None,
            bidirectional_attention: cfg.text_config.bidirectional_attention(),
            vision_attention_on_full_layers: false,
        }))
    }
    fn supports_paged_attention(&self, config: &str) -> bool {
        supports_gemma4_incremental_cache(config)
    }
    fn supports_encoder_cache(&self, _config: &str) -> bool {
        true
    }
    fn supports_prefix_cacher(&self, config: &str) -> bool {
        supports_gemma4_incremental_cache(config)
    }
    fn prefixer(&self, _config: &str) -> Arc<dyn MultimodalPromptPrefixer> {
        Arc::new(Gemma4Prefixer)
    }
    fn modalities(&self, config: &str) -> Result<Modalities> {
        let cfg: Gemma4Config = serde_json::from_str(config)?;
        let mut input = vec![SupportedModality::Text];
        if cfg.vision_config.is_some() {
            input.push(SupportedModality::Vision);
            input.push(SupportedModality::Video);
        }
        if cfg.audio_config.is_some() {
            input.push(SupportedModality::Audio);
        }
        Ok(Modalities {
            input,
            output: vec![SupportedModality::Text],
        })
    }
}

impl IsqModelLoader for Gemma4Loader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"^model\.language_model\.embed_tokens\.weight$")?,
            Regex::new(r"^model\.language_model\.embed_tokens_per_layer\.weight$")?,
            Regex::new(r"^model\.language_model\.lm_head\.(weight|bias)$")?,
        ])
    }

    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        // `embed_vision.embedding_projection` is intentionally excluded.
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.moe\.gate_up_proj\.weight$")?,
            Regex::new(r"layers\.(\d+)\.moe\.down_proj\.weight$")?,
            Regex::new(r"layers\.(\d+)\.experts\.gate_up_proj\.weight$")?,
            Regex::new(r"layers\.(\d+)\.experts\.down_proj\.weight$")?,
            Regex::new(r"layers\.(\d+)\.(moe|experts)\.(gate_proj|up_proj|down_proj)\.weight$")?,
            Regex::new(r"per_layer_model_projection\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.per_layer_input_gate\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.per_layer_projection\.(weight|bias)$")?,
        ])
    }
    fn immediate_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"model\.language_model\.embed_tokens\.weight$")?,
            Regex::new(r"model\.language_model\.embed_tokens_per_layer\.weight$")?,
            Regex::new(r"lm_head\.(weight|bias)$")?,
            Regex::new(r"model\.language_model\.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"model\.language_model\.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"model\.language_model\.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"model\.language_model\.layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            Regex::new(r"model\.language_model\.layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"model\.language_model\.layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"model\.language_model\.layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
            Regex::new(r"model\.language_model\.layers\.(\d+)\.moe\.gate_up_proj\.weight$")?,
            Regex::new(r"model\.language_model\.layers\.(\d+)\.moe\.down_proj\.weight$")?,
            Regex::new(r"model\.language_model\.layers\.(\d+)\.experts\.gate_up_proj\.weight$")?,
            Regex::new(r"model\.language_model\.layers\.(\d+)\.experts\.down_proj\.weight$")?,
            Regex::new(
                r"model\.language_model\.layers\.(\d+)\.(moe|experts)\.(gate_proj|up_proj|down_proj)\.weight$",
            )?,
            Regex::new(r"model\.language_model\.per_layer_model_projection\.(weight|bias)$")?,
            Regex::new(
                r"model\.language_model\.layers\.(\d+)\.per_layer_input_gate\.(weight|bias)$",
            )?,
            Regex::new(
                r"model\.language_model\.layers\.(\d+)\.per_layer_projection\.(weight|bias)$",
            )?,
        ])
    }
}

impl DeviceMappedModelLoader for Gemma4Loader {
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

        let cfg: Gemma4Config = serde_json::from_str(config)?;
        let tc = &cfg.text_config;

        let vision_tokens_per_image = if cfg.vision_config.is_some() {
            cfg.vision_soft_tokens_per_image.unwrap_or(280)
        } else {
            0
        };
        let audio_tokens = if cfg.audio_config.is_some() { 750 } else { 0 };
        let total_seq_len = *max_seq_len + vision_tokens_per_image * max_num_images + audio_tokens;
        let max_text_attn = max_batch_size * tc.num_attention_heads * total_seq_len * total_seq_len;

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

        let cfg: Gemma4Config = serde_json::from_str(config)?;
        let (max_vision_attn, max_vision_hidden) =
            cfg.vision_config.as_ref().map_or((0, 0), |vc| {
                let (max_patches, hidden_size, intermediate_size, num_attention_heads) = if cfg
                    .is_unified()
                {
                    (
                        vc.default_output_length,
                        vc.hidden_size(),
                        vc.hidden_size(),
                        0,
                    )
                } else {
                    (
                        vc.default_output_length * vc.pooling_kernel_size * vc.pooling_kernel_size,
                        vc.hidden_size,
                        vc.intermediate_size,
                        vc.num_attention_heads,
                    )
                };
                let max_vision_attn = max_batch_size
                    * max_num_images
                    * num_attention_heads
                    * max_patches
                    * max_patches;
                let max_vision_hidden = max_batch_size
                    * max_num_images
                    * max_patches
                    * hidden_size.max(intermediate_size);
                (max_vision_attn, max_vision_hidden)
            });

        let max_audio_activation = cfg.audio_config.as_ref().map_or(0, |audio_cfg| {
            if cfg.is_unified() {
                max_batch_size * 750 * audio_cfg.input_feat_size()
            } else {
                let subsample_factor: usize = audio_cfg
                    .sscp_conv_stride_size
                    .iter()
                    .map(|stride| stride[0])
                    .product();
                let max_audio_frames = 750 * subsample_factor.max(1);
                let audio_seq_after_subsample = max_audio_frames / subsample_factor.max(1);
                let audio_encoder_act = audio_seq_after_subsample * (audio_cfg.hidden_size * 4);
                let chunk_size = audio_cfg.conf_attention_chunk_size;
                let context_size = chunk_size + audio_cfg.conf_attention_context_left - 1
                    + audio_cfg.conf_attention_context_right;
                let num_chunks = audio_seq_after_subsample.div_ceil(chunk_size);
                let audio_attn_act =
                    audio_cfg.conf_num_attention_heads * num_chunks * chunk_size * context_size;

                max_batch_size * audio_encoder_act.max(audio_attn_act)
            }
        });

        Ok(max_vision_attn
            .max(max_vision_hidden)
            .max(max_audio_activation))
    }
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: Gemma4Config = serde_json::from_str(config)?;
        let tc = &cfg.text_config;
        let text_elems = {
            let (resolved_embed_tokens_pack_factor, lm_head_pack_factor) =
                super::language_model_pack_factors(
                    _quantization,
                    "model.language_model.embed_tokens.weight",
                    "model.language_model.lm_head.weight",
                    tc.tie_word_embeddings,
                    dtype,
                    weight_pack_factor,
                )?;
            let embed_tokens_pack_factor =
                if tc.tie_word_embeddings && tc.keep_tied_lm_head_unquantized {
                    1
                } else {
                    resolved_embed_tokens_pack_factor
                };
            let embed_tokens = tc.hidden_size * tc.vocab_size / embed_tokens_pack_factor;
            let lm_head = if !tc.tie_word_embeddings {
                tc.hidden_size * tc.vocab_size / lm_head_pack_factor
            } else {
                0
            };
            let norm = tc.hidden_size;

            let ple_dim = tc.hidden_size_per_layer_input.unwrap_or(0);
            let ple_vocab = tc.vocab_size_per_layer_input.unwrap_or(tc.vocab_size);
            let ple_embedding_pack_factor = super::promoted_tensor_pack_factor(
                _quantization,
                "model.language_model.embed_tokens_per_layer.weight",
                dtype,
                weight_pack_factor,
            )?;
            let embed_tokens_per_layer = if ple_dim > 0 {
                ple_vocab * tc.num_hidden_layers * ple_dim / ple_embedding_pack_factor
            } else {
                0
            };
            let per_layer_model_projection = if ple_dim > 0 {
                tc.hidden_size * tc.num_hidden_layers * ple_dim / weight_pack_factor
            } else {
                0
            };
            let per_layer_projection_norm = ple_dim;

            embed_tokens
                + lm_head
                + norm
                + embed_tokens_per_layer
                + per_layer_model_projection
                + per_layer_projection_norm
        };

        let vision_elems = cfg.vision_config.as_ref().map_or(0, |vc| {
            if cfg.is_unified() {
                let hidden_size = vc.hidden_size();
                let patch_dim = vc.patch_size() * vc.patch_size() * 3;
                let patch_norms = 2 * patch_dim + 4 * hidden_size;
                let patch_dense = hidden_size * patch_dim + hidden_size;
                let pos_embedding = 2 * vc.position_embedding_size * hidden_size;
                let embed_vision = hidden_size * tc.hidden_size;
                patch_norms + patch_dense + pos_embedding + embed_vision
            } else {
                let vision_layer_elems = {
                    let quantized = vc.hidden_size * vc.num_attention_heads * vc.head_dim
                        + 3 * (vc.hidden_size * vc.num_key_value_heads * vc.head_dim)
                        + 2 * (vc.hidden_size * vc.intermediate_size)
                        + vc.intermediate_size * vc.hidden_size;
                    let norms = 2 * vc.head_dim + 4 * vc.hidden_size;
                    quantized + norms
                };
                let patch_embed = vc.patch_size * vc.patch_size * 3 * vc.hidden_size;
                let position_embedding_table = 2 * vc.position_embedding_size * vc.hidden_size;
                let patch_embedder = patch_embed + position_embedding_table;
                let encoder = vc.num_hidden_layers * vision_layer_elems;
                let embed_vision = vc.hidden_size * tc.hidden_size;

                patch_embedder + encoder + embed_vision
            }
        });

        let audio_elems = cfg.audio_config.as_ref().map_or(0, |audio_cfg| {
            if cfg.is_unified() {
                audio_cfg.input_feat_size() * tc.hidden_size
            } else {
                let mut f_out = audio_cfg.input_feat_size();
                for i in 0..2 {
                    let kernel_w = audio_cfg.sscp_conv_kernel_size[i][1];
                    let stride_w = audio_cfg.sscp_conv_stride_size[i][1];
                    let pad_left = 1;
                    let pad_right = 1;
                    f_out = (f_out + pad_left + pad_right + stride_w - kernel_w) / stride_w;
                }

                let subsample_conv_projection = {
                    let conv_0 = audio_cfg.sscp_conv_channel_size[0]
                        * audio_cfg.sscp_conv_kernel_size[0][0]
                        * audio_cfg.sscp_conv_kernel_size[0][1];
                    let conv_1 = audio_cfg.sscp_conv_channel_size[0]
                        * audio_cfg.sscp_conv_channel_size[1]
                        * audio_cfg.sscp_conv_kernel_size[1][0]
                        * audio_cfg.sscp_conv_kernel_size[1][1];
                    let norms =
                        audio_cfg.sscp_conv_channel_size[0] + audio_cfg.sscp_conv_channel_size[1];
                    let input_proj =
                        audio_cfg.sscp_conv_channel_size[1] * f_out * audio_cfg.hidden_size;
                    conv_0 + conv_1 + norms + input_proj
                };

                let conformer_block = {
                    let attention = 5 * (audio_cfg.hidden_size * audio_cfg.hidden_size)
                        + 2 * audio_cfg.hidden_size
                        + audio_cfg.hidden_size / audio_cfg.conf_num_attention_heads
                        + audio_cfg.hidden_size / 2
                        + (audio_cfg.conf_attention_context_left
                            + audio_cfg.conf_attention_context_right
                            + 1)
                        + (audio_cfg.conf_attention_chunk_size
                            * (audio_cfg.conf_attention_chunk_size
                                + audio_cfg.conf_attention_context_left
                                - 1
                                + audio_cfg.conf_attention_context_right))
                        + 1;
                    let ffw = 2
                        * (2 * audio_cfg.hidden_size
                            + 2 * (audio_cfg.hidden_size * (audio_cfg.hidden_size * 4)));
                    let conv = 2 * audio_cfg.hidden_size
                        + audio_cfg.hidden_size * (audio_cfg.hidden_size * 2)
                        + audio_cfg.hidden_size * audio_cfg.hidden_size
                        + audio_cfg.hidden_size * audio_cfg.conf_conv_kernel_size;
                    attention + ffw + conv + audio_cfg.hidden_size
                };

                let output_proj = audio_cfg.output_proj_dims.map_or(0, |output_dim| {
                    audio_cfg.hidden_size * output_dim + output_dim
                });
                let audio_embed_hidden =
                    audio_cfg.output_proj_dims.unwrap_or(audio_cfg.hidden_size);
                let embed_audio = audio_embed_hidden * tc.hidden_size;

                subsample_conv_projection
                    + audio_cfg.conf_num_hidden_layers * conformer_block
                    + output_proj
                    + embed_audio
            }
        });

        let vision_dtype = if dtype == DType::F16 {
            DType::F32
        } else {
            dtype
        };

        Ok(text_elems * dtype.size_in_bytes()
            + vision_elems * vision_dtype.size_in_bytes()
            + audio_elems * DType::F32.size_in_bytes())
    }
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: Gemma4Config = serde_json::from_str(config)?;
        let tc = &cfg.text_config;
        let sizes: Vec<usize> = (0..tc.num_hidden_layers)
            .map(|layer_idx| {
                let is_sliding = {
                    let is_last = layer_idx == tc.num_hidden_layers - 1;
                    !is_last && (layer_idx + 1) % tc.sliding_window_pattern != 0
                };
                let hd = if is_sliding {
                    tc.head_dim
                } else {
                    tc.global_head_dim
                };
                let nkv = if is_sliding {
                    tc.num_key_value_heads
                } else {
                    tc.num_global_key_value_heads
                        .unwrap_or(tc.num_key_value_heads)
                };
                let use_k_eq_v = tc.attention_k_eq_v && !is_sliding;

                let mut attn = tc.hidden_size * tc.num_attention_heads * hd
                    + tc.hidden_size * nkv * hd
                    + tc.num_attention_heads * hd * tc.hidden_size;
                if !use_k_eq_v {
                    attn += tc.hidden_size * nkv * hd;
                }
                attn += 2 * hd;

                let mlp = 3 * tc.hidden_size * tc.intermediate_size;

                let moe = if tc.enable_moe_block {
                    let ne = tc.num_experts.unwrap_or(0);
                    let ei = tc.expert_intermediate_size().unwrap_or(0);
                    ne * tc.hidden_size * ei * 2
                        + ne * ei * tc.hidden_size
                        + ne
                        + ne * tc.hidden_size
                        + tc.hidden_size
                        + 3 * tc.hidden_size
                } else {
                    0
                };

                let ple = if tc.hidden_size_per_layer_input.unwrap_or(0) > 0 {
                    let pd = tc.hidden_size_per_layer_input.unwrap();
                    tc.hidden_size * pd + pd * tc.hidden_size + tc.hidden_size
                } else {
                    0
                };

                let norms = 4 * tc.hidden_size + 1;

                (attn + mlp + moe + ple + norms) * dtype.size_in_bytes() / weight_pack_factor
            })
            .collect();
        Ok(sizes)
    }
    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: Gemma4Config = serde_json::from_str(config)?;
        Ok(cfg.text_config.num_hidden_layers)
    }

    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Vision, NonMappedSubModel::Audio])
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: Gemma4Config = serde_json::from_str(config)?;
        let tc = &cfg.text_config;

        let cfg = ModelConfigMetadata {
            max_seq_len: tc.max_position_embeddings,
            num_layers: tc.num_hidden_layers,
            hidden_size: tc.hidden_size,
            num_kv_heads: tc.num_key_value_heads,
            num_attn_heads: tc.num_attention_heads,
            sliding_window: Some(tc.sliding_window),
            k_head_dim: tc.global_head_dim,
            v_head_dim: tc.global_head_dim,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }
}

#[allow(dead_code)]
pub struct Gemma4Prefixer;

impl MultimodalPromptPrefixer for Gemma4Prefixer {
    fn prefix_image(&self, _image_indexes: Vec<usize>, prompt: &str) -> String {
        prompt.to_string()
    }
    fn prefix_video(&self, _video_indexes: Vec<usize>, prompt: &str) -> String {
        prompt.to_string()
    }
}
