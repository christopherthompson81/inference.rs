use super::*;

// ======================== DiffusionGemma loader

pub struct DiffusionGemmaLoader;

impl DiffusionGemmaLoader {
    fn is_sliding(tc: &crate::vision_models::gemma4::config::Gemma4TextConfig, i: usize) -> bool {
        tc.layer_types[i] == "sliding_attention"
    }
}

impl MultimodalModelLoader for DiffusionGemmaLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn MultimodalModel + Send + Sync>> {
        let cfg: DiffusionGemmaConfig = serde_json::from_str(config)?;
        Ok(Box::new(DiffusionGemmaModel::new(
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
        let config: DiffusionGemmaConfig = serde_json::from_str(config)?;
        Ok(Box::new(config))
    }
    fn get_processor(
        &self,
        config: &str,
        processor_config: Option<ProcessorConfig>,
        _preprocessor_config: PreProcessorConfig,
        _max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        let cfg: DiffusionGemmaConfig =
            serde_json::from_str(config).expect("Failed to parse DiffusionGemmaConfig");
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
        Arc::new(Gemma4Processor::new(Gemma4ProcessorSettings {
            processor_config: processor_config.unwrap_or_default(),
            patch_size,
            pooling_kernel_size,
            default_output_length,
            supports_images,
            supports_audio: false,
            raw_audio_frame_size: None,
            is_unified: false,
            decode_window: Some(cfg.canvas_length),
            bidirectional_attention: cfg.text_config.bidirectional_attention(),
            vision_attention_on_full_layers: true,
        }))
    }
    fn supports_paged_attention(&self, config: &str) -> bool {
        supports_gemma4_incremental_cache(config)
    }
    fn supports_prefix_cacher(&self, config: &str) -> bool {
        supports_gemma4_incremental_cache(config)
    }
    fn prefixer(&self, _config: &str) -> Arc<dyn MultimodalPromptPrefixer> {
        Arc::new(Gemma4Prefixer)
    }
    fn modalities(&self, config: &str) -> Result<Modalities> {
        let cfg: DiffusionGemmaConfig = serde_json::from_str(config)?;
        let mut input = vec![SupportedModality::Text];
        if cfg.vision_config.is_some() {
            input.push(SupportedModality::Vision);
        }
        Ok(Modalities {
            input,
            output: vec![SupportedModality::Text],
        })
    }
}

impl IsqModelLoader for DiffusionGemmaLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"^model\.decoder\.embed_tokens\.weight$")?,
            Regex::new(r"^model\.decoder\.lm_head\.(weight|bias)$")?,
        ])
    }

    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
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
        ])
    }
    fn immediate_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            Regex::new(r"model\.decoder\.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"model\.decoder\.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"model\.decoder\.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"model\.decoder\.layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            Regex::new(r"model\.decoder\.layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"model\.decoder\.layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"model\.decoder\.layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
            Regex::new(r"model\.decoder\.layers\.(\d+)\.moe\.gate_up_proj\.weight$")?,
            Regex::new(r"model\.decoder\.layers\.(\d+)\.moe\.down_proj\.weight$")?,
            Regex::new(r"model\.decoder\.layers\.(\d+)\.experts\.gate_up_proj\.weight$")?,
            Regex::new(r"model\.decoder\.layers\.(\d+)\.experts\.down_proj\.weight$")?,
        ])
    }
}

impl DeviceMappedModelLoader for DiffusionGemmaLoader {
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

        let cfg: DiffusionGemmaConfig = serde_json::from_str(config)?;
        let tc = &cfg.text_config;

        let vision_tokens_per_image = if cfg.vision_config.is_some() {
            cfg.vision_soft_tokens_per_image.unwrap_or(280)
        } else {
            0
        };
        let total_seq_len = *max_seq_len + vision_tokens_per_image * max_num_images;
        Ok(max_batch_size * tc.num_attention_heads * total_seq_len * total_seq_len)
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

        let cfg: DiffusionGemmaConfig = serde_json::from_str(config)?;
        let (max_vision_attn, max_vision_hidden) =
            cfg.vision_config.as_ref().map_or((0, 0), |vc| {
                let max_patches =
                    vc.default_output_length * vc.pooling_kernel_size * vc.pooling_kernel_size;
                let max_vision_attn = max_batch_size
                    * max_num_images
                    * vc.num_attention_heads
                    * max_patches
                    * max_patches;
                let max_vision_hidden = max_batch_size
                    * max_num_images
                    * max_patches
                    * vc.hidden_size.max(vc.intermediate_size);
                (max_vision_attn, max_vision_hidden)
            });

        // The denoising loop materializes several fp32 [canvas, vocab] transients
        // (logits, log-probs, probs) on the output device.
        let canvas_logits = 4 * cfg.canvas_length * cfg.text_config.vocab_size;

        Ok(max_vision_attn.max(max_vision_hidden).max(canvas_logits))
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: DiffusionGemmaConfig = serde_json::from_str(config)?;
        let tc = &cfg.text_config;
        let text_elems = {
            let (resolved_embed_tokens_pack_factor, lm_head_pack_factor) =
                super::language_model_pack_factors(
                    _quantization,
                    "model.decoder.embed_tokens.weight",
                    "model.decoder.lm_head.weight",
                    tc.tie_word_embeddings,
                    dtype,
                    weight_pack_factor,
                )?;
            let embed_tokens_pack_factor = if tc.tie_word_embeddings {
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
            let self_conditioning = 3 * tc.hidden_size * tc.intermediate_size + tc.hidden_size;
            embed_tokens + lm_head + norm + self_conditioning
        };

        let vision_elems = cfg.vision_config.as_ref().map_or(0, |vc| {
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
        });

        let vision_dtype = if dtype == DType::F16 {
            DType::F32
        } else {
            dtype
        };

        Ok(text_elems * dtype.size_in_bytes() + vision_elems * vision_dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: DiffusionGemmaConfig = serde_json::from_str(config)?;
        let tc = &cfg.text_config;
        let sizes: Vec<usize> = (0..tc.num_hidden_layers)
            .map(|layer_idx| {
                let is_sliding = Self::is_sliding(tc, layer_idx);
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

                // 4 norms + decoder layer_scalar + encoder layer_scalar
                let norms = 4 * tc.hidden_size + 2;

                (attn + mlp + moe + norms) * dtype.size_in_bytes() / weight_pack_factor
            })
            .collect();
        Ok(sizes)
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: DiffusionGemmaConfig = serde_json::from_str(config)?;
        Ok(cfg.text_config.num_hidden_layers)
    }

    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Vision])
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: DiffusionGemmaConfig = serde_json::from_str(config)?;
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
