use super::*;
use crate::qwen2vl::inputs_processor::{IMAGE_PAD, VISION_END, VISION_START};

/// `MultimodalLoader` for an Qwen2_5VL model.
pub struct Qwen2_5VLLoader;

pub struct Qwen2_5VLPrefixer;

impl MultimodalPromptPrefixer for Qwen2_5VLPrefixer {
    fn prefix_image(&self, image_indexes: Vec<usize>, prompt: &str) -> String {
        format!(
            "{}{prompt}",
            format!("{VISION_START}{IMAGE_PAD}{VISION_END}").repeat(image_indexes.len())
        )
    }
}

impl MultimodalModelLoader for Qwen2_5VLLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn MultimodalModel + Send + Sync>> {
        let cfg = Qwen2_5VLConfig::from_json(config)?;
        Ok(Box::new(Qwen2_5VLModel::new(
            &cfg,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let config = Qwen2_5VLConfig::from_json(config)?;
        Ok(Box::new(config))
    }
    fn supports_paged_attention(&self, _config: &str) -> bool {
        true
    }
    fn supports_encoder_cache(&self, _config: &str) -> bool {
        true
    }
    fn prefixer(&self, _config: &str) -> Arc<dyn MultimodalPromptPrefixer> {
        Arc::new(Qwen2_5VLPrefixer)
    }
    fn modalities(&self, _config: &str) -> Result<Modalities> {
        Ok(Modalities {
            input: vec![SupportedModality::Text, SupportedModality::Vision],
            output: vec![SupportedModality::Text],
        })
    }
}

impl IsqModelLoader for Qwen2_5VLLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"^(model|language_model\.model)\.embed_tokens\.weight$",
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
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for Qwen2_5VLLoader {
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

        let cfg = Qwen2_5VLConfig::from_json(config)?;

        let img_seq_len = {
            let cfg = &cfg.vision_config;
            let grid_t = max_num_images / cfg.temporal_patch_size;
            let grid_h = max_image_shape.0 / cfg.patch_size;
            let grid_w = max_image_shape.1 / cfg.patch_size;
            grid_t * grid_h * grid_w
        };
        let img_seq_len = img_seq_len * max_num_images;

        let max_text_attn = {
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

        let cfg = Qwen2_5VLConfig::from_json(config)?;

        let img_seq_len = {
            let cfg = &cfg.vision_config;
            let grid_t = max_num_images / cfg.temporal_patch_size;
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
        let cfg = Qwen2_5VLConfig::from_json(config)?;
        let text_elems = {
            let (embed_tokens_pack_factor, lm_head_pack_factor) =
                super::language_model_pack_factors_with_aliases(
                    _quantization,
                    &[
                        "model.embed_tokens.weight",
                        "language_model.model.embed_tokens.weight",
                    ],
                    &["lm_head.weight"],
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

        let patch_merger = {
            let cfg = &cfg.vision_config;
            let hidden_size = cfg.hidden_size * cfg.spatial_merge_size.pow(2);

            let mlp0 = hidden_size * hidden_size + hidden_size;
            let mlp2 = hidden_size * cfg.hidden_size + cfg.hidden_size;

            let ln_q = cfg.hidden_size + bias_if!(true, cfg.hidden_size);

            mlp0 + mlp2 + ln_q
        };

        let patch_embed = {
            let cfg = &cfg.vision_config;
            let conv_cfg = Conv3dConfig {
                stride: cfg.patch_size,
                ..Default::default()
            };
            let kernel_sizes = [cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size];
            cfg.in_chans * cfg.hidden_size / conv_cfg.groups
                * kernel_sizes[0]
                * kernel_sizes[1]
                * kernel_sizes[2]
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

        let elems =
            text_elems + patch_merger + patch_embed + encoder_layer * cfg.vision_config.depth;

        Ok(elems * dtype.size_in_bytes())
    }
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg = Qwen2_5VLConfig::from_json(config)?;
        Ok(super::qwen2vl::decoder_shape(&cfg).layer_sizes_in_bytes(
            cfg.num_hidden_layers,
            dtype,
            weight_pack_factor,
        ))
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = Qwen2_5VLConfig::from_json(config)?;
        Ok(Box::new(super::qwen2vl::decoder_shape(&cfg).model_config(
            cfg.num_hidden_layers,
            cfg.max_position_embeddings,
            cfg.sliding_window,
        )))
    }

    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Vision])
    }
}
