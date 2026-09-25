use super::*;

// ======================== Phi 4MM loader

/// [`MultimodalLoader`] for a Phi 4MM Vision model.
///
/// [`MultimodalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.MultimodalLoader.html
pub struct Phi4MMLoader;

pub struct Phi4MMPrefixer;

impl MultimodalPromptPrefixer for Phi4MMPrefixer {
    fn prefix_image(&self, image_indexes: Vec<usize>, prompt: &str) -> String {
        // Image indexing starts at 0.

        format!(
            "{}{prompt}",
            image_indexes
                .into_iter()
                .map(|image_index| format!("<|image_{}|>", image_index + 1))
                .join("")
        )
    }
    fn prefix_audio(&self, audio_indexes: Vec<usize>, prompt: &str) -> String {
        // Image indexing starts at 0.

        format!(
            "{}{prompt}",
            audio_indexes
                .into_iter()
                .map(|audio_index| format!("<|audio_{}|>", audio_index + 1))
                .join("")
        )
    }
}

impl MultimodalModelLoader for Phi4MMLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn MultimodalModel + Send + Sync>> {
        let cfg: crate::vision_models::phi4::Phi4MMConfig = serde_json::from_str(config)?;
        Ok(Box::new(Phi4MMModel::new(
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
        let cfg: crate::vision_models::phi4::Phi4MMConfig = serde_json::from_str(config)?;
        Ok(Box::new(cfg))
    }
    fn get_processor(
        &self,
        _model_config: &str,
        processor_config: Option<ProcessorConfig>,
        preprocessor_config: PreProcessorConfig,
        _max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        Phi4MMProcessor::new_processor(processor_config, preprocessor_config)
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
        Arc::new(Phi4MMPrefixer)
    }
    fn modalities(&self, _config: &str) -> Result<Modalities> {
        Ok(Modalities {
            input: vec![
                SupportedModality::Text,
                SupportedModality::Vision,
                SupportedModality::Audio,
            ],
            output: vec![SupportedModality::Text],
        })
    }
}

impl IsqModelLoader for Phi4MMLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"^model\.embed_tokens\.weight$")?,
            Regex::new(r"^lm_head\.(weight|bias)$")?,
        ])
    }

    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"layers\.(\d+)\.self_attn\.qkv_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // MLP
            Regex::new(r"layers\.(\d+)\.mlp\.gate_up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for Phi4MMLoader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        // NOTE: we ignore max_num_images although it can only be one...
        let AutoDeviceMapParams::Multimodal {
            max_seq_len,
            max_batch_size,
            max_image_shape: _,
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected multimodal AutoDeviceMapParams for this model!")
        };

        let cfg: Phi4MMConfig = serde_json::from_str(config)?;

        let vcfg = &PHI4_MM_VISION_CFG;

        let num_patches = (vcfg.image_size / vcfg.patch_size).pow(2);
        let img_seq_len = (num_patches + 1) * max_num_images;

        let max_text_attn = {
            // This model injects the vision information directly into the input embeddings
            let max_seq_len = img_seq_len + max_seq_len.min(&ATTENTION_CHUNK_SIZE);
            max_batch_size * cfg.num_attention_heads * max_seq_len * max_seq_len
        };

        Ok(max_text_attn)
    }

    fn non_mapped_max_act_size_elems(
        &self,
        _config: &str,
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

        let vcfg = &PHI4_MM_VISION_CFG;

        let num_patches = (vcfg.image_size / vcfg.patch_size).pow(2);
        let img_seq_len = num_patches + 1;

        let max_batch_size = max_batch_size
            * (max_image_shape
                .0
                .div_ceil(phi4::inputs_processor::DYHD_BASE_RESOLUTION)
                * max_image_shape
                    .1
                    .div_ceil(phi4::inputs_processor::DYHD_BASE_RESOLUTION)
                + 1);

        let max_vision_attn = (max_batch_size * max_num_images)
            * vcfg.num_attention_heads
            * img_seq_len
            * img_seq_len;
        let max_qkv = 3
            * (max_batch_size
                * vcfg.num_attention_heads
                * img_seq_len
                * (vcfg.hidden_size / vcfg.num_attention_heads));

        Ok(max_vision_attn + max_qkv)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: Phi4MMConfig = serde_json::from_str(config)?;
        let elems = {
            let (embed_tokens_pack_factor, lm_head_pack_factor) =
                super::language_model_pack_factors(
                    _quantization,
                    "model.embed_tokens.weight",
                    "lm_head.weight",
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

            let image_embed = if let Some(img_embed) = &cfg.embd_layer.image_embd_layer {
                let projection_cls = img_embed
                    .projection_cls
                    .clone()
                    .unwrap_or("linear".to_string());
                let with_learnable_separator = img_embed.with_learnable_separator.unwrap_or(false);
                let use_hd_transform = img_embed.use_hd_transform.unwrap_or(false);
                let image_dim_out = PHI4_MM_VISION_CFG.hidden_size;

                let proj = match (projection_cls.as_str(), use_hd_transform) {
                    ("linear", _) => image_dim_out * cfg.hidden_size + cfg.hidden_size,
                    ("mlp", true) => {
                        let a = (image_dim_out * 4) * cfg.hidden_size + cfg.hidden_size;
                        let b = cfg.hidden_size * cfg.hidden_size + cfg.hidden_size;
                        a + b
                    }
                    ("mlp", false) => {
                        let a = image_dim_out * cfg.hidden_size + cfg.hidden_size;
                        let b = cfg.hidden_size * cfg.hidden_size + cfg.hidden_size;
                        a + b
                    }
                    _ => {
                        anyhow::bail!("projection_cls=`{projection_cls}` not implemented.");
                    }
                };

                let (glb_gn, sub_gn) = if with_learnable_separator {
                    let glb_gn = image_dim_out * 4;
                    let sub_gn = image_dim_out * 4;
                    (glb_gn, sub_gn)
                } else {
                    (0, 0)
                };

                let vision_transformer = {
                    let cfg = &PHI4_MM_VISION_CFG;

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

                proj + glb_gn + sub_gn + vision_transformer
            } else {
                0
            };

            embed_tokens + lm_head + norm + image_embed
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
        let cfg: Phi4MMConfig = serde_json::from_str(config)?;
        let per_layer_elems = {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let size_in = cfg.hidden_size;
            let head_dim = cfg.head_dim();
            let op_size =
                cfg.num_attention_heads * head_dim + 2 * cfg.num_key_value_heads() * head_dim;
            let qkv_proj = size_in * op_size / weight_pack_factor;
            let o_proj = (cfg.num_attention_heads * head_dim) * size_in / weight_pack_factor;

            let h_size = cfg.hidden_size;
            let i_size = cfg.intermediate_size;
            let gate_up_proj = h_size * (2 * i_size) / weight_pack_factor;
            let down_proj = h_size * i_size / weight_pack_factor;

            input_layernorm
                + post_attention_layernorm
                + qkv_proj
                + o_proj
                + gate_up_proj
                + down_proj
        };
        Ok(vec![
            per_layer_elems * dtype.size_in_bytes();
            cfg.num_hidden_layers
        ])
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: Phi4MMConfig = serde_json::from_str(config)?;
        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: Phi4MMConfig = serde_json::from_str(config)?;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads(),
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: cfg.sliding_window,
            k_head_dim: cfg.head_dim(),
            v_head_dim: cfg.head_dim(),
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }

    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Vision, NonMappedSubModel::Audio])
    }
}
