use super::*;

/// [`MultimodalLoader`] for an Llama Vision model.
///
/// [`MultimodalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.MultimodalLoader.html
pub struct VLlama4Loader;

pub struct VLlama4Prefixer;

impl MultimodalPromptPrefixer for VLlama4Prefixer {
    fn prefix_image(&self, image_indexes: Vec<usize>, prompt: &str) -> String {
        format!(
            "{}{prompt}",
            llama4::IMAGE_TOKEN.repeat(image_indexes.len())
        )
    }
}

impl MultimodalModelLoader for VLlama4Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn MultimodalModel + Send + Sync>> {
        let mut cfg: crate::vision_models::llama4::Llama4Config = serde_json::from_str(config)?;
        cfg.propagate_quantization_config();
        Ok(Box::new(Llama4Model::new(
            &cfg,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn is_gptx(&self, _config: &str) -> bool {
        false
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let mut cfg: crate::vision_models::llama4::Llama4Config = serde_json::from_str(config)?;
        cfg.propagate_quantization_config();
        Ok(Box::new(cfg))
    }
    fn get_processor(
        &self,
        _model_config: &str,
        processor_config: Option<ProcessorConfig>,
        _preprocessor_config: PreProcessorConfig,
        _max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        Arc::new(Llama4Processor::new(&processor_config.unwrap_or_default()))
    }
    fn supports_paged_attention(&self, _config: &str) -> bool {
        true
    }
    fn supports_encoder_cache(&self, _config: &str) -> bool {
        true
    }
    fn prefixer(&self, _config: &str) -> Arc<dyn MultimodalPromptPrefixer> {
        Arc::new(VLlama4Prefixer)
    }
    fn modalities(&self, _config: &str) -> Result<Modalities> {
        Ok(Modalities {
            input: vec![SupportedModality::Text, SupportedModality::Vision],
            output: vec![SupportedModality::Text],
        })
    }
}

impl IsqModelLoader for VLlama4Loader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"^language_model\.model\.embed_tokens\.weight$")?,
            Regex::new(r"^language_model\.lm_head\.(weight|bias)$")?,
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
            // FF MoE
            Regex::new(r"layers\.(\d+)\.feed_forward\.experts\.gate_up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.feed_forward\.experts\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.feed_forward\.experts\.up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.feed_forward\.experts\.down_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.feed_forward\.router\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.feed_forward\.shared_expert\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.feed_forward\.shared_expert\.up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.feed_forward\.shared_expert\.down_proj\.(weight|bias)$")?,
            // FF MLP
            Regex::new(r"layers\.(\d+)\.feed_forward\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.feed_forward\.up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.feed_forward\.down_proj\.(weight|bias)$")?,
        ])
    }
    fn immediate_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"language_model\.model\.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"language_model\.model\.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"language_model\.model\.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"language_model\.model\.layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // FF MoE
            Regex::new(
                r"language_model\.model\.layers\.(\d+)\.feed_forward\.experts\.(\d+)\.gate_up_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"language_model\.model\.layers\.(\d+)\.feed_forward\.experts\.(\d+)\.gate_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"language_model\.model\.layers\.(\d+)\.feed_forward\.experts\.(\d+)\.up_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"language_model\.model\.layers\.(\d+)\.feed_forward\.experts\.(\d+)\.down_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"language_model\.model\.layers\.(\d+)\.feed_forward\.experts\.(gate_proj|up_proj|down_proj)\.weight$",
            )?,
            Regex::new(
                r"language_model\.model\.layers\.(\d+)\.feed_forward\.router\.(weight|bias)$",
            )?,
            Regex::new(
                r"language_model\.model\.layers\.(\d+)\.feed_forward\.shared_expert\.gate_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"language_model\.model\.layers\.(\d+)\.feed_forward\.shared_expert\.up_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"language_model\.model\.layers\.(\d+)\.feed_forward\.shared_expert\.down_proj\.(weight|bias)$",
            )?,
            // FF MLP
            Regex::new(
                r"language_model\.model\.layers\.(\d+)\.feed_forward\.gate_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"language_model\.model\.layers\.(\d+)\.feed_forward\.up_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"language_model\.model\.layers\.(\d+)\.feed_forward\.down_proj\.(weight|bias)$",
            )?,
        ])
    }
}

impl VLlama4Loader {
    /// This incorporates the max batch size!
    /// Returns (pixels max batch size, num text image tokens)
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn run_dummy_processing(
        &self,
        cfg: &Llama4Config,
        height: usize,
        width: usize,
        max_num_images: usize,
        max_batch_size: usize,
    ) -> Result<(usize, usize)> {
        let cfg = &cfg.vision_config;

        let img_processor =
            Llama4ImageProcessor::new(Some(cfg.patch_size), Some(cfg.pixel_shuffle_ratio));
        let image = DynamicImage::new(width as u32, height as u32, ColorType::Rgb8);
        let res = img_processor.preprocess(
            vec![image; max_num_images],
            vec![],
            &PreProcessorConfig::default(),
            &Device::Cpu,
            (max_batch_size, max_num_images),
        )?;

        let pixels_batch_size = res.pixel_values.dim(0)?;
        let pixels_max_batch_size = pixels_batch_size * max_batch_size;

        let (image_h, image_w) = (
            res.pixel_values.dim(D::Minus2).unwrap(),
            res.pixel_values.dim(D::Minus1).unwrap(),
        );
        let num_patches_per_chunk = (image_h / img_processor.patch_size)
            * (image_w / img_processor.patch_size)
            / img_processor.downsample_ratio;

        Ok((
            pixels_max_batch_size,
            num_patches_per_chunk * pixels_max_batch_size,
        ))
    }
}

impl DeviceMappedModelLoader for VLlama4Loader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Multimodal {
            max_seq_len,
            max_batch_size,
            max_image_shape: (height, width),
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected multimodal AutoDeviceMapParams for this model!")
        };

        let cfg: Llama4Config = serde_json::from_str(config)?;

        let (_pixels_batch_size, num_text_image_toks) =
            self.run_dummy_processing(&cfg, *height, *width, *max_num_images, *max_batch_size)?;

        let max_seq_len = max_seq_len.min(&ATTENTION_CHUNK_SIZE) + num_text_image_toks;

        Ok(max_batch_size * cfg.text_config.num_attention_heads * max_seq_len * max_seq_len)
    }
    fn non_mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Multimodal {
            max_seq_len: _,
            max_batch_size,
            max_image_shape: (height, width),
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected multimodal AutoDeviceMapParams for this model!")
        };

        let cfg: Llama4Config = serde_json::from_str(config)?;

        let (pixels_batch_size, _num_text_image_toks) =
            self.run_dummy_processing(&cfg, *height, *width, *max_num_images, *max_batch_size)?;
        let max_seq_len = cfg.vision_config.num_patches();

        Ok((max_batch_size * pixels_batch_size)
            * cfg.vision_config.num_attention_heads
            * max_seq_len
            * max_seq_len)
    }
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: Llama4Config = serde_json::from_str(config)?;
        let tcfg = &cfg.text_config;

        let text_elems = {
            let (embed_tokens_pack_factor, lm_head_pack_factor) =
                super::language_model_pack_factors(
                    _quantization,
                    "language_model.model.embed_tokens.weight",
                    "language_model.lm_head.weight",
                    tcfg.tie_word_embeddings,
                    dtype,
                    weight_pack_factor,
                )?;
            let embed_tokens = tcfg.hidden_size * tcfg.vocab_size / embed_tokens_pack_factor;
            let lm_head = if !tcfg.tie_word_embeddings {
                tcfg.hidden_size * tcfg.vocab_size / lm_head_pack_factor
            } else {
                0
            };
            let norm = tcfg.hidden_size;
            embed_tokens + lm_head + norm
        };

        let vision_elems = {
            let cfg = &cfg.vision_config;

            let num_patches = cfg.num_patches();

            let unfold_elems =
                (cfg.num_channels * cfg.patch_size * cfg.patch_size) * cfg.hidden_size;
            let class_embeddng_elems = cfg.hidden_size;
            let positional_embedding_vlm_elems = num_patches * cfg.hidden_size;
            let layernorm_pre_elems = cfg.hidden_size;
            let layernorm_post_elems = cfg.hidden_size;

            let pixel_shuffle_elems = cfg.intermediate_size * cfg.projector_input_dim
                + cfg.projector_input_dim * cfg.projector_output_dim;

            let encoder_layer = {
                let input_layernorm = cfg.hidden_size + cfg.hidden_size;
                let post_attention_layernorm = cfg.hidden_size + cfg.hidden_size;

                let head_dim = cfg.hidden_size / cfg.num_attention_heads;
                let q_proj = cfg.hidden_size * cfg.num_attention_heads * head_dim
                    + cfg.num_attention_heads * head_dim;
                let k_proj = cfg.hidden_size * cfg.num_attention_heads * head_dim
                    + cfg.num_attention_heads * head_dim;
                let v_proj = cfg.hidden_size * cfg.num_attention_heads * head_dim
                    + cfg.num_attention_heads * head_dim;
                let o_proj = cfg.hidden_size * cfg.num_attention_heads * head_dim
                    + cfg.num_attention_heads * head_dim;

                let fc1 = cfg.hidden_size * cfg.intermediate_size + cfg.intermediate_size;
                let fc2 = cfg.intermediate_size * cfg.hidden_size + cfg.hidden_size;

                input_layernorm
                    + post_attention_layernorm
                    + q_proj
                    + k_proj
                    + v_proj
                    + o_proj
                    + fc1
                    + fc2
            };

            unfold_elems
                + class_embeddng_elems
                + positional_embedding_vlm_elems
                + layernorm_post_elems
                + layernorm_pre_elems
                + pixel_shuffle_elems
                + encoder_layer * cfg.num_hidden_layers
        };

        let elems = text_elems + vision_elems;
        Ok(elems * dtype.size_in_bytes())
    }
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: Llama4Config = serde_json::from_str(config)?;
        let tcfg = &cfg.text_config;

        let mut per_layer_elems = Vec::new();

        for layer_idx in 0..tcfg.num_hidden_layers {
            let input_layernorm = tcfg.hidden_size;
            let post_attention_layernorm = tcfg.hidden_size;

            let size_in = tcfg.hidden_size;
            let size_q = (tcfg.hidden_size / tcfg.num_attention_heads) * tcfg.num_attention_heads;
            let size_kv = (tcfg.hidden_size / tcfg.num_attention_heads) * tcfg.num_key_value_heads;
            let q_proj = size_in * size_q / weight_pack_factor;
            let k_proj = size_in * size_kv / weight_pack_factor;
            let v_proj = size_in * size_kv / weight_pack_factor;
            let o_proj = size_q * size_in / weight_pack_factor;

            let use_moe = tcfg.moe_layers().contains(&layer_idx);
            let moe_block = if use_moe {
                let h_size = tcfg.hidden_size;
                let i_size = tcfg.intermediate_size;
                let gate_proj = tcfg.num_local_experts * h_size * i_size / weight_pack_factor;
                let up_proj = tcfg.num_local_experts * h_size * i_size / weight_pack_factor;
                let down_proj = tcfg.num_local_experts * i_size * h_size / weight_pack_factor;

                gate_proj + up_proj + down_proj
            } else {
                let h_size = tcfg.hidden_size;
                let i_size = tcfg.intermediate_size_mlp;
                let gate_proj = h_size * i_size / weight_pack_factor;
                let up_proj = h_size * i_size / weight_pack_factor;
                let down_proj = i_size * h_size / weight_pack_factor;
                gate_proj + up_proj + down_proj
            };

            per_layer_elems.push(
                input_layernorm
                    + post_attention_layernorm
                    + q_proj
                    + k_proj
                    + v_proj
                    + o_proj
                    + moe_block,
            );
        }

        Ok(per_layer_elems
            .into_iter()
            .map(|x| x * dtype.size_in_bytes())
            .collect())
    }
    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: Llama4Config = serde_json::from_str(config)?;
        Ok(cfg.text_config.num_hidden_layers)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: Llama4Config = serde_json::from_str(config)?;
        let cfg = &cfg.text_config;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: Some(cfg.attention_chunk_size),
            k_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            v_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::StandardNoFlashInfer,
        };

        Ok(Box::new(cfg))
    }

    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Vision])
    }
}
