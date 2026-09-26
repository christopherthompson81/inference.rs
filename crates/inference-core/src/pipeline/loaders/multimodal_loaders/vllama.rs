use super::*;

/// [`MultimodalLoader`] for an Llama Vision model.
///
/// [`MultimodalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.MultimodalLoader.html
pub struct VLlamaLoader;

pub struct VLlamaPrefixer;

impl MultimodalPromptPrefixer for VLlamaPrefixer {
    fn prefix_image(&self, image_indexes: Vec<usize>, prompt: &str) -> String {
        format!("{}{prompt}", "<|image|>".repeat(image_indexes.len()))
    }
}

impl MultimodalModelLoader for VLlamaLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn MultimodalModel + Send + Sync>> {
        let cfg: crate::vision_models::mllama::MLlamaConfig = serde_json::from_str(config)?;
        Ok(Box::new(MLlamaModel::new(
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
        let cfg: crate::vision_models::mllama::MLlamaConfig = serde_json::from_str(config)?;
        Ok(Box::new(cfg))
    }
    fn get_processor(
        &self,
        _model_config: &str,
        _processor_config: Option<ProcessorConfig>,
        _preprocessor_config: PreProcessorConfig,
        _max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        Arc::new(MLlamaProcessor::new())
    }
    fn supports_paged_attention(&self, _config: &str) -> bool {
        true
    }
    fn supports_encoder_cache(&self, _config: &str) -> bool {
        true
    }
    fn supports_prefix_cacher(&self, _config: &str) -> bool {
        false
    }
    fn prefixer(&self, _config: &str) -> Arc<dyn MultimodalPromptPrefixer> {
        Arc::new(VLlamaPrefixer)
    }
    fn modalities(&self, _config: &str) -> Result<Modalities> {
        Ok(Modalities {
            input: vec![SupportedModality::Text, SupportedModality::Vision],
            output: vec![SupportedModality::Text],
        })
    }
}

impl IsqModelLoader for VLlamaLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"^language_model\.model\.embed_tokens\.weight$")?,
            Regex::new(r"^language_model\.lm_head\.(weight|bias)$")?,
        ])
    }

    fn isq_layer_regexes(&self, config: &str) -> Result<Vec<Regex>> {
        let config: MLlamaConfig = serde_json::from_str(config)?;
        let cross_attn_layers = &config.text_config.cross_attention_layers;
        let transformer_layers =
            (0..config.text_config.num_hidden_layers).filter(|i| !cross_attn_layers.contains(i));
        let mut text_regexes = Vec::new();
        for layer in transformer_layers {
            text_regexes.extend(vec![
                // Attention text
                Regex::new(&format!(
                    r"language_model.model.layers\.{layer}\.self_attn\.q_proj\.(weight|bias)$"
                ))?,
                Regex::new(&format!(
                    r"language_model.model.layers\.{layer}\.self_attn\.k_proj\.(weight|bias)$"
                ))?,
                Regex::new(&format!(
                    r"language_model.model.layers\.{layer}\.self_attn\.v_proj\.(weight|bias)$"
                ))?,
                Regex::new(&format!(
                    r"language_model.model.layers\.{layer}\.self_attn\.o_proj\.(weight|bias)$"
                ))?,
                // MLP text
                Regex::new(&format!(
                    r"language_model.model.layers\.{layer}\.mlp\.gate_proj\.(weight|bias)$"
                ))?,
                Regex::new(&format!(
                    r"language_model.model.layers\.{layer}\.mlp\.up_proj\.(weight|bias)$"
                ))?,
                Regex::new(&format!(
                    r"language_model.model.layers\.{layer}\.mlp\.down_proj\.(weight|bias)$"
                ))?,
            ]);
        }
        let vision_regexes = vec![
            // Vision attention (transformer)
            Regex::new(
                r"vision_model.transformer.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"vision_model.transformer.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"vision_model.transformer.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"vision_model.transformer.layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$",
            )?,
            // Vision attention (global transforemr)
            Regex::new(
                r"vision_model.global_transformer.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"vision_model.global_transformer.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"vision_model.global_transformer.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"vision_model.global_transformer.layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$",
            )?,
            // MLP vision
            Regex::new(r"layers\.(\d+)\.mlp\.fc1\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.fc2\.(weight|bias)$")?,
        ];

        Ok([text_regexes, vision_regexes].concat())
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for VLlamaLoader {
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

        let config: MLlamaConfig = serde_json::from_str(config)?;

        let img_seq_len = {
            let cfg = &config.vision_config;
            let num_patches = (cfg.image_size / cfg.patch_size).pow(2) + 1;
            let num_padding_patches = (8 - (num_patches as isize % 8)) % 8;
            cfg.max_num_tiles * (num_patches as isize + num_padding_patches) as usize
        };
        let img_seq_len = img_seq_len * max_num_images;

        let max_cross_text_attn = {
            let cfg = &config.text_config;
            max_batch_size * cfg.num_attention_heads * img_seq_len * img_seq_len
        };

        let max_self_text_attn = {
            let cfg = &config.text_config;
            max_batch_size * cfg.num_attention_heads * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2)
        };

        Ok(max_self_text_attn.max(max_cross_text_attn))
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

        let config: MLlamaConfig = serde_json::from_str(config)?;

        let img_seq_len = {
            let cfg = &config.vision_config;
            let num_patches = (cfg.image_size / cfg.patch_size).pow(2) + 1;
            let num_padding_patches = (8 - (num_patches as isize % 8)) % 8;
            cfg.max_num_tiles * (num_patches as isize + num_padding_patches) as usize
        };
        let max_vision_attn = {
            let cfg = &config.vision_config;
            (max_batch_size * max_num_images) * cfg.num_attention_heads * img_seq_len * img_seq_len
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
        let config: MLlamaConfig = serde_json::from_str(config)?;
        let text_elems = {
            let cfg = &config.text_config;
            let (embed_tokens_pack_factor, lm_head_pack_factor) =
                super::language_model_pack_factors(
                    _quantization,
                    "language_model.model.embed_tokens.weight",
                    "language_model.lm_head.weight",
                    cfg.tie_word_embeddings,
                    dtype,
                    weight_pack_factor,
                )?;
            let embed_tokens = cfg.hidden_size * (cfg.vocab_size + 8) / embed_tokens_pack_factor;
            let lm_head = if !cfg.tie_word_embeddings {
                cfg.hidden_size * cfg.vocab_size / lm_head_pack_factor
            } else {
                0
            };
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };

        let vision_elems = {
            let cfg = &config.vision_config;

            let conv_cfg = Conv2dConfig {
                stride: cfg.patch_size,
                ..Default::default()
            };
            let patch_embedding = cfg.num_channels * cfg.hidden_size / conv_cfg.groups
                * cfg.patch_size
                * cfg.patch_size;

            let class_embedding = cfg.hidden_size;

            let gated_positional_embedding = {
                let num_patches = (cfg.image_size / cfg.patch_size).pow(2) + 1;
                let embedding = num_patches * cfg.hidden_size;
                let tile_embedding = (cfg.max_aspect_ratio_id() + 1)
                    * (cfg.max_num_tiles * num_patches * cfg.hidden_size);

                embedding + tile_embedding
            };

            let pre_tile_positional_embedding =
                (cfg.max_aspect_ratio_id() + 1) * (cfg.max_num_tiles * cfg.hidden_size);
            let post_tile_positional_embedding =
                (cfg.max_aspect_ratio_id() + 1) * (cfg.max_num_tiles * cfg.hidden_size);

            let layernorm_pre = cfg.hidden_size;
            let layernorm_post = cfg.hidden_size;

            let encoder_layer = {
                let input_layernorm = cfg.hidden_size + cfg.hidden_size;
                let post_attention_layernorm = cfg.hidden_size + cfg.hidden_size;

                let head_dim = cfg.hidden_size / cfg.num_attention_heads;
                let q_proj =
                    cfg.hidden_size * cfg.num_attention_heads * head_dim / weight_pack_factor;
                let k_proj =
                    cfg.hidden_size * cfg.num_attention_heads * head_dim / weight_pack_factor;
                let v_proj =
                    cfg.hidden_size * cfg.num_attention_heads * head_dim / weight_pack_factor;
                let o_proj =
                    cfg.hidden_size * cfg.num_attention_heads * head_dim / weight_pack_factor;

                let fc1 = (cfg.hidden_size * cfg.intermediate_size) / weight_pack_factor
                    + cfg.intermediate_size;
                let fc2 = (cfg.intermediate_size * cfg.hidden_size) / weight_pack_factor
                    + cfg.hidden_size;

                input_layernorm
                    + post_attention_layernorm
                    + q_proj
                    + k_proj
                    + v_proj
                    + o_proj
                    + fc1
                    + fc2
            };

            patch_embedding
                + class_embedding
                + gated_positional_embedding
                + pre_tile_positional_embedding
                + post_tile_positional_embedding
                + layernorm_pre
                + layernorm_post
                + encoder_layer * (cfg.num_hidden_layers + cfg.num_global_layers)
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
        let config: MLlamaConfig = serde_json::from_str(config)?;
        let cfg = &config.text_config;

        let mut layer_sizes = Vec::new();

        for i in 0..cfg.num_hidden_layers {
            let weight_pack_factor = if cfg.cross_attention_layers.contains(&i) {
                // No isq for cross attention
                1
            } else {
                weight_pack_factor
            };

            let per_layer_elems = {
                let input_layernorm = cfg.hidden_size;
                let post_attention_layernorm = cfg.hidden_size;

                let size_in = cfg.hidden_size;
                let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
                let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
                let q_proj = size_in * size_q / weight_pack_factor;
                let k_proj = size_in * size_kv / weight_pack_factor;
                let v_proj = size_in * size_kv / weight_pack_factor;
                let o_proj = size_q * size_in / weight_pack_factor;

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

            layer_sizes.push(per_layer_elems * dtype.size_in_bytes());
        }

        Ok(layer_sizes)
    }
    fn num_layers(&self, config: &str) -> Result<usize> {
        let config: MLlamaConfig = serde_json::from_str(config)?;
        Ok(config.text_config.num_hidden_layers)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: MLlamaConfig = serde_json::from_str(config)?;
        let cfg = &cfg.text_config;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: None,
            k_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            v_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }

    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Vision])
    }
}
