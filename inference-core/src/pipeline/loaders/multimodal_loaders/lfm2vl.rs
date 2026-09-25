use super::*;

// ======================== LFM2-VL loader

/// [`MultimodalLoader`] for an LFM2-VL model.
///
/// [`MultimodalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.MultimodalLoader.html
pub struct Lfm2VlLoader;

pub struct Lfm2VlPrefixer;

impl MultimodalPromptPrefixer for Lfm2VlPrefixer {
    fn prefix_image(&self, _image_indexes: Vec<usize>, prompt: &str) -> String {
        prompt.to_string()
    }
}

impl Lfm2VlLoader {
    fn max_image_seq_len(cfg: &Lfm2VlConfig) -> usize {
        let num_patches = cfg.tile_size / cfg.encoder_patch_size;
        let downsampled_patches = num_patches.div_ceil(cfg.downsample_factor);
        let tokens_per_tile = downsampled_patches * downsampled_patches;
        if cfg.do_image_splitting {
            cfg.max_tiles * tokens_per_tile
                + if cfg.use_thumbnail {
                    cfg.max_image_tokens
                } else {
                    0
                }
        } else {
            cfg.max_image_tokens
        }
    }

    fn max_num_patches(cfg: &Lfm2VlConfig) -> usize {
        let max_thumbnail_image_patches = cfg.max_image_tokens * cfg.downsample_factor.pow(2);
        let tile_size_patches = if cfg.do_image_splitting {
            (cfg.tile_size / cfg.encoder_patch_size).pow(2)
        } else {
            0
        };
        max_thumbnail_image_patches.max(tile_size_patches)
    }

    fn max_crops_per_image(cfg: &Lfm2VlConfig) -> usize {
        if cfg.do_image_splitting {
            cfg.max_tiles + if cfg.use_thumbnail { 1 } else { 0 }
        } else {
            1
        }
    }
}

impl MultimodalModelLoader for Lfm2VlLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn MultimodalModel + Send + Sync>> {
        let cfg: Lfm2VlConfig = serde_json::from_str(config)?;
        Ok(Box::new(Lfm2VlModel::new(
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
        let cfg: Lfm2VlConfig = serde_json::from_str(config)?;
        Ok(Box::new(cfg))
    }

    fn get_processor(
        &self,
        model_config: &str,
        _processor_config: Option<ProcessorConfig>,
        preprocessor_config: PreProcessorConfig,
        _max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        let cfg: Lfm2VlConfig =
            serde_json::from_str(model_config).expect("Failed to parse LFM2-VL config");
        Arc::new(Lfm2VlProcessor::new(&cfg, &preprocessor_config))
    }

    fn supports_paged_attention(&self, _config: &str) -> bool {
        true
    }

    fn supports_prefix_cacher(&self, _config: &str) -> bool {
        true
    }

    fn modalities(&self, _config: &str) -> Result<Modalities> {
        Ok(Modalities {
            input: vec![SupportedModality::Text, SupportedModality::Vision],
            output: vec![SupportedModality::Text],
        })
    }

    fn prefixer(&self, _config: &str) -> Arc<dyn MultimodalPromptPrefixer> {
        Arc::new(Lfm2VlPrefixer)
    }

    fn get_device_for_tensor(
        &self,
        config: &str,
        _mapper: &dyn DeviceMapper,
        loading_isq: bool,
    ) -> Result<Arc<dyn Fn(String) -> DeviceForLoadTensor + Send + Sync + 'static>> {
        if loading_isq {
            Ok(Arc::new(|_| DeviceForLoadTensor::Base))
        } else {
            let re = Regex::new(r"model\.language_model\.layers\.(\d+)\.").unwrap();
            let cfg: Lfm2VlConfig = serde_json::from_str(config)?;
            let num_layers = cfg.text_config.num_hidden_layers;
            Ok(Arc::new(move |name: String| {
                if let Some(captures) = re.captures(&name) {
                    captures
                        .get(1)
                        .and_then(|m| m.as_str().parse::<usize>().ok())
                        .map(|l| l.min(num_layers))
                        .map(DeviceForLoadTensor::Idx)
                        .unwrap_or(DeviceForLoadTensor::Base)
                } else {
                    DeviceForLoadTensor::Base
                }
            }))
        }
    }
}

impl IsqModelLoader for Lfm2VlLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"^model\.language_model\.embed_tokens\.weight$")?,
            Regex::new(r"^lm_head\.(weight|bias)$")?,
        ])
    }

    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            Regex::new(
                r"(model\.)?language_model\.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"(model\.)?language_model\.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"(model\.)?language_model\.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"(model\.)?language_model\.layers\.(\d+)\.self_attn\.out_proj\.(weight|bias)$",
            )?,
            Regex::new(r"(model\.)?language_model\.layers\.(\d+)\.conv\.in_proj\.(weight|bias)$")?,
            Regex::new(r"(model\.)?language_model\.layers\.(\d+)\.conv\.out_proj\.(weight|bias)$")?,
            Regex::new(
                r"(model\.)?language_model\.layers\.(\d+)\.feed_forward\.w1\.(weight|bias)$",
            )?,
            Regex::new(
                r"(model\.)?language_model\.layers\.(\d+)\.feed_forward\.w2\.(weight|bias)$",
            )?,
            Regex::new(
                r"(model\.)?language_model\.layers\.(\d+)\.feed_forward\.w3\.(weight|bias)$",
            )?,
            Regex::new(r"model\.multi_modal_projector\.linear_1\.(weight|bias)$")?,
            Regex::new(r"model\.multi_modal_projector\.linear_2\.(weight|bias)$")?,
        ])
    }

    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for Lfm2VlLoader {
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

        let cfg: Lfm2VlConfig = serde_json::from_str(config)?;
        let seq_len =
            max_seq_len.min(&ATTENTION_CHUNK_SIZE) + Self::max_image_seq_len(&cfg) * max_num_images;
        Ok(max_batch_size * cfg.text_config.num_attention_heads * seq_len * seq_len)
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

        let cfg: Lfm2VlConfig = serde_json::from_str(config)?;
        let max_crops = max_num_images * Self::max_crops_per_image(&cfg);
        let max_patches = Self::max_num_patches(&cfg);
        let max_vision_attn = max_batch_size
            * max_crops
            * cfg.vision_config.num_attention_heads
            * max_patches
            * max_patches;
        let max_vision_hidden = max_batch_size
            * max_crops
            * max_patches
            * cfg
                .vision_config
                .hidden_size
                .max(cfg.vision_config.intermediate_size);
        Ok(max_vision_attn.max(max_vision_hidden))
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: Lfm2VlConfig = serde_json::from_str(config)?;
        let text = {
            let tc = &cfg.text_config;
            let tied = tc.tie_word_embeddings();
            let (embed_tokens_pack_factor, lm_head_pack_factor) =
                super::language_model_pack_factors(
                    _quantization,
                    "model.language_model.embed_tokens.weight",
                    "lm_head.weight",
                    tied,
                    dtype,
                    weight_pack_factor,
                )?;
            let embed_tokens = tc.hidden_size * tc.vocab_size / embed_tokens_pack_factor;
            let lm_head = if tied {
                0
            } else {
                tc.hidden_size * tc.vocab_size / lm_head_pack_factor
            };
            embed_tokens + lm_head + tc.hidden_size
        };
        let vision = {
            let vc = &cfg.vision_config;
            let patch_embedding =
                vc.num_channels * vc.patch_size * vc.patch_size * vc.hidden_size + vc.hidden_size;
            let position_embedding = vc.num_patches * vc.hidden_size;
            let post_layernorm = 2 * vc.hidden_size;
            let layer = {
                let attn = 4 * (vc.hidden_size * vc.hidden_size + vc.hidden_size);
                let mlp = vc.hidden_size * vc.intermediate_size
                    + vc.intermediate_size
                    + vc.intermediate_size * vc.hidden_size
                    + vc.hidden_size;
                let norms = 4 * vc.hidden_size;
                attn + mlp + norms
            };
            patch_embedding + position_embedding + post_layernorm + vc.num_hidden_layers * layer
        };
        let projector = {
            let in_channels = cfg.vision_config.hidden_size * cfg.downsample_factor.pow(2);
            let linears = in_channels * cfg.projector_hidden_size / weight_pack_factor
                + cfg.projector_hidden_size * cfg.text_config.hidden_size / weight_pack_factor;
            let bias = if cfg.projector_bias {
                cfg.projector_hidden_size + cfg.text_config.hidden_size
            } else {
                0
            };
            let norm = if cfg.projector_use_layernorm {
                2 * in_channels
            } else {
                0
            };
            linears + bias + norm
        };
        Ok((text + vision + projector) * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: Lfm2VlConfig = serde_json::from_str(config)?;
        let cfg = cfg.text_config;
        let head_dim = cfg.head_dim();
        let hidden = cfg.hidden_size;
        let intermediate = cfg.intermediate_size();
        let mut sizes = Vec::with_capacity(cfg.num_hidden_layers);

        for layer_type in cfg.layer_types() {
            let operator_norm = hidden;
            let ffn_norm = hidden;
            let feed_forward = 3 * hidden * intermediate / weight_pack_factor;
            let operator = match layer_type {
                crate::models::lfm2::LayerType::Attention => {
                    let q_dim = cfg.num_attention_heads * head_dim;
                    let kv_dim = cfg.num_key_value_heads * head_dim;
                    let projections = (hidden * q_dim + hidden * kv_dim * 2 + q_dim * hidden)
                        / weight_pack_factor;
                    projections + 2 * head_dim
                }
                crate::models::lfm2::LayerType::Conv => {
                    let projections = (hidden * 3 * hidden + hidden * hidden) / weight_pack_factor;
                    let conv = hidden * cfg.conv_l_cache;
                    let bias = if cfg.conv_bias { 5 * hidden } else { 0 };
                    projections + conv + bias
                }
            };

            sizes
                .push((operator_norm + ffn_norm + operator + feed_forward) * dtype.size_in_bytes());
        }

        Ok(sizes)
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: Lfm2VlConfig = serde_json::from_str(config)?;
        Ok(cfg.text_config.num_hidden_layers)
    }

    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Vision])
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: Lfm2VlConfig = serde_json::from_str(config)?;
        let tc = cfg.text_config;
        let head_dim = tc.head_dim();
        Ok(Box::new(ModelConfigMetadata {
            max_seq_len: tc.max_position_embeddings,
            num_layers: tc.num_hidden_layers,
            hidden_size: tc.hidden_size,
            num_kv_heads: tc.num_key_value_heads,
            num_attn_heads: tc.num_attention_heads,
            sliding_window: None,
            k_head_dim: head_dim,
            v_head_dim: head_dim,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        }))
    }
}
