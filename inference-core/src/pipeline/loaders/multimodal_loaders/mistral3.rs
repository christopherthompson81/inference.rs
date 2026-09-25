use super::*;

// ======================== Mistral 3 Loader

/// [`MultimodalLoader`] for an Mistral 3 model.
///
/// [`MultimodalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.MultimodalLoader.html
pub struct Mistral3Loader;

pub struct Mistral3Prefixer;

impl MultimodalPromptPrefixer for Mistral3Prefixer {
    fn prefix_image(&self, _image_indexes: Vec<usize>, prompt: &str) -> String {
        prompt.to_string()
    }
}

impl MultimodalModelLoader for Mistral3Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn MultimodalModel + Send + Sync>> {
        let mut cfg: crate::vision_models::mistral3::Mistral3Config = serde_json::from_str(config)?;
        cfg.propagate_quantization_config();
        Ok(Box::new(Mistral3Model::new(
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
        let cfg: crate::vision_models::mistral3::Mistral3Config = serde_json::from_str(config)?;
        Ok(Box::new(cfg))
    }
    fn get_processor(
        &self,
        _model_config: &str,
        processor_config: Option<ProcessorConfig>,
        _preprocessor_config: PreProcessorConfig,
        _max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        Arc::new(Mistral3Processor::new(processor_config.unwrap_or_default()))
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
        Arc::new(Mistral3Prefixer)
    }
    fn modalities(&self, _config: &str) -> Result<Modalities> {
        Ok(Modalities {
            input: vec![SupportedModality::Text, SupportedModality::Vision],
            output: vec![SupportedModality::Text],
        })
    }
}

impl IsqModelLoader for Mistral3Loader {
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
            // MLP
            Regex::new(r"layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
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
            // MLP
            Regex::new(r"language_model\.model\.layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"language_model\.model\.layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"language_model\.model\.layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
        ])
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
impl DeviceMappedModelLoader for Mistral3Loader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let cfg: Mistral3Config = serde_json::from_str(config)?;
        let vcfg = &cfg.vision_config;
        let tcfg = &cfg.text_config;

        let AutoDeviceMapParams::Multimodal {
            max_seq_len,
            max_batch_size,
            max_image_shape: (mut height, mut width),
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected multimodal AutoDeviceMapParams for this model!")
        };

        let img_seq_len = {
            // Reshaping algorithm

            // https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/blob/main/preprocessor_config.json#L29
            let (max_height, max_width) = (1540, 1540);
            let ratio = (height as f64 / max_height as f64).max(width as f64 / max_width as f64);
            if ratio > 1. {
                height = (height as f64 / ratio).floor() as usize;
                width = (width as f64 / ratio).floor() as usize;
            }

            let num_height_tokens = (height - 1) / vcfg.patch_size + 1;
            let num_width_tokens = (width - 1) / vcfg.patch_size + 1;

            height = num_height_tokens * vcfg.patch_size;
            width = num_width_tokens * vcfg.patch_size;

            let num_height_tokens = height / vcfg.patch_size;
            let num_width_tokens = width / vcfg.patch_size;

            (num_width_tokens + 1) * num_height_tokens
        };

        // This model injects the vision information directly into the input embeddings
        let max_seq_len = img_seq_len * max_num_images + *max_seq_len.min(&ATTENTION_CHUNK_SIZE);
        Ok(max_batch_size * tcfg.num_attention_heads * max_seq_len * max_seq_len)
    }

    fn non_mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let cfg: Mistral3Config = serde_json::from_str(config)?;
        let cfg = &cfg.vision_config;

        let AutoDeviceMapParams::Multimodal {
            max_seq_len: _,
            max_batch_size,
            max_image_shape: (mut height, mut width),
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected multimodal AutoDeviceMapParams for this model!")
        };

        let img_seq_len = {
            // Reshaping algorithm

            // https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/blob/main/preprocessor_config.json#L29
            let (max_height, max_width) = (1540, 1540);
            let ratio = (height as f64 / max_height as f64).max(width as f64 / max_width as f64);
            if ratio > 1. {
                height = (height as f64 / ratio).floor() as usize;
                width = (width as f64 / ratio).floor() as usize;
            }

            let num_height_tokens = (height - 1) / cfg.patch_size + 1;
            let num_width_tokens = (width - 1) / cfg.patch_size + 1;

            height = num_height_tokens * cfg.patch_size;
            width = num_width_tokens * cfg.patch_size;

            let num_height_tokens = height / cfg.patch_size;
            let num_width_tokens = width / cfg.patch_size;

            (num_width_tokens + 1) * num_height_tokens
        };

        Ok((max_batch_size * max_num_images) * cfg.num_attention_heads * img_seq_len * img_seq_len)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: Mistral3Config = serde_json::from_str(config)?;

        let text_elems = {
            let cfg = &cfg.text_config;

            let (embed_tokens_pack_factor, lm_head_pack_factor) =
                super::language_model_pack_factors(
                    _quantization,
                    "language_model.model.embed_tokens.weight",
                    "language_model.lm_head.weight",
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

        let vision_elems = {
            let cfg = &cfg.vision_config;

            let patch_embed = {
                let conv_cfg = Conv2dConfig {
                    stride: cfg.patch_size,
                    ..Default::default()
                };
                cfg.num_channels * cfg.hidden_size / conv_cfg.groups
                    * cfg.patch_size
                    * cfg.patch_size
                    * cfg.patch_size
            };
            let ln_pre = cfg.hidden_size;
            let vision_layer = {
                let attn_norm = cfg.hidden_size;
                let ffn_norm = cfg.hidden_size;

                let gate = cfg.hidden_size * cfg.intermediate_size;
                let up = cfg.hidden_size * cfg.intermediate_size;
                let down = cfg.hidden_size * cfg.intermediate_size;

                let q = cfg.hidden_size * cfg.hidden_size;
                let k = cfg.hidden_size * cfg.hidden_size;
                let v = cfg.hidden_size * cfg.hidden_size;
                let o = cfg.hidden_size * cfg.hidden_size;

                attn_norm + ffn_norm + gate + up + down + q + k + v + o
            };

            patch_embed + ln_pre + vision_layer * cfg.num_hidden_layers
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
        let cfg: Mistral3Config = serde_json::from_str(config)?;
        let cfg = &cfg.text_config;

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
        Ok(vec![
            per_layer_elems * dtype.size_in_bytes();
            cfg.num_hidden_layers
        ])
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: Mistral3Config = serde_json::from_str(config)?;
        let cfg = &cfg.text_config;
        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: Mistral3Config = serde_json::from_str(config)?;
        let cfg = &cfg.text_config;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: cfg.sliding_window,
            k_head_dim: cfg.head_dim(),
            v_head_dim: cfg.head_dim(),
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }

    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Vision])
    }
}
