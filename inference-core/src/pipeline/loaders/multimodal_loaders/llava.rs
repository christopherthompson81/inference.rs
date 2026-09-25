use super::*;

/// [`MultimodalLoader`] for an LLaVA Vision model.
///
/// [`MultimodalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.MultimodalLoader.html
pub struct LLaVALoader;

pub struct LLaVAPrefixer;

impl MultimodalPromptPrefixer for LLaVAPrefixer {
    fn prefix_image(&self, image_indexes: Vec<usize>, prompt: &str) -> String {
        format!("{}{prompt}", "<image>".repeat(image_indexes.len()))
    }
}

impl MultimodalModelLoader for LLaVALoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn MultimodalModel + Send + Sync>> {
        let cfg: crate::vision_models::llava::config::Config = serde_json::from_str(config)?;
        Ok(Box::new(LLaVA::new(
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
        let cfg: crate::vision_models::llava::config::Config = serde_json::from_str(config)?;
        Ok(Box::new(cfg))
    }
    fn get_processor(
        &self,
        model_config: &str,
        _processor_config: Option<ProcessorConfig>,
        _preprocessor_config: PreProcessorConfig,
        _max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        Arc::new(LLaVAProcessor::new(model_config))
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
        Arc::new(LLaVAPrefixer)
    }
    fn modalities(&self, _config: &str) -> Result<Modalities> {
        Ok(Modalities {
            input: vec![SupportedModality::Text, SupportedModality::Vision],
            output: vec![SupportedModality::Text],
        })
    }
}

impl IsqModelLoader for LLaVALoader {
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

impl DeviceMappedModelLoader for LLaVALoader {
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

        let config: LLaVAConfig = serde_json::from_str(config)?;

        let img_seq_len =
            llava_inputs_processor::LLaVAInputProcessor::get_num_image_tokens(&config);
        let img_seq_len = img_seq_len * max_num_images;

        let max_text_attn = {
            let cfg = &config.text_config;
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
            max_image_shape: _,
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected multimodal AutoDeviceMapParams for this model!")
        };

        let config: LLaVAConfig = serde_json::from_str(config)?;

        let img_seq_len =
            llava_inputs_processor::LLaVAInputProcessor::get_num_image_tokens(&config);

        let max_vision_attn = {
            (max_batch_size * max_num_images)
                * config.vision_config.num_attention_heads
                * img_seq_len
                * img_seq_len
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
        let cfg: LLaVAConfig = serde_json::from_str(config)?;
        let text_elems = {
            let cfg = &cfg.text_config;
            let embed_tokens_pack_factor = super::promoted_tensor_pack_factor(
                _quantization,
                "language_model.model.embed_tokens.weight",
                dtype,
                weight_pack_factor,
            )?;
            let lm_head_pack_factor = super::promoted_tensor_pack_factor(
                _quantization,
                "language_model.lm_head.weight",
                dtype,
                1,
            )?;
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / embed_tokens_pack_factor;
            let lm_head = cfg.hidden_size * cfg.vocab_size / lm_head_pack_factor;
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };

        let image_newline = cfg.text_config.hidden_size;
        let mmproj = {
            let linear_1 = cfg.vision_config.hidden_size * cfg.text_config.hidden_size
                + cfg.text_config.hidden_size;
            let linear_2 = cfg.text_config.hidden_size * cfg.text_config.hidden_size
                + cfg.text_config.hidden_size;

            linear_1 + linear_2
        };
        let vision_tower = get_clip_vit_num_elems(&cfg.to_clip_config());

        let elems = text_elems + image_newline + mmproj + vision_tower;
        Ok(elems * dtype.size_in_bytes())
    }
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: LLaVAConfig = serde_json::from_str(config)?;
        let per_layer_elems = {
            let cfg = &cfg.text_config;
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
            cfg.text_config.num_hidden_layers
        ])
    }
    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: LLaVAConfig = serde_json::from_str(config)?;
        Ok(cfg.text_config.num_hidden_layers)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: LLaVAConfig = serde_json::from_str(config)?;
        let cfg = &cfg.text_config;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: cfg.sliding_window,
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
