use super::*;

/// [`MultimodalLoader`] for a PaddleOCR-VL (1.5, 1.6) model.
///
/// [`MultimodalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.MultimodalLoader.html
pub struct PaddleOcrVlLoader;

pub struct PaddleOcrVlPrefixer;

impl MultimodalPromptPrefixer for PaddleOcrVlPrefixer {
    // No-op: the chat template emits the image tokens itself (MessagesAction::Keep).
}

impl MultimodalModelLoader for PaddleOcrVlLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn MultimodalModel + Send + Sync>> {
        let cfg: PaddleOcrVlConfig = serde_json::from_str(config)?;
        Ok(Box::new(PaddleOcrVlModel::new(
            &cfg,
            vb,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn is_gptx(&self, _config: &str) -> bool {
        true
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let config: PaddleOcrVlConfig = serde_json::from_str(config)?;
        Ok(Box::new(config))
    }
    fn get_processor(
        &self,
        _model_config: &str,
        _processor_config: Option<ProcessorConfig>,
        _preprocessor_config: PreProcessorConfig,
        _max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        Arc::new(PaddleOcrVlProcessor)
    }
    fn supports_paged_attention(&self, _config: &str) -> bool {
        true
    }
    fn supports_prefix_cacher(&self, _config: &str) -> bool {
        // Safe only because the inputs processor hashes the image span into its blocks.
        true
    }
    fn prefixer(&self, _config: &str) -> Arc<dyn MultimodalPromptPrefixer> {
        Arc::new(PaddleOcrVlPrefixer)
    }
    fn modalities(&self, _config: &str) -> Result<Modalities> {
        Ok(Modalities {
            input: vec![SupportedModality::Text, SupportedModality::Vision],
            output: vec![SupportedModality::Text],
        })
    }
}

impl IsqModelLoader for PaddleOcrVlLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"^model\.embed_tokens\.weight$")?,
            Regex::new(r"^lm_head\.(weight|bias)$")?,
        ])
    }

    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention (ERNIE keys have no `language_model` infix)
            Regex::new(r"model\.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"model\.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"model\.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"model\.layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // MLP
            Regex::new(r"model\.layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"model\.layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"model\.layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for PaddleOcrVlLoader {
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

        let cfg: PaddleOcrVlConfig = serde_json::from_str(config)?;
        let tcfg = cfg.text_config();
        let vcfg = cfg.vision_config();

        // Post spatial merge.
        let img_seq_len = {
            let grid_t = 1;
            let grid_h = (max_image_shape.0 / vcfg.patch_size) / vcfg.spatial_merge_size;
            let grid_w = (max_image_shape.1 / vcfg.patch_size) / vcfg.spatial_merge_size;
            grid_t * grid_h * grid_w * max_num_images
        };

        // Vision embeds are scattered into the token stream, so text attends over image + text tokens.
        let max_text_attn = {
            let max_seq_len = img_seq_len + max_seq_len.min(&ATTENTION_CHUNK_SIZE);
            max_batch_size * tcfg.num_attention_heads * max_seq_len * max_seq_len
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

        let cfg: PaddleOcrVlConfig = serde_json::from_str(config)?;
        let vcfg = cfg.vision_config();

        // Vision self-attention runs over the full patch grid, before the spatial merge.
        let img_seq_len = {
            let grid_h = max_image_shape.0 / vcfg.patch_size;
            let grid_w = max_image_shape.1 / vcfg.patch_size;
            grid_h * grid_w
        };

        let max_vision_attn = (max_batch_size * max_num_images)
            * vcfg.num_attention_heads
            * img_seq_len
            * img_seq_len;

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
        let cfg: PaddleOcrVlConfig = serde_json::from_str(config)?;
        let tcfg = cfg.text_config();
        let vcfg = cfg.vision_config();

        let text_elems = {
            let embed_tokens = tcfg.hidden_size * tcfg.vocab_size / weight_pack_factor;
            // tie_word_embeddings=false
            let lm_head = tcfg.hidden_size * tcfg.vocab_size / weight_pack_factor;
            let norm = tcfg.hidden_size;
            embed_tokens + lm_head + norm
        };

        let connector = {
            let merged = vcfg.hidden_size * vcfg.spatial_merge_size.pow(2);
            let pre_norm = vcfg.hidden_size + vcfg.hidden_size; // LayerNorm weight + bias
            let linear_1 = merged * merged + merged;
            let linear_2 = merged * tcfg.hidden_size + tcfg.hidden_size;
            pre_norm + linear_1 + linear_2
        };

        let patch_embed = {
            let weight = vcfg.num_channels * vcfg.hidden_size * vcfg.patch_size * vcfg.patch_size;
            weight + vcfg.hidden_size
        };
        let pos_embed = vcfg.num_positions * vcfg.hidden_size;
        let post_layernorm = vcfg.hidden_size + vcfg.hidden_size;

        let encoder_layer = {
            let norm1 = vcfg.hidden_size + vcfg.hidden_size;
            let norm2 = vcfg.hidden_size + vcfg.hidden_size;
            let attn = 4 * (vcfg.hidden_size * vcfg.hidden_size + vcfg.hidden_size);
            let fc1 = vcfg.hidden_size * vcfg.intermediate_size + vcfg.intermediate_size;
            let fc2 = vcfg.intermediate_size * vcfg.hidden_size + vcfg.hidden_size;
            norm1 + norm2 + attn + fc1 + fc2
        };

        let elems = text_elems
            + connector
            + patch_embed
            + pos_embed
            + post_layernorm
            + encoder_layer * vcfg.num_hidden_layers;

        Ok(elems * dtype.size_in_bytes())
    }
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: PaddleOcrVlConfig = serde_json::from_str(config)?;
        let tcfg = cfg.text_config();
        let per_layer_elems = {
            let input_layernorm = tcfg.hidden_size;
            let post_attention_layernorm = tcfg.hidden_size;

            let size_in = tcfg.hidden_size;
            let size_q = tcfg.head_dim * tcfg.num_attention_heads;
            let size_kv = tcfg.head_dim * tcfg.num_key_value_heads;
            // ERNIE projections are bias-free.
            let q_proj = size_in * size_q / weight_pack_factor;
            let k_proj = size_in * size_kv / weight_pack_factor;
            let v_proj = size_in * size_kv / weight_pack_factor;
            let o_proj = size_q * size_in / weight_pack_factor;

            let h_size = tcfg.hidden_size;
            let i_size = tcfg.intermediate_size;
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
            tcfg.num_hidden_layers
        ])
    }
    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: PaddleOcrVlConfig = serde_json::from_str(config)?;
        Ok(cfg.text_config().num_hidden_layers)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: PaddleOcrVlConfig = serde_json::from_str(config)?;
        let tcfg = cfg.text_config();

        let meta = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: tcfg.num_hidden_layers,
            hidden_size: tcfg.hidden_size,
            num_kv_heads: tcfg.num_key_value_heads,
            num_attn_heads: tcfg.num_attention_heads,
            sliding_window: None,
            k_head_dim: tcfg.head_dim,
            v_head_dim: tcfg.head_dim,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(meta))
    }

    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Vision])
    }
}
