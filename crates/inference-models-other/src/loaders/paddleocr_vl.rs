use super::*;

/// `MultimodalLoader` for a PaddleOCR-VL (1.5, 1.6) model.
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
        let cfg = PaddleOcrVlConfig::from_json(config)?;
        Ok(Box::new(PaddleOcrVlModel::new(
            &cfg,
            vb,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let config = PaddleOcrVlConfig::from_json(config)?;
        Ok(Box::new(config))
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
        isq_regexes(&[
            r"^model\.embed_tokens\.weight$",
            r"^lm_head\.(weight|bias)$",
        ])
    }

    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"lm_head\.(weight|bias)$",
            // Attention (ERNIE keys have no `language_model` infix)
            r"model\.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$",
            r"model\.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$",
            r"model\.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$",
            r"model\.layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$",
            // MLP
            r"model\.layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$",
            r"model\.layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$",
            r"model\.layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$",
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

        let cfg = PaddleOcrVlConfig::from_json(config)?;
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

        let cfg = PaddleOcrVlConfig::from_json(config)?;
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
        let cfg = PaddleOcrVlConfig::from_json(config)?;
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
        let tcfg = PaddleOcrVlConfig::from_json(config)?.text_config();
        Ok(text_decoder_shape(&tcfg).layer_sizes_in_bytes(
            tcfg.num_hidden_layers,
            dtype,
            weight_pack_factor,
        ))
    }
    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg = PaddleOcrVlConfig::from_json(config)?;
        Ok(cfg.text_config().num_hidden_layers)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = PaddleOcrVlConfig::from_json(config)?;
        let tcfg = cfg.text_config();
        Ok(Box::new(text_decoder_shape(&tcfg).model_config(
            tcfg.num_hidden_layers,
            cfg.max_position_embeddings,
            None,
        )))
    }

    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Vision])
    }
}

fn text_decoder_shape(tcfg: &crate::paddleocr_vl::config::TextConfig) -> DecoderLayerShape {
    DecoderLayerShape {
        hidden_size: tcfg.hidden_size,
        num_attention_heads: tcfg.num_attention_heads,
        num_key_value_heads: tcfg.num_key_value_heads,
        head_dim: tcfg.head_dim,
        qkv_bias: false,
        o_bias: false,
        qk_norm: false,
        norms: 2,
        mlp: MlpShape::Gated {
            intermediate_size: tcfg.intermediate_size,
        },
    }
}
