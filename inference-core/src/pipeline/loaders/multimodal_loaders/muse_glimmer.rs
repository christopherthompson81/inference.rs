use super::*;

pub struct MuseGlimmerLoader;

pub struct MuseGlimmerPrefixer;

impl MultimodalPromptPrefixer for MuseGlimmerPrefixer {}

fn muse_glimmer_runtime_config(config: &str, max_model_len: Option<usize>) -> Result<Cow<'_, str>> {
    let Some(max_model_len) = max_model_len else {
        return Ok(Cow::Borrowed(config));
    };
    anyhow::ensure!(max_model_len > 0, "max_model_len must be greater than zero");

    let parsed: MuseGlimmerConfig = serde_json::from_str(config)?;
    if parsed.text_config.max_position_embeddings <= max_model_len {
        return Ok(Cow::Borrowed(config));
    }

    let mut value: serde_json::Value = serde_json::from_str(config)?;
    value
        .get_mut("text_config")
        .and_then(serde_json::Value::as_object_mut)
        .ok_or_else(|| anyhow::anyhow!("Muse-Glimmer config is missing text_config"))?
        .insert(
            "max_position_embeddings".to_string(),
            serde_json::Value::from(max_model_len),
        );
    Ok(Cow::Owned(serde_json::to_string(&value)?))
}

impl MultimodalModelLoader for MuseGlimmerLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn MultimodalModel + Send + Sync>> {
        let cfg: MuseGlimmerConfig = serde_json::from_str(config)?;
        Ok(Box::new(MuseGlimmerModel::new(
            &cfg,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }

    fn runtime_config<'a>(
        &self,
        config: &'a str,
        max_model_len: Option<usize>,
    ) -> Result<Cow<'a, str>> {
        muse_glimmer_runtime_config(config, max_model_len)
    }

    fn is_gptx(&self, _config: &str) -> bool {
        true
    }

    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        Ok(Box::new(serde_json::from_str::<MuseGlimmerConfig>(config)?))
    }

    fn get_processor(
        &self,
        model_config: &str,
        _processor_config: Option<ProcessorConfig>,
        preprocessor_config: PreProcessorConfig,
        max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        let cfg: MuseGlimmerConfig =
            serde_json::from_str(model_config).expect("Failed to parse Muse-Glimmer config");
        Arc::new(
            MuseGlimmerProcessor::new(&preprocessor_config, max_edge, cfg.gguf_collapsed_temporal)
                .expect("Failed to create Muse-Glimmer processor"),
        )
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

    fn modalities(&self, config: &str) -> Result<Modalities> {
        let cfg: MuseGlimmerConfig = serde_json::from_str(config)?;
        let mut input = vec![SupportedModality::Text, SupportedModality::Vision];
        if !cfg.gguf_collapsed_temporal {
            input.push(SupportedModality::Video);
        }
        Ok(Modalities {
            input,
            output: vec![SupportedModality::Text],
        })
    }

    fn prefixer(&self, _config: &str) -> Arc<dyn MultimodalPromptPrefixer> {
        Arc::new(MuseGlimmerPrefixer)
    }

    fn get_device_for_tensor(
        &self,
        config: &str,
        _mapper: &dyn DeviceMapper,
        loading_isq: bool,
    ) -> Result<Arc<dyn Fn(String) -> DeviceForLoadTensor + Send + Sync + 'static>> {
        if loading_isq {
            return Ok(Arc::new(|_| DeviceForLoadTensor::Base));
        }
        let re = Regex::new(r"^model\.language_model\.layers\.(\d+)\.")?;
        let num_layers = serde_json::from_str::<MuseGlimmerConfig>(config)?
            .text_config
            .num_hidden_layers;
        Ok(Arc::new(move |name: String| {
            re.captures(&name)
                .and_then(|captures| captures.get(1))
                .and_then(|index| index.as_str().parse::<usize>().ok())
                .filter(|&index| index < num_layers)
                .map(DeviceForLoadTensor::Idx)
                .unwrap_or(DeviceForLoadTensor::Base)
        }))
    }
}

impl IsqModelLoader for MuseGlimmerLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"^model\.language_model\.embed_tokens\.weight$")?,
            Regex::new(r"^lm_head\.(weight|bias)$")?,
        ])
    }

    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"^lm_head\.(weight|bias)$")?,
            Regex::new(
                r"^model\.language_model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj|gate_proj)\.(weight|bias)$",
            )?,
            Regex::new(
                r"^model\.language_model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.(weight|bias)$",
            )?,
        ])
    }

    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for MuseGlimmerLoader {
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
        let cfg: MuseGlimmerConfig = serde_json::from_str(config)?;
        let vc = &cfg.vision_config;
        let visual_tokens = (max_image_shape.0 / vc.patch_size / vc.merge_size)
            * (max_image_shape.1 / vc.patch_size / vc.merge_size)
            * max_num_images;
        let total_seq_len = *max_seq_len + visual_tokens;
        let query_len = total_seq_len.min(ATTENTION_CHUNK_SIZE);
        Ok(max_batch_size * cfg.text_config.num_attention_heads * query_len * total_seq_len)
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
        let cfg: MuseGlimmerConfig = serde_json::from_str(config)?;
        let vc = &cfg.vision_config;
        let raw_patches = (max_image_shape.0 / vc.patch_size) * (max_image_shape.1 / vc.patch_size);
        let items = max_batch_size * max_num_images;
        let attention = items * vc.num_attention_heads * raw_patches * raw_patches;
        let hidden = items * raw_patches * vc.hidden_size.max(vc.intermediate_size);
        Ok(attention.max(hidden))
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: MuseGlimmerConfig = serde_json::from_str(config)?;
        let tc = &cfg.text_config;
        let (embed_pack_factor, head_pack_factor) =
            super::language_model_pack_factors_with_aliases(
                quantization,
                &["model.language_model.embed_tokens.weight"],
                &["lm_head.weight"],
                tc.tie_word_embeddings,
                dtype,
                weight_pack_factor,
            )?;
        let text = tc.vocab_size * tc.hidden_size / embed_pack_factor
            + if tc.tie_word_embeddings {
                0
            } else {
                tc.vocab_size * tc.hidden_size / head_pack_factor
            }
            + tc.hidden_size;

        let vc = &cfg.vision_config;
        let patch = vc.hidden_size * vc.patch_temporal * 3 * vc.patch_size.pow(2);
        let position = vc.pos_emb_height * vc.pos_emb_width * vc.hidden_size;
        let tower_norms = 4 * vc.hidden_size;
        let vision_layer = 4 * vc.hidden_size.pow(2)
            + 2 * vc.hidden_size * vc.intermediate_size
            + vc.intermediate_size
            + 9 * vc.hidden_size;
        let adapter =
            cfg.out_hidden_size * cfg.projector_hidden_size + cfg.projector_hidden_size.pow(2);
        let projection = cfg.projector_hidden_size * tc.hidden_size;
        let vision = patch
            + position
            + tower_norms
            + vc.num_hidden_layers * vision_layer
            + adapter
            + projection;
        Ok((text + vision) * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: MuseGlimmerConfig = serde_json::from_str(config)?;
        let tc = &cfg.text_config;
        let query_size = tc.num_attention_heads * tc.head_dim;
        let kv_size = tc.num_key_value_heads * tc.head_dim;
        let projections =
            (tc.hidden_size * query_size * 3 + tc.hidden_size * kv_size * 2) / weight_pack_factor;
        let attention_biases = if tc.attention_bias {
            query_size + kv_size * 2 + tc.hidden_size
        } else {
            0
        };
        let mlp = 3 * tc.hidden_size * tc.intermediate_size / weight_pack_factor;
        let norms = 4 * tc.hidden_size;
        let layer = (projections + attention_biases + mlp + norms) * dtype.size_in_bytes();
        Ok(vec![layer; tc.num_hidden_layers])
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        Ok(serde_json::from_str::<MuseGlimmerConfig>(config)?
            .text_config
            .num_hidden_layers)
    }

    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Vision])
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: MuseGlimmerConfig = serde_json::from_str(config)?;
        let tc = cfg.text_config;
        Ok(Box::new(ModelConfigMetadata {
            max_seq_len: tc.max_position_embeddings,
            num_layers: tc.num_hidden_layers,
            hidden_size: tc.hidden_size,
            num_kv_heads: tc.num_key_value_heads,
            num_attn_heads: tc.num_attention_heads,
            sliding_window: None,
            k_head_dim: tc.head_dim,
            v_head_dim: tc.head_dim,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        }))
    }
}
