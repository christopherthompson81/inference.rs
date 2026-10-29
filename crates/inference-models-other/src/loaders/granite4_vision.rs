use super::*;

use crate::granite4_vision::{Granite4VisionModel, config::Config as Granite4VisionConfig};

const TEXT_LAYER_INDEX_PATTERN: &str = r"model\.language_model\.layers\.(\d+)\.";

/// `MultimodalLoader` for IBM Granite 4 Vision.
pub struct Granite4VisionLoader;

pub struct Granite4VisionPrefixer;

impl MultimodalPromptPrefixer for Granite4VisionPrefixer {
    // the chat template writes `<image>` for each image part
    fn prefix_image(&self, _image_indexes: Vec<usize>, prompt: &str) -> String {
        prompt.to_string()
    }
}

impl Granite4VisionLoader {
    fn text_config(cfg: &Granite4VisionConfig) -> Result<String> {
        Ok(serde_json::to_string(&cfg.text_config)?)
    }

    /// The most image tokens and crops one image can take: the largest pinpoint, filled exactly.
    fn image_budget(cfg: &Granite4VisionConfig) -> Result<(usize, usize)> {
        let tile = cfg.vision_config.image_size;
        cfg.image_grid_pinpoints
            .iter()
            .map(|&(height, width)| {
                let (height, width) = (height as usize, width as usize);
                let tokens = crate::granite4_vision::image_token_count(cfg, (height, width))?;
                Ok((tokens, 1 + height / tile * (width / tile)))
            })
            .try_fold((0, 0), |(tokens, crops), size: Result<(usize, usize)>| {
                let (t, c) = size?;
                Ok((tokens.max(t), crops.max(c)))
            })
    }

    fn projectors(cfg: &Granite4VisionConfig) -> usize {
        cfg.deepstack_layer_map.len()
            + if cfg.use_spatial_sampling {
                cfg.spatial_target_layers.len()
            } else {
                0
            }
    }
}

impl MultimodalModelLoader for Granite4VisionLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn MultimodalModel + Send + Sync>> {
        let cfg = Granite4VisionConfig::from_json(config)?;
        Ok(Box::new(Granite4VisionModel::new(
            &cfg,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        Ok(Box::new(Granite4VisionConfig::from_json(config)?))
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
        Arc::new(Granite4VisionPrefixer)
    }
    fn get_device_for_tensor(
        &self,
        config: &str,
        _mapper: &dyn DeviceMapper,
        loading_isq: bool,
    ) -> Result<Arc<dyn Fn(String) -> DeviceForLoadTensor + Send + Sync + 'static>> {
        let cfg = Granite4VisionConfig::from_json(config)?;
        layer_indexed_device(
            TEXT_LAYER_INDEX_PATTERN,
            cfg.text_config.num_hidden_layers,
            loading_isq,
        )
    }
}

impl IsqModelLoader for Granite4VisionLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"^model\.language_model\.embed_tokens\.weight$",
            r"^lm_head\.(weight|bias)$",
        ])
    }

    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"lm_head\.(weight|bias)$",
            r"(model\.)?language_model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.(weight|bias)$",
            r"(model\.)?language_model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.(weight|bias)$",
            r"(model\.)?language_model\.layers\.(\d+)\.shared_mlp\.(input_linear|output_linear)\.(weight|bias)$",
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for Granite4VisionLoader {
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
        let cfg = Granite4VisionConfig::from_json(config)?;
        let (image_tokens, _) = Self::image_budget(&cfg)?;
        let seq_len = max_seq_len.min(&ATTENTION_CHUNK_SIZE) + image_tokens * max_num_images;
        Ok(max_batch_size * cfg.text_config.num_attention_heads * seq_len * seq_len)
    }
    fn non_mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Multimodal {
            max_batch_size,
            max_num_images,
            ..
        } = params
        else {
            anyhow::bail!("Expected multimodal AutoDeviceMapParams for this model!")
        };
        let cfg = Granite4VisionConfig::from_json(config)?;
        let (_, crops) = Self::image_budget(&cfg)?;
        let vc = &cfg.vision_config;
        let patches = cfg.patches_per_side().pow(2);
        let crops = max_batch_size * max_num_images * crops;
        let attention = crops * vc.num_attention_heads * patches * patches;
        // every hidden state stays alive for the projectors that read it
        let states = crops * patches * vc.hidden_size * (vc.num_hidden_layers + 1);
        Ok(attention.max(crops * patches * vc.intermediate_size) + states)
    }
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg = Granite4VisionConfig::from_json(config)?;
        let tc = &cfg.text_config;
        let (embed_pack, head_pack) = super::language_model_pack_factors(
            quantization,
            "model.language_model.embed_tokens.weight",
            "lm_head.weight",
            tc.tie_word_embeddings,
            dtype,
            weight_pack_factor,
        )?;
        let text = tc.hidden_size * tc.vocab_size / embed_pack
            + if tc.tie_word_embeddings {
                0
            } else {
                tc.hidden_size * tc.vocab_size / head_pack
            }
            + tc.hidden_size;
        let vc = &cfg.vision_config;
        let vision = {
            let embeddings = vc.num_channels * vc.patch_size * vc.patch_size * vc.hidden_size
                + vc.hidden_size
                + cfg.patches_per_side().pow(2) * vc.hidden_size;
            let layer = 4 * (vc.hidden_size * vc.hidden_size + vc.hidden_size)
                + 2 * vc.hidden_size * vc.intermediate_size
                + vc.intermediate_size
                + 5 * vc.hidden_size;
            embeddings + vc.num_hidden_layers * layer + 2 * vc.hidden_size
        };
        let projector = {
            let vision_hidden = vc.hidden_size;
            let q = &cfg.qformer_config;
            let hidden = q.hidden(vision_hidden);
            let encoder = q.encoder_hidden(vision_hidden);
            let (query_side, window_side) = cfg.downsample_sides()?;
            let attention = |context: usize| {
                2 * (hidden * hidden + hidden) + 2 * (context * hidden + hidden) + 2 * hidden
            };
            let layer = attention(hidden)
                + attention(encoder)
                + 2 * hidden * q.intermediate_size
                + q.intermediate_size
                + 3 * hidden;
            2 * vision_hidden
                + (query_side.pow(2) + window_side.pow(2)) * vision_hidden
                + 2 * hidden
                + q.num_hidden_layers * layer
                + hidden * tc.hidden_size
                + tc.hidden_size
        };
        Ok(
            (text + vision + Self::projectors(&cfg) * projector + tc.hidden_size)
                * dtype.size_in_bytes(),
        )
    }
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg = Granite4VisionConfig::from_json(config)?;
        GraniteMoeHybridLoader.layer_sizes_in_bytes(
            &Self::text_config(&cfg)?,
            dtype,
            weight_pack_factor,
            matformer_config,
        )
    }
    fn num_layers(&self, config: &str) -> Result<usize> {
        Ok(Granite4VisionConfig::from_json(config)?
            .text_config
            .num_hidden_layers)
    }
    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Vision])
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = Granite4VisionConfig::from_json(config)?;
        GraniteMoeHybridLoader.model_config(&Self::text_config(&cfg)?)
    }
}
