use super::*;
use crate::exaone4_5::{Config as Exaone4_5Config, exaone4_5_model};
use crate::qwen2vl::inputs_processor::{
    EXAONE_VISION_END, EXAONE_VISION_START, IMAGE_PAD, VIDEO_PAD,
};

/// `MultimodalLoader` for an EXAONE 4.5 model.
pub struct Exaone4_5Loader;

pub struct Exaone4_5Prefixer;

impl MultimodalPromptPrefixer for Exaone4_5Prefixer {
    fn prefix_image(&self, image_indexes: Vec<usize>, prompt: &str) -> String {
        format!(
            "{}{prompt}",
            format!("{EXAONE_VISION_START}{IMAGE_PAD}{EXAONE_VISION_END}\n")
                .repeat(image_indexes.len())
        )
    }
    fn prefix_video(&self, video_indexes: Vec<usize>, prompt: &str) -> String {
        format!(
            "{}{prompt}",
            format!("{EXAONE_VISION_START}{VIDEO_PAD}{EXAONE_VISION_END}\n")
                .repeat(video_indexes.len())
        )
    }
}

impl MultimodalModelLoader for Exaone4_5Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn MultimodalModel + Send + Sync>> {
        let cfg = Exaone4_5Config::from_json(config)?;
        Ok(Box::new(exaone4_5_model(
            &cfg,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        Ok(Box::new(Exaone4_5Config::from_json(config)?))
    }
    fn supports_paged_attention(&self, _config: &str) -> bool {
        true
    }
    fn supports_encoder_cache(&self, _config: &str) -> bool {
        true
    }
    fn supports_prefix_cacher(&self, _config: &str) -> bool {
        // the shared Qwen-VL inputs processor registers each media span in the cache key
        true
    }
    fn prefixer(&self, _config: &str) -> Arc<dyn MultimodalPromptPrefixer> {
        Arc::new(Exaone4_5Prefixer)
    }
    fn modalities(&self, _config: &str) -> Result<Modalities> {
        Ok(Modalities {
            input: vec![SupportedModality::Text, SupportedModality::Vision],
            output: vec![SupportedModality::Text],
        })
    }
}

impl IsqModelLoader for Exaone4_5Loader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"^model\.language_model\.embed_tokens\.weight$",
            r"^lm_head\.(weight|bias)$",
        ])
    }

    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"lm_head\.(weight|bias)$",
            r"language_model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.(weight|bias)$",
            r"language_model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.(weight|bias)$",
            r"^mtp\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.(weight|bias)$",
            r"^mtp\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.(weight|bias)$",
            r"^mtp\.fc\.weight$",
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

fn vision_seq_len(
    cfg: &Exaone4_5Config,
    max_image_shape: (usize, usize),
    max_num_images: usize,
) -> usize {
    let vision = &cfg.vision_config;
    let grid_t = max_num_images / vision.temporal_patch_size;
    let grid_h = max_image_shape.0 / vision.patch_size;
    let grid_w = max_image_shape.1 / vision.patch_size;
    grid_t * grid_h * grid_w
}

impl DeviceMappedModelLoader for Exaone4_5Loader {
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
        let cfg = Exaone4_5Config::from_json(config)?;
        let img_seq_len = vision_seq_len(&cfg, *max_image_shape, *max_num_images) * max_num_images;
        // the vision embeddings sit in the text sequence
        let max_seq_len = img_seq_len + max_seq_len.min(&ATTENTION_CHUNK_SIZE);
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
            max_image_shape,
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected multimodal AutoDeviceMapParams for this model!")
        };
        let cfg = Exaone4_5Config::from_json(config)?;
        let img_seq_len = vision_seq_len(&cfg, *max_image_shape, *max_num_images);
        Ok((max_batch_size * max_num_images)
            * cfg.vision_config.num_heads
            * img_seq_len
            * img_seq_len)
    }
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg = Exaone4_5Config::from_json(config)?;
        let text = &cfg.text_config;
        let text_elems = {
            let (embed_tokens_pack_factor, lm_head_pack_factor) =
                super::language_model_pack_factors_with_aliases(
                    quantization,
                    &["model.language_model.embed_tokens.weight"],
                    &["lm_head.weight"],
                    cfg.tie_word_embeddings,
                    dtype,
                    weight_pack_factor,
                )?;
            let embed_tokens = text.hidden_size * text.vocab_size / embed_tokens_pack_factor;
            let lm_head = if cfg.tie_word_embeddings {
                0
            } else {
                text.hidden_size * text.vocab_size / lm_head_pack_factor
            };
            embed_tokens + lm_head + text.hidden_size
        };
        let vision = &cfg.vision_config;
        let patch_merger = {
            let hidden_size = vision.hidden_size * vision.spatial_merge_size.pow(2);
            let mlp0 = hidden_size * hidden_size + hidden_size;
            let mlp2 = hidden_size * vision.out_hidden_size + vision.out_hidden_size;
            mlp0 + mlp2 + vision.hidden_size
        };
        let patch_embed = vision.in_chans
            * vision.hidden_size
            * vision.temporal_patch_size
            * vision.patch_size
            * vision.patch_size;
        let block = {
            let head_dim = vision.hidden_size / vision.num_heads;
            let qkv_out = vision.hidden_size
                + 2 * vision.num_key_value_heads.unwrap_or(vision.num_heads) * head_dim;
            let norms = 2 * vision.hidden_size;
            let mlp = 3 * vision.hidden_size * vision.intermediate_size
                + 2 * vision.intermediate_size
                + vision.hidden_size;
            let qkv = vision.hidden_size * qkv_out + qkv_out;
            let out = vision.hidden_size * vision.hidden_size + vision.hidden_size;
            norms + mlp + qkv + out
        };
        // the MTP head runs beside the head: its fc over both halves, three norms and one layer
        let mtp_head = if cfg.mtp {
            2 * text.hidden_size * text.hidden_size / weight_pack_factor
                + 3 * text.hidden_size
                + decoder_shape(&cfg).elems(weight_pack_factor)
        } else {
            0
        };
        let elems = text_elems + patch_merger + patch_embed + block * vision.depth + mtp_head;
        Ok(elems * dtype.size_in_bytes())
    }
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg = Exaone4_5Config::from_json(config)?;
        Ok(decoder_shape(&cfg).layer_sizes_in_bytes(
            cfg.text_config.num_hidden_layers,
            dtype,
            weight_pack_factor,
        ))
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = Exaone4_5Config::from_json(config)?;
        // the MTP head's attention is the paged layer after the stack's
        Ok(Box::new(decoder_shape(&cfg).model_config(
            cfg.text_config.num_hidden_layers + usize::from(cfg.mtp),
            cfg.text_config.max_position_embeddings,
            cfg.text_config.sliding_window,
        )))
    }

    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Vision])
    }
}

fn decoder_shape(cfg: &Exaone4_5Config) -> DecoderLayerShape {
    let text = &cfg.text_config;
    DecoderLayerShape {
        hidden_size: text.hidden_size,
        num_attention_heads: text.num_attention_heads,
        num_key_value_heads: text.num_key_value_heads,
        head_dim: text.head_dim(),
        qkv_bias: false,
        o_bias: false,
        qk_norm: true,
        norms: 2,
        mlp: MlpShape::Gated {
            intermediate_size: text.intermediate_size,
        },
    }
}
