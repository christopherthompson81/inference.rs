use super::*;

use crate::deepseek_ocr::{Config, DeepseekOcrModel};

// the text stack's layers only: CLIP's `vision_model.transformer.layers.N` stays with the tower
const TEXT_LAYER_INDEX_PATTERN: &str = r"^model\.layers\.(\d+)\.";
// the reference's plain format: the prompt's own text after BOS, `<image>` written for each image part
const CHAT_TEMPLATE: &str = include_str!("../deepseek_ocr/chat_template.jinja");

const EXPERT_REGEXES: [&str; 1] =
    [r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|bias)$"];
const LAYER_REGEXES: [&str; 3] = [
    r"^lm_head\.(weight|bias)$",
    r"^model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.(weight|bias)$",
    r"^model\.layers\.(\d+)\.mlp\.(shared_experts\.)?(gate_proj|up_proj|down_proj)\.(weight|bias)$",
];

/// `MultimodalLoader` for DeepSeek-OCR.
pub struct DeepseekOcrLoader;

pub struct DeepseekOcrPrefixer;

// the chat template writes `<image>` for each image part
impl MultimodalPromptPrefixer for DeepseekOcrPrefixer {}

fn shape(cfg: &Config, intermediate_size: usize) -> DecoderLayerShape {
    let text = &cfg.language_config;
    DecoderLayerShape {
        hidden_size: text.hidden_size,
        num_attention_heads: text.num_attention_heads,
        num_key_value_heads: text.num_key_value_heads,
        head_dim: text.hidden_size / text.num_attention_heads,
        qkv_bias: false,
        o_bias: false,
        qk_norm: false,
        norms: 2,
        mlp: MlpShape::Gated { intermediate_size },
    }
}

// one image's tokens and views at most: every tile the grid search allows, plus the global view
fn image_budget(cfg: &Config) -> (usize, usize) {
    let tiles = cfg.max_tiles;
    (cfg.image_token_count((tiles, 1)), tiles + 1)
}

impl MultimodalModelLoader for DeepseekOcrLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn MultimodalModel + Send + Sync>> {
        let cfg = Config::from_json(config)?;
        Ok(Box::new(DeepseekOcrModel::new(
            &cfg,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        Ok(Box::new(Config::from_json(config)?))
    }
    fn supports_paged_attention(&self, _config: &str) -> bool {
        true
    }
    fn supports_prefix_cacher(&self, _config: &str) -> bool {
        true
    }
    fn supports_encoder_cache(&self, _config: &str) -> bool {
        true
    }
    fn modalities(&self, _config: &str) -> Result<Modalities> {
        Ok(Modalities {
            input: vec![SupportedModality::Text, SupportedModality::Vision],
            output: vec![SupportedModality::Text],
        })
    }
    fn prefixer(&self, _config: &str) -> Arc<dyn MultimodalPromptPrefixer> {
        Arc::new(DeepseekOcrPrefixer)
    }
    // the layout (`<|ref|>`, `<|det|>`) and table (`<td>`, ...) tags are the tokenizer's special tokens
    fn output_special_tokens(&self, _config: &str) -> bool {
        true
    }
    fn default_chat_template(&self, _config: &str) -> Option<String> {
        Some(CHAT_TEMPLATE.to_string())
    }
    // llama.cpp's GGUFs embed a bare concatenation, which writes no `<image>` for the image parts
    fn template_overrides_gguf(&self, _config: &str) -> bool {
        true
    }
    fn get_device_for_tensor(
        &self,
        config: &str,
        _mapper: &dyn DeviceMapper,
        loading_isq: bool,
    ) -> Result<Arc<dyn Fn(String) -> DeviceForLoadTensor + Send + Sync + 'static>> {
        let cfg = Config::from_json(config)?;
        layer_indexed_device(
            TEXT_LAYER_INDEX_PATTERN,
            cfg.language_config.num_hidden_layers,
            loading_isq,
        )
    }
}

impl IsqModelLoader for DeepseekOcrLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"^model\.embed_tokens\.weight$",
            r"^lm_head\.(weight|bias)$",
        ])
    }
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        let mut regexes = LAYER_REGEXES.to_vec();
        regexes.extend(EXPERT_REGEXES);
        isq_regexes(&regexes)
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
    fn isq_layer_regexes_moqe(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&EXPERT_REGEXES)
    }
    fn immediate_isq_predicates_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes_moqe(config)
    }
}

impl DeviceMappedModelLoader for DeepseekOcrLoader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Multimodal {
            max_seq_len,
            max_batch_size,
            max_num_images,
            ..
        } = params
        else {
            anyhow::bail!("Expected multimodal AutoDeviceMapParams for this model!")
        };
        let cfg = Config::from_json(config)?;
        let (image_tokens, _) = image_budget(&cfg);
        let seq_len = max_seq_len.min(&ATTENTION_CHUNK_SIZE) + image_tokens * max_num_images;
        Ok(max_batch_size * cfg.language_config.num_attention_heads * seq_len * seq_len)
    }
    fn non_mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Multimodal { max_batch_size, .. } = params else {
            anyhow::bail!("Expected multimodal AutoDeviceMapParams for this model!")
        };
        let cfg = Config::from_json(config)?;
        // SAM's global layers over the global view: one score per head per token pair, images encoded one at a time
        let sam = cfg.sam_config()?;
        let tokens = (sam.image_size / sam.patch_size).pow(2);
        Ok(max_batch_size * sam.num_heads * tokens * tokens)
    }
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg = Config::from_json(config)?;
        let text = &cfg.language_config;
        let ends = standard_non_mapped_size_in_bytes(
            LanguageModelEnds {
                hidden_size: text.hidden_size,
                vocab_size: text.vocab_size,
                tie_word_embeddings: text.tie_word_embeddings,
            },
            quantization,
            dtype,
            weight_pack_factor,
        )?;
        let clip = &cfg.vision_config.width.clip;
        let clip_layer = 4 * clip.width * clip.width
            + 2 * 4 * clip.width * clip.width
            + 4 * clip.width
            + 4 * 4 * clip.width;
        let clip_elems = clip.layers * clip_layer
            + ((clip.image_size / clip.patch_size).pow(2) + 2) * clip.width
            + 2 * clip.width;
        let projector = &cfg.projector_config;
        let projector_elems = (projector.input_dim + 3) * projector.n_embed;
        let tower = cfg.sam_config()?.num_elems() + clip_elems + projector_elems;
        Ok(ends + tower * dtype.size_in_bytes())
    }
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg = Config::from_json(config)?;
        let text = &cfg.language_config;
        let dense = cfg
            .language_config
            .decoder_spec()?
            .moe
            .map(|moe| moe.dense_layers)
            .unwrap_or_default();
        let h = text.hidden_size;
        let gated = |intermediate_size: usize| 3 * h * intermediate_size / weight_pack_factor;
        let attention = shape(&cfg, 0).elems(weight_pack_factor);
        Ok((0..text.num_hidden_layers)
            .map(|layer| {
                let mlp = if dense.contains(&layer) {
                    gated(text.intermediate_size)
                } else {
                    text.n_routed_experts * gated(text.moe_intermediate_size)
                        + gated(text.moe_intermediate_size * text.n_shared_experts)
                        + text.n_routed_experts * h
                };
                (attention + mlp) * dtype.size_in_bytes()
            })
            .collect())
    }
    fn num_layers(&self, config: &str) -> Result<usize> {
        Ok(Config::from_json(config)?.language_config.num_hidden_layers)
    }
    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Vision])
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = Config::from_json(config)?;
        Ok(Box::new(shape(&cfg, 0).model_config(
            cfg.language_config.num_hidden_layers,
            cfg.language_config.max_position_embeddings,
            None,
        )))
    }
}
