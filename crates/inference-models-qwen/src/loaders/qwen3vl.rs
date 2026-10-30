use super::*;
use crate::qwen3_vl::config::TextConfig;
use inference_nn::vision::qwen3_vl::VisionConfig;

/// `MultimodalLoader` for an Qwen3VL model.
pub struct Qwen3VLLoader;

/// `MultimodalLoader` for a Qwen3VLMoE model.
pub struct Qwen3VLMoELoader;

pub struct Qwen3VLPrefixer;

impl MultimodalPromptPrefixer for Qwen3VLPrefixer {
    // No-op: With MessagesAction::Keep, the chat template handles image tokens
    // when it sees {"type": "image"} entries in the content.
}

const LANGUAGE_MODEL: &str = r"^(language_model\.model|model\.language_model)";
const ATTENTION_ISQ: &str = r"\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.(weight|bias)$";
const DENSE_MLP_ISQ: &str = r"\.mlp\.(gate_proj|up_proj|down_proj)\.(weight|bias)$";
// Per-expert and stacked checkpoint layouts
const EXPERT_ISQ: &str = r"\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|bias)$";
const STACKED_EXPERTS_ISQ: &str = r"\.mlp\.experts\.(gate_proj|up_proj|down_proj)\.weight$";
const ROUTER_ISQ: &str = r"\.mlp\.gate\.(weight|bias)$";

fn layer_regexes(patterns: &[&str]) -> Result<Vec<Regex>> {
    let names = patterns
        .iter()
        .map(|pattern| format!(r"{LANGUAGE_MODEL}\.layers\.(\d+){pattern}"))
        .collect::<Vec<_>>();
    isq_regexes(&names)
}

fn isq_layer_regexes(moe: bool) -> Result<Vec<Regex>> {
    let mut regexes = isq_regexes(&[r"lm_head\.(weight|bias)$"])?;
    regexes.extend(match moe {
        true => layer_regexes(&[
            ATTENTION_ISQ,
            DENSE_MLP_ISQ,
            ROUTER_ISQ,
            EXPERT_ISQ,
            STACKED_EXPERTS_ISQ,
        ])?,
        false => layer_regexes(&[ATTENTION_ISQ, DENSE_MLP_ISQ])?,
    });
    Ok(regexes)
}

/// The text attention over the images' merged tokens and a prompt chunk.
pub(super) fn qwen3_vision_mapped_act_elems(
    vision: &VisionConfig,
    num_attention_heads: usize,
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
    // the vision information goes straight into the input embeddings
    let seq_len = vision.merged_image_tokens(*max_image_shape, *max_num_images)
        + max_seq_len.min(&ATTENTION_CHUNK_SIZE);
    Ok(max_batch_size * num_attention_heads * seq_len * seq_len)
}

/// The vision encoder's attention over a batch of images.
pub(super) fn qwen3_vision_non_mapped_act_elems(
    vision: &VisionConfig,
    params: &AutoDeviceMapParams,
) -> Result<usize> {
    let AutoDeviceMapParams::Multimodal {
        max_batch_size,
        max_image_shape,
        max_num_images,
        ..
    } = params
    else {
        anyhow::bail!("Expected multimodal AutoDeviceMapParams for this model!")
    };
    Ok(vision.encoder_attention_elems(*max_image_shape, max_batch_size * max_num_images))
}

fn non_mapped_size_in_bytes(
    cfg: &Qwen3VLConfig,
    dtype: DType,
    weight_pack_factor: usize,
    quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
) -> Result<usize> {
    let tie = cfg.tie_word_embeddings;
    let text = &cfg.text_config;
    let (embed_tokens_pack_factor, lm_head_pack_factor) =
        super::language_model_pack_factors_with_aliases(
            quantization,
            &[
                "language_model.model.embed_tokens.weight",
                "model.language_model.embed_tokens.weight",
            ],
            &["lm_head.weight"],
            tie,
            dtype,
            weight_pack_factor,
        )?;
    let embed_tokens = text.hidden_size * text.vocab_size / embed_tokens_pack_factor;
    let lm_head = if tie {
        0
    } else {
        text.hidden_size * text.vocab_size / lm_head_pack_factor
    };
    let elems = embed_tokens + lm_head + text.hidden_size + cfg.vision_config.weight_elems();
    Ok(elems * dtype.size_in_bytes())
}

fn layer_sizes_in_bytes(cfg: &TextConfig, dtype: DType, weight_pack_factor: usize) -> Vec<usize> {
    let dense = decoder_shape(cfg, cfg.intermediate_size);
    // the attention and norms alone, beside a layer's experts
    let attention = decoder_shape(cfg, 0).elems(weight_pack_factor);
    let experts = cfg.hidden_size * cfg.num_experts
        + cfg.num_experts * 3 * (cfg.hidden_size * cfg.moe_intermediate_size / weight_pack_factor);
    (0..cfg.num_hidden_layers)
        .map(|layer| {
            let is_moe = !cfg.mlp_only_layers.contains(&layer)
                && cfg.num_experts > 0
                && (layer + 1) % cfg.decoder_sparse_step == 0;
            let elems = match is_moe {
                true => attention + experts,
                false => dense.elems(weight_pack_factor),
            };
            elems * dtype.size_in_bytes()
        })
        .collect()
}

fn parse_config(config: &str, moe: bool) -> Result<Qwen3VLConfig> {
    let cfg = Qwen3VLConfig::from_json(config)?;
    cfg.text_config.check_experts(moe)?;
    Ok(cfg)
}

fn decoder_shape(cfg: &TextConfig, intermediate_size: usize) -> DecoderLayerShape {
    DecoderLayerShape {
        hidden_size: cfg.hidden_size,
        num_attention_heads: cfg.num_attention_heads,
        num_key_value_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim,
        qkv_bias: false,
        o_bias: false,
        qk_norm: true,
        norms: 2,
        mlp: MlpShape::Gated { intermediate_size },
    }
}

macro_rules! qwen3vl_loader {
    ($loader:ident, $model:ident, $moe:expr) => {
        impl MultimodalModelLoader for $loader {
            fn load(
                &self,
                config: &str,
                vb: ShardedVarBuilder,
                normal_loading_metadata: NormalLoadingMetadata,
                attention_mechanism: AttentionImplementation,
            ) -> Result<Box<dyn MultimodalModel + Send + Sync>> {
                let cfg = parse_config(config, $moe)?;
                Ok(Box::new($model::new(
                    &cfg,
                    vb,
                    self.is_gptx_for(config, &normal_loading_metadata)?,
                    normal_loading_metadata,
                    attention_mechanism,
                )?))
            }
            fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
                Ok(Box::new(parse_config(config, $moe)?))
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
                Arc::new(Qwen3VLPrefixer)
            }
            fn video_frame_sampling(&self, _config: &str) -> VideoFrameSampling {
                QWEN3_VIDEO_SAMPLING
            }
            fn modalities(&self, _config: &str) -> Result<Modalities> {
                Ok(Modalities {
                    input: vec![
                        SupportedModality::Text,
                        SupportedModality::Vision,
                        SupportedModality::Video,
                    ],
                    output: vec![SupportedModality::Text],
                })
            }
        }

        impl IsqModelLoader for $loader {
            fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
                isq_regexes(&[
                    r"^(language_model\.model|model\.language_model)\.embed_tokens\.weight$",
                    r"^lm_head\.(weight|bias)$",
                ])
            }
            fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
                isq_layer_regexes($moe)
            }
            fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
                self.isq_layer_regexes(config)
            }
            fn isq_layer_regexes_moqe(&self, _config: &str) -> Result<Vec<Regex>> {
                match $moe {
                    true => layer_regexes(&[EXPERT_ISQ, STACKED_EXPERTS_ISQ]),
                    false => Ok(Vec::new()),
                }
            }
            fn immediate_isq_predicates_moqe(&self, config: &str) -> Result<Vec<Regex>> {
                self.isq_layer_regexes_moqe(config)
            }
        }

        impl DeviceMappedModelLoader for $loader {
            fn mapped_max_act_size_elems(
                &self,
                config: &str,
                params: &AutoDeviceMapParams,
            ) -> Result<usize> {
                let cfg = parse_config(config, $moe)?;
                qwen3_vision_mapped_act_elems(
                    &cfg.vision_config,
                    cfg.text_config.num_attention_heads,
                    params,
                )
            }
            fn non_mapped_max_act_size_elems(
                &self,
                config: &str,
                params: &AutoDeviceMapParams,
            ) -> Result<usize> {
                qwen3_vision_non_mapped_act_elems(
                    &parse_config(config, $moe)?.vision_config,
                    params,
                )
            }
            fn non_mapped_size_in_bytes(
                &self,
                config: &str,
                dtype: DType,
                weight_pack_factor: usize,
                quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
                _matformer_config: Option<&MatformerSliceConfig>,
            ) -> Result<usize> {
                non_mapped_size_in_bytes(
                    &parse_config(config, $moe)?,
                    dtype,
                    weight_pack_factor,
                    quantization,
                )
            }
            fn layer_sizes_in_bytes(
                &self,
                config: &str,
                dtype: DType,
                weight_pack_factor: usize,
                _matformer_config: Option<&MatformerSliceConfig>,
            ) -> Result<Vec<usize>> {
                Ok(layer_sizes_in_bytes(
                    &parse_config(config, $moe)?.text_config,
                    dtype,
                    weight_pack_factor,
                ))
            }
            fn num_layers(&self, config: &str) -> Result<usize> {
                Ok(parse_config(config, $moe)?.text_config.num_hidden_layers)
            }
            fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
                let cfg = parse_config(config, $moe)?.text_config;
                Ok(Box::new(
                    decoder_shape(&cfg, cfg.intermediate_size).model_config(
                        cfg.num_hidden_layers,
                        cfg.max_position_embeddings,
                        cfg.decoder_spec(false).sliding_window(),
                    ),
                ))
            }
            fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
                Some(vec![NonMappedSubModel::Vision])
            }
        }
    };
}

qwen3vl_loader!(Qwen3VLLoader, Qwen3VLModel, false);
qwen3vl_loader!(Qwen3VLMoELoader, Qwen3VLMoEModel, true);
