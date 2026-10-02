use super::*;
use crate::qwen3_5::config::{LayerType, TextConfig};

/// `MultimodalLoader` for a Qwen3.5 dense (hybrid GDN + full attention) model.
pub struct Qwen3_5Loader;

/// `MultimodalLoader` for a Qwen3.5 MoE (hybrid GDN + full attention) model.
pub struct Qwen3_5MoeLoader;

pub struct Qwen3_5Prefixer;

impl MultimodalPromptPrefixer for Qwen3_5Prefixer {
    // No-op: With MessagesAction::Keep, the chat template handles image tokens
    // when it sees {"type": "image"} entries in the content.
}

const LANGUAGE_MODEL: &str = r"^(language_model\.model|model\.language_model)\.layers\.(\d+)";
const ATTENTION_ISQ: &[&str] = &[
    // Full attention projections
    r"\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.(weight|bias)$",
    // GDN linear attention projections
    r"\.linear_attn\.(in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)\.(weight|bias)$",
];
const DENSE_MLP_ISQ: &[&str] = &[r"\.mlp\.(gate_proj|up_proj|down_proj)\.(weight|bias)$"];
// Per-expert and stacked (`experts.gate_up_proj`) checkpoint layouts
const EXPERTS_ISQ: &[&str] = &[
    r"\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|bias)$",
    r"\.mlp\.experts\.(gate_up_proj|gate_proj|up_proj|down_proj)\.weight$",
];
const SHARED_EXPERT_ISQ: &[&str] =
    &[r"\.mlp\.shared_expert\.(gate_proj|up_proj|down_proj)\.(weight|bias)$"];
const MTP_LAYERS: &str = r"^mtp\.layers\.(\d+)";

// Every pattern under both the main stack's and the built-in MTP head's layer prefixes.
fn layer_regexes(patterns: &[&[&str]]) -> Result<Vec<Regex>> {
    let patterns = patterns.iter().flat_map(|group| group.iter());
    let names = patterns
        .flat_map(|pattern| [LANGUAGE_MODEL, MTP_LAYERS].map(|prefix| format!("{prefix}{pattern}")))
        .collect::<Vec<_>>();
    isq_regexes(&names.iter().map(String::as_str).collect::<Vec<_>>())
}

fn isq_layer_regexes(moe: bool) -> Result<Vec<Regex>> {
    let mut regexes = isq_regexes(&[r"lm_head\.(weight|bias)$", r"^mtp\.fc\.weight$"])?;
    regexes.extend(if moe {
        layer_regexes(&[ATTENTION_ISQ, EXPERTS_ISQ, SHARED_EXPERT_ISQ])?
    } else {
        layer_regexes(&[ATTENTION_ISQ, DENSE_MLP_ISQ])?
    });
    Ok(regexes)
}

fn parse_config(config: &str, moe: bool) -> Result<Qwen3_5Config> {
    let cfg = Qwen3_5Config::from_json(config)?;
    cfg.text_config.check_experts(moe)?;
    Ok(cfg)
}

fn full_attention_elems(cfg: &TextConfig, weight_pack_factor: usize) -> usize {
    let q_dim = cfg.head_dim * cfg.num_attention_heads;
    let kv_dim = cfg.head_dim * cfg.num_key_value_heads;
    // q_proj carries the output gate, so it is twice q_dim wide
    let projections = cfg.hidden_size * (q_dim * 2 + kv_dim * 2) + q_dim * cfg.hidden_size;
    projections / weight_pack_factor + cfg.head_dim * 2
}

fn linear_attention_elems(cfg: &TextConfig, weight_pack_factor: usize) -> usize {
    let value_dim = cfg.linear_value_dim();
    let conv_dim = cfg.linear_conv_dim();
    let projections = cfg.hidden_size * (conv_dim + value_dim + cfg.linear_num_value_heads * 2)
        + value_dim * cfg.hidden_size;
    // conv1d, dt_bias, A_log and the gated norm over the per-head value dim stay unpacked
    let residual = conv_dim * cfg.linear_conv_kernel_dim
        + cfg.linear_num_value_heads * 2
        + cfg.linear_value_head_dim;
    projections / weight_pack_factor + residual
}

fn feed_forward_elems(cfg: &TextConfig, weight_pack_factor: usize) -> Result<usize> {
    if !cfg.is_moe() {
        return Ok(cfg.hidden_size * cfg.dense_intermediate_size()? * 3 / weight_pack_factor);
    }
    let experts = cfg.hidden_size * cfg.moe_intermediate_size * 3 * cfg.num_experts;
    let shared_expert = cfg.hidden_size * cfg.shared_expert_intermediate_size * 3;
    // router and shared expert gate stay unpacked
    let gates = cfg.hidden_size * cfg.num_experts + cfg.hidden_size;
    Ok((experts + shared_expert) / weight_pack_factor + gates)
}

/// Elements of one decoder layer: both norms, its attention and its feed-forward.
pub(super) fn decoder_layer_elems(
    cfg: &TextConfig,
    layer_type: LayerType,
    weight_pack_factor: usize,
) -> Result<usize> {
    let attention = match layer_type {
        LayerType::FullAttention => full_attention_elems(cfg, weight_pack_factor),
        LayerType::LinearAttention => linear_attention_elems(cfg, weight_pack_factor),
    };
    Ok(cfg.hidden_size * 2 + attention + feed_forward_elems(cfg, weight_pack_factor)?)
}

// The built-in MTP head sits on the non-mapped device: fc, three norms and its full-attention layers.
pub(super) fn mtp_head_elems(cfg: &TextConfig, weight_pack_factor: usize) -> Result<usize> {
    let fc = 2 * cfg.hidden_size * cfg.hidden_size / weight_pack_factor;
    let layer = decoder_layer_elems(cfg, LayerType::FullAttention, weight_pack_factor)?;
    Ok(fc + cfg.hidden_size * 3 + layer * cfg.mtp_num_hidden_layers)
}

macro_rules! qwen3_5_loader {
    ($loader:ident, $moe:expr) => {
        impl MultimodalModelLoader for $loader {
            fn runtime_config<'a>(
                &self,
                config: &'a str,
                max_model_len: Option<usize>,
            ) -> Result<Cow<'a, str>> {
                match max_model_len {
                    Some(max_model_len) => Ok(Cow::Owned(
                        crate::qwen3_5::config::apply_max_model_len(config, max_model_len)?,
                    )),
                    None => Ok(Cow::Borrowed(config)),
                }
            }

            fn load(
                &self,
                config: &str,
                vb: ShardedVarBuilder,
                normal_loading_metadata: NormalLoadingMetadata,
                attention_mechanism: AttentionImplementation,
            ) -> Result<Box<dyn MultimodalModel + Send + Sync>> {
                let cfg = parse_config(config, $moe)?;
                Ok(Box::new(Qwen3_5Model::new(
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
                Arc::new(Qwen3_5Prefixer)
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
                if $moe {
                    layer_regexes(&[EXPERTS_ISQ])
                } else {
                    Ok(Vec::new())
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
                let AutoDeviceMapParams::Multimodal {
                    max_seq_len,
                    max_batch_size,
                    max_image_shape,
                    max_num_images,
                } = params
                else {
                    anyhow::bail!("Expected multimodal AutoDeviceMapParams for this model!")
                };

                let cfg = parse_config(config, $moe)?;

                let img_seq_len = {
                    let cfg = &cfg.vision_config;
                    let grid_t = 1;
                    let grid_h = (max_image_shape.0 / cfg.patch_size) / cfg.spatial_merge_size;
                    let grid_w = (max_image_shape.1 / cfg.patch_size) / cfg.spatial_merge_size;
                    grid_t * grid_h * grid_w * max_num_images
                };

                let max_text_attn = {
                    let cfg = &cfg.text_config;
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
                    max_image_shape,
                    max_num_images,
                } = params
                else {
                    anyhow::bail!("Expected multimodal AutoDeviceMapParams for this model!")
                };

                let cfg = parse_config(config, $moe)?;

                let img_seq_len = {
                    let cfg = &cfg.vision_config;
                    let grid_t = 1;
                    let grid_h = max_image_shape.0 / cfg.patch_size;
                    let grid_w = max_image_shape.1 / cfg.patch_size;
                    grid_t * grid_h * grid_w
                };
                let max_vision_attn = {
                    let cfg = &cfg.vision_config;
                    (max_batch_size * max_num_images) * cfg.num_heads * img_seq_len * img_seq_len
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
                let cfg = parse_config(config, $moe)?;
                let tie = cfg.tie_word_embeddings;
                let text_elems = {
                    let cfg = &cfg.text_config;
                    let (embed_tokens_pack_factor, lm_head_pack_factor) =
                        super::language_model_pack_factors_with_aliases(
                            _quantization,
                            &[
                                "language_model.model.embed_tokens.weight",
                                "model.language_model.embed_tokens.weight",
                            ],
                            &["lm_head.weight"],
                            tie,
                            dtype,
                            weight_pack_factor,
                        )?;
                    let embed_tokens = cfg.hidden_size * cfg.vocab_size / embed_tokens_pack_factor;
                    let lm_head = if !tie {
                        cfg.hidden_size * cfg.vocab_size / lm_head_pack_factor
                    } else {
                        0
                    };
                    let norm = cfg.hidden_size;
                    embed_tokens + lm_head + norm
                };

                let (patch_merger, deepstack_mergers) = {
                    let cfg = &cfg.vision_config;
                    let hidden_size = cfg.hidden_size * cfg.spatial_merge_size.pow(2);

                    let mlp0 = hidden_size * hidden_size + hidden_size;
                    let mlp2 = hidden_size * cfg.out_hidden_size + cfg.out_hidden_size;

                    let ln_q = cfg.hidden_size + bias_if!(true, cfg.hidden_size);
                    let merger = mlp0 + mlp2 + ln_q;

                    let ds_ln = hidden_size + bias_if!(true, hidden_size);
                    let ds_merger = mlp0 + mlp2 + ds_ln;
                    let deepstack = cfg.deepstack_visual_indexes.len() * ds_merger;

                    (merger, deepstack)
                };

                let patch_embed = {
                    let cfg = &cfg.vision_config;
                    let conv_cfg = Conv3dConfig {
                        stride: cfg.patch_size,
                        ..Default::default()
                    };
                    let kernel_sizes = [cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size];
                    let weight = cfg.in_chans * cfg.hidden_size / conv_cfg.groups
                        * kernel_sizes[0]
                        * kernel_sizes[1]
                        * kernel_sizes[2];
                    let bias = cfg.hidden_size;
                    weight + bias
                };

                let pos_embed = {
                    let cfg = &cfg.vision_config;
                    cfg.num_position_embeddings * cfg.hidden_size
                };

                let encoder_layer = {
                    let cfg = &cfg.vision_config;
                    let norm1 = cfg.hidden_size + bias_if!(true, cfg.hidden_size);
                    let norm2 = cfg.hidden_size + bias_if!(true, cfg.hidden_size);

                    let fc1 = cfg.hidden_size * cfg.intermediate_size + cfg.intermediate_size;
                    let fc2 = cfg.hidden_size * cfg.intermediate_size + cfg.hidden_size;

                    let qkv = cfg.hidden_size * cfg.hidden_size * 3 + cfg.hidden_size * 3;
                    let out = cfg.hidden_size * cfg.hidden_size + cfg.hidden_size;

                    norm1 + norm2 + fc1 + fc2 + qkv + out
                };

                let mtp_head = if cfg.mtp {
                    mtp_head_elems(&cfg.text_config, weight_pack_factor)?
                } else {
                    0
                };

                let elems = text_elems
                    + mtp_head
                    + patch_merger
                    + deepstack_mergers
                    + patch_embed
                    + pos_embed
                    + encoder_layer * cfg.vision_config.depth;

                Ok(elems * dtype.size_in_bytes())
            }
            fn layer_sizes_in_bytes(
                &self,
                config: &str,
                dtype: DType,
                weight_pack_factor: usize,
                _matformer_config: Option<&MatformerSliceConfig>,
            ) -> Result<Vec<usize>> {
                let cfg = parse_config(config, $moe)?.text_config;
                cfg.layer_types()
                    .into_iter()
                    .map(|layer_type| {
                        Ok(decoder_layer_elems(&cfg, layer_type, weight_pack_factor)?
                            * dtype.size_in_bytes())
                    })
                    .collect()
            }
            fn num_layers(&self, config: &str) -> Result<usize> {
                let cfg = parse_config(config, $moe)?;
                Ok(cfg.text_config.num_hidden_layers)
            }
            fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
                let cfg = parse_config(config, $moe)?;
                let mtp = cfg.mtp;
                let cfg = &cfg.text_config;

                let base = ModelConfigMetadata {
                    max_seq_len: cfg.max_position_embeddings,
                    num_layers: cfg.num_hidden_layers + cfg.mtp_layers(mtp),
                    hidden_size: cfg.hidden_size,
                    num_kv_heads: cfg.num_key_value_heads,
                    num_attn_heads: cfg.num_attention_heads,
                    sliding_window: None,
                    k_head_dim: cfg.head_dim,
                    v_head_dim: cfg.head_dim,
                    kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
                };

                Ok(Box::new(
                    HybridPagedKvCacheConfig::new(base, cfg.paged_kv_layers(mtp))
                        .with_uniform_prefix_prefill_attention_features(Default::default()),
                ))
            }

            fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
                Some(vec![NonMappedSubModel::Vision])
            }
        }
    };
}

qwen3_5_loader!(Qwen3_5Loader, false);
qwen3_5_loader!(Qwen3_5MoeLoader, true);
