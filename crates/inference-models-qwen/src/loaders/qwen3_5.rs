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

// Multimodal checkpoints nest the text stack; text-only ones keep it at `model`
const LANGUAGE_MODEL: &str = r"^(language_model\.model|model\.language_model|model)\.layers\.(\d+)";
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
// Qwen3.8-Flash-Next's hyper-connections, PLE projections and QSA indexer, which ISQ promotes: they run every token
const FLASH_NEXT_ISQ: &[&str] = &[
    r"\.(attn|mlp)_hyper_connection\.(input_mix_weight_down|input_mix_weight_up|block_inject_weight)\.weight$",
    r"\.ple\.(key_proj|value_proj)\.weight$",
    r"\.self_attn\.indexer\.(index_qk_proj|index_q_proj|index_k_proj)\.weight$",
];
const FLASH_NEXT_MIXER_ISQ: &str = r"^(language_model\.model|model\.language_model|model)\.hyper_connection_mixer\.(input_mix_weight_down|input_mix_weight_up)\.weight$";
const MTP_LAYERS: &str = r"^mtp\.layers\.(\d+)";
// Bytes a prompt token takes in an MoE prefill chunk, per hidden or intermediate value of each routed expert and per
// hidden value of the residual stream's temporaries: fit to peak memory over 512-8192 token chunks of two models
// (Qwen3.6-35B-A3B and Qwen3.8-Flash-Next) with offloaded experts
const PROMPT_ROUTED_HIDDEN_BYTES: usize = 6;
const PROMPT_ROUTED_INTERMEDIATE_BYTES: usize = 16;
const PROMPT_RESIDUAL_HIDDEN_BYTES: usize = 40;
// A dense model's prefill bytes a token: the most peak memory a token took over 512-4096 token chunks of Qwen3.5 0.8B,
// 9B and 27B (135, 266 and 397 KB against intermediate sizes 3584, 12288 and 17408), covered by base plus slope
const PROMPT_DENSE_BASE_BYTES: usize = 66 * 1024;
const PROMPT_DENSE_INTERMEDIATE_BYTES: usize = 19;

// Every pattern under both the main stack's and the built-in MTP head's layer prefixes.
fn layer_regexes(patterns: &[&[&str]]) -> Result<Vec<Regex>> {
    let patterns = patterns.iter().flat_map(|group| group.iter());
    let names = patterns
        .flat_map(|pattern| [LANGUAGE_MODEL, MTP_LAYERS].map(|prefix| format!("{prefix}{pattern}")))
        .collect::<Vec<_>>();
    isq_regexes(&names.iter().map(String::as_str).collect::<Vec<_>>())
}

pub(super) fn isq_layer_regexes(moe: bool) -> Result<Vec<Regex>> {
    let mut regexes = isq_regexes(&[r"lm_head\.(weight|bias)$", r"^mtp\.fc\.weight$"])?;
    regexes.extend(if moe {
        layer_regexes(&[ATTENTION_ISQ, EXPERTS_ISQ, SHARED_EXPERT_ISQ])?
    } else {
        layer_regexes(&[ATTENTION_ISQ, DENSE_MLP_ISQ])?
    });
    regexes.extend(flash_next_promoted_regexes()?);
    Ok(regexes)
}

/// The Flash-Next layers ISQ quantizes at the promoted type; they match nothing in other Qwen3.5 checkpoints.
pub(super) fn flash_next_promoted_regexes() -> Result<Vec<Regex>> {
    let mut regexes = layer_regexes(&[FLASH_NEXT_ISQ])?;
    regexes.extend(isq_regexes(&[FLASH_NEXT_MIXER_ISQ])?);
    Ok(regexes)
}

pub(super) fn isq_layer_regexes_moqe(moe: bool) -> Result<Vec<Regex>> {
    if moe {
        layer_regexes(&[EXPERTS_ISQ])
    } else {
        Ok(Vec::new())
    }
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
    // QSA's indexer projections count unpacked, as ISQ promotes them
    let indexer = cfg.qsa().map_or(0, |qsa| {
        cfg.hidden_size * (qsa.n_heads + qsa.kv_heads) * qsa.head_dim + 2 * qsa.head_dim
    });
    projections / weight_pack_factor + cfg.head_dim * 2 + indexer
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
    Ok(layer_norm_elems(cfg) + attention + feed_forward_elems(cfg, weight_pack_factor)?)
}

// the two RMSNorms, or Qwen3.8-Flash-Next's two hyper-connections (grouped norm, low-rank mix, write-back weights)
// the hyper-connections' matrices count unpacked: ISQ promotes them, and a GGUF may keep them wider than the bulk
fn layer_norm_elems(cfg: &TextConfig) -> usize {
    match cfg.hyper_connections() {
        Some(hc) => {
            let width = hc.streams * cfg.hidden_size;
            2 * (width + 2 * width * hc.lowrank + width * hc.streams)
        }
        None => cfg.hidden_size * 2,
    }
}

/// What layer `layer_idx` adds beyond `decoder_layer_elems`: PLE's projections (unpacked, as ISQ promotes them),
/// norms and conv, its table aside (it stays in host memory).
pub(super) fn layer_extra_elems(cfg: &TextConfig, layer_idx: usize) -> Result<usize> {
    let Some(ple) = cfg.ple()?.filter(|ple| ple.layer == layer_idx) else {
        return Ok(0);
    };
    let width = cfg.residual_size();
    let projections = ple.embed_dim * (width + cfg.hidden_size);
    Ok(projections + 3 * width + width * ple.conv_kernel)
}

/// The cache config with QSA's indexer keys appended as auxiliary layers, when the model has an indexer.
pub(super) fn with_qsa_layers(
    cfg: &TextConfig,
    mtp: bool,
    config: impl ModelConfigLike + Send + Sync + 'static,
) -> Box<dyn ModelConfigLike> {
    match cfg.qsa_auxiliary_layers(mtp) {
        Some(layers) => Box::new(crate::paged_attention::AuxiliaryPagedKvCacheConfig::new(
            std::sync::Arc::new(config),
            layers,
        )),
        None => Box::new(config),
    }
}

// The built-in MTP head sits on the non-mapped device: fc, three norms and its full-attention layers.
/// Elements of the token embedding and of the LM head (0 when tied), as the weights will load.
pub(super) fn embedding_and_head_elems(
    text: &TextConfig,
    tied: bool,
    embedding_names: &[&str],
    quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
    dtype: DType,
    weight_pack_factor: usize,
) -> Result<(usize, usize)> {
    let (embed_tokens_pack_factor, lm_head_pack_factor) =
        super::language_model_pack_factors_with_aliases(
            quantization,
            embedding_names,
            &["lm_head.weight"],
            tied,
            dtype,
            weight_pack_factor,
        )?;
    let embed_tokens = text.hidden_size * text.vocab_size / embed_tokens_pack_factor;
    let lm_head = if tied {
        0
    } else {
        text.hidden_size * text.vocab_size / lm_head_pack_factor
    };
    Ok((embed_tokens, lm_head))
}

const MULTIMODAL_EMBEDDING_NAMES: [&str; 2] = [
    "language_model.model.embed_tokens.weight",
    "model.language_model.embed_tokens.weight",
];

pub(super) fn prompt_activation_bytes_per_token(cfg: &TextConfig) -> usize {
    if cfg.num_experts == 0 {
        return PROMPT_DENSE_BASE_BYTES
            + PROMPT_DENSE_INTERMEDIATE_BYTES * cfg.intermediate_size.unwrap_or_default();
    }
    let routed = PROMPT_ROUTED_HIDDEN_BYTES * cfg.hidden_size
        + PROMPT_ROUTED_INTERMEDIATE_BYTES * cfg.moe_intermediate_size;
    cfg.num_experts_per_tok * routed + PROMPT_RESIDUAL_HIDDEN_BYTES * cfg.hidden_size
}

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
                let mut regexes = isq_regexes(&[
                    r"^(language_model\.model|model\.language_model)\.embed_tokens\.weight$",
                    r"^lm_head\.(weight|bias)$",
                ])?;
                regexes.extend(flash_next_promoted_regexes()?);
                Ok(regexes)
            }
            fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
                isq_layer_regexes($moe)
            }
            fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
                self.isq_layer_regexes(config)
            }
            fn isq_layer_regexes_moqe(&self, _config: &str) -> Result<Vec<Regex>> {
                isq_layer_regexes_moqe($moe)
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
                super::qwen3vl::qwen3_vision_mapped_act_elems(
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
                super::qwen3vl::qwen3_vision_non_mapped_act_elems(
                    &parse_config(config, $moe)?.vision_config,
                    params,
                )
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
                    let (embed_tokens, lm_head) = embedding_and_head_elems(
                        &cfg.text_config,
                        tie,
                        &MULTIMODAL_EMBEDDING_NAMES,
                        _quantization,
                        dtype,
                        weight_pack_factor,
                    )?;
                    embed_tokens + lm_head + cfg.text_config.hidden_size
                };

                let mtp_head = if cfg.mtp {
                    mtp_head_elems(&cfg.text_config, weight_pack_factor)?
                } else {
                    0
                };

                let elems = text_elems + mtp_head + cfg.vision_config.weight_elems();

                Ok(elems * dtype.size_in_bytes())
            }
            fn untied_embedding_size_in_bytes(
                &self,
                config: &str,
                dtype: DType,
                weight_pack_factor: usize,
                quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
            ) -> Result<Option<usize>> {
                let cfg = parse_config(config, $moe)?;
                if cfg.tie_word_embeddings {
                    return Ok(None);
                }
                let (embed_tokens, _) = embedding_and_head_elems(
                    &cfg.text_config,
                    false,
                    &MULTIMODAL_EMBEDDING_NAMES,
                    quantization,
                    dtype,
                    weight_pack_factor,
                )?;
                Ok(Some(embed_tokens * dtype.size_in_bytes()))
            }
            fn prompt_activation_bytes_per_token(&self, config: &str) -> Result<Option<usize>> {
                let cfg = parse_config(config, $moe)?;
                Ok(Some(prompt_activation_bytes_per_token(&cfg.text_config)))
            }
            fn non_mapped_layer_sizes_in_bytes(
                &self,
                config: &str,
                dtype: DType,
                weight_pack_factor: usize,
            ) -> Result<Vec<usize>> {
                let cfg = parse_config(config, $moe)?;
                if !cfg.mtp {
                    return Ok(Vec::new());
                }
                let head = mtp_head_elems(&cfg.text_config, weight_pack_factor)?;
                Ok(vec![head * dtype.size_in_bytes()])
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
                    .enumerate()
                    .map(|(layer_idx, layer_type)| {
                        Ok((decoder_layer_elems(&cfg, layer_type, weight_pack_factor)?
                            + layer_extra_elems(&cfg, layer_idx)?)
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
                    max_seq_len: cfg.supported_context_len(true),
                    num_layers: cfg.num_hidden_layers + cfg.mtp_layers(mtp),
                    hidden_size: cfg.hidden_size,
                    num_kv_heads: cfg.num_key_value_heads,
                    num_attn_heads: cfg.num_attention_heads,
                    sliding_window: None,
                    k_head_dim: cfg.head_dim,
                    v_head_dim: cfg.head_dim,
                    kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
                };

                Ok(with_qsa_layers(
                    cfg,
                    mtp,
                    HybridPagedKvCacheConfig::new(base, cfg.paged_kv_layers(mtp))
                        .with_uniform_prefix_prefill_attention_features(Default::default())
                        .with_recurrent_state_bytes(cfg.recurrent_state_bytes()),
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
