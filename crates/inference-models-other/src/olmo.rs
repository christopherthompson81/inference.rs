//! AllenAI's OLMo family on the shared decoder: OLMo 1 (weightless LayerNorms, clamped q/k/v), OLMo 2 and Olmo 3
//! (post-norm layers, q/k norms over the whole projection, Olmo 3's sliding layers on plain RoPE beside YaRN full
//! ones) and OLMoE (64 experts, unrenormalized top-k).
#![allow(clippy::cast_possible_truncation)]

use inference_quant::QuantizedConfig;
use serde::Serialize;

use crate::{
    decoder::{DecoderSpec, MoeRouting, MoeSpec, NormKind, NormNames, QkNorm, RopeKind},
    layers::{Activation, YarnRopeConfig},
    moe::ExpertProjNames,
    serde_default_fn,
};

serde_default_fn!(f64, default_rope_theta, 10_000.0);
serde_default_fn!(Activation, default_act, Activation::Silu);
// OLMo 1 hard-codes its non-parametric LayerNorm's eps
const OLMO1_NORM_EPS: f64 = 1e-5;
const SLIDING_ATTENTION: &str = "sliding_attention";
const FULL_ATTENTION: &str = "full_attention";
const YARN: &str = "yarn";
const DEFAULT_ROPE: &str = "default";

pub const OLMO2_NORMS: NormNames = NormNames {
    input: None,
    pre_ffn: None,
    post_attn: Some("post_attention_layernorm"),
    post_ffn: Some("post_feedforward_layernorm"),
    last: "norm",
};
const FULL_WIDTH_QK_NORM: QkNorm = QkNorm::FullWidth {
    q: "q_norm",
    k: "k_norm",
};

/// The fields every OLMo config shares.
#[derive(Debug, Clone, inference_json::FromJson, Serialize)]
pub struct Common {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    #[serde(default = "default_act")]
    pub hidden_act: Activation,
    pub max_position_embeddings: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub attention_bias: bool,
    pub clip_qkv: Option<f64>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub quantization_config: Option<QuantizedConfig>,
}

impl Common {
    fn kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    fn base_spec(&self, norm: NormKind, eps: f64) -> DecoderSpec {
        DecoderSpec {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_heads: self.num_attention_heads,
            num_kv_heads: self.kv_heads(),
            head_dim: self.head_dim(),
            hidden_act: self.hidden_act,
            rms_norm_eps: eps,
            rope: RopeKind::Default {
                theta: self.rope_theta as f32,
            },
            max_position_embeddings: self.max_position_embeddings,
            qkv_bias: self.attention_bias,
            o_bias: self.attention_bias,
            clip_qkv: self.clip_qkv,
            layer_windows: vec![None; self.num_hidden_layers],
            tie_word_embeddings: self.tie_word_embeddings,
            quantization_config: self.quantization_config.clone(),
            norm,
            ..Default::default()
        }
    }
}

/// `OlmoForCausalLM`.
#[derive(Debug, Clone, inference_json::FromJson, Serialize)]
pub struct OlmoConfig {
    #[serde(flatten)]
    pub common: Common,
}

impl OlmoConfig {
    pub fn decoder_spec(&self) -> DecoderSpec {
        self.common
            .base_spec(NormKind::LayerNoAffine, OLMO1_NORM_EPS)
    }
}

/// A YaRN `rope_scaling` (or one layer type's `rope_parameters`).
#[derive(Debug, Clone, inference_json::FromJson, Serialize)]
pub struct RopeParameters {
    #[serde(alias = "type")]
    pub rope_type: Option<String>,
    pub rope_theta: Option<f64>,
    pub factor: Option<f32>,
    pub original_max_position_embeddings: Option<usize>,
    pub beta_fast: Option<f32>,
    pub beta_slow: Option<f32>,
    pub attention_factor: Option<f32>,
}

/// `Olmo2ForCausalLM` and `Olmo3ForCausalLM`.
#[derive(Debug, Clone, inference_json::FromJson, Serialize)]
pub struct Olmo2Config {
    #[serde(flatten)]
    pub common: Common,
    pub rms_norm_eps: f64,
    /// Olmo 3: `sliding_attention` or `full_attention` per layer.
    pub layer_types: Option<Vec<String>>,
    pub sliding_window: Option<usize>,
    pub rope_scaling: Option<RopeParameters>,
}

impl Olmo2Config {
    fn rope(&self, scaling: Option<&RopeParameters>) -> inference_tensor::Result<RopeKind> {
        let c = &self.common;
        let theta = c.rope_theta as f32;
        let Some(scaling) = scaling else {
            return Ok(RopeKind::Default { theta });
        };
        match scaling.rope_type.as_deref().unwrap_or(DEFAULT_ROPE) {
            DEFAULT_ROPE => Ok(RopeKind::Default {
                theta: scaling.rope_theta.map_or(theta, |t| t as f32),
            }),
            YARN => {
                let required = |value: Option<f32>, name: &str| {
                    value.ok_or_else(|| {
                        inference_tensor::Error::msg(format!("OLMo YaRN `{name}` is required"))
                    })
                };
                Ok(RopeKind::Yarn(YarnRopeConfig {
                    base: scaling.rope_theta.map_or(theta, |t| t as f32),
                    head_dim: c.head_dim(),
                    max_position_embeddings: c.max_position_embeddings,
                    original_max_position_embeddings: scaling
                        .original_max_position_embeddings
                        .ok_or_else(|| {
                            inference_tensor::Error::msg(
                                "OLMo YaRN `original_max_position_embeddings` is required",
                            )
                        })?,
                    factor: required(scaling.factor, "factor")?,
                    beta_fast: required(scaling.beta_fast, "beta_fast")?,
                    beta_slow: required(scaling.beta_slow, "beta_slow")?,
                    mscale: 1.0,
                    mscale_all_dim: 0.0,
                    attention_factor: scaling.attention_factor,
                }))
            }
            other => inference_tensor::bail!("OLMo RoPE type `{other}` is not supported"),
        }
    }

    pub fn decoder_spec(&self) -> inference_tensor::Result<DecoderSpec> {
        let c = &self.common;
        let mut spec = c.base_spec(NormKind::Rms, self.rms_norm_eps);
        spec.norm_names = OLMO2_NORMS;
        spec.qk_norm = Some(FULL_WIDTH_QK_NORM);
        spec.rope = self.rope(self.rope_scaling.as_ref())?;
        if let Some(layer_types) = &self.layer_types {
            if layer_types.len() != c.num_hidden_layers {
                inference_tensor::bail!(
                    "Olmo 3 lists {} layer types for {} layers",
                    layer_types.len(),
                    c.num_hidden_layers
                );
            }
            spec.layer_windows = layer_types
                .iter()
                .map(|kind| match kind.as_str() {
                    SLIDING_ATTENTION => Ok(self.sliding_window),
                    FULL_ATTENTION => Ok(None),
                    other => {
                        inference_tensor::bail!("Olmo 3 layer type `{other}` is not supported")
                    }
                })
                .collect::<inference_tensor::Result<_>>()?;
            // transformers gives the sliding layers plain RoPE, the scaling only to the full ones
            if layer_types.iter().any(|kind| kind == SLIDING_ATTENTION) {
                spec.local_rope = Some(RopeKind::Default {
                    theta: c.rope_theta as f32,
                });
            }
        }
        Ok(spec)
    }
}

/// `OlmoeForCausalLM`.
#[derive(Debug, Clone, inference_json::FromJson, Serialize)]
pub struct OlmoeConfig {
    #[serde(flatten)]
    pub common: Common,
    pub rms_norm_eps: f64,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    #[serde(default)]
    pub norm_topk_prob: bool,
}

impl OlmoeConfig {
    pub fn decoder_spec(&self) -> DecoderSpec {
        let mut spec = self.common.base_spec(NormKind::Rms, self.rms_norm_eps);
        spec.qk_norm = Some(FULL_WIDTH_QK_NORM);
        spec.moe = Some(MoeSpec {
            num_experts: self.num_experts,
            intermediate_size: self.common.intermediate_size,
            routing: MoeRouting::TopK {
                k: self.num_experts_per_tok,
                renormalize: self.norm_topk_prob,
            },
            quantized_router: true,
            expert_names: ExpertProjNames::DEFAULT,
            name: "mlp",
            dense_layers: Vec::new(),
            cuda_decode_graphs: false,
        });
        spec
    }
}

pub type Model = crate::decoder::CausalLm;
