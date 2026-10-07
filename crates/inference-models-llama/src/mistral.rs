//! Mistral: the shared decoder with a uniform sliding window, optional YaRN RoPE and attention temperature.
#![allow(clippy::cast_possible_truncation)]

use inference_quant::QuantizedConfig;
use inference_tensor::Result;
use serde::{Deserialize, Serialize};

use crate::{
    decoder::{AttentionTemperature, DecoderSpec, RopeKind},
    layers::{Activation, YarnRopeConfig},
    serde_default_fn,
};

serde_default_fn!(bool, tie_word_embeddings, false);
serde_default_fn!(f64, default_rope_theta, 10000.0);

/// RoPE type for Mistral models
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum MistralRopeType {
    #[default]
    #[serde(rename = "default")]
    Default,
    #[serde(rename = "yarn")]
    Yarn,
}

/// RoPE parameters for Mistral models, supporting YARN scaling
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MistralRopeParameters {
    pub rope_theta: f64,
    #[serde(default)]
    pub rope_type: MistralRopeType,
    // YARN parameters (optional)
    pub factor: Option<f32>,
    pub beta_fast: Option<f32>,
    pub beta_slow: Option<f32>,
    pub mscale: Option<f32>,
    pub mscale_all_dim: Option<f32>,
    pub original_max_position_embeddings: Option<usize>,
    #[serde(default)]
    pub llama_4_scaling_beta: Option<f32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub hidden_act: Activation,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    // Support both flat rope_theta and nested rope_parameters
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub rope_parameters: Option<MistralRopeParameters>,
    pub sliding_window: Option<usize>,
    pub head_dim: Option<usize>,
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default = "tie_word_embeddings")]
    pub tie_word_embeddings: bool,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Get rope_theta from either flat field or rope_parameters
    pub fn get_rope_theta(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .map(|p| p.rope_theta)
            .unwrap_or(self.rope_theta)
    }

    fn attention_temperature(&self) -> Option<AttentionTemperature> {
        let rope = self.rope_parameters.as_ref()?;
        Some(AttentionTemperature {
            scale: rope.llama_4_scaling_beta?,
            floor_scale: rope.original_max_position_embeddings?,
        })
    }

    fn rope(&self) -> Result<RopeKind> {
        let yarn = match self.rope_parameters.as_ref() {
            Some(rope) if matches!(rope.rope_type, MistralRopeType::Yarn) => rope,
            _ => {
                return Ok(RopeKind::Default {
                    theta: self.get_rope_theta() as f32,
                });
            }
        };
        let required = |value: Option<f32>, name: &str| {
            value.ok_or_else(|| inference_tensor::Error::msg(format!("YARN {name} is required")))
        };
        Ok(RopeKind::Yarn(YarnRopeConfig {
            base: yarn.rope_theta as f32,
            head_dim: self.head_dim(),
            max_position_embeddings: self.max_position_embeddings,
            original_max_position_embeddings: yarn.original_max_position_embeddings.ok_or_else(
                || inference_tensor::Error::msg("YARN original context length is required"),
            )?,
            factor: required(yarn.factor, "factor")?,
            beta_fast: required(yarn.beta_fast, "beta_fast")?,
            beta_slow: required(yarn.beta_slow, "beta_slow")?,
            mscale: yarn.mscale.unwrap_or(1.),
            mscale_all_dim: yarn.mscale_all_dim.unwrap_or(0.),
            attention_factor: None,
        }))
    }

    pub fn decoder_spec(&self) -> Result<DecoderSpec> {
        Ok(DecoderSpec {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_heads: self.num_attention_heads,
            num_kv_heads: self.num_key_value_heads,
            head_dim: self.head_dim(),
            hidden_act: self.hidden_act,
            rms_norm_eps: self.rms_norm_eps,
            rope: self.rope()?,
            max_position_embeddings: self.max_position_embeddings,
            qkv_bias: false,
            qk_norm: None,
            no_rope_layers: Vec::new(),
            attention_temperature: self.attention_temperature(),
            layer_windows: vec![self.sliding_window; self.num_hidden_layers],
            tie_word_embeddings: self.tie_word_embeddings,
            quantization_config: self.quantization_config.clone(),
        })
    }
}

pub type Model = crate::decoder::CausalLm;
