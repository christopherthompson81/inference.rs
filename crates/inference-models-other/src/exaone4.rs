//! EXAONE 4.0: post-norm layers (OLMo 2's), per-head q/k norms before RoPE, and with a sliding window its global
//! layers skip RoPE (NoPE) beside the sliding ones.
#![allow(clippy::cast_possible_truncation)]

use inference_quant::QuantizedConfig;
use inference_tensor::Result;
use serde::Serialize;

use crate::{
    decoder::{DecoderSpec, QkNorm, RopeKind},
    layers::{Activation, Llama3RopeConfig},
    olmo::OLMO2_NORMS,
};

const SLIDING_ATTENTION: &str = "sliding_attention";
const FULL_ATTENTION: &str = "full_attention";

#[derive(Debug, Clone, Serialize, inference_json::FromJson)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: Option<usize>,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub rope_scaling: Option<Llama3RopeConfig>,
    pub hidden_act: Activation,
    pub sliding_window: Option<usize>,
    pub layer_types: Option<Vec<String>>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub quantization_config: Option<QuantizedConfig>,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Per layer, its sliding window; all `None` without one.
    pub fn layer_windows(&self) -> Result<Vec<Option<usize>>> {
        let Some(window) = self.sliding_window else {
            return Ok(vec![None; self.num_hidden_layers]);
        };
        let Some(layer_types) = &self.layer_types else {
            inference_tensor::bail!("EXAONE 4 with a sliding window needs `layer_types`")
        };
        if layer_types.len() != self.num_hidden_layers {
            inference_tensor::bail!(
                "EXAONE 4 lists {} layer types for {} layers",
                layer_types.len(),
                self.num_hidden_layers
            );
        }
        layer_types
            .iter()
            .map(|kind| match kind.as_str() {
                SLIDING_ATTENTION => Ok(Some(window)),
                FULL_ATTENTION => Ok(None),
                other => inference_tensor::bail!("EXAONE 4 layer type `{other}` is not supported"),
            })
            .collect()
    }

    pub fn decoder_spec(&self) -> Result<DecoderSpec> {
        let layer_windows = self.layer_windows()?;
        // transformers: RoPE on the sliding layers, or on every layer of a model without a window
        let no_rope_layers = if self.sliding_window.is_some() {
            (0..self.num_hidden_layers)
                .filter(|&layer| layer_windows[layer].is_none())
                .collect()
        } else {
            Vec::new()
        };
        Ok(DecoderSpec {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_heads: self.num_attention_heads,
            num_kv_heads: self.num_key_value_heads,
            head_dim: self.head_dim(),
            hidden_act: self.hidden_act,
            rms_norm_eps: self.rms_norm_eps,
            rope: RopeKind::Llama3 {
                theta: self.rope_theta,
                scaling: self.rope_scaling.clone(),
            },
            max_position_embeddings: self.max_position_embeddings,
            qk_norm: Some(QkNorm::BEFORE_ROPE),
            no_rope_layers,
            layer_windows,
            norm_names: OLMO2_NORMS,
            tie_word_embeddings: self.tie_word_embeddings,
            quantization_config: self.quantization_config.clone(),
            ..Default::default()
        })
    }
}

pub type Model = crate::decoder::CausalLm;
