//! Qwen3: the shared decoder with per-head q/k norm and layer windows past `max_window_layers`.

use inference_quant::QuantizedConfig;
use serde::{Deserialize, Serialize};

use crate::{
    decoder::{DecoderSpec, QkNorm, RopeKind},
    layers::Activation,
    serde_default_fn,
};

serde_default_fn!(bool, tie_word_embeddings, false);

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
    pub rope_theta: f64,
    pub sliding_window: Option<usize>,
    pub head_dim: Option<usize>,
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default = "tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    pub max_window_layers: usize,
    pub use_sliding_window: bool,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Only layers past `max_window_layers` slide, and only when `use_sliding_window` is set.
    pub fn layer_window(&self, layer_idx: usize) -> Option<usize> {
        self.sliding_window
            .filter(|_| self.use_sliding_window && layer_idx >= self.max_window_layers)
    }

    #[allow(clippy::cast_possible_truncation)]
    pub fn decoder_spec(&self) -> DecoderSpec {
        DecoderSpec {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_heads: self.num_attention_heads,
            num_kv_heads: self.num_key_value_heads,
            head_dim: self.head_dim(),
            hidden_act: self.hidden_act,
            rms_norm_eps: self.rms_norm_eps,
            rope: RopeKind::Default {
                theta: self.rope_theta as f32,
            },
            max_position_embeddings: self.max_position_embeddings,
            qkv_bias: false,
            qk_norm: Some(QkNorm::BeforeRope),
            no_rope_layers: Vec::new(),
            attention_temperature: None,
            layer_windows: (0..self.num_hidden_layers)
                .map(|layer_idx| self.layer_window(layer_idx))
                .collect(),
            tie_word_embeddings: self.tie_word_embeddings,
            quantization_config: self.quantization_config.clone(),
            ..Default::default()
        }
    }
}

pub type Model = crate::decoder::CausalLm;
