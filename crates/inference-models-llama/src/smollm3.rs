//! SmolLM3: the Llama decoder with RoPE skipped on every `no_rope_layer_interval`-th layer (NoPE).

use inference_quant::QuantizedConfig;
use serde::{Deserialize, Serialize};

use crate::{
    decoder::{DecoderSpec, RopeKind},
    layers::{Activation, Llama3RopeConfig},
    serde_default_fn,
};

serde_default_fn!(bool, word_emb_default, true);

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub hidden_act: Activation,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub rope_scaling: Option<Llama3RopeConfig>,
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default = "word_emb_default")]
    pub tie_word_embeddings: bool,
    /// Per layer, 1 where the layer applies RoPE and 0 where it does not.
    pub no_rope_layers: Option<Vec<usize>>,
    pub no_rope_layer_interval: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    fn uses_rope(&self, layer_idx: usize) -> bool {
        match &self.no_rope_layers {
            Some(flags) => flags[layer_idx] != 0,
            None => !(layer_idx + 1).is_multiple_of(self.no_rope_layer_interval),
        }
    }

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
            rope: RopeKind::Llama3 {
                theta: self.rope_theta,
                scaling: self.rope_scaling.clone(),
            },
            max_position_embeddings: self.max_position_embeddings,
            qkv_bias: false,
            qk_norm: None,
            no_rope_layers: (0..self.num_hidden_layers)
                .filter(|&layer_idx| !self.uses_rope(layer_idx))
                .collect(),
            attention_temperature: None,
            layer_windows: vec![None; self.num_hidden_layers],
            tie_word_embeddings: self.tie_word_embeddings,
            quantization_config: self.quantization_config.clone(),
        }
    }
}

pub type SmolLm3 = crate::decoder::CausalLm;
