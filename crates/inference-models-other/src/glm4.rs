//! GLM4: the shared decoder with sandwich norms, partial adjacent-pair RoPE, q/k/v bias and a fused gate/up MLP.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use inference_quant::QuantizedConfig;
use serde::{Deserialize, Serialize};

use crate::{
    decoder::{DecoderSpec, NormNames, RopeKind},
    layers::Activation,
    serde_default_fn,
};

pub const SANDWICH_NORMS: NormNames = NormNames {
    input: "input_layernorm",
    pre_ffn: "post_attention_layernorm",
    post_attn: Some("post_self_attn_layernorm"),
    post_ffn: Some("post_mlp_layernorm"),
};

serde_default_fn!(bool, tie_word_embeddings, false);
serde_default_fn!(usize, max_position_embeddings, 32768);

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub hidden_act: Activation,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub sliding_window: Option<usize>,
    pub partial_rotary_factor: Option<f32>,
    #[serde(default = "max_position_embeddings")]
    pub max_position_embeddings: usize,
    pub attention_bias: Option<bool>,
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

    pub fn decoder_spec(&self) -> DecoderSpec {
        let head_dim = self.head_dim();
        let rotary_dim = self
            .partial_rotary_factor
            .map_or(head_dim, |factor| (factor * head_dim as f32) as usize);
        let bias = self.attention_bias.unwrap_or(false);
        DecoderSpec {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_heads: self.num_attention_heads,
            num_kv_heads: self.num_key_value_heads,
            head_dim,
            hidden_act: self.hidden_act,
            rms_norm_eps: self.rms_norm_eps,
            // GLM pairs adjacent features within the rotated part, whatever the loader reports
            rope: RopeKind::Partial {
                theta: self.rope_theta as f32,
                rotary_dim,
                is_gpt_neox: false,
            },
            max_position_embeddings: self.max_position_embeddings,
            qkv_bias: bias,
            layer_windows: vec![self.sliding_window; self.num_hidden_layers],
            tie_word_embeddings: self.tie_word_embeddings,
            quantization_config: self.quantization_config.clone(),
            norm_names: SANDWICH_NORMS,
            merged_gate_up: true,
            ..Default::default()
        }
    }
}

pub type Model = crate::decoder::CausalLm;
