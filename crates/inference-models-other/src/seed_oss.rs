//! Seed-OSS: the shared decoder with q/k/v bias (no o_proj or MLP bias) and an explicit head_dim.
#![allow(clippy::cast_possible_truncation)]

use inference_quant::QuantizedConfig;
use inference_tensor::Result;
use serde::Serialize;

use crate::{
    decoder::{DecoderSpec, RopeKind},
    layers::Activation,
    serde_default_fn,
};

serde_default_fn!(bool, attention_bias_default, true);

#[derive(Debug, Clone, Serialize, inference_json::FromJson)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub hidden_act: Activation,
    #[serde(default = "attention_bias_default")]
    pub attention_bias: bool,
    #[serde(default)]
    pub attention_out_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub quantization_config: Option<QuantizedConfig>,
}

impl Config {
    pub fn decoder_spec(&self) -> Result<DecoderSpec> {
        if self.mlp_bias {
            inference_tensor::bail!("SeedOss mlp_bias=true is not implemented")
        }
        Ok(DecoderSpec {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_heads: self.num_attention_heads,
            num_kv_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
            hidden_act: self.hidden_act,
            rms_norm_eps: self.rms_norm_eps,
            rope: RopeKind::Default {
                theta: self.rope_theta as f32,
            },
            max_position_embeddings: self.max_position_embeddings,
            qkv_bias: self.attention_bias,
            o_bias: self.attention_out_bias,
            layer_windows: vec![None; self.num_hidden_layers],
            tie_word_embeddings: self.tie_word_embeddings,
            quantization_config: self.quantization_config.clone(),
            ..Default::default()
        })
    }
}

pub type Model = crate::decoder::CausalLm;
