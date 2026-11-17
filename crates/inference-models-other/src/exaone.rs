//! EXAONE 3.0 / 3.5: a Llama decoder (llama3 RoPE scaling) under its checkpoint's names, `transformer.h.N.attn.attention`.
#![allow(clippy::cast_possible_truncation)]

use inference_quant::QuantizedConfig;
use inference_tensor::Result;
use serde::Serialize;

use crate::{
    decoder::{DecoderSpec, ModuleNames, NormNames, RopeKind},
    layers::{Activation, Llama3RopeConfig},
    moe::ExpertProjNames,
};

const O_PROJ: &str = "out_proj";
pub const NORM_NAMES: NormNames = NormNames {
    input: Some("ln_1"),
    pre_ffn: Some("ln_2"),
    post_attn: None,
    post_ffn: None,
    last: "ln_f",
};
pub const MODULE_NAMES: ModuleNames = ModuleNames {
    model: "transformer",
    embed_tokens: "wte",
    layers: "h",
    self_attn: "attn.attention",
    qkv_proj: "qkv_proj",
    mlp: ExpertProjNames {
        gate: "c_fc_0",
        up: "c_fc_1",
        down: "c_proj",
    },
};

#[derive(Debug, Clone, Serialize, inference_json::FromJson)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: Option<usize>,
    pub max_position_embeddings: usize,
    pub layer_norm_epsilon: f64,
    pub rope_theta: f32,
    pub rope_scaling: Option<Llama3RopeConfig>,
    pub activation_function: Activation,
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

    pub fn decoder_spec(&self) -> Result<DecoderSpec> {
        Ok(DecoderSpec {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_heads: self.num_attention_heads,
            num_kv_heads: self.num_key_value_heads,
            head_dim: self.head_dim(),
            hidden_act: self.activation_function,
            rms_norm_eps: self.layer_norm_epsilon,
            rope: RopeKind::Llama3 {
                theta: self.rope_theta,
                scaling: self.rope_scaling.clone(),
            },
            max_position_embeddings: self.max_position_embeddings,
            o_proj_name: Some(O_PROJ),
            norm_names: NORM_NAMES,
            module_names: MODULE_NAMES,
            layer_windows: vec![None; self.num_layers],
            tie_word_embeddings: self.tie_word_embeddings,
            quantization_config: self.quantization_config.clone(),
            ..Default::default()
        })
    }
}

pub type Model = crate::decoder::CausalLm;
