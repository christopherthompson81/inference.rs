//! Phi-2: <https://huggingface.co/microsoft/phi-2>, as of commit cb2f4533604d8b67de604e7df03bfe6f3ca22869.
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use inference_quant::QuantizedConfig;
use serde::{Deserialize, Serialize};

use crate::{
    amoe::AnyMoeLoraTarget,
    decoder::{DecoderSpec, MlpKind, NormKind, NormNames, QkNorm, RopeKind},
    layers::Activation,
    serde_default_fn,
};

serde_default_fn!(bool, word_emb_default, false);

const MLP_PROJECTIONS: &[AnyMoeLoraTarget; 2] =
    &[AnyMoeLoraTarget::up("fc1"), AnyMoeLoraTarget::down("fc2")];

// attention and MLP both read the one input norm
const PARALLEL_NORMS: NormNames = NormNames {
    input: "input_layernorm",
    pre_ffn: None,
    post_attn: None,
    post_ffn: None,
    last: "final_layernorm",
};

const QK_LAYERNORM: QkNorm = QkNorm::BeforeRope {
    q: "q_layernorm",
    k: "k_layernorm",
};

// https://huggingface.co/microsoft/phi-2/blob/main/configuration_phi.py
#[derive(Debug, Clone, Deserialize, Default, Serialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub hidden_act: Activation,
    pub max_position_embeddings: usize,
    pub layer_norm_eps: f64,
    pub rope_theta: f32,
    pub partial_rotary_factor: f64,
    pub qk_layernorm: bool,
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default = "word_emb_default")]
    pub tie_word_embeddings: bool,
}

impl Config {
    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn decoder_spec(&self) -> DecoderSpec {
        DecoderSpec {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_heads: self.num_attention_heads,
            num_kv_heads: self.num_key_value_heads(),
            head_dim: self.head_dim(),
            hidden_act: self.hidden_act,
            rms_norm_eps: self.layer_norm_eps,
            rope: RopeKind::Partial {
                theta: self.rope_theta,
                rotary_dim: (self.partial_rotary_factor * self.head_dim() as f64) as usize,
                is_gpt_neox: None,
            },
            max_position_embeddings: self.max_position_embeddings,
            qkv_bias: true,
            o_bias: true,
            o_proj_name: Some("dense"),
            qk_norm: self.qk_layernorm.then_some(QK_LAYERNORM),
            layer_windows: vec![None; self.num_hidden_layers],
            tie_word_embeddings: self.tie_word_embeddings,
            quantization_config: self.quantization_config.clone(),
            norm: NormKind::Layer,
            norm_names: PARALLEL_NORMS,
            mlp: MlpKind::Plain {
                projections: MLP_PROJECTIONS,
                bias: true,
            },
            // HF's PhiForCausalLM lm_head is biased
            lm_head_bias: true,
            ..Default::default()
        }
    }
}

pub type Model = crate::decoder::CausalLm;
