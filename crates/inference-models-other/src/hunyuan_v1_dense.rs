//! HunYuan dense: the shared decoder with q/k norm after RoPE and dynamic-alpha RoPE theta.
#![allow(clippy::cast_possible_truncation)]

use super::hunyuan_rope::{RopeScalingConfig, effective_rope_theta};
use inference_quant::QuantizedConfig;
use inference_tensor::Result;
use serde::{Deserialize, Serialize};

use crate::{
    decoder::{DecoderSpec, QkNorm, RopeKind},
    layers::Activation,
    serde_default_fn,
};

serde_default_fn!(bool, tie_word_embeddings_default, false);
serde_default_fn!(bool, use_cla_default, false);
serde_default_fn!(usize, pretraining_tp_default, 1);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub hidden_act: Activation,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub rope_scaling: Option<RopeScalingConfig>,
    #[serde(default = "use_cla_default")]
    pub use_cla: bool,
    #[serde(default)]
    pub cla_share_factor: Option<usize>,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
    #[serde(default = "pretraining_tp_default")]
    pub pretraining_tp: usize,
    #[serde(default)]
    pub add_classification_head: bool,
    #[serde(default = "tie_word_embeddings_default")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub quantization_config: Option<QuantizedConfig>,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .filter(|&d| d > 0)
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn effective_rope_theta(&self) -> Result<f64> {
        effective_rope_theta(self.rope_theta, self.head_dim(), self.rope_scaling.as_ref())
    }

    pub fn decoder_spec(&self) -> Result<DecoderSpec> {
        if self.use_cla {
            inference_tensor::bail!("HunYuanDenseV1 CLA is not implemented")
        }
        if self.attention_bias {
            inference_tensor::bail!("HunYuanDenseV1 attention_bias=true is not implemented")
        }
        if self.mlp_bias {
            inference_tensor::bail!("HunYuanDenseV1 mlp_bias=true is not implemented")
        }
        if self.pretraining_tp != 1 {
            inference_tensor::bail!("HunYuanDenseV1 pretraining_tp>1 is not implemented")
        }
        if self.add_classification_head {
            inference_tensor::bail!("HunYuanDenseV1 classification head is not implemented")
        }
        Ok(DecoderSpec {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_heads: self.num_attention_heads,
            num_kv_heads: self.num_key_value_heads,
            head_dim: self.head_dim(),
            hidden_act: self.hidden_act,
            rms_norm_eps: self.rms_norm_eps,
            rope: RopeKind::Default {
                theta: self.effective_rope_theta()? as f32,
            },
            max_position_embeddings: self.max_position_embeddings,
            qkv_bias: false,
            qk_norm: Some(QkNorm::AfterRope {
                q: "query_layernorm",
                k: "key_layernorm",
            }),
            no_rope_layers: Vec::new(),
            attention_temperature: None,
            layer_windows: vec![None; self.num_hidden_layers],
            tie_word_embeddings: self.tie_word_embeddings,
            quantization_config: self.quantization_config.clone(),
            ..Default::default()
        })
    }
}

pub type Model = crate::decoder::CausalLm;
