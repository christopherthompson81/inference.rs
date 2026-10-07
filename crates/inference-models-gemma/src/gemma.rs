//! Gemma: the shared decoder with Gemma's `1 + weight` RMS norm and embeddings scaled by `sqrt(hidden_size)`.
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use inference_quant::QuantizedConfig;
use inference_tensor::Result;

use crate::{
    decoder::{DecoderSpec, NormKind, RopeKind},
    layers::Activation,
    serde_default_fn,
};

fn default_max_position_embeddings() -> usize {
    4096
}

serde_default_fn!(bool, word_emb_default, false);

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone, Default)]
pub struct Config {
    pub attention_bias: bool,
    pub head_dim: usize,
    // The code gemma configs include both hidden_act and hidden_activation.
    pub hidden_act: Option<Activation>,
    pub hidden_activation: Option<Activation>,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub vocab_size: usize,

    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default = "word_emb_default")]
    pub tie_word_embeddings: bool,
}

impl Config {
    pub fn decoder_spec(&self) -> Result<DecoderSpec> {
        Ok(DecoderSpec {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_heads: self.num_attention_heads,
            num_kv_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
            hidden_act: self.hidden_act()?,
            rms_norm_eps: self.rms_norm_eps,
            rope: RopeKind::Default {
                theta: self.rope_theta as f32,
            },
            max_position_embeddings: self.max_position_embeddings,
            qkv_bias: self.attention_bias,
            o_bias: self.attention_bias,
            layer_windows: vec![None; self.num_hidden_layers],
            tie_word_embeddings: self.tie_word_embeddings,
            quantization_config: self.quantization_config.clone(),
            norm: NormKind::Gemma,
            embed_scale: Some((self.hidden_size as f64).sqrt()),
            ..Default::default()
        })
    }

    pub fn hidden_act(&self) -> Result<Activation> {
        match (self.hidden_act, self.hidden_activation) {
            (None, Some(act)) | (Some(act), None) => Ok(act),
            (Some(_), Some(_)) => {
                inference_tensor::bail!("both hidden_act and hidden_activation are set")
            }
            (None, None) => {
                inference_tensor::bail!("none of hidden_act and hidden_activation are set")
            }
        }
    }
}

pub type Model = crate::decoder::CausalLm;
