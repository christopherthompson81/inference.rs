//! Gemma 2: Gemma with sandwich norms, attention and final-logit softcaps, and every even layer sliding.
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use inference_quant::QuantizedConfig;
use inference_tensor::Result;

use crate::{
    decoder::{DecoderSpec, NormKind, NormNames, RopeKind},
    layers::Activation,
    serde_default_fn,
};

serde_default_fn!(bool, word_emb_default, false);

pub const SANDWICH_NORMS: NormNames = NormNames {
    input: "input_layernorm",
    pre_ffn: Some("pre_feedforward_layernorm"),
    post_attn: Some("post_attention_layernorm"),
    post_ffn: Some("post_feedforward_layernorm"),
    last: "norm",
};

#[derive(Debug, Clone, Default, serde::Deserialize)]
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
    pub sliding_window: usize,
    pub attn_logit_softcapping: Option<f64>,
    pub final_logit_softcapping: Option<f64>,
    pub query_pre_attn_scalar: usize,
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
            layer_windows: (0..self.num_hidden_layers)
                .map(|layer_idx| layer_idx.is_multiple_of(2).then_some(self.sliding_window))
                .collect(),
            tie_word_embeddings: self.tie_word_embeddings,
            quantization_config: self.quantization_config.clone(),
            norm: NormKind::Gemma,
            norm_names: SANDWICH_NORMS,
            attn_softcap: self.attn_logit_softcapping.map(|cap| cap as f32),
            softmax_scale: Some(1.0 / (self.query_pre_attn_scalar as f32).sqrt()),
            final_logit_softcap: self.final_logit_softcapping.map(|cap| cap as f32),
            embed_scale: Some((self.hidden_size as f64).sqrt()),
            ..Default::default()
        })
    }

    pub fn hidden_act(&self) -> Result<Activation> {
        match (self.hidden_act, self.hidden_activation) {
            (None, Some(act)) | (Some(act), None) => Ok(act),
            (Some(act), Some(_)) => {
                // If both are set just use hidden_act
                Ok(act)
            }
            (None, None) => {
                inference_tensor::bail!("none of hidden_act and hidden_activation are set")
            }
        }
    }
}

pub type Model = crate::decoder::CausalLm;

#[cfg(test)]
mod tests {
    use super::*;

    const SLIDING_WINDOW: usize = 4096;

    #[test]
    fn even_layers_slide_and_odd_layers_attend_fully() -> Result<()> {
        let config = Config {
            num_hidden_layers: 5,
            sliding_window: SLIDING_WINDOW,
            hidden_activation: Some(Activation::GeluPytorchTanh),
            query_pre_attn_scalar: 1,
            ..Default::default()
        };
        let window = Some(SLIDING_WINDOW);
        assert_eq!(
            config.decoder_spec()?.layer_windows,
            [window, None, window, None, window]
        );
        Ok(())
    }
}
