//! Qwen2: the shared decoder with q/k/v bias and per-layer windows from `layer_types` or `max_window_layers`.

use inference_quant::QuantizedConfig;
use inference_tensor::Result;

use crate::{
    decoder::{DecoderSpec, RopeKind},
    layers::Activation,
    serde_default_fn,
};

serde_default_fn!(bool, word_emb_default, false);
serde_default_fn!(usize, max_window_layers_default, 28);

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Qwen2AttentionType {
    FullAttention,
    SlidingAttention,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub use_sliding_window: bool,
    #[serde(default = "max_window_layers_default")]
    pub max_window_layers: usize,
    #[serde(default)]
    pub layer_types: Option<Vec<Qwen2AttentionType>>,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub hidden_act: Activation,
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default = "word_emb_default")]
    pub tie_word_embeddings: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vocab_size: 0,
            hidden_size: 0,
            intermediate_size: 0,
            num_hidden_layers: 0,
            num_attention_heads: 0,
            num_key_value_heads: 0,
            max_position_embeddings: 0,
            sliding_window: None,
            use_sliding_window: false,
            max_window_layers: max_window_layers_default(),
            layer_types: None,
            rope_theta: 0.0,
            rms_norm_eps: 0.0,
            hidden_act: Activation::default(),
            quantization_config: None,
            tie_word_embeddings: word_emb_default(),
        }
    }
}

impl Config {
    #[allow(clippy::cast_possible_truncation)]
    pub fn decoder_spec(&self) -> Result<DecoderSpec> {
        Ok(DecoderSpec {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_heads: self.num_attention_heads,
            num_kv_heads: self.num_key_value_heads,
            head_dim: self.hidden_size / self.num_attention_heads,
            hidden_act: self.hidden_act,
            rms_norm_eps: self.rms_norm_eps,
            rope: RopeKind::Default {
                theta: self.rope_theta as f32,
            },
            max_position_embeddings: self.max_position_embeddings,
            qkv_bias: true,
            qk_norm: None,
            no_rope_layers: Vec::new(),
            attention_temperature: None,
            layer_windows: self.layer_sliding_windows()?,
            tie_word_embeddings: self.tie_word_embeddings,
            quantization_config: self.quantization_config.clone(),
        })
    }

    fn layer_sliding_windows(&self) -> Result<Vec<Option<usize>>> {
        let sliding_window = self
            .use_sliding_window
            .then_some(self.sliding_window)
            .flatten();
        let layer_types = self.layer_types.clone().unwrap_or_else(|| {
            (0..self.num_hidden_layers)
                .map(|layer_idx| {
                    if sliding_window.is_some() && layer_idx >= self.max_window_layers {
                        Qwen2AttentionType::SlidingAttention
                    } else {
                        Qwen2AttentionType::FullAttention
                    }
                })
                .collect()
        });
        if layer_types.len() != self.num_hidden_layers {
            inference_tensor::bail!(
                "Qwen2 layer_types has {} entries for {} layers",
                layer_types.len(),
                self.num_hidden_layers
            );
        }
        layer_types
            .into_iter()
            .map(|layer_type| match layer_type {
                Qwen2AttentionType::FullAttention => Ok(None),
                Qwen2AttentionType::SlidingAttention => {
                    sliding_window.map(Some).ok_or_else(|| {
                        inference_tensor::Error::msg(
                            "Qwen2 sliding_attention layer requires use_sliding_window and sliding_window",
                        )
                    })
                }
            })
            .collect()
    }
}

pub type Model = crate::decoder::CausalLm;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_default_matches_the_serde_window_default() {
        assert_eq!(
            Config::default().max_window_layers,
            max_window_layers_default()
        );
    }

    #[test]
    fn serialized_sliding_window_is_disabled_without_use_flag() -> Result<()> {
        let config = Config {
            num_hidden_layers: 4,
            sliding_window: Some(128),
            max_window_layers: 2,
            ..Default::default()
        };

        assert_eq!(config.layer_sliding_windows()?, vec![None; 4]);
        Ok(())
    }

    #[test]
    fn generated_layer_types_match_transformers_normalization() -> Result<()> {
        let config = Config {
            num_hidden_layers: 4,
            sliding_window: Some(128),
            use_sliding_window: true,
            max_window_layers: 2,
            ..Default::default()
        };

        assert_eq!(
            config.layer_sliding_windows()?,
            vec![None, None, Some(128), Some(128)]
        );
        Ok(())
    }

    #[test]
    fn explicit_layer_types_are_validated() {
        let config = Config {
            num_hidden_layers: 2,
            sliding_window: Some(128),
            use_sliding_window: true,
            layer_types: Some(vec![Qwen2AttentionType::SlidingAttention]),
            ..Default::default()
        };

        assert!(config.layer_sliding_windows().is_err());
    }
}
