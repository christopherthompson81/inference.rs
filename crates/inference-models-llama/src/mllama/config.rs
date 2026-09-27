use candle_core::{Context, Result, Tensor};
use candle_nn::Module;
use inference_quant::QuantizedConfig;

use crate::{
    layers::{Llama3RopeConfig, Llama3RopeType},
    serde_default_fn,
};

#[derive(Debug, Clone, Copy, serde::Deserialize)]
pub enum VisionActivation {
    QuickGelu,
    #[serde(alias = "gelu")]
    Gelu,
    #[serde(alias = "gelu_new")]
    NewGelu,
    Relu,
    Silu,
}

impl Module for VisionActivation {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::QuickGelu => xs * candle_nn::ops::sigmoid(&(xs * 1.702f64)?),
            Self::Gelu => xs.gelu_erf(),
            // https://github.com/huggingface/transformers/blob/12f043eaeaabfef6f6efea411d98e6f6d3c094b7/src/transformers/activations.py#L49-L78
            Self::NewGelu => xs.gelu(),
            Self::Relu => xs.relu(),
            Self::Silu => xs.silu(),
        }
    }
}

serde_default_fn!(usize, d_attn_heads, 16);

#[derive(Debug, Clone, serde::Deserialize)]
pub struct MLlamaVisionConfig {
    pub hidden_size: usize,
    pub hidden_act: VisionActivation,
    pub num_hidden_layers: usize,
    pub num_global_layers: usize,
    #[serde(default = "d_attn_heads", alias = "attention_heads")]
    pub num_attention_heads: usize,
    pub num_channels: usize,
    pub intermediate_size: usize,
    pub vision_output_dim: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub norm_eps: f64,
    pub max_num_tiles: usize,
    pub intermediate_layers_indices: Vec<usize>,
    pub supported_aspect_ratios: Vec<(usize, usize)>,
}

impl MLlamaVisionConfig {
    pub fn max_aspect_ratio_id(&self) -> usize {
        self.supported_aspect_ratios.len()
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
pub enum MLlamaRopeType {
    #[serde(rename = "default")]
    Default,
    #[serde(rename = "linear")]
    Linear,
    #[serde(rename = "dynamic")]
    Dynamic,
    #[serde(rename = "yarn")]
    Yarn,
    #[serde(rename = "longrope")]
    Longrope,
    #[serde(rename = "llama3")]
    Llama3,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[allow(dead_code)]
pub struct MLlamaRopeScaling {
    pub rope_type: MLlamaRopeType,
    pub factor: Option<f32>,
    pub original_max_position_embeddings: usize,
    pub attention_factor: Option<f32>,
    pub beta_fast: Option<f32>,
    pub beta_slow: Option<f32>,
    pub short_factor: Option<Vec<f64>>,
    pub long_factor: Option<Vec<f64>>,
    pub low_freq_factor: Option<f32>,
    pub high_freq_factor: Option<f32>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct MLlamaTextConfig {
    pub rope_scaling: Option<MLlamaRopeScaling>,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub hidden_act: candle_nn::Activation,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f64,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    pub cross_attention_layers: Vec<usize>,
    pub quantization_config: Option<QuantizedConfig>,
}

impl MLlamaTextConfig {
    pub fn llama3_rope_scaling(&self) -> Result<Option<Llama3RopeConfig>> {
        match &self.rope_scaling {
            None
            | Some(MLlamaRopeScaling {
                rope_type: MLlamaRopeType::Default,
                ..
            }) => Ok(None),
            Some(MLlamaRopeScaling {
                rope_type: MLlamaRopeType::Llama3,
                factor,
                original_max_position_embeddings,
                low_freq_factor,
                high_freq_factor,
                ..
            }) => Ok(Some(Llama3RopeConfig {
                factor: factor.context("MLlama Llama3 RoPE needs `factor` parameter.")?,
                low_freq_factor: *low_freq_factor,
                high_freq_factor: *high_freq_factor,
                original_max_position_embeddings: Some(*original_max_position_embeddings),
                rope_type: Llama3RopeType::Llama3,
            })),
            Some(MLlamaRopeScaling {
                rope_type: other, ..
            }) => {
                candle_core::bail!(
                    "MLlama doesn't support any other RoPE type than `llama3`, got {other:?}"
                )
            }
        }
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct MLlamaConfig {
    pub vision_config: MLlamaVisionConfig,
    pub text_config: MLlamaTextConfig,
}

#[cfg(test)]
mod tests {
    use super::MLlamaVisionConfig;

    #[test]
    fn vision_attention_heads_alias_is_supported() {
        let config: MLlamaVisionConfig = serde_json::from_value(serde_json::json!({
            "hidden_size": 1280,
            "hidden_act": "gelu",
            "num_hidden_layers": 32,
            "num_global_layers": 8,
            "attention_heads": 20,
            "num_channels": 3,
            "intermediate_size": 5120,
            "vision_output_dim": 7680,
            "image_size": 560,
            "patch_size": 14,
            "norm_eps": 0.00001,
            "max_num_tiles": 4,
            "intermediate_layers_indices": [3, 7, 15, 23, 30],
            "supported_aspect_ratios": [[1, 1], [1, 2]]
        }))
        .unwrap();

        assert_eq!(config.num_attention_heads, 20);
    }
}
