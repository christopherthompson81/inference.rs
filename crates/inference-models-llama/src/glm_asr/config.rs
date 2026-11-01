#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use inference_quant::QuantizedConfig;

use crate::{
    audio::whisper::{
        WhisperEncoderConfig, WhisperLayerConfig, WhisperLayerNames, WhisperPositions,
    },
    layers::Activation,
    mistral,
};

const LAYER_NORM_EPS: f64 = 1e-5;

#[derive(Debug, Clone, inference_json::FromJson)]
pub struct RopeParameters {
    pub rope_theta: f64,
    #[serde(default)]
    pub partial_rotary_factor: Option<f64>,
}

#[derive(Debug, Clone, inference_json::FromJson)]
pub struct AudioConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_mel_bins: usize,
    pub max_position_embeddings: usize,
    pub hidden_act: Activation,
    pub rope_parameters: RopeParameters,
}

#[derive(Debug, Clone, inference_json::FromJson)]
pub struct GlmAsrConfig {
    pub audio_config: AudioConfig,
    pub text_config: mistral::Config,
    pub audio_token_id: u32,
    pub projector_hidden_act: Activation,
    #[serde(default)]
    pub quantization_config: Option<QuantizedConfig>,
}

impl GlmAsrConfig {
    pub fn encoder(&self) -> WhisperEncoderConfig {
        let ac = &self.audio_config;
        let head_dim = ac.hidden_size / ac.num_attention_heads;
        let factor = ac.rope_parameters.partial_rotary_factor.unwrap_or(1.0);
        WhisperEncoderConfig {
            layer: WhisperLayerConfig {
                d_model: ac.hidden_size,
                num_heads: ac.num_attention_heads,
                ffn_dim: ac.intermediate_size,
                activation: ac.hidden_act,
                layer_norm_eps: LAYER_NORM_EPS,
                k_bias: false,
                names: WhisperLayerNames::Llama,
            },
            num_layers: ac.num_hidden_layers,
            num_mel_bins: ac.num_mel_bins,
            max_source_positions: ac.max_position_embeddings,
            positions: WhisperPositions::Rotary {
                theta: ac.rope_parameters.rope_theta as f32,
                dims: (head_dim as f64 * factor) as usize,
            },
        }
    }

    /// The width of one projector input: transformers regroups the encoder output by the tower's FFN width.
    pub fn stacked_width(&self) -> usize {
        self.audio_config.intermediate_size
    }

    pub fn text_config(&self) -> mistral::Config {
        let mut text = self.text_config.clone();
        if text.quantization_config.is_none() {
            text.quantization_config = self.quantization_config.clone();
        }
        text
    }
}
