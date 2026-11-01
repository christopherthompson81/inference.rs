use inference_quant::QuantizedConfig;

use crate::{
    audio::whisper::{
        WhisperEncoderConfig, WhisperLayerConfig, WhisperLayerNames, WhisperPositions,
    },
    layers::Activation,
    llama, serde_default_fn,
};

const LAYER_NORM_EPS: f64 = 1e-5;
// the projector's RMSNorms (Ultravox's `RMSNorm` default)
serde_default_fn!(f64, default_norm_eps, 1e-6);

#[derive(Debug, Clone, inference_json::FromJson)]
pub struct AudioConfig {
    #[serde(alias = "d_model")]
    pub hidden_size: usize,
    #[serde(alias = "encoder_ffn_dim")]
    pub intermediate_size: usize,
    #[serde(alias = "encoder_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(alias = "encoder_layers")]
    pub num_hidden_layers: usize,
    pub num_mel_bins: usize,
    pub max_source_positions: usize,
}

/// Ultravox's tower, projector and text model as its GGUFs describe them; the published safetensors repos hold no
/// text model (it comes from another repository), so this is the GGUF loader's synthesized config.
#[derive(Debug, Clone, inference_json::FromJson)]
pub struct UltravoxConfig {
    pub audio_config: AudioConfig,
    pub text_config: llama::Config,
    pub stack_factor: usize,
    /// The projector's inner width, halved by its SwiGLU.
    pub hidden_size: usize,
    pub audio_token_id: u32,
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f64,
    #[serde(default)]
    pub quantization_config: Option<QuantizedConfig>,
}

impl UltravoxConfig {
    pub fn encoder(&self) -> WhisperEncoderConfig {
        let ac = &self.audio_config;
        WhisperEncoderConfig {
            layer: WhisperLayerConfig {
                d_model: ac.hidden_size,
                num_heads: ac.num_attention_heads,
                ffn_dim: ac.intermediate_size,
                activation: Activation::Gelu,
                layer_norm_eps: LAYER_NORM_EPS,
                k_bias: false,
                names: WhisperLayerNames::Whisper,
            },
            num_layers: ac.num_hidden_layers,
            num_mel_bins: ac.num_mel_bins,
            max_source_positions: ac.max_source_positions,
            positions: WhisperPositions::Learned,
        }
    }

    pub fn stacked_width(&self) -> usize {
        self.audio_config.hidden_size * self.stack_factor
    }

    pub fn text_config(&self) -> llama::Config {
        let mut text = self.text_config.clone();
        if text.quantization_config.is_none() {
            text.quantization_config = self.quantization_config.clone();
        }
        text
    }
}
