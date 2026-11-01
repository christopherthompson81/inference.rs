use inference_quant::QuantizedConfig;

use crate::{
    audio::whisper::{
        WhisperEncoderConfig, WhisperLayerConfig, WhisperLayerNames, WhisperPositions,
    },
    gemma2,
    layers::Activation,
};

const LAYER_NORM_EPS: f64 = 1e-5;

/// The Whisper tower, under transformers' `WhisperConfig` names.
#[derive(Debug, Clone, inference_json::FromJson)]
pub struct SpeechConfig {
    pub d_model: usize,
    pub encoder_layers: usize,
    pub encoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub num_mel_bins: usize,
    pub max_source_positions: usize,
    pub activation_function: Activation,
}

#[derive(Debug, Clone, inference_json::FromJson)]
pub struct MERaLiON2Config {
    pub speech_config: SpeechConfig,
    pub text_config: gemma2::Config,
    /// Encoder positions stacked into one adapter input.
    pub speech_mlp_scale_factor: usize,
    pub speech_token_index: u32,
    #[serde(default)]
    pub quantization_config: Option<QuantizedConfig>,
}

impl MERaLiON2Config {
    pub fn encoder(&self) -> WhisperEncoderConfig {
        let sc = &self.speech_config;
        WhisperEncoderConfig {
            layer: WhisperLayerConfig {
                d_model: sc.d_model,
                num_heads: sc.encoder_attention_heads,
                ffn_dim: sc.encoder_ffn_dim,
                activation: sc.activation_function,
                layer_norm_eps: LAYER_NORM_EPS,
                k_bias: false,
                names: WhisperLayerNames::Whisper,
            },
            num_layers: sc.encoder_layers,
            num_mel_bins: sc.num_mel_bins,
            max_source_positions: sc.max_source_positions,
            positions: WhisperPositions::Learned,
        }
    }

    pub fn stacked_width(&self) -> usize {
        self.speech_config.d_model * self.speech_mlp_scale_factor
    }

    /// The adapter's inner width, five times the tower's.
    pub fn adapter_width(&self) -> usize {
        5 * self.speech_config.d_model
    }

    // transformers' Gemma2Config ties by default and the checkpoint has no head
    pub fn text_config(&self) -> gemma2::Config {
        let mut text = self.text_config.clone();
        text.tie_word_embeddings = true;
        if text.quantization_config.is_none() {
            text.quantization_config = self.quantization_config.clone();
        }
        text
    }
}
