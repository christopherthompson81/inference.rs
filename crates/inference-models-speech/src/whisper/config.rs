use std::collections::HashMap;

use inference_nn::audio::whisper::{WhisperEncoderConfig, WhisperLayerConfig};
use inference_nn::layers::Activation;

const LAYER_NORM_EPS: f64 = 1e-5;

/// transformers' `WhisperConfig`, the fields the model reads.
#[derive(Debug, Clone, serde::Serialize, inference_json::FromJson)]
pub struct WhisperConfig {
    pub model_type: String,
    pub d_model: usize,
    pub encoder_layers: usize,
    pub decoder_layers: usize,
    pub encoder_attention_heads: usize,
    pub decoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub decoder_ffn_dim: usize,
    pub num_mel_bins: usize,
    pub vocab_size: usize,
    pub max_source_positions: usize,
    pub max_target_positions: usize,
    pub activation_function: Activation,
}

impl WhisperConfig {
    pub fn encoder(&self) -> WhisperEncoderConfig {
        WhisperEncoderConfig {
            layer: WhisperLayerConfig {
                d_model: self.d_model,
                num_heads: self.encoder_attention_heads,
                ffn_dim: self.encoder_ffn_dim,
                activation: self.activation_function,
                layer_norm_eps: LAYER_NORM_EPS,
                k_bias: false,
            },
            num_layers: self.encoder_layers,
            num_mel_bins: self.num_mel_bins,
            max_source_positions: self.max_source_positions,
        }
    }

    pub fn layer_norm_eps(&self) -> f64 {
        LAYER_NORM_EPS
    }
}

/// `generation_config.json`: the special token ids, which move between Whisper versions, and the decoding rules.
#[derive(Debug, Clone, serde::Serialize, inference_json::FromJson)]
pub struct WhisperGenerationConfig {
    pub decoder_start_token_id: u32,
    pub eos_token_id: u32,
    pub no_timestamps_token_id: u32,
    #[serde(default)]
    pub begin_suppress_tokens: Vec<u32>,
    #[serde(default)]
    pub suppress_tokens: Vec<u32>,
    /// `[layer, head]` pairs whose cross-attention follows the audio, for word times.
    #[serde(default)]
    pub alignment_heads: Vec<Vec<usize>>,
    #[serde(default)]
    pub lang_to_id: HashMap<String, u32>,
    #[serde(default)]
    pub task_to_id: HashMap<String, u32>,
    pub max_initial_timestamp_index: Option<usize>,
    #[serde(default)]
    pub is_multilingual: bool,
}
