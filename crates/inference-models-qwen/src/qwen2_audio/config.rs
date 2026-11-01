use inference_quant::QuantizedConfig;

use crate::{
    audio::whisper::{WhisperEncoderConfig, WhisperLayerConfig},
    layers::Activation,
    qwen2, serde_default_fn,
};

const LAYER_NORM_EPS: f64 = 1e-5;

// transformers' `Qwen2Config` defaults, which the released config leaves the text model to
serde_default_fn!(usize, default_hidden_size, 4096);
serde_default_fn!(usize, default_layers, 32);
serde_default_fn!(usize, default_heads, 32);
serde_default_fn!(usize, default_max_window_layers, 28);
serde_default_fn!(f64, default_rope_theta, 10000.0);
serde_default_fn!(f64, default_rms_norm_eps, 1e-6);
serde_default_fn!(Activation, default_hidden_act, Activation::Silu);

#[derive(Debug, Clone, inference_json::FromJson)]
pub struct AudioConfig {
    pub d_model: usize,
    pub encoder_layers: usize,
    pub encoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub num_mel_bins: usize,
    pub max_source_positions: usize,
    pub activation_function: Activation,
}

#[derive(Debug, Clone, inference_json::FromJson)]
pub struct TextConfig {
    pub vocab_size: usize,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    pub intermediate_size: usize,
    #[serde(default = "default_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_heads")]
    pub num_attention_heads: usize,
    // transformers falls back to the attention heads
    pub num_key_value_heads: Option<usize>,
    pub max_position_embeddings: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: Activation,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_max_window_layers")]
    pub max_window_layers: usize,
}

#[derive(Debug, Clone, inference_json::FromJson)]
pub struct Qwen2AudioConfig {
    pub audio_config: AudioConfig,
    pub text_config: TextConfig,
    pub audio_token_index: u32,
    #[serde(default)]
    pub quantization_config: Option<QuantizedConfig>,
}

impl Qwen2AudioConfig {
    pub fn encoder(&self) -> WhisperEncoderConfig {
        let ac = &self.audio_config;
        WhisperEncoderConfig {
            layer: WhisperLayerConfig {
                d_model: ac.d_model,
                num_heads: ac.encoder_attention_heads,
                ffn_dim: ac.encoder_ffn_dim,
                activation: ac.activation_function,
                layer_norm_eps: LAYER_NORM_EPS,
                k_bias: false,
            },
            num_layers: ac.encoder_layers,
            num_mel_bins: ac.num_mel_bins,
            max_source_positions: ac.max_source_positions,
        }
    }

    /// The Qwen2 text model; the released checkpoints use no sliding window.
    pub fn text(&self) -> qwen2::Config {
        let tc = &self.text_config;
        qwen2::Config {
            vocab_size: tc.vocab_size,
            hidden_size: tc.hidden_size,
            intermediate_size: tc.intermediate_size,
            num_hidden_layers: tc.num_hidden_layers,
            num_attention_heads: tc.num_attention_heads,
            num_key_value_heads: tc.num_key_value_heads.unwrap_or(tc.num_attention_heads),
            max_position_embeddings: tc.max_position_embeddings,
            sliding_window: None,
            use_sliding_window: false,
            max_window_layers: tc.max_window_layers,
            layer_types: None,
            rope_theta: tc.rope_theta,
            rms_norm_eps: tc.rms_norm_eps,
            hidden_act: tc.hidden_act,
            quantization_config: self.quantization_config.clone(),
            tie_word_embeddings: tc.tie_word_embeddings,
        }
    }
}
