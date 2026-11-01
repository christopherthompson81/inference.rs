use inference_quant::QuantizedConfig;

use crate::{
    audio::whisper::{
        WhisperEncoderConfig, WhisperLayerConfig, WhisperLayerNames, WhisperPositions,
    },
    layers::Activation,
    qwen2, serde_default_fn,
};

const LAYER_NORM_EPS: f64 = 1e-5;
// Qwen2-Audio's processor reads one 30 s window per clip; Audio Flamingo 3's up to 600 s of them
const QWEN2_AUDIO_WINDOWS: usize = 1;
const AUDIO_FLAMINGO3_WINDOWS: usize = 20;

// transformers' `Qwen2Config` defaults, which Qwen2-Audio's released config leaves the text model to
serde_default_fn!(usize, default_hidden_size, 4096);
serde_default_fn!(usize, default_layers, 32);
serde_default_fn!(usize, default_heads, 32);
serde_default_fn!(usize, default_max_window_layers, 28);
serde_default_fn!(f64, default_rope_theta, 10000.0);
serde_default_fn!(f64, default_rms_norm_eps, 1e-6);
serde_default_fn!(Activation, default_hidden_act, Activation::Silu);

/// The Whisper tower, under Qwen2-Audio's names or transformers 5's (Audio Flamingo 3).
#[derive(Debug, Clone, inference_json::FromJson)]
pub struct AudioConfig {
    #[serde(alias = "hidden_size")]
    pub d_model: usize,
    #[serde(alias = "num_hidden_layers")]
    pub encoder_layers: usize,
    #[serde(alias = "num_attention_heads")]
    pub encoder_attention_heads: usize,
    #[serde(alias = "intermediate_size")]
    pub encoder_ffn_dim: usize,
    pub num_mel_bins: usize,
    pub max_source_positions: usize,
    pub activation_function: Activation,
}

#[derive(Debug, Clone, inference_json::FromJson)]
pub struct RopeParameters {
    pub rope_theta: f64,
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
    // the older repos give `rope_theta`; transformers 5 nests it in `rope_parameters`
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: Activation,
    #[serde(default)]
    pub tie_word_embeddings: Option<bool>,
    #[serde(default = "default_max_window_layers")]
    pub max_window_layers: usize,
}

impl TextConfig {
    /// The Qwen2 text model; the released checkpoints use no sliding window.
    fn qwen2(&self, quantization_config: Option<QuantizedConfig>) -> qwen2::Config {
        qwen2::Config {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads.unwrap_or(self.num_attention_heads),
            max_position_embeddings: self.max_position_embeddings,
            sliding_window: None,
            use_sliding_window: false,
            max_window_layers: self.max_window_layers,
            layer_types: None,
            rope_theta: self
                .rope_parameters
                .as_ref()
                .map_or(self.rope_theta, |rope| rope.rope_theta),
            rms_norm_eps: self.rms_norm_eps,
            hidden_act: self.hidden_act,
            quantization_config,
            tie_word_embeddings: self.tie_word_embeddings.unwrap_or(false),
        }
    }
}

/// How encoded audio reaches the text width.
#[derive(Debug, Clone, Copy)]
pub enum Projector {
    /// One linear layer with bias (Qwen2-Audio).
    Linear,
    /// Linear, activation, linear, at the text width (Audio Flamingo 3).
    Mlp { activation: Activation, bias: bool },
}

/// What the model and its processor need: Qwen2-Audio and Audio Flamingo 3 differ only in these.
#[derive(Debug, Clone)]
pub struct PooledAudioSpec {
    pub audio: AudioConfig,
    pub text: qwen2::Config,
    pub audio_token: u32,
    pub projector: Projector,
    pub max_windows: usize,
}

impl PooledAudioSpec {
    pub fn encoder(&self) -> WhisperEncoderConfig {
        let ac = &self.audio;
        WhisperEncoderConfig {
            layer: WhisperLayerConfig {
                d_model: ac.d_model,
                num_heads: ac.encoder_attention_heads,
                ffn_dim: ac.encoder_ffn_dim,
                activation: ac.activation_function,
                layer_norm_eps: LAYER_NORM_EPS,
                k_bias: false,
                names: WhisperLayerNames::Whisper,
            },
            num_layers: ac.encoder_layers,
            num_mel_bins: ac.num_mel_bins,
            max_source_positions: ac.max_source_positions,
            positions: WhisperPositions::Learned,
        }
    }
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
    pub fn spec(&self) -> PooledAudioSpec {
        PooledAudioSpec {
            audio: self.audio_config.clone(),
            text: self.text_config.qwen2(self.quantization_config.clone()),
            audio_token: self.audio_token_index,
            projector: Projector::Linear,
            max_windows: QWEN2_AUDIO_WINDOWS,
        }
    }
}

#[derive(Debug, Clone, inference_json::FromJson)]
pub struct AudioFlamingo3Config {
    pub audio_config: AudioConfig,
    pub text_config: TextConfig,
    pub audio_token_id: u32,
    pub projector_hidden_act: Activation,
    #[serde(default)]
    pub projector_bias: bool,
    #[serde(default)]
    pub quantization_config: Option<QuantizedConfig>,
}

impl AudioFlamingo3Config {
    pub fn spec(&self) -> PooledAudioSpec {
        PooledAudioSpec {
            audio: self.audio_config.clone(),
            text: self.text_config.qwen2(self.quantization_config.clone()),
            audio_token: self.audio_token_id,
            projector: Projector::Mlp {
                activation: self.projector_hidden_act,
                bias: self.projector_bias,
            },
            max_windows: AUDIO_FLAMINGO3_WINDOWS,
        }
    }
}
