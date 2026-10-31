use inference_quant::QuantizedConfig;
use serde::Serialize;

use crate::layers::Activation;

#[derive(Debug, Clone, Serialize, inference_json::FromJson)]
pub struct AudioConfig {
    pub d_model: usize,
    pub encoder_layers: usize,
    pub encoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub num_mel_bins: usize,
    pub downsample_hidden_size: usize,
    pub output_dim: usize,
    pub n_window: usize,
    pub n_window_infer: usize,
    pub activation_function: Activation,
}

#[derive(Debug, Clone, Serialize, inference_json::FromJson)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: Option<usize>,
    pub hidden_act: Activation,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub tie_word_embeddings: bool,
    pub quantization_config: Option<QuantizedConfig>,
}

impl TextConfig {
    // the interleaved M-RoPE sections are 1-D RoPE here: text and audio positions match on all three axes
    pub fn qwen3(&self) -> crate::qwen3::Config {
        crate::qwen3::Config {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            hidden_act: self.hidden_act,
            max_position_embeddings: self.max_position_embeddings,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            head_dim: self.head_dim,
            quantization_config: self.quantization_config.clone(),
            tie_word_embeddings: self.tie_word_embeddings,
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone, Serialize, inference_json::FromJson)]
pub struct ThinkerConfig {
    pub audio_config: AudioConfig,
    pub text_config: TextConfig,
    pub audio_token_id: u32,
    pub audio_start_token_id: u32,
    pub audio_end_token_id: u32,
}

/// The released checkpoints' config, the model under `thinker_config` and its weights under `thinker.`.
#[derive(Debug, Clone, Serialize, inference_json::FromJson)]
pub struct Config {
    pub thinker_config: ThinkerConfig,
}
