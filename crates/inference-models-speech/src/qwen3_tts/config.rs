use std::collections::BTreeMap;

use crate::layers::Activation;

const QWEN3_TTS: &str = "qwen3_tts";
const CODEC_12HZ: &str = "qwen3_tts_tokenizer_12hz";

/// The code predictor: a small Qwen3 decoder filling codebooks 1.. of each frame.
#[derive(Debug, Clone, inference_json::FromJson)]
pub struct CodePredictorConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub hidden_act: Activation,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
}

#[derive(Debug, Clone, inference_json::FromJson)]
pub struct TalkerConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub hidden_act: Activation,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    /// The codec vocabulary: audio codes, then control ids (bos, eos, think tags, languages, speakers).
    pub vocab_size: usize,
    pub text_hidden_size: usize,
    pub text_vocab_size: usize,
    pub num_code_groups: usize,
    pub codec_bos_id: u32,
    pub codec_eos_token_id: u32,
    pub codec_pad_id: u32,
    pub codec_think_id: u32,
    pub codec_nothink_id: u32,
    pub codec_think_bos_id: u32,
    pub codec_think_eos_id: u32,
    pub codec_language_id: BTreeMap<String, u32>,
    #[serde(default)]
    pub spk_id: BTreeMap<String, u32>,
    // false, or the dialect whose language id the speaker takes in Chinese
    #[serde(default)]
    pub spk_is_dialect: BTreeMap<String, serde_json::Value>,
    pub code_predictor_config: CodePredictorConfig,
}

#[derive(Debug, Clone, inference_json::FromJson)]
pub struct Qwen3TtsConfig {
    pub model_type: String,
    pub tokenizer_type: String,
    /// `custom_voice`, `voice_design` or `base`.
    pub tts_model_type: String,
    /// `0b6` or `1b7`.
    pub tts_model_size: String,
    pub tts_bos_token_id: u32,
    pub tts_eos_token_id: u32,
    pub tts_pad_token_id: u32,
    pub talker_config: TalkerConfig,
    /// The Base checkpoints' speaker encoder.
    #[serde(default)]
    pub speaker_encoder_config: Option<SpeakerEncoderConfig>,
}

impl Qwen3TtsConfig {
    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        let cfg: Self = serde_json::from_str(json)?;
        anyhow::ensure!(
            cfg.model_type == QWEN3_TTS,
            "not a Qwen3-TTS config: model_type `{}`",
            cfg.model_type
        );
        anyhow::ensure!(
            cfg.tokenizer_type == CODEC_12HZ,
            "Qwen3-TTS with the `{}` codec is not supported, only the 12 Hz one",
            cfg.tokenizer_type
        );
        Ok(cfg)
    }

    /// The dialect a speaker speaks Chinese in, by its `codec_language_id` name.
    pub fn dialect(&self, speaker: &str) -> Option<&str> {
        self.talker_config
            .spk_is_dialect
            .get(speaker)
            .and_then(serde_json::Value::as_str)
    }
}

/// The 12 Hz codec's decoder (`speech_tokenizer/config.json`, `decoder_config`).
#[derive(Debug, Clone, inference_json::FromJson)]
pub struct CodecDecoderConfig {
    pub codebook_size: usize,
    pub codebook_dim: usize,
    pub latent_dim: usize,
    pub decoder_dim: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub hidden_act: Activation,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub sliding_window: usize,
    pub num_quantizers: usize,
    pub num_semantic_quantizers: usize,
    pub upsample_rates: Vec<usize>,
    pub upsampling_ratios: Vec<usize>,
}

/// The 12 Hz codec's encoder, a Mimi (`encoder_config`): read for voice cloning from a reference clip.
#[derive(Debug, Clone, inference_json::FromJson)]
pub struct CodecEncoderConfig {
    pub num_filters: usize,
    pub kernel_size: usize,
    pub last_kernel_size: usize,
    pub residual_kernel_size: usize,
    pub dilation_growth_rate: usize,
    pub num_residual_layers: usize,
    pub compress: usize,
    pub upsampling_ratios: Vec<usize>,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub codebook_size: usize,
    pub codebook_dim: usize,
    pub num_quantizers: usize,
    pub num_semantic_quantizers: usize,
    pub vector_quantization_hidden_dimension: usize,
}

#[derive(Debug, Clone, inference_json::FromJson)]
pub struct CodecConfig {
    pub decoder_config: CodecDecoderConfig,
    #[serde(default)]
    pub encoder_config: Option<CodecEncoderConfig>,
    /// The encoder's codebooks the talker reads (of the Mimi's `num_quantizers`).
    #[serde(default)]
    pub encoder_valid_num_quantizers: Option<usize>,
    pub output_sample_rate: usize,
}

/// The Base checkpoints' speaker encoder, an ECAPA-TDNN over a 24 kHz log-mel.
#[derive(Debug, Clone, inference_json::FromJson)]
pub struct SpeakerEncoderConfig {
    #[serde(default = "default_mel_dim")]
    pub mel_dim: usize,
    pub enc_dim: usize,
    #[serde(default = "default_channels")]
    pub enc_channels: Vec<usize>,
    #[serde(default = "default_kernels")]
    pub enc_kernel_sizes: Vec<usize>,
    #[serde(default = "default_dilations")]
    pub enc_dilations: Vec<usize>,
    #[serde(default = "default_attention_channels")]
    pub enc_attention_channels: usize,
    #[serde(default = "default_res2net_scale")]
    pub enc_res2net_scale: usize,
    #[serde(default = "default_se_channels")]
    pub enc_se_channels: usize,
    pub sample_rate: u32,
}

impl SpeakerEncoderConfig {
    /// The reference's encoder at `mel_dim` mels and `enc_dim` outputs, its other settings at their defaults.
    pub fn ecapa(mel_dim: usize, enc_dim: usize, sample_rate: u32) -> Self {
        Self {
            mel_dim,
            enc_dim,
            enc_channels: default_channels(),
            enc_kernel_sizes: default_kernels(),
            enc_dilations: default_dilations(),
            enc_attention_channels: default_attention_channels(),
            enc_res2net_scale: default_res2net_scale(),
            enc_se_channels: default_se_channels(),
            sample_rate,
        }
    }
}

// the reference's `Qwen3TTSSpeakerEncoderConfig` defaults, which the released configs leave out
fn default_mel_dim() -> usize {
    128
}
fn default_channels() -> Vec<usize> {
    vec![512, 512, 512, 512, 1536]
}
fn default_kernels() -> Vec<usize> {
    vec![5, 3, 3, 3, 1]
}
fn default_dilations() -> Vec<usize> {
    vec![1, 2, 3, 4, 1]
}
fn default_attention_channels() -> usize {
    128
}
fn default_res2net_scale() -> usize {
    8
}
fn default_se_channels() -> usize {
    128
}
