use std::collections::HashMap;

use serde::Deserialize;

// NeMo's and transformers' default when a config leaves it out
const DEFAULT_MAX_SYMBOLS_PER_STEP: usize = 10;
/// The encoder `model_type` of cache-aware streaming checkpoints: causal convs, chunked-limited attention.
pub const STREAMING_ENCODER_TYPE: &str = "nemotron_asr_streaming_encoder";

/// The encoder half of a Parakeet `config.json` (transformers' `ParakeetEncoderConfig`).
#[derive(Debug, Clone, Deserialize)]
pub struct EncoderConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_mel_bins: usize,
    pub conv_kernel_size: usize,
    pub subsampling_conv_channels: usize,
    pub subsampling_conv_kernel_size: usize,
    pub subsampling_conv_stride: usize,
    pub subsampling_factor: usize,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub convolution_bias: bool,
    #[serde(default)]
    pub scale_input: bool,
    #[serde(default)]
    pub model_type: Option<String>,
    /// Streaming encoders attend this many frames back, the query's own included.
    #[serde(default)]
    pub sliding_window: Option<usize>,
    /// Streaming encoders' right context in encoder frames, which also sets their attention chunk.
    #[serde(default)]
    pub default_num_lookahead_tokens: Option<usize>,
}

/// The attention context of a chunked-limited (streaming) encoder, in encoder frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkedContext {
    pub left: usize,
    pub right: usize,
}

impl EncoderConfig {
    /// Causal convs and chunked-limited attention, as cache-aware streaming checkpoints are trained.
    pub fn is_streaming(&self) -> bool {
        self.model_type.as_deref() == Some(STREAMING_ENCODER_TYPE)
    }

    pub fn chunked_context(&self) -> Option<ChunkedContext> {
        if !self.is_streaming() {
            return None;
        }
        Some(ChunkedContext {
            left: self.sliding_window?.saturating_sub(1),
            right: self.default_num_lookahead_tokens?,
        })
    }
}

/// Which head turns encoder frames into tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadKind {
    Ctc,
    Rnnt,
    Tdt,
}

/// A Parakeet `config.json`: transformers' CTC, RNN-T or TDT config around the encoder's.
#[derive(Debug, Clone, Deserialize)]
pub struct ParakeetConfig {
    pub model_type: String,
    pub encoder_config: EncoderConfig,
    pub vocab_size: usize,
    #[serde(default)]
    pub pad_token_id: Option<u32>,
    #[serde(default)]
    pub blank_token_id: Option<u32>,
    #[serde(default)]
    pub decoder_hidden_size: Option<usize>,
    #[serde(default)]
    pub num_decoder_layers: Option<usize>,
    #[serde(default)]
    pub durations: Vec<usize>,
    #[serde(default = "default_max_symbols_per_step")]
    pub max_symbols_per_step: usize,
    /// Nemotron-3.5's language prompts: a one-hot of this many slots joins every encoder frame.
    #[serde(default)]
    pub num_prompts: Option<usize>,
    #[serde(default)]
    pub prompt_intermediate_size: Option<usize>,
    #[serde(default)]
    pub default_prompt_id: Option<usize>,
}

fn default_max_symbols_per_step() -> usize {
    DEFAULT_MAX_SYMBOLS_PER_STEP
}

/// The `model_type`s a Parakeet checkpoint carries.
pub const MODEL_TYPES: [(&str, HeadKind); 5] = [
    ("parakeet_ctc", HeadKind::Ctc),
    ("parakeet_rnnt", HeadKind::Rnnt),
    ("parakeet_tdt", HeadKind::Tdt),
    ("nemotron_asr_streaming", HeadKind::Rnnt),
    ("nemotron3_5_asr", HeadKind::Rnnt),
];

impl ParakeetConfig {
    pub fn head(&self) -> Option<HeadKind> {
        MODEL_TYPES
            .iter()
            .find(|(name, _)| *name == self.model_type)
            .map(|(_, kind)| *kind)
    }
}

/// The feature extractor half of `processor_config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct FeatureConfig {
    pub feature_size: usize,
    pub sampling_rate: u32,
    pub n_fft: usize,
    pub win_length: usize,
    pub hop_length: usize,
    #[serde(default)]
    pub preemphasis: Option<f32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ProcessorConfig {
    pub feature_extractor: FeatureConfig,
    /// Nemotron-3.5's language names (`de-DE`, `de`, `auto`) to prompt ids.
    #[serde(default)]
    pub prompt_dictionary: HashMap<String, usize>,
}
