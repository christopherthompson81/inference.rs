use serde::Deserialize;

// NeMo's and transformers' default when a config leaves it out
const DEFAULT_MAX_SYMBOLS_PER_STEP: usize = 10;

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
}

fn default_max_symbols_per_step() -> usize {
    DEFAULT_MAX_SYMBOLS_PER_STEP
}

/// The `model_type`s a Parakeet checkpoint carries.
pub const MODEL_TYPES: [(&str, HeadKind); 3] = [
    ("parakeet_ctc", HeadKind::Ctc),
    ("parakeet_rnnt", HeadKind::Rnnt),
    ("parakeet_tdt", HeadKind::Tdt),
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
}
