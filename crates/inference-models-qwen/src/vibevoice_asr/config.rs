use inference_quant::QuantizedConfig;

use crate::layers::Activation;
use crate::qwen2;
use crate::qwen2_audio::config::TextConfig;

/// One of the two speech-tokenizer encoders (`VibeVoiceAcousticTokenizerEncoderConfig`): a causal conv stem, strided
/// causal convs between ConvNeXt-1d stages, and a head conv to the latent width.
#[derive(Debug, Clone, inference_json::FromJson)]
pub struct EncoderConfig {
    pub channels: usize,
    pub depths: Vec<usize>,
    pub downsampling_ratios: Vec<usize>,
    pub ffn_expansion: usize,
    pub hidden_act: Activation,
    pub hidden_size: usize,
    pub kernel_size: usize,
    pub num_filters: usize,
    pub rms_norm_eps: f64,
}

impl EncoderConfig {
    /// Samples per latent frame: the product of the downsampling ratios.
    pub fn hop_length(&self) -> usize {
        self.downsampling_ratios.iter().product()
    }
}

#[derive(Debug, Clone, inference_json::FromJson)]
pub struct VibeVoiceAsrConfig {
    pub acoustic_tokenizer_encoder_config: EncoderConfig,
    pub semantic_tokenizer_encoder_config: EncoderConfig,
    pub text_config: TextConfig,
    pub audio_token_id: u32,
    /// Samples the encoders take at once, their causal context carried between; a multiple of the hop.
    pub acoustic_tokenizer_chunk_size: usize,
    #[serde(default)]
    pub quantization_config: Option<QuantizedConfig>,
}

impl VibeVoiceAsrConfig {
    pub fn text(&self) -> qwen2::Config {
        self.text_config.qwen2(self.quantization_config.clone())
    }

    pub fn hop_length(&self) -> usize {
        self.acoustic_tokenizer_encoder_config.hop_length()
    }
}
