use inference_quant::QuantizedConfig;

use inference_tensor::Result;

use super::encoder::EncoderNames;
use crate::layers::Activation;
use crate::qwen2;
use crate::qwen2_audio::config::TextConfig;

// Microsoft's offline checkpoint class (`microsoft/VibeVoice-ASR`), in its own layout rather than transformers'
pub const UPSTREAM_ARCHITECTURE: &str = "VibeVoiceForASRTraining";
// the upstream encoders' fixed sizes, which their config does not state
const ENCODER_KERNEL: usize = 7;
const ENCODER_FFN_EXPANSION: usize = 4;
const DEPTH_SEPARATOR: char = '-';
// upstream's speech placeholder, `<|box_start|>` in Qwen2.5's vocabulary, which transformers' config names
const UPSTREAM_AUDIO_TOKEN: u32 = 151_648;
// a minute at 24 kHz, the encoders' chunk in both layouts
const UPSTREAM_CHUNK_SAMPLES: usize = 1_440_000;

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
pub(crate) struct UpstreamTokenizer {
    channels: usize,
    encoder_ratios: Vec<usize>,
    encoder_depths: String,
    encoder_n_filters: usize,
    vae_dim: usize,
    layernorm_eps: f64,
}

impl UpstreamTokenizer {
    // as transformers' port names it: its ratios run stem-first, the reverse of how the upstream config lists them
    pub(crate) fn encoder(&self) -> Result<EncoderConfig> {
        let depths = self
            .encoder_depths
            .split(DEPTH_SEPARATOR)
            .map(|d| d.trim().parse::<usize>())
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| inference_tensor::Error::Msg(format!("`encoder_depths`: {e}")))?;
        Ok(EncoderConfig {
            channels: self.channels,
            depths,
            downsampling_ratios: self.encoder_ratios.iter().rev().copied().collect(),
            ffn_expansion: ENCODER_FFN_EXPANSION,
            hidden_act: Activation::Gelu,
            hidden_size: self.vae_dim,
            kernel_size: ENCODER_KERNEL,
            num_filters: self.encoder_n_filters,
            rms_norm_eps: self.layernorm_eps,
        })
    }
}

// Microsoft's offline config: the encoders as the streaming checkpoints list them, the decoder beside them
#[derive(Debug, Clone, inference_json::FromJson)]
struct UpstreamConfig {
    acoustic_tokenizer_config: UpstreamTokenizer,
    semantic_tokenizer_config: UpstreamTokenizer,
    decoder_config: TextConfig,
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
    /// Which layout the weights are in.
    #[serde(skip)]
    pub names: EncoderNames,
}

impl VibeVoiceAsrConfig {
    /// A `config.json` of either layout: transformers' port or Microsoft's own.
    pub fn parse(config: &str) -> Result<Self> {
        let value: serde_json::Value =
            serde_json::from_str(config).map_err(inference_tensor::Error::wrap)?;
        let upstream = value["architectures"]
            .as_array()
            .is_some_and(|a| a.iter().any(|a| a.as_str() == Some(UPSTREAM_ARCHITECTURE)));
        if !upstream {
            return serde_json::from_str(config).map_err(inference_tensor::Error::wrap);
        }
        let cfg: UpstreamConfig =
            serde_json::from_str(config).map_err(inference_tensor::Error::wrap)?;
        Ok(Self {
            acoustic_tokenizer_encoder_config: cfg.acoustic_tokenizer_config.encoder()?,
            semantic_tokenizer_encoder_config: cfg.semantic_tokenizer_config.encoder()?,
            text_config: cfg.decoder_config,
            audio_token_id: UPSTREAM_AUDIO_TOKEN,
            acoustic_tokenizer_chunk_size: UPSTREAM_CHUNK_SAMPLES,
            quantization_config: None,
            names: EncoderNames::Upstream,
        })
    }

    pub fn text(&self) -> qwen2::Config {
        self.text_config.qwen2(self.quantization_config.clone())
    }

    pub fn hop_length(&self) -> usize {
        self.acoustic_tokenizer_encoder_config.hop_length()
    }

    /// Where the decoder's stack and its head are.
    pub fn text_prefixes(&self) -> (&'static str, &'static str) {
        match self.names {
            EncoderNames::Transformers => ("language_model.model", "language_model.lm_head"),
            EncoderNames::Upstream => ("model.language_model", "lm_head"),
        }
    }
}
