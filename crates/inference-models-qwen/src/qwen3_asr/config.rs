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
    // the original repos give `rope_theta`; transformers' `-hf` ones nest it in `rope_parameters`
    pub rope_theta: Option<f64>,
    pub rope_parameters: Option<RopeParameters>,
    pub tie_word_embeddings: bool,
    pub quantization_config: Option<QuantizedConfig>,
}

#[derive(Debug, Clone, Serialize, inference_json::FromJson)]
pub struct RopeParameters {
    pub rope_theta: f64,
}

impl TextConfig {
    fn theta(&self) -> Option<f64> {
        self.rope_theta
            .or_else(|| self.rope_parameters.as_ref().map(|rope| rope.rope_theta))
    }

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
            rope_theta: self.theta().unwrap_or_default(),
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
}

/// Where a checkpoint keeps the model's weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Layout {
    /// The released repos: everything under `thinker.`, the projector inside the audio tower.
    Original,
    /// transformers' `-hf` conversions: `model.audio_tower`, `model.multi_modal_projector`, `model.language_model`.
    Transformers,
}

impl Layout {
    pub fn audio_tower(self) -> &'static str {
        match self {
            Self::Original => "thinker.audio_tower",
            Self::Transformers => "model.audio_tower",
        }
    }

    // the projector's two linears
    pub fn projector(self) -> [&'static str; 2] {
        match self {
            Self::Original => ["thinker.audio_tower.proj1", "thinker.audio_tower.proj2"],
            Self::Transformers => [
                "model.multi_modal_projector.linear_1",
                "model.multi_modal_projector.linear_2",
            ],
        }
    }

    pub fn text_model(self) -> &'static str {
        match self {
            Self::Original => "thinker.model",
            Self::Transformers => "model.language_model",
        }
    }

    pub fn lm_head(self) -> &'static str {
        match self {
            Self::Original => "thinker.lm_head",
            Self::Transformers => "lm_head",
        }
    }
}

#[derive(inference_json::FromJson)]
struct RawConfig {
    thinker_config: Option<ThinkerConfig>,
    audio_config: Option<AudioConfig>,
    text_config: Option<TextConfig>,
    audio_token_id: Option<u32>,
}

/// Either layout's config: the original repos nest the model under `thinker_config`, the `-hf` ones keep it flat.
#[derive(Debug, Clone, Serialize)]
pub struct Config {
    pub thinker_config: ThinkerConfig,
    pub layout: Layout,
}

impl Config {
    pub fn from_json(json: &str) -> serde_json::Result<Self> {
        use serde::de::Error;
        let raw: RawConfig = serde_json::from_str(json)?;
        let (thinker_config, layout) = match raw {
            RawConfig {
                thinker_config: Some(thinker),
                ..
            } => (thinker, Layout::Original),
            RawConfig {
                audio_config: Some(audio_config),
                text_config: Some(text_config),
                audio_token_id: Some(audio_token_id),
                ..
            } => (
                ThinkerConfig {
                    audio_config,
                    text_config,
                    audio_token_id,
                },
                Layout::Transformers,
            ),
            _ => {
                return Err(serde_json::Error::custom(
                    "Qwen3-ASR config needs `thinker_config`, or `audio_config`, `text_config` and `audio_token_id`",
                ));
            }
        };
        if thinker_config.text_config.theta().is_none() {
            return Err(serde_json::Error::custom(
                "Qwen3-ASR text config needs `rope_theta` or `rope_parameters.rope_theta`",
            ));
        }
        Ok(Self {
            thinker_config,
            layout,
        })
    }
}
