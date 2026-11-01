use inference_quant::QuantizedConfig;

use super::inputs_processor::AUDIO_TOKEN;
use crate::{
    audio::whisper::{
        WhisperEncoderConfig, WhisperLayerConfig, WhisperLayerNames, WhisperPositions,
    },
    layers::Activation,
    llama, serde_default_fn,
};

const LAYER_NORM_EPS: f64 = 1e-5;
const TEXT_CONFIG_KEY: &str = "text_config";
const AUDIO_TOKEN_ID_KEY: &str = "audio_token_id";
// the projector's RMSNorms (Ultravox's `RMSNorm` default)
serde_default_fn!(f64, default_norm_eps, 1e-6);

#[derive(Debug, Clone, inference_json::FromJson)]
pub struct AudioConfig {
    #[serde(alias = "d_model")]
    pub hidden_size: usize,
    #[serde(alias = "encoder_ffn_dim")]
    pub intermediate_size: usize,
    #[serde(alias = "encoder_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(alias = "encoder_layers")]
    pub num_hidden_layers: usize,
    pub num_mel_bins: usize,
    pub max_source_positions: usize,
}

/// Ultravox's tower, projector and text model: synthesized from the GGUFs, or a safetensors repo's config with its
/// `text_model_id`'s merged in ([`with_text_config`]).
#[derive(Debug, Clone, inference_json::FromJson)]
pub struct UltravoxConfig {
    pub audio_config: AudioConfig,
    pub text_config: llama::Config,
    pub stack_factor: usize,
    /// The projector's inner width, halved by its SwiGLU.
    pub hidden_size: usize,
    pub audio_token_id: u32,
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f64,
    #[serde(default)]
    pub quantization_config: Option<QuantizedConfig>,
}

impl UltravoxConfig {
    pub fn encoder(&self) -> WhisperEncoderConfig {
        let ac = &self.audio_config;
        WhisperEncoderConfig {
            layer: WhisperLayerConfig {
                d_model: ac.hidden_size,
                num_heads: ac.num_attention_heads,
                ffn_dim: ac.intermediate_size,
                activation: Activation::Gelu,
                layer_norm_eps: LAYER_NORM_EPS,
                k_bias: false,
                names: WhisperLayerNames::Whisper,
            },
            num_layers: ac.num_hidden_layers,
            num_mel_bins: ac.num_mel_bins,
            max_source_positions: ac.max_source_positions,
            positions: WhisperPositions::Learned,
        }
    }

    pub fn stacked_width(&self) -> usize {
        self.audio_config.hidden_size * self.stack_factor
    }

    pub fn text_config(&self) -> llama::Config {
        let mut text = self.text_config.clone();
        if text.quantization_config.is_none() {
            text.quantization_config = self.quantization_config.clone();
        }
        text
    }
}

/// A safetensors repo's config with the config of the text model it names, and the placeholder's id in `tokenizer`:
/// the repo's own `<|audio|>` lies past the text model's embeddings.
pub fn with_text_config(config: &str, text: &str, tokenizer: &str) -> anyhow::Result<String> {
    let mut config: serde_json::Value = serde_json::from_str(config)?;
    let tokenizer: serde_json::Value = serde_json::from_str(tokenizer)?;
    let audio_token = tokenizer["added_tokens"]
        .as_array()
        .and_then(|tokens| {
            tokens
                .iter()
                .find(|token| token["content"] == AUDIO_TOKEN)
                .and_then(|token| token["id"].as_u64())
        })
        .ok_or_else(|| anyhow::anyhow!("Ultravox's tokenizer has no `{AUDIO_TOKEN}`"))?;
    config[TEXT_CONFIG_KEY] = serde_json::from_str(text)?;
    config[AUDIO_TOKEN_ID_KEY] = audio_token.into();
    Ok(config.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_repo_config_takes_its_text_model_and_the_placeholder_id() {
        let config = r#"{"architectures": ["UltravoxModel"], "text_model_id": "org/text", "stack_factor": 8,
            "hidden_size": 16, "audio_token_index": 300, "audio_config": {"d_model": 4, "encoder_layers": 1,
            "encoder_attention_heads": 1, "encoder_ffn_dim": 8, "num_mel_bins": 128, "max_source_positions": 10}}"#;
        let text = r#"{"hidden_size": 8, "intermediate_size": 16, "vocab_size": 300, "num_hidden_layers": 1,
            "num_attention_heads": 2, "num_key_value_heads": 1, "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
            "max_position_embeddings": 64, "tie_word_embeddings": true, "hidden_act": "silu"}"#;
        let tokenizer = format!(
            r#"{{"added_tokens": [{{"id": 7, "content": "{AUDIO_TOKEN}"}}, {{"id": 300, "content": "<|audio|>"}}]}}"#
        );
        let merged =
            UltravoxConfig::from_json(&with_text_config(config, text, &tokenizer).unwrap())
                .unwrap();
        assert_eq!(merged.audio_token_id, 7);
        assert_eq!(merged.text_config.hidden_size, 8);
        assert_eq!(merged.audio_config.hidden_size, 4);
    }
}
