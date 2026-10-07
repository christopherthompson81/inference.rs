use std::collections::HashMap;

use inference_quant::{QuantizedConfig, StaticLoraConfig};
use serde::{Deserialize, Serialize};

use crate::{
    conformer::config::ConformerEncoderConfig,
    decoder::{DecoderSpec, MlpKind, RopeKind},
    layers::{Activation, PhiRopeConfig, PhiRopeScalingConfig, ScaledRopeType},
};

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Phi4MMScaledRopeType {
    #[serde(alias = "longrope")]
    LongRope,
    #[default]
    Default,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Phi4MMRopeScalingConfig {
    short_factor: Option<Vec<f64>>,
    long_factor: Option<Vec<f64>>,
    #[serde(rename = "type")]
    scaling_type: Phi4MMScaledRopeType,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Phi4MMImageEmbedConfig {
    pub n_embd: Option<usize>,
    pub crop_size: Option<usize>,
    pub embedding_cls: String,
    pub enable_gradient_checkpointing: bool,
    pub hd_transform_order: Option<String>,
    pub image_token_compression_cls: Option<String>,
    pub projection_cls: Option<String>,
    pub use_hd_transform: Option<bool>,
    pub with_learnable_separator: Option<bool>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Phi4MMAudioEmbedConfig {
    pub n_embd: Option<usize>,
    pub compression_rate: usize,
    pub downsample_rate: usize,
    pub embedding_cls: String,
    pub projection_cls: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Phi4MMEmbdLayerConfig {
    pub image_embd_layer: Option<Phi4MMImageEmbedConfig>,
    pub audio_embd_layer: Option<Phi4MMAudioEmbedConfig>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Phi4MMImgProcessorConfig {
    pub layer_idx: Option<isize>,
    pub type_feature: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Phi4MMAudioConfig {
    pub config: ConformerEncoderConfig,
    pub name: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Phi4MMConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub resid_pdrop: f64,
    pub embd_pdrop: f64,
    pub attention_dropout: f64,
    pub hidden_act: Activation,
    pub max_position_embeddings: usize,
    pub original_max_position_embeddings: usize,
    pub initializer_range: f64,
    pub rms_norm_eps: f64,
    pub use_cache: bool,
    pub tie_word_embeddings: bool,
    pub rope_theta: f64,
    pub rope_scaling: Option<Phi4MMRopeScalingConfig>,
    pub partial_rotary_factor: f64,
    pub bos_token_id: usize,
    pub eos_token_id: usize,
    pub pad_token_id: usize,
    pub image_input_id: Option<f64>,
    pub sliding_window: Option<usize>,
    pub embd_layer: Phi4MMEmbdLayerConfig,
    pub img_processor: Option<Phi4MMImgProcessorConfig>,
    pub audio_processor: Option<Phi4MMAudioConfig>,
    pub vision_lora: StaticLoraConfig,
    pub speech_lora: StaticLoraConfig,
    pub quantization_config: Option<QuantizedConfig>,
}

impl Phi4MMConfig {
    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Phi's LongRoPE over the rotated part, when the config names both factor lists; plain RoPE otherwise.
    fn rope_config(&self) -> PhiRopeConfig {
        let rope_scaling = match &self.rope_scaling {
            Some(Phi4MMRopeScalingConfig {
                scaling_type: Phi4MMScaledRopeType::LongRope,
                short_factor: Some(short_factor),
                long_factor: Some(long_factor),
            }) => Some(PhiRopeScalingConfig::Classic {
                short_factor: short_factor.clone(),
                long_factor: long_factor.clone(),
                scaling_type: ScaledRopeType::Su,
            }),
            _ => None,
        };
        PhiRopeConfig {
            rope_scaling,
            scaling_attn_factor: None,
            max_position_embeddings: self.max_position_embeddings,
            original_max_position_embeddings: self.original_max_position_embeddings,
            rope_theta: self.rope_theta,
            head_dim: self.head_dim(),
            partial_rotary_factor: Some(self.partial_rotary_factor),
        }
    }

    pub fn decoder_spec(&self) -> DecoderSpec {
        DecoderSpec {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_heads: self.num_attention_heads,
            num_kv_heads: self.num_key_value_heads(),
            head_dim: self.head_dim(),
            hidden_act: self.hidden_act,
            rms_norm_eps: self.rms_norm_eps,
            rope: RopeKind::Phi(self.rope_config()),
            max_position_embeddings: self.max_position_embeddings,
            layer_windows: vec![self.sliding_window; self.num_hidden_layers],
            tie_word_embeddings: self.tie_word_embeddings,
            quantization_config: self.quantization_config.clone(),
            fused_qkv: true,
            static_loras: Some(self.loras()),
            mlp: MlpKind::FusedGateUp,
            ..Default::default()
        }
    }

    pub fn loras(&self) -> HashMap<String, StaticLoraConfig> {
        let mut accum = HashMap::new();
        // Add all the loras
        // accum.insert("speech".to_string(), self.speech_lora.clone());
        accum.insert("vision".to_string(), self.vision_lora.clone());
        accum
    }
}
