#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use inference_quant::QuantizedConfig;

use crate::{
    amoe::AnyMoeLoraTarget,
    decoder::{DecoderSpec, MlpKind, NormKind, NormNames, RopeKind},
    layers::Activation,
    serde_default_fn,
};

serde_default_fn!(bool, word_emb_default, false);

const MLP_PROJECTIONS: &[AnyMoeLoraTarget; 2] = &[
    AnyMoeLoraTarget::up("c_fc"),
    AnyMoeLoraTarget::down("c_proj"),
];

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize, Default)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub hidden_act: Activation,
    pub max_position_embeddings: usize,
    pub norm_epsilon: f64,
    pub rope_theta: f64,
    pub use_bias: bool,
    pub sliding_window: Option<usize>,
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default = "word_emb_default")]
    pub tie_word_embeddings: bool,
}

impl Config {
    pub fn decoder_spec(&self) -> DecoderSpec {
        DecoderSpec {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_heads: self.num_attention_heads,
            num_kv_heads: self.num_key_value_heads,
            head_dim: self.hidden_size / self.num_attention_heads,
            hidden_act: self.hidden_act,
            rms_norm_eps: self.norm_epsilon,
            rope: RopeKind::Default {
                theta: self.rope_theta as f32,
            },
            max_position_embeddings: self.max_position_embeddings,
            qkv_bias: self.use_bias,
            o_bias: self.use_bias,
            layer_windows: vec![self.sliding_window; self.num_hidden_layers],
            tie_word_embeddings: self.tie_word_embeddings,
            quantization_config: self.quantization_config.clone(),
            norm: NormKind::Layer,
            norm_names: NormNames::PRE,
            mlp: MlpKind::Plain {
                projections: MLP_PROJECTIONS,
                bias: self.use_bias,
            },
            ..Default::default()
        }
    }
}

pub type Model = crate::decoder::CausalLm;
