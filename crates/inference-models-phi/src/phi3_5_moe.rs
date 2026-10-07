//! Phi-3.5-MoE: LayerNorm layers with biases, Phi's LongRoPE, and sparsemixer routing over `block_sparse_moe`.

use inference_quant::QuantizedConfig;

use crate::{
    decoder::{BLOCK_SPARSE_MOE, DecoderSpec, MoeRouting, MoeSpec, NormKind, RopeKind},
    layers::{Activation, PhiRopeConfig, PhiRopeScalingConfig},
    moe::ExpertProjNames,
    serde_default_fn,
};

serde_default_fn!(bool, word_emb_default, false);

// https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/config.json
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize, Default)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_act: Activation,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub rope_scaling: Option<PhiRopeScalingConfig>,
    #[serde(default)]
    pub rope_scaling_attn_factor: Option<f64>,
    pub max_position_embeddings: usize,
    pub sliding_window: Option<usize>,
    pub original_max_position_embeddings: usize,

    pub quantization_config: Option<QuantizedConfig>,
    pub lm_head_bias: bool,
    pub attention_bias: bool,
    pub num_local_experts: usize,
    pub router_jitter_noise: f64,
    #[serde(default = "word_emb_default")]
    pub tie_word_embeddings: bool,
}

impl From<Config> for PhiRopeConfig {
    fn from(val: Config) -> Self {
        PhiRopeConfig {
            rope_scaling: val.rope_scaling,
            scaling_attn_factor: val.rope_scaling_attn_factor,
            max_position_embeddings: val.max_position_embeddings,
            original_max_position_embeddings: val.original_max_position_embeddings,
            rope_theta: val.rope_theta,
            head_dim: val.hidden_size / val.num_attention_heads,
            partial_rotary_factor: None,
        }
    }
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn decoder_spec(&self) -> DecoderSpec {
        DecoderSpec {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_heads: self.num_attention_heads,
            num_kv_heads: self.num_key_value_heads,
            head_dim: self.head_dim(),
            hidden_act: self.hidden_act,
            rms_norm_eps: self.rms_norm_eps,
            rope: RopeKind::Phi(PhiRopeConfig::from(self.clone())),
            max_position_embeddings: self.max_position_embeddings,
            qkv_bias: self.attention_bias,
            o_bias: self.attention_bias,
            layer_windows: vec![self.sliding_window; self.num_hidden_layers],
            tie_word_embeddings: self.tie_word_embeddings,
            quantization_config: self.quantization_config.clone(),
            norm: NormKind::Layer,
            moe: Some(MoeSpec {
                num_experts: self.num_local_experts,
                intermediate_size: self.intermediate_size,
                routing: MoeRouting::SparseMixer {
                    jitter_eps: self.router_jitter_noise,
                },
                quantized_router: false,
                expert_names: ExpertProjNames::MIXTRAL,
                name: BLOCK_SPARSE_MOE,
                dense_layers: Vec::new(),
                cuda_decode_graphs: false,
            }),
            lm_head_bias: self.lm_head_bias,
            ..Default::default()
        }
    }
}

pub type Model = crate::decoder::CausalLm;
