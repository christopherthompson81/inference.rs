//! Qwen3-MoE: Qwen3's attention with routed experts, except on the layers that keep a dense MLP.

use inference_quant::QuantizedConfig;
use serde::{Deserialize, Serialize};

use crate::{
    decoder::{DecoderSpec, MLP, MoeRouting, MoeSpec, QkNorm, RopeKind},
    layers::Activation,
    moe::ExpertProjNames,
    serde_default_fn,
};

serde_default_fn!(bool, tie_word_embeddings, false);

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub hidden_act: Activation,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub sliding_window: Option<usize>,
    pub head_dim: Option<usize>,
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default = "tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    pub max_window_layers: usize,
    pub use_sliding_window: bool,
    pub moe_intermediate_size: usize,
    pub num_experts: usize,
    pub mlp_only_layers: Vec<usize>,
    pub decoder_sparse_step: usize,
    pub norm_topk_prob: bool,
    pub num_experts_per_tok: usize,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Only layers past `max_window_layers` slide, and only when `use_sliding_window` is set.
    pub fn layer_window(&self, layer_idx: usize) -> Option<usize> {
        self.sliding_window
            .filter(|_| self.use_sliding_window && layer_idx >= self.max_window_layers)
    }

    fn is_moe_layer(&self, layer_idx: usize) -> bool {
        !self.mlp_only_layers.contains(&layer_idx)
            && self.num_experts > 0
            && (layer_idx + 1).is_multiple_of(self.decoder_sparse_step)
    }

    #[allow(clippy::cast_possible_truncation)]
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
            rope: RopeKind::Default {
                theta: self.rope_theta as f32,
            },
            max_position_embeddings: self.max_position_embeddings,
            qk_norm: Some(QkNorm::BEFORE_ROPE),
            layer_windows: (0..self.num_hidden_layers)
                .map(|layer_idx| self.layer_window(layer_idx))
                .collect(),
            tie_word_embeddings: self.tie_word_embeddings,
            quantization_config: self.quantization_config.clone(),
            moe: Some(MoeSpec {
                num_experts: self.num_experts,
                intermediate_size: self.moe_intermediate_size,
                routing: MoeRouting::TopK {
                    k: self.num_experts_per_tok,
                    renormalize: self.norm_topk_prob,
                },
                quantized_router: false,
                expert_names: ExpertProjNames::DEFAULT,
                name: MLP,
                dense_layers: (0..self.num_hidden_layers)
                    .filter(|&layer_idx| !self.is_moe_layer(layer_idx))
                    .collect(),
                cuda_decode_graphs: true,
            }),
            ..Default::default()
        }
    }
}

pub type Model = crate::decoder::CausalLm;
