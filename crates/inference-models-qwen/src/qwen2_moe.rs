//! Qwen2-MoE (Qwen1.5-MoE, Qwen2-57B-A14B): Qwen2's attention with routed experts and a gated shared expert.

use inference_quant::QuantizedConfig;
use inference_tensor::Result;

use crate::{
    decoder::{DecoderSpec, MLP, MoeRouting, MoeSpec, RopeKind},
    layers::Activation,
    moe::ExpertProjNames,
    qwen2::Qwen2AttentionType,
    serde_default_fn,
};

serde_default_fn!(bool, qkv_bias_default, true);
serde_default_fn!(usize, decoder_sparse_step_default, 1);
serde_default_fn!(usize, max_window_layers_default, 28);

#[derive(Debug, Clone, Default, serde::Serialize, inference_json::FromJson)]
pub struct Config {
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
    pub rope_theta: f64,
    #[serde(default = "qkv_bias_default")]
    pub qkv_bias: bool,
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub use_sliding_window: bool,
    #[serde(default = "max_window_layers_default")]
    pub max_window_layers: usize,
    #[serde(default)]
    pub layer_types: Option<Vec<Qwen2AttentionType>>,
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "decoder_sparse_step_default")]
    pub decoder_sparse_step: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub num_experts_per_tok: usize,
    pub num_experts: usize,
    #[serde(default)]
    pub norm_topk_prob: bool,
    #[serde(default)]
    pub mlp_only_layers: Vec<usize>,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        !self.mlp_only_layers.contains(&layer_idx)
            && self.num_experts > 0
            && (layer_idx + 1).is_multiple_of(self.decoder_sparse_step)
    }

    // transformers' default: the even layers below `max_window_layers` slide
    pub fn layer_windows(&self) -> Result<Vec<Option<usize>>> {
        let window = self
            .use_sliding_window
            .then_some(self.sliding_window)
            .flatten();
        let Some(layer_types) = &self.layer_types else {
            return Ok((0..self.num_hidden_layers)
                .map(|i| window.filter(|_| i.is_multiple_of(2) && i < self.max_window_layers))
                .collect());
        };
        if layer_types.len() != self.num_hidden_layers {
            inference_tensor::bail!(
                "Qwen2-MoE layer_types has {} entries for {} layers",
                layer_types.len(),
                self.num_hidden_layers
            );
        }
        layer_types
            .iter()
            .map(|layer_type| match layer_type {
                Qwen2AttentionType::FullAttention => Ok(None),
                Qwen2AttentionType::SlidingAttention => window.map(Some).ok_or_else(|| {
                    inference_tensor::Error::msg(
                        "Qwen2-MoE sliding_attention layer requires use_sliding_window and sliding_window",
                    )
                }),
            })
            .collect()
    }

    #[allow(clippy::cast_possible_truncation)]
    pub fn decoder_spec(&self) -> Result<DecoderSpec> {
        Ok(DecoderSpec {
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
            qkv_bias: self.qkv_bias,
            layer_windows: self.layer_windows()?,
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
                shared_expert: Some(self.shared_expert_intermediate_size),
            }),
            ..Default::default()
        })
    }
}

pub type Model = crate::decoder::CausalLm;
