use inference_quant::QuantizedConfig;
use serde::Deserialize;

use crate::deepseek_family::{
    FamilyConfig, FamilyModel, MlaAttention, MlaConfig, MlaKvLayout, MoeSpec, SharedExpert,
    mla_softmax_scale,
};
use crate::{
    layers::Activation,
    moe::{GroupedRouterConfig, RouterMethod, RouterRenorm, RouterScoring},
    serde_default_fn,
};

serde_default_fn!(f64, routed_scaling_factor, 1.0);
serde_default_fn!(usize, moe_layer_freq, 1);
serde_default_fn!(usize, first_k_dense_replace, 0);
serde_default_fn!(Activation, hidden_act, Activation::Silu);
serde_default_fn!(bool, tie_word_embeddings, false);
serde_default_fn!(usize, n_group, 1);
serde_default_fn!(usize, topk_group, 1);

#[derive(Deserialize, Clone, Debug)]
pub struct Glm4MoeLiteConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    #[serde(default = "first_k_dense_replace")]
    pub first_k_dense_replace: usize,
    #[serde(default = "routed_scaling_factor")]
    pub routed_scaling_factor: f64,
    #[serde(default = "n_group")]
    pub n_group: usize,
    #[serde(default = "topk_group")]
    pub topk_group: usize,
    #[serde(
        default = "moe_layer_freq",
        deserialize_with = "crate::deepseek_family::nonzero_moe_layer_freq"
    )]
    pub moe_layer_freq: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    #[serde(default = "hidden_act")]
    pub hidden_act: Activation,
    #[serde(default = "tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    #[serde(alias = "quantization")]
    pub quantization_config: Option<QuantizedConfig>,
}

impl Glm4MoeLiteConfig {
    fn router_config(&self) -> GroupedRouterConfig {
        GroupedRouterConfig {
            scoring: RouterScoring::Sigmoid,
            method: RouterMethod::NoAuxTc,
            n_group: self.n_group,
            topk_group: self.topk_group,
            routed_scaling_factor: self.routed_scaling_factor,
            renorm: RouterRenorm::Always,
        }
    }

    pub fn q_head_dim(&self) -> usize {
        self.qk_rope_head_dim + self.qk_nope_head_dim
    }

    pub fn family(&self) -> FamilyConfig<MlaConfig> {
        FamilyConfig {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            max_position_embeddings: self.max_position_embeddings,
            rms_norm_eps: self.rms_norm_eps,
            tie_word_embeddings: self.tie_word_embeddings,
            hidden_act: self.hidden_act,
            quantization_config: self.quantization_config.clone(),
            moe: Some(MoeSpec {
                n_routed_experts: self.n_routed_experts,
                num_experts_per_tok: Some(self.num_experts_per_tok),
                moe_intermediate_size: self.moe_intermediate_size,
                first_k_dense_replace: self.first_k_dense_replace,
                moe_layer_freq: Some(self.moe_layer_freq),
                shared_expert: (self.n_shared_experts > 0).then_some(SharedExpert::Replicated(
                    self.moe_intermediate_size * self.n_shared_experts,
                )),
                router: self.router_config(),
            }),
            attn: MlaConfig {
                q_lora_rank: Some(self.q_lora_rank),
                kv_lora_rank: self.kv_lora_rank,
                qk_nope_head_dim: self.qk_nope_head_dim,
                qk_rope_head_dim: self.qk_rope_head_dim,
                v_head_dim: self.v_head_dim,
                attention_bias: false,
                softmax_scale: mla_softmax_scale(self.q_head_dim(), None),
                rope_theta: self.rope_theta,
                rope_scaling: None,
                kv_layout: MlaKvLayout::PagedOnCudaDevice,
                label: "GLM4 MoE",
            },
        }
    }
}

pub type Glm4MoeLite = FamilyModel<MlaAttention>;

#[cfg(test)]
#[path = "deepseek_family_tests/glm4_moe_lite.rs"]
mod family_tests;
