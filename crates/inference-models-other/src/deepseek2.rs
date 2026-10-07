use inference_quant::QuantizedConfig;
use serde::Deserialize;

use crate::deepseek_family::{
    FamilyConfig, FamilyModel, MlaAttention, MlaConfig, MoeSpec, SharedExpert, mla_softmax_scale,
};
use crate::{
    layers::{Activation, DeepSeekV2RopeScaling},
    moe::{GroupedRouterConfig, RouterMethod, RouterRenorm, RouterScoring},
    serde_default_fn,
};

serde_default_fn!(f64, routed_scaling_factor, 1.0);
serde_default_fn!(TopkMethod, topk_method, TopkMethod::Greedy);
serde_default_fn!(usize, moe_layer_freq, 1);
serde_default_fn!(usize, first_k_dense_replace, 0);
serde_default_fn!(bool, norm_topk_prob, false);
serde_default_fn!(ScoringFunc, scoring_func, ScoringFunc::Softmax);
serde_default_fn!(Activation, hidden_act, Activation::Silu);
serde_default_fn!(bool, tie_word_embeddings, false);

#[derive(Deserialize, Clone, Debug)]
enum TopkMethod {
    #[serde(rename = "greedy")]
    Greedy,
    #[serde(rename = "group_limited_greedy")]
    GroupLimitedGreedy,
}

#[derive(Deserialize, Clone, Debug)]
enum ScoringFunc {
    #[serde(rename = "softmax")]
    Softmax,
}

#[derive(Deserialize, Clone, Debug)]
pub struct DeepSeekV2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub n_shared_experts: Option<usize>,
    pub n_routed_experts: Option<usize>,
    #[serde(default = "routed_scaling_factor")]
    pub routed_scaling_factor: f64,
    #[serde(default = "topk_method")]
    topk_method: TopkMethod,
    pub num_experts_per_tok: Option<usize>,
    #[serde(
        default = "moe_layer_freq",
        deserialize_with = "crate::deepseek_family::nonzero_moe_layer_freq"
    )]
    pub moe_layer_freq: usize,
    #[serde(default = "first_k_dense_replace")]
    pub first_k_dense_replace: usize,
    // k dense layers
    #[serde(default = "norm_topk_prob")]
    pub norm_topk_prob: bool,
    #[serde(default = "scoring_func")]
    scoring_func: ScoringFunc,
    #[serde(default = "hidden_act")]
    pub hidden_act: Activation,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    #[serde(default = "tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    pub rope_theta: f32,
    pub rope_scaling: Option<DeepSeekV2RopeScaling>,
    pub attention_bias: bool,
    pub q_lora_rank: Option<usize>,
    pub qk_rope_head_dim: usize,
    pub kv_lora_rank: usize,
    pub v_head_dim: usize,
    pub qk_nope_head_dim: usize,
    pub quantization_config: Option<QuantizedConfig>,
    pub n_group: usize,
    pub topk_group: usize,
}

impl DeepSeekV2Config {
    pub(crate) fn router_config(&self) -> GroupedRouterConfig {
        GroupedRouterConfig {
            scoring: match self.scoring_func {
                ScoringFunc::Softmax => RouterScoring::Softmax,
            },
            method: match self.topk_method {
                TopkMethod::Greedy => RouterMethod::Greedy,
                TopkMethod::GroupLimitedGreedy => RouterMethod::GroupLimitedGreedy,
            },
            n_group: self.n_group,
            topk_group: self.topk_group,
            routed_scaling_factor: self.routed_scaling_factor,
            renorm: RouterRenorm::TopkProbSkipsScale {
                norm_topk_prob: self.norm_topk_prob,
            },
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
            moe: self.n_routed_experts.map(|n_routed_experts| MoeSpec {
                n_routed_experts,
                num_experts_per_tok: self.num_experts_per_tok,
                moe_intermediate_size: self.moe_intermediate_size,
                first_k_dense_replace: self.first_k_dense_replace,
                moe_layer_freq: Some(self.moe_layer_freq),
                shared_expert: self
                    .n_shared_experts
                    .map(|n| SharedExpert::Sharded(self.moe_intermediate_size * n)),
                router: self.router_config(),
            }),
            attn: MlaConfig {
                q_lora_rank: self.q_lora_rank,
                kv_lora_rank: self.kv_lora_rank,
                qk_nope_head_dim: self.qk_nope_head_dim,
                qk_rope_head_dim: self.qk_rope_head_dim,
                v_head_dim: self.v_head_dim,
                attention_bias: self.attention_bias,
                softmax_scale: mla_softmax_scale(self.q_head_dim(), self.rope_scaling.as_ref()),
                rope_theta: self.rope_theta,
                rope_scaling: self.rope_scaling.clone(),
                label: "DeepSeek",
            },
        }
    }
}

pub type DeepSeekV2 = FamilyModel<MlaAttention>;

#[cfg(test)]
#[path = "deepseek_family_tests/deepseek2.rs"]
mod family_tests;
