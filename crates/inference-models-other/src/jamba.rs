//! Jamba (`JambaForCausalLM`: AI21's Jamba 1.x, Jamba Reasoning, Jamba2): Mamba 1 layers (with RMSNorms over dt, B
//! and C) interleaved with attention layers that have no position embedding, each followed by a SwiGLU MLP or a
//! softmax-routed MoE. Runs as Granite's hybrid model.

#![allow(clippy::cast_precision_loss)]

use inference_quant::QuantizedConfig;
use inference_tensor::Result;
use serde::Serialize;

use crate::granite::{
    BlockNames, DtBcNorm, FfnKind, FfnNames, GraniteLayerType, HybridBlock, HybridLayout,
    Mamba1Spec, MoeRouting, NemotronMoeSpec,
};
use crate::serde_default_fn;

const EMBEDDINGS: &str = "embed_tokens";
const FINAL_NORM: &str = "final_layernorm";
// transformers' `mamba_dt_rank` default: the hidden size over 16, rounded up
const AUTO: &str = "auto";
const AUTO_RANK_DIVISOR: usize = 16;

serde_default_fn!(f64, default_norm_eps, 1e-6);
serde_default_fn!(usize, default_max_position_embeddings, 262_144);
serde_default_fn!(usize, default_attn_layer_period, 8);
serde_default_fn!(usize, default_attn_layer_offset, 4);
serde_default_fn!(usize, default_expert_layer_period, 2);
serde_default_fn!(usize, default_expert_layer_offset, 1);
serde_default_fn!(usize, default_num_experts, 16);
serde_default_fn!(usize, default_experts_per_tok, 2);
serde_default_fn!(usize, default_d_state, 16);
serde_default_fn!(usize, default_d_conv, 4);
serde_default_fn!(usize, default_expand, 2);
serde_default_fn!(bool, default_true, true);

#[derive(Debug, Clone, inference_json::FromJson, Serialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    #[serde(default = "default_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_attn_layer_period")]
    pub attn_layer_period: usize,
    #[serde(default = "default_attn_layer_offset")]
    pub attn_layer_offset: usize,
    #[serde(default = "default_expert_layer_period")]
    pub expert_layer_period: usize,
    #[serde(default = "default_expert_layer_offset")]
    pub expert_layer_offset: usize,
    #[serde(default = "default_num_experts")]
    pub num_experts: usize,
    #[serde(default = "default_experts_per_tok")]
    pub num_experts_per_tok: usize,
    #[serde(default = "default_d_state")]
    pub mamba_d_state: usize,
    #[serde(default = "default_d_conv")]
    pub mamba_d_conv: usize,
    #[serde(default = "default_expand")]
    pub mamba_expand: usize,
    pub mamba_dt_rank: Option<serde_json::Value>,
    #[serde(default = "default_true")]
    pub mamba_conv_bias: bool,
    #[serde(default)]
    pub mamba_proj_bias: bool,
    pub sliding_window: Option<usize>,
    pub quantization_config: Option<QuantizedConfig>,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn inner_size(&self) -> usize {
        self.mamba_expand * self.hidden_size
    }

    pub fn dt_rank(&self) -> Result<usize> {
        match &self.mamba_dt_rank {
            None => Ok(self.hidden_size.div_ceil(AUTO_RANK_DIVISOR)),
            Some(serde_json::Value::String(auto)) if auto == AUTO => {
                Ok(self.hidden_size.div_ceil(AUTO_RANK_DIVISOR))
            }
            Some(value) => value
                .as_u64()
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(|| {
                    inference_tensor::Error::msg(format!(
                        "Jamba `mamba_dt_rank` {value} is not a size"
                    ))
                }),
        }
    }

    pub fn is_attention(&self, layer: usize) -> bool {
        layer % self.attn_layer_period == self.attn_layer_offset
    }

    pub fn is_moe(&self, layer: usize) -> bool {
        self.num_experts > 1 && layer % self.expert_layer_period == self.expert_layer_offset
    }

    pub fn moe_spec(&self) -> NemotronMoeSpec {
        NemotronMoeSpec {
            num_experts: self.num_experts,
            top_k: self.num_experts_per_tok,
            intermediate_size: self.intermediate_size,
            shared_intermediate_size: 0,
            routed_scaling_factor: 1.0,
            latent_size: None,
            routing: MoeRouting::Jamba,
        }
    }

    pub fn layout(&self) -> HybridLayout {
        let blocks = (0..self.num_hidden_layers)
            .map(|i| {
                let layer = format!("layers.{i}");
                let attention = self.is_attention(i);
                let moe = self.is_moe(i);
                HybridBlock {
                    mixer: if attention {
                        GraniteLayerType::Attention
                    } else {
                        GraniteLayerType::Mamba
                    },
                    names: BlockNames {
                        norm: format!("{layer}.input_layernorm"),
                        mixer: format!("{layer}.{}", if attention { "self_attn" } else { "mamba" }),
                        mamba: None,
                        ffn: Some(FfnNames {
                            norm: format!("{layer}.pre_ff_layernorm"),
                            module: format!("{layer}.feed_forward"),
                            kind: if moe { FfnKind::Moe } else { FfnKind::SwiGlu },
                            moe: moe.then(|| self.moe_spec()),
                        }),
                    },
                }
            })
            .collect();
        HybridLayout {
            blocks,
            embeddings: EMBEDDINGS,
            final_norm: FINAL_NORM,
        }
    }

    pub fn granite(&self) -> Result<crate::granite::Config> {
        if self.mamba_proj_bias {
            inference_tensor::bail!("Jamba with Mamba projection biases is not supported");
        }
        if self.sliding_window.is_some() {
            inference_tensor::bail!("Jamba with a sliding window is not supported");
        }
        let inner = self.inner_size();
        let layer_types = (0..self.num_hidden_layers)
            .map(|i| {
                if self.is_attention(i) {
                    GraniteLayerType::Attention
                } else {
                    GraniteLayerType::Mamba
                }
            })
            .collect();
        Ok(crate::granite::Config {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            shared_intermediate_size: None,
            vocab_size: self.vocab_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: 0.0,
            max_position_embeddings: self.max_position_embeddings,
            rope_scaling: None,
            quantization_config: self.quantization_config.clone(),
            tie_word_embeddings: self.tie_word_embeddings,
            layer_types,
            attention_multiplier: 1.0 / (self.head_dim() as f32).sqrt(),
            embedding_multiplier: 1.0,
            residual_multiplier: 1.0,
            logits_scaling: 1.0,
            // Mamba 1: each channel a head of width one, all in one group
            mamba_n_heads: Some(inner),
            mamba_n_groups: 1,
            mamba_d_state: self.mamba_d_state,
            mamba_d_head: Some(1),
            mamba_d_conv: self.mamba_d_conv,
            mamba_expand: self.mamba_expand,
            mamba_chunk_size: 0,
            mamba_conv_bias: self.mamba_conv_bias,
            mamba_proj_bias: self.mamba_proj_bias,
            num_local_experts: 0,
            num_experts_per_tok: self.num_experts_per_tok,
            position_embedding_type: "nope".to_string(),
            sliding_window: None,
            layer_rope_theta: None,
            no_rope_layers: None,
            rope_parameters: None,
            head_dim: Some(self.head_dim()),
            mamba_inner_size: Some(inner),
            mamba_norm_group_size: None,
            mamba_dt_limit: None,
            mamba_f32_state: true,
            mamba_gated_norm: false,
            mup: None,
            mamba1: Some(Mamba1Spec {
                dt_rank: self.dt_rank()?,
                dt_b_c_norm: Some(DtBcNorm {
                    eps: self.rms_norm_eps,
                    weighted: true,
                }),
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(extra: &str) -> Config {
        Config::from_json(&format!(
            r#"{{"vocab_size": 16, "hidden_size": 64, "intermediate_size": 96, "num_hidden_layers": 8,
                "num_attention_heads": 4, "num_key_value_heads": 2{extra}}}"#
        ))
        .unwrap()
    }

    // transformers' defaults: attention every 8 layers from 4, experts every other layer from 1
    #[test]
    fn attention_and_experts_follow_their_periods() {
        let layout = config("").layout();
        let kinds = layout
            .blocks
            .iter()
            .map(|b| (b.mixer.clone(), b.names.ffn.as_ref().unwrap().kind))
            .collect::<Vec<_>>();
        assert_eq!(kinds[4], (GraniteLayerType::Attention, FfnKind::SwiGlu));
        assert_eq!(layout.blocks[4].names.mixer, "layers.4.self_attn");
        assert_eq!(layout.blocks[1].names.mixer, "layers.1.mamba");
        assert_eq!(kinds[1].1, FfnKind::Moe);
        assert_eq!(kinds[2].1, FfnKind::SwiGlu);
        // one expert is a dense MLP everywhere, as Jamba Reasoning 3B has it
        let dense = config(r#", "num_experts": 1"#).layout();
        assert!(
            dense
                .blocks
                .iter()
                .all(|b| b.names.ffn.as_ref().unwrap().kind == FfnKind::SwiGlu)
        );
        let spec = config("").granite().unwrap().mamba1.unwrap();
        assert_eq!(spec.dt_rank, 4);
        assert!(spec.dt_b_c_norm.unwrap().weighted);
    }
}
