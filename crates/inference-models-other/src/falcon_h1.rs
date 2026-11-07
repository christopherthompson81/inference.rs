//! Falcon-H1: every layer runs attention and a Mamba-2 mixer side by side on one norm and sums them into the
//! residual, then a SwiGLU MLP; muP multipliers scale the activations. Runs as Granite's hybrid model with parallel
//! blocks. A GGUF has the multipliers folded into its weights.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use inference_quant::QuantizedConfig;
use inference_tensor::Result;
use serde::Serialize;

use crate::granite::{
    BlockNames, FfnKind, FfnNames, GraniteLayerType, GraniteRopeParameters, HybridBlock,
    HybridLayout, MupMultipliers,
};
use crate::serde_default_fn;

const EMBEDDINGS: &str = "embed_tokens";
const FINAL_NORM: &str = "final_layernorm";
// transformers' `mamba_d_head` default: the inner width over the heads
const AUTO: &str = "auto";

serde_default_fn!(f64, default_norm_eps, 1e-5);
serde_default_fn!(usize, default_max_position_embeddings, 8192);
serde_default_fn!(usize, default_mamba_d_ssm, 1024);
serde_default_fn!(usize, default_mamba_n_heads, 128);
serde_default_fn!(usize, default_one, 1);
serde_default_fn!(usize, default_d_state, 256);
serde_default_fn!(usize, default_d_conv, 4);
serde_default_fn!(usize, default_expand, 2);
serde_default_fn!(usize, default_chunk_size, 256);
serde_default_fn!(bool, default_true, true);
serde_default_fn!(f32, default_multiplier, 1.0);
serde_default_fn!(f32, default_rope_theta, 10_000.0);
serde_default_fn!([f32; 5], default_ssm_multipliers, [1.0; 5]);
serde_default_fn!([f32; 2], default_mlp_multipliers, [1.0; 2]);

#[derive(Debug, Clone, inference_json::FromJson, Serialize)]
pub struct Config {
    pub vocab_size: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub head_dim: Option<usize>,
    #[serde(default = "default_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    pub rope_parameters: Option<GraniteRopeParameters>,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
    #[serde(default)]
    pub projectors_bias: bool,
    #[serde(default = "default_mamba_d_ssm")]
    pub mamba_d_ssm: usize,
    #[serde(default = "default_mamba_n_heads")]
    pub mamba_n_heads: usize,
    pub mamba_d_head: Option<serde_json::Value>,
    #[serde(default = "default_one")]
    pub mamba_n_groups: usize,
    #[serde(default = "default_d_state")]
    pub mamba_d_state: usize,
    #[serde(default = "default_d_conv")]
    pub mamba_d_conv: usize,
    #[serde(default = "default_expand")]
    pub mamba_expand: usize,
    #[serde(default = "default_chunk_size")]
    pub mamba_chunk_size: usize,
    #[serde(default = "default_true")]
    pub mamba_conv_bias: bool,
    #[serde(default)]
    pub mamba_proj_bias: bool,
    #[serde(default = "default_true")]
    pub mamba_norm_before_gate: bool,
    #[serde(default)]
    pub mamba_rms_norm: bool,
    #[serde(default = "default_multiplier")]
    pub embedding_multiplier: f32,
    #[serde(default = "default_multiplier")]
    pub lm_head_multiplier: f32,
    #[serde(default = "default_multiplier")]
    pub attention_in_multiplier: f32,
    #[serde(default = "default_multiplier")]
    pub attention_out_multiplier: f32,
    #[serde(default = "default_multiplier")]
    pub key_multiplier: f32,
    #[serde(default = "default_multiplier")]
    pub ssm_in_multiplier: f32,
    #[serde(default = "default_multiplier")]
    pub ssm_out_multiplier: f32,
    #[serde(default = "default_ssm_multipliers")]
    pub ssm_multipliers: [f32; 5],
    #[serde(default = "default_mlp_multipliers")]
    pub mlp_multipliers: [f32; 2],
    pub quantization_config: Option<QuantizedConfig>,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn mamba_d_head(&self) -> Result<usize> {
        match &self.mamba_d_head {
            None => Ok(self.mamba_d_ssm / self.mamba_n_heads),
            Some(serde_json::Value::String(auto)) if auto == AUTO => {
                Ok(self.mamba_d_ssm / self.mamba_n_heads)
            }
            Some(value) => value.as_u64().map(|v| v as usize).ok_or_else(|| {
                inference_tensor::Error::msg(format!(
                    "Falcon-H1 `mamba_d_head` {value} is not a size"
                ))
            }),
        }
    }

    fn rope_theta(&self) -> f32 {
        self.rope_parameters
            .as_ref()
            .and_then(|p| p.rope_theta)
            .unwrap_or(self.rope_theta)
    }

    // all ones once a GGUF folded them into the weights
    fn mup(&self) -> Option<MupMultipliers> {
        let mup = MupMultipliers {
            attention_in: self.attention_in_multiplier,
            attention_out: self.attention_out_multiplier,
            key: self.key_multiplier,
            ssm_in: self.ssm_in_multiplier,
            ssm_out: self.ssm_out_multiplier,
            ssm: self.ssm_multipliers,
            mlp: self.mlp_multipliers,
        };
        let ones = [
            mup.attention_in,
            mup.attention_out,
            mup.key,
            mup.ssm_in,
            mup.ssm_out,
        ]
        .into_iter()
        .chain(mup.ssm)
        .chain(mup.mlp)
        .all(|m| m == 1.0);
        (!ones).then_some(mup)
    }

    pub fn layout(&self) -> HybridLayout {
        let blocks = (0..self.num_hidden_layers)
            .map(|i| {
                let layer = format!("layers.{i}");
                HybridBlock {
                    mixer: GraniteLayerType::Parallel,
                    names: BlockNames {
                        norm: format!("{layer}.input_layernorm"),
                        mixer: format!("{layer}.self_attn"),
                        mamba: Some(format!("{layer}.mamba")),
                        ffn: Some(FfnNames {
                            norm: format!("{layer}.pre_ff_layernorm"),
                            module: format!("{layer}.feed_forward"),
                            kind: FfnKind::SwiGlu,
                            moe: None,
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
        if self.attention_bias || self.mlp_bias || self.projectors_bias {
            inference_tensor::bail!(
                "Falcon-H1 with attention, MLP or projector biases is not supported"
            );
        }
        if self.mamba_rms_norm && self.mamba_norm_before_gate {
            inference_tensor::bail!(
                "Falcon-H1 with its Mamba norm before the gate is not supported"
            );
        }
        let mamba_d_head = self.mamba_d_head()?;
        Ok(crate::granite::Config {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            shared_intermediate_size: None,
            vocab_size: self.vocab_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta(),
            max_position_embeddings: self.max_position_embeddings,
            rope_scaling: None,
            quantization_config: self.quantization_config.clone(),
            tie_word_embeddings: self.tie_word_embeddings,
            layer_types: vec![GraniteLayerType::Parallel; self.num_hidden_layers],
            attention_multiplier: 1.0 / (self.head_dim() as f32).sqrt(),
            embedding_multiplier: self.embedding_multiplier,
            residual_multiplier: 1.0,
            logits_scaling: 1.0 / self.lm_head_multiplier,
            mamba_n_heads: Some(self.mamba_n_heads),
            mamba_n_groups: self.mamba_n_groups,
            mamba_d_state: self.mamba_d_state,
            mamba_d_head: Some(mamba_d_head),
            mamba_d_conv: self.mamba_d_conv,
            mamba_expand: self.mamba_expand,
            mamba_chunk_size: self.mamba_chunk_size,
            mamba_conv_bias: self.mamba_conv_bias,
            mamba_proj_bias: self.mamba_proj_bias,
            num_local_experts: 0,
            num_experts_per_tok: 0,
            position_embedding_type: "rope".to_string(),
            sliding_window: None,
            layer_rope_theta: None,
            no_rope_layers: None,
            rope_parameters: None,
            head_dim: Some(self.head_dim()),
            mamba_inner_size: Some(self.mamba_d_ssm),
            mamba_norm_group_size: Some(self.mamba_d_ssm / self.mamba_n_groups),
            mamba_dt_limit: None,
            mamba_f32_state: true,
            mamba_gated_norm: self.mamba_rms_norm,
            mup: self.mup(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(extra: &str) -> Config {
        Config::from_json(&format!(
            r#"{{"vocab_size": 16, "hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 2,
                "num_attention_heads": 2, "mamba_d_ssm": 8, "mamba_n_heads": 2{extra}}}"#
        ))
        .unwrap()
    }

    #[test]
    fn every_layer_is_a_parallel_block_over_falcons_names() {
        let layout = config("").layout();
        assert_eq!(layout.blocks.len(), 2);
        let names = &layout.blocks[1].names;
        assert_eq!(layout.blocks[1].mixer, GraniteLayerType::Parallel);
        assert_eq!(names.mixer, "layers.1.self_attn");
        assert_eq!(names.mamba.as_deref(), Some("layers.1.mamba"));
        let ffn = names.ffn.as_ref().unwrap();
        assert_eq!(
            (ffn.norm.as_str(), ffn.kind),
            ("layers.1.pre_ff_layernorm", FfnKind::SwiGlu)
        );
    }

    #[test]
    fn folded_multipliers_leave_no_mup_and_the_auto_head_splits_the_inner_width() {
        let granite = config(r#", "mamba_d_head": "auto""#).granite().unwrap();
        assert!(granite.mup.is_none());
        assert_eq!(granite.mamba_d_head, Some(4));
        let granite = config(r#", "lm_head_multiplier": 0.5, "ssm_multipliers": [1, 2, 3, 4, 5]"#)
            .granite()
            .unwrap();
        assert_eq!(granite.logits_scaling, 2.0);
        assert_eq!(granite.mup.unwrap().ssm, [1.0, 2.0, 3.0, 4.0, 5.0]);
    }
}
