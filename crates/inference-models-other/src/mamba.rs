//! Mamba 1 (`MambaForCausalLM`, state-spaces' HF conversions) and Falcon-Mamba (`FalconMambaForCausalLM`): a stack of
//! Mamba 1 mixers, run as Granite's hybrid model with only Mamba blocks whose heads are the mixer's channels.

use inference_quant::QuantizedConfig;
use inference_tensor::Result;
use serde::Serialize;

use crate::granite::{
    BlockNames, DtBcNorm, GraniteLayerType, HybridBlock, HybridLayout, Mamba1Spec,
};
use crate::serde_default_fn;

const EMBEDDINGS: &str = "embeddings";
// `mamba_ssm`'s own checkpoints
const MAMBA_SSM_EMBEDDINGS: &str = "embedding";
const FINAL_NORM: &str = "norm_f";
// normalizes dt, B and C after `x_proj`
pub const FALCON_MAMBA: &str = "falcon_mamba";
// transformers' `time_step_rank` default: the hidden size over 16, rounded up
const AUTO: &str = "auto";
const AUTO_RANK_DIVISOR: usize = 16;
// no attention layers, so no heads or positions; the context the planner may offer a sequence
const NO_ATTENTION_HEADS: usize = 1;
pub const MAX_POSITIONS: usize = 1 << 20;

serde_default_fn!(usize, default_state_size, 16);
serde_default_fn!(f64, default_norm_eps, 1e-5);
serde_default_fn!(f64, default_mixer_eps, 1e-6);
serde_default_fn!(usize, default_expand, 2);
serde_default_fn!(usize, default_conv_kernel, 4);
serde_default_fn!(bool, default_true, true);

#[derive(Debug, Clone, inference_json::FromJson, Serialize)]
pub struct Config {
    #[serde(default)]
    pub model_type: String,
    /// Falcon-Mamba's dt, B and C norms when the model type does not say so (a GGUF's `ssm.dt_b_c_rms`).
    #[serde(default)]
    pub dt_b_c_norm: bool,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: Option<usize>,
    #[serde(default = "default_state_size")]
    pub state_size: usize,
    #[serde(default = "default_norm_eps")]
    pub layer_norm_epsilon: f64,
    #[serde(default = "default_mixer_eps")]
    pub mixer_rms_eps: f64,
    #[serde(default = "default_expand")]
    pub expand: usize,
    #[serde(default = "default_conv_kernel")]
    pub conv_kernel: usize,
    pub time_step_rank: Option<serde_json::Value>,
    #[serde(default)]
    pub use_bias: bool,
    #[serde(default = "default_true")]
    pub use_conv_bias: bool,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
    pub quantization_config: Option<QuantizedConfig>,
    /// Set when the config was translated from `mamba_ssm`'s (see `MAMBA_SSM_CONFIG_KEY`).
    #[serde(default, rename = "_inference_mamba_ssm")]
    pub mamba_ssm: bool,
}

impl Config {
    // Falcon-Mamba's `expand` (16) is not its width; `intermediate_size` is
    pub fn inner_size(&self) -> usize {
        self.intermediate_size
            .unwrap_or(self.expand * self.hidden_size)
    }

    pub fn dt_rank(&self) -> Result<usize> {
        match &self.time_step_rank {
            None => Ok(self.hidden_size.div_ceil(AUTO_RANK_DIVISOR)),
            Some(serde_json::Value::String(auto)) if auto == AUTO => {
                Ok(self.hidden_size.div_ceil(AUTO_RANK_DIVISOR))
            }
            Some(value) => value
                .as_u64()
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(|| {
                    inference_tensor::Error::msg(format!(
                        "Mamba `time_step_rank` {value} is not a size"
                    ))
                }),
        }
    }

    pub fn layout(&self) -> HybridLayout {
        let blocks = (0..self.num_hidden_layers)
            .map(|i| HybridBlock {
                mixer: GraniteLayerType::Mamba,
                names: BlockNames {
                    norm: format!("layers.{i}.norm"),
                    mixer: format!("layers.{i}.mixer"),
                    mamba: None,
                    ffn: None,
                },
            })
            .collect();
        HybridLayout {
            blocks,
            embeddings: if self.mamba_ssm {
                MAMBA_SSM_EMBEDDINGS
            } else {
                EMBEDDINGS
            },
            final_norm: FINAL_NORM,
        }
    }

    pub fn granite(&self) -> Result<crate::granite::Config> {
        if self.use_bias {
            inference_tensor::bail!("Mamba with projection biases is not supported");
        }
        let inner = self.inner_size();
        Ok(crate::granite::Config {
            hidden_size: self.hidden_size,
            intermediate_size: 0,
            shared_intermediate_size: None,
            vocab_size: self.vocab_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: NO_ATTENTION_HEADS,
            num_key_value_heads: None,
            rms_norm_eps: self.layer_norm_epsilon,
            rope_theta: 0.0,
            max_position_embeddings: MAX_POSITIONS,
            rope_scaling: None,
            quantization_config: self.quantization_config.clone(),
            tie_word_embeddings: self.tie_word_embeddings,
            layer_types: vec![GraniteLayerType::Mamba; self.num_hidden_layers],
            attention_multiplier: 1.0,
            embedding_multiplier: 1.0,
            residual_multiplier: 1.0,
            logits_scaling: 1.0,
            // each channel a head of width one, all in one group
            mamba_n_heads: Some(inner),
            mamba_n_groups: 1,
            mamba_d_state: self.state_size,
            mamba_d_head: Some(1),
            mamba_d_conv: self.conv_kernel,
            mamba_expand: self.expand,
            mamba_chunk_size: 0,
            mamba_conv_bias: self.use_conv_bias,
            mamba_proj_bias: self.use_bias,
            num_local_experts: 0,
            num_experts_per_tok: 0,
            position_embedding_type: "nope".to_string(),
            sliding_window: None,
            layer_rope_theta: None,
            no_rope_layers: None,
            rope_parameters: None,
            head_dim: Some(self.hidden_size),
            mamba_inner_size: Some(inner),
            mamba_norm_group_size: None,
            mamba_dt_limit: None,
            mamba_f32_state: true,
            mamba_gated_norm: false,
            mup: None,
            mamba1: Some(Mamba1Spec {
                dt_rank: self.dt_rank()?,
                dt_b_c_norm: (self.dt_b_c_norm || self.model_type == FALCON_MAMBA).then_some(
                    DtBcNorm {
                        eps: self.mixer_rms_eps,
                        weighted: false,
                    },
                ),
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(extra: &str) -> Config {
        Config::from_json(&format!(
            r#"{{"vocab_size": 16, "hidden_size": 40, "num_hidden_layers": 2{extra}}}"#
        ))
        .unwrap()
    }

    #[test]
    fn the_channels_are_the_heads_and_only_falcon_mamba_normalizes_its_selection() {
        let granite = config("").granite().unwrap();
        assert_eq!(
            (
                granite.mamba_n_heads,
                granite.mamba_d_head,
                granite.mamba_conv_dim()
            ),
            (Some(80), Some(1), 80)
        );
        let spec = granite.mamba1.unwrap();
        assert_eq!(spec.dt_rank, 3);
        assert!(spec.dt_b_c_norm.is_none());
        let falcon = config(r#", "model_type": "falcon_mamba", "intermediate_size": 64, "expand": 16, "time_step_rank": 8"#)
            .granite()
            .unwrap();
        assert_eq!(falcon.mamba_inner_size, Some(64));
        let spec = falcon.mamba1.unwrap();
        let norm = spec.dt_b_c_norm.unwrap();
        assert_eq!((spec.dt_rank, norm.eps, norm.weighted), (8, 1e-6, false));
    }
}
