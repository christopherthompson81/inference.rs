//! Mamba-2 (`Mamba2ForCausalLM`: state-spaces' HF conversions, Mistral's Mamba-Codestral): a stack of Mamba-2 mixers
//! with no attention or feed-forward, run as Granite's hybrid model with only Mamba blocks.

use inference_quant::QuantizedConfig;
use inference_tensor::Result;
use serde::Serialize;

use crate::granite::{BlockNames, GraniteLayerType, HybridBlock, HybridLayout};
use crate::nemotron_h::float_pair;
use crate::serde_default_fn;

const EMBEDDINGS: &str = "embeddings";
const FINAL_NORM: &str = "norm_f";
// no attention layers, so no heads or positions; the context the planner may offer a sequence
const NO_ATTENTION_HEADS: usize = 1;
pub const MAX_POSITIONS: usize = 1 << 20;

serde_default_fn!(usize, default_num_heads, 128);
serde_default_fn!(usize, default_head_dim, 64);
serde_default_fn!(usize, default_state_size, 128);
serde_default_fn!(f64, default_norm_eps, 1e-5);
serde_default_fn!(usize, default_expand, 2);
serde_default_fn!(usize, default_conv_kernel, 4);
serde_default_fn!(usize, default_n_groups, 8);
serde_default_fn!(bool, default_true, true);
serde_default_fn!(usize, default_chunk_size, 256);

#[derive(Debug, Clone, inference_json::FromJson, Serialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_heads")]
    pub num_heads: usize,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_state_size")]
    pub state_size: usize,
    #[serde(default = "default_norm_eps")]
    pub layer_norm_epsilon: f64,
    #[serde(default = "default_expand")]
    pub expand: usize,
    #[serde(default = "default_conv_kernel")]
    pub conv_kernel: usize,
    #[serde(default = "default_n_groups")]
    pub n_groups: usize,
    #[serde(default)]
    pub use_bias: bool,
    #[serde(default = "default_true")]
    pub use_conv_bias: bool,
    #[serde(default = "default_chunk_size")]
    pub chunk_size: usize,
    #[serde(default, with = "float_pair")]
    pub time_step_limit: Option<[f64; 2]>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub quantization_config: Option<QuantizedConfig>,
}

impl Config {
    pub fn inner_size(&self) -> usize {
        self.num_heads * self.head_dim
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
            embeddings: EMBEDDINGS,
            final_norm: FINAL_NORM,
        }
    }

    /// The Granite config the blocks load with. The gated norm runs after the gate and per group whatever
    /// `norm_before_gate` says, as mamba_ssm's fused kernel and llama.cpp run it (transformers' unfused fallback
    /// normalizes the whole width, which differs once `n_groups` is above one).
    pub fn granite(&self) -> Result<crate::granite::Config> {
        if self.use_bias {
            inference_tensor::bail!("Mamba-2 with projection biases is not supported");
        }
        let inner = self.inner_size();
        if inner != self.expand * self.hidden_size {
            inference_tensor::bail!(
                "Mamba-2 heads ({} x {}) do not make expand x hidden_size ({} x {})",
                self.num_heads,
                self.head_dim,
                self.expand,
                self.hidden_size
            );
        }
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
            mamba_n_heads: Some(self.num_heads),
            mamba_n_groups: self.n_groups,
            mamba_d_state: self.state_size,
            mamba_d_head: Some(self.head_dim),
            mamba_d_conv: self.conv_kernel,
            mamba_expand: self.expand,
            mamba_chunk_size: self.chunk_size,
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
            mamba_norm_group_size: Some(inner / self.n_groups),
            mamba_dt_limit: self.time_step_limit,
            mamba_f32_state: true,
            mamba_gated_norm: true,
            mup: None,
            mamba1: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(extra: &str) -> Config {
        Config::from_json(&format!(
            r#"{{"vocab_size": 16, "hidden_size": 8, "num_hidden_layers": 2, "num_heads": 4, "head_dim": 4,
                "n_groups": 2{extra}}}"#
        ))
        .unwrap()
    }

    #[test]
    fn every_layer_is_a_mamba_block_with_a_norm_per_group() {
        let cfg = config("");
        let layout = cfg.layout();
        assert_eq!(layout.blocks.len(), 2);
        assert_eq!(layout.blocks[1].names.mixer, "layers.1.mixer");
        assert!(
            layout
                .blocks
                .iter()
                .all(|b| b.mixer == GraniteLayerType::Mamba && b.names.ffn.is_none())
        );
        let granite = cfg.granite().unwrap();
        assert_eq!(granite.mamba_norm_group_size, Some(8));
        assert_eq!(granite.mamba_dt_limit, None);
    }

    #[test]
    fn python_infinity_in_the_dt_limit_reads_as_no_upper_clamp() {
        let text = inference_nn::loaders::encode_non_finite_floats(
            r#", "time_step_limit": [0.0, Infinity]"#,
        );
        assert_eq!(config(&text).time_step_limit, Some([0.0, f64::INFINITY]));
        assert!(config(r#", "expand": 3"#).granite().is_err());
    }
}
