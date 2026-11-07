//! Nemotron-H: Mamba-2, attention, MoE and MLP layers in a per-layer pattern, each layer one mixer or one
//! feed-forward. A mixer layer and the feed-forward layer after it run as one of Granite's hybrid blocks.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use inference_quant::QuantizedConfig;
use inference_tensor::Result;
use serde::Serialize;

use crate::decoder::MtpSpec;
use crate::granite::{
    BlockNames, FfnKind, FfnNames, GraniteLayerType, HybridBlock, HybridLayout, HybridMtpLayout,
    NemotronMoeSpec,
};
use crate::serde_default_fn;

const EMBEDDINGS: &str = "embeddings";
const FINAL_NORM: &str = "norm_f";
const SQUARED_RELU: &str = "relu2";
const F32_STATE: &str = "float32";

serde_default_fn!(usize, default_head_dim, 128);
serde_default_fn!(usize, default_intermediate_size, 21504);
serde_default_fn!(String, default_mlp_act, SQUARED_RELU.to_string());
serde_default_fn!(usize, default_ssm_state_size, 128);
serde_default_fn!(usize, default_mamba_num_heads, 128);
serde_default_fn!(usize, default_mamba_head_dim, 64);
serde_default_fn!(usize, default_n_groups, 8);
serde_default_fn!(usize, default_conv_kernel, 4);
serde_default_fn!(usize, default_expand, 2);
serde_default_fn!(bool, default_true, true);
serde_default_fn!(usize, default_chunk_size, 128);
serde_default_fn!(String, default_state_dtype, F32_STATE.to_string());
serde_default_fn!(usize, default_n_routed_experts, 8);
serde_default_fn!(usize, default_experts_per_tok, 2);
serde_default_fn!(usize, default_moe_intermediate_size, 7688);
serde_default_fn!(f32, default_routed_scaling_factor, 1.0);
serde_default_fn!(usize, default_one, 1);
serde_default_fn!(f64, default_norm_eps, 1e-5);
serde_default_fn!(usize, default_max_position_embeddings, 4096);

// NeMo-exported configs write non-finite floats as `{"__float__": "Infinity"}`; serde_json would write `null`
pub(crate) mod float_pair {
    use serde::{Deserialize, Serialize, de::Error};
    use serde_json::{Value, json};

    const ENCODED: &str = inference_nn::loaders::NON_FINITE_FLOAT_KEY;

    pub fn serialize<S: serde::Serializer>(
        pair: &Option<[f64; 2]>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let encode = |x: f64| match x.is_finite() {
            true => json!(x),
            false if x.is_nan() => json!({ ENCODED: "NaN" }),
            false => json!({ ENCODED: if x > 0.0 { "Infinity" } else { "-Infinity" } }),
        };
        pair.map(|[a, b]| [encode(a), encode(b)])
            .serialize(serializer)
    }

    pub fn deserialize<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<[f64; 2]>, D::Error> {
        let Some(pair) = Option::<[Value; 2]>::deserialize(deserializer)? else {
            return Ok(None);
        };
        let float = |v: &Value| {
            v.as_f64()
                .or_else(|| v.get(ENCODED)?.as_str()?.parse().ok())
                .ok_or_else(|| D::Error::custom(format!("expected a float, got {v}")))
        };
        Ok(Some([float(&pair[0])?, float(&pair[1])?]))
    }
}

/// `Config::moe_spec`'s layer for the MTP head's MoE, past every layer of the stack.
pub const MTP_LAYER: usize = usize::MAX;

/// One layer of a Puzzle checkpoint (`nemotron_h_puzzle`: Nemotron Labs 3 Puzzle, Bolt): its kind and, for MoE, the
/// experts it was pruned to.
#[derive(Debug, Clone, inference_json::FromJson, Serialize)]
pub struct BlockConfig {
    pub block_type: String,
    pub moe_intermediate_size: Option<usize>,
    pub n_routed_experts: Option<usize>,
    pub num_experts_per_tok: Option<usize>,
}

#[derive(Debug, Clone, inference_json::FromJson, Serialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    #[serde(alias = "layer_types")]
    pub layers_block_type: Option<Vec<String>>,
    pub hybrid_override_pattern: Option<String>,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_mlp_act")]
    pub mlp_hidden_act: String,
    #[serde(default = "default_ssm_state_size")]
    pub ssm_state_size: usize,
    #[serde(default = "default_mamba_num_heads")]
    pub mamba_num_heads: usize,
    #[serde(default = "default_mamba_head_dim")]
    pub mamba_head_dim: usize,
    #[serde(default = "default_n_groups", alias = "mamba_n_groups")]
    pub n_groups: usize,
    #[serde(default = "default_conv_kernel", alias = "mamba_d_conv")]
    pub conv_kernel: usize,
    #[serde(default = "default_expand", alias = "mamba_expand")]
    pub expand: usize,
    // the Mamba dt clamp, prompts and decode alike (`time_step_min` only seeds the dt bias at init)
    #[serde(default, with = "float_pair")]
    pub time_step_limit: Option<[f64; 2]>,
    #[serde(default = "default_true", alias = "mamba_conv_bias")]
    pub use_conv_bias: bool,
    #[serde(default)]
    pub mamba_proj_bias: bool,
    #[serde(default = "default_chunk_size", alias = "mamba_chunk_size")]
    pub chunk_size: usize,
    #[serde(default = "default_state_dtype")]
    pub mamba_ssm_cache_dtype: String,
    #[serde(default = "default_n_routed_experts", alias = "num_experts")]
    pub n_routed_experts: usize,
    #[serde(default = "default_experts_per_tok")]
    pub num_experts_per_tok: usize,
    #[serde(default = "default_moe_intermediate_size")]
    pub moe_intermediate_size: usize,
    #[serde(default = "default_moe_intermediate_size")]
    pub moe_shared_expert_intermediate_size: usize,
    pub moe_latent_size: Option<usize>,
    pub block_configs: Option<Vec<BlockConfig>>,
    #[serde(default = "default_routed_scaling_factor")]
    pub routed_scaling_factor: f32,
    #[serde(default = "default_one")]
    pub n_group: usize,
    #[serde(default = "default_one")]
    pub topk_group: usize,
    #[serde(default = "default_true")]
    pub norm_topk_prob: bool,
    #[serde(default = "default_norm_eps", alias = "norm_eps")]
    pub layer_norm_epsilon: f64,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default)]
    pub num_nextn_predict_layers: usize,
    pub mtp_layers_block_type: Option<Vec<String>>,
    /// A Puzzle checkpoint's MTP head, layer by layer as `block_configs` gives the stack's (its MoE's experts there).
    pub mtp_block_configs: Option<Vec<BlockConfig>>,
    /// The loader's flag that `--mtp` asked for the checkpoint's MTP head.
    #[serde(default, rename = "_inference_mtp")]
    pub mtp: bool,
}

/// What one checkpoint layer holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    Mamba,
    Attention,
    Moe,
    Mlp,
}

impl LayerKind {
    fn parse(name: &str) -> Result<Self> {
        Ok(match name {
            "linear_attention" | "mamba" | "M" => Self::Mamba,
            "full_attention" | "attention" | "*" => Self::Attention,
            "moe" | "E" => Self::Moe,
            "mlp" | "-" => Self::Mlp,
            other => inference_tensor::bail!("unknown Nemotron-H layer type `{other}`"),
        })
    }
}

impl Config {
    pub fn layer_kinds(&self) -> Result<Vec<LayerKind>> {
        if let Some(blocks) = &self.block_configs {
            return blocks
                .iter()
                .map(|b| LayerKind::parse(&b.block_type))
                .collect();
        }
        match (&self.layers_block_type, &self.hybrid_override_pattern) {
            (Some(types), _) => types.iter().map(|t| LayerKind::parse(t)).collect(),
            (None, Some(pattern)) => pattern
                .chars()
                .map(|c| LayerKind::parse(&c.to_string()))
                .collect(),
            (None, None) => inference_tensor::bail!(
                "Nemotron-H config has neither `layers_block_type` nor `hybrid_override_pattern`"
            ),
        }
    }

    fn validate(&self) -> Result<()> {
        if self.mlp_hidden_act != SQUARED_RELU {
            inference_tensor::bail!(
                "Nemotron-H with `mlp_hidden_act` {} is not supported",
                self.mlp_hidden_act
            );
        }
        if self.n_group != 1 || self.topk_group != 1 {
            inference_tensor::bail!("Nemotron-H group-limited routing is not supported yet");
        }
        if !self.norm_topk_prob {
            inference_tensor::bail!("Nemotron-H without `norm_topk_prob` is not supported");
        }
        Ok(())
    }

    /// Granite's blocks over the checkpoint's layers: each mixer layer with the feed-forward layer after it.
    pub fn layout(&self) -> Result<HybridLayout> {
        self.validate()?;
        let mut blocks: Vec<HybridBlock> = Vec::new();
        for (i, kind) in self.layer_kinds()?.into_iter().enumerate() {
            let layer = format!("layers.{i}");
            let mixer = match kind {
                LayerKind::Mamba => Some(GraniteLayerType::Mamba),
                LayerKind::Attention => Some(GraniteLayerType::Attention),
                LayerKind::Moe | LayerKind::Mlp => None,
            };
            if let Some(mixer) = mixer {
                blocks.push(HybridBlock {
                    mixer,
                    names: BlockNames {
                        norm: format!("{layer}.norm"),
                        mixer: format!("{layer}.mixer"),
                        mamba: None,
                        ffn: None,
                    },
                });
                continue;
            }
            let block = blocks
                .last_mut()
                .filter(|block| block.names.ffn.is_none())
                .ok_or_else(|| {
                    inference_tensor::Error::msg(format!(
                        "Nemotron-H layer {i} is a feed-forward with no mixer layer before it"
                    ))
                })?;
            block.names.ffn = Some(FfnNames {
                norm: format!("{layer}.norm"),
                module: format!("{layer}.mixer"),
                kind: match kind {
                    LayerKind::Moe => FfnKind::Moe,
                    _ => FfnKind::Mlp,
                },
                moe: (kind == LayerKind::Moe).then(|| self.moe_spec(i)),
            });
        }
        Ok(HybridLayout {
            blocks,
            embeddings: EMBEDDINGS,
            final_norm: FINAL_NORM,
        })
    }

    /// Layer `layer`'s experts: the model's, as a Puzzle checkpoint's `block_configs` prunes them; at `MTP_LAYER`
    /// the head's, as its `mtp_block_configs` gives them.
    pub fn moe_spec(&self, layer: usize) -> NemotronMoeSpec {
        let block = if layer == MTP_LAYER {
            self.mtp_block_configs.as_ref().and_then(|blocks| {
                blocks
                    .iter()
                    .find(|b| LayerKind::parse(&b.block_type).ok() == Some(LayerKind::Moe))
            })
        } else {
            self.block_configs
                .as_ref()
                .and_then(|blocks| blocks.get(layer))
        };
        NemotronMoeSpec {
            num_experts: block
                .and_then(|b| b.n_routed_experts)
                .unwrap_or(self.n_routed_experts),
            top_k: block
                .and_then(|b| b.num_experts_per_tok)
                .unwrap_or(self.num_experts_per_tok),
            intermediate_size: block
                .and_then(|b| b.moe_intermediate_size)
                .unwrap_or(self.moe_intermediate_size),
            shared_intermediate_size: self.moe_shared_expert_intermediate_size,
            routed_scaling_factor: self.routed_scaling_factor,
            latent_size: self.moe_latent_size,
        }
    }

    /// The checkpoint's MTP head when `--mtp` asked for it: an attention layer and the MoE after it, under `mtp.`, its
    /// input fusion on the first and its final LayerNorm on the second.
    pub fn mtp_layout(&self) -> Result<Option<HybridMtpLayout>> {
        if !self.mtp || self.num_nextn_predict_layers == 0 {
            return Ok(None);
        }
        if self.num_nextn_predict_layers != 1 {
            inference_tensor::bail!(
                "Nemotron-H with {} MTP layers is not supported",
                self.num_nextn_predict_layers
            );
        }
        let types = self.mtp_layers_block_type.clone().or_else(|| {
            self.mtp_block_configs
                .as_ref()
                .map(|blocks| blocks.iter().map(|b| b.block_type.clone()).collect())
        });
        let kinds = match types {
            Some(types) => types
                .iter()
                .map(|t| LayerKind::parse(t))
                .collect::<Result<Vec<_>>>()?,
            None => vec![LayerKind::Attention, LayerKind::Moe],
        };
        if kinds != [LayerKind::Attention, LayerKind::Moe] {
            inference_tensor::bail!(
                "Nemotron-H MTP layers {kinds:?} are not supported, only an attention layer and an MoE"
            );
        }
        Ok(Some(HybridMtpLayout {
            block: HybridBlock {
                mixer: GraniteLayerType::Attention,
                names: BlockNames {
                    norm: "layers.0.norm".into(),
                    mixer: "layers.0.mixer".into(),
                    mamba: None,
                    ffn: Some(FfnNames {
                        norm: "layers.1.norm".into(),
                        module: "layers.1.mixer".into(),
                        kind: FfnKind::Moe,
                        moe: Some(self.moe_spec(MTP_LAYER)),
                    }),
                },
            },
            names: MtpSpec {
                embedding_norm: "layers.0.enorm",
                hidden_norm: "layers.0.hnorm",
                projection: "layers.0.eh_proj",
                block: "layers.0",
                final_norm: Some("layers.1.final_layernorm"),
                dense_block: false,
                mask_first_position: false,
            },
            layer_norm_final: true,
        }))
    }

    /// The Granite config the blocks load with: NoPE attention scaled by `1/sqrt(head_dim)`, the Mamba widths from
    /// its heads, a gated norm per group, no multipliers.
    pub fn granite(&self) -> Result<crate::granite::Config> {
        let layout = self.layout()?;
        Ok(crate::granite::Config {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            shared_intermediate_size: None,
            vocab_size: self.vocab_size,
            num_hidden_layers: layout.blocks.len(),
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            rms_norm_eps: self.layer_norm_epsilon,
            rope_theta: 0.0,
            max_position_embeddings: self.max_position_embeddings,
            rope_scaling: None,
            quantization_config: self.quantization_config.clone(),
            tie_word_embeddings: self.tie_word_embeddings,
            layer_types: layout.blocks.iter().map(|b| b.mixer.clone()).collect(),
            attention_multiplier: 1.0 / (self.head_dim as f32).sqrt(),
            embedding_multiplier: 1.0,
            residual_multiplier: 1.0,
            logits_scaling: 1.0,
            mamba_n_heads: Some(self.mamba_num_heads),
            mamba_n_groups: self.n_groups,
            mamba_d_state: self.ssm_state_size,
            mamba_d_head: Some(self.mamba_head_dim),
            mamba_d_conv: self.conv_kernel,
            mamba_expand: self.expand,
            mamba_chunk_size: self.chunk_size,
            mamba_conv_bias: self.use_conv_bias,
            mamba_proj_bias: self.mamba_proj_bias,
            num_local_experts: 0,
            num_experts_per_tok: self.num_experts_per_tok,
            position_embedding_type: "nope".to_string(),
            sliding_window: None,
            layer_rope_theta: None,
            no_rope_layers: None,
            rope_parameters: None,
            head_dim: Some(self.head_dim),
            mamba_inner_size: Some(self.mamba_num_heads * self.mamba_head_dim),
            mamba_norm_group_size: Some(self.mamba_num_heads * self.mamba_head_dim / self.n_groups),
            mamba_dt_limit: self.time_step_limit,
            mamba_f32_state: self.mamba_ssm_cache_dtype == F32_STATE,
            mamba_gated_norm: true,
            mup: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PATTERN: &str = "MEM*EM-";

    fn config(pattern: &str) -> Config {
        Config::from_json(&format!(
            r#"{{"vocab_size": 16, "hidden_size": 8, "num_attention_heads": 2, "hybrid_override_pattern": "{pattern}"}}"#
        ))
        .unwrap()
    }

    #[test]
    fn a_mixer_layer_and_the_feed_forward_after_it_make_one_block() {
        let layout = config(PATTERN).layout().unwrap();
        let blocks = layout
            .blocks
            .iter()
            .map(|b| {
                (
                    b.mixer.clone(),
                    b.names.mixer.clone(),
                    b.names.ffn.as_ref().map(|f| (f.module.clone(), f.kind)),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            blocks,
            [
                (
                    GraniteLayerType::Mamba,
                    "layers.0.mixer".to_string(),
                    Some(("layers.1.mixer".to_string(), FfnKind::Moe))
                ),
                (GraniteLayerType::Mamba, "layers.2.mixer".to_string(), None),
                (
                    GraniteLayerType::Attention,
                    "layers.3.mixer".to_string(),
                    Some(("layers.4.mixer".to_string(), FfnKind::Moe))
                ),
                (
                    GraniteLayerType::Mamba,
                    "layers.5.mixer".to_string(),
                    Some(("layers.6.mixer".to_string(), FfnKind::Mlp))
                ),
            ]
        );
        assert_eq!(layout.block_of_layer()[&4], 2);
    }

    #[test]
    fn a_feed_forward_without_a_mixer_before_it_is_refused() {
        assert!(config("EM").layout().is_err());
        assert!(config("MEE").layout().is_err());
    }

    #[test]
    fn the_list_and_the_pattern_name_the_same_layers() {
        let listed = Config::from_json(
            r#"{"vocab_size": 16, "hidden_size": 8, "num_attention_heads": 2,
                "layers_block_type": ["mamba", "moe", "linear_attention", "full_attention", "moe", "mamba", "mlp"]}"#,
        )
        .unwrap();
        let pattern = config("MEM*EM-");
        assert_eq!(
            listed.layer_kinds().unwrap(),
            pattern.layer_kinds().unwrap()
        );
    }

    // dt is clamped by `time_step_limit` alone, as NVIDIA's modeling code and llama.cpp do: transformers' port clamps
    // prompts at `time_step_min`, which tripled the text KLD against llama.cpp's BF16 logits
    #[test]
    fn the_dt_clamp_is_the_time_step_limit_not_the_init_minimum() {
        let unset = config(PATTERN).granite().unwrap();
        assert_eq!(unset.mamba_dt_limit, None);
        let set = Config::from_json(
            r#"{"vocab_size": 16, "hidden_size": 8, "num_attention_heads": 2, "hybrid_override_pattern": "ME",
                "time_step_min": 0.001, "time_step_limit": [0.0, 1e30]}"#,
        )
        .unwrap();
        assert_eq!(set.granite().unwrap().mamba_dt_limit, Some([0.0, 1e30]));
        let nemo = Config::from_json(
            r#"{"vocab_size": 16, "hidden_size": 8, "num_attention_heads": 2, "hybrid_override_pattern": "ME",
                "time_step_limit": [0.0, {"__float__": "Infinity"}]}"#,
        )
        .unwrap();
        assert_eq!(nemo.time_step_limit, Some([0.0, f64::INFINITY]));
        let round_trip = Config::from_json(&serde_json::to_string(&nemo).unwrap()).unwrap();
        assert_eq!(round_trip.time_step_limit, Some([0.0, f64::INFINITY]));
    }
}
