#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::layers::masker::CausalMaskConfig;
use inference_quant::{
    ColumnParallelLayer, QuantMethod, QuantizedConfig, ReplicatedLayer, RowParallelLayer,
    ShardedVarBuilder, apply_immediate_isq, should_apply_immediate_isq,
};
use inference_tensor::nn::Module;
use inference_tensor::{DType, Device, IndexOp, Result, Tensor};
use serde::Serialize;
use std::{
    collections::HashMap,
    ops::Range,
    sync::{Arc, Mutex},
};

use crate::decoder::{MtpBlock, MtpHead, MtpHeadParts, MtpSpec, Norm};
use crate::gdn::RecurrentBatchKind;
use crate::kv_cache::EitherCache;
use crate::kv_cache::KvCache;
use crate::kv_cache::NormalCacheType;
use crate::kv_cache::RecurrentCapturePoint;
use crate::model::ForwardMaskCache;
use crate::model::IsqModel;
use crate::model::ModelForwardContext;
use crate::model::NormalLoadingMetadata;
use crate::model::NormalModel;
use crate::{
    amoe::{
        AnyMoeBaseModelMixin, AnyMoeConfig, AnyMoeExpertType, AnyMoeLoraTarget, MlpLayer, MoeMlp,
    },
    attention::{AttentionDispatch, AttentionMask, SdpaParams},
    device_map::{DeviceMappedMask, DeviceMapper},
    kv_cache::{
        HybridCache, HybridCacheConfig, HybridLayerCache, HybridLayerType, RecurrentLayerConfig,
    },
    layers::masker::PastKvLenCache,
    layers::{CausalMasker, PlainMlp, RmsNorm, RotaryEmbedding, embedding_with_legacy_tied_uqff},
    moe::{ExpertProjNames, MoEExperts, MoEExpertsConfig},
    paged_attention::{
        AttentionImplementation, HybridPagedKvCacheConfig, ModelConfigLike, ModelConfigMetadata,
        PagedAttention,
    },
    serde_default_fn,
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};
use inference_drafters::dflash::DFlashLoadTarget;
use inference_drafters::host::{DrafterAttach, DrafterTarget, ExternalDrafter};

// the Mamba dt clamp without a configured one (transformers' `time_step_limit` default)
const NO_DT_LIMIT: [f64; 2] = [0.0, f64::MAX];

// stacked expert tensors: Granite 4's `input_linear.weight`, or transformers v5's bare parameters
const EXPERT_WEIGHT: &str = "weight";
const V5_GATE_UP: &str = "gate_up_proj";
const V5_DOWN: &str = "down_proj";
// Nemotron-H's MoE: `mixer.gate` with its `e_score_correction_bias`, `mixer.shared_experts`
const NEMOTRON_ROUTER: &str = "gate";
const JAMBA_ROUTER: &str = "router";
const NEMOTRON_SELECTION_BIAS: &str = "e_score_correction_bias";
const NEMOTRON_SHARED_EXPERTS: &str = "shared_experts";
// latent MoE (Super, Bolt): the routed experts run between these two, the router and shared expert at full width
const NEMOTRON_LATENT_DOWN: &str = "fc1_latent_proj";
const NEMOTRON_LATENT_UP: &str = "fc2_latent_proj";
const NEMO_FP32_PARAMS: &str = "_fp32_params";
// transformers divides the selected scores by their sum plus this
const NEMOTRON_NORM_MIN: f32 = 1e-20;
const NEMOTRON_MLP: [AnyMoeLoraTarget; 2] = [
    AnyMoeLoraTarget::up("up_proj"),
    AnyMoeLoraTarget::down("down_proj"),
];

serde_default_fn!(bool, word_emb_default, true);
serde_default_fn!(bool, default_true, true);
serde_default_fn!(f32, default_one, 1.0);
serde_default_fn!(f32, default_rope_theta, 10_000.0);
serde_default_fn!(usize, default_mamba_d_conv, 4);
serde_default_fn!(usize, default_mamba_d_state, 256);
serde_default_fn!(usize, default_mamba_expand, 2);
serde_default_fn!(usize, default_mamba_n_groups, 1);
serde_default_fn!(usize, default_mamba_chunk_size, 256);
serde_default_fn!(bool, default_mamba_conv_bias, true);
serde_default_fn!(bool, default_mamba_proj_bias, false);
serde_default_fn!(usize, default_num_local_experts, 0);
serde_default_fn!(usize, default_num_experts_per_tok, 2);
serde_default_fn!(String, default_position_embedding_type, "rope".to_string());

// where a checkpoint keeps its MTP head, beside the decoder's own prefix
pub const MTP_PREFIX: &str = "mtp";

#[derive(Debug, Clone, inference_json::FromJson, Serialize, Default, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum GraniteLayerType {
    #[default]
    #[serde(alias = "full_attention")]
    Attention,
    #[serde(rename = "sliding_attention")]
    SlidingAttention,
    Mamba,
    /// Attention and a Mamba mixer side by side on one norm, their outputs summed (Falcon-H1).
    Parallel,
}

impl GraniteLayerType {
    pub fn is_attention(&self) -> bool {
        !matches!(self, Self::Mamba)
    }

    pub fn has_recurrent_state(&self) -> bool {
        matches!(self, Self::Mamba | Self::Parallel)
    }
}

#[derive(Debug, Clone, inference_json::FromJson, Serialize)]
pub struct GraniteRopeParameters {
    pub rope_theta: Option<f32>,
}

#[derive(Debug, Clone, inference_json::FromJson, Serialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum GraniteRopeType {
    #[default]
    Default,
    Granite,
}

#[derive(Debug, Clone, inference_json::FromJson, Serialize, Default)]
pub struct GraniteRopeConfig {
    pub factor: Option<f32>,
    pub low_freq_factor: Option<f32>,
    pub high_freq_factor: Option<f32>,
    pub original_max_position_embeddings: Option<usize>,
    #[serde(default)]
    pub rope_type: GraniteRopeType,
}

#[derive(Debug, Clone, inference_json::FromJson, Serialize)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub shared_intermediate_size: Option<usize>,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub rope_scaling: Option<GraniteRopeConfig>,
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default = "word_emb_default")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub layer_types: Vec<GraniteLayerType>,
    #[serde(default = "default_one")]
    pub attention_multiplier: f32,
    #[serde(default = "default_one")]
    pub embedding_multiplier: f32,
    #[serde(default = "default_one")]
    pub residual_multiplier: f32,
    #[serde(default = "default_one")]
    pub logits_scaling: f32,
    // Mamba configuration
    pub mamba_n_heads: Option<usize>,
    #[serde(default = "default_mamba_n_groups")]
    pub mamba_n_groups: usize,
    #[serde(default = "default_mamba_d_state")]
    pub mamba_d_state: usize,
    pub mamba_d_head: Option<usize>,
    #[serde(default = "default_mamba_d_conv")]
    pub mamba_d_conv: usize,
    #[serde(default = "default_mamba_expand")]
    pub mamba_expand: usize,
    #[serde(default = "default_mamba_chunk_size")]
    pub mamba_chunk_size: usize,
    #[serde(default = "default_mamba_conv_bias")]
    pub mamba_conv_bias: bool,
    #[serde(default = "default_mamba_proj_bias")]
    pub mamba_proj_bias: bool,
    // MoE configuration
    #[serde(default = "default_num_local_experts")]
    pub num_local_experts: usize,
    #[serde(default = "default_num_experts_per_tok")]
    pub num_experts_per_tok: usize,
    // Position embedding type: "rope" or "nope" (no position embedding)
    #[serde(default = "default_position_embedding_type")]
    pub position_embedding_type: String,
    pub sliding_window: Option<usize>,
    // transformers' per-layer base theta, 0 for a layer without RoPE
    pub layer_rope_theta: Option<Vec<f32>>,
    // the same as llama.cpp reads it: 1 for a RoPE layer, 0 for NoPE
    pub no_rope_layers: Option<Vec<u8>>,
    pub rope_parameters: Option<GraniteRopeParameters>,
    // Nemotron-H sets these: its head and Mamba widths, a gated norm per group, its dt clamp, an f32 state
    pub head_dim: Option<usize>,
    pub mamba_inner_size: Option<usize>,
    pub mamba_norm_group_size: Option<usize>,
    #[serde(default)]
    pub mamba_dt_limit: Option<[f64; 2]>,
    #[serde(default)]
    pub mamba_f32_state: bool,
    // Falcon-H1: a Mamba output gated without the grouped norm, and the muP multipliers (a GGUF has them folded in)
    #[serde(default = "default_true")]
    pub mamba_gated_norm: bool,
    #[serde(default)]
    pub mup: Option<MupMultipliers>,
    /// Mamba 1's selection projections in place of Mamba-2's (Mamba, Falcon-Mamba): the heads are its channels.
    #[serde(default)]
    pub mamba1: Option<Mamba1Spec>,
}

/// Mamba 1's mixer: `x_proj` to dt (`dt_rank` wide), B and C after the conv, `dt_proj` back to the channels, and A
/// per (channel, state item), with Falcon-Mamba's or Jamba's RMSNorms over dt, B and C when `dt_b_c_norm` is set.
#[derive(Debug, Clone, inference_json::FromJson, Serialize)]
pub struct Mamba1Spec {
    pub dt_rank: usize,
    pub dt_b_c_norm: Option<DtBcNorm>,
}

/// Mamba 1's RMSNorms over dt, B and C: Falcon-Mamba's without weights, Jamba's `dt_layernorm`, `b_layernorm` and
/// `c_layernorm` with them.
#[derive(Debug, Clone, inference_json::FromJson, Serialize)]
pub struct DtBcNorm {
    pub eps: f64,
    pub weighted: bool,
}

const DT_B_C_NORMS: [&str; 3] = ["dt_layernorm", "b_layernorm", "c_layernorm"];

/// Falcon-H1's muP multipliers on the activations of a parallel block's attention, Mamba and MLP.
#[derive(Debug, Clone, inference_json::FromJson, Serialize)]
pub struct MupMultipliers {
    pub attention_in: f32,
    pub attention_out: f32,
    pub key: f32,
    pub ssm_in: f32,
    pub ssm_out: f32,
    /// Over the Mamba input projection's z, x, B, C and dt.
    pub ssm: [f32; 5],
    /// The MLP's gate and down projections.
    pub mlp: [f32; 2],
}

impl Config {
    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub fn shared_intermediate_size(&self) -> usize {
        self.shared_intermediate_size
            .unwrap_or(self.intermediate_size)
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn rope_theta(&self) -> f32 {
        self.rope_parameters
            .as_ref()
            .and_then(|p| p.rope_theta)
            .unwrap_or(self.rope_theta)
    }

    /// The RoPE base theta of layer `layer_idx`, or None for a layer without position embeddings.
    pub fn layer_rope_theta(&self, layer_idx: usize) -> Option<f32> {
        if let Some(thetas) = &self.layer_rope_theta {
            return thetas.get(layer_idx).copied().filter(|theta| *theta != 0.0);
        }
        if self.position_embedding_type == "nope"
            || self
                .no_rope_layers
                .as_ref()
                .is_some_and(|layers| layers.get(layer_idx) == Some(&0))
        {
            return None;
        }
        Some(self.rope_theta())
    }

    pub fn layer_types(&self) -> Vec<GraniteLayerType> {
        if self.layer_types.is_empty() {
            vec![GraniteLayerType::Attention; self.num_hidden_layers]
        } else {
            self.layer_types.clone()
        }
    }

    // Mamba helper methods
    pub fn mamba_intermediate_size(&self) -> usize {
        self.mamba_inner_size
            .unwrap_or(self.mamba_expand * self.hidden_size)
    }

    pub fn mamba_n_heads(&self) -> usize {
        self.mamba_n_heads.unwrap_or(128)
    }

    pub fn mamba_d_head(&self) -> usize {
        self.mamba_d_head
            .unwrap_or(self.mamba_intermediate_size() / self.mamba_n_heads())
    }

    /// The SSM state's dtype when it is not the model's.
    pub fn mamba_state_dtype(&self) -> Option<DType> {
        self.mamba_f32_state.then_some(DType::F32)
    }

    /// The conv's channels: Mamba-2's x, B and C, Mamba 1's x alone.
    pub fn mamba_conv_dim(&self) -> usize {
        if self.mamba1.is_some() {
            return self.mamba_intermediate_size();
        }
        self.mamba_intermediate_size() + 2 * self.mamba_n_groups * self.mamba_d_state
    }
}

/// GraniteMLP uses a fused gate-up projection followed by output projection
/// Input: shared_mlp.input_linear (hidden -> shared_intermediate * 2)
/// Output: shared_mlp.output_linear (shared_intermediate -> hidden)
#[derive(Clone)]
pub struct GraniteMlp {
    input_linear: Arc<dyn QuantMethod>,
    output_linear: Arc<dyn QuantMethod>,
    params: Vec<usize>,
}

impl GraniteMlp {
    pub fn new(
        vb: ShardedVarBuilder,
        cfg: &Config,
        comm: &Arc<inference_quant::Comm>,
    ) -> Result<Self> {
        let shared_intermediate_size = cfg.shared_intermediate_size();
        let input_linear = ColumnParallelLayer::new(
            cfg.hidden_size,
            shared_intermediate_size * 2,
            &cfg.quantization_config,
            false,
            comm,
            vb.pp("shared_mlp").pp("input_linear"),
        )?;
        let output_linear = RowParallelLayer::new(
            shared_intermediate_size,
            cfg.hidden_size,
            &cfg.quantization_config,
            false,
            comm,
            vb.pp("shared_mlp").pp("output_linear"),
        )?;
        Ok(Self {
            input_linear,
            output_linear,
            params: vec![cfg.hidden_size, shared_intermediate_size],
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let projected = self.input_linear.forward(x)?;
        let chunks = projected.chunk(2, inference_tensor::D::Minus1)?;
        let gated =
            crate::ops::mul_and_act(&chunks[0], &chunks[1], crate::layers::Activation::Silu)?;
        let res = self.output_linear.forward(&gated)?;
        Ok(res)
    }
}

impl MlpLayer for GraniteMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward(xs)
    }
    fn clone(&self) -> Box<dyn MlpLayer> {
        Box::new(Self {
            input_linear: self.input_linear.clone(),
            output_linear: self.output_linear.clone(),
            params: self.params.clone(),
        })
    }
    fn get_params(&self) -> &[usize] {
        &self.params
    }
    fn hidden_act(&self) -> crate::layers::Activation {
        crate::layers::Activation::Silu
    }
    fn new_added_delta(&self, _deltas: Vec<Option<Tensor>>) -> Result<Box<dyn MlpLayer>> {
        inference_tensor::bail!("LoRA adapter not supported for GraniteMlp")
    }
    fn dtype_device(&self) -> (inference_tensor::DType, inference_tensor::Device) {
        self.input_linear.dtype_and_device()
    }
}

impl crate::amoe::AnyMoeTrainableLayer for GraniteMlp {}

/// Falcon-H1's SwiGLU with its muP multipliers: the gate scaled inside the activation, the down projection after.
#[derive(Clone)]
struct MupMlp {
    gate: Arc<dyn QuantMethod>,
    up: Arc<dyn QuantMethod>,
    down: Arc<dyn QuantMethod>,
    gate_scale: f32,
    down_scale: f32,
    params: Vec<usize>,
}

impl MupMlp {
    fn new(
        vb: ShardedVarBuilder,
        cfg: &Config,
        [gate_scale, down_scale]: [f32; 2],
        comm: &Arc<inference_quant::Comm>,
    ) -> Result<Self> {
        let column = |name| {
            ColumnParallelLayer::new(
                cfg.hidden_size,
                cfg.intermediate_size,
                &cfg.quantization_config,
                false,
                comm,
                vb.pp(name),
            )
        };
        Ok(Self {
            gate: column("gate_proj")?,
            up: column("up_proj")?,
            down: RowParallelLayer::new(
                cfg.intermediate_size,
                cfg.hidden_size,
                &cfg.quantization_config,
                false,
                comm,
                vb.pp("down_proj"),
            )?,
            gate_scale,
            down_scale,
            params: vec![cfg.hidden_size, cfg.intermediate_size],
        })
    }
}

impl MlpLayer for MupMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = scale_tensor(self.gate.forward(xs)?, self.gate_scale)?;
        let gated = crate::ops::mul_and_act(
            &gate,
            &self.up.forward(xs)?,
            crate::layers::Activation::Silu,
        )?;
        scale_tensor(self.down.forward(&gated)?, self.down_scale)
    }
    fn clone(&self) -> Box<dyn MlpLayer> {
        Box::new(Clone::clone(self))
    }
    fn get_params(&self) -> &[usize] {
        &self.params
    }
    fn hidden_act(&self) -> crate::layers::Activation {
        crate::layers::Activation::Silu
    }
    fn new_added_delta(&self, _deltas: Vec<Option<Tensor>>) -> Result<Box<dyn MlpLayer>> {
        inference_tensor::bail!("LoRA adapter not supported for Falcon-H1's MLP")
    }
    fn dtype_device(&self) -> (inference_tensor::DType, inference_tensor::Device) {
        self.gate.dtype_and_device()
    }
}

impl crate::amoe::AnyMoeTrainableLayer for MupMlp {}

// MoE (Mixture of Experts) Implementation

/// Top-K gating router for sparse MoE
struct GraniteTopKGating {
    layer: Arc<dyn QuantMethod>,
    num_experts: usize,
    top_k: usize,
}

impl GraniteTopKGating {
    fn new(
        input_size: usize,
        num_experts: usize,
        top_k: usize,
        quantization_config: &Option<QuantizedConfig>,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let layer = ReplicatedLayer::new(input_size, num_experts, quantization_config, false, vb)?;
        Ok(Self {
            layer,
            num_experts,
            top_k,
        })
    }

    fn topk(&self, x: &Tensor) -> Result<crate::ops::TopKOutput> {
        let logits = self.layer.forward(x)?;
        crate::ops::moe_router_topk(
            &logits,
            crate::ops::MoeRouterTopKConfig {
                top_k: self.top_k,
                score_function: crate::ops::MoeRouterScoreFunction::Softmax,
                selected_weight: crate::ops::MoeRouterSelectedWeight::Score,
                renormalize: true,
                norm_min: 0.0,
                output_scale: 1.0,
                logit_clip: None,
            },
            None,
            None,
        )
    }

    fn grouped_routes(
        &self,
        topk: &crate::ops::TopKOutput,
        dtype: DType,
        device: &Device,
    ) -> Result<(Tensor, Tensor, Vec<usize>)> {
        let selected_experts = topk.indices.to_vec2::<u32>()?;
        let routing_weights = topk
            .values
            .to_dtype(inference_tensor::DType::F32)?
            .to_vec2::<f32>()?;

        // Collect (expert_idx, token_idx, gate) tuples
        let mut expert_token_gates: Vec<(usize, usize, f32)> = Vec::new();
        let mut expert_counts = vec![0usize; self.num_experts];

        for (token_idx, (experts, weights)) in selected_experts
            .iter()
            .zip(routing_weights.iter())
            .enumerate()
        {
            for (&expert_idx, &gate) in experts.iter().zip(weights.iter()) {
                let expert_idx = expert_idx as usize;
                expert_token_gates.push((expert_idx, token_idx, gate));
                expert_counts[expert_idx] += 1;
            }
        }

        // Sort by expert index so tokens are grouped by expert
        expert_token_gates.sort_by_key(|(expert_idx, _, _)| *expert_idx);

        // Extract sorted batch_index and batch_gates
        let all_batch_indices: Vec<u32> = expert_token_gates
            .iter()
            .map(|(_, token_idx, _)| *token_idx as u32)
            .collect();
        let all_batch_gates: Vec<f32> = expert_token_gates
            .iter()
            .map(|(_, _, gate)| *gate)
            .collect();

        let indices_len = all_batch_indices.len();
        let gates_len = all_batch_gates.len();
        let batch_index = Tensor::from_vec(all_batch_indices, (indices_len,), device)?;
        let batch_gates =
            Tensor::from_vec(all_batch_gates, (gates_len,), device)?.to_dtype(dtype)?;

        Ok((batch_index, batch_gates, expert_counts))
    }
}

/// Parallel experts layer - processes all experts in a batched manner
enum GraniteParallelExpertWeights {
    Dense(Vec<Tensor>),
    Quantized(Arc<dyn QuantMethod>),
}

pub struct GraniteParallelExperts {
    weights: GraniteParallelExpertWeights,
    output_size: usize,
    name: &'static str,
}

impl GraniteParallelExperts {
    pub fn new(
        num_experts: usize,
        input_size: usize,
        output_size: usize,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        Self::new_named(num_experts, input_size, output_size, vb, EXPERT_WEIGHT)
    }

    // transformers v5 stores experts as bare `experts.gate_up_proj` / `experts.down_proj` parameters
    fn new_named(
        num_experts: usize,
        input_size: usize,
        output_size: usize,
        stacked_vb: ShardedVarBuilder,
        name: &'static str,
    ) -> Result<Self> {
        let vb = match name {
            EXPERT_WEIGHT => stacked_vb.clone(),
            name => stacked_vb.pp(name),
        };
        if let Some(source) = vb.weight_source() {
            let load_device = inference_quant::weight_source_load_device(&vb);
            if let Some(weight) = source.load_linear(
                &vb.prefix(),
                &load_device,
                inference_quant::Shard::default(),
            )? {
                let weight = apply_immediate_isq(weight, vb)?;
                return Ok(Self {
                    weights: GraniteParallelExpertWeights::Quantized(weight),
                    output_size,
                    name,
                });
            }
        }
        let all_weights = stacked_vb.get((num_experts, output_size, input_size), name)?;
        if should_apply_immediate_isq(&vb) {
            let layer = Arc::new(inference_quant::UnquantLinear::new(
                inference_quant::QuantMethodConfig::Unquantized(inference_tensor::nn::Linear::new(
                    all_weights,
                    None,
                )),
            )?);
            return Ok(Self {
                weights: GraniteParallelExpertWeights::Quantized(apply_immediate_isq(layer, vb)?),
                output_size,
                name,
            });
        }
        let weights = (0..num_experts)
            .map(|i| all_weights.i(i))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            weights: GraniteParallelExpertWeights::Dense(weights),
            output_size,
            name,
        })
    }

    pub fn quantized(&self) -> Option<&Arc<dyn QuantMethod>> {
        match &self.weights {
            GraniteParallelExpertWeights::Dense(_) => None,
            GraniteParallelExpertWeights::Quantized(weight) => Some(weight),
        }
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        match &self.weights {
            GraniteParallelExpertWeights::Dense(weights) => {
                let weight = Tensor::stack(weights, 0)
                    .expect("Granite expert weight reconstruction should succeed");
                uvb.add_tensor(self.name, weight);
            }
            GraniteParallelExpertWeights::Quantized(weight) => match self.name {
                EXPERT_WEIGHT => uvb.add(weight),
                name => uvb.pp(name).add(weight),
            },
        }
        uvb.to_safetensors()
    }

    fn forward(&self, x: &Tensor, expert_size: &[usize]) -> Result<Tensor> {
        let dtype = x.dtype();
        let device = x.device();

        if let GraniteParallelExpertWeights::Quantized(weight) = &self.weights {
            let expert_ids = expert_size
                .iter()
                .enumerate()
                .flat_map(|(expert, count)| std::iter::repeat_n(expert as u32, *count))
                .collect::<Vec<_>>();
            let rows = expert_ids.len();
            let expert_ids = Tensor::from_vec(expert_ids, (rows, 1), device)?;
            let routed_x = x.unsqueeze(1)?;
            weight.process_routed_stats(&routed_x, &expert_ids)?;
            return weight.gather_forward(&routed_x, &expert_ids)?.squeeze(1);
        }
        let GraniteParallelExpertWeights::Dense(weights) = &self.weights else {
            unreachable!()
        };
        let mut outputs = Vec::new();
        let mut offset = 0;

        for (expert_idx, &size) in expert_size.iter().enumerate() {
            if size == 0 {
                continue;
            }
            let expert_input = x.narrow(0, offset, size)?;
            let expert_output = expert_input.matmul(&weights[expert_idx].t()?)?;
            outputs.push(expert_output);
            offset += size;
        }

        if outputs.is_empty() {
            Tensor::zeros((0, self.output_size), dtype, device)
        } else {
            Tensor::cat(&outputs, 0)
        }
    }
}

/// Sparse Mixture of Experts layer
struct GraniteMoE {
    input_linear: GraniteParallelExperts,
    output_linear: GraniteParallelExperts,
    router: GraniteTopKGating,
    input_size: usize,
    v5_names: bool,
}

impl GraniteMoE {
    fn new(cfg: &Config, vb: ShardedVarBuilder) -> Result<Self> {
        let input_size = cfg.hidden_size;
        let hidden_size = cfg.intermediate_size;
        let num_experts = cfg.num_local_experts;
        let top_k = cfg.num_experts_per_tok;

        let v5_names =
            crate::layers::contains_tensor_or_weight_source(&vb.pp("experts"), V5_GATE_UP);
        let (input_vb, input_name, output_vb, output_name, router_vb) = if v5_names {
            (
                vb.pp("experts"),
                V5_GATE_UP,
                vb.pp("experts"),
                V5_DOWN,
                vb.pp("router"),
            )
        } else {
            (
                vb.pp("input_linear"),
                EXPERT_WEIGHT,
                vb.pp("output_linear"),
                EXPERT_WEIGHT,
                vb.pp("router").pp("layer"),
            )
        };
        Ok(Self {
            input_linear: GraniteParallelExperts::new_named(
                num_experts,
                input_size,
                hidden_size * 2, // Gated
                input_vb,
                input_name,
            )?,
            output_linear: GraniteParallelExperts::new_named(
                num_experts,
                hidden_size,
                input_size,
                output_vb,
                output_name,
            )?,
            router: GraniteTopKGating::new(
                input_size,
                num_experts,
                top_k,
                &cfg.quantization_config,
                router_vb,
            )?,
            input_size,
            v5_names,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, emb_size) = x.dims3()?;
        let dtype = x.dtype();
        let device = x.device();
        let num_tokens = batch_size * seq_len;

        let x_flat = x.reshape((num_tokens, emb_size))?;
        let topk = self.router.topk(&x_flat)?;
        if let (Some(input_linear), Some(output_linear)) = (
            self.input_linear.quantized(),
            self.output_linear.quantized(),
        ) {
            input_linear.process_routed_stats(&x_flat, &topk.indices)?;
            let hidden = input_linear.gather_forward(&x_flat.unsqueeze(1)?, &topk.indices)?;
            let chunks = hidden.chunk(2, inference_tensor::D::Minus1)?;
            let hidden =
                crate::ops::mul_and_act(&chunks[0], &chunks[1], crate::layers::Activation::Silu)?;
            output_linear.process_routed_stats(&hidden, &topk.indices)?;
            let expert_outputs = output_linear.gather_forward(&hidden, &topk.indices)?;
            // a quantized gather on CUDA returns f32 whatever the activation dtype
            return expert_outputs
                .broadcast_mul(&topk.values.to_dtype(expert_outputs.dtype())?.unsqueeze(2)?)?
                .sum(1)?
                .to_dtype(dtype)?
                .reshape((batch_size, seq_len, self.input_size));
        }
        let (batch_index, batch_gates, expert_size) =
            self.router.grouped_routes(&topk, dtype, device)?;

        if batch_index.dim(0)? == 0 {
            return Tensor::zeros((batch_size, seq_len, self.input_size), dtype, device);
        }

        // Route tokens through experts
        let expert_inputs = x_flat.index_select(&batch_index, 0)?;
        let hidden = self.input_linear.forward(&expert_inputs, &expert_size)?;

        // Gated activation: silu(first_half) * second_half
        let chunks = hidden.chunk(2, inference_tensor::D::Minus1)?;
        let hidden =
            crate::ops::mul_and_act(&chunks[0], &chunks[1], crate::layers::Activation::Silu)?;

        let expert_outputs = self.output_linear.forward(&hidden, &expert_size)?;
        let expert_outputs = expert_outputs.broadcast_mul(&batch_gates.unsqueeze(1)?)?;

        // Scatter-add outputs back to token positions
        let batch_index_vec: Vec<i64> = batch_index
            .to_dtype(inference_tensor::DType::I64)?
            .to_vec1()?;
        let expert_outputs_f32 = expert_outputs.to_dtype(inference_tensor::DType::F32)?;
        let num_outputs = expert_outputs_f32.dim(0)?;

        let expert_outputs_vec: Vec<Vec<f32>> = (0..num_outputs)
            .map(|i| expert_outputs_f32.i(i)?.to_vec1())
            .collect::<Result<Vec<_>>>()?;

        let mut output_vec = vec![vec![0.0f32; self.input_size]; batch_size * seq_len];
        for (i, &token_idx) in batch_index_vec.iter().enumerate() {
            let token_idx = token_idx as usize;
            for (j, &val) in expert_outputs_vec[i].iter().enumerate() {
                output_vec[token_idx][j] += val;
            }
        }

        let flat_output: Vec<f32> = output_vec.into_iter().flatten().collect();
        Tensor::from_vec(flat_output, (num_tokens, self.input_size), device)?
            .to_dtype(dtype)?
            .reshape((batch_size, seq_len, self.input_size))
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        if self.v5_names {
            uvb.pp("experts")
                .extend(self.input_linear.residual_tensors());
            uvb.pp("experts")
                .extend(self.output_linear.residual_tensors());
            uvb.pp("router").add(&self.router.layer);
        } else {
            uvb.pp("input_linear")
                .extend(self.input_linear.residual_tensors());
            uvb.pp("output_linear")
                .extend(self.output_linear.residual_tensors());
            uvb.pp("router").pp("layer").add(&self.router.layer);
        }
        uvb.to_safetensors()
    }
}

// Mamba Implementation

/// Per-layer Mamba state cache (local to granite model).
/// Stores conv state and SSM state for recurrent processing.
#[derive(Debug)]
struct MambaLayerCache {
    /// Convolution state: (batch, conv_dim, d_conv)
    pub conv_state: Tensor,
    /// SSM state: (batch, n_heads, head_dim, d_state)
    pub ssm_state: Tensor,
}

impl MambaLayerCache {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        batch_size: usize,
        conv_dim: usize,
        d_conv: usize,
        n_heads: usize,
        head_dim: usize,
        d_state: usize,
        dtype: inference_tensor::DType,
        state_dtype: inference_tensor::DType,
        device: &Device,
    ) -> Result<Self> {
        let conv_state = Tensor::zeros((batch_size, conv_dim, d_conv), dtype, device)?;
        let ssm_state = Tensor::zeros(
            (batch_size, n_heads, head_dim, d_state),
            state_dtype,
            device,
        )?;
        Ok(Self {
            conv_state,
            ssm_state,
        })
    }

    pub fn reset(&mut self) -> Result<()> {
        self.conv_state = self.conv_state.zeros_like()?;
        self.ssm_state = self.ssm_state.zeros_like()?;
        Ok(())
    }
}

impl Clone for MambaLayerCache {
    fn clone(&self) -> Self {
        Self {
            conv_state: self.conv_state.clone(),
            ssm_state: self.ssm_state.clone(),
        }
    }
}

#[derive(Clone, Copy)]
struct PackedMambaShape {
    physical_batch: usize,
    physical_tokens: usize,
    conv_state_batch: usize,
    conv_dim: usize,
    conv_width: usize,
    ssm_state_batch: usize,
    ssm_heads: usize,
    ssm_head_dim: usize,
    ssm_state_width: usize,
    expected_conv_dim: usize,
    expected_conv_width: usize,
    expected_ssm_heads: usize,
    expected_ssm_head_dim: usize,
    expected_ssm_state_width: usize,
}

fn packed_mamba_query_ranges(
    physical_batch: usize,
    physical_tokens: usize,
    query_lens: &[usize],
) -> Result<Vec<Range<usize>>> {
    if physical_batch != 1 {
        inference_tensor::bail!(
            "Granite packed Mamba requires physical batch size 1, got {physical_batch}"
        );
    }
    if query_lens.is_empty() {
        inference_tensor::bail!("Granite packed Mamba requires at least one logical sequence");
    }

    let mut offset = 0usize;
    let mut ranges = Vec::with_capacity(query_lens.len());
    for (sequence_index, &query_len) in query_lens.iter().enumerate() {
        if query_len == 0 {
            inference_tensor::bail!(
                "Granite packed Mamba logical sequence {sequence_index} has zero tokens"
            );
        }
        let end = offset.checked_add(query_len).ok_or_else(|| {
            inference_tensor::Error::msg("Granite packed Mamba query length overflow")
        })?;
        ranges.push(offset..end);
        offset = end;
    }
    if offset != physical_tokens {
        inference_tensor::bail!(
            "Granite packed Mamba has {offset} logical tokens but {physical_tokens} physical tokens"
        );
    }
    Ok(ranges)
}

fn packed_mamba_ranges(shape: PackedMambaShape, query_lens: &[usize]) -> Result<Vec<Range<usize>>> {
    if shape.conv_state_batch != query_lens.len() {
        inference_tensor::bail!(
            "Granite packed Mamba has {} convolution state rows but {} logical sequences",
            shape.conv_state_batch,
            query_lens.len()
        );
    }
    if shape.ssm_state_batch != query_lens.len() {
        inference_tensor::bail!(
            "Granite packed Mamba has {} SSM state rows but {} logical sequences",
            shape.ssm_state_batch,
            query_lens.len()
        );
    }
    if shape.conv_dim != shape.expected_conv_dim || shape.conv_width != shape.expected_conv_width {
        inference_tensor::bail!(
            "Granite packed Mamba convolution state shape mismatch: expected ({}, {}), got ({}, {})",
            shape.expected_conv_dim,
            shape.expected_conv_width,
            shape.conv_dim,
            shape.conv_width
        );
    }
    if shape.ssm_heads != shape.expected_ssm_heads
        || shape.ssm_head_dim != shape.expected_ssm_head_dim
        || shape.ssm_state_width != shape.expected_ssm_state_width
    {
        inference_tensor::bail!(
            "Granite packed Mamba SSM state shape mismatch: expected ({}, {}, {}), got ({}, {}, {})",
            shape.expected_ssm_heads,
            shape.expected_ssm_head_dim,
            shape.expected_ssm_state_width,
            shape.ssm_heads,
            shape.ssm_head_dim,
            shape.ssm_state_width
        );
    }
    packed_mamba_query_ranges(shape.physical_batch, shape.physical_tokens, query_lens)
}

fn softplus(x: &Tensor) -> Result<Tensor> {
    (Tensor::ones_like(x)? + x.exp()?)?.log()
}

fn create_mamba_cache(
    batch_size: usize,
    cfg: &Config,
    dtype: inference_tensor::DType,
    device: &Device,
) -> Result<MambaLayerCache> {
    let conv_dim = cfg.mamba_conv_dim();
    MambaLayerCache::new(
        batch_size,
        conv_dim,
        cfg.mamba_d_conv,
        cfg.mamba_n_heads(),
        cfg.mamba_d_head(),
        cfg.mamba_d_state,
        dtype,
        cfg.mamba_state_dtype().unwrap_or(dtype),
        device,
    )
}

/// RMSNorm with optional gating (for Mamba output), over the whole width or each group of `group_size`
struct RmsNormGated {
    weight: Tensor,
    // what the fused CUDA norm reads
    #[cfg(feature = "cuda")]
    weight_f32: Tensor,
    eps: f64,
    group_size: usize,
}

impl RmsNormGated {
    fn new(
        hidden_size: usize,
        group_size: usize,
        eps: f64,
        vb: ShardedVarBuilder,
        isq_target_device: Option<&Device>,
    ) -> Result<Self> {
        let mut weight = vb.get((hidden_size,), "weight")?;
        // Move weight to target device for ISQ compatibility
        if let Some(target_dev) = isq_target_device {
            weight = weight.to_device(target_dev)?;
        }
        Ok(Self {
            #[cfg(feature = "cuda")]
            weight_f32: weight.to_dtype(inference_tensor::DType::F32)?,
            weight,
            eps,
            group_size,
        })
    }

    fn forward(&self, hidden_states: &Tensor, gate: Option<&Tensor>) -> Result<Tensor> {
        let dtype = hidden_states.dtype();
        #[cfg(feature = "cuda")]
        if let Some(gate) = gate
            && hidden_states.device().is_cuda()
            && dtype == inference_tensor::DType::F32
        {
            return crate::cuda::ssm::mamba_gated_rmsnorm_cuda(
                hidden_states,
                gate,
                &self.weight_f32,
                self.eps,
                self.group_size,
            );
        }
        if let Some(gate) = gate
            && hidden_states.device().is_vulkan()
            && dtype == inference_tensor::DType::F32
            && gate.dtype() == inference_tensor::DType::F32
        {
            return inference_tensor::vulkan::gated_rmsnorm(
                hidden_states,
                gate,
                &self.weight.to_dtype(inference_tensor::DType::F32)?,
                self.eps as f32,
                self.group_size,
            );
        }
        let mut hidden_states = hidden_states.to_dtype(inference_tensor::DType::F32)?;

        // Apply gating if provided
        if let Some(gate) = gate {
            let gate =
                inference_tensor::nn::ops::silu(&gate.to_dtype(inference_tensor::DType::F32)?)?;
            hidden_states = hidden_states.broadcast_mul(&gate)?;
        }

        let shape = hidden_states.dims().to_vec();
        let width = shape[shape.len() - 1];
        let mut grouped = shape[..shape.len() - 1].to_vec();
        grouped.extend([width / self.group_size, self.group_size]);
        let hidden_states = hidden_states.reshape(grouped)?;
        let variance = hidden_states
            .sqr()?
            .mean_keepdim(inference_tensor::D::Minus1)?;
        let hidden_states = hidden_states
            .broadcast_div(&(variance + self.eps)?.sqrt()?)?
            .reshape(shape)?;

        // Apply weight and convert back to original dtype
        hidden_states
            .to_dtype(dtype)?
            .broadcast_mul(&self.weight.to_dtype(dtype)?)
    }
}

/// The scan's per-head constants in F32, as every step reads them: `A` as -exp(A_log), `D` and the dt bias.
struct ScanConstants {
    a: Tensor,
    d: Tensor,
    dt_bias: Tensor,
}

impl ScanConstants {
    fn new(a_log: &Tensor, d: &Tensor, dt_bias: &Tensor) -> Result<Self> {
        let f32 = inference_tensor::DType::F32;
        Ok(Self {
            a: a_log.to_dtype(f32)?.exp()?.neg()?,
            d: d.to_dtype(f32)?,
            dt_bias: dt_bias.to_dtype(f32)?,
        })
    }
}

/// Mamba 1's selection: dt, B and C from the conv's output, dt back to the channels with `dt_proj`'s bias (a GGUF linear
/// carries the bias its file has, so it stays there and the scan's dt bias is zero).
struct Selection {
    x_proj: Arc<dyn QuantMethod>,
    dt_proj: Arc<dyn QuantMethod>,
    dt_rank: usize,
    // the RMSNorms over dt, B and C (Falcon-Mamba's with unit weights)
    norm: Option<[RmsNorm; 3]>,
    weighted_norm: bool,
}

impl Selection {
    // `x` the conv's output `(..., channels)`: `[b, c, dt]`, dt before its bias
    fn project(&self, x: &Tensor, d_state: usize) -> Result<[Tensor; 3]> {
        let flat = x.dims().len() == 2;
        let x = if flat { x.unsqueeze(1)? } else { x.clone() };
        let params = self.x_proj.forward(&x)?;
        let last = inference_tensor::D::Minus1;
        let mut dt = params.narrow(last, 0, self.dt_rank)?;
        let mut b = params.narrow(last, self.dt_rank, d_state)?;
        let mut c = params.narrow(last, self.dt_rank + d_state, d_state)?;
        if let Some([dt_norm, b_norm, c_norm]) = &self.norm {
            // the norm kernel sums its squares in f32, as transformers runs them
            dt = dt_norm.forward(&dt.contiguous()?)?;
            b = b_norm.forward(&b.contiguous()?)?;
            c = c_norm.forward(&c.contiguous()?)?;
        }
        // a slice of x_proj's rows, which a GGUF matmul reads as if contiguous
        let dt = self.dt_proj.forward(&dt.contiguous()?)?;
        let out = [b, c, dt];
        if flat {
            return Ok(out.map(|t| t.squeeze(1).expect("a one-token row")));
        }
        Ok(out)
    }
}

/// The Mamba mixer: Mamba-2's, or Mamba 1's with a `selection`
struct MambaLayer {
    in_proj: Arc<dyn QuantMethod>,
    selection: Option<Selection>,
    conv1d_weight: Tensor,
    conv1d_bias: Option<Tensor>,
    dt_bias: Tensor,
    a_log: Tensor,
    d: Tensor,
    scan: ScanConstants,
    norm: Option<RmsNormGated>,
    out_proj: Arc<dyn QuantMethod>,
    // muP (Falcon-H1): the input and each of z, x, B, C and dt scaled as one vector over the projection, the output
    in_scale: Option<Tensor>,
    out_scale: f32,
    num_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
    ssm_state_size: usize,
    conv_kernel_size: usize,
    n_groups: usize,
    time_step_min: f64,
    time_step_max: f64,
    // the Vulkan conv and scan steps' identity slots for the last batch size, made once rather than once a step
    identity_slots: Mutex<Option<Tensor>>,
}

impl MambaLayer {
    fn identity_slots(&self, rows: usize, device: &Device) -> Result<Tensor> {
        let mut held = self.identity_slots.lock().expect("identity slots poisoned");
        if let Some(slots) = held
            .as_ref()
            .filter(|slots| slots.dim(0).ok() == Some(rows) && slots.device().same_device(device))
        {
            return Ok(slots.clone());
        }
        #[allow(clippy::cast_possible_truncation)]
        let slots = Tensor::arange(0u32, rows as u32, device)?;
        *held = Some(slots.clone());
        Ok(slots)
    }

    fn load(
        vb: ShardedVarBuilder,
        cfg: &Config,
        isq_target_device: Option<&Device>,
    ) -> Result<Self> {
        let intermediate_size = cfg.mamba_intermediate_size();
        let conv_dim = cfg.mamba_conv_dim();
        let num_heads = cfg.mamba_n_heads();
        let head_dim = cfg.mamba_d_head();
        let ssm_state_size = cfg.mamba_d_state;
        let conv_kernel_size = cfg.mamba_d_conv;
        let n_groups = cfg.mamba_n_groups;

        // Mamba 1 projects x and z alone; dt, B and C come after the conv
        let projection_size = match cfg.mamba1 {
            Some(_) => 2 * intermediate_size,
            None => intermediate_size + conv_dim + num_heads,
        };
        let projection_vb = |name: &str| {
            let vb = vb.pp(name);
            if let Some(device) = isq_target_device {
                vb.set_device(device.clone())
            } else {
                vb
            }
        };
        let in_proj = ReplicatedLayer::new(
            cfg.hidden_size,
            projection_size,
            &cfg.quantization_config,
            cfg.mamba_proj_bias,
            projection_vb("in_proj"),
        )?;

        let mut conv1d_weight = vb
            .pp("conv1d")
            .get((conv_dim, 1, conv_kernel_size), "weight")?;
        let mut conv1d_bias = if cfg.mamba_conv_bias {
            Some(vb.pp("conv1d").get(conv_dim, "bias")?)
        } else {
            None
        };

        // NeMo keeps these in float32 under a module of their own
        let params = if vb.contains_tensor(&format!("{NEMO_FP32_PARAMS}.A_log")) {
            vb.pp(NEMO_FP32_PARAMS)
        } else {
            vb.clone()
        };
        let selection = cfg
            .mamba1
            .as_ref()
            .map(|spec| -> Result<Selection> {
                let device = isq_target_device.unwrap_or(vb.device());
                Ok(Selection {
                    x_proj: ReplicatedLayer::new(
                        intermediate_size,
                        spec.dt_rank + 2 * ssm_state_size,
                        &cfg.quantization_config,
                        false,
                        projection_vb("x_proj"),
                    )?,
                    dt_proj: ReplicatedLayer::new(
                        spec.dt_rank,
                        intermediate_size,
                        &cfg.quantization_config,
                        true,
                        projection_vb("dt_proj"),
                    )?,
                    dt_rank: spec.dt_rank,
                    norm: spec
                        .dt_b_c_norm
                        .as_ref()
                        .map(|norm| -> Result<[RmsNorm; 3]> {
                            let make = |name: &str, n: usize| {
                                if norm.weighted {
                                    RmsNorm::new(n, norm.eps, projection_vb(name))
                                } else {
                                    RmsNorm::from_w(Tensor::ones(n, vb.dtype(), device)?, norm.eps)
                                }
                            };
                            let [dt, b, c] = DT_B_C_NORMS;
                            Ok([
                                make(dt, spec.dt_rank)?,
                                make(b, ssm_state_size)?,
                                make(c, ssm_state_size)?,
                            ])
                        })
                        .transpose()?,
                    weighted_norm: spec.dt_b_c_norm.as_ref().is_some_and(|norm| norm.weighted),
                })
            })
            .transpose()?;
        let (mut dt_bias, mut a_log) = match &selection {
            Some(_) => (
                Tensor::zeros(intermediate_size, DType::F32, vb.device())?,
                params.get((num_heads, ssm_state_size), "A_log")?,
            ),
            None => (
                params.get(num_heads, "dt_bias")?,
                params.get(num_heads, "A_log")?,
            ),
        };
        let mut d = params.get(num_heads, "D")?;
        let norm = cfg
            .mamba_gated_norm
            .then(|| {
                RmsNormGated::new(
                    intermediate_size,
                    cfg.mamba_norm_group_size.unwrap_or(intermediate_size),
                    cfg.rms_norm_eps,
                    vb.pp("norm"),
                    isq_target_device,
                )
            })
            .transpose()?;
        let in_scale = cfg
            .mup
            .as_ref()
            .map(|mup| {
                let groups = n_groups * ssm_state_size;
                let [z, x, b, c, dt] = mup.ssm.map(|m| m * mup.ssm_in);
                let scale = [
                    (z, intermediate_size),
                    (x, intermediate_size),
                    (b, groups),
                    (c, groups),
                    (dt, num_heads),
                ]
                .into_iter()
                .flat_map(|(m, n)| std::iter::repeat_n(m, n))
                .collect::<Vec<f32>>();
                Tensor::new(scale, isq_target_device.unwrap_or(vb.device()))
            })
            .transpose()?;

        let out_proj = ReplicatedLayer::new(
            intermediate_size,
            cfg.hidden_size,
            &cfg.quantization_config,
            cfg.mamba_proj_bias,
            projection_vb("out_proj"),
        )?;

        if let Some(target_dev) = isq_target_device {
            tracing::debug!(
                "Moving Mamba weights to {:?} for ISQ compatibility",
                target_dev
            );
            conv1d_weight = conv1d_weight.to_device(target_dev)?;
            if let Some(ref bias) = conv1d_bias {
                conv1d_bias = Some(bias.to_device(target_dev)?);
            }
            dt_bias = dt_bias.to_device(target_dev)?;
            a_log = a_log.to_device(target_dev)?;
            d = d.to_device(target_dev)?;
        }

        let [time_step_min, time_step_max] = cfg.mamba_dt_limit.unwrap_or(NO_DT_LIMIT);
        Ok(Self {
            in_proj,
            selection,
            conv1d_weight,
            conv1d_bias,
            scan: ScanConstants::new(&a_log, &d, &dt_bias)?,
            dt_bias,
            a_log,
            d,
            norm,
            out_proj,
            in_scale,
            out_scale: cfg.mup.as_ref().map_or(1.0, |mup| mup.ssm_out),
            num_heads,
            head_dim,
            intermediate_size,
            ssm_state_size,
            conv_kernel_size,
            n_groups,
            time_step_min,
            time_step_max,
            identity_slots: Mutex::new(None),
        })
    }

    // the scan's output gated by `z` (through the grouped norm when the layer has one), projected back
    fn output(&self, y: &Tensor, gate: &Tensor, dtype: DType) -> Result<Tensor> {
        let y = match &self.norm {
            Some(norm) => norm.forward(y, Some(gate))?,
            None => y
                .to_dtype(DType::F32)?
                .broadcast_mul(&inference_tensor::nn::ops::silu(
                    &gate.to_dtype(DType::F32)?,
                )?)?,
        };
        scale_tensor(self.out_proj.forward(&y.to_dtype(dtype)?)?, self.out_scale)
    }

    fn projected_parts(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let groups_time_state_size = self.n_groups * self.ssm_state_size;
        let mut projected = self.in_proj.forward(x)?;
        if self.selection.is_some() {
            // x then z; dt comes after the conv, so its slot is empty
            let last = inference_tensor::D::Minus1;
            let hidden = projected.narrow(last, 0, self.intermediate_size)?;
            let gate = projected.narrow(last, self.intermediate_size, self.intermediate_size)?;
            let dt = projected.narrow(last, 0, 0)?;
            return Ok((gate, hidden, dt));
        }
        if let Some(scale) = &self.in_scale {
            projected = projected.broadcast_mul(&scale.to_dtype(projected.dtype())?)?;
        }
        let gate = projected.narrow(inference_tensor::D::Minus1, 0, self.intermediate_size)?;
        let hidden_states_b_c = projected.narrow(
            inference_tensor::D::Minus1,
            self.intermediate_size,
            self.intermediate_size + 2 * groups_time_state_size,
        )?;
        let dt = projected.narrow(
            inference_tensor::D::Minus1,
            self.intermediate_size + self.intermediate_size + 2 * groups_time_state_size,
            self.num_heads,
        )?;
        Ok((gate, hidden_states_b_c, dt))
    }

    // the conv's output as the scan reads it, `[x, b, c, dt]`: Mamba-2's x, B and C slices and the projected dt, or
    // Mamba 1's x with its selection
    fn scan_parts(&self, conv_out: &Tensor, dt: &Tensor) -> Result<[Tensor; 4]> {
        let last = inference_tensor::D::Minus1;
        let x = conv_out.narrow(last, 0, self.intermediate_size)?;
        if let Some(selection) = &self.selection {
            let [b, c, dt] = selection.project(&x, self.ssm_state_size)?;
            return Ok([x, b, c, dt]);
        }
        let groups = self.n_groups * self.ssm_state_size;
        let b = conv_out.narrow(last, self.intermediate_size, groups)?;
        let c = conv_out.narrow(last, self.intermediate_size + groups, groups)?;
        Ok([x, b, c, dt.clone()])
    }

    // A for every state item, `(heads, head_dim, state)`: Mamba-2's per head, Mamba 1's per (channel, item)
    fn a_items(&self) -> Result<Tensor> {
        let a = &self.scan.a;
        let a = match a.dims().len() {
            1 => a.unsqueeze(1)?.unsqueeze(2)?,
            _ => a.unsqueeze(1)?,
        };
        a.expand((self.num_heads, self.head_dim, self.ssm_state_size))?
            .to_dtype(DType::F32)
    }

    // a multi-token verify step's pass that also hands back its conv input and dt, for a replay over the kept rows
    fn forward_replayable(
        &self,
        x: &Tensor,
        cache: &mut MambaLayerCache,
    ) -> Result<(Tensor, ReplayInputs)> {
        let (batch_size, seq_len, _) = x.dims3()?;
        let (gate, hidden_states_b_c, dt) = self.projected_parts(x)?;
        let y = self.forward_full(&hidden_states_b_c, &dt, cache, batch_size, seq_len)?;
        let out = self.output(&y, &gate, x.dtype())?;
        Ok((
            out,
            ReplayInputs {
                hidden_states_b_c,
                dt,
            },
        ))
    }

    /// Advances `cache` over the first `rows` positions of a stashed verify step.
    fn replay(
        &self,
        inputs: &ReplayInputs,
        cache: &mut MambaLayerCache,
        rows: usize,
    ) -> Result<Tensor> {
        let batch_size = inputs.dt.dim(0)?;
        self.forward_full(
            &inputs.hidden_states_b_c.narrow(1, 0, rows)?,
            &inputs.dt.narrow(1, 0, rows)?,
            cache,
            batch_size,
            rows,
        )
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.pp("in_proj").add(&self.in_proj);
        let uvb_conv = uvb.pp("conv1d");
        uvb_conv.add_tensor("weight", self.conv1d_weight.clone());
        if let Some(bias) = &self.conv1d_bias {
            uvb_conv.add_tensor("bias", bias.clone());
        }
        match &self.selection {
            Some(selection) => {
                uvb.pp("x_proj").add(&selection.x_proj);
                uvb.pp("dt_proj").add(&selection.dt_proj);
                if let Some(norms) = selection.norm.as_ref().filter(|_| selection.weighted_norm) {
                    for (name, norm) in DT_B_C_NORMS.iter().zip(norms) {
                        uvb.pp(name).add_tensor("weight", norm.weight().clone());
                    }
                }
            }
            None => uvb.add_tensor("dt_bias", self.dt_bias.clone()),
        }
        uvb.add_tensor("A_log", self.a_log.clone());
        uvb.add_tensor("D", self.d.clone());
        if let Some(norm) = &self.norm {
            uvb.pp("norm").add_tensor("weight", norm.weight.clone());
        }
        uvb.pp("out_proj").add(&self.out_proj);
        uvb.to_safetensors()
    }

    // `pool_slots`: `cache` holds the recurrent pool itself, a decode step updating these rows of it in place
    fn forward(
        &self,
        x: &Tensor,
        cache: &mut MambaLayerCache,
        batch_kind: RecurrentBatchKind,
        pool_slots: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;
        let dtype = x.dtype();
        let (gate, hidden_states_b_c, dt) = self.projected_parts(x)?;

        let y = if matches!(batch_kind, RecurrentBatchKind::Decode) {
            if seq_len != 1 {
                inference_tensor::bail!("Mamba decode expects a single-token query.");
            }
            self.forward_cached(
                &hidden_states_b_c.squeeze(1)?,
                &dt.squeeze(1)?,
                cache,
                batch_size,
                pool_slots,
            )?
            .unsqueeze(1)?
        } else if pool_slots.is_some() {
            inference_tensor::bail!(
                "Only a Mamba decode step updates the recurrent pool in place."
            );
        } else {
            self.forward_full(&hidden_states_b_c, &dt, cache, batch_size, seq_len)?
        };

        self.output(&y, &gate, dtype)
    }

    fn forward_packed_prefill(
        &self,
        x: &Tensor,
        cache: &mut MambaLayerCache,
        query_lens: &[usize],
    ) -> Result<Tensor> {
        let (physical_batch, physical_tokens, _) = x.dims3()?;
        let (conv_state_batch, conv_dim, conv_width) = cache.conv_state.dims3()?;
        let (ssm_state_batch, ssm_heads, ssm_head_dim, ssm_state_width) =
            cache.ssm_state.dims4()?;
        let ranges = packed_mamba_ranges(
            PackedMambaShape {
                physical_batch,
                physical_tokens,
                conv_state_batch,
                conv_dim,
                conv_width,
                ssm_state_batch,
                ssm_heads,
                ssm_head_dim,
                ssm_state_width,
                expected_conv_dim: self.conv1d_weight.dim(0)?,
                expected_conv_width: self.conv_kernel_size,
                expected_ssm_heads: self.num_heads,
                expected_ssm_head_dim: self.head_dim,
                expected_ssm_state_width: self.ssm_state_size,
            },
            query_lens,
        )?;

        let dtype = x.dtype();
        let (gate, hidden_states_b_c, dt) = self.projected_parts(x)?;
        let mut outputs = Vec::with_capacity(ranges.len());
        let mut conv_states = Vec::with_capacity(ranges.len());
        let mut ssm_states = Vec::with_capacity(ranges.len());
        for (state_index, range) in ranges.into_iter().enumerate() {
            let mut segment_cache = MambaLayerCache {
                conv_state: cache.conv_state.narrow(0, state_index, 1)?,
                ssm_state: cache.ssm_state.narrow(0, state_index, 1)?,
            };
            outputs.push(self.forward_full(
                &hidden_states_b_c.narrow(1, range.start, range.len())?,
                &dt.narrow(1, range.start, range.len())?,
                &mut segment_cache,
                1,
                range.len(),
            )?);
            conv_states.push(segment_cache.conv_state);
            ssm_states.push(segment_cache.ssm_state);
        }

        cache.conv_state = Tensor::cat(&conv_states, 0)?;
        cache.ssm_state = Tensor::cat(&ssm_states, 0)?;
        let y = Tensor::cat(&outputs, 1)?;
        self.output(&y, &gate, dtype)
    }

    // a prefill whose scan also stops after each capture point's tokens: the output and each point's (conv, ssm)
    // state in the order given; `rows` are each sequence's tokens, along the packed sequence or one batch row apiece
    fn forward_captured(
        &self,
        x: &Tensor,
        cache: &mut MambaLayerCache,
        rows: &[Range<usize>],
        packed: bool,
        points: &[RecurrentCapturePoint],
    ) -> Result<(Tensor, Vec<(Tensor, Tensor)>)> {
        let dtype = x.dtype();
        let (gate, hidden_states_b_c, dt) = self.projected_parts(x)?;
        let mut captured: Vec<Option<(Tensor, Tensor)>> = vec![None; points.len()];
        let mut row_outputs = Vec::with_capacity(rows.len());
        let mut conv_states = Vec::with_capacity(rows.len());
        let mut ssm_states = Vec::with_capacity(rows.len());
        for (state_index, range) in rows.iter().enumerate() {
            let batch = if packed { 0 } else { state_index };
            let (hbc, dt) = (
                hidden_states_b_c.narrow(0, batch, 1)?,
                dt.narrow(0, batch, 1)?,
            );
            let mut segment_cache = MambaLayerCache {
                conv_state: cache.conv_state.narrow(0, state_index, 1)?,
                ssm_state: cache.ssm_state.narrow(0, state_index, 1)?,
            };
            let mut cuts = points
                .iter()
                .enumerate()
                .filter(|(_, point)| point.row == state_index)
                .map(|(i, point)| (point.offset, Some(i)))
                .collect::<Vec<_>>();
            if let Some((offset, _)) = cuts
                .iter()
                .find(|(offset, _)| *offset == 0 || *offset >= range.len())
            {
                inference_tensor::bail!(
                    "a Mamba capture after {offset} of a {}-token sequence is not inside it",
                    range.len()
                );
            }
            cuts.sort_unstable();
            cuts.push((range.len(), None));
            let mut start = 0;
            let mut segments = Vec::with_capacity(cuts.len());
            for (end, point) in cuts {
                segments.push(self.forward_full(
                    &hbc.narrow(1, range.start + start, end - start)?,
                    &dt.narrow(1, range.start + start, end - start)?,
                    &mut segment_cache,
                    1,
                    end - start,
                )?);
                // the scan advances the state it is handed in place, so a capture is a copy
                if let Some(i) = point {
                    captured[i] = Some((
                        segment_cache.conv_state.copy()?,
                        segment_cache.ssm_state.copy()?,
                    ));
                }
                start = end;
            }
            row_outputs.push(Tensor::cat(&segments, 1)?);
            conv_states.push(segment_cache.conv_state);
            ssm_states.push(segment_cache.ssm_state);
        }
        cache.conv_state = Tensor::cat(&conv_states, 0)?;
        cache.ssm_state = Tensor::cat(&ssm_states, 0)?;
        let y = Tensor::cat(&row_outputs, if packed { 1 } else { 0 })?;
        let captured = captured
            .into_iter()
            .map(|state| {
                state.ok_or_else(|| {
                    inference_tensor::Error::msg("a Mamba capture names no sequence")
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((self.output(&y, &gate, dtype)?, captured))
    }

    fn forward_cached(
        &self,
        hidden_states_b_c: &Tensor, // (batch, conv_dim)
        dt: &Tensor,                // (batch, num_heads)
        cache: &mut MambaLayerCache,
        batch_size: usize,
        pool_slots: Option<&Tensor>,
    ) -> Result<Tensor> {
        let hidden_states_b_c = self.conv_step(hidden_states_b_c, cache, pool_slots)?;
        let [hidden_states, b, c, dt] = self.scan_parts(&hidden_states_b_c, dt)?;
        let dt = &dt;

        if pool_slots.is_some() && hidden_states.device().is_metal() {
            inference_tensor::bail!(
                "Only CUDA and the CPU update a Mamba recurrent pool in place."
            );
        }
        // Vulkan runs the CPU's step on its storage until it has its own kernels
        let device = hidden_states.device();
        if device.is_cuda() || device.is_cpu() || device.is_vulkan() {
            return self.scan_step([&hidden_states, &b, &c, dt], cache, batch_size, pool_slots);
        }

        // SSM computation for single token
        // dt with bias and softplus
        let dt_dtype = dt.dtype();
        let dt_bias = self
            .dt_bias
            .to_dtype(dt_dtype)?
            .unsqueeze(0)?
            .expand((batch_size, self.num_heads))?;
        let dt = dt.broadcast_add(&dt_bias)?;
        let dt = softplus(&dt.to_dtype(inference_tensor::DType::F32)?)?;
        // Clamp dt
        let dt = dt.clamp(self.time_step_min, self.time_step_max)?;

        // Expand dimensions for broadcasting
        // dt: (batch, num_heads) -> (batch, num_heads, head_dim)
        let dt = dt
            .unsqueeze(2)?
            .expand((batch_size, self.num_heads, self.head_dim))?;

        let a = self.a_items()?;

        // dA = exp(dt * A): (batch, num_heads, head_dim, state_size)
        let da = dt
            .unsqueeze(3)?
            .to_dtype(inference_tensor::DType::F32)?
            .broadcast_mul(&a.unsqueeze(0)?)?
            .exp()?;

        // Reshape B: (batch, n_groups * state_size) -> (batch, num_heads, state_size)
        let b = b
            .reshape((batch_size, self.n_groups, self.ssm_state_size))?
            .to_dtype(inference_tensor::DType::F32)?;
        let b = b
            .unsqueeze(2)?
            .expand((
                batch_size,
                self.n_groups,
                self.num_heads / self.n_groups,
                self.ssm_state_size,
            ))?
            .reshape((batch_size, self.num_heads, self.ssm_state_size))?;

        // dB = dt * B: (batch, num_heads, head_dim, state_size)
        let dt_f32 = dt.to_dtype(inference_tensor::DType::F32)?;
        let db = dt_f32.unsqueeze(3)?.broadcast_mul(&b.unsqueeze(2)?)?;

        // hidden_states: (batch, intermediate_size) -> (batch, num_heads, head_dim)
        let hidden_states = hidden_states
            .reshape((batch_size, self.num_heads, self.head_dim))?
            .to_dtype(inference_tensor::DType::F32)?;

        // dBx = dB * x: (batch, num_heads, head_dim, state_size)
        let dbx = db.broadcast_mul(&hidden_states.unsqueeze(3)?)?;

        // Update SSM state: state = state * dA + dBx
        let ssm_state = cache
            .ssm_state
            .to_dtype(inference_tensor::DType::F32)?
            .broadcast_mul(&da)?
            .broadcast_add(&dbx)?;
        cache.ssm_state = ssm_state.to_dtype(cache.ssm_state.dtype())?;

        // Reshape C: (batch, n_groups * state_size) -> (batch, num_heads, state_size)
        let c = c
            .reshape((batch_size, self.n_groups, self.ssm_state_size))?
            .to_dtype(inference_tensor::DType::F32)?;
        let c = c
            .unsqueeze(2)?
            .expand((
                batch_size,
                self.n_groups,
                self.num_heads / self.n_groups,
                self.ssm_state_size,
            ))?
            .reshape((batch_size, self.num_heads, self.ssm_state_size))?;

        // y = (state @ C^T): (batch, num_heads, head_dim)
        // state: (batch, num_heads, head_dim, state_size)
        // C: (batch, num_heads, state_size)
        let y = cache
            .ssm_state
            .to_dtype(inference_tensor::DType::F32)?
            .matmul(&c.unsqueeze(3)?)?
            .squeeze(3)?;

        // D skip connection: y = y + x * D
        let d = self
            .d
            .to_dtype(inference_tensor::DType::F32)?
            .unsqueeze(0)?
            .unsqueeze(2)?
            .expand((batch_size, self.num_heads, self.head_dim))?;
        let y = y.broadcast_add(&hidden_states.broadcast_mul(&d)?)?;

        // Reshape output: (batch, num_heads, head_dim) -> (batch, intermediate_size)
        let y = y.reshape((batch_size, self.intermediate_size))?;

        Ok(y)
    }

    // one decode step through the scan kernel (a one-token chunk): `[x, b, c, dt]` as
    // `forward_cached` splits them, the result `(batch, intermediate_size)`
    fn scan_step(
        &self,
        [x, b, c, dt]: [&Tensor; 4],
        cache: &mut MambaLayerCache,
        batch_size: usize,
        pool_slots: Option<&Tensor>,
    ) -> Result<Tensor> {
        let f32 = inference_tensor::DType::F32;
        let by_group = |t: &Tensor| -> Result<Tensor> {
            t.reshape((batch_size, 1, self.n_groups, self.ssm_state_size))?
                .to_dtype(f32)
        };
        // an F32 state is the cache's own storage, which the kernel updates in place
        let mut ssm_state = cache.ssm_state.to_dtype(f32)?;
        let (x, dt, b, c) = (
            x.reshape((batch_size, 1, self.num_heads, self.head_dim))?
                .to_dtype(f32)?,
            dt.reshape((batch_size, 1, self.num_heads))?.to_dtype(f32)?,
            by_group(b)?,
            by_group(c)?,
        );
        let scan = crate::ops::SelectiveScan {
            x: &x,
            dt: &dt,
            a: &self.scan.a,
            b: &b,
            c: &c,
            d: &self.scan.d,
            dt_bias: &self.scan.dt_bias,
            dt_min: self.time_step_min as f32,
            dt_max: self.time_step_max as f32,
        };
        let y = if x.device().is_cuda() {
            crate::cuda::ssm::selective_scan_cuda(scan, &mut ssm_state, pool_slots)?
        } else if let Some(slots) = pool_slots.filter(|_| {
            x.device().is_vulkan()
                && self.ssm_state_size == inference_tensor::vulkan::SCAN_STEP_STATE
                && self.selection.is_none()
        }) {
            inference_tensor::vulkan::ssm_scan_step(&inference_tensor::vulkan::ScanStep {
                x: &x.contiguous()?,
                dt: &dt.contiguous()?,
                a: scan.a,
                b: &b.contiguous()?,
                c: &c.contiguous()?,
                d: scan.d,
                dt_bias: scan.dt_bias,
                dt_range: [scan.dt_min, scan.dt_max],
                pool: &ssm_state,
                slots: &slots
                    .to_device(x.device())?
                    .to_dtype(inference_tensor::DType::U32)?,
            })?
        } else if let Some(slots) = pool_slots {
            crate::ops::selective_scan_cpu_slots(scan, &ssm_state, &slots.to_vec1::<u32>()?)?
        } else {
            crate::ops::selective_scan_cpu(scan, &mut ssm_state)?
        };
        cache.ssm_state = ssm_state.to_dtype(cache.ssm_state.dtype())?;
        y.reshape((batch_size, self.intermediate_size))
    }

    // one decode step's conv and SiLU, `(batch, conv_dim)`, shifting the step into the conv state
    fn conv_step(
        &self,
        hidden_states_b_c: &Tensor,
        cache: &mut MambaLayerCache,
        pool_slots: Option<&Tensor>,
    ) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if hidden_states_b_c.device().is_cuda() {
            // the update kernel shifts the state in place, and a gathered state may share a prefix snapshot's storage
            if pool_slots.is_none() {
                cache.conv_state = cache.conv_state.copy()?;
            }
            return self
                .conv_cuda(&hidden_states_b_c.unsqueeze(1)?, cache, true, pool_slots)?
                .squeeze(1);
        }
        if let Some(slots) = pool_slots.filter(|_| hidden_states_b_c.device().is_vulkan()) {
            let f32 = inference_tensor::DType::F32;
            return inference_tensor::vulkan::ssm_conv_step(
                &hidden_states_b_c
                    .unsqueeze(1)?
                    .to_dtype(f32)?
                    .contiguous()?,
                &self.conv1d_weight.squeeze(1)?.to_dtype(f32)?.contiguous()?,
                self.conv1d_bias
                    .as_ref()
                    .map(|b| b.to_dtype(f32))
                    .transpose()?
                    .as_ref(),
                &cache.conv_state,
                &slots
                    .to_device(hidden_states_b_c.device())?
                    .to_dtype(inference_tensor::DType::U32)?,
            )?
            .squeeze(1)?
            .to_dtype(hidden_states_b_c.dtype());
        }
        if let Some(slots) = pool_slots {
            return crate::ops::causal_conv1d_step_cpu_slots(
                hidden_states_b_c,
                &self.conv1d_weight.squeeze(1)?,
                self.conv1d_bias.as_ref(),
                &cache.conv_state,
                &slots.to_vec1::<u32>()?,
            )?
            .to_dtype(hidden_states_b_c.dtype());
        }
        // Update conv state: roll and insert new values
        // conv_state: (batch, conv_dim, d_conv)
        let conv_state = cache.conv_state.narrow(2, 1, self.conv_kernel_size - 1)?;
        let new_col = hidden_states_b_c.unsqueeze(2)?;
        cache.conv_state = Tensor::cat(&[conv_state, new_col], 2)?;

        // Apply convolution: sum(conv_state * weight)
        // weight: (conv_dim, 1, kernel_size) -> squeeze to (conv_dim, kernel_size)
        let weight = self.conv1d_weight.squeeze(1)?;
        let mut hidden_states_b_c =
            (cache.conv_state.clone() * weight.unsqueeze(0)?)?.sum(inference_tensor::D::Minus1)?;

        if let Some(ref bias) = self.conv1d_bias {
            hidden_states_b_c = hidden_states_b_c.broadcast_add(bias)?;
        }
        let hidden_states_b_c = inference_tensor::nn::ops::silu(&hidden_states_b_c)?;
        Ok(hidden_states_b_c)
    }

    // the causal conv over a prompt chunk and its SiLU, `(batch, seq_len, conv_dim)`, rolling the chunk into the
    // conv state
    fn conv_prefill(
        &self,
        hidden_states_b_c: &Tensor,
        cache: &mut MambaLayerCache,
    ) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if hidden_states_b_c.device().is_cuda() {
            return self.conv_cuda(hidden_states_b_c, cache, false, None);
        }
        let seq_len = hidden_states_b_c.dim(1)?;
        if hidden_states_b_c.device().is_vulkan() {
            let f32 = inference_tensor::DType::F32;
            let device = hidden_states_b_c.device();
            // the kernel shifts the state in place, and a gathered state may share a snapshot's storage
            cache.conv_state = cache.conv_state.to_dtype(f32)?.force_contiguous()?;
            let slots = self.identity_slots(hidden_states_b_c.dim(0)?, device)?;
            return inference_tensor::vulkan::ssm_conv_step(
                &hidden_states_b_c.to_dtype(f32)?.contiguous()?,
                &self.conv1d_weight.squeeze(1)?.to_dtype(f32)?.contiguous()?,
                self.conv1d_bias
                    .as_ref()
                    .map(|b| b.to_dtype(f32))
                    .transpose()?
                    .as_ref(),
                &cache.conv_state,
                &slots,
            )?
            .to_dtype(hidden_states_b_c.dtype());
        }
        let hidden_states_b_c_t = hidden_states_b_c.transpose(1, 2)?;
        let state_width = cache.conv_state.dim(2)?;
        if state_width != self.conv_kernel_size {
            inference_tensor::bail!(
                "Mamba convolution state width is {state_width}, expected {}",
                self.conv_kernel_size
            );
        }
        let prior_state = cache.conv_state.clone();
        let state_and_input = Tensor::cat(&[prior_state.clone(), hidden_states_b_c_t.clone()], 2)?;
        cache.conv_state = state_and_input.narrow(
            2,
            state_and_input.dim(2)? - self.conv_kernel_size,
            self.conv_kernel_size,
        )?;
        let padded_t = Tensor::cat(
            &[
                prior_state.narrow(2, 1, self.conv_kernel_size - 1)?,
                hidden_states_b_c_t,
            ],
            2,
        )?;
        let weight = self.conv1d_weight.squeeze(1)?.to_dtype(padded_t.dtype())?;

        let mut hidden_states_b_c = if padded_t.device().is_cpu() {
            crate::ops::causal_conv1d_cpu(&padded_t, &weight)?
        } else {
            let mut conv_outputs = Vec::with_capacity(seq_len);
            for i in 0..seq_len {
                let window = padded_t.narrow(2, i, self.conv_kernel_size)?;
                let out = (window * weight.unsqueeze(0)?)?.sum(inference_tensor::D::Minus1)?;
                conv_outputs.push(out);
            }
            Tensor::stack(&conv_outputs, 1)? // (batch, seq_len, conv_dim)
        };

        if let Some(ref bias) = self.conv1d_bias {
            let bias = bias.to_dtype(hidden_states_b_c.dtype())?;
            hidden_states_b_c =
                hidden_states_b_c.broadcast_add(&bias.unsqueeze(0)?.unsqueeze(0)?)?;
        }
        let hidden_states_b_c = inference_tensor::nn::ops::silu(&hidden_states_b_c)?;
        Ok(hidden_states_b_c)
    }

    // the fused kernel's conv, bias and SiLU over `(batch, seq_len, conv_dim)` (one step when `is_update`)
    #[cfg(feature = "cuda")]
    fn conv_cuda(
        &self,
        hidden_states_b_c: &Tensor,
        cache: &mut MambaLayerCache,
        is_update: bool,
        pool_slots: Option<&Tensor>,
    ) -> Result<Tensor> {
        let dtype = hidden_states_b_c.dtype();
        let weight = self.conv1d_weight.squeeze(1)?.to_dtype(dtype)?;
        let bias = self
            .conv1d_bias
            .as_ref()
            .map(|bias| bias.to_dtype(dtype))
            .transpose()?;
        let state = cache.conv_state.to_dtype(dtype)?;
        let (activated, state) = crate::cuda::gdn::causal_conv1d_cuda_with_bias(
            hidden_states_b_c,
            &weight,
            bias.as_ref(),
            &state,
            self.conv_kernel_size,
            is_update,
            crate::cuda::gdn::GdnStateSlots::from_option(pool_slots),
        )?;
        cache.conv_state = state.to_dtype(cache.conv_state.dtype())?;
        Ok(activated)
    }

    fn forward_full(
        &self,
        hidden_states_b_c: &Tensor, // (batch, seq_len, conv_dim)
        dt: &Tensor,                // (batch, seq_len, num_heads)
        cache: &mut MambaLayerCache,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        let hidden_states_b_c = self.conv_prefill(hidden_states_b_c, cache)?;
        let [hidden_states, b, c, dt] = self.scan_parts(&hidden_states_b_c, dt)?;
        let dt = &dt;

        // Reshape for SSM first
        let hidden_states = hidden_states
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .to_dtype(inference_tensor::DType::F32)?;
        let b = b
            .reshape((batch_size, seq_len, self.n_groups, self.ssm_state_size))?
            .to_dtype(inference_tensor::DType::F32)?;
        let c = c
            .reshape((batch_size, seq_len, self.n_groups, self.ssm_state_size))?
            .to_dtype(inference_tensor::DType::F32)?;

        let a = self.scan.a.clone();

        // B and C expanded from groups to heads for Metal's scan; the CPU and CUDA ones read each head's group
        let expand = |t: &Tensor| -> Result<Tensor> {
            t.unsqueeze(3)?
                .expand((
                    batch_size,
                    seq_len,
                    self.n_groups,
                    self.num_heads / self.n_groups,
                    self.ssm_state_size,
                ))?
                .reshape((batch_size, seq_len, self.num_heads, self.ssm_state_size))
        };
        #[cfg(feature = "cuda")]
        let use_cuda = matches!(hidden_states.device(), Device::Cuda(_));
        #[cfg(not(feature = "cuda"))]
        let use_cuda = false;

        // Metal's scan takes A per head, which Mamba 1's per-item A is not
        #[cfg(feature = "metal")]
        let use_metal = hidden_states.device().is_metal() && self.selection.is_none();
        #[cfg(not(feature = "metal"))]
        let use_metal = false;

        if use_cuda {
            // the kernel adds dt_bias, then softplus and the clamp
            let dt_f32 = dt.to_dtype(inference_tensor::DType::F32)?;
            let mut ssm_state = cache.ssm_state.to_dtype(inference_tensor::DType::F32)?;
            let y = crate::cuda::ssm::selective_scan_cuda(
                crate::ops::SelectiveScan {
                    x: &hidden_states,
                    dt: &dt_f32,
                    a: &a,
                    b: &b,
                    c: &c,
                    d: &self.scan.d,
                    dt_bias: &self.scan.dt_bias,
                    dt_min: self.time_step_min as f32,
                    dt_max: self.time_step_max as f32,
                },
                &mut ssm_state,
                None,
            )?;

            cache.ssm_state = ssm_state.to_dtype(cache.ssm_state.dtype())?;
            y.reshape((batch_size, seq_len, self.intermediate_size))
        } else if use_metal {
            // Metal kernel handles dt_bias + softplus + clamp internally
            let dt_f32 = dt.to_dtype(inference_tensor::DType::F32)?;
            let dt_bias_f32 = self.dt_bias.to_dtype(inference_tensor::DType::F32)?;
            let d_f32 = self.d.to_dtype(inference_tensor::DType::F32)?;
            let mut ssm_state = cache.ssm_state.to_dtype(inference_tensor::DType::F32)?;

            let y = crate::metal::ssm::selective_scan_metal(
                &hidden_states,
                &dt_f32,
                &a,
                &expand(&b)?,
                &expand(&c)?,
                &d_f32,
                &dt_bias_f32,
                &mut ssm_state,
                self.time_step_min as f32,
                self.time_step_max as f32,
            )?;

            cache.ssm_state = ssm_state.to_dtype(cache.ssm_state.dtype())?;
            y.reshape((batch_size, seq_len, self.intermediate_size))
        } else if hidden_states.device().is_vulkan()
            && self.ssm_state_size == inference_tensor::vulkan::SCAN_STEP_STATE
            && self.selection.is_none()
        {
            // the kernel advances the state in place, and a gathered state may share a snapshot's storage
            let ssm_state = cache
                .ssm_state
                .to_dtype(inference_tensor::DType::F32)?
                .force_contiguous()?;
            let slots = self.identity_slots(batch_size, hidden_states.device())?;
            let y = inference_tensor::vulkan::ssm_scan_step(&inference_tensor::vulkan::ScanStep {
                x: &hidden_states.contiguous()?,
                dt: &dt.to_dtype(inference_tensor::DType::F32)?.contiguous()?,
                a: &a,
                b: &b.contiguous()?,
                c: &c.contiguous()?,
                d: &self.scan.d,
                dt_bias: &self.scan.dt_bias,
                dt_range: [self.time_step_min as f32, self.time_step_max as f32],
                pool: &ssm_state,
                slots: &slots,
            })?;
            cache.ssm_state = ssm_state.to_dtype(cache.ssm_state.dtype())?;
            y.reshape((batch_size, seq_len, self.intermediate_size))
        } else if hidden_states.device().is_cpu() || hidden_states.device().is_vulkan() {
            let mut ssm_state = cache.ssm_state.to_dtype(inference_tensor::DType::F32)?;
            let y = crate::ops::selective_scan_cpu(
                crate::ops::SelectiveScan {
                    x: &hidden_states,
                    dt,
                    a: &a,
                    b: &b,
                    c: &c,
                    d: &self.scan.d,
                    dt_bias: &self.scan.dt_bias,
                    dt_min: self.time_step_min as f32,
                    dt_max: self.time_step_max as f32,
                },
                &mut ssm_state,
            )?;
            cache.ssm_state = ssm_state.to_dtype(cache.ssm_state.dtype())?;
            y.reshape((batch_size, seq_len, self.intermediate_size))
        } else {
            // a device with no scan kernel: per-timestep tensor ops
            let dt_dtype = dt.dtype();
            let dt_bias = self
                .dt_bias
                .to_dtype(dt_dtype)?
                .unsqueeze(0)?
                .unsqueeze(0)?
                .expand((batch_size, seq_len, self.num_heads))?;
            let mut ssm_state = cache.ssm_state.to_dtype(inference_tensor::DType::F32)?;

            let dt = dt.broadcast_add(&dt_bias)?;
            let dt = softplus(&dt.to_dtype(inference_tensor::DType::F32)?)?;
            let dt = dt.clamp(self.time_step_min, self.time_step_max)?;

            let d_coeff = self.d.to_dtype(inference_tensor::DType::F32)?;

            let mut outputs = Vec::with_capacity(seq_len);

            for t in 0..seq_len {
                let dt_t = dt.i((.., t, ..))?.unsqueeze(2)?.expand((
                    batch_size,
                    self.num_heads,
                    self.head_dim,
                ))?;
                let x_t = hidden_states.i((.., t, .., ..))?;
                let b_t = b.i((.., t, .., ..))?;
                let c_t = c.i((.., t, .., ..))?;

                // dA = exp(dt * A)
                let a_expanded = self.a_items()?.unsqueeze(0)?.expand((
                    batch_size,
                    self.num_heads,
                    self.head_dim,
                    self.ssm_state_size,
                ))?;
                let da = dt_t.unsqueeze(3)?.broadcast_mul(&a_expanded)?.exp()?;

                // dB = dt * B
                let db = dt_t.unsqueeze(3)?.broadcast_mul(&b_t.unsqueeze(2)?)?;

                // dBx = dB * x
                let dbx = db.broadcast_mul(&x_t.unsqueeze(3)?)?;

                // Update state: state = state * dA + dBx
                ssm_state = ssm_state.broadcast_mul(&da)?.broadcast_add(&dbx)?;

                // Output: y = state @ C^T
                let y_t = ssm_state.matmul(&c_t.unsqueeze(3)?)?.squeeze(3)?;

                // D skip connection
                let d_expanded = d_coeff.unsqueeze(0)?.unsqueeze(2)?.expand((
                    batch_size,
                    self.num_heads,
                    self.head_dim,
                ))?;
                let y_t = y_t.broadcast_add(&x_t.broadcast_mul(&d_expanded)?)?;

                outputs.push(y_t);
            }

            cache.ssm_state = ssm_state.to_dtype(cache.ssm_state.dtype())?;

            let y = Tensor::stack(&outputs, 1)?;
            y.reshape((batch_size, seq_len, self.intermediate_size))
        }
    }
}

// a block on the CPU beside a GPU computes in f32 over a cache allocated in the model's dtype
fn in_activation_dtype(
    cache: &mut MambaLayerCache,
    dtype: DType,
    run: impl FnOnce(&mut MambaLayerCache) -> Result<Tensor>,
) -> Result<Tensor> {
    let state_dtype = cache.conv_state.dtype();
    if state_dtype == dtype {
        return run(cache);
    }
    cache.conv_state = cache.conv_state.to_dtype(dtype)?;
    let out = run(cache)?;
    cache.conv_state = cache.conv_state.to_dtype(state_dtype)?;
    Ok(out)
}

/// A Mamba layer's conv input `(batch, seq, conv_dim)` and dt `(batch, seq, heads)` from one verify step.
#[derive(Clone)]
struct ReplayInputs {
    hidden_states_b_c: Tensor,
    dt: Tensor,
}

/// One Mamba layer's verify step from the state before it, so a commit can re-advance the pool over only the rows
/// verification kept.
#[derive(Clone)]
struct MambaReplay {
    layer_idx: usize,
    // the cache layer holding its state: `layer_idx` but for a parallel block's
    slot: usize,
    // the pool slots of the step's batch rows, and on the host where the step had them
    indices: Tensor,
    host_indices: Option<Vec<u32>>,
    conv_state: Tensor,
    ssm_state: Tensor,
    inputs: ReplayInputs,
}

/// What a verify step's forward leaves for drafting and the commit: the MTP head's capture or an external drafter's
/// taps (over the logits rows), and the Mamba replays.
#[derive(Clone)]
struct GraniteSpecGraphState {
    mtp: Option<Arc<dyn crate::speculative::SpeculativeGraphState>>,
    taps: Vec<Tensor>,
    replays: Vec<MambaReplay>,
}

impl MambaReplay {
    // the device indices are the graph's slot buffer, which a replay refreshes, so they are not part of the state
    fn tensors(&self) -> [&Tensor; 4] {
        [
            &self.conv_state,
            &self.ssm_state,
            &self.inputs.hidden_states_b_c,
            &self.inputs.dt,
        ]
    }

    fn tensors_mut(&mut self) -> [&mut Tensor; 4] {
        [
            &mut self.conv_state,
            &mut self.ssm_state,
            &mut self.inputs.hidden_states_b_c,
            &mut self.inputs.dt,
        ]
    }
}

impl crate::speculative::SpeculativeGraphState for GraniteSpecGraphState {
    fn tensors(&self) -> Vec<Tensor> {
        let mut out = self.mtp.as_ref().map_or_else(Vec::new, |mtp| mtp.tensors());
        out.extend(self.taps.iter().cloned());
        for replay in &self.replays {
            out.extend(replay.tensors().into_iter().cloned());
        }
        out
    }

    fn with_tensors(
        &self,
        tensors: Vec<Tensor>,
    ) -> Result<Box<dyn crate::speculative::SpeculativeGraphState>> {
        let mtp_len = self.mtp.as_ref().map_or(0, |mtp| mtp.tensors().len());
        let expected = mtp_len + self.taps.len() + self.replays.len() * 4;
        if tensors.len() != expected {
            inference_tensor::bail!(
                "Granite speculative graph state holds {expected} tensors, got {}",
                tensors.len()
            );
        }
        let mut tensors = tensors.into_iter();
        let mtp = self
            .mtp
            .as_ref()
            .map(
                |mtp| -> Result<Arc<dyn crate::speculative::SpeculativeGraphState>> {
                    Ok(Arc::from(
                        mtp.with_tensors(tensors.by_ref().take(mtp_len).collect())?,
                    ))
                },
            )
            .transpose()?;
        let taps = tensors.by_ref().take(self.taps.len()).collect();
        let mut replays = self.replays.clone();
        for replay in &mut replays {
            for (dst, src) in replay.tensors_mut().into_iter().zip(tensors.by_ref()) {
                *dst = src;
            }
        }
        Ok(Box::new(Self { mtp, taps, replays }))
    }

    fn for_real_batch(
        &self,
        real_batch: usize,
    ) -> Result<Box<dyn crate::speculative::SpeculativeGraphState>> {
        let mtp = self
            .mtp
            .as_ref()
            .map(
                |mtp| -> Result<Arc<dyn crate::speculative::SpeculativeGraphState>> {
                    Ok(Arc::from(mtp.for_real_batch(real_batch)?))
                },
            )
            .transpose()?;
        let taps = self
            .taps
            .iter()
            .map(|tap| {
                let captured = tap.dim(0)?;
                if real_batch > captured {
                    inference_tensor::bail!(
                        "tap batch {real_batch} exceeds the captured batch {captured}"
                    );
                }
                tap.narrow(0, 0, real_batch)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut replays = self.replays.clone();
        for replay in &mut replays {
            let captured = replay.indices.dim(0)?;
            if real_batch > captured {
                inference_tensor::bail!(
                    "Mamba replay batch {real_batch} exceeds the captured batch {captured}"
                );
            }
            replay.indices = replay.indices.narrow(0, 0, real_batch)?;
            if let Some(host) = replay.host_indices.as_mut() {
                host.truncate(real_batch);
            }
            for tensor in replay.tensors_mut() {
                *tensor = tensor.narrow(0, 0, real_batch)?;
            }
        }
        Ok(Box::new(Self { mtp, taps, replays }))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Mamba decoder block (replaces attention block for Mamba layers)
struct MambaBlock {
    rms_1: RmsNorm,
    mamba: MambaLayer,
    ffn: Option<BlockFfn>,
    residual_multiplier: f32,
    mixer_tap: MixerTap,
}

/// A Mamba block's input: the residual stream, its norm, and (in a parallel block) the attention output beside it.
struct MambaIn<'a> {
    x: &'a Tensor,
    normed: Tensor,
    side: Option<Tensor>,
}

impl MambaBlock {
    fn input<'a>(&self, x: &'a Tensor) -> Result<MambaIn<'a>> {
        Ok(MambaIn {
            x,
            normed: self.rms_1.forward(x)?,
            side: None,
        })
    }

    // the mixer's output (and the attention beside it) into the residual, then the FFN
    fn finish(&self, input: &MambaIn<'_>, out: Tensor) -> Result<Tensor> {
        let mixed = match &input.side {
            Some(side) => (out + side)?,
            None => out,
        };
        let x = (scale_tensor(mixed, self.residual_multiplier)? + input.x)?;
        self.mixer_tap.record(&x);
        BlockFfn::residual(self.ffn.as_ref(), x, self.residual_multiplier)
    }

    fn forward(
        &self,
        input: MambaIn<'_>,
        cache: &mut MambaLayerCache,
        batch_kind: RecurrentBatchKind,
        pool_slots: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mamba_out = in_activation_dtype(cache, input.normed.dtype(), |cache| {
            self.mamba
                .forward(&input.normed, cache, batch_kind, pool_slots)
        })?;
        self.finish(&input, mamba_out)
    }

    fn forward_replayable(
        &self,
        input: MambaIn<'_>,
        cache: &mut MambaLayerCache,
    ) -> Result<(Tensor, ReplayInputs)> {
        let mut inputs = None;
        let mamba_out = in_activation_dtype(cache, input.normed.dtype(), |cache| {
            let (out, replay) = self.mamba.forward_replayable(&input.normed, cache)?;
            inputs = Some(replay);
            Ok(out)
        })?;
        let out = self.finish(&input, mamba_out)?;
        Ok((out, inputs.expect("the Mamba pass ran")))
    }

    fn replay(
        &self,
        inputs: &ReplayInputs,
        cache: &mut MambaLayerCache,
        rows: usize,
    ) -> Result<()> {
        in_activation_dtype(cache, inputs.dt.dtype(), |cache| {
            self.mamba.replay(inputs, cache, rows)
        })?;
        Ok(())
    }

    fn forward_packed_prefill(
        &self,
        input: MambaIn<'_>,
        cache: &mut MambaLayerCache,
        query_lens: &[usize],
    ) -> Result<Tensor> {
        let mamba_out = in_activation_dtype(cache, input.normed.dtype(), |cache| {
            self.mamba
                .forward_packed_prefill(&input.normed, cache, query_lens)
        })?;
        self.finish(&input, mamba_out)
    }

    // `forward`'s prefill with `MambaLayer::forward_captured`'s states, the conv states in the cache's dtype
    fn forward_captured(
        &self,
        input: MambaIn<'_>,
        cache: &mut MambaLayerCache,
        rows: &[Range<usize>],
        packed: bool,
        points: &[RecurrentCapturePoint],
    ) -> Result<(Tensor, Vec<(Tensor, Tensor)>)> {
        let conv_dtype = cache.conv_state.dtype();
        let mut captured = Vec::new();
        let mamba_out = in_activation_dtype(cache, input.normed.dtype(), |cache| {
            let (out, states) =
                self.mamba
                    .forward_captured(&input.normed, cache, rows, packed, points)?;
            captured = states;
            Ok(out)
        })?;
        let captured = captured
            .into_iter()
            .map(|(conv, ssm)| Ok((conv.to_dtype(conv_dtype)?, ssm)))
            .collect::<Result<Vec<_>>>()?;
        let out = self.finish(&input, mamba_out)?;
        Ok((out, captured))
    }

    fn load(vb_m: &ShardedVarBuilder, layout: &HybridLayout, load: BlockLoad<'_>) -> Result<Self> {
        let names = &layout.blocks[load.layer_idx].names;
        let isq_target_device = if load.loading_isq {
            load.mapper.device_for(load.layer_idx, false)
        } else {
            None
        };
        let mixer = names.mamba.as_ref().unwrap_or(&names.mixer);
        let mamba = MambaLayer::load(
            load.mapper
                .set_device(load.layer_idx, vb_m.pp(mixer), load.loading_isq),
            load.cfg,
            isq_target_device,
        )?;
        Ok(Self {
            rms_1: load.norm(vb_m.pp(&names.norm))?,
            mamba,
            ffn: BlockFfn::load(vb_m, layout, &load)?,
            residual_multiplier: load.cfg.residual_multiplier,
            mixer_tap: MixerTap::default(),
        })
    }
}

/// Falcon-H1's layer: attention and a Mamba mixer on one input norm, their outputs summed into the residual. The
/// Mamba block holds the norm and the FFN; its recurrent state is a cache layer of its own after the stack's.
struct ParallelBlock {
    attn: Box<CausalSelfAttention>,
    mamba: MambaBlock,
}

impl ParallelBlock {
    fn load(
        vb_m: &ShardedVarBuilder,
        layout: &HybridLayout,
        load: BlockLoad<'_>,
        rope: Option<Arc<RotaryEmbedding>>,
        paged_attn: Option<PagedAttention>,
    ) -> Result<Self> {
        let names = &layout.blocks[load.layer_idx].names;
        let attn = CausalSelfAttention::load(
            load.mapper
                .set_device(load.layer_idx, vb_m.pp(&names.mixer), load.loading_isq),
            load.cfg,
            load.layer_idx,
            rope,
            paged_attn,
            load.comm,
        )?;
        Ok(Self {
            attn: Box::new(attn),
            mamba: MambaBlock::load(vb_m, layout, load)?,
        })
    }
}

/// Enum to represent either an attention or mamba layer
enum DecoderLayer {
    Attention(Block),
    Mamba(MambaBlock),
    Parallel(ParallelBlock),
}

impl DecoderLayer {
    fn ffn(&self) -> Option<&BlockFfn> {
        match self {
            Self::Attention(block) => block.ffn.as_ref(),
            Self::Mamba(block) | Self::Parallel(ParallelBlock { mamba: block, .. }) => {
                block.ffn.as_ref()
            }
        }
    }

    fn ffn_mut(&mut self) -> &mut Option<BlockFfn> {
        match self {
            Self::Attention(block) => &mut block.ffn,
            Self::Mamba(block) | Self::Parallel(ParallelBlock { mamba: block, .. }) => {
                &mut block.ffn
            }
        }
    }

    fn mixer_tap(&self) -> &MixerTap {
        match self {
            Self::Attention(block) => &block.mixer_tap,
            Self::Mamba(block) | Self::Parallel(ParallelBlock { mamba: block, .. }) => {
                &block.mixer_tap
            }
        }
    }

    fn mixer_tap_mut(&mut self) -> &mut MixerTap {
        match self {
            Self::Attention(block) => &mut block.mixer_tap,
            Self::Mamba(block) | Self::Parallel(ParallelBlock { mamba: block, .. }) => {
                &mut block.mixer_tap
            }
        }
    }

    fn mamba(&self) -> Option<&MambaBlock> {
        match self {
            Self::Attention(_) => None,
            Self::Mamba(block) | Self::Parallel(ParallelBlock { mamba: block, .. }) => Some(block),
        }
    }
}

/// A block's residual stream between its mixer and its FFN, kept when an external drafter taps a checkpoint layer
/// there (Nemotron-H counts the MoE after a mixer as a layer of its own).
#[derive(Default)]
struct MixerTap {
    enabled: bool,
    last: Mutex<Option<Tensor>>,
}

impl MixerTap {
    fn record(&self, x: &Tensor) {
        if self.enabled {
            *self.last.lock().expect("mixer tap poisoned") = Some(x.clone());
        }
    }

    fn take(&self) -> Option<Tensor> {
        self.last.lock().expect("mixer tap poisoned").take()
    }
}

/// Where a drafter's tapped checkpoint layer ends inside the model's blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TapPoint {
    AfterMixer(usize),
    AfterBlock(usize),
}

/// Nemotron-H's MTP block: an attention mixer and the MoE after it, as the stack's attention blocks are.
struct GraniteMtpBlock {
    block: Block,
    names: BlockNames,
}

impl MtpBlock for GraniteMtpBlock {
    fn forward(
        &self,
        xs: &Tensor,
        kv_cache: &mut KvCache,
        ctx: &mut ModelForwardContext<'_>,
        kv_layer_idx: usize,
    ) -> Result<Tensor> {
        self.block
            .forward(xs, &AttentionMask::None, kv_cache, ctx, kv_layer_idx)
    }

    fn add_residual(&self, uvb_m: &UnVarBuilder) {
        uvb_m.pp(&self.names.norm).add(&self.block.rms_1);
        if let (Some(ffn), Some(names)) = (&self.block.ffn, &self.names.ffn) {
            uvb_m.pp(&names.norm).add(&ffn.norm);
            if let Some(moe) = &ffn.moe {
                uvb_m.pp(&names.module).extend(moe.residual_tensors());
            }
        }
    }
}

/// Where a checkpoint's MTP head sits under its own prefix: its norms and projection, and one block.
#[derive(Clone, Debug)]
pub struct HybridMtpLayout {
    pub block: HybridBlock,
    pub names: MtpSpec,
    /// The head's final norm is a weight-only LayerNorm (Nemotron-H's `final_layernorm`), not an RMS norm.
    pub layer_norm_final: bool,
}

// Use HybridLayerCache from kv_cache instead of a local type alias

// End Mamba Implementation

struct CausalSelfAttention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rotary_emb: Option<Arc<RotaryEmbedding>>, // Optional - None when position_embedding_type == "nope"
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
    // muP (Falcon-H1): the input, the keys and the output scaled
    in_scale: f32,
    key_scale: f32,
    out_scale: f32,
}

impl CausalSelfAttention {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        attention_mask: &AttentionMask,
        kv_cache: &mut KvCache,
        ctx: &mut ModelForwardContext<'_>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;
        let x = &scale_tensor(x.clone(), self.in_scale)?;

        let (mut q, mut k, mut v) =
            crate::ops::qkv_projections(x, &*self.q_proj, &*self.k_proj, &*self.v_proj)?;
        k = scale_tensor(k, self.key_scale)?;
        (q, k, v) = if seq_len != 1 {
            let q = q
                .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
                .transpose(1, 2)?;
            let k = k
                .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v)
        } else {
            let q = q.reshape((b_sz, self.num_attention_heads, seq_len, self.head_dim))?;
            let k = k.reshape((b_sz, self.num_key_value_heads, seq_len, self.head_dim))?;
            let v = v.reshape((b_sz, self.num_key_value_heads, seq_len, self.head_dim))?;
            (q, k, v)
        };

        (q, k) = if let Some(ref rotary_emb) = self.rotary_emb {
            let positions = ctx
                .text_positions(q.device(), q.dim(2)?)?
                .ok_or_else(|| inference_tensor::Error::msg("missing RoPE positions"))?;
            rotary_emb.forward(&q, &k, positions)?
        } else {
            (q, k)
        };

        let metadata = ctx.paged_layer(layer_idx);
        let flash_params = ctx.flash_params();
        let mut y = AttentionDispatch {
            paged_attn: self.paged_attn.as_ref(),
            paged_layer: metadata,
            kv_cache,
            sdpa_params: &self.sdpa_params,
            flash_params,
        }
        .run(&q, &k, &v, attention_mask)?;

        y = if !matches!(attention_mask, AttentionMask::None) {
            y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?
        } else {
            y.reshape((b_sz, seq_len, ()))?
        };
        scale_tensor(self.o_proj.forward(&y)?, self.out_scale)
    }

    fn load(
        vb: ShardedVarBuilder,
        cfg: &Config,
        layer_idx: usize,
        rope: Option<Arc<RotaryEmbedding>>,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<inference_quant::Comm>,
    ) -> Result<Self> {
        let num_attention_heads = cfg.num_attention_heads / comm.world_size();
        let sliding_window = match cfg.layer_types().get(layer_idx) {
            Some(GraniteLayerType::SlidingAttention) => cfg.sliding_window,
            _ => None,
        };
        // Granite SWA's per-head sinks: softmax(QK)V scaled by sigmoid(lse - sink), the gpt-oss sink
        let sinks = if vb.contains_tensor("sinks") {
            let sinks = vb.get(cfg.num_attention_heads, "sinks")?;
            Some(sinks.narrow(0, comm.rank() * num_attention_heads, num_attention_heads)?)
        } else {
            None
        };
        let size_in = cfg.hidden_size;
        let size_q = cfg.head_dim() * cfg.num_attention_heads;
        let size_kv = cfg.head_dim() * cfg.num_key_value_heads();
        let q_proj = ColumnParallelLayer::new(
            size_in,
            size_q,
            &cfg.quantization_config,
            false,
            comm,
            vb.pp("q_proj"),
        )?;
        let kv_shard =
            inference_quant::compute_kv_shard(cfg.num_key_value_heads(), cfg.head_dim(), comm)?;
        let k_proj = ColumnParallelLayer::new_with_shard(
            size_in,
            size_kv,
            &cfg.quantization_config,
            false,
            comm,
            kv_shard,
            vb.pp("k_proj"),
        )?;
        let v_proj = ColumnParallelLayer::new_with_shard(
            size_in,
            size_kv,
            &cfg.quantization_config,
            false,
            comm,
            kv_shard,
            vb.pp("v_proj"),
        )?;
        let o_proj = RowParallelLayer::new(
            size_q,
            size_in,
            &cfg.quantization_config,
            false,
            comm,
            vb.pp("o_proj"),
        )?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads,
            num_key_value_heads: (cfg.num_key_value_heads() / comm.world_size()).max(1),
            head_dim: cfg.head_dim(),
            rotary_emb: rope, // Now optional
            paged_attn,
            in_scale: cfg.mup.as_ref().map_or(1.0, |mup| mup.attention_in),
            key_scale: cfg.mup.as_ref().map_or(1.0, |mup| mup.key),
            out_scale: cfg.mup.as_ref().map_or(1.0, |mup| mup.attention_out),
            sdpa_params: SdpaParams {
                n_kv_groups: inference_quant::compute_n_kv_groups(
                    cfg.num_key_value_heads(),
                    cfg.num_attention_heads,
                    comm,
                )?,
                softcap: None,
                // GraniteMoeHybrid uses attention_multiplier instead of 1/sqrt(d)
                softmax_scale: cfg.attention_multiplier,
                sliding_window,
                sinks,
                chunk: None,
            },
        })
    }
}

struct Block {
    rms_1: RmsNorm,
    attn: CausalSelfAttention,
    ffn: Option<BlockFfn>,
    residual_multiplier: f32,
    mixer_tap: MixerTap,
}

impl Block {
    fn forward(
        &self,
        x: &Tensor,
        attention_mask: &AttentionMask,
        kv_cache: &mut KvCache,
        ctx: &mut ModelForwardContext<'_>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let attn_out = self.attn.forward(
            &self.rms_1.forward(x)?,
            attention_mask,
            kv_cache,
            ctx,
            layer_idx,
        )?;
        let x = (scale_tensor(attn_out, self.residual_multiplier)? + x)?;
        self.mixer_tap.record(&x);
        BlockFfn::residual(self.ffn.as_ref(), x, self.residual_multiplier)
    }

    fn load(
        vb_m: &ShardedVarBuilder,
        layout: &HybridLayout,
        load: BlockLoad<'_>,
        rope: Option<Arc<RotaryEmbedding>>, // None when position_embedding_type == "nope"
        paged_attn: Option<PagedAttention>,
    ) -> Result<Self> {
        let names = &layout.blocks[load.layer_idx].names;
        let attn = CausalSelfAttention::load(
            load.mapper
                .set_device(load.layer_idx, vb_m.pp(&names.mixer), load.loading_isq),
            load.cfg,
            load.layer_idx,
            rope,
            paged_attn,
            load.comm,
        )?;
        Ok(Self {
            rms_1: load.norm(vb_m.pp(&names.norm))?,
            attn,
            ffn: BlockFfn::load(vb_m, layout, &load)?,
            residual_multiplier: load.cfg.residual_multiplier,
            mixer_tap: MixerTap::default(),
        })
    }
}

/// What every block's load takes.
#[derive(Clone, Copy)]
struct BlockLoad<'a> {
    cfg: &'a Config,
    mapper: &'a dyn DeviceMapper,
    layer_idx: usize,
    loading_isq: bool,
    comm: &'a Arc<inference_quant::Comm>,
}

impl BlockLoad<'_> {
    fn norm(&self, vb: ShardedVarBuilder) -> Result<RmsNorm> {
        RmsNorm::new(
            self.cfg.hidden_size,
            self.cfg.rms_norm_eps,
            self.mapper.set_device(self.layer_idx, vb, false),
        )
    }
}

/// A block's feed-forward half, after its own norm: Granite's dense MLP and/or experts, or one Nemotron-H layer.
struct BlockFfn {
    norm: RmsNorm,
    mlp: Option<Box<dyn MlpLayer>>,
    moe: Option<HybridMoe>,
}

enum HybridMoe {
    Granite(GraniteMoE),
    Nemotron(NemotronMoe),
}

impl BlockFfn {
    fn load(
        vb_m: &ShardedVarBuilder,
        layout: &HybridLayout,
        load: &BlockLoad<'_>,
    ) -> Result<Option<Self>> {
        let Some(ffn) = &layout.blocks[load.layer_idx].names.ffn else {
            return Ok(None);
        };
        let (cfg, layer_idx, isq) = (load.cfg, load.layer_idx, load.loading_isq);
        let vb = load.mapper.set_device(layer_idx, vb_m.pp(&ffn.module), isq);
        let (mlp, moe) = match ffn.kind {
            FfnKind::Granite => (
                load_dense_ffn(vb.clone(), cfg, load.comm)?,
                (cfg.num_local_experts > 0)
                    .then(|| GraniteMoE::new(cfg, vb.pp("block_sparse_moe")))
                    .transpose()?
                    .map(HybridMoe::Granite),
            ),
            FfnKind::SwiGlu => (
                Some(match &cfg.mup {
                    Some(mup) => Box::new(MupMlp::new(vb, cfg, mup.mlp, load.comm)?),
                    None => Box::new(crate::layers::Mlp::new(
                        vb,
                        cfg.hidden_size,
                        cfg.intermediate_size,
                        &cfg.quantization_config,
                        crate::layers::Activation::Silu,
                        load.comm,
                    )?) as Box<dyn MlpLayer>,
                }),
                None,
            ),
            FfnKind::Mlp => (
                Some(Box::new(PlainMlp::new(
                    vb,
                    &[cfg.hidden_size, cfg.intermediate_size],
                    &NEMOTRON_MLP,
                    false,
                    &cfg.quantization_config,
                    crate::layers::Activation::Relu2,
                    load.comm,
                )?) as Box<dyn MlpLayer>),
                None,
            ),
            FfnKind::Moe => {
                let spec = ffn.moe.as_ref().ok_or_else(|| {
                    inference_tensor::Error::msg("a Nemotron-H MoE layer without its MoE settings")
                })?;
                let device = load
                    .mapper
                    .device_for(layer_idx, false)
                    .cloned()
                    .unwrap_or_else(|| vb.device().clone());
                (
                    None,
                    Some(HybridMoe::Nemotron(NemotronMoe::load(
                        cfg, spec, vb, device, load.comm, isq,
                    )?)),
                )
            }
        };
        Ok(Some(Self {
            norm: load.norm(vb_m.pp(&ffn.norm))?,
            mlp,
            moe,
        }))
    }

    /// `x` plus the feed-forward of its norm, when the block has one.
    fn residual(ffn: Option<&Self>, x: Tensor, residual_multiplier: f32) -> Result<Tensor> {
        let Some(ffn) = ffn else {
            return Ok(x);
        };
        let normed = ffn.norm.forward(&x)?;
        let out = match (ffn.mlp.as_deref(), &ffn.moe) {
            (Some(mlp), Some(moe)) => (mlp.forward(&normed)? + moe.forward(&normed)?)?,
            (Some(mlp), None) => mlp.forward(&normed)?,
            (None, Some(moe)) => moe.forward(&normed)?,
            (None, None) => {
                inference_tensor::bail!("Granite layer has neither dense nor expert FFN weights")
            }
        };
        scale_tensor(out, residual_multiplier)? + x
    }
}

impl HybridMoe {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Granite(moe) => moe.forward(xs),
            Self::Nemotron(moe) => moe.forward(xs),
        }
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        match self {
            Self::Granite(moe) => {
                let uvb = UnVarBuilder::new();
                uvb.pp("block_sparse_moe").extend(moe.residual_tensors());
                uvb.to_safetensors()
            }
            Self::Nemotron(moe) => moe.residual_tensors(),
        }
    }
}

/// Nemotron-H's experts: a sigmoid router choosing by score plus a correction bias (weighing by the unbiased scores,
/// renormalized and scaled), squared-ReLU experts with no gate, and one shared expert beside them.
struct NemotronMoe {
    gate: inference_tensor::nn::Linear,
    // Nemotron-H's selection bias; Jamba routes without one
    bias: Option<Tensor>,
    experts: MoEExperts,
    latent: Option<NemotronLatent>,
    shared: Option<PlainMlp>,
    top_k: usize,
    scale: f32,
    routing: MoeRouting,
}

impl NemotronMoe {
    fn load(
        cfg: &Config,
        spec: &NemotronMoeSpec,
        vb: ShardedVarBuilder,
        device: Device,
        comm: &Arc<inference_quant::Comm>,
        loading_isq: bool,
    ) -> Result<Self> {
        let jamba = spec.routing == MoeRouting::Jamba;
        let router_vb = vb
            .pp(if jamba { JAMBA_ROUTER } else { NEMOTRON_ROUTER })
            .set_device(device.clone())
            .set_dtype(DType::F32);
        // a GGUF of a Puzzle checkpoint gives each layer's expert count only by its router's rows
        let router = router_vb.get_unchecked("weight")?;
        let num_experts = router.dim(0)?;
        let gate = inference_tensor::nn::Linear::new(router, None);
        let bias = (!jamba)
            .then(|| router_vb.get(num_experts, NEMOTRON_SELECTION_BIAS))
            .transpose()?;
        let latent = spec
            .latent_size
            .map(|size| NemotronLatent::load(cfg, size, &vb))
            .transpose()?;
        let experts = MoEExperts::new(
            &MoEExpertsConfig {
                num_experts,
                num_experts_per_tok: spec.top_k,
                hidden_size: spec.latent_size.unwrap_or(cfg.hidden_size),
                moe_intermediate_size: spec.intermediate_size,
                expert_proj_names: ExpertProjNames::DEFAULT,
                gated: jamba,
            },
            vb.clone(),
            device,
            comm,
            loading_isq,
            &cfg.quantization_config,
            if jamba {
                crate::layers::Activation::Silu
            } else {
                crate::layers::Activation::Relu2
            },
        )?;
        let shared = (spec.shared_intermediate_size > 0)
            .then(|| {
                PlainMlp::new(
                    vb.pp(NEMOTRON_SHARED_EXPERTS),
                    &[cfg.hidden_size, spec.shared_intermediate_size],
                    &NEMOTRON_MLP,
                    false,
                    &cfg.quantization_config,
                    crate::layers::Activation::Relu2,
                    comm,
                )
            })
            .transpose()?;
        Ok(Self {
            gate,
            bias,
            experts,
            latent,
            shared,
            top_k: spec.top_k,
            scale: spec.routed_scaling_factor,
            routing: spec.routing,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_size, seq_len, hidden) = xs.dims3()?;
        let xs_flat = xs.reshape(((), hidden))?;
        let logits = self.gate.forward(&xs_flat.to_dtype(DType::F32)?)?;
        // Jamba: the softmax's top k as they are, unnormalized
        let (score_function, renormalize) = match self.routing {
            MoeRouting::Nemotron => (crate::ops::MoeRouterScoreFunction::Sigmoid, true),
            MoeRouting::Jamba => (crate::ops::MoeRouterScoreFunction::Softmax, false),
        };
        let topk = crate::ops::moe_router_topk(
            &logits,
            crate::ops::MoeRouterTopKConfig {
                top_k: self.top_k,
                score_function,
                selected_weight: crate::ops::MoeRouterSelectedWeight::Score,
                renormalize,
                norm_min: NEMOTRON_NORM_MIN,
                output_scale: self.scale,
                logit_clip: None,
            },
            self.bias.as_ref(),
            None,
        )?;
        let routed_in = match &self.latent {
            Some(latent) => latent.down.forward(xs)?,
            None => xs.clone(),
        };
        let Some(shared_mlp) = &self.shared else {
            let routed = self
                .experts
                .forward(&routed_in, topk.values, &topk.indices)?;
            return routed
                .reshape((b_size, seq_len, routed_in.dims3()?.2))?
                .to_dtype(xs.dtype());
        };
        // the shared expert runs while offloaded experts' misses are on the CPU
        let (routed, shared) = self.experts.forward_overlapping(
            &routed_in,
            topk.values,
            &topk.indices,
            &mut || MlpLayer::forward(shared_mlp, xs),
        )?;
        let routed = routed
            .reshape((b_size, seq_len, routed_in.dims3()?.2))?
            .to_dtype(xs.dtype())?;
        let routed = match &self.latent {
            Some(latent) => latent.up.forward(&routed)?,
            None => routed,
        };
        routed + shared
    }

    // the router stays as the checkpoint has it; ISQ takes the experts and the shared expert
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        let router = uvb.pp(match self.routing {
            MoeRouting::Nemotron => NEMOTRON_ROUTER,
            MoeRouting::Jamba => JAMBA_ROUTER,
        });
        router.add(&self.gate);
        if let Some(bias) = &self.bias {
            router.add_tensor(NEMOTRON_SELECTION_BIAS, bias.clone());
        }
        if let Some(latent) = &self.latent {
            uvb.pp(NEMOTRON_LATENT_DOWN).add(&latent.down);
            uvb.pp(NEMOTRON_LATENT_UP).add(&latent.up);
        }
        uvb.to_safetensors()
    }
}

struct NemotronLatent {
    down: Arc<dyn QuantMethod>,
    up: Arc<dyn QuantMethod>,
}

impl NemotronLatent {
    fn load(cfg: &Config, size: usize, vb: &ShardedVarBuilder) -> Result<Self> {
        let projection = |(input, output), name| {
            ReplicatedLayer::new(input, output, &cfg.quantization_config, false, vb.pp(name))
        };
        Ok(Self {
            down: projection((cfg.hidden_size, size), NEMOTRON_LATENT_DOWN)?,
            up: projection((size, cfg.hidden_size), NEMOTRON_LATENT_UP)?,
        })
    }
}

fn scale_tensor(tensor: Tensor, scale: f32) -> Result<Tensor> {
    if (scale - 1.0).abs() < f32::EPSILON {
        Ok(tensor)
    } else {
        tensor.affine(scale as f64, 0.)
    }
}

// Granite 4 fuses gate and up into shared_mlp.input_linear; dense Granite 3.x / 4.1 keeps a Llama-style mlp.
fn load_dense_ffn(
    vb: ShardedVarBuilder,
    cfg: &Config,
    comm: &Arc<inference_quant::Comm>,
) -> Result<Option<Box<dyn MlpLayer>>> {
    if crate::layers::contains_tensor_or_weight_source(
        &vb.pp("shared_mlp").pp("input_linear"),
        "weight",
    ) {
        return Ok(Some(Box::new(GraniteMlp::new(vb, cfg, comm)?)));
    }
    let mlp_vb = vb.pp("mlp");
    if crate::layers::contains_tensor_or_weight_source(&mlp_vb.pp("gate_proj"), "weight") {
        return Ok(Some(Box::new(crate::layers::Mlp::new(
            mlp_vb,
            cfg.hidden_size,
            cfg.intermediate_size,
            &cfg.quantization_config,
            crate::layers::Activation::Silu,
            comm,
        )?)));
    }
    Ok(None)
}

/// The routed experts of a Nemotron-H MoE layer.
#[derive(Clone, Debug)]
pub struct NemotronMoeSpec {
    pub num_experts: usize,
    pub top_k: usize,
    pub intermediate_size: usize,
    pub shared_intermediate_size: usize,
    pub routed_scaling_factor: f32,
    /// The width the routed experts run at, when it is not the model's.
    pub latent_size: Option<usize>,
    pub routing: MoeRouting,
}

/// How a hybrid MoE routes and what its experts are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeRouting {
    /// Nemotron-H: a biased sigmoid, its top k renormalized and scaled; squared-ReLU experts beside a shared one.
    Nemotron,
    /// Jamba: `router`'s softmax top k, unnormalized; SwiGLU experts, no shared expert.
    Jamba,
}

/// Which feed-forward a block's FFN half loads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FfnKind {
    /// Granite's: a dense MLP (fused `shared_mlp` or Llama-style `mlp`) and/or `block_sparse_moe`.
    Granite,
    /// Nemotron-H's squared-ReLU `up_proj`/`down_proj`.
    Mlp,
    /// Falcon-H1's SwiGLU `gate_proj`/`up_proj`/`down_proj`.
    SwiGlu,
    /// Nemotron-H's experts ([`NemotronMoeSpec`]).
    Moe,
}

/// A block's FFN half: its norm's prefix, its module's, and which feed-forward that is.
#[derive(Clone, Debug)]
pub struct FfnNames {
    pub norm: String,
    pub module: String,
    pub kind: FfnKind,
    /// A Nemotron-H MoE layer's experts.
    pub moe: Option<NemotronMoeSpec>,
}

/// Where one block's weights sit, relative to the model's prefix.
#[derive(Clone, Debug)]
pub struct BlockNames {
    pub norm: String,
    pub mixer: String,
    /// A parallel block's Mamba mixer, beside its attention `mixer`.
    pub mamba: Option<String>,
    pub ffn: Option<FfnNames>,
}

#[derive(Clone, Debug)]
pub struct HybridBlock {
    pub mixer: GraniteLayerType,
    pub names: BlockNames,
}

/// A sequence's recurrent state over `mamba_layers` layers: the conv window in `dtype`, the SSM state in the config's
/// state dtype (else `dtype`).
pub fn recurrent_state_bytes(cfg: &Config, mamba_layers: usize, dtype: DType) -> usize {
    mamba_layers
        * (cfg.mamba_conv_dim() * cfg.mamba_d_conv * dtype.size_in_bytes()
            + cfg.mamba_n_heads()
                * cfg.mamba_d_head()
                * cfg.mamba_d_state
                * cfg.mamba_state_dtype().unwrap_or(dtype).size_in_bytes())
}

/// The paged cache's view of a hybrid layout: only attention layers page (a parallel block's recurrent slots follow
/// the stack's, an MTP head's after those), and each sequence holds `recurrent_state_bytes`.
pub fn hybrid_paged_kv_config(
    base: ModelConfigMetadata,
    layout: &HybridLayout,
    mtp: bool,
    recurrent_state_bytes: usize,
) -> HybridPagedKvCacheConfig {
    let parallel = layout
        .blocks
        .iter()
        .filter(|block| block.mixer == GraniteLayerType::Parallel)
        .count();
    let paged_layers = layout
        .blocks
        .iter()
        .map(|block| !matches!(block.mixer, GraniteLayerType::Mamba))
        .chain(std::iter::repeat_n(false, parallel))
        .chain(mtp.then_some(true))
        .collect();
    HybridPagedKvCacheConfig::new(base, paged_layers)
        .with_recurrent_state_bytes(recurrent_state_bytes)
}

/// The model's blocks and the names around them. Granite keeps one block per checkpoint layer; Nemotron-H, whose
/// layers each hold one mixer or one feed-forward, makes a block of a mixer layer and the feed-forward after it.
#[derive(Clone, Debug)]
pub struct HybridLayout {
    pub blocks: Vec<HybridBlock>,
    pub embeddings: &'static str,
    pub final_norm: &'static str,
}

impl HybridLayout {
    pub fn granite(cfg: &Config) -> Self {
        let blocks = cfg
            .layer_types()
            .into_iter()
            .enumerate()
            .map(|(i, mixer)| {
                let layer = format!("layers.{i}");
                let mixer_name = match mixer {
                    GraniteLayerType::Mamba => "mamba",
                    _ => "self_attn",
                };
                HybridBlock {
                    names: BlockNames {
                        norm: format!("{layer}.input_layernorm"),
                        mixer: format!("{layer}.{mixer_name}"),
                        mamba: None,
                        ffn: Some(FfnNames {
                            norm: format!("{layer}.post_attention_layernorm"),
                            module: layer,
                            kind: FfnKind::Granite,
                            moe: None,
                        }),
                    },
                    mixer,
                }
            })
            .collect();
        Self {
            blocks,
            embeddings: "embed_tokens",
            final_norm: "norm",
        }
    }

    /// Where each checkpoint layer's output is, by the `layers.N` of its block's mixer and FFN names, and how many
    /// checkpoint layers there are.
    fn tap_points(&self, layers: &[usize]) -> Result<(Vec<TapPoint>, usize)> {
        let layer_of = |name: &str| {
            name.strip_prefix("layers.")
                .and_then(|rest| rest.split('.').next())
                .and_then(|n| n.parse::<usize>().ok())
        };
        let mut points = HashMap::new();
        for (block, b) in self.blocks.iter().enumerate() {
            let mixer = layer_of(&b.names.mixer);
            let ffn = b.names.ffn.as_ref().and_then(|ffn| layer_of(&ffn.module));
            match (mixer, ffn) {
                (Some(mixer), Some(ffn)) if mixer != ffn => {
                    points.insert(mixer, TapPoint::AfterMixer(block));
                    points.insert(ffn, TapPoint::AfterBlock(block));
                }
                (Some(layer), _) | (None, Some(layer)) => {
                    points.insert(layer, TapPoint::AfterBlock(block));
                }
                (None, None) => {}
            }
        }
        let count = points.keys().max().map_or(0, |last| last + 1);
        let taps = layers
            .iter()
            .map(|layer| {
                points.get(layer).copied().ok_or_else(|| {
                    inference_tensor::Error::msg(format!(
                        "a drafter taps checkpoint layer {layer}, which this model's blocks do not hold"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((taps, count))
    }

    /// The checkpoint layer each tensor name's `layers.N` belongs to, mapped to its block.
    pub fn block_of_layer(&self) -> HashMap<usize, usize> {
        let layer_of = |name: &str| {
            name.strip_prefix("layers.")
                .and_then(|rest| rest.split('.').next())
                .and_then(|n| n.parse::<usize>().ok())
        };
        let mut map = HashMap::new();
        for (block, b) in self.blocks.iter().enumerate() {
            let names = [
                Some(&b.names.mixer),
                b.names.ffn.as_ref().map(|f| &f.module),
            ];
            for layer in names.into_iter().flatten().filter_map(|n| layer_of(n)) {
                map.insert(layer, block);
            }
        }
        map
    }
}

// a layer's KV cache: the model's normal cache, else the pipeline's hybrid cache, else the model's own
fn attention_cache<'a>(
    layer_idx: usize,
    normal: &'a mut Option<std::sync::MutexGuard<'_, crate::kv_cache::NormalCache>>,
    pipeline: &'a mut Option<std::sync::MutexGuard<'_, HybridCache>>,
    internal: &'a mut GraniteHybridCache,
) -> Option<&'a mut KvCache> {
    if let Some(normal) = normal.as_mut() {
        return normal.0.get_mut(layer_idx);
    }
    if let Some(HybridLayerCache::Attention(kv_cache)) =
        pipeline.as_mut().and_then(|cache| cache.get_mut(layer_idx))
    {
        return Some(kv_cache);
    }
    match &mut internal.caches[layer_idx] {
        GraniteLayerCache::Attention(kv_cache) => Some(kv_cache),
        GraniteLayerCache::Mamba(_) => None,
    }
}

/// The cache's layers: each block's own (a parallel block's as attention), then a recurrent one per parallel block;
/// and each block's recurrent slot.
fn cache_slots(layer_types: &[GraniteLayerType]) -> (Vec<GraniteLayerType>, Vec<Option<usize>>) {
    let mut slots = layer_types
        .iter()
        .map(|t| match t {
            GraniteLayerType::Parallel => GraniteLayerType::Attention,
            t => t.clone(),
        })
        .collect::<Vec<_>>();
    let recurrent = layer_types
        .iter()
        .enumerate()
        .map(|(layer, t)| match t {
            GraniteLayerType::Mamba => Some(layer),
            GraniteLayerType::Parallel => {
                slots.push(GraniteLayerType::Mamba);
                Some(slots.len() - 1)
            }
            _ => None,
        })
        .collect();
    (slots, recurrent)
}

/// Local enum to represent either a KV cache or Mamba cache for a layer
/// (used internally by GraniteMoeHybrid - not exported)
enum GraniteLayerCache {
    Attention(KvCache),
    Mamba(MambaLayerCache),
}

/// Hybrid cache that can store either KV cache or Mamba cache per layer
/// (local to granite model - wraps kv_cache::HybridCache for pipeline integration)
struct GraniteHybridCache {
    pub caches: Vec<GraniteLayerCache>,
}

impl GraniteHybridCache {
    pub fn new(
        layer_types: &[GraniteLayerType],
        cfg: &Config,
        layer_devices: &[Device],
        dtype: inference_tensor::DType,
    ) -> Result<Self> {
        if layer_devices.len() != layer_types.len() {
            inference_tensor::bail!(
                "Granite hybrid cache has {} layers but {} layer devices",
                layer_types.len(),
                layer_devices.len()
            );
        }
        let mut caches = Vec::with_capacity(layer_types.len());
        for (layer_type, device) in layer_types.iter().zip(layer_devices) {
            match layer_type {
                GraniteLayerType::Attention
                | GraniteLayerType::SlidingAttention
                | GraniteLayerType::Parallel => {
                    caches.push(GraniteLayerCache::Attention(KvCache::new_normal(
                        2,
                        cfg.max_position_embeddings,
                        HybridCache::CACHE_GROW_SIZE,
                    )));
                }
                GraniteLayerType::Mamba => {
                    caches.push(GraniteLayerCache::Mamba(create_mamba_cache(
                        1, cfg, dtype, device,
                    )?));
                }
            }
        }
        Ok(Self { caches })
    }

    pub fn seqlen(&self) -> usize {
        // Return the seqlen from the first attention layer
        for cache in &self.caches {
            if let GraniteLayerCache::Attention(kv) = cache {
                return kv.current_seq_len();
            }
        }
        // If no attention layers, return 0
        0
    }
}

impl PastKvLenCache for GraniteHybridCache {
    fn get_past_kv_len(&self) -> Result<usize> {
        Ok(self.seqlen())
    }
}

pub struct GraniteMoeHybrid {
    wte: Arc<dyn QuantMethod>,
    layout: HybridLayout,
    layers: Vec<DecoderLayer>,
    layer_types: Vec<GraniteLayerType>,
    // each layer's recurrent state's cache slot: its own index for a Mamba block, one after the stack's for a parallel
    recurrent_slots: Vec<Option<usize>>,
    ln_f: RmsNorm,
    lm_head: Arc<dyn QuantMethod>,
    // the lm_head's weights, which an MTP draft head re-quantizes
    lm_head_elements: usize,
    dtype: DType,
    hybrid_cache: Arc<Mutex<GraniteHybridCache>>,
    // EitherCache for pipeline integration
    kv_cache: EitherCache,
    device: Device,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    cfg: ModelConfigMetadata,
    embedding_multiplier: f32,
    logits_scaling: f32,
    max_seq_len: usize,
    sliding_window: Option<usize>,
    has_sinks: bool,
    // one sequence's conv and SSM state over the Mamba layers
    recurrent_state_bytes: usize,
    mtp: Option<MtpHead<GraniteMtpBlock>>,
    // the last verify step's Mamba layers, which a commit replays over the rows it keeps
    mamba_replay: Mutex<Vec<MambaReplay>>,
    // an external drafter (DFlash, DSpark), the layers whose outputs it reads, and the head it drafts with
    drafter: Mutex<Option<Arc<ExternalDrafter>>>,
    tap_points: Vec<TapPoint>,
    draft_lm_head: Option<Arc<dyn QuantMethod>>,
    // the last forward's taps: over its logits rows, and over every row of a prompt
    spec_taps: Mutex<Option<Vec<Tensor>>>,
    prompt_taps: Mutex<Option<Vec<Tensor>>>,
}

/// Runs before decoder layer `layer_idx` on its input hidden states, as Granite 4 Vision injects image features.
pub type GraniteLayerHook<'a> = &'a dyn Fn(usize, Tensor) -> Result<Tensor>;

impl GraniteMoeHybrid {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        Self::new_inner(
            cfg,
            vb.pp("model"),
            vb.pp("lm_head"),
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )
    }

    pub fn new_inner(
        cfg: &Config,
        vb_m: ShardedVarBuilder,
        vb_lm_head: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        Self::new_with_layout(
            cfg,
            HybridLayout::granite(cfg),
            vb_m,
            vb_lm_head,
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )
    }

    /// As [`Self::new_inner`], with the blocks and names `layout` gives (Nemotron-H's).
    pub fn new_with_layout(
        cfg: &Config,
        layout: HybridLayout,
        vb_m: ShardedVarBuilder,
        vb_lm_head: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        if let Some(quant_cfg) = &cfg.quantization_config {
            tracing::info!(
                "Using {} quantization: {}.",
                quant_cfg.name(),
                quant_cfg.get_bits_name(&vb_m)
            );
        }
        let mapper = normal_loading_metadata.mapper;
        let dtype = vb_m.dtype();

        let wte = embedding_with_legacy_tied_uqff(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(
                vb_m.pp(layout.embeddings),
                normal_loading_metadata.loading_isq,
            ),
            cfg.tie_word_embeddings.then(|| {
                mapper.set_nm_device(vb_lm_head.clone(), normal_loading_metadata.loading_isq)
            }),
            &cfg.quantization_config,
        )?;
        let lm_head = if !cfg.tie_word_embeddings {
            ReplicatedLayer::new(
                cfg.hidden_size,
                cfg.vocab_size,
                &cfg.quantization_config,
                false,
                mapper.set_nm_device(vb_lm_head, normal_loading_metadata.loading_isq),
            )?
        } else {
            wte.clone()
        };
        let ln_f = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp(layout.final_norm), false),
        )?;

        let head_dim = cfg.head_dim();

        // Note: granite rope_type scaling is not yet supported, using default rope
        if let Some(GraniteRopeConfig {
            rope_type: GraniteRopeType::Granite,
            ..
        }) = &cfg.rope_scaling
        {
            tracing::warn!(
                "Granite-style rope scaling is not yet fully supported. Using default rope scaling."
            );
        }

        let mut ropes = HashMap::new();
        for i in 0..cfg.num_hidden_layers {
            let Some(theta) = cfg.layer_rope_theta(i) else {
                continue;
            };
            let device = mapper
                .device_for(i, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            if let std::collections::hash_map::Entry::Vacant(e) =
                ropes.entry((device.location(), theta.to_bits()))
            {
                let rope = RotaryEmbedding::new(
                    theta,
                    head_dim,
                    cfg.max_position_embeddings,
                    device,
                    is_gptx,
                    vb_m.dtype(),
                )?;
                e.insert(Arc::new(rope));
            }
        }

        let layer_types = layout
            .blocks
            .iter()
            .map(|block| block.mixer.clone())
            .collect::<Vec<_>>();
        let layer_devices = (0..layer_types.len())
            .map(|layer_idx| {
                mapper
                    .device_for(layer_idx, false)
                    .unwrap_or(&normal_loading_metadata.real_device)
                    .clone()
            })
            .collect::<Vec<_>>();

        // Log layer configuration
        let num_mamba = layer_types
            .iter()
            .filter(|t| t.has_recurrent_state())
            .count();
        let num_attn = layer_types.iter().filter(|t| t.is_attention()).count();
        tracing::info!(
            "GraniteMoeHybrid: {} attention layers, {} mamba layers",
            num_attn,
            num_mamba
        );

        // Build layers based on layer_types
        let num_blocks = layer_types.len();
        let mut layers = Vec::with_capacity(num_blocks);
        for i in NiceProgressBar::<_, 'b'>(
            0..layer_types.len(),
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        ) {
            let device = mapper
                .device_for(i, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            let comm = mapper.get_comm_for(i)?;
            let load = BlockLoad {
                cfg,
                mapper: &*mapper,
                layer_idx: i,
                loading_isq: normal_loading_metadata.loading_isq,
                comm: &comm,
            };

            let layer = match &layer_types[i] {
                GraniteLayerType::Attention
                | GraniteLayerType::SlidingAttention
                | GraniteLayerType::Parallel => {
                    let rotary_emb = cfg.layer_rope_theta(i).map(|theta| {
                        ropes
                            .get(&(device.location(), theta.to_bits()))
                            .expect("No RoPE for device location!")
                            .clone()
                    });
                    let paged_attn = match &attention_mechanism {
                        AttentionImplementation::Eager => None,
                        AttentionImplementation::PagedAttention => {
                            Some(PagedAttention::new(head_dim, device, None)?)
                        }
                    };
                    if layer_types[i] == GraniteLayerType::Parallel {
                        DecoderLayer::Parallel(ParallelBlock::load(
                            &vb_m, &layout, load, rotary_emb, paged_attn,
                        )?)
                    } else {
                        DecoderLayer::Attention(Block::load(
                            &vb_m, &layout, load, rotary_emb, paged_attn,
                        )?)
                    }
                }
                GraniteLayerType::Mamba => {
                    DecoderLayer::Mamba(MambaBlock::load(&vb_m, &layout, load)?)
                }
            };
            layers.push(layer);
        }

        // a parallel block pages its attention at its own index; its recurrent state is a slot after the stack's
        let (slot_types, recurrent_slots) = cache_slots(&layer_types);
        let slot_devices = (0..slot_types.len())
            .map(|slot| {
                let layer = recurrent_slots
                    .iter()
                    .position(|&s| s == Some(slot))
                    .unwrap_or(slot);
                layer_devices[layer].clone()
            })
            .collect::<Vec<_>>();

        // Create hybrid cache for internal use
        let hybrid_cache = Arc::new(Mutex::new(GraniteHybridCache::new(
            &slot_types,
            cfg,
            &slot_devices,
            vb_m.dtype(),
        )?));

        // Create pipeline-compatible hybrid cache config
        let pipeline_layer_types: Vec<HybridLayerType> = slot_types
            .iter()
            .map(|lt| match lt {
                GraniteLayerType::Mamba => HybridLayerType::Recurrent,
                _ => HybridLayerType::Attention,
            })
            .collect();

        let hybrid_cache_config = HybridCacheConfig {
            layer_types: pipeline_layer_types,
            max_seq_len: cfg.max_position_embeddings,
            recurrent: RecurrentLayerConfig {
                conv_dim: cfg.mamba_conv_dim(),
                conv_width: cfg.mamba_d_conv,
                state: crate::kv_cache::RecurrentStateSpec::Opaque {
                    dims: vec![cfg.mamba_n_heads(), cfg.mamba_d_head(), cfg.mamba_d_state],
                },
                recurrent_dtype: cfg.mamba_state_dtype(),
            },
            auxiliary: Vec::new(),
        };

        let pipeline_cache = Arc::new(Mutex::new(
            HybridCache::new(hybrid_cache_config, vb_m.dtype(), &slot_devices).map_err(|e| {
                inference_tensor::Error::Msg(format!("Failed to create hybrid cache: {}", e))
            })?,
        ));

        let num_attention_heads = cfg.num_attention_heads / mapper.get_comm_for(0)?.world_size();
        let has_mamba = layer_types
            .iter()
            .any(GraniteLayerType::has_recurrent_state);
        let has_sliding = layer_types.contains(&GraniteLayerType::SlidingAttention);
        if has_mamba && has_sliding {
            inference_tensor::bail!(
                "Granite with both Mamba and sliding-window layers is not supported"
            );
        }
        let has_sinks = layers.iter().any(|layer| {
            matches!(layer, DecoderLayer::Attention(block) if block.attn.sdpa_params.sinks.is_some())
        });
        let mamba_layers = layer_types
            .iter()
            .filter(|layer_type| layer_type.has_recurrent_state())
            .count();
        let recurrent_state_bytes = recurrent_state_bytes(cfg, mamba_layers, vb_m.dtype());

        Ok(Self {
            layout,
            sliding_window: has_sliding.then_some(cfg.sliding_window).flatten(),
            has_sinks,
            recurrent_state_bytes,
            mtp: None,
            mamba_replay: Mutex::new(Vec::new()),
            drafter: Mutex::new(None),
            tap_points: Vec::new(),
            draft_lm_head: None,
            spec_taps: Mutex::new(None),
            prompt_taps: Mutex::new(None),
            lm_head_elements: cfg.vocab_size * cfg.hidden_size,
            wte,
            layers,
            layer_types,
            recurrent_slots,
            ln_f,
            lm_head,
            dtype,
            hybrid_cache,
            // an all-attention Granite keeps plain KV caches: a hybrid one would reserve recurrent state it has none of
            kv_cache: match has_mamba {
                true => EitherCache::Hybrid(pipeline_cache),
                false => EitherCache::Normal(crate::kv_cache::NormalCache::from_types(
                    cfg.layer_types()
                        .iter()
                        .map(|layer_type| match (layer_type, cfg.sliding_window) {
                            (GraniteLayerType::SlidingAttention, Some(window)) => {
                                NormalCacheType::SlidingWindow { window }
                            }
                            _ => NormalCacheType::Normal {
                                max_seq_len: cfg.max_position_embeddings,
                            },
                        })
                        .collect(),
                )),
            },
            device: normal_loading_metadata.real_device,
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: slot_types.len(),
                hidden_size: cfg.hidden_size,
                num_kv_heads: (cfg.num_key_value_heads() / mapper.get_comm_for(0)?.world_size())
                    .max(1),
                num_attn_heads: num_attention_heads,
                sliding_window: has_sliding.then_some(cfg.sliding_window).flatten(),
                k_head_dim: cfg.head_dim(),
                v_head_dim: cfg.head_dim(),
                kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
            },
            mapper,
            embedding_multiplier: cfg.embedding_multiplier,
            logits_scaling: if cfg.logits_scaling == 0.0 {
                1.0
            } else {
                1.0 / cfg.logits_scaling
            },
            max_seq_len: cfg.max_position_embeddings,
        })
    }

    /// The paged cache's view: a hybrid model pages only its attention layers, whose prefix prefill reads the paged
    /// cache directly unless sinks need the gathered path.
    /// The checkpoint's MTP head as the paged layer after the stack's, its block loaded as the stack's attention blocks.
    pub fn with_mtp(
        mut self,
        cfg: &Config,
        mtp: HybridMtpLayout,
        vb: ShardedVarBuilder,
        loading_isq: bool,
        attention_mechanism: &AttentionImplementation,
    ) -> Result<Self> {
        if self.layer_types.contains(&GraniteLayerType::Parallel) {
            inference_tensor::bail!(
                "an MTP head beside parallel attention and Mamba blocks is not supported"
            );
        }
        let kv_layer_idx = self.layers.len();
        let mut layout = self.layout.clone();
        layout.blocks.push(mtp.block.clone());
        let comm = self.mapper.get_comm_for(kv_layer_idx)?;
        let load = BlockLoad {
            cfg,
            mapper: &*self.mapper,
            layer_idx: kv_layer_idx,
            loading_isq,
            comm: &comm,
        };
        let device = self.device.clone();
        let paged_attn = match attention_mechanism {
            AttentionImplementation::Eager => None,
            AttentionImplementation::PagedAttention => {
                Some(PagedAttention::new(cfg.head_dim(), &device, None)?)
            }
        };
        // with the model's experts offloaded the head's stay on the GPU, requantized small by the loader: its misses
        // would cost each draft more than its forward
        let block = Block::load(
            &vb.clone().with_moe_expert_cache(None),
            &layout,
            load,
            None,
            paged_attn,
        )?;
        let norm = |name: &str| -> Result<Norm> {
            Ok(Norm::Rms(RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                self.mapper.set_nm_device(vb.pp(name), false),
            )?))
        };
        let final_norm = match mtp.names.final_norm {
            Some(name) if mtp.layer_norm_final => Some(Norm::Layer(crate::layers::layer_norm(
                cfg.hidden_size,
                inference_tensor::nn::LayerNormConfig {
                    eps: cfg.rms_norm_eps,
                    remove_mean: true,
                    affine: false,
                },
                self.mapper.set_nm_device(vb.pp(name), false),
            )?)),
            Some(name) => Some(norm(name)?),
            None => None,
        };
        let head = MtpHead::from_parts(MtpHeadParts {
            embedding_norm: norm(mtp.names.embedding_norm)?,
            hidden_norm: norm(mtp.names.hidden_norm)?,
            projection: ReplicatedLayer::new(
                2 * cfg.hidden_size,
                cfg.hidden_size,
                &cfg.quantization_config,
                false,
                self.mapper
                    .set_nm_device(vb.pp(mtp.names.projection), loading_isq),
            )?,
            block: GraniteMtpBlock {
                block,
                names: mtp.block.names,
            },
            final_norm,
            names: mtp.names,
            chain_normed: true,
            kv_layer_idx,
            device,
            dtype: self.dtype,
        });
        self.mtp = Some(head);
        self.cfg.num_layers += 1;
        Ok(self)
    }

    pub fn paged_kv_config(&self) -> Arc<dyn ModelConfigLike + Send + Sync> {
        if !self
            .layer_types
            .iter()
            .any(GraniteLayerType::has_recurrent_state)
        {
            return Arc::new(self.cfg.clone());
        }
        let config = hybrid_paged_kv_config(
            self.cfg.clone(),
            &self.layout,
            self.mtp.is_some(),
            self.recurrent_state_bytes,
        );
        Arc::new(match self.has_sinks {
            true => config,
            false => config.with_uniform_prefix_prefill_attention_features(Default::default()),
        })
    }

    pub fn forward(&self, input_ids: &Tensor, ctx: &mut ModelForwardContext<'_>) -> Result<Tensor> {
        let x = self.embed(input_ids)?;
        self.forward_embeds(input_ids, x, ctx, &|_, x| Ok(x))
    }

    /// Leaves the logits undivided by `logits_scaling`, as Granite Speech's own `lm_head` does in transformers.
    pub fn without_logits_scaling(mut self) -> Self {
        self.logits_scaling = 1.0;
        self
    }

    /// The token embeddings, scaled by the embedding multiplier.
    pub fn embed(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.scale_embeddings(self.wte.embedding_forward(input_ids, self.dtype)?)
    }

    /// The token embeddings before the embedding multiplier, which transformers applies to whatever it is given.
    pub fn token_embeddings(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.wte.embedding_forward(input_ids, self.dtype)
    }

    pub fn scale_embeddings(&self, embeddings: Tensor) -> Result<Tensor> {
        scale_tensor(embeddings, self.embedding_multiplier)
    }

    pub fn forward_embeds(
        &self,
        input_ids: &Tensor,
        mut x: Tensor,
        ctx: &mut ModelForwardContext<'_>,
        before_layer: GraniteLayerHook<'_>,
    ) -> Result<Tensor> {
        // Get both internal cache and pipeline cache
        let mut internal_cache = self.hybrid_cache.lock().unwrap();
        let mut normal_cache =
            matches!(self.kv_cache, EitherCache::Normal(_)).then(|| self.kv_cache.normal());
        let mut pipeline_cache =
            matches!(self.kv_cache, EitherCache::Hybrid(_)).then(|| self.kv_cache.hybrid());

        let recurrent_metadata = ctx.recurrent_metadata().cloned();
        let has_mamba_layers = self
            .layer_types
            .iter()
            .any(|layer_type| matches!(layer_type, GraniteLayerType::Mamba));
        let packed_query_lens = if ctx.flash_params().packed {
            Some(
                ctx.paged_input_metadata()
                    .and_then(|metadata| metadata.query_lens.clone())
                    .ok_or_else(|| {
                        inference_tensor::Error::msg(
                            "Granite packed prefill requires logical query lengths",
                        )
                    })?,
            )
        } else {
            None
        };
        if has_mamba_layers && let Some(query_lens) = packed_query_lens.as_deref() {
            let metadata = recurrent_metadata.as_ref().ok_or_else(|| {
                inference_tensor::Error::msg(
                    "Granite packed Mamba requires hybrid recurrent metadata",
                )
            })?;
            if metadata.batch_kind() != RecurrentBatchKind::Prefill {
                inference_tensor::bail!("Granite packed Mamba cannot run a decode batch");
            }
            let (physical_batch, physical_tokens, _) = x.dims3()?;
            packed_mamba_query_ranges(physical_batch, physical_tokens, query_lens)?;
            let index_count = metadata.state_indices().dims1()?;
            if index_count != query_lens.len() {
                inference_tensor::bail!(
                    "Granite packed Mamba has {index_count} recurrent state indices but {} logical sequences",
                    query_lens.len()
                );
            }
            if let Some(host_indices) = metadata.state_indices_host()
                && host_indices.len() != query_lens.len()
            {
                inference_tensor::bail!(
                    "Granite packed Mamba has {} host state indices but {} logical sequences",
                    host_indices.len(),
                    query_lens.len()
                );
            }
        }
        let recurrent_batch_kind = recurrent_metadata
            .as_ref()
            .map(|metadata| metadata.batch_kind())
            .or_else(|| ctx.recurrent_batch_kind())
            .unwrap_or(RecurrentBatchKind::Prefill);

        let paged_mask_cache = ForwardMaskCache::Paged(ctx.seqlen_offsets());
        let mask_cache = match (&normal_cache, &pipeline_cache) {
            _ if ctx.is_paged() => &paged_mask_cache as &dyn PastKvLenCache,
            (Some(normal), _) => &normal.0 as &dyn PastKvLenCache,
            (None, Some(hybrid)) => &**hybrid as &dyn PastKvLenCache,
            (None, None) => unreachable!("Granite keeps a normal or a hybrid cache"),
        };
        // later chunks attend through the paged cache, so only the first needs a mask
        let first_chunk = ctx.is_first_prompt_chunk();
        let mask = if first_chunk {
            CausalMasker.make_causal_mask(
                input_ids,
                mask_cache,
                x.dtype(),
                &CausalMaskConfig::default(),
            )?
        } else {
            AttentionMask::None
        };
        let sliding_mask = match self.sliding_window {
            Some(window) if first_chunk => Some(CausalMasker.make_causal_mask(
                input_ids,
                mask_cache,
                x.dtype(),
                &CausalMaskConfig {
                    sliding_window: Some(window),
                    ..Default::default()
                },
            )?),
            Some(_) => Some(AttentionMask::None),
            None => None,
        };
        let mask = DeviceMappedMask::new(mask, &*self.mapper)?;
        let sliding_mask = sliding_mask
            .map(|mask| DeviceMappedMask::new(mask, &*self.mapper))
            .transpose()?;
        let stash_replays =
            crate::speculative::SpeculativeTargetMixin::has_speculative_proposer(self)
                && packed_query_lens.is_none()
                && recurrent_metadata.as_ref().is_some_and(|metadata| {
                    metadata.batch_kind() == RecurrentBatchKind::SpeculativeDecode
                })
                && x.dim(1)? > 1;
        let mut replays = Vec::new();
        let mut taps: Vec<Option<Tensor>> = vec![None; self.tap_points.len()];

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // a block's output is the next one's input (Mamba decode steps end their iteration early)
            if layer_idx > 0 {
                self.collect_taps(&mut taps, layer_idx - 1, &x)?;
            }
            x = self.mapper.map(x, layer_idx)?;
            x = before_layer(layer_idx, x)?;

            let layer_mask = match (&self.layer_types[layer_idx], &sliding_mask) {
                (GraniteLayerType::SlidingAttention, Some(sliding)) => sliding,
                _ => &mask,
            };
            let (block, input, slot) = match layer {
                DecoderLayer::Attention(block) => {
                    if let Some(kv_cache) = attention_cache(
                        layer_idx,
                        &mut normal_cache,
                        &mut pipeline_cache,
                        &mut internal_cache,
                    ) {
                        let mask_for_layer = &layer_mask.get(x.device());
                        x = block.forward(&x, mask_for_layer, kv_cache, ctx, layer_idx)?;
                    }
                    continue;
                }
                DecoderLayer::Mamba(block) => (block, block.input(&x)?, layer_idx),
                DecoderLayer::Parallel(parallel) => {
                    let mut input = parallel.mamba.input(&x)?;
                    let kv_cache = attention_cache(
                        layer_idx,
                        &mut normal_cache,
                        &mut pipeline_cache,
                        &mut internal_cache,
                    )
                    .ok_or_else(|| {
                        inference_tensor::Error::msg(format!(
                            "parallel block {layer_idx} has no attention cache"
                        ))
                    })?;
                    let mask_for_layer = &layer_mask.get(x.device());
                    input.side = Some(parallel.attn.forward(
                        &input.normed,
                        mask_for_layer,
                        kv_cache,
                        ctx,
                        layer_idx,
                    )?);
                    let slot = self.recurrent_slots[layer_idx].ok_or_else(|| {
                        inference_tensor::Error::msg(format!(
                            "parallel block {layer_idx} has no recurrent slot"
                        ))
                    })?;
                    (&parallel.mamba, input, slot)
                }
            };
            {
                {
                    if let (Some(metadata), Some(pipeline_cache)) =
                        (recurrent_metadata.as_ref(), pipeline_cache.as_mut())
                    {
                        let indices = pipeline_cache.state_indices_for_layer(slot)?.ok_or_else(
                            || {
                                inference_tensor::Error::msg(format!(
                                    "Hybrid cache layer {slot} is missing recurrent state indices"
                                ))
                            },
                        )?;
                        let captures = pipeline_cache.recurrent_captures_for_layer(slot)?;
                        let Some(HybridLayerCache::Recurrent(pool)) = pipeline_cache.get_mut(slot)
                        else {
                            inference_tensor::bail!(
                                "Hybrid cache layer {slot} is not recurrent for Granite"
                            );
                        };
                        // a CUDA or CPU decode step runs on the pool's rows in place, skipping the gather and scatter
                        if (x.device().is_cuda()
                            || ((x.device().is_cpu() || x.device().is_vulkan())
                                && x.dtype() == DType::F32))
                            && packed_query_lens.is_none()
                            && matches!(metadata.batch_kind(), RecurrentBatchKind::Decode)
                            && pool.conv_dtype() == x.dtype()
                            && pool.recurrent_dtype() == DType::F32
                        {
                            let mut pool_cache = MambaLayerCache {
                                conv_state: pool.conv_state.clone(),
                                ssm_state: pool.recurrent_state.clone(),
                            };
                            x = block.forward(
                                input,
                                &mut pool_cache,
                                RecurrentBatchKind::Decode,
                                Some(&indices),
                            )?;
                            continue;
                        }
                        let conv_state = pool.gather_conv_state(&indices)?;
                        let ssm_state = pool.gather_recurrent_state(&indices)?;

                        let mut temp_cache = MambaLayerCache {
                            conv_state: conv_state.clone(),
                            ssm_state: ssm_state.clone(),
                        };

                        x = if !captures.is_empty() {
                            if stash_replays
                                || matches!(metadata.batch_kind(), RecurrentBatchKind::Decode)
                            {
                                inference_tensor::bail!("Mamba captures run only in a prompt pass");
                            }
                            let (rows, packed) = match packed_query_lens.as_deref() {
                                Some(query_lens) => {
                                    let (batch, tokens, _) = x.dims3()?;
                                    (packed_mamba_query_ranges(batch, tokens, query_lens)?, true)
                                }
                                None => {
                                    let (batch, tokens, _) = x.dims3()?;
                                    (vec![0..tokens; batch], false)
                                }
                            };
                            let points =
                                captures.iter().map(|(point, _)| *point).collect::<Vec<_>>();
                            let (out, states) = block.forward_captured(
                                input,
                                &mut temp_cache,
                                &rows,
                                packed,
                                &points,
                            )?;
                            for ((_, slot), (conv, ssm)) in captures.iter().zip(states) {
                                pool.scatter_conv_state(slot, &conv)?;
                                pool.scatter_recurrent_state(slot, &ssm)?;
                            }
                            out
                        } else if let Some(query_lens) = packed_query_lens.as_deref() {
                            block.forward_packed_prefill(input, &mut temp_cache, query_lens)?
                        } else if stash_replays {
                            // the CUDA scan advances the state it is handed in place, so the replay's starting
                            // state is a copy taken before the pass
                            let (conv_state, ssm_state) = (conv_state.copy()?, ssm_state.copy()?);
                            let (out, inputs) = block.forward_replayable(input, &mut temp_cache)?;
                            replays.push(MambaReplay {
                                layer_idx,
                                slot,
                                indices: indices.clone(),
                                host_indices: metadata.state_indices_host().map(<[u32]>::to_vec),
                                conv_state,
                                ssm_state,
                                inputs,
                            });
                            out
                        } else {
                            block.forward(input, &mut temp_cache, metadata.batch_kind(), None)?
                        };

                        pool.scatter_conv_state_with_host_indices(
                            &indices,
                            metadata.state_indices_host(),
                            &temp_cache.conv_state,
                        )?;
                        pool.scatter_recurrent_state_with_host_indices(
                            &indices,
                            metadata.state_indices_host(),
                            &temp_cache.ssm_state,
                        )?;
                    } else {
                        if let GraniteLayerCache::Mamba(mamba_cache) =
                            &mut internal_cache.caches[slot]
                        {
                            if recurrent_batch_kind == RecurrentBatchKind::Prefill {
                                mamba_cache.reset()?;
                            }
                            x = block.forward(input, mamba_cache, recurrent_batch_kind, None)?;
                        }
                    }
                }
            }
        }

        *self.mamba_replay.lock().expect("mamba replay poisoned") = replays;
        if !self.tap_points.is_empty() {
            self.collect_taps(&mut taps, self.layers.len() - 1, &x)?;
        }

        let x = self.mapper.cast_nm_device(&x, false)?;
        let full = self.ln_f.forward(&x)?;
        let x = ctx.logits(&full)?;
        if let Some(mtp) = &self.mtp {
            mtp.record(&full, &x);
        }
        if !self.tap_points.is_empty() {
            let taps = taps
                .into_iter()
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| inference_tensor::Error::msg("a drafter tap was not recorded"))?;
            let selected = taps
                .iter()
                .map(|tap| ctx.logits(tap))
                .collect::<Result<Vec<_>>>()?;
            *self.prompt_taps.lock().expect("prompt taps poisoned") =
                (recurrent_batch_kind == RecurrentBatchKind::Prefill).then_some(taps);
            *self.spec_taps.lock().expect("spec taps poisoned") = Some(selected);
        }

        let mut logits = ctx.lm_head(&*self.lm_head, &x)?;

        // Scale logits
        logits = scale_tensor(logits, self.logits_scaling)?;

        Ok(logits)
    }

    pub fn residual_tensors_m(&self, uvb_m: UnVarBuilder) -> Vec<(String, Tensor)> {
        uvb_m.pp(self.layout.embeddings).add(&self.wte);
        uvb_m.pp(self.layout.final_norm).add(&self.ln_f);

        for (layer, block) in self.layers.iter().zip(&self.layout.blocks) {
            let names = &block.names;
            let (rms_1, ffn) = match layer {
                DecoderLayer::Attention(block) => (&block.rms_1, &block.ffn),
                DecoderLayer::Mamba(block)
                | DecoderLayer::Parallel(ParallelBlock { mamba: block, .. }) => {
                    uvb_m
                        .pp(names.mamba.as_ref().unwrap_or(&names.mixer))
                        .extend(block.mamba.residual_tensors());
                    (&block.rms_1, &block.ffn)
                }
            };
            uvb_m.pp(&names.norm).add(rms_1);
            if let (Some(ffn), Some(ffn_names)) = (ffn, &names.ffn) {
                uvb_m.pp(&ffn_names.norm).add(&ffn.norm);
                if let Some(moe) = &ffn.moe {
                    uvb_m.pp(&ffn_names.module).extend(moe.residual_tensors());
                }
            }
        }

        uvb_m.to_safetensors()
    }
}

impl IsqModel for GraniteMoeHybrid {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        if let Some(mtp) = &self.mtp {
            mtp.add_residual(&uvb.pp(MTP_PREFIX));
        }
        self.residual_tensors_m(uvb.pp("model"))
    }
}

impl GraniteMoeHybrid {
    // the pool rows of a verify step that kept fewer rows than it ran, re-advanced from the stashed state
    fn replay_mamba_layers(&self, rows: &[crate::speculative::SpeculativeCommitRow]) -> Result<()> {
        let replays =
            std::mem::take(&mut *self.mamba_replay.lock().expect("mamba replay poisoned"));
        let mut groups: std::collections::BTreeMap<usize, Vec<u32>> = Default::default();
        for row in rows.iter().filter(|row| !row.accepted_all) {
            #[allow(clippy::cast_possible_truncation)]
            groups
                .entry(row.keep_rows)
                .or_default()
                .push(row.batch_idx as u32);
        }
        if replays.is_empty() || groups.is_empty() {
            return Ok(());
        }
        let mut cache = self.kv_cache.hybrid();
        // each group's rows on the device once, not once a layer and tensor; `None` where a group is the whole batch
        let batch = replays[0].indices.dim(0)?;
        let device = replays[0].indices.device().clone();
        let group_rows = groups
            .values()
            .map(|rows| {
                #[allow(clippy::cast_possible_truncation)]
                let whole =
                    rows.len() == batch && rows.iter().enumerate().all(|(i, &r)| r == i as u32);
                (!whole)
                    .then(|| Tensor::new(rows.as_slice(), &device))
                    .transpose()
            })
            .collect::<Result<Vec<_>>>()?;
        for replay in &replays {
            let Some(block) = self.layers[replay.layer_idx].mamba() else {
                inference_tensor::bail!("Mamba replay of non-Mamba layer {}", replay.layer_idx);
            };
            for ((&keep, batch_rows), rows_on_device) in groups.iter().zip(&group_rows) {
                // the stashed conv input and dt are views into the input projection
                let pick = |t: &Tensor| -> Result<Tensor> {
                    match rows_on_device {
                        Some(rows) => t.contiguous()?.index_select(rows, 0),
                        None => Ok(t.clone()),
                    }
                };
                let mut state = MambaLayerCache {
                    conv_state: pick(&replay.conv_state)?,
                    ssm_state: pick(&replay.ssm_state)?,
                };
                let inputs = ReplayInputs {
                    hidden_states_b_c: pick(&replay.inputs.hidden_states_b_c)?,
                    dt: pick(&replay.inputs.dt)?,
                };
                block.replay(&inputs, &mut state, keep)?;
                let slots = pick(&replay.indices)?;
                let host_slots: Option<Vec<u32>> = replay
                    .host_indices
                    .as_ref()
                    .map(|host| batch_rows.iter().map(|&row| host[row as usize]).collect());
                let Some(HybridLayerCache::Recurrent(pool)) = cache.get_mut(replay.slot) else {
                    inference_tensor::bail!("Hybrid cache layer {} is not recurrent", replay.slot);
                };
                let host_slots = host_slots.as_deref();
                pool.scatter_conv_state_with_host_indices(&slots, host_slots, &state.conv_state)?;
                pool.scatter_recurrent_state_with_host_indices(
                    &slots,
                    host_slots,
                    &state.ssm_state,
                )?;
            }
        }
        Ok(())
    }
}

impl GraniteMoeHybrid {
    fn mtp_drafting(&self) -> Option<&MtpHead<GraniteMtpBlock>> {
        self.mtp.as_ref().filter(|mtp| mtp.n_predict() > 0)
    }

    fn drafter(&self) -> Option<Arc<ExternalDrafter>> {
        self.drafter.lock().expect("drafter poisoned").clone()
    }

    // block `block`'s taps, once it has run: its mixer's output, and `output`, the block's own
    fn collect_taps(
        &self,
        taps: &mut [Option<Tensor>],
        block: usize,
        output: &Tensor,
    ) -> Result<()> {
        for (slot, point) in taps.iter_mut().zip(&self.tap_points) {
            let tap = match *point {
                TapPoint::AfterMixer(b) if b == block => self.layers[b].mixer_tap().take(),
                TapPoint::AfterBlock(b) if b == block => Some(output.clone()),
                _ => continue,
            };
            *slot = tap
                .map(|tap| self.mapper.cast_nm_device(&tap, false))
                .transpose()?;
        }
        Ok(())
    }

    fn attach_external_drafter(
        &mut self,
        config: &crate::speculative::MtpConfig,
        runtime: crate::speculative::MtpRuntimeConfig,
    ) -> Result<crate::speculative::SpeculativeAttachInfo> {
        let (recurrent_capacity, serving_capacity) = {
            let cache = self.kv_cache.hybrid();
            let recurrent = cache.recurrent_capacity();
            let serving = cache.fitted_serving_capacity().unwrap_or_else(|| {
                recurrent.saturating_sub(crate::kv_cache::RECURRENT_GRAPH_PAD_SLOTS)
            });
            (recurrent, serving)
        };
        let checkpoint_lanes = self.kv_cache.hybrid().checkpoint_lanes();
        let checkpoint_layers = self.layout.tap_points(&[])?.1;
        let (drafter, info) = ExternalDrafter::attach(
            config,
            DrafterAttach {
                load: DFlashLoadTarget {
                    num_layers: checkpoint_layers,
                    hidden_size: self.cfg.hidden_size,
                    // Nemotron-H's attention has no positions; the drafter brings its own YaRN
                    yarn_rope_config: None,
                    device: &self.device,
                    dtype: self.dtype,
                },
                checkpoint_lanes,
                recurrent_capacity,
                serving_capacity,
                // after the target's paged layers and its MTP head's
                paged_first_layer: self.paged_kv_config().num_layers(),
                runtime,
                recurrent_pad_slots: crate::kv_cache::RECURRENT_GRAPH_PAD_SLOTS,
            },
        )?;
        self.tap_points = self.layout.tap_points(&drafter.target_layer_ids)?.0;
        for point in &self.tap_points {
            if let TapPoint::AfterMixer(block) = *point {
                self.layers[block].mixer_tap_mut().enabled = true;
            }
        }
        self.draft_lm_head = config
            .draft_lm_head_isq
            .map(|ty| {
                self.lm_head.clone().apply_isq(
                    Some(ty),
                    self.device.clone(),
                    &std::sync::atomic::AtomicUsize::new(0),
                    None,
                    inference_quant::QuantizeOntoGuard::new(),
                )
            })
            .transpose()?;
        *self.drafter.lock().expect("drafter poisoned") = Some(Arc::new(drafter));
        Ok(info)
    }

    fn detach_external_drafter(&mut self) {
        *self.drafter.lock().expect("drafter poisoned") = None;
        self.tap_points.clear();
        for layer in &mut self.layers {
            *layer.mixer_tap_mut() = MixerTap::default();
        }
        self.draft_lm_head = None;
        *self.spec_taps.lock().expect("spec taps poisoned") = None;
        *self.prompt_taps.lock().expect("prompt taps poisoned") = None;
    }
}

impl DrafterTarget for GraniteMoeHybrid {
    fn spec_taps(&self) -> Option<Vec<Tensor>> {
        self.spec_taps.lock().expect("spec taps poisoned").clone()
    }

    fn prompt_taps(&self) -> Option<Vec<Tensor>> {
        self.prompt_taps
            .lock()
            .expect("prompt taps poisoned")
            .clone()
    }

    fn embed_tokens(&self, ids: &Tensor) -> Result<Tensor> {
        self.embed(ids)
    }

    fn token_embedding(&self) -> &Arc<dyn QuantMethod> {
        &self.wte
    }

    fn draft_lm_head(&self) -> Arc<dyn QuantMethod> {
        self.draft_lm_head
            .clone()
            .unwrap_or_else(|| self.lm_head.clone())
    }
}

impl crate::speculative::SpeculativeTargetMixin for GraniteMoeHybrid {
    fn attach_speculative(
        &mut self,
        config: crate::speculative::SpeculativeConfig,
    ) -> Result<Option<crate::speculative::SpeculativeAttachInfo>> {
        self.attach_speculative_with_runtime(config, Default::default())
    }

    fn mtp_draft_head_bytes(&self, isq: inference_quant::IsqType) -> Option<usize> {
        self.mtp.as_ref()?;
        let dtype = inference_tensor::quantized::GgmlDType::try_from(isq).ok()?;
        Some(self.lm_head_elements / dtype.block_size() * dtype.type_size())
    }

    fn attach_speculative_with_runtime(
        &mut self,
        config: crate::speculative::SpeculativeConfig,
        runtime: crate::speculative::MtpRuntimeConfig,
    ) -> Result<Option<crate::speculative::SpeculativeAttachInfo>> {
        self.detach_external_drafter();
        if let Some(mtp) = &mut self.mtp {
            mtp.set_n_predict(0);
        }
        let crate::speculative::SpeculativeConfig::Mtp(config) = config else {
            return Ok(None);
        };
        if !config.is_builtin() {
            return self.attach_external_drafter(&config, runtime).map(Some);
        }
        let Some(mtp) = &mut self.mtp else {
            inference_tensor::bail!(
                "`--mtp` asked for this model's MTP head, which its checkpoint does not carry"
            );
        };
        let offloaded = self.layers.iter().any(|layer| {
            matches!(
                layer.ffn().and_then(|ffn| ffn.moe.as_ref()),
                Some(HybridMoe::Nemotron(moe)) if moe.experts.is_offloaded()
            )
        });
        let n_predict = config.n_predict.unwrap_or(if offloaded {
            crate::decoder::DEFAULT_MTP_N_PREDICT_OFFLOADED
        } else {
            crate::decoder::DEFAULT_MTP_N_PREDICT
        });
        if n_predict == 0 {
            inference_tensor::bail!("MTP needs at least one draft token");
        }
        mtp.set_draft_lm_head(&self.lm_head, config.drafter_isq)?;
        mtp.set_n_predict(n_predict);
        Ok(Some(crate::speculative::SpeculativeAttachInfo::mtp(
            "built-in".to_string(),
            n_predict,
        )))
    }

    fn has_speculative_proposer(&self) -> bool {
        self.mtp_drafting().is_some() || self.drafter().is_some()
    }

    fn supports_speculative_prompt_bootstrap(&self) -> bool {
        self.drafter().is_some()
    }

    fn speculative_plan(
        &self,
        batch_size: usize,
    ) -> Option<crate::speculative::SpeculativeBatchPlan> {
        if let Some(drafter) = self.drafter() {
            return Some(drafter.plan(drafter.n_predict(), batch_size));
        }
        self.mtp_drafting()
            .map(|mtp| crate::speculative::SpeculativeBatchPlan::new(mtp.n_predict()))
    }

    fn speculative_graph_plans(&self) -> Vec<crate::speculative::SpeculativeGraphPlan> {
        if let Some(drafter) = self.drafter() {
            return drafter.graph_plans(drafter.n_predict());
        }
        self.mtp_drafting()
            .map(|mtp| crate::speculative::SpeculativeGraphPlan::new(mtp.n_predict(), None))
            .into_iter()
            .collect()
    }

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    fn precapture_speculative_cuda_graphs(&self) -> Result<()> {
        match self.drafter() {
            Some(drafter) => drafter.precapture(self, drafter.n_predict()),
            None => Ok(()),
        }
    }

    fn evict_speculative_cuda_graphs(&self, max_entries: usize) -> usize {
        self.drafter()
            .map_or(0, |drafter| drafter.evict_cuda_graphs_lru(max_entries))
    }

    fn speculative_bypass(&mut self, seq_ids: &[usize]) -> Result<()> {
        if let Some(drafter) = self.drafter() {
            drafter.mark_seqs_dormant(seq_ids);
        }
        Ok(())
    }

    fn release_speculative_sequences(&mut self, seq_ids: &[usize]) -> Result<()> {
        if let Some(drafter) = self.drafter() {
            drafter.release_seqs(seq_ids);
        }
        Ok(())
    }

    inference_drafters::external_drafter_prefix_hooks!(|this| this.drafter());

    fn speculative_target_hiddens(&self, rows: &[(usize, usize)]) -> Result<Option<Tensor>> {
        match self.mtp_drafting() {
            Some(mtp) => mtp.target_hiddens(rows),
            None => Ok(None),
        }
    }

    fn speculative_prefill(
        &mut self,
        ctx: crate::speculative::SpeculativePrefillCtx<'_>,
    ) -> Result<()> {
        if let Some(drafter) = self.drafter() {
            return drafter.prefill(self, ctx);
        }
        match self.mtp_drafting() {
            Some(mtp) => mtp.prefill(&|tokens| self.embed(tokens), ctx),
            None => Ok(()),
        }
    }

    fn speculative_prepare_propose(
        &mut self,
        ctx: crate::speculative::SpeculativeProposePrepareCtx<'_>,
    ) -> Result<Option<Box<dyn crate::speculative::SpeculativeProposePreparation>>> {
        match self.drafter() {
            Some(drafter) => drafter.prepare_propose(self, ctx),
            None => Ok(None),
        }
    }

    fn speculative_propose(
        &mut self,
        ctx: crate::speculative::SpeculativeProposeBatchCtx<'_>,
    ) -> Result<Option<crate::speculative::SpeculativeProposalBatch>> {
        if let Some(drafter) = self.drafter() {
            return drafter.propose(self, ctx, drafter.n_predict(), &self.device);
        }
        match self.mtp_drafting() {
            Some(mtp) => mtp.propose(&|tokens| self.embed(tokens), &*self.lm_head, ctx),
            None => Ok(None),
        }
    }

    fn speculative_commit(
        &mut self,
        rows: &[crate::speculative::SpeculativeCommitRow],
    ) -> Result<()> {
        self.replay_mamba_layers(rows)?;
        match self.drafter() {
            Some(drafter) => drafter.commit_terminal(self, rows),
            None => Ok(()),
        }
    }

    fn take_speculative_graph_state(
        &self,
    ) -> Option<Box<dyn crate::speculative::SpeculativeGraphState>> {
        let (mtp, taps) = match (self.mtp_drafting(), self.drafter()) {
            (_, Some(_)) => (None, self.spec_taps()?),
            (Some(mtp), None) => (Some(Arc::from(mtp.take_graph_state()?)), Vec::new()),
            (None, None) => return None,
        };
        let replays = self
            .mamba_replay
            .lock()
            .expect("mamba replay poisoned")
            .clone();
        Some(Box::new(GraniteSpecGraphState { mtp, taps, replays }))
    }

    fn install_speculative_graph_state(
        &self,
        state: &dyn crate::speculative::SpeculativeGraphState,
    ) -> Result<()> {
        if !self.has_speculative_proposer() {
            return Ok(());
        }
        let state = state
            .as_any()
            .downcast_ref::<GraniteSpecGraphState>()
            .ok_or_else(|| {
                inference_tensor::Error::msg("foreign speculative graph state for Granite")
            })?;
        if let (Some(mtp), Some(captured)) = (self.mtp_drafting(), &state.mtp) {
            mtp.install(&**captured)?;
        }
        if self.drafter().is_some() {
            *self.spec_taps.lock().expect("spec taps poisoned") = Some(state.taps.clone());
        }
        let mut replays = state.replays.clone();
        // a capture baked in its own step's host slots; graphs run on CUDA, whose scatter reads the device slots
        for replay in &mut replays {
            replay.host_indices = None;
        }
        *self.mamba_replay.lock().expect("mamba replay poisoned") = replays;
        Ok(())
    }
}

impl NormalModel for GraniteMoeHybrid {
    fn forward(&self, input_ids: &Tensor, ctx: &mut ModelForwardContext<'_>) -> Result<Tensor> {
        self.forward(input_ids, ctx)
    }
    fn cache(&self) -> &crate::kv_cache::EitherCache {
        &self.kv_cache
    }
    fn device(&self) -> &Device {
        &self.device
    }
    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    fn config(&self) -> &ModelConfigMetadata {
        &self.cfg
    }
    fn model_config(&self) -> Arc<dyn ModelConfigLike + Send + Sync> {
        self.paged_kv_config()
    }
    // a hybrid model's decode reads and writes its recurrent pool through device slot indices, so a step captures
    // unless its experts route on the host (Granite's own, or dense ungated stacks)
    #[cfg(feature = "cuda")]
    fn supports_cuda_decode_graphs(&self) -> bool {
        self.layer_types
            .iter()
            .any(GraniteLayerType::has_recurrent_state)
            && self
                .layers
                .iter()
                .all(|layer| match layer.ffn().and_then(|ffn| ffn.moe.as_ref()) {
                    None => true,
                    Some(HybridMoe::Granite(_)) => false,
                    Some(HybridMoe::Nemotron(moe)) => moe.experts.decode_is_device_only(),
                })
    }
    // offloaded experts route a larger batch on the host; the MTP head's count as the trunk's
    #[cfg(feature = "cuda")]
    fn cuda_decode_graph_max_tokens(&self) -> Option<usize> {
        let offloaded = |ffn: Option<&BlockFfn>| match ffn.and_then(|ffn| ffn.moe.as_ref()) {
            Some(HybridMoe::Nemotron(moe)) => moe.experts.is_offloaded(),
            _ => false,
        };
        (self.layers.iter().any(|layer| offloaded(layer.ffn()))
            || self
                .mtp
                .as_ref()
                .is_some_and(|head| offloaded(head.block().block.ffn.as_ref())))
        .then_some(crate::moe::OFFLOADED_GRAPH_MAX_TOKENS)
    }
    fn captures_recurrent_state_mid_chunk(&self) -> bool {
        self.layer_types
            .iter()
            .any(GraniteLayerType::has_recurrent_state)
    }
    fn supports_packed_prefill(&self) -> bool {
        !self.has_sinks
            || crate::attention::sinks_backend_supports(
                self.dtype,
                self.device.location(),
                self.cfg.k_head_dim,
            )
    }
}

impl AnyMoeBaseModelMixin for GraniteMoeHybrid {
    fn get_mlps(&self) -> Vec<&dyn MlpLayer> {
        let mut mlps = Vec::new();
        for layer in &self.layers {
            if let Some(mlp) = layer.ffn().and_then(|ffn| ffn.mlp.as_deref()) {
                mlps.push(mlp);
            }
        }
        mlps
    }
    fn get_mlps_mut(&mut self) -> Vec<&mut Box<dyn MlpLayer>> {
        let mut mlps = Vec::new();
        for layer in &mut self.layers {
            if let Some(mlp) = layer.ffn_mut().as_mut().and_then(|ffn| ffn.mlp.as_mut()) {
                mlps.push(mlp);
            }
        }
        mlps
    }
    fn create_anymoe_layers(
        &mut self,
        additional_vbs: Vec<ShardedVarBuilder>,
        config: AnyMoeConfig,
        (prefix, _mlp): (String, String),
        mut layers: Vec<usize>,
        expert_type: AnyMoeExpertType,
        gate_vb: Option<ShardedVarBuilder>,
    ) -> Result<()> {
        let mut experts: Vec<Vec<Box<dyn MlpLayer>>> = Vec::new();
        if layers.is_empty() {
            layers = (0..self.layers.len()).collect::<Vec<_>>();
        }
        for _ in 0..layers.len() {
            experts.push(Vec::new());
        }

        // Helper to get MLP from a layer
        fn get_mlp(layer: &DecoderLayer) -> Result<&dyn MlpLayer> {
            layer
                .ffn()
                .and_then(|ffn| ffn.mlp.as_deref())
                .ok_or_else(|| inference_tensor::Error::msg("Granite layer has no shared MLP"))
        }

        for vb in additional_vbs {
            let vb = vb.pp(&prefix);
            for (layer_idx, row) in experts.iter_mut().enumerate() {
                if !layers.contains(&layer_idx) {
                    continue;
                }

                match expert_type {
                    AnyMoeExpertType::FineTuned => {
                        let layer_mlp = get_mlp(&self.layers[layer_idx])?;
                        let (dtype, device) = layer_mlp.dtype_device();
                        // For GraniteMlp, we need custom handling
                        let cfg_for_layer = Config {
                            hidden_size: layer_mlp.get_params()[0],
                            shared_intermediate_size: Some(layer_mlp.get_params()[1]),
                            intermediate_size: layer_mlp.get_params()[1],
                            vocab_size: 0,
                            num_hidden_layers: 0,
                            num_attention_heads: 0,
                            num_key_value_heads: None,
                            rms_norm_eps: 0.0,
                            rope_theta: 0.0,
                            max_position_embeddings: 0,
                            rope_scaling: None,
                            quantization_config: None,
                            tie_word_embeddings: false,
                            layer_types: vec![],
                            attention_multiplier: 1.0,
                            embedding_multiplier: 1.0,
                            residual_multiplier: 1.0,
                            logits_scaling: 1.0,
                            // Mamba fields (not used for MLP but needed for struct)
                            mamba_n_heads: None,
                            mamba_n_groups: 1,
                            mamba_d_state: 256,
                            mamba_d_head: None,
                            mamba_d_conv: 4,
                            mamba_expand: 2,
                            mamba_chunk_size: 256,
                            mamba_conv_bias: true,
                            mamba_proj_bias: false,
                            // MoE fields (not used for MLP but needed for struct)
                            num_local_experts: 0,
                            num_experts_per_tok: 2,
                            // Position embedding type (not used for MLP)
                            position_embedding_type: "rope".to_string(),
                            sliding_window: None,
                            layer_rope_theta: None,
                            no_rope_layers: None,
                            rope_parameters: None,
                            head_dim: None,
                            mamba_inner_size: None,
                            mamba_norm_group_size: None,
                            mamba_dt_limit: None,
                            mamba_f32_state: false,
                            mamba_gated_norm: true,
                            mup: None,
                            mamba1: None,
                        };
                        row.push(Box::new(GraniteMlp::new(
                            vb.pp(layer_idx)
                                .pp("mlp")
                                .set_dtype(dtype)
                                .set_device(device),
                            &cfg_for_layer,
                            &self.mapper.get_comm_for(layer_idx)?,
                        )?));
                    }
                    AnyMoeExpertType::LoraAdapter { .. } => {
                        inference_tensor::bail!(
                            "LoRA adapters not supported for GraniteMoeHybrid MLP"
                        )
                    }
                }
            }
        }
        for (layer_idx, expert) in layers.into_iter().zip(experts) {
            let mlp_box = self.layers[layer_idx]
                .ffn_mut()
                .as_mut()
                .and_then(|ffn| ffn.mlp.as_mut())
                .ok_or_else(|| inference_tensor::Error::msg("Granite layer has no shared MLP"))?;
            let mut experts_all = vec![mlp_box.clone()];
            experts_all.extend(expert);
            let (dtype, device) = mlp_box.dtype_device();
            *mlp_box = Box::new(MoeMlp::new(
                experts_all,
                config.clone(),
                dtype,
                &device,
                layer_idx,
                gate_vb.as_ref(),
            )?);
        }
        Ok(())
    }
    fn amoe_supported(&self) -> bool {
        self.layers
            .iter()
            .all(|layer| layer.ffn().is_some_and(|ffn| ffn.mlp.is_some()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HIDDEN_SIZE: usize = 4;
    const INTERMEDIATE_SIZE: usize = 4;
    const NUM_HEADS: usize = 2;
    const HEAD_DIM: usize = 2;
    const STATE_SIZE: usize = 2;
    const NUM_GROUPS: usize = 1;
    const CONV_WIDTH: usize = 3;
    const CONV_DIM: usize = INTERMEDIATE_SIZE + 2 * NUM_GROUPS * STATE_SIZE;
    const PROJECTION_SIZE: usize = INTERMEDIATE_SIZE + CONV_DIM + NUM_HEADS;
    const ASSERT_EPS: f32 = 1e-4;

    fn patterned(len: usize, salt: usize, scale: f32, offset: f32) -> Vec<f32> {
        (0..len)
            .map(|index| {
                let value = ((index.wrapping_mul(37) + salt.wrapping_mul(17)) % 257) as f32;
                ((value / 128.0) - 1.0) * scale + offset
            })
            .collect()
    }

    fn unquant_linear(weight: Tensor, bias: Option<Tensor>) -> Result<Arc<dyn QuantMethod>> {
        Ok(Arc::new(inference_quant::UnquantLinear::new(
            inference_quant::QuantMethodConfig::Unquantized(inference_tensor::nn::Linear::new(
                weight, bias,
            )),
        )?))
    }

    fn mamba_layer(device: &Device) -> Result<MambaLayer> {
        let in_proj_weight = Tensor::from_vec(
            patterned(PROJECTION_SIZE * HIDDEN_SIZE, 1, 0.08, 0.01),
            (PROJECTION_SIZE, HIDDEN_SIZE),
            device,
        )?;
        let in_proj_bias = Tensor::from_vec(
            patterned(PROJECTION_SIZE, 2, 0.02, 0.0),
            (PROJECTION_SIZE,),
            device,
        )?;
        let out_proj_weight = Tensor::from_vec(
            patterned(HIDDEN_SIZE * INTERMEDIATE_SIZE, 3, 0.1, -0.01),
            (HIDDEN_SIZE, INTERMEDIATE_SIZE),
            device,
        )?;
        let out_proj_bias =
            Tensor::from_vec(patterned(HIDDEN_SIZE, 4, 0.02, 0.0), (HIDDEN_SIZE,), device)?;
        let dt_bias = Tensor::from_vec(patterned(NUM_HEADS, 7, 0.1, 0.2), (NUM_HEADS,), device)?;
        let a_log = Tensor::from_vec(patterned(NUM_HEADS, 8, 0.1, -0.5), (NUM_HEADS,), device)?;
        let d = Tensor::from_vec(patterned(NUM_HEADS, 9, 0.1, 0.3), (NUM_HEADS,), device)?;
        let norm_weight = Tensor::from_vec(
            patterned(INTERMEDIATE_SIZE, 10, 0.1, 1.0),
            (INTERMEDIATE_SIZE,),
            device,
        )?;
        Ok(MambaLayer {
            in_proj: unquant_linear(in_proj_weight, Some(in_proj_bias))?,
            selection: None,
            conv1d_weight: Tensor::from_vec(
                patterned(CONV_DIM * CONV_WIDTH, 5, 0.08, 0.01),
                (CONV_DIM, 1, CONV_WIDTH),
                device,
            )?,
            conv1d_bias: Some(Tensor::from_vec(
                patterned(CONV_DIM, 6, 0.03, 0.0),
                (CONV_DIM,),
                device,
            )?),
            scan: ScanConstants::new(&a_log, &d, &dt_bias)?,
            dt_bias,
            a_log,
            d,
            norm: Some(RmsNormGated {
                weight: norm_weight.clone(),
                #[cfg(feature = "cuda")]
                weight_f32: norm_weight.to_dtype(DType::F32)?,
                eps: 1e-5,
                group_size: INTERMEDIATE_SIZE,
            }),
            out_proj: unquant_linear(out_proj_weight, Some(out_proj_bias))?,
            in_scale: None,
            out_scale: 1.0,
            num_heads: NUM_HEADS,
            head_dim: HEAD_DIM,
            intermediate_size: INTERMEDIATE_SIZE,
            ssm_state_size: STATE_SIZE,
            conv_kernel_size: CONV_WIDTH,
            n_groups: NUM_GROUPS,
            time_step_min: NO_DT_LIMIT[0],
            time_step_max: NO_DT_LIMIT[1],
            identity_slots: Mutex::new(None),
        })
    }

    fn mamba_cache(device: &Device) -> Result<MambaLayerCache> {
        Ok(MambaLayerCache {
            conv_state: Tensor::from_vec(
                patterned(2 * CONV_DIM * CONV_WIDTH, 11, 0.03, 0.0),
                (2, CONV_DIM, CONV_WIDTH),
                device,
            )?,
            ssm_state: Tensor::from_vec(
                patterned(2 * NUM_HEADS * HEAD_DIM * STATE_SIZE, 12, 0.03, 0.0),
                (2, NUM_HEADS, HEAD_DIM, STATE_SIZE),
                device,
            )?,
        })
    }

    #[test]
    fn mamba_residual_contains_every_checkpoint_tensor() -> Result<()> {
        let names = mamba_layer(&Device::Cpu)?
            .residual_tensors()
            .into_iter()
            .map(|(name, _)| name)
            .collect::<std::collections::HashSet<_>>();

        for name in [
            "in_proj.weight",
            "in_proj.bias",
            "conv1d.weight",
            "conv1d.bias",
            "dt_bias",
            "A_log",
            "D",
            "norm.weight",
            "out_proj.weight",
            "out_proj.bias",
        ] {
            assert!(names.contains(name), "missing Mamba residual tensor {name}");
        }
        Ok(())
    }

    #[test]
    fn dense_granite_expert_residual_reconstructs_stacked_weight() -> Result<()> {
        let experts = GraniteParallelExperts {
            weights: GraniteParallelExpertWeights::Dense(vec![
                Tensor::zeros((4, 3), DType::F32, &Device::Cpu)?,
                Tensor::ones((4, 3), DType::F32, &Device::Cpu)?,
            ]),
            output_size: 4,
            name: EXPERT_WEIGHT,
        };
        let residual = experts.residual_tensors();

        assert_eq!(residual.len(), 1);
        assert_eq!(residual[0].0, "weight");
        assert_eq!(residual[0].1.dims(), &[2, 4, 3]);
        Ok(())
    }

    fn assert_close(lhs: &Tensor, rhs: &Tensor) -> Result<()> {
        let lhs = lhs.flatten_all()?.to_vec1::<f32>()?;
        let rhs = rhs.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(lhs.len(), rhs.len());
        for (index, (&lhs, &rhs)) in lhs.iter().zip(rhs.iter()).enumerate() {
            let diff = (lhs - rhs).abs();
            assert!(
                diff <= ASSERT_EPS,
                "index={index} lhs={lhs} rhs={rhs} diff={diff}"
            );
        }
        Ok(())
    }

    fn packed_shape() -> PackedMambaShape {
        PackedMambaShape {
            physical_batch: 1,
            physical_tokens: 7,
            conv_state_batch: 3,
            conv_dim: CONV_DIM,
            conv_width: CONV_WIDTH,
            ssm_state_batch: 3,
            ssm_heads: NUM_HEADS,
            ssm_head_dim: HEAD_DIM,
            ssm_state_width: STATE_SIZE,
            expected_conv_dim: CONV_DIM,
            expected_conv_width: CONV_WIDTH,
            expected_ssm_heads: NUM_HEADS,
            expected_ssm_head_dim: HEAD_DIM,
            expected_ssm_state_width: STATE_SIZE,
        }
    }

    #[test]
    fn packed_mamba_matches_independent_unequal_prefills() -> Result<()> {
        let device = Device::Cpu;
        let layer = mamba_layer(&device)?;
        let x = Tensor::from_vec(
            patterned(5 * HIDDEN_SIZE, 13, 0.2, 0.01),
            (1, 5, HIDDEN_SIZE),
            &device,
        )?;
        let initial_cache = mamba_cache(&device)?;
        let mut packed_cache = initial_cache.clone();
        let packed = layer.forward_packed_prefill(&x, &mut packed_cache, &[2, 3])?;

        let mut reference_outputs = Vec::new();
        let mut reference_conv_states = Vec::new();
        let mut reference_ssm_states = Vec::new();
        let mut offset = 0;
        for (state_index, segment_len) in [2, 3].into_iter().enumerate() {
            let mut segment_cache = MambaLayerCache {
                conv_state: initial_cache.conv_state.narrow(0, state_index, 1)?,
                ssm_state: initial_cache.ssm_state.narrow(0, state_index, 1)?,
            };
            reference_outputs.push(layer.forward(
                &x.narrow(1, offset, segment_len)?,
                &mut segment_cache,
                RecurrentBatchKind::Prefill,
                None,
            )?);
            reference_conv_states.push(segment_cache.conv_state);
            reference_ssm_states.push(segment_cache.ssm_state);
            offset += segment_len;
        }
        let reference = Tensor::cat(&reference_outputs, 1)?;
        let reference_conv_state = Tensor::cat(&reference_conv_states, 0)?;
        let reference_ssm_state = Tensor::cat(&reference_ssm_states, 0)?;

        assert_close(&packed, &reference)?;
        assert_close(&packed_cache.conv_state, &reference_conv_state)?;
        assert_close(&packed_cache.ssm_state, &reference_ssm_state)
    }

    #[test]
    fn packed_expert_forward_matches_dense_grouping() -> Result<()> {
        const EXPERTS: usize = 3;
        const TOP_K: usize = 2;
        const EXPERT_INTERMEDIATE: usize = 3;

        let device = Device::Cpu;
        let input_weights = Tensor::from_vec(
            patterned(
                EXPERTS * EXPERT_INTERMEDIATE * 2 * HIDDEN_SIZE,
                20,
                0.08,
                0.01,
            ),
            (EXPERTS, EXPERT_INTERMEDIATE * 2, HIDDEN_SIZE),
            &device,
        )?;
        let output_weights = Tensor::from_vec(
            patterned(EXPERTS * HIDDEN_SIZE * EXPERT_INTERMEDIATE, 21, 0.09, -0.01),
            (EXPERTS, HIDDEN_SIZE, EXPERT_INTERMEDIATE),
            &device,
        )?;
        let router = unquant_linear(
            Tensor::from_vec(
                patterned(EXPERTS * HIDDEN_SIZE, 22, 0.1, 0.0),
                (EXPERTS, HIDDEN_SIZE),
                &device,
            )?,
            None,
        )?;
        let dense = GraniteMoE {
            input_linear: GraniteParallelExperts {
                weights: GraniteParallelExpertWeights::Dense(
                    (0..EXPERTS)
                        .map(|expert| input_weights.i(expert))
                        .collect::<Result<Vec<_>>>()?,
                ),
                output_size: EXPERT_INTERMEDIATE * 2,
                name: EXPERT_WEIGHT,
            },
            output_linear: GraniteParallelExperts {
                weights: GraniteParallelExpertWeights::Dense(
                    (0..EXPERTS)
                        .map(|expert| output_weights.i(expert))
                        .collect::<Result<Vec<_>>>()?,
                ),
                output_size: HIDDEN_SIZE,
                name: EXPERT_WEIGHT,
            },
            router: GraniteTopKGating {
                layer: router.clone(),
                num_experts: EXPERTS,
                top_k: TOP_K,
            },
            input_size: HIDDEN_SIZE,
            v5_names: false,
        };
        let packed_input = unquant_linear(input_weights, None)?;
        let packed_output = unquant_linear(output_weights, None)?;
        let packed = GraniteMoE {
            input_linear: GraniteParallelExperts {
                weights: GraniteParallelExpertWeights::Quantized(packed_input.clone()),
                output_size: EXPERT_INTERMEDIATE * 2,
                name: EXPERT_WEIGHT,
            },
            output_linear: GraniteParallelExperts {
                weights: GraniteParallelExpertWeights::Quantized(packed_output.clone()),
                output_size: HIDDEN_SIZE,
                name: EXPERT_WEIGHT,
            },
            router: GraniteTopKGating {
                layer: router,
                num_experts: EXPERTS,
                top_k: TOP_K,
            },
            input_size: HIDDEN_SIZE,
            v5_names: false,
        };
        let input = Tensor::from_vec(
            patterned(2 * 3 * HIDDEN_SIZE, 23, 0.2, 0.01),
            (2, 3, HIDDEN_SIZE),
            &device,
        )?;
        packed_input.begin_track_stats()?;
        packed_output.begin_track_stats()?;
        let packed_output_tensor = packed.forward(&input)?;
        assert_eq!(packed_input.stats_snapshot(), Some((1, 12)));
        assert_eq!(packed_output.stats_snapshot(), Some((1, 12)));
        assert_close(&packed_output_tensor, &dense.forward(&input)?)
    }

    #[test]
    fn packed_expert_forward_records_routed_stats() -> Result<()> {
        let device = Device::Cpu;
        let weight = unquant_linear(Tensor::ones((2, 2, 3), DType::F32, &device)?, None)?;
        let experts = GraniteParallelExperts {
            weights: GraniteParallelExpertWeights::Quantized(weight.clone()),
            output_size: 2,
            name: EXPERT_WEIGHT,
        };
        let input = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.], [7., 8., 9.]], &device)?;

        weight.begin_track_stats()?;
        experts.forward(&input, &[1, 2])?;
        assert_eq!(weight.stats_snapshot(), Some((1, 3)));
        assert_eq!(
            weight.end_track_stats()?.to_vec2::<f32>()?,
            vec![
                vec![1., 4., 9.],
                vec![(16. + 49.) / 2., (25. + 64.) / 2., (36. + 81.) / 2.]
            ]
        );
        Ok(())
    }

    #[test]
    fn mamba_prefill_continuation_matches_one_shot() -> Result<()> {
        let device = Device::Cpu;
        let layer = mamba_layer(&device)?;
        let seq_len = 6;
        let split = 2;
        let x = Tensor::from_vec(
            patterned(seq_len * HIDDEN_SIZE, 14, 0.2, 0.01),
            (1, seq_len, HIDDEN_SIZE),
            &device,
        )?;
        let initial = mamba_cache(&device)?;
        let initial = MambaLayerCache {
            conv_state: initial.conv_state.narrow(0, 0, 1)?,
            ssm_state: initial.ssm_state.narrow(0, 0, 1)?,
        };
        let mut one_shot_cache = initial.clone();
        let mut chunked_cache = initial;

        let one_shot = layer.forward(&x, &mut one_shot_cache, RecurrentBatchKind::Prefill, None)?;
        let first = layer.forward(
            &x.narrow(1, 0, split)?,
            &mut chunked_cache,
            RecurrentBatchKind::Prefill,
            None,
        )?;
        let second = layer.forward(
            &x.narrow(1, split, seq_len - split)?,
            &mut chunked_cache,
            RecurrentBatchKind::Prefill,
            None,
        )?;

        assert_close(&one_shot, &Tensor::cat(&[first, second], 1)?)?;
        assert_close(&one_shot_cache.conv_state, &chunked_cache.conv_state)?;
        assert_close(&one_shot_cache.ssm_state, &chunked_cache.ssm_state)
    }

    #[test]
    fn packed_mamba_maps_tokens_to_state_rows() {
        assert_eq!(
            packed_mamba_ranges(packed_shape(), &[2, 1, 4]).unwrap(),
            vec![0..2, 2..3, 3..7]
        );
    }

    #[test]
    fn packed_mamba_rejects_cardinality_mismatches() {
        let mut wrong_conv_batch = packed_shape();
        wrong_conv_batch.conv_state_batch = 2;
        assert!(packed_mamba_ranges(wrong_conv_batch, &[2, 1, 4]).is_err());

        let mut wrong_ssm_batch = packed_shape();
        wrong_ssm_batch.ssm_state_batch = 2;
        assert!(packed_mamba_ranges(wrong_ssm_batch, &[2, 1, 4]).is_err());

        assert!(packed_mamba_ranges(packed_shape(), &[2, 4]).is_err());
        assert!(packed_mamba_ranges(packed_shape(), &[2, 0, 5]).is_err());
    }

    #[test]
    fn packed_mamba_rejects_incompatible_shapes() {
        let mut non_packed_batch = packed_shape();
        non_packed_batch.physical_batch = 3;
        assert!(packed_mamba_ranges(non_packed_batch, &[2, 1, 4]).is_err());

        let mut wrong_conv_width = packed_shape();
        wrong_conv_width.conv_width = CONV_WIDTH - 1;
        assert!(packed_mamba_ranges(wrong_conv_width, &[2, 1, 4]).is_err());

        let mut wrong_ssm_width = packed_shape();
        wrong_ssm_width.ssm_state_width = STATE_SIZE + 1;
        assert!(packed_mamba_ranges(wrong_ssm_width, &[2, 1, 4]).is_err());
    }
}
