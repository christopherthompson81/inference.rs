//! The sparse-MoE feed-forward: a router over experts, in place of a layer's dense MLP.

use std::sync::Arc;

use inference_quant::{LoraSiteHandle, QuantMethod, ShardedVarBuilder};
use inference_tensor::{D, DType, Module, Result, Tensor, nn::Linear};

use crate::{
    layers::{self, Activation, masker::masked_fill},
    moe::{ExpertProjNames, MoEExperts, MoEExpertsConfig},
    ops::{MoeRouterScoreFunction, MoeRouterSelectedWeight, MoeRouterTopKConfig},
    utils::unvarbuilder::UnVarBuilder,
};

use super::LayerLoad;

/// Mixtral's and Phi-3.5-MoE's name for a layer's experts and router.
pub const BLOCK_SPARSE_MOE: &str = "block_sparse_moe";
const ROUTER: &str = "gate";
const SHARED_EXPERT_GATE: &str = "shared_expert_gate";
const SELECTION_BIAS: (&str, &str) = ("moe_statics", "e_score_correction_bias");
// sparsemixer routes each token to two experts by construction
const SPARSE_MIXER_EXPERTS: usize = 2;

/// How a layer's router picks its experts and weighs them.
#[derive(Clone, Copy, Debug)]
pub enum MoeRouting {
    /// Softmax over the router logits, then the top `k`, renormalized to sum to one when asked.
    TopK { k: usize, renormalize: bool },
    /// Phi-3.5-MoE's sparsemixer: the top two experts, each weighed by a softmax masked at the jitter threshold.
    SparseMixer { jitter_eps: f64 },
    /// ERNIE 4.5: the router and a per-expert selection bias (`moe_statics.e_score_correction_bias`) in f32; the top
    /// `k` of softmax plus bias, weighed by their softmax renormalized over a sum clamped at `norm_min`.
    BiasedTopK { k: usize, norm_min: f32 },
}

impl MoeRouting {
    fn experts_per_token(self) -> usize {
        match self {
            Self::TopK { k, .. } | Self::BiasedTopK { k, .. } => k,
            Self::SparseMixer { .. } => SPARSE_MIXER_EXPERTS,
        }
    }
}

/// The experts a stack's MoE layers route over.
#[derive(Clone, Debug)]
pub struct MoeSpec {
    pub num_experts: usize,
    pub intermediate_size: usize,
    pub routing: MoeRouting,
    /// The router is quantized like a projection; otherwise it stays a plain linear with a LoRA site.
    pub quantized_router: bool,
    pub expert_names: ExpertProjNames,
    /// The feed-forward's name in the layer: `mlp`, or Mixtral's `block_sparse_moe`.
    pub name: &'static str,
    /// Layers that keep a dense MLP instead.
    pub dense_layers: Vec<usize>,
    /// Whether a captured CUDA decode graph has been trusted with these experts.
    pub cuda_decode_graphs: bool,
    pub shared_expert: Option<SharedExpertSpec>,
}

/// An MLP every token runs beside the routed experts.
#[derive(Clone, Copy, Debug)]
pub struct SharedExpertSpec {
    pub intermediate_size: usize,
    /// `shared_expert` (Qwen2-MoE) or `shared_experts` (ERNIE 4.5, its experts fused into one MLP).
    pub name: &'static str,
    /// Scaled by `sigmoid(shared_expert_gate(x))`, as Qwen2-MoE's is.
    pub gated: bool,
}

enum Router {
    Quantized(Arc<dyn QuantMethod>),
    Plain {
        gate: Linear,
        lora: Option<Arc<LoraSiteHandle>>,
    },
    /// An f32 router and its selection bias, as the checkpoint stores them (`[1, experts]`).
    F32 {
        gate: Linear,
        bias: Tensor,
    },
}

/// A router and its experts, as [`MoeSpec`] describes them.
pub struct SparseMoe {
    router: Router,
    experts: MoEExperts,
    routing: MoeRouting,
    name: &'static str,
    cuda_decode_graphs: bool,
    shared_expert: Option<Box<SharedExpert>>,
}

struct SharedExpert {
    mlp: layers::Mlp,
    name: &'static str,
    gate: Option<(Linear, Option<Arc<LoraSiteHandle>>)>,
}

impl SharedExpert {
    fn forward(&self, xs: &Tensor, xs_flat: &Tensor) -> Result<Tensor> {
        let out = self.mlp.forward(xs)?;
        let Some((gate, lora)) = &self.gate else {
            return Ok(out);
        };
        let (b_size, seq_len, _) = xs.dims3()?;
        let gate_logits = gate.forward(xs_flat)?;
        let gate_logits = match lora {
            Some(site) => inference_quant::apply_dynamic_lora_delta(site, xs_flat, gate_logits)?,
            None => gate_logits,
        };
        let gate =
            inference_tensor::nn::ops::sigmoid(&gate_logits)?.reshape((b_size, seq_len, 1))?;
        out.broadcast_mul(&gate)
    }
}

impl SparseMoe {
    pub(super) fn new(
        spec: &MoeSpec,
        hidden_size: usize,
        quantization_config: &Option<inference_quant::QuantizedConfig>,
        act: Activation,
        load: &LayerLoad<'_>,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let router = if let MoeRouting::BiasedTopK { .. } = spec.routing {
            let router_vb = vb
                .pp(ROUTER)
                .set_device(load.device.clone())
                .set_dtype(DType::F32);
            let gate = layers::linear_no_bias(hidden_size, spec.num_experts, router_vb)?;
            let bias = vb
                .pp(SELECTION_BIAS.0)
                .set_device(load.device.clone())
                .get_with_hints_dtype(
                    (1, spec.num_experts),
                    SELECTION_BIAS.1,
                    Default::default(),
                    DType::F32,
                )?;
            Router::F32 { gate, bias }
        } else if spec.quantized_router {
            Router::Quantized(inference_quant::linear_no_bias(
                hidden_size,
                spec.num_experts,
                quantization_config,
                vb.pp(ROUTER),
            )?)
        } else {
            let gate_vb = vb.pp(ROUTER).set_device(load.device.clone());
            Router::Plain {
                gate: layers::linear_no_bias(hidden_size, spec.num_experts, gate_vb.clone())?,
                lora: inference_quant::register_dynamic_lora_site(
                    &gate_vb,
                    inference_quant::LoraLinearSpec::replicated(hidden_size, spec.num_experts),
                )?,
            }
        };
        let experts = MoEExperts::new(
            &MoEExpertsConfig {
                num_experts: spec.num_experts,
                num_experts_per_tok: spec.routing.experts_per_token(),
                hidden_size,
                moe_intermediate_size: spec.intermediate_size,
                expert_proj_names: spec.expert_names,
            },
            vb.clone(),
            load.device.clone(),
            load.comm,
            load.loading_isq,
            quantization_config,
            act,
        )?;
        let shared_expert = spec
            .shared_expert
            .map(|shared| -> Result<Box<SharedExpert>> {
                let gate = shared
                    .gated
                    .then(|| -> Result<_> {
                        let gate_vb = vb.pp(SHARED_EXPERT_GATE).set_device(load.device.clone());
                        Ok((
                            layers::linear_no_bias(hidden_size, 1, gate_vb.clone())?,
                            inference_quant::register_dynamic_lora_site(
                                &gate_vb,
                                inference_quant::LoraLinearSpec::replicated(hidden_size, 1),
                            )?,
                        ))
                    })
                    .transpose()?;
                Ok(Box::new(SharedExpert {
                    mlp: layers::Mlp::new(
                        vb.pp(shared.name),
                        hidden_size,
                        shared.intermediate_size,
                        quantization_config,
                        act,
                        load.comm,
                    )?,
                    name: shared.name,
                    gate,
                }))
            })
            .transpose()?;
        Ok(Self {
            router,
            experts,
            routing: spec.routing,
            name: spec.name,
            cuda_decode_graphs: spec.cuda_decode_graphs,
            shared_expert,
        })
    }

    pub(super) fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_size, seq_len, hidden) = xs.dims3()?;
        let xs_flat = xs.reshape(((), hidden))?;
        let logits = match &self.router {
            Router::Quantized(gate) => gate.forward(&xs_flat)?,
            Router::Plain { gate, lora } => {
                let logits = gate.forward(&xs_flat)?;
                match lora {
                    Some(site) => {
                        inference_quant::apply_dynamic_lora_delta(site, &xs_flat, logits)?
                    }
                    None => logits,
                }
            }
            Router::F32 { gate, .. } => gate.forward(&xs_flat.to_dtype(DType::F32)?)?,
        };
        let (weights, experts) = match self.routing {
            MoeRouting::TopK { k, renormalize } => {
                let topk = crate::ops::moe_router_topk(
                    &logits,
                    MoeRouterTopKConfig {
                        top_k: k,
                        score_function: MoeRouterScoreFunction::Softmax,
                        selected_weight: MoeRouterSelectedWeight::Score,
                        renormalize,
                        norm_min: 0.0,
                        output_scale: 1.0,
                        logit_clip: None,
                    },
                    None,
                    None,
                )?;
                (topk.values, topk.indices)
            }
            MoeRouting::SparseMixer { jitter_eps } => {
                let (weights, experts) = sparsemixer(&logits, jitter_eps)?;
                (weights.to_dtype(DType::F32)?, experts)
            }
            MoeRouting::BiasedTopK { k, norm_min } => {
                let Router::F32 { bias, .. } = &self.router else {
                    inference_tensor::bail!("biased top-k routing without its f32 router");
                };
                let topk = crate::ops::moe_router_topk(
                    &logits,
                    MoeRouterTopKConfig {
                        top_k: k,
                        score_function: MoeRouterScoreFunction::Softmax,
                        selected_weight: MoeRouterSelectedWeight::Score,
                        renormalize: true,
                        norm_min,
                        output_scale: 1.0,
                        logit_clip: None,
                    },
                    Some(bias),
                    None,
                )?;
                (topk.values, topk.indices)
            }
        };
        let Some(shared_expert) = &self.shared_expert else {
            return self
                .experts
                .forward(xs, weights, &experts)?
                .reshape((b_size, seq_len, hidden))?
                .to_dtype(xs.dtype());
        };
        // the shared expert runs while offloaded experts' misses are on the CPU
        let (routed, shared) =
            self.experts
                .forward_overlapping(xs, weights, &experts, &mut || {
                    shared_expert.forward(xs, &xs_flat)
                })?;
        routed
            .reshape((b_size, seq_len, hidden))?
            .to_dtype(xs.dtype())?
            + shared
    }

    pub(super) fn name(&self) -> &'static str {
        self.name
    }

    pub(super) fn cuda_decode_graphs(&self) -> bool {
        self.cuda_decode_graphs
    }

    /// A plain router, which ISQ leaves alone.
    pub(super) fn add_residual(&self, uvb: &UnVarBuilder) {
        match &self.router {
            Router::Plain { gate, .. } => uvb.pp(ROUTER).add(gate),
            Router::F32 { gate, bias } => {
                uvb.pp(ROUTER).add(gate);
                uvb.pp(SELECTION_BIAS.0)
                    .add_tensor(SELECTION_BIAS.1, bias.clone());
            }
            Router::Quantized(_) => {}
        }
        if let Some((gate, _)) = self
            .shared_expert
            .as_ref()
            .and_then(|shared| shared.gate.as_ref())
        {
            uvb.pp(SHARED_EXPERT_GATE).add(gate);
        }
    }

    /// A quantized router, which a MoE-experts-only ISQ leaves alone too.
    pub(super) fn add_projections(&self, uvb: &UnVarBuilder) {
        if let Router::Quantized(gate) = &self.router {
            uvb.pp(ROUTER).add(gate);
        }
        if let Some(shared_expert) = &self.shared_expert {
            let uvb = uvb.pp(shared_expert.name);
            uvb.pp("gate_proj").add(&shared_expert.mlp.gate);
            uvb.pp("up_proj").add(&shared_expert.mlp.up);
            uvb.pp("down_proj").add(&shared_expert.mlp.down);
        }
    }
}

/// Phi-3.5-MoE's top-2 routing: (weights, experts), each `[tokens, 2]`.
fn sparsemixer(scores: &Tensor, jitter_eps: f64) -> Result<(Tensor, Tensor)> {
    let top1 = scores.argmax_keepdim(D::Minus1)?;
    let weight1 = masked_softmax_at(scores, scores, &top1, jitter_eps)?;
    let masked_scores = scores.scatter_add(
        &top1.broadcast_as(scores.shape())?.contiguous()?,
        &(scores.ones_like()? * f64::NEG_INFINITY)?,
        D::Minus1,
    )?;
    let top2 = masked_scores.argmax_keepdim(D::Minus1)?;
    let weight2 = masked_softmax_at(scores, &masked_scores, &top2, jitter_eps)?;
    Ok((
        Tensor::cat(&[weight1, weight2], D::Minus1)?,
        Tensor::cat(&[top1, top2], D::Minus1)?,
    ))
}

/// The softmax weight of `selected` over `candidates`, masking those further than `2 * jitter_eps` below it.
fn masked_softmax_at(
    scores: &Tensor,
    candidates: &Tensor,
    selected: &Tensor,
    jitter_eps: f64,
) -> Result<Tensor> {
    let threshold = candidates.gather(selected, D::Minus1)?;
    let factor = scores.abs()?.broadcast_minimum(&threshold)?;
    let mask = threshold
        .broadcast_sub(scores)?
        .broadcast_div(&factor)?
        .gt(2. * jitter_eps)?;
    let gates = inference_tensor::nn::ops::softmax_last_dim(&masked_fill(
        candidates,
        &mask,
        f64::NEG_INFINITY,
    )?)?;
    gates.gather(selected, D::Minus1)
}
