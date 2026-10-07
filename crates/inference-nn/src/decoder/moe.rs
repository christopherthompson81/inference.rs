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
// sparsemixer routes each token to two experts by construction
const SPARSE_MIXER_EXPERTS: usize = 2;

/// How a layer's router picks its experts and weighs them.
#[derive(Clone, Copy, Debug)]
pub enum MoeRouting {
    /// Softmax over the router logits, then the top `k`, renormalized to sum to one when asked.
    TopK { k: usize, renormalize: bool },
    /// Phi-3.5-MoE's sparsemixer: the top two experts, each weighed by a softmax masked at the jitter threshold.
    SparseMixer { jitter_eps: f64 },
}

impl MoeRouting {
    fn experts_per_token(self) -> usize {
        match self {
            Self::TopK { k, .. } => k,
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
}

enum Router {
    Quantized(Arc<dyn QuantMethod>),
    Plain {
        gate: Linear,
        lora: Option<Arc<LoraSiteHandle>>,
    },
}

/// A router and its experts, as [`MoeSpec`] describes them.
pub struct SparseMoe {
    router: Router,
    experts: MoEExperts,
    routing: MoeRouting,
    name: &'static str,
    cuda_decode_graphs: bool,
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
        let router = if spec.quantized_router {
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
            vb,
            load.device.clone(),
            load.comm,
            load.loading_isq,
            quantization_config,
            act,
        )?;
        Ok(Self {
            router,
            experts,
            routing: spec.routing,
            name: spec.name,
            cuda_decode_graphs: spec.cuda_decode_graphs,
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
        };
        self.experts
            .forward(xs, weights, &experts)?
            .reshape((b_size, seq_len, hidden))?
            .to_dtype(xs.dtype())
    }

    pub(super) fn name(&self) -> &'static str {
        self.name
    }

    pub(super) fn cuda_decode_graphs(&self) -> bool {
        self.cuda_decode_graphs
    }

    /// A plain router, which ISQ leaves alone.
    pub(super) fn add_residual(&self, uvb: &UnVarBuilder) {
        if let Router::Plain { gate, .. } = &self.router {
            uvb.pp(ROUTER).add(gate);
        }
    }

    /// A quantized router, which a MoE-experts-only ISQ leaves alone too.
    pub(super) fn add_projections(&self, uvb: &UnVarBuilder) {
        if let Router::Quantized(gate) = &self.router {
            uvb.pp(ROUTER).add(gate);
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
