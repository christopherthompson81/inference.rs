use std::sync::Arc;

use inference_quant::ShardedVarBuilder;
use inference_tensor::nn::Linear;
use inference_tensor::{Device, Module, Result, Tensor};

use super::config::TextConfig;
use crate::{
    layers::{self, GemmaRmsNorm, Mlp},
    moe::{MoEExperts, MoEExpertsConfig},
    utils::unvarbuilder::UnVarBuilder,
};

/// A decoder layer's feed-forward: the dense MLP, or the routed experts plus a gated shared expert.
pub(super) enum FeedForward {
    Dense(Mlp),
    Sparse(SparseMoeBlock),
}

impl FeedForward {
    /// `vb` is the layer's `mlp` prefix, already placed for ISQ; `layer_device` is where the router lives.
    pub(super) fn load(
        cfg: &TextConfig,
        vb: ShardedVarBuilder,
        layer_device: Device,
        loading_isq: bool,
        comm: &Arc<inference_quant::Comm>,
    ) -> Result<Self> {
        if cfg.is_moe() {
            return Ok(Self::Sparse(SparseMoeBlock::new(
                cfg,
                vb,
                layer_device,
                loading_isq,
                comm,
            )?));
        }
        Ok(Self::Dense(Self::dense_mlp(cfg, vb, comm)?))
    }

    pub(super) fn dense_mlp(
        cfg: &TextConfig,
        vb: ShardedVarBuilder,
        comm: &Arc<inference_quant::Comm>,
    ) -> Result<Mlp> {
        Mlp::new(
            vb,
            cfg.hidden_size,
            cfg.dense_intermediate_size()?,
            &cfg.quantization_config,
            cfg.hidden_act,
            comm,
        )
    }

    /// Adds `branch` to `residual`, normalizes, and returns the new residual with the feed-forward output.
    pub(super) fn forward_with_add_rms_norm(
        &self,
        branch: &Tensor,
        residual: &Tensor,
        norm: &GemmaRmsNorm,
    ) -> Result<(Tensor, Tensor)> {
        match self {
            Self::Dense(mlp) => mlp.forward_with_add_rms_norm(branch, residual, norm),
            Self::Sparse(moe) => {
                let (residual, normalized) = norm.forward_add_rms_norm(branch, residual)?;
                Ok((residual, moe.forward(&normalized)?))
            }
        }
    }

    pub(super) fn residual_tensors(&self, uvb_mlp: &UnVarBuilder) {
        if let Self::Sparse(moe) = self {
            uvb_mlp
                .pp("gate")
                .add_tensor("weight", moe.gate.weight().clone());
            uvb_mlp
                .pp("shared_expert_gate")
                .add_tensor("weight", moe.shared_expert_gate.weight().clone());
        }
    }
}

pub(super) struct SparseMoeBlock {
    gate: Linear,
    gate_lora: Option<Arc<inference_quant::LoraSiteHandle>>,
    experts: MoEExperts,
    shared_expert: Mlp,
    shared_expert_gate: Linear,
    shared_expert_gate_lora: Option<Arc<inference_quant::LoraSiteHandle>>,
    num_experts_per_tok: usize,
    norm_topk_prob: bool,
}

impl SparseMoeBlock {
    fn new(
        cfg: &TextConfig,
        vb: ShardedVarBuilder,
        layer_device: Device,
        loading_isq: bool,
        comm: &Arc<inference_quant::Comm>,
    ) -> Result<Self> {
        let gate_vb = vb.pp("gate").set_device(layer_device.clone());
        let gate = layers::linear_no_bias(cfg.hidden_size, cfg.num_experts, gate_vb.clone())?;
        let gate_lora = inference_quant::register_dynamic_lora_site(
            &gate_vb,
            inference_quant::LoraLinearSpec::replicated(cfg.hidden_size, cfg.num_experts),
        )?;

        let moe_cfg = MoEExpertsConfig {
            num_experts: cfg.num_experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
            hidden_size: cfg.hidden_size,
            moe_intermediate_size: cfg.moe_intermediate_size,
            expert_proj_names: crate::moe::ExpertProjNames::DEFAULT,
        };
        let experts = MoEExperts::new(
            &moe_cfg,
            vb.clone(),
            layer_device.clone(),
            comm,
            loading_isq,
            &cfg.quantization_config,
            cfg.hidden_act,
        )?;

        let shared_expert = Mlp::new(
            vb.pp("shared_expert"),
            cfg.hidden_size,
            cfg.shared_expert_intermediate_size,
            &cfg.quantization_config,
            cfg.hidden_act,
            comm,
        )?;

        let shared_expert_gate_vb = vb.pp("shared_expert_gate");
        let mut seg_w = shared_expert_gate_vb.get((1, cfg.hidden_size), "weight")?;
        if loading_isq {
            seg_w = seg_w.to_device(&layer_device)?;
        }
        let shared_expert_gate = Linear::new(seg_w, None);
        let shared_expert_gate_lora = inference_quant::register_dynamic_lora_site(
            &shared_expert_gate_vb.set_device(layer_device),
            inference_quant::LoraLinearSpec::replicated(cfg.hidden_size, 1),
        )?;

        Ok(Self {
            gate,
            gate_lora,
            experts,
            shared_expert,
            shared_expert_gate,
            shared_expert_gate_lora,
            num_experts_per_tok: cfg.num_experts_per_tok,
            norm_topk_prob: cfg.norm_topk_prob,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_size, seq_len, hidden_dim) = xs.dims3()?;
        let xs_flat = xs.reshape(((), hidden_dim))?;

        let router_logits = self.gate.forward(&xs_flat)?;
        let router_logits = match &self.gate_lora {
            Some(site) => inference_quant::apply_dynamic_lora_delta(site, &xs_flat, router_logits)?,
            None => router_logits,
        };
        let topk = crate::ops::moe_router_topk(
            &router_logits,
            crate::ops::MoeRouterTopKConfig {
                top_k: self.num_experts_per_tok,
                score_function: crate::ops::MoeRouterScoreFunction::Softmax,
                selected_weight: crate::ops::MoeRouterSelectedWeight::Score,
                renormalize: self.norm_topk_prob,
                norm_min: 0.0,
                output_scale: 1.0,
                logit_clip: None,
            },
            None,
            None,
        )?;

        let mut y = self.experts.forward(xs, topk.values, &topk.indices)?;
        y = y.reshape((b_size, seq_len, hidden_dim))?;

        let shared_out = self.shared_expert.forward(xs)?;
        let shared_gate = self.shared_expert_gate.forward(&xs_flat)?;
        let shared_gate = match &self.shared_expert_gate_lora {
            Some(site) => inference_quant::apply_dynamic_lora_delta(site, &xs_flat, shared_gate)?,
            None => shared_gate,
        };
        let shared_gate = inference_tensor::nn::ops::sigmoid(&shared_gate)?;
        let shared_gate = shared_gate.reshape((b_size, seq_len, 1))?;
        let shared_out = shared_out.broadcast_mul(&shared_gate)?;

        y + shared_out
    }
}
