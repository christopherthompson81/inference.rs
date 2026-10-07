use inference_quant::ShardedVarBuilder;
use inference_tensor::{D, DType, Result, Tensor};

use crate::ops::{
    MoeRouterScoreFunction, MoeRouterSelectedWeight, MoeRouterTopKConfig, TopKLastDimOp,
    TopKOutput, moe_router_topk,
};

const NORM_MIN: f32 = 1e-20;
const RENORM_EPS: f64 = 1e-20;
// noaux_tc ranks each group by the sum of its two best experts
const NOAUX_TC_GROUP_TOP: usize = 2;
const CORRECTION_BIAS: &str = "e_score_correction_bias";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterScoring {
    Softmax,
    Sigmoid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterMethod {
    Greedy,
    GroupLimitedGreedy,
    NoAuxTc,
}

/// When the selected weights are renormalised, one variant per model rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterRenorm {
    /// DeepSeek-V2: renormalise when `top_k > 1 && norm_topk_prob`, and then skip the scale.
    TopkProbSkipsScale { norm_topk_prob: bool },
    /// GLM4-MoE-Lite: always renormalise, then scale.
    Always,
    /// DeepSeek-V3 and GLM4-MoE: renormalise when `norm_topk_prob`, then scale.
    TopkProb { norm_topk_prob: bool },
}

#[derive(Debug, Clone, Copy)]
pub struct GroupedRouterConfig {
    pub scoring: RouterScoring,
    pub method: RouterMethod,
    pub n_group: usize,
    pub topk_group: usize,
    pub routed_scaling_factor: f64,
    pub renorm: RouterRenorm,
}

/// The DeepSeek-V2/V3 and GLM4-MoE router: scores gate logits and picks `top_k` experts per token.
pub struct GroupedRouter {
    cfg: GroupedRouterConfig,
    top_k: usize,
    n_routed_experts: usize,
    e_score_correction_bias: Option<Tensor>,
}

impl GroupedRouter {
    /// Loads `e_score_correction_bias` from the gate's builder when the method is noaux_tc.
    pub fn load(
        cfg: GroupedRouterConfig,
        top_k: usize,
        vb: &ShardedVarBuilder,
        n_routed_experts: usize,
    ) -> Result<Self> {
        let e_score_correction_bias = if cfg.method == RouterMethod::NoAuxTc {
            Some(vb.get_with_hints_dtype(
                n_routed_experts,
                CORRECTION_BIAS,
                Default::default(),
                DType::F32,
            )?)
        } else {
            None
        };
        Ok(Self {
            cfg,
            top_k,
            n_routed_experts,
            e_score_correction_bias,
        })
    }

    pub fn e_score_correction_bias(&self) -> Option<&Tensor> {
        self.e_score_correction_bias.as_ref()
    }

    fn renormalises(&self) -> bool {
        match self.cfg.renorm {
            RouterRenorm::TopkProbSkipsScale { norm_topk_prob } => self.top_k > 1 && norm_topk_prob,
            RouterRenorm::Always => true,
            RouterRenorm::TopkProb { norm_topk_prob } => norm_topk_prob,
        }
    }

    /// Routes F32 gate logits of shape (tokens, experts) to (topk_idx, topk_weight).
    pub fn route(&self, logits: &Tensor) -> Result<(Tensor, Tensor)> {
        let cfg = &self.cfg;
        let renormalize = self.renormalises();
        let skip_scale =
            renormalize && matches!(cfg.renorm, RouterRenorm::TopkProbSkipsScale { .. });
        if cfg.method == RouterMethod::Greedy {
            #[allow(clippy::cast_possible_truncation)]
            let scale = cfg.routed_scaling_factor as f32;
            let topk = moe_router_topk(
                logits,
                MoeRouterTopKConfig {
                    top_k: self.top_k,
                    score_function: match cfg.scoring {
                        RouterScoring::Softmax => MoeRouterScoreFunction::Softmax,
                        RouterScoring::Sigmoid => MoeRouterScoreFunction::Sigmoid,
                    },
                    selected_weight: MoeRouterSelectedWeight::Score,
                    renormalize,
                    norm_min: NORM_MIN,
                    output_scale: if skip_scale { 1.0 } else { scale },
                    logit_clip: None,
                },
                None,
                None,
            )?;
            return Ok((topk.indices, topk.values));
        }
        let n = logits.dim(0)?;
        let scores = match cfg.scoring {
            RouterScoring::Softmax => inference_tensor::nn::ops::softmax_last_dim(logits)?,
            RouterScoring::Sigmoid => inference_tensor::nn::ops::sigmoid(logits)?,
        };

        let (mut topk_weight, topk_idx) = match cfg.method {
            RouterMethod::Greedy => unreachable!(),
            RouterMethod::NoAuxTc => {
                let Some(e_score_correction_bias) = &self.e_score_correction_bias else {
                    inference_tensor::bail!("Expected e_score_correction_bias")
                };
                let scores_for_choice = scores
                    .reshape((n, ()))?
                    .broadcast_add(&e_score_correction_bias.unsqueeze(0)?)?;
                // (n, n_group)
                let group_scores = scores_for_choice
                    .reshape((n, cfg.n_group, ()))?
                    .topk(NOAUX_TC_GROUP_TOP)?
                    .values
                    .sum(D::Minus1)?;
                // (n, topk_group)
                let group_idx = group_scores.topk(cfg.topk_group)?.indices;
                let score_mask = self.group_score_mask(&group_scores, &group_idx, n)?;
                let tmp_scores = scores_for_choice.broadcast_mul(&score_mask)?;
                let topk_idx = tmp_scores.topk(self.top_k)?.indices;
                (scores.gather(&topk_idx, 1)?, topk_idx)
            }
            RouterMethod::GroupLimitedGreedy => {
                // (n, n_group)
                let group_scores = scores.reshape((n, cfg.n_group, ()))?.max(D::Minus1)?;
                // (n, topk_group)
                let group_idx = group_scores.topk_unsorted(cfg.topk_group)?.indices;
                let score_mask = self.group_score_mask(&group_scores, &group_idx, n)?;
                // Experts outside the chosen groups score 0, as HF's `scores.masked_fill(~score_mask, 0.0)`.
                let tmp_scores = scores.broadcast_mul(&score_mask)?;
                let TopKOutput { values, indices } = tmp_scores.topk_unsorted(self.top_k)?;
                (values, indices)
            }
        };

        if renormalize {
            let denominator = (topk_weight.sum_keepdim(D::Minus1)? + RENORM_EPS)?;
            topk_weight = topk_weight.broadcast_div(&denominator)?;
        }
        if !skip_scale {
            topk_weight = (topk_weight * cfg.routed_scaling_factor)?;
        }
        Ok((topk_idx, topk_weight))
    }

    // (n, e): 1 for every expert in a selected group
    fn group_score_mask(
        &self,
        group_scores: &Tensor,
        group_idx: &Tensor,
        n: usize,
    ) -> Result<Tensor> {
        let group_mask = group_scores.zeros_like()?.scatter_add(
            group_idx,
            &group_idx.ones_like()?.to_dtype(group_scores.dtype())?,
            1,
        )?;
        group_mask
            .unsqueeze(D::Minus1)?
            .expand((
                n,
                self.cfg.n_group,
                self.n_routed_experts / self.cfg.n_group,
            ))?
            .reshape((n, ()))
    }
}
