//! Experts with no gate projection, `down(act(up(x)))` (Nemotron-H's squared ReLU), on the gather path.

use inference_quant::{
    QuantMethod, QuantMethodConfig, QuantizedExpertKeys, Shard, ShardedVarBuilder, UnquantLinear,
    apply_immediate_isq_with_key, should_apply_immediate_isq,
};
use inference_tensor::nn::Linear;
use inference_tensor::{D, DType, Device, Result, Tensor};
use std::sync::Arc;

use super::checkpoint::{ExpertCheckpoint, ExpertSourceLayout};
use super::config::{ExpertProj, MoEExpertsConfig};
use super::forward::{MoEForward, MoEForwardConfig};

// NeMo Automodel's stacked experts: an ungated layer's "gate and up" holds only up
const NEMO_UP: &str = "gate_and_up_projs";
const NEMO_DOWN: &str = "down_projs";

pub(super) struct UngatedExpertsWeights {
    up: Arc<dyn QuantMethod>,
    down: Arc<dyn QuantMethod>,
}

impl UngatedExpertsWeights {
    /// The stacked `up_proj`/`down_proj` of a weight source (GGUF), or of any per-expert checkpoint layout.
    pub(super) fn load(
        cfg: &MoEExpertsConfig,
        experts_vb: ShardedVarBuilder,
        comm: &Arc<inference_quant::Comm>,
    ) -> Result<Self> {
        if comm.world_size() > 1 {
            inference_tensor::bail!("Experts without a gate projection run on one device.");
        }
        let keys = QuantizedExpertKeys::new(&experts_vb.prefix());
        if let Some(source) = experts_vb.weight_source()
            && source.contains(&format!("{}.weight", keys.up))
        {
            let load =
                |key: &str, predicate_vb: ShardedVarBuilder| -> Result<Arc<dyn QuantMethod>> {
                    let device = inference_quant::weight_source_load_device(&predicate_vb);
                    let layer = source
                        .load_linear(key, &device, Shard::default())?
                        .ok_or_else(|| {
                            inference_tensor::Error::Msg(format!("Missing expert weight `{key}`."))
                        })?;
                    apply_immediate_isq_with_key(
                        layer,
                        predicate_vb,
                        Some(key.to_string()),
                        Some(Shard::default()),
                    )
                };
            return Ok(Self {
                up: load(&keys.up, experts_vb.pp("up_proj"))?,
                down: load(&keys.down, experts_vb.pp("down_proj"))?,
            });
        }
        if experts_vb.contains_tensor(NEMO_UP) {
            return Self::load_nemo(cfg, experts_vb, keys);
        }
        let shape_of = |rel: &str| experts_vb.tensor_shape(rel).map(|s| s.to_vec());
        let Some(ExpertSourceLayout::PerExpert { names, .. }) =
            ExpertSourceLayout::detect(&shape_of, ExpertProj::Up)
        else {
            inference_tensor::bail!(
                "No per-expert up/down projections under `{}`.",
                experts_vb.prefix()
            );
        };
        // ISQ predicates match source names, so the predicate VBs follow the on-disk layout
        let expert0 = experts_vb.pp("0");
        let (vb_up, vb_down) = (expert0.pp(names.up), expert0.pp(names.down));
        let isq = should_apply_immediate_isq(&vb_up) || should_apply_immediate_isq(&vb_down);
        // read on the CPU when ISQ quantizes the stack straight away, as the gated loader does
        let read_vb = if isq && !experts_vb.device().is_cpu() {
            experts_vb.clone().set_device(Device::Cpu)
        } else {
            experts_vb.clone()
        };
        let checkpoint = ExpertCheckpoint::new(cfg, read_vb, comm)?;
        let load = |proj: ExpertProj, vb: ShardedVarBuilder, key: String| {
            let stacked = Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
                Linear::new(checkpoint.stacked_proj(proj)?, None),
            ))?);
            apply_immediate_isq_with_key(stacked, vb, Some(key), Some(Shard::default()))
        };
        Ok(Self {
            up: load(ExpertProj::Up, vb_up, keys.up)?,
            down: load(ExpertProj::Down, vb_down, keys.down)?,
        })
    }

    // NeMo Automodel's stacks, `[experts, in, out]`: transposed to `[experts, out, in]`
    fn load_nemo(
        cfg: &MoEExpertsConfig,
        experts_vb: ShardedVarBuilder,
        keys: QuantizedExpertKeys,
    ) -> Result<Self> {
        let isq = should_apply_immediate_isq(&experts_vb.pp(NEMO_UP))
            || should_apply_immediate_isq(&experts_vb.pp(NEMO_DOWN));
        let read_vb = if isq && !experts_vb.device().is_cpu() {
            experts_vb.clone().set_device(Device::Cpu)
        } else {
            experts_vb.clone()
        };
        let (e, h, i) = (cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size);
        let load = |name: &str, shape: (usize, usize, usize), key: String| {
            let stacked = read_vb.get(shape, name)?.transpose(1, 2)?.contiguous()?;
            let layer = Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
                Linear::new(stacked, None),
            ))?);
            apply_immediate_isq_with_key(
                layer,
                experts_vb.pp(name),
                Some(key),
                Some(Shard::default()),
            )
        };
        Ok(Self {
            up: load(NEMO_UP, (e, h, i), keys.up)?,
            down: load(NEMO_DOWN, (e, i, h), keys.down)?,
        })
    }

    pub(super) fn forward_impl(
        &self,
        forward: &MoEForward,
        config: MoEForwardConfig,
    ) -> Result<Tensor> {
        let (up, down) = (
            self.up.resolved_layer().unwrap_or_else(|| self.up.clone()),
            self.down
                .resolved_layer()
                .unwrap_or_else(|| self.down.clone()),
        );
        let ids = forward.topk_ids;
        up.process_routed_stats(forward.xs_flat, ids)?;
        let shape = forward.shape;
        let ys = if forward.xs.device().is_cuda() {
            let xs = forward
                .xs_flat
                .reshape((shape.num_tokens, 1, shape.hidden_dim))?;
            let down_in = up.gather_forward(&xs, ids)?.apply(&config.act)?;
            down.process_routed_stats(&down_in, ids)?;
            down.gather_forward(&down_in, ids)?
        } else {
            let xs =
                forward
                    .xs
                    .reshape((shape.batch_size, shape.seq_len, 1, 1, shape.hidden_dim))?;
            let indices =
                ids.reshape((shape.batch_size, shape.seq_len, config.num_experts_per_tok))?;
            let down_in = up.gather_forward(&xs, &indices)?.apply(&config.act)?;
            let inter = down_in.dim(D::Minus1)?;
            down.process_routed_stats(
                &down_in.reshape((shape.num_tokens, config.num_experts_per_tok, inter))?,
                ids,
            )?;
            down.gather_forward(&down_in, &indices)?
                .squeeze(D::Minus2)?
                .reshape((
                    shape.num_tokens,
                    config.num_experts_per_tok,
                    shape.hidden_dim,
                ))?
        };
        ys.to_dtype(DType::F32)?
            .broadcast_mul(&forward.topk_weights.unsqueeze(D::Minus1)?)?
            .sum(D::Minus2)?
            .to_dtype(forward.original_dtype)
    }
}
