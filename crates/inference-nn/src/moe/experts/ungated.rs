//! Experts with no gate projection, `down(act(up(x)))` (Nemotron-H's squared ReLU): grouped by expert where the
//! weights are dense, from their GGUF blocks on the CPU, else on the gather path.

use inference_quant::{
    QuantMethod, QuantMethodConfig, QuantizedExpertKeys, Shard, ShardedVarBuilder, UnquantLinear,
    apply_immediate_isq_with_key, should_apply_immediate_isq,
};
use inference_tensor::nn::Linear;
use inference_tensor::{D, DType, Device, Result, Tensor};
use rayon::prelude::*;
use std::sync::Arc;

use super::checkpoint::{ExpertCheckpoint, ExpertSourceLayout};
use super::config::{ExpertProj, MoEExpertsConfig};
use super::forward::{MoEForward, MoEForwardConfig};

// NeMo Automodel's stacked experts: an ungated layer's "gate and up" holds only up
const NEMO_UP: &str = "gate_and_up_projs";
const NEMO_DOWN: &str = "down_projs";

pub(super) struct UngatedExpertsWeights {
    pub(super) up: Arc<dyn QuantMethod>,
    pub(super) down: Arc<dyn QuantMethod>,
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
        let stacks = [
            checkpoint.stacked_proj(ExpertProj::Up)?,
            checkpoint.stacked_proj(ExpertProj::Down)?,
        ];
        Self::from_stacks(stacks, [(vb_up, keys.up), (vb_down, keys.down)])
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
        let stack = |name: &str, shape: (usize, usize, usize)| -> Result<Tensor> {
            read_vb.get(shape, name)?.transpose(1, 2)?.contiguous()
        };
        Self::from_stacks(
            [stack(NEMO_UP, (e, h, i))?, stack(NEMO_DOWN, (e, i, h))?],
            [
                (experts_vb.pp(NEMO_UP), keys.up),
                (experts_vb.pp(NEMO_DOWN), keys.down),
            ],
        )
    }

    // `[experts, out, in]` stacks as layers, quantized when ISQ asks
    fn from_stacks(stacks: [Tensor; 2], names: [(ShardedVarBuilder, String); 2]) -> Result<Self> {
        let [(up_vb, up_key), (down_vb, down_key)] = names;
        let layer = |stacked: &Tensor, vb: ShardedVarBuilder, key: String| {
            let layer = Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
                Linear::new(stacked.clone(), None),
            ))?);
            apply_immediate_isq_with_key(layer, vb, Some(key), Some(Shard::default()))
        };
        Ok(Self {
            up: layer(&stacks[0], up_vb, up_key)?,
            down: layer(&stacks[1], down_vb, down_key)?,
        })
    }

    // each expert's rows through its own matmuls, weighed and added back into their tokens' rows; unquantized stacks
    // (pending ones too, while calibration collects their routed statistics) run this way, as gathering them copies a
    // weight matrix per routed token
    fn forward_grouped(
        layers: [&Arc<dyn QuantMethod>; 2],
        [up, down]: [Tensor; 2],
        forward: &MoEForward,
        config: MoEForwardConfig,
    ) -> Result<Tensor> {
        layers[0].process_routed_stats(forward.xs_flat, forward.topk_ids)?;
        let tracking = layers[1].stats_snapshot().is_some();
        let ids = forward.topk_ids.to_dtype(DType::U32)?.to_vec2::<u32>()?;
        let weights = forward
            .topk_weights
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?;
        let mut routes: Vec<(Vec<u32>, Vec<f32>)> = vec![(Vec::new(), Vec::new()); up.dim(0)?];
        for (token, (experts, gates)) in ids.iter().zip(&weights).enumerate() {
            for (&expert, &gate) in experts.iter().zip(gates) {
                let route = &mut routes[expert as usize];
                route
                    .0
                    .push(u32::try_from(token).expect("token indices fit u32"));
                route.1.push(gate);
            }
        }
        let device = forward.xs_flat.device();
        // one expert's rows through its matmuls: the hidden rows (for calibration) and the gated output rows
        let run = |(expert, (tokens, gates)): (usize, (Vec<u32>, Vec<f32>))| -> Result<_> {
            let rows = tokens.len();
            let tokens = Tensor::from_vec(tokens, rows, device)?;
            let xs = forward.xs_flat.index_select(&tokens, 0)?;
            let hidden = xs.matmul(&up.get(expert)?.t()?)?.apply(&config.act)?;
            let ys = hidden
                .matmul(&down.get(expert)?.t()?)?
                .to_dtype(DType::F32)?
                .broadcast_mul(&Tensor::from_vec(gates, (rows, 1), device)?)?;
            Ok((expert, tokens, hidden, ys))
        };
        let routed = routes
            .into_iter()
            .enumerate()
            .filter(|(_, (tokens, _))| !tokens.is_empty());
        // on the CPU a prompt gives each expert a few dozen rows, too few for a matmul to spread over the cores, so
        // the experts run side by side instead
        let outputs = if device.is_cpu() {
            routed
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(run)
                .collect::<Result<Vec<_>>>()?
        } else {
            routed.map(run).collect::<Result<Vec<_>>>()?
        };
        let mut out = Tensor::zeros(forward.shape.flat(), DType::F32, device)?;
        for (expert, tokens, hidden, ys) in outputs {
            if tracking {
                let ids = Tensor::full(
                    u32::try_from(expert).expect("expert ids fit u32"),
                    (hidden.dim(0)?, 1),
                    device,
                )?;
                layers[1].process_routed_stats(&hidden.unsqueeze(1)?, &ids)?;
            }
            out = out.index_add(&tokens, &ys, 0)?;
        }
        out.to_dtype(forward.original_dtype)
    }

    // quantized stacks in host memory run from their blocks, on every core; gathering dequantizes a matrix per route
    fn forward_cpu_blocks(
        [up, down]: [&Arc<dyn QuantMethod>; 2],
        forward: &MoEForward,
        config: MoEForwardConfig,
    ) -> Result<Option<Tensor>> {
        use inference_quant::cpu_experts::{CpuExpertGate, CpuExperts, supports};
        if !forward.xs.device().is_cpu() || !matches!(config.act, crate::layers::Activation::Relu2)
        {
            return Ok(None);
        }
        let (Some(up_blocks), Some(down_blocks)) = (up.cpu_gguf_blocks(), down.cpu_gguf_blocks())
        else {
            return Ok(None);
        };
        let (Some(up_proj), Some(down_proj)) = (
            up_blocks.expert_projection(),
            down_blocks.expert_projection(),
        ) else {
            return Ok(None);
        };
        if ![up_proj, down_proj].iter().all(|p| supports(p.ty, p.cols)) {
            return Ok(None);
        }
        let experts = CpuExperts {
            gate: CpuExpertGate::SquaredRelu,
            up: up_proj,
            down: down_proj,
        };
        if down.stats_snapshot().is_none() {
            return super::backends::run_cpu_experts(&experts, forward, config, None).map(Some);
        }
        up.process_routed_stats(forward.xs_flat, forward.topk_ids)?;
        let mut hidden = Vec::new();
        let ys = super::backends::run_cpu_experts(&experts, forward, config, Some(&mut hidden))?;
        let rows = forward.shape.num_tokens;
        let hidden = Tensor::from_vec(
            hidden,
            (rows, config.num_experts_per_tok, up_proj.rows),
            &Device::Cpu,
        )?;
        down.process_routed_stats(&hidden, forward.topk_ids)?;
        Ok(Some(ys))
    }

    // a prompt's routes sorted by expert through grouped mmq, each routed expert's blocks read once per chunk
    // instead of once per routed token; calibration keeps the gather path, which hands back the routed rows
    #[cfg(feature = "cuda")]
    pub(super) fn forward_cuda_grouped(
        [up, down]: [&Arc<dyn QuantMethod>; 2],
        forward: &MoEForward,
        config: MoEForwardConfig,
    ) -> Result<Option<Tensor>> {
        if !forward.xs.device().is_cuda()
            || !forward.shape.phase.is_prefill()
            || forward.shape.num_tokens < super::backends::GROUPED_PREFILL_MIN_TOKENS
            || forward.lora.is_some()
            || up.stats_snapshot().is_some()
            || down.stats_snapshot().is_some()
        {
            return Ok(None);
        }
        let (Some(up_qt), Some(down_qt)) = (up.kernel_weight(), down.kernel_weight()) else {
            return Ok(None);
        };
        grouped_ungated_with([up_qt, down_qt], forward, config, false)
    }

    // the indexed matvec runs both projections; dense stacks route on the host, as grouped matmuls by expert
    pub(super) fn decode_is_device_only(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            let resolved =
                |w: &Arc<dyn QuantMethod>| w.resolved_layer().unwrap_or_else(|| w.clone());
            let (up, down) = (resolved(&self.up), resolved(&self.down));
            if let (Some(up), Some(down)) = (up.kernel_weight(), down.kernel_weight()) {
                return inference_quant::supports_indexed_moe_mmvq_ungated(up, down);
            }
        }
        false
    }

    // a decode step's few routes through the expert-indexed matvec, reading only the routed experts' blocks
    #[cfg(feature = "cuda")]
    pub(super) fn forward_cuda_decode(
        [up, down]: [&Arc<dyn QuantMethod>; 2],
        forward: &MoEForward,
        config: MoEForwardConfig,
    ) -> Result<Option<Tensor>> {
        if !forward.xs.device().is_cuda()
            || up.stats_snapshot().is_some()
            || down.stats_snapshot().is_some()
        {
            return Ok(None);
        }
        let (Some(up_qt), Some(down_qt)) = (up.kernel_weight(), down.kernel_weight()) else {
            return Ok(None);
        };
        indexed_mmvq_ungated_with([up_qt, down_qt], forward, config)
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
        if let (Some((up_w, None)), Some((down_w, None))) =
            (up.unquant_weight_bias(), down.unquant_weight_bias())
            && up_w.rank() == 3
        {
            return Self::forward_grouped([&up, &down], [up_w, down_w], forward, config);
        }
        if let Some(ys) = Self::forward_cpu_blocks([&up, &down], forward, config)? {
            return Ok(ys);
        }
        #[cfg(feature = "cuda")]
        if let Some(ys) = Self::forward_cuda_grouped([&up, &down], forward, config)? {
            return Ok(ys);
        }
        #[cfg(feature = "cuda")]
        if let Some(ys) = Self::forward_cuda_decode([&up, &down], forward, config)? {
            return Ok(ys);
        }
        Self::forward_gather([&up, &down], forward, config)
    }

    // each route's rows gathered through the indexed kernels, the path for decode and for calibration
    pub(super) fn forward_gather(
        [up, down]: [&Arc<dyn QuantMethod>; 2],
        forward: &MoEForward,
        config: MoEForwardConfig,
    ) -> Result<Tensor> {
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

/// The grouped mmq over `[up, down]` kernel weights; `parked` sorts routes to expert `config.num_experts` past the rest
/// and never runs them (their routing weight must be 0, which the weighted reduce skips).
#[cfg(feature = "cuda")]
pub(super) fn grouped_ungated_with(
    [up, down]: [&dyn inference_quant::KernelWeight; 2],
    forward: &MoEForward,
    config: MoEForwardConfig,
    parked: bool,
) -> Result<Option<Tensor>> {
    if !inference_quant::supports_mmq_weight(up) || !inference_quant::supports_mmq_weight(down) {
        return Ok(None);
    }
    let Some(super::backends::GroupedDispatch {
        dev,
        expert_bounds,
        sorted_token_ids,
        sorted_source_ids,
    }) = super::backends::grouped_dispatch(forward, config, parked)?
    else {
        return Ok(None);
    };
    let (tokens, topk) = (forward.shape.num_tokens, config.num_experts_per_tok);
    let total = tokens * topk;
    let hidden = inference_quant::grouped_moe_mmq(
        up,
        forward.xs_flat,
        &sorted_source_ids,
        &sorted_token_ids,
        &expert_bounds,
        total,
        tokens,
        config.num_experts,
        dev,
    )?
    .apply(&config.act)?;
    let inter = hidden.dim(D::Minus1)?;
    let down_rows = inference_quant::grouped_moe_mmq(
        down,
        &hidden.reshape((total, inter))?,
        &sorted_token_ids,
        &sorted_token_ids,
        &expert_bounds,
        total,
        tokens,
        config.num_experts,
        dev,
    )?
    .to_dtype(hidden.dtype())?;
    inference_quant::moe_weighted_reduce_flat_same_dtype(
        &down_rows.reshape((total, forward.shape.hidden_dim))?,
        forward.topk_weights,
        tokens,
        topk,
        dev,
    )?
    .to_dtype(forward.original_dtype)
    .map(Some)
}

/// The expert-indexed matvec over `[up, down]` kernel weights, for decode-sized batches.
#[cfg(feature = "cuda")]
pub(super) fn indexed_mmvq_ungated_with(
    [up, down]: [&dyn inference_quant::KernelWeight; 2],
    forward: &MoEForward,
    config: MoEForwardConfig,
) -> Result<Option<Tensor>> {
    if forward.lora.is_some()
        || forward.shape.num_tokens > super::backends::INDEXED_MMVQ_MAX_TOKENS
        || !inference_quant::supports_indexed_moe_mmvq_ungated(up, down)
    {
        return Ok(None);
    }
    let routed = inference_quant::indexed_moe_mmvq_ungated(
        up,
        down,
        forward.xs_flat,
        forward.topk_ids,
        config.num_experts_per_tok,
        &|rows| rows.apply(&config.act),
    )?;
    let out = super::backends::weighted_reduce_f32(forward, &routed, config.num_experts_per_tok)?;
    out.to_dtype(forward.original_dtype).map(Some)
}
