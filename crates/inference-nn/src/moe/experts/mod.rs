//! Unified MoE experts layer supporting multiple backends and weight formats.
//!
//! `MoEExperts` does NOT carry the gate (router); the caller computes routing. Three orthogonal
//! axes, each in one place: [`config`] (which backend to run), [`checkpoint`] (what's on disk ->
//! canonical ENK), and [`backends`] (materialize + forward). [`forward`] holds the per-call shapes.

mod backends;
mod checkpoint;
pub use checkpoint::{expert_stack_available, rebuild_expert_projection};
mod config;
mod forward;
#[cfg(feature = "cuda")]
mod offload;

use inference_quant::{
    IsqType, LoraExpertExecution, LoraExpertProjectionNames, LoraExpertSiteHandle,
    LoraExpertSiteSpec, LoraSiteKey, QuantizedConfig, Shard, ShardedVarBuilder, SumAllReduce,
};
use inference_tensor::{Device, Result, Tensor};
use std::sync::Arc;

use crate::layers::Activation;

pub use config::{ExpertProj, ExpertProjNames, MoEExpertsConfig, prelog_moe_backend};

#[cfg(feature = "cutile")]
use backends::CutileExpertsWeights;
#[cfg(feature = "cutile")]
use backends::CutileFp8ExpertsWeights;
#[cfg(feature = "cuda")]
use backends::CutlassExpertsWeights;
use backends::{
    FastExpertsWeights, FusedExpertsWeights, StackedExpertWeights, experts_are_prequantized,
};
use checkpoint::ExpertCheckpoint;
use config::{BackendChoice, MoEExpertsBackend};
use forward::{MoEForward, MoEForwardConfig, MoEForwardShape};

/// MoE experts layer without the gate; the caller computes routing weights + topk indices.
pub struct MoEExperts {
    backend: MoEExpertsBackendImpl,
    lora_site: Option<Arc<LoraExpertSiteHandle>>,
    act: Activation,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    num_experts: usize,
    num_experts_per_tok: usize,
    all_reduce: SumAllReduce,
    world_size: usize,
}

enum MoEExpertsBackendImpl {
    Fused(FusedExpertsWeights),
    #[cfg(feature = "cutile")]
    Cutile(CutileExpertsWeights),
    #[cfg(feature = "cutile")]
    CutileFp8(CutileFp8ExpertsWeights),
    #[cfg(feature = "cuda")]
    Cutlass(CutlassExpertsWeights),
    Fast(FastExpertsWeights),
    #[cfg(feature = "cuda")]
    Offloaded(Box<offload::OffloadedExperts>),
}

impl MoEExpertsBackendImpl {
    fn forward(&self, forward: &MoEForward, config: MoEForwardConfig) -> Result<Tensor> {
        match self {
            Self::Fused(w) => w.forward_impl(forward, config),
            #[cfg(feature = "cutile")]
            Self::Cutile(w) => w
                .forward_impl(forward, config)
                .map_err(|err| err.context("moe experts cutile")),
            #[cfg(feature = "cutile")]
            Self::CutileFp8(w) => w
                .forward_impl(forward, config)
                .map_err(|err| err.context("moe experts cutile fp8")),
            #[cfg(feature = "cuda")]
            Self::Cutlass(w) => w
                .forward_impl(forward, config)
                .map_err(|err| err.context("moe experts cutlass")),
            Self::Fast(w) => w.forward_impl(forward, config),
            #[cfg(feature = "cuda")]
            Self::Offloaded(w) => w
                .forward(forward, config)
                .map_err(|err| err.context("moe experts offloaded")),
        }
    }
}

// The gather-based forward requires `gather_forward` support from the quantized layer.
fn check_isq_gather_support() -> Result<()> {
    let Some(params) = inference_quant::get_immediate_isq() else {
        return Ok(());
    };
    if let Some(ty) = params.ty
        && matches!(
            ty,
            IsqType::HQQ4 | IsqType::HQQ8 | IsqType::F8E4M3 | IsqType::F8Q8
        )
    {
        inference_tensor::bail!("ISQ type {ty} is not supported for MoE experts.");
    }
    Ok(())
}

impl MoEExperts {
    /// Create MoEExperts for a standard model: experts live under `vb.pp("experts")`.
    pub fn new(
        cfg: &MoEExpertsConfig,
        vb: ShardedVarBuilder,
        layer_device: Device,
        comm: &Arc<inference_quant::Comm>,
        loading_isq: bool,
        quantization_config: &Option<QuantizedConfig>,
        act: Activation,
    ) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if let (Some(slots), Some(device)) =
            (vb.moe_expert_cache(), offload::cuda_device(&layer_device))
            && let Some(offloaded) = Self::load_offloaded(cfg, &vb, comm, slots, device, act)?
        {
            return Ok(Self::from_backend(
                MoEExpertsBackendImpl::Offloaded(offloaded),
                None,
                cfg,
                comm,
                act,
            ));
        }
        let experts_vb = vb.pp("experts").set_device(layer_device.clone());
        if let Some(fast) = FastExpertsWeights::from_weight_source(cfg, &experts_vb, comm)? {
            let lora_site = Self::register_lora_site(cfg, &experts_vb, comm, fast.sharded)?;
            return Ok(Self::from_backend(
                MoEExpertsBackendImpl::Fast(fast),
                lora_site,
                cfg,
                comm,
                act,
            ));
        }
        #[cfg(feature = "cutile")]
        if let Some(fp8) = CutileFp8ExpertsWeights::try_from_checkpoint(
            cfg,
            &experts_vb,
            comm,
            quantization_config,
            act,
        )? {
            let lora_site = Self::register_lora_site(cfg, &experts_vb, comm, false)?;
            return Ok(Self::from_backend(
                MoEExpertsBackendImpl::CutileFp8(fp8),
                lora_site,
                cfg,
                comm,
                act,
            ));
        }
        check_isq_gather_support()?;
        let choice = BackendChoice::new(
            layer_device,
            experts_vb.dtype(),
            loading_isq,
            quantization_config,
            act,
        );
        // only the stacked backends need a readable checkpoint; Fast detects with a dummy fallback
        let backend_impl = match MoEExpertsBackend::resolve(&choice)? {
            MoEExpertsBackend::Fused => MoEExpertsBackendImpl::Fused(FusedExpertsWeights {
                w: StackedExpertWeights::from_checkpoint(&ExpertCheckpoint::new(
                    cfg,
                    experts_vb.clone(),
                    comm,
                )?)?,
            }),
            #[cfg(feature = "cutile")]
            MoEExpertsBackend::Cutile => {
                MoEExpertsBackendImpl::Cutile(CutileExpertsWeights::from_checkpoint(
                    &ExpertCheckpoint::new(cfg, experts_vb.clone(), comm)?,
                )?)
            }
            #[cfg(feature = "cuda")]
            MoEExpertsBackend::Cutlass => {
                MoEExpertsBackendImpl::Cutlass(CutlassExpertsWeights::from_checkpoint(
                    &ExpertCheckpoint::new(cfg, experts_vb.clone(), comm)?,
                )?)
            }
            MoEExpertsBackend::Fast => {
                if experts_are_prequantized(quantization_config, &experts_vb) {
                    MoEExpertsBackendImpl::Fast(FastExpertsWeights::load_prequantized(
                        cfg,
                        vb,
                        quantization_config,
                    )?)
                } else {
                    MoEExpertsBackendImpl::Fast(FastExpertsWeights::load_unquantized(
                        cfg,
                        experts_vb.clone(),
                        comm,
                    )?)
                }
            }
        };
        let lora_sharded = match &backend_impl {
            MoEExpertsBackendImpl::Fast(weights) => weights.sharded,
            _ => comm.world_size() > 1,
        };
        let lora_site = Self::register_lora_site(cfg, &experts_vb, comm, lora_sharded)?;

        Ok(Self::from_backend(backend_impl, lora_site, cfg, comm, act))
    }

    /// Create MoEExperts from a VB already at the experts level (gemma4's flat `moe.*`, no
    /// `experts.*` sublevel).
    pub fn new_direct(
        cfg: &MoEExpertsConfig,
        experts_vb: ShardedVarBuilder,
        comm: &Arc<inference_quant::Comm>,
        loading_isq: bool,
        quantization_config: &Option<QuantizedConfig>,
        act: Activation,
    ) -> Result<Self> {
        if let Some(fast) = FastExpertsWeights::from_weight_source(cfg, &experts_vb, comm)? {
            let lora_site = Self::register_lora_site(cfg, &experts_vb, comm, fast.sharded)?;
            return Ok(Self::from_backend(
                MoEExpertsBackendImpl::Fast(fast),
                lora_site,
                cfg,
                comm,
                act,
            ));
        }
        check_isq_gather_support()?;
        let choice = BackendChoice::new(
            experts_vb.device().clone(),
            experts_vb.dtype(),
            loading_isq,
            quantization_config,
            act,
        );
        // only the stacked backends need a readable checkpoint; Fast detects with a dummy fallback
        let backend_impl = match MoEExpertsBackend::resolve(&choice)? {
            MoEExpertsBackend::Fused => MoEExpertsBackendImpl::Fused(FusedExpertsWeights {
                w: StackedExpertWeights::from_checkpoint(&ExpertCheckpoint::new(
                    cfg,
                    experts_vb.clone(),
                    comm,
                )?)?,
            }),
            #[cfg(feature = "cutile")]
            MoEExpertsBackend::Cutile => {
                MoEExpertsBackendImpl::Cutile(CutileExpertsWeights::from_checkpoint(
                    &ExpertCheckpoint::new(cfg, experts_vb.clone(), comm)?,
                )?)
            }
            #[cfg(feature = "cuda")]
            MoEExpertsBackend::Cutlass => {
                MoEExpertsBackendImpl::Cutlass(CutlassExpertsWeights::from_checkpoint(
                    &ExpertCheckpoint::new(cfg, experts_vb.clone(), comm)?,
                )?)
            }
            MoEExpertsBackend::Fast => {
                if experts_are_prequantized(quantization_config, &experts_vb) {
                    inference_tensor::bail!(
                        "Pre-quantized experts are not supported for flat expert trees."
                    );
                }
                MoEExpertsBackendImpl::Fast(FastExpertsWeights::load_unquantized(
                    cfg,
                    experts_vb.clone(),
                    comm,
                )?)
            }
        };
        let lora_sharded = match &backend_impl {
            MoEExpertsBackendImpl::Fast(weights) => weights.sharded,
            _ => comm.world_size() > 1,
        };
        let lora_site = Self::register_lora_site(cfg, &experts_vb, comm, lora_sharded)?;

        Ok(Self::from_backend(backend_impl, lora_site, cfg, comm, act))
    }

    // GGUF experts read on the host, kept there in pinned memory with a device cache of `slots` per projection
    #[cfg(feature = "cuda")]
    fn load_offloaded(
        cfg: &MoEExpertsConfig,
        vb: &ShardedVarBuilder,
        comm: &Arc<inference_quant::Comm>,
        slots: usize,
        device: &inference_tensor::CudaDevice,
        act: Activation,
    ) -> Result<Option<Box<offload::OffloadedExperts>>> {
        let Some(glu) = backends::glu_activation(act) else {
            return Ok(None);
        };
        if comm.world_size() > 1 {
            return Ok(None);
        }
        let host_vb = vb.pp("experts").set_device(Device::Cpu);
        // read in place from a GGUF mapping when the source holds the stacks there, sparing two copies of each
        if let Some(source) = host_vb.weight_source() {
            let keys = inference_quant::QuantizedExpertKeys::new(&host_vb.prefix());
            let mapped = [&keys.gate, &keys.up, &keys.down].map(|key| source.mapped_blocks(key));
            if let [Some(gate), Some(up), Some(down)] = mapped
                && [&gate, &up, &down].iter().all(|blocks| {
                    blocks.dims.len() == 3
                        && inference_quant::cpu_experts::supports(blocks.ty, blocks.dims[2])
                })
            {
                let stacks = [gate, up, down]
                    .map(|blocks| inference_quant::HostExpertStack::from_mapped(blocks, device));
                let [gate, up, down] = stacks;
                return offload::OffloadedExperts::new(
                    [gate?, up?, down?],
                    slots,
                    cfg.num_experts_per_tok,
                    device,
                    glu,
                )
                .map(|experts| Some(Box::new(experts)));
            }
        }
        let Some(fast) = FastExpertsWeights::from_weight_source(cfg, &host_vb, comm)? else {
            return Ok(None);
        };
        let mut stacks = Vec::with_capacity(3);
        for projection in [
            &fast.fused_gate_proj,
            &fast.fused_up_proj,
            &fast.fused_down_proj,
        ] {
            let Some(blocks) = projection.cpu_gguf_blocks() else {
                return Ok(None);
            };
            let Ok(dims) = <[usize; 3]>::try_from(blocks.dims) else {
                return Ok(None);
            };
            if !inference_quant::cpu_experts::supports(blocks.ty, dims[2]) {
                return Ok(None);
            }
            stacks.push(inference_quant::HostExpertStack::new(
                blocks.ty,
                dims,
                &blocks.bytes,
                device,
            )?);
        }
        let [gate, up, down]: [inference_quant::HostExpertStack; 3] = stacks
            .try_into()
            .map_err(|_| inference_tensor::Error::msg("three expert projections"))?;
        offload::OffloadedExperts::new(
            [gate, up, down],
            slots,
            cfg.num_experts_per_tok,
            device,
            glu,
        )
        .map(|experts| Some(Box::new(experts)))
    }

    fn from_backend(
        backend: MoEExpertsBackendImpl,
        lora_site: Option<Arc<LoraExpertSiteHandle>>,
        cfg: &MoEExpertsConfig,
        comm: &Arc<inference_quant::Comm>,
        act: Activation,
    ) -> Self {
        Self {
            backend,
            lora_site,
            act,
            num_experts: cfg.num_experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
            all_reduce: SumAllReduce::new(comm),
            world_size: comm.world_size(),
        }
    }

    fn register_lora_site(
        cfg: &MoEExpertsConfig,
        experts_vb: &ShardedVarBuilder,
        comm: &Arc<inference_quant::Comm>,
        sharded: bool,
    ) -> Result<Option<Arc<LoraExpertSiteHandle>>> {
        let Some(registry) = experts_vb.lora_registry() else {
            return Ok(None);
        };
        let names = cfg.expert_proj_names;
        let (gate_up_output_shard, down_input_shard) = if sharded {
            (
                Shard::Simple {
                    dim: 0,
                    rank: comm.rank(),
                    world_size: comm.world_size(),
                },
                Shard::Simple {
                    dim: 1,
                    rank: comm.rank(),
                    world_size: comm.world_size(),
                },
            )
        } else {
            (Shard::default(), Shard::default())
        };
        let spec = LoraExpertSiteSpec::new(
            cfg.num_experts,
            cfg.hidden_size,
            cfg.moe_intermediate_size,
            LoraExpertProjectionNames::new(names.gate, names.up, names.down),
            gate_up_output_shard,
            down_input_shard,
        )?;
        registry
            .register_expert(
                LoraSiteKey::new(experts_vb.prefix()),
                spec,
                experts_vb.dtype(),
                experts_vb.device().clone(),
            )
            .map(Some)
    }
    /// Forward pass through experts
    ///
    /// # Arguments
    /// * `xs` - Input tensor of shape [batch, seq_len, hidden_dim]
    /// * `topk_weights` - Top-k routing weights of shape [num_tokens, num_experts_per_tok]
    /// * `topk_ids` - Top-k expert indices of shape [num_tokens, num_experts_per_tok]
    ///
    /// # Returns
    /// Output tensor of shape [batch, seq_len, hidden_dim]
    pub fn forward(&self, xs: &Tensor, topk_weights: Tensor, topk_ids: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, hidden_dim) = xs.dims3()?;
        let shape = MoEForwardShape::new(batch_size, seq_len, hidden_dim);
        let xs_flat = xs.reshape(shape.flat())?;
        let lora = self
            .lora_site
            .as_ref()
            .map(LoraExpertExecution::current)
            .transpose()?
            .flatten();
        let forward = MoEForward {
            xs,
            xs_flat: &xs_flat,
            topk_weights: &topk_weights,
            topk_ids,
            original_dtype: xs.dtype(),
            shape,
            lora,
        };

        let config = self.forward_config();
        let mut ys = self
            .backend
            .forward(&forward, config)
            .map_err(|err| err.context("moe experts forward"))?;

        // Sharded experts produce partial sums; replicated ones are already complete.
        let sharded = match &self.backend {
            MoEExpertsBackendImpl::Fast(w) => w.sharded,
            _ => true,
        };
        if self.world_size > 1 && sharded {
            ys = self.all_reduce.sum_all_reduce(&ys)?;
        }

        ys.reshape(forward.shape.output())
    }

    fn forward_config(&self) -> MoEForwardConfig {
        MoEForwardConfig {
            num_experts: self.num_experts,
            num_experts_per_tok: self.num_experts_per_tok,
            act: self.act,
        }
    }
}
