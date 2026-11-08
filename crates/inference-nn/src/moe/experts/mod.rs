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
mod ungated;
#[cfg(feature = "cuda")]
pub use offload::{SPLIT_MAX_TOKENS as OFFLOADED_GRAPH_MAX_TOKENS, offload_activity};

/// Experts an offloaded layer's prompt stages on the device at once: its uncached experts run in groups of this many,
/// so the staging takes a group's room rather than a whole layer's.
pub const OFFLOAD_STAGING_EXPERTS: usize = 64;

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
    MappedCpu(Box<MappedCpuExperts>),
    Vulkan(Box<VulkanExperts>),
    #[cfg(feature = "cuda")]
    Offloaded(Box<offload::OffloadedExperts>),
    Ungated(ungated::UngatedExpertsWeights),
}

/// Resident CPU experts read in place from a GGUF mapping: the block kernels over the file's own bytes, so loading
/// copies nothing and the pages stay reclaimable.
struct MappedCpuExperts {
    // gate, up, down; each `[experts, rows, cols]`
    stacks: [inference_quant::MappedBlocks; 3],
    act: inference_quant::GluActivationType,
}

impl MappedCpuExperts {
    fn forward(&self, forward: &MoEForward, config: MoEForwardConfig) -> Result<Tensor> {
        let [gate, up, down] = self.stacks.each_ref().map(|blocks| {
            inference_quant::cpu_experts::CpuExpertProjection {
                ty: blocks.ty,
                bytes: blocks.bytes(),
                rows: blocks.dims[1],
                cols: blocks.dims[2],
            }
        });
        let experts = inference_quant::cpu_experts::CpuExperts {
            gate: inference_quant::cpu_experts::CpuExpertGate::Glu {
                gate,
                act: self.act,
            },
            up,
            down,
        };
        backends::run_cpu_experts(&experts, forward, config, None)
    }
}

/// GGUF experts in a Vulkan device's memory, their blocks copied once: gate and up through the expert matvec, the
/// activation, then down.
struct VulkanExperts {
    // gate, up, down; each `(experts, rows, cols)`
    stacks: [inference_tensor::vulkan::BlockWeight; 3],
}

impl VulkanExperts {
    fn forward(&self, forward: &MoEForward, config: MoEForwardConfig) -> Result<Tensor> {
        let [gate, up, down] = &self.stacks;
        let (tokens, k) = (forward.shape.num_tokens, config.num_experts_per_tok);
        let xs = forward
            .xs_flat
            .to_dtype(inference_tensor::DType::F32)?
            .contiguous()?;
        let ids = forward
            .topk_ids
            .to_dtype(inference_tensor::DType::U32)?
            .reshape((tokens, k))?
            .contiguous()?;
        let hidden = (inference_tensor::nn::ops::silu(&gate.expert_mat_vec(&xs, &ids, k)?)?
            * up.expert_mat_vec(&xs, &ids, k)?)?;
        let ys = down.expert_mat_vec(&hidden, &ids, 1)?;
        let weights = forward
            .topk_weights
            .to_dtype(inference_tensor::DType::F32)?
            .reshape((tokens, 1, k))?;
        weights
            .matmul(&ys.reshape((tokens, k, ys.dim(1)?))?)?
            .reshape(forward.shape.flat())?
            .to_dtype(forward.original_dtype)
    }
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
            Self::MappedCpu(w) => w
                .forward(forward, config)
                .map_err(|err| err.context("moe experts mapped cpu")),
            Self::Vulkan(w) => w
                .forward(forward, config)
                .map_err(|err| err.context("moe experts vulkan")),
            #[cfg(feature = "cuda")]
            Self::Offloaded(w) => w
                .forward(forward, config, None)
                .map(|(ys, _)| ys)
                .map_err(|err| err.context("moe experts offloaded")),
            Self::Ungated(w) => w
                .forward_impl(forward, config)
                .map_err(|err| err.context("moe experts ungated")),
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
        if !cfg.gated {
            #[cfg(feature = "cuda")]
            if let (Some(slots), Some(device)) =
                (vb.moe_expert_cache(), offload::cuda_device(&layer_device))
                && let Some(offloaded) =
                    Self::load_offloaded_ungated(cfg, &vb, comm, slots, device, act)?
            {
                return Ok(Self::from_backend(
                    MoEExpertsBackendImpl::Offloaded(offloaded),
                    None,
                    cfg,
                    comm,
                    act,
                ));
            }
            let experts_vb = vb.pp("experts").set_device(layer_device);
            return Self::new_ungated(cfg, experts_vb, comm, act);
        }
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
        if layer_device.is_vulkan()
            && let Some(vulkan) = Self::load_vulkan(&experts_vb, comm, act, &layer_device)?
        {
            return Ok(Self::from_backend(
                MoEExpertsBackendImpl::Vulkan(Box::new(vulkan)),
                None,
                cfg,
                comm,
                act,
            ));
        }
        // Vulkan's experts of a type its kernels do not take run on the CPU's kernels from host memory
        if (layer_device.is_cpu() || layer_device.is_vulkan())
            && let Some(mapped) = Self::load_mapped_cpu(&experts_vb, comm, act)
        {
            return Ok(Self::from_backend(
                MoEExpertsBackendImpl::MappedCpu(Box::new(mapped)),
                None,
                cfg,
                comm,
                act,
            ));
        }
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
        if !cfg.gated {
            return Self::new_ungated(cfg, experts_vb, comm, act);
        }
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

    fn new_ungated(
        cfg: &MoEExpertsConfig,
        experts_vb: ShardedVarBuilder,
        comm: &Arc<inference_quant::Comm>,
        act: Activation,
    ) -> Result<Self> {
        let weights = ungated::UngatedExpertsWeights::load(cfg, experts_vb, comm)?;
        Ok(Self::from_backend(
            MoEExpertsBackendImpl::Ungated(weights),
            None,
            cfg,
            comm,
            act,
        ))
    }

    // GGUF stacks the block kernels run, left in the mapping; not when ISQ, LoRA or sharding needs layers of its own
    // SiLU-gated GGUF experts of types Vulkan's kernels take, copied to the device from their mapping
    fn load_vulkan(
        experts_vb: &ShardedVarBuilder,
        comm: &Arc<inference_quant::Comm>,
        act: Activation,
        device: &Device,
    ) -> Result<Option<VulkanExperts>> {
        if !matches!(
            backends::glu_activation(act),
            Some(inference_quant::GluActivationType::Silu)
        ) || comm.world_size() > 1
            || inference_quant::get_immediate_isq().is_some()
            || experts_vb.lora_registry().is_some()
        {
            return Ok(None);
        }
        let Some(source) = experts_vb.weight_source() else {
            return Ok(None);
        };
        let keys = inference_quant::QuantizedExpertKeys::new(&experts_vb.prefix());
        let [Some(gate), Some(up), Some(down)] = [&keys.gate, &keys.up, &keys.down].map(|key| {
            source
                .mapped_blocks(key)
                .filter(|blocks| blocks.dims.len() == 3)
                .and_then(|blocks| Some((blocks.ty.vulkan_block_kind()?, blocks)))
        }) else {
            return Ok(None);
        };
        let upload = |(kind, blocks): (
            inference_tensor::vulkan::BlockKind,
            inference_quant::MappedBlocks,
        )| {
            inference_tensor::vulkan::BlockWeight::new(
                device,
                kind,
                (blocks.dims[0], blocks.dims[1], blocks.dims[2]),
                blocks.bytes(),
            )
        };
        Ok(Some(VulkanExperts {
            stacks: [upload(gate)?, upload(up)?, upload(down)?],
        }))
    }

    fn load_mapped_cpu(
        experts_vb: &ShardedVarBuilder,
        comm: &Arc<inference_quant::Comm>,
        act: Activation,
    ) -> Option<MappedCpuExperts> {
        let act = backends::glu_activation(act)?;
        if comm.world_size() > 1
            || inference_quant::get_immediate_isq().is_some()
            || experts_vb.lora_registry().is_some()
        {
            return None;
        }
        let source = experts_vb.weight_source()?;
        let keys = inference_quant::QuantizedExpertKeys::new(&experts_vb.prefix());
        let [gate, up, down] = [&keys.gate, &keys.up, &keys.down].map(|key| {
            source.mapped_blocks(key).filter(|blocks| {
                blocks.dims.len() == 3
                    && inference_quant::cpu_experts::supports(blocks.ty, blocks.dims[2])
            })
        });
        Some(MappedCpuExperts {
            stacks: [gate?, up?, down?],
            act,
        })
    }

    // ungated (squared ReLU) GGUF experts left in their mapping, with a device cache of `slots` per projection
    #[cfg(feature = "cuda")]
    fn load_offloaded_ungated(
        cfg: &MoEExpertsConfig,
        vb: &ShardedVarBuilder,
        comm: &Arc<inference_quant::Comm>,
        slots: usize,
        device: &inference_tensor::CudaDevice,
        act: Activation,
    ) -> Result<Option<Box<offload::OffloadedExperts>>> {
        if act != Activation::Relu2 || comm.world_size() > 1 {
            return Ok(None);
        }
        let host_vb = vb.pp("experts").set_device(Device::Cpu);
        let Some(source) = host_vb.weight_source() else {
            return Ok(None);
        };
        let keys = inference_quant::QuantizedExpertKeys::new(&host_vb.prefix());
        let [Some(up), Some(down)] = [&keys.up, &keys.down].map(|key| {
            source.mapped_blocks(key).filter(|blocks| {
                blocks.dims.len() == 3
                    && inference_quant::cpu_experts::supports(blocks.ty, blocks.dims[2])
            })
        }) else {
            return Ok(None);
        };
        let [up, down] =
            [up, down].map(|blocks| inference_quant::HostExpertStack::from_mapped(blocks, device));
        offload::OffloadedExperts::new(
            vec![up?, down?],
            slots,
            cfg.num_experts_per_tok,
            device,
            offload::OffloadGate::SquaredRelu,
        )
        .map(|experts| Some(Box::new(experts)))
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
                    vec![gate?, up?, down?],
                    slots,
                    cfg.num_experts_per_tok,
                    device,
                    offload::OffloadGate::Glu(glu),
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
            vec![gate, up, down],
            slots,
            cfg.num_experts_per_tok,
            device,
            offload::OffloadGate::Glu(glu),
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

    /// As [`Self::forward`], also returning `overlap`'s output: with experts offloaded it runs on the GPU while the CPU
    /// computes the step's missed experts, otherwise after the experts.
    pub fn forward_overlapping(
        &self,
        xs: &Tensor,
        topk_weights: Tensor,
        topk_ids: &Tensor,
        overlap: &mut dyn FnMut() -> Result<Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        #[cfg(feature = "cuda")]
        if let MoEExpertsBackendImpl::Offloaded(offloaded) = &self.backend
            && self.world_size == 1
        {
            let (batch_size, seq_len, hidden_dim) = xs.dims3()?;
            let shape = MoEForwardShape::new(batch_size, seq_len, hidden_dim);
            let xs_flat = xs.reshape(shape.flat())?;
            let forward = MoEForward {
                xs,
                xs_flat: &xs_flat,
                topk_weights: &topk_weights,
                topk_ids,
                original_dtype: xs.dtype(),
                shape,
                lora: None,
            };
            let (ys, side) = offloaded
                .forward(&forward, self.forward_config(), Some(overlap))
                .map_err(|err| err.context("moe experts offloaded"))?;
            let side = side.expect("an overlapping forward runs its overlap");
            return Ok((ys.reshape(shape.output())?, side));
        }
        let ys = self.forward(xs, topk_weights, topk_ids)?;
        Ok((ys, overlap()?))
    }

    /// Whether a decode step's experts run without reading their routes on the host, as a CUDA graph capture needs.
    pub fn decode_is_device_only(&self) -> bool {
        match &self.backend {
            MoEExpertsBackendImpl::Ungated(w) => w.decode_is_device_only(),
            _ => true,
        }
    }

    /// Whether the experts live in host memory, a cache of them on the GPU.
    pub fn is_offloaded(&self) -> bool {
        #[cfg(feature = "cuda")]
        if matches!(self.backend, MoEExpertsBackendImpl::Offloaded(_)) {
            return true;
        }
        false
    }

    fn forward_config(&self) -> MoEForwardConfig {
        MoEForwardConfig {
            num_experts: self.num_experts,
            num_experts_per_tok: self.num_experts_per_tok,
            act: self.act,
        }
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod vulkan_block_tests {
    use inference_quant::GgufType;
    use inference_tensor::vulkan::{BlockKind, BlockWeight};
    use inference_tensor::{Device, Result, Tensor};

    const EXPERTS: usize = 3;
    // not a multiple of the kernels' rows a workgroup, so a partial last group runs
    const ROWS: usize = 37;
    const COLS: usize = 512;
    // against the f32 product of the dequantized rows: the activations are int8 blocks on the device
    const RELATIVE_TOLERANCE: f32 = 2e-2;

    // random blocks with each block's leading f16 scale small and finite
    fn blocks(ty: GgufType, rows: usize, seed: u64) -> Vec<u8> {
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut bytes: Vec<u8> = (0..rows * ty.row_bytes(COLS).expect("whole blocks"))
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 56) as u8
            })
            .collect();
        for block in bytes.chunks_exact_mut(ty.type_size()) {
            let d = half::f16::from_f32(0.002 + f32::from(block[2]) * 1e-5);
            block[..2].copy_from_slice(&d.to_le_bytes());
        }
        bytes
    }

    fn assert_close(got: &Tensor, want: &Tensor, what: &str) -> Result<()> {
        let got = got.to_device(&Device::Cpu)?;
        let scale = want.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        let diff = (got - want)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(
            diff <= RELATIVE_TOLERANCE * scale,
            "{what}: {diff} of {scale}"
        );
        Ok(())
    }

    // each i-quant's blocks through the expert matvec (decode's few routes and a prompt's many) and the dense matmul
    // (a step's rows and a prompt's), against ggml's reference dequantization
    #[test]
    fn vulkan_iq_block_weights_match_their_dequantized_rows() -> Result<()> {
        let Ok(vk) = Device::new_vulkan(0) else {
            return Ok(());
        };
        let cpu = Device::Cpu;
        for (ty, kind) in [
            (GgufType::Iq4Xs, BlockKind::Iq4Xs),
            (GgufType::Iq3S, BlockKind::Iq3S),
        ] {
            let bytes = blocks(ty, EXPERTS * ROWS, 7);
            let w = Tensor::from_vec(
                inference_quant::dequantize_rows(ty, COLS, &bytes)?,
                (EXPERTS, ROWS, COLS),
                &cpu,
            )?;
            let stack = BlockWeight::new(&vk, kind, (EXPERTS, ROWS, COLS), &bytes)?;
            for tokens in [3usize, 200] {
                let k = 2;
                let x = Tensor::randn(0f32, 1.0, (tokens, COLS), &cpu)?;
                let ids: Vec<u32> = (0..tokens * k)
                    .map(|s| ((s * 7 + s / k) % EXPERTS) as u32)
                    .collect();
                let id_t = Tensor::from_vec(ids.clone(), (tokens, k), &cpu)?;
                let got = stack.expert_mat_vec(&x.to_device(&vk)?, &id_t.to_device(&vk)?, k)?;
                let want = Tensor::cat(
                    &ids.iter()
                        .enumerate()
                        .map(|(slot, &e)| {
                            x.narrow(0, slot / k, 1)?.matmul(&w.get(e as usize)?.t()?)
                        })
                        .collect::<Result<Vec<_>>>()?,
                    0,
                )?;
                assert_close(&got, &want, &format!("{ty:?} experts, {tokens} tokens"))?;
            }
            let dense = BlockWeight::new(
                &vk,
                kind,
                (1, ROWS, COLS),
                &bytes[..ROWS * ty.row_bytes(COLS).expect("rows")],
            )?;
            for tokens in [3usize, 300] {
                let x = Tensor::randn(0f32, 1.0, (tokens, COLS), &cpu)?;
                let got = dense.matmul(&x.to_device(&vk)?)?;
                assert_close(
                    &got,
                    &x.matmul(&w.get(0)?.t()?)?,
                    &format!("{ty:?} dense, {tokens} rows"),
                )?;
            }
        }
        Ok(())
    }

    // a raw GGUF i-quant weight moved to Vulkan: its matmul (a step's rows and a prompt's) and its rows at ids
    #[test]
    fn vulkan_raw_gguf_iq_weights_match_the_cpu() -> Result<()> {
        use inference_quant::QuantMethod;
        let Ok(vk) = Device::new_vulkan(0) else {
            return Ok(());
        };
        let cpu = Device::Cpu;
        for ty in [GgufType::Iq4Xs, GgufType::Iq3S] {
            let bytes = blocks(ty, ROWS, 11);
            let w = Tensor::from_vec(
                inference_quant::dequantize_rows(ty, COLS, &bytes)?,
                (ROWS, COLS),
                &cpu,
            )?;
            let raw = inference_quant::RawGgufTensor::new(ty, &[ROWS, COLS], bytes, &vk)?;
            let layer = inference_quant::GgufRawMatMul::new(raw.clone(), None);
            for tokens in [2usize, 300] {
                let x = Tensor::randn(0f32, 1.0, (tokens, COLS), &cpu)?;
                let got = layer.forward_raw(&x.to_device(&vk)?)?;
                assert_close(
                    &got,
                    &x.matmul(&w.t()?)?,
                    &format!("{ty:?} raw, {tokens} rows"),
                )?;
            }
            let ids = Tensor::new(&[5u32, 0, 36, 5], &cpu)?;
            let got = raw.embedding(&ids.to_device(&vk)?)?;
            let diff = (got.to_device(&cpu)? - w.index_select(&ids, 0)?)?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_scalar::<f32>()?;
            assert!(diff < 1e-5, "{ty:?} rows: {diff}");
        }
        Ok(())
    }
}
