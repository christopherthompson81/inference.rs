//! MoE experts kept in host memory: a per-layer VRAM cache of recently routed experts serves decode while the CPU
//! runs the misses, and prefill stages the whole layer on the GPU.

use std::sync::{Arc, Mutex};

use inference_quant::cpu_experts::{CpuExpertProjection, CpuExpertRoute, CpuExperts};
use inference_quant::{DeviceExpertStack, GluActivationType, HostExpertStack, KernelWeight};
use inference_tensor::cuda::cudarc::driver::{CudaEvent, CudaStream};
use inference_tensor::{CudaDevice, DType, Device, Result, Tensor};

use super::backends::{grouped_with, indexed_mmvq_with};
use super::forward::{MoEForward, MoEForwardConfig};

// Largest batch the cache and CPU split serves; past it the layer is staged whole on the GPU
const SPLIT_MAX_TOKENS: usize = 8;
// A slot left zeroed for the picks the CPU serves (weight zeroed too): a slot mid-copy could hold non-finite scales
const MISS_SLOT: u32 = 0;

/// Which projection of a layer, in `[gate, up, down]` order.
const PROJECTIONS: usize = 3;

/// Prefill staging every offloaded layer shares on a device: room for all experts of the largest layer.
struct Staging {
    stacks: [DeviceExpertStack; PROJECTIONS],
}

static STAGING: Mutex<Vec<(usize, Arc<Staging>)>> = Mutex::new(Vec::new());

fn staging(
    device: &CudaDevice,
    layouts: &[[(inference_quant::GgufType, usize, usize); PROJECTIONS]],
    experts: usize,
) -> Result<Arc<Staging>> {
    let mut all = STAGING.lock().expect("expert staging poisoned");
    let ordinal = device.cuda_stream().context().ordinal();
    if let Some((_, staging)) = all.iter().find(|(o, _)| *o == ordinal) {
        return Ok(staging.clone());
    }
    let stack = |p: usize| {
        let layouts = layouts.iter().map(|l| l[p]).collect::<Vec<_>>();
        DeviceExpertStack::zeros(&layouts, experts, device)
    };
    let staging = Arc::new(Staging {
        stacks: [stack(0)?, stack(1)?, stack(2)?],
    });
    all.push((ordinal, staging.clone()));
    Ok(staging)
}

// Every offloaded layer's projection layouts, so the shared staging fits the largest
static LAYOUTS: Mutex<Vec<[(inference_quant::GgufType, usize, usize); PROJECTIONS]>> =
    Mutex::new(Vec::new());

struct CacheState {
    // expert -> slot whose copy has landed
    slot_of: Vec<Option<usize>>,
    // slot -> expert it holds or is receiving
    expert_in: Vec<Option<usize>>,
    last_use: Vec<u64>,
    pending: Vec<(usize, CudaEvent)>,
    clock: u64,
}

impl CacheState {
    fn new(experts: usize, slots: usize) -> Self {
        Self {
            slot_of: vec![None; experts],
            expert_in: vec![None; slots],
            last_use: vec![0; slots],
            pending: Vec::new(),
            clock: 0,
        }
    }

    // copies that have landed make their slots usable
    fn retire(&mut self) {
        let expert_in = &self.expert_in;
        let slot_of = &mut self.slot_of;
        self.pending.retain(|(slot, event)| {
            if event.is_complete() {
                if let Some(expert) = expert_in[*slot] {
                    slot_of[expert] = Some(*slot);
                }
                false
            } else {
                true
            }
        });
    }

    // the least recently used slot no in-flight copy targets and this forward does not read
    fn victim(&self, in_use: &[bool]) -> Option<usize> {
        (0..self.expert_in.len())
            .filter(|&s| {
                s != MISS_SLOT as usize && !in_use[s] && !self.pending.iter().any(|(p, _)| *p == s)
            })
            .min_by_key(|&s| (self.expert_in[s].is_some(), self.last_use[s]))
    }
}

pub(super) struct OffloadedExperts {
    host: [HostExpertStack; PROJECTIONS],
    cache: [DeviceExpertStack; PROJECTIONS],
    state: Mutex<CacheState>,
    copy_stream: Arc<CudaStream>,
    device: CudaDevice,
    act: GluActivationType,
}

impl OffloadedExperts {
    /// `host` holds `[gate, up, down]`; `slots` experts of each stay cached on `device`.
    pub(super) fn new(
        host: [HostExpertStack; PROJECTIONS],
        slots: usize,
        device: &CudaDevice,
        act: GluActivationType,
    ) -> Result<Self> {
        let experts = host[0].dims()[0];
        let layout = |h: &HostExpertStack| {
            let [_, rows, cols] = h.dims();
            (h.ty(), rows, cols)
        };
        let layouts = [layout(&host[0]), layout(&host[1]), layout(&host[2])];
        LAYOUTS
            .lock()
            .expect("expert layouts poisoned")
            .push(layouts);
        let slots = slots.min(experts) + 1;
        crate::perf_flags::rule_out_cuda_graphs();
        let cache = |p: usize| DeviceExpertStack::zeros(&[layouts[p]], slots, device);
        Ok(Self {
            cache: [cache(0)?, cache(1)?, cache(2)?],
            host,
            state: Mutex::new(CacheState::new(experts, slots)),
            copy_stream: device
                .cuda_stream()
                .context()
                .new_stream()
                .map_err(inference_tensor::Error::wrap)?,
            device: device.clone(),
            act,
        })
    }

    fn experts(&self) -> usize {
        self.host[0].dims()[0]
    }

    pub(super) fn forward(&self, forward: &MoEForward, config: MoEForwardConfig) -> Result<Tensor> {
        if forward.lora.is_some() {
            inference_tensor::bail!("offloaded MoE experts do not take LoRA adapters");
        }
        if forward.shape.num_tokens <= SPLIT_MAX_TOKENS {
            self.forward_split(forward, config)
        } else {
            self.forward_staged(forward, config)
        }
    }

    fn cpu_experts(&self) -> Result<CpuExperts<'_>> {
        fn projection(h: &HostExpertStack) -> Result<CpuExpertProjection<'_>> {
            let [_, rows, cols] = h.dims();
            Ok(CpuExpertProjection {
                ty: h.ty(),
                bytes: h.bytes(),
                rows,
                cols,
            })
        }
        Ok(CpuExperts {
            gate: projection(&self.host[0])?,
            up: projection(&self.host[1])?,
            down: projection(&self.host[2])?,
            act: self.act,
        })
    }

    // cached experts on the GPU and the rest on the CPU at the same time, then the misses are copied in
    fn forward_split(&self, forward: &MoEForward, config: MoEForwardConfig) -> Result<Tensor> {
        let topk = config.num_experts_per_tok;
        let (tokens, hidden) = forward.shape.flat();
        let ids = forward
            .topk_ids
            .to_dtype(DType::U32)?
            .flatten_all()?
            .to_vec1::<u32>()?;
        let weights = forward
            .topk_weights
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        let mut state = self.state.lock().expect("expert cache poisoned");
        state.retire();
        state.clock += 1;
        let clock = state.clock;
        let mut slot_ids = vec![MISS_SLOT; ids.len()];
        let mut hit_weights = vec![0f32; ids.len()];
        let mut in_use = vec![false; state.expert_in.len()];
        let mut misses = Vec::new();
        for (i, (&expert, &weight)) in ids.iter().zip(&weights).enumerate() {
            match state.slot_of[expert as usize] {
                Some(slot) => {
                    slot_ids[i] = u32::try_from(slot).expect("slot ids fit u32");
                    hit_weights[i] = weight;
                    in_use[slot] = true;
                    state.last_use[slot] = clock;
                }
                None => misses.push(CpuExpertRoute {
                    token: i / topk,
                    expert: expert as usize,
                    weight,
                }),
            }
        }

        let gpu = if misses.len() < ids.len() {
            let device = forward.xs.device();
            let slot_ids = Tensor::from_vec(slot_ids, (tokens, topk), device)?;
            let hit_weights = Tensor::from_vec(hit_weights, (tokens, topk), device)?;
            let hits = MoEForward {
                xs: forward.xs,
                xs_flat: forward.xs_flat,
                topk_weights: &hit_weights,
                topk_ids: &slot_ids,
                original_dtype: forward.original_dtype,
                shape: forward.shape,
                lora: None,
            };
            let cache = [
                &self.cache[0] as &dyn KernelWeight,
                &self.cache[1],
                &self.cache[2],
            ];
            let hit_config = MoEForwardConfig {
                num_experts: self.cache[0].slots(),
                ..config
            };
            let routed = if tokens <= SPLIT_MAX_TOKENS {
                indexed_mmvq_with(cache, &hits, hit_config)?
            } else {
                grouped_with(cache, &hits, hit_config)?
            };
            Some(routed.ok_or_else(|| {
                inference_tensor::Error::msg("cached experts cannot run the GGUF kernels")
            })?)
        } else {
            None
        };

        let cpu = if misses.is_empty() {
            None
        } else {
            let xs = forward
                .xs_flat
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let mut out = vec![0f32; tokens * hidden];
            self.cpu_experts()?.forward(&xs, &misses, &mut out)?;
            Some(
                Tensor::from_vec(out, (tokens, hidden), forward.xs.device())?
                    .to_dtype(forward.original_dtype)?,
            )
        };

        self.admit(&mut state, &misses, &mut in_use)?;
        drop(state);
        match (gpu, cpu) {
            (Some(gpu), Some(cpu)) => gpu + cpu,
            (Some(y), None) | (None, Some(y)) => Ok(y),
            (None, None) => unreachable!("every pick is a hit or a miss"),
        }
    }

    // each missed expert takes the least recently used slot, copied on the side stream after the work queued so far
    fn admit(
        &self,
        state: &mut CacheState,
        misses: &[CpuExpertRoute],
        in_use: &mut [bool],
    ) -> Result<()> {
        let mut experts = misses.iter().map(|m| m.expert).collect::<Vec<_>>();
        experts.sort_unstable();
        experts.dedup();
        if experts.is_empty() {
            return Ok(());
        }
        let main = self.device.cuda_stream();
        let queued = main
            .record_event(None)
            .map_err(inference_tensor::Error::wrap)?;
        self.copy_stream
            .wait(&queued)
            .map_err(inference_tensor::Error::wrap)?;
        for expert in experts {
            if state.expert_in.contains(&Some(expert)) {
                continue;
            }
            let Some(slot) = state.victim(in_use) else {
                break;
            };
            if let Some(old) = state.expert_in[slot].take() {
                state.slot_of[old] = None;
            }
            for (host, cache) in self.host.iter().zip(&self.cache) {
                let size = host.expert_bytes();
                let src = &host.bytes()[expert * size..][..size];
                // SAFETY: host experts live as long as the layer, and the slot is unread until its event completes
                unsafe { cache.fill_async(slot, src, &self.copy_stream)? };
            }
            let landed = self
                .copy_stream
                .record_event(None)
                .map_err(inference_tensor::Error::wrap)?;
            state.expert_in[slot] = Some(expert);
            state.last_use[slot] = state.clock;
            state.pending.push((slot, landed));
            in_use[slot] = true;
        }
        Ok(())
    }

    // the whole layer copied into the shared staging, then the grouped kernels as for a resident layer
    fn forward_staged(&self, forward: &MoEForward, config: MoEForwardConfig) -> Result<Tensor> {
        let experts = self.experts();
        let staging = {
            let layouts = LAYOUTS.lock().expect("expert layouts poisoned").clone();
            staging(&self.device, &layouts, experts)?
        };
        let main = self.device.cuda_stream();
        let mut views = Vec::with_capacity(PROJECTIONS);
        for (host, stack) in self.host.iter().zip(&staging.stacks) {
            let [_, rows, cols] = host.dims();
            let view = stack.view(host.ty(), experts, rows, cols)?;
            // SAFETY: host experts live as long as the layer; the main stream orders this copy after the previous
            // layer's reads of the staging and before this layer's
            unsafe { view.fill_async(0, host.bytes(), &main)? };
            views.push(view);
        }
        let weights = [&views[0] as &dyn KernelWeight, &views[1], &views[2]];
        grouped_with(weights, forward, config)?.ok_or_else(|| {
            inference_tensor::Error::msg("staged experts cannot run the grouped kernels")
        })
    }
}

/// The CUDA device a layer runs on, when it has one.
pub(super) fn cuda_device(device: &Device) -> Option<&CudaDevice> {
    match device {
        Device::Cuda(device) => Some(device),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use inference_tensor::quantized::{GgmlDType, QTensor};

    use super::*;
    use crate::layers::Activation;
    use crate::moe::experts::forward::MoEForwardShape;

    const EXPERTS: usize = 8;
    const HIDDEN: usize = 256;
    const INTER: usize = 256;
    const TOPK: usize = 2;
    const SLOTS: usize = 3;
    const PROMPT: usize = 40;

    fn patterned(dims: &[usize], salt: usize, scale: f32) -> Result<Tensor> {
        let len = dims.iter().product::<usize>();
        let values = (0..len)
            .map(|i| {
                let step = u16::try_from((i * 37 + salt * 17) % 257).expect("below 257");
                (f32::from(step) / 128.0 - 1.0) * scale
            })
            .collect::<Vec<_>>();
        Tensor::from_vec(values, dims, &Device::Cpu)
    }

    fn q4k_bytes(rows: usize, cols: usize, salt: usize) -> Result<Vec<u8>> {
        let weights = patterned(&[EXPERTS, rows, cols], salt, 0.1)?;
        Ok(QTensor::quantize(&weights, GgmlDType::Q4K)?
            .data()?
            .into_owned())
    }

    fn relative_error(actual: &Tensor, expected: &Tensor) -> Result<f32> {
        let err = (actual - expected)?
            .sqr()?
            .sum_all()?
            .sqrt()?
            .to_scalar::<f32>()?;
        let norm = expected.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        Ok(err / norm)
    }

    struct Run<'a> {
        xs: Tensor,
        xs_flat: Tensor,
        ids: Tensor,
        weights: Tensor,
        shape: MoEForwardShape,
        _device: &'a Device,
    }

    impl Run<'_> {
        fn forward(&self) -> MoEForward<'_> {
            MoEForward {
                xs: &self.xs,
                xs_flat: &self.xs_flat,
                topk_weights: &self.weights,
                topk_ids: &self.ids,
                original_dtype: DType::F32,
                shape: self.shape,
                lora: None,
            }
        }
    }

    fn run(device: &Device, tokens: usize, step: usize) -> Result<Run<'_>> {
        let xs = patterned(&[1, tokens, HIDDEN], step + 5, 1.0)?.to_device(device)?;
        let ids = (0..tokens * TOPK)
            .map(|i| u32::try_from((i * 3 + step * 5) % EXPERTS).expect("few experts"))
            .collect::<Vec<_>>();
        let weights = (0..tokens * TOPK)
            .map(|i| if i % 2 == 0 { 0.7 } else { 0.3 })
            .collect::<Vec<f32>>();
        Ok(Run {
            xs_flat: xs.reshape((tokens, HIDDEN))?,
            xs,
            ids: Tensor::from_vec(ids, (tokens, TOPK), device)?,
            weights: Tensor::from_vec(weights, (tokens, TOPK), device)?,
            shape: MoEForwardShape::new(1, tokens, HIDDEN),
            _device: device,
        })
    }

    // the cache, its misses on the CPU and the staged prompt path against every expert resident on the device
    #[test]
    fn offloaded_experts_match_resident_ones() -> Result<()> {
        crate::skip_without_cuda!();
        let device = Device::new_cuda(0)?;
        let cuda = device.as_cuda_device()?;
        let bytes = [
            q4k_bytes(INTER, HIDDEN, 1)?,
            q4k_bytes(INTER, HIDDEN, 2)?,
            q4k_bytes(HIDDEN, INTER, 3)?,
        ];
        let dims = [
            [EXPERTS, INTER, HIDDEN],
            [EXPERTS, INTER, HIDDEN],
            [EXPERTS, HIDDEN, INTER],
        ];
        let host = |p: usize| {
            HostExpertStack::new(inference_quant::GgufType::Q4K, dims[p], &bytes[p], cuda)
        };
        let offloaded = OffloadedExperts::new(
            [host(0)?, host(1)?, host(2)?],
            SLOTS,
            cuda,
            GluActivationType::Silu,
        )?;
        let resident = (0..PROJECTIONS)
            .map(|p| {
                let [_, rows, cols] = dims[p];
                let stack = DeviceExpertStack::zeros(
                    &[(inference_quant::GgufType::Q4K, rows, cols)],
                    EXPERTS,
                    cuda,
                )?;
                // SAFETY: `bytes` outlives the test, and the synchronize below lands the copy before any read
                unsafe { stack.fill_async(0, &bytes[p], &cuda.cuda_stream())? };
                Ok(stack)
            })
            .collect::<Result<Vec<_>>>()?;
        cuda.cuda_stream()
            .synchronize()
            .map_err(inference_tensor::Error::wrap)?;
        let resident = [
            &resident[0] as &dyn KernelWeight,
            &resident[1],
            &resident[2],
        ];
        let config = MoEForwardConfig {
            num_experts: EXPERTS,
            num_experts_per_tok: TOPK,
            act: Activation::Silu,
        };

        // later steps hit what earlier ones copied in, and evict as the routes move on
        for step in 0..12 {
            let run = run(&device, 2, step)?;
            let actual = offloaded.forward(&run.forward(), config)?;
            let expected = indexed_mmvq_with(resident, &run.forward(), config)?
                .expect("Q4_K runs the indexed matvec");
            let err = relative_error(&actual, &expected)?;
            // the CPU quantizes activations to Q8_K, the GPU kernels to Q8_1
            assert!(err < 0.03, "step {step}: relative error {err}");
            cuda.cuda_stream()
                .synchronize()
                .map_err(inference_tensor::Error::wrap)?;
        }

        let run = run(&device, PROMPT, 0)?;
        let actual = offloaded.forward(&run.forward(), config)?;
        let expected =
            grouped_with(resident, &run.forward(), config)?.expect("Q4_K runs the grouped mmq");
        assert!(
            relative_error(&actual, &expected)? < 1e-5,
            "the staged prompt path runs the same kernels"
        );
        Ok(())
    }
}
