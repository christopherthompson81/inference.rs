//! MoE experts kept in host memory: a per-layer VRAM cache of recently routed experts serves decode while the CPU
//! runs the misses, and prefill stages the experts each batch routes to on the GPU.
//!
//! A decode step never waits on the host, so it can be captured in a CUDA graph: the GPU looks each pick's slot up in a
//! table in mapped host memory (0, the zeroed slot, for a miss), writes the picks to mapped buffers and rings the
//! layer's doorbell, runs the cached experts, and spins until a host thread polling the doorbells has run the misses on
//! the CPU; the two outputs are added. A host function in their place costs a step ~2.5% on Qwen3.8-Flash-Next: its
//! dispatch waits behind the fills' landing callbacks on the driver's one callback thread (qwen4exp investigation Run
//! 55). Cache fills run on a worker thread and a copy stream, and the table opens a slot once its copy has landed.

use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use inference_quant::cpu_experts::{CpuExpertProjection, CpuExpertRoute, CpuExperts};
use inference_quant::{
    BounceRing, DeviceExpertStack, GluActivationType, HostExpertStack, KernelWeight,
};
use inference_tensor::cuda::cudarc::driver::{CudaSlice, CudaStream, DevicePtr, result};

use crate::cuda::mapped::{mapped_buffer, mapped_host, tensor_ptr};
use inference_tensor::{CudaDevice, DType, Device, Result, Tensor};

use super::OFFLOAD_STAGING_EXPERTS;
use super::backends::{grouped_parked_with, indexed_mmvq_with};
use super::forward::{MoEForward, MoEForwardConfig};

// Largest batch the cache and CPU split serves (the indexed matvec's limit); a larger one stages the experts it routes
// to on the GPU, since copying an expert costs about what the CPU takes for one token through it
const SPLIT_MAX_TOKENS: usize = 8;
// Copies a layer has in flight before its misses stop being admitted: a speculative verify misses far more experts than
// the copy stream lands before they are evicted again, so a deeper queue only delays every fill
const MAX_PENDING_FILLS: usize = 4;
// The slot a miss maps to: left zeroed, so the GPU's share of a CPU-served pick is exactly 0 (a slot mid-copy could
// hold non-finite scales)
const MISS_SLOT: u32 = 0;
// The doorbell thread spins this long after the last ring or step start, then parks, polling this often while parked
const DOORBELL_IDLE: Duration = Duration::from_millis(50);
const DOORBELL_PARKED_POLL: Duration = Duration::from_millis(1);

/// Which projection of a layer, in `[gate, up, down]` order.
const PROJECTIONS: usize = 3;

/// Prefill staging every offloaded layer shares on a device: room for a group of the largest layer's experts, and the
/// pinned chunks uploads from pageable experts go through.
struct Staging {
    stacks: [DeviceExpertStack; PROJECTIONS],
    ring: Mutex<BounceRing>,
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
        ring: Mutex::new(BounceRing::new(&device.cuda_stream())?),
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
    pending: Vec<bool>,
    clock: u64,
}

impl CacheState {
    fn new(experts: usize, slots: usize) -> Self {
        Self {
            slot_of: vec![None; experts],
            expert_in: vec![None; slots],
            last_use: vec![0; slots],
            pending: vec![false; slots],
            clock: 0,
        }
    }

    // the least recently used slot no in-flight copy targets and this forward does not read
    fn victim(&self, in_use: &[bool]) -> Option<usize> {
        (0..self.expert_in.len())
            .filter(|&s| s != MISS_SLOT as usize && !in_use[s] && !self.pending[s])
            .min_by_key(|&s| (self.expert_in[s].is_some(), self.last_use[s]))
    }
}

/// Pinned buffers a decode step's routes and results pass through, sized for the largest split batch.
struct Exchange {
    ids: CudaSlice<u32>,
    weights: CudaSlice<f32>,
    slots: CudaSlice<u32>,
    xs: CudaSlice<f32>,
    ys: CudaSlice<f32>,
}

/// What a layer's host functions and fill worker share.
struct Shared {
    host: [HostExpertStack; PROJECTIONS],
    cache: [DeviceExpertStack; PROJECTIONS],
    state: Mutex<CacheState>,
    exchange: Exchange,
    // expert -> slot in mapped host memory, which the GPU reads and only `state`'s holder writes
    table: Tensor,
    copy_stream: Arc<CudaStream>,
    // mapped: the sequence the GPU last rang and that ring's token count; the sequence the host thread last answered;
    // set when the GPU gave up waiting for an answer
    bell: CudaSlice<u32>,
    done: CudaSlice<u32>,
    timed_out: CudaSlice<u32>,
    // on the device: the sequence the next ring takes, which the wait compares the answer with
    counter: CudaSlice<u32>,
    act: GluActivationType,
    topk: usize,
}

impl Shared {
    fn cpu_experts(&self) -> CpuExperts<'_> {
        fn projection(h: &HostExpertStack) -> CpuExpertProjection<'_> {
            let [_, rows, cols] = h.dims();
            CpuExpertProjection {
                ty: h.ty(),
                bytes: h.bytes(),
                rows,
                cols,
            }
        }
        CpuExperts {
            gate: projection(&self.host[0]),
            up: projection(&self.host[1]),
            down: projection(&self.host[2]),
            act: self.act,
        }
    }

    fn hidden(&self) -> usize {
        self.host[0].dims()[2]
    }

    #[allow(clippy::mut_from_ref)]
    fn table(&self) -> &mut [u32] {
        let (storage, _) = self.table.storage_and_layout();
        let inference_tensor::Storage::Cuda(cuda) = &*storage else {
            unreachable!("the slot table is built on CUDA")
        };
        let slice = cuda.as_cuda_slice::<u32>().expect("a u32 slot table");
        let ptr = slice
            .mapped_host_ptr()
            .expect("the slot table is mapped host memory");
        // SAFETY: the table outlives the layer, and its writers hold `state`
        unsafe { std::slice::from_raw_parts_mut(ptr, slice.len()) }
    }

    // On the side stream's host thread: the picks the GPU found no slot for run on the CPU, and their experts are queued
    // for the slots least recently used
    fn split(self: &Arc<Self>, tokens: usize) -> Result<()> {
        let (picks, hidden) = (tokens * self.topk, self.hidden());
        let ex = &self.exchange;
        let (ids, weights, slots) = (
            mapped_host(&ex.ids, picks),
            mapped_host(&ex.weights, picks),
            mapped_host(&ex.slots, picks),
        );
        let ys = mapped_host(&ex.ys, tokens * hidden);
        let mut state = self.state.lock().expect("expert cache poisoned");
        state.clock += 1;
        let clock = state.clock;
        let mut in_use = vec![false; state.expert_in.len()];
        let mut misses = Vec::new();
        for i in 0..picks {
            if slots[i] == MISS_SLOT {
                misses.push(CpuExpertRoute {
                    token: i / self.topk,
                    expert: ids[i] as usize,
                    weight: weights[i],
                });
            } else {
                let slot = slots[i] as usize;
                in_use[slot] = true;
                state.last_use[slot] = clock;
            }
        }
        ys.fill(0.0);
        if !misses.is_empty() {
            self.cpu_experts()
                .forward(mapped_host(&ex.xs, tokens * hidden), &misses, ys)?;
        }

        // the experts most of this step's tokens missed are admitted first
        let mut experts = misses.iter().map(|m| m.expert).collect::<Vec<_>>();
        experts.sort_unstable();
        let mut runs = experts
            .chunk_by(|a, b| a == b)
            .map(|run| (run.len(), run[0]))
            .collect::<Vec<_>>();
        runs.sort_by_key(|&(n, _)| std::cmp::Reverse(n));
        let mut pending = state.pending.iter().filter(|&&p| p).count();
        for (_, expert) in runs {
            if pending >= MAX_PENDING_FILLS {
                break;
            }
            if state.expert_in.contains(&Some(expert)) {
                continue;
            }
            let Some(slot) = state.victim(&in_use) else {
                break;
            };
            pending += 1;
            if let Some(old) = state.expert_in[slot].take() {
                state.slot_of[old] = None;
                self.table()[old] = MISS_SLOT;
            }
            state.expert_in[slot] = Some(expert);
            state.last_use[slot] = clock;
            state.pending[slot] = true;
            in_use[slot] = true;
            fills()
                .send(FillJob::Fill(Fill {
                    shared: self.clone(),
                    slot,
                    expert,
                }))
                .ok();
        }
        Ok(())
    }

    // On the fill worker: the copy, then a host function that opens the slot once the copy has landed
    fn fill(self: Arc<Self>, slot: usize, expert: usize) -> Result<()> {
        self.copy_stream
            .context()
            .bind_to_thread()
            .map_err(inference_tensor::Error::wrap)?;
        for (host, cache) in self.host.iter().zip(&self.cache) {
            let size = host.expert_bytes();
            let src = &host.bytes()[expert * size..][..size];
            // SAFETY: host experts live as long as the layer, and the slot is unread until the copy has landed
            unsafe { cache.fill_async(slot, src, &self.copy_stream) }
                .map_err(|err| err.context("expert cache copy"))?;
        }
        let stream = self.copy_stream.cu_stream();
        let landed = Box::into_raw(Box::new(Fill {
            shared: self,
            slot,
            expert,
        }));
        // SAFETY: `landed` is reclaimed once by `fill_landed`
        unsafe { result::stream::launch_host_function(stream, fill_landed, landed.cast()) }.map_err(
            |err| inference_tensor::Error::wrap(err).context("expert cache landing callback"),
        )
    }
}

struct Fill {
    shared: Arc<Shared>,
    slot: usize,
    expert: usize,
}

enum FillJob {
    Fill(Fill),
    // a landed fill's layer, dropped on the fill thread: the last reference frees the layer's stream, a CUDA call a
    // host function may not make (the driver deadlocks), and a fill can land after its engine is gone
    Release(Arc<Shared>),
}

unsafe extern "C" fn fill_landed(data: *mut std::ffi::c_void) {
    // SAFETY: `data` is the box `Shared::fill` leaked for this call
    let Fill {
        shared,
        slot,
        expert,
    } = *unsafe { Box::from_raw(data.cast::<Fill>()) };
    {
        let mut state = shared.state.lock().expect("expert cache poisoned");
        state.pending[slot] = false;
        if state.expert_in[slot] == Some(expert) {
            state.slot_of[expert] = Some(slot);
            shared.table()[expert] = u32::try_from(slot).expect("slot ids fit u32");
        }
    }
    if let Err(std::sync::mpsc::SendError(FillJob::Release(shared))) =
        fills().send(FillJob::Release(shared))
    {
        // the fill thread is gone, so the process is exiting: leak the layer rather than free it here
        std::mem::forget(shared);
    }
}

impl Shared {
    // a fill that never queued: its slot goes back to the free pool instead of staying pending
    fn release(&self, slot: usize, expert: usize) {
        let mut state = self.state.lock().expect("expert cache poisoned");
        state.pending[slot] = false;
        if state.expert_in[slot] == Some(expert) {
            state.expert_in[slot] = None;
        }
    }
}

// One thread issues every layer's cache fills: host functions may not call CUDA themselves
fn fills() -> &'static Sender<FillJob> {
    static FILLS: OnceLock<Sender<FillJob>> = OnceLock::new();
    FILLS.get_or_init(|| {
        let (send, receive): (Sender<FillJob>, Receiver<FillJob>) = std::sync::mpsc::channel();
        std::thread::Builder::new()
            .name("moe-expert-fills".into())
            .spawn(move || {
                for job in receive {
                    let FillJob::Fill(Fill {
                        shared,
                        slot,
                        expert,
                    }) = job
                    else {
                        continue;
                    };
                    if let Err(err) = shared.clone().fill(slot, expert) {
                        tracing::error!("MoE expert cache fill failed: {err}");
                        shared.release(slot, expert);
                    }
                }
            })
            .expect("spawn the MoE expert fill thread");
        send
    })
}

impl Shared {
    fn word(buffer: &CudaSlice<u32>, index: usize) -> &AtomicU32 {
        let word = &mut mapped_host(buffer, index + 1)[index];
        // SAFETY: an aligned u32 of mapped memory the layer owns; the GPU and the host thread only touch it atomically
        unsafe { AtomicU32::from_ptr(word) }
    }

    // On the doorbell thread: the ring's misses on the CPU, then the answer the GPU waits for
    fn answer(self: &Arc<Self>, seq: u32) {
        let tokens = Self::word(&self.bell, 1).load(Ordering::Acquire) as usize;
        if let Err(err) = self.split(tokens) {
            // the misses then add nothing rather than stale values
            tracing::error!("offloaded MoE experts failed on the host: {err}");
            mapped_host(&self.exchange.ys, tokens * self.hidden()).fill(0.0);
        }
        Self::word(&self.done, 0).store(seq, Ordering::Release);
    }
}

/// The host thread every offloaded layer's decode rings, and the layers it polls.
struct Doorbell {
    layers: Mutex<Vec<Arc<Shared>>>,
    generation: AtomicUsize,
    // nanoseconds past `epoch` of the latest step start
    activity: AtomicU64,
    epoch: Instant,
    thread: OnceLock<std::thread::Thread>,
}

fn doorbell() -> &'static Doorbell {
    static DOORBELL: OnceLock<Doorbell> = OnceLock::new();
    DOORBELL.get_or_init(|| Doorbell {
        layers: Mutex::new(Vec::new()),
        generation: AtomicUsize::new(0),
        activity: AtomicU64::new(0),
        epoch: Instant::now(),
        thread: OnceLock::new(),
    })
}

impl Doorbell {
    fn now(&self) -> u64 {
        u64::try_from(self.epoch.elapsed().as_nanos()).unwrap_or(u64::MAX)
    }

    fn register(&'static self, layer: Arc<Shared>) {
        self.layers
            .lock()
            .expect("doorbell layers poisoned")
            .push(layer);
        self.generation.fetch_add(1, Ordering::Release);
        self.thread.get_or_init(|| {
            std::thread::Builder::new()
                .name("moe-doorbell".into())
                .spawn(move || self.run())
                .expect("spawn the MoE doorbell thread")
                .thread()
                .clone()
        });
        offload_activity();
    }

    fn unregister(&self, layer: &Arc<Shared>) {
        self.layers
            .lock()
            .expect("doorbell layers poisoned")
            .retain(|l| !Arc::ptr_eq(l, layer));
        self.generation.fetch_add(1, Ordering::Release);
    }

    fn run(&self) {
        let (mut layers, mut answered, mut seen) = (Vec::new(), Vec::new(), usize::MAX);
        let mut last_ring = self.now();
        loop {
            let generation = self.generation.load(Ordering::Acquire);
            if generation != seen {
                layers = self
                    .layers
                    .lock()
                    .expect("doorbell layers poisoned")
                    .clone();
                answered = layers
                    .iter()
                    .map(|l: &Arc<Shared>| Shared::word(&l.done, 0).load(Ordering::Acquire))
                    .collect();
                seen = generation;
            }
            let mut rang = false;
            for (layer, answered) in layers.iter().zip(answered.iter_mut()) {
                let seq = Shared::word(&layer.bell, 0).load(Ordering::Acquire);
                if seq != *answered {
                    layer.answer(seq);
                    *answered = seq;
                    rang = true;
                }
            }
            let now = self.now();
            if rang {
                last_ring = now;
                continue;
            }
            let idle = now.saturating_sub(last_ring.max(self.activity.load(Ordering::Acquire)));
            if idle > u64::try_from(DOORBELL_IDLE.as_nanos()).unwrap_or(u64::MAX) {
                for layer in &layers {
                    if Shared::word(&layer.timed_out, 0).swap(0, Ordering::AcqRel) != 0 {
                        tracing::error!(
                            "an offloaded MoE layer's decode gave up waiting for the CPU's experts"
                        );
                    }
                }
                std::thread::park_timeout(DOORBELL_PARKED_POLL);
            } else {
                std::hint::spin_loop();
            }
        }
    }
}

/// Keeps the doorbell thread spinning: called as a step starts, so its first ring is answered at once.
pub fn offload_activity() {
    let bell = doorbell();
    bell.activity.store(bell.now(), Ordering::Release);
    if let Some(thread) = bell.thread.get() {
        thread.unpark();
    }
}

pub(super) struct OffloadedExperts {
    shared: Arc<Shared>,
    device: CudaDevice,
    // experts a prompt pass stages at once
    staging_experts: usize,
}

impl OffloadedExperts {
    /// `host` holds `[gate, up, down]`; `slots` experts of each stay cached on `device`, `topk` routed per token.
    pub(super) fn new(
        host: [HostExpertStack; PROJECTIONS],
        slots: usize,
        topk: usize,
        device: &CudaDevice,
        act: GluActivationType,
    ) -> Result<Self> {
        let experts = host[0].dims()[0];
        let hidden = host[0].dims()[2];
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
        let cache = |p: usize| DeviceExpertStack::zeros(&[layouts[p]], slots, device);
        let stream = device.cuda_stream();
        let picks = SPLIT_MAX_TOKENS * topk;
        let exchange = Exchange {
            ids: mapped_buffer(&stream, picks)?,
            weights: mapped_buffer(&stream, picks)?,
            slots: mapped_buffer(&stream, picks)?,
            xs: mapped_buffer(&stream, SPLIT_MAX_TOKENS * hidden)?,
            ys: mapped_buffer(&stream, SPLIT_MAX_TOKENS * hidden)?,
        };
        let table = Tensor::zeros(experts, DType::U32, &Device::Cuda(device.clone()))?
            .to_mapped_host()?
            .ok_or_else(|| inference_tensor::Error::msg("the slot table needs a CUDA device"))?;
        let shared = Arc::new(Shared {
            cache: [cache(0)?, cache(1)?, cache(2)?],
            host,
            state: Mutex::new(CacheState::new(experts, slots)),
            exchange,
            table,
            copy_stream: stream
                .context()
                .new_stream()
                .map_err(inference_tensor::Error::wrap)?,
            bell: zeroed_mapped(&stream, 2)?,
            done: zeroed_mapped(&stream, 1)?,
            timed_out: zeroed_mapped(&stream, 1)?,
            counter: stream
                .alloc_zeros(1)
                .map_err(inference_tensor::Error::wrap)?,
            act,
            topk,
        });
        doorbell().register(shared.clone());
        Ok(Self {
            shared,
            device: device.clone(),
            staging_experts: OFFLOAD_STAGING_EXPERTS,
        })
    }

    fn experts(&self) -> usize {
        self.shared.host[0].dims()[0]
    }

    /// The layer's output, and `overlap`'s: on a decode step it runs on the stream while the CPU computes the misses.
    pub(super) fn forward(
        &self,
        forward: &MoEForward,
        config: MoEForwardConfig,
        overlap: Option<&mut dyn FnMut() -> Result<Tensor>>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        if forward.lora.is_some() {
            inference_tensor::bail!("offloaded MoE experts do not take LoRA adapters");
        }
        if config.num_experts_per_tok != self.shared.topk {
            inference_tensor::bail!(
                "offloaded MoE experts were built for top-{} routing",
                self.shared.topk
            );
        }
        if forward.shape.num_tokens <= SPLIT_MAX_TOKENS {
            self.forward_split(forward, config, overlap)
        } else {
            let ys = self.forward_staged(forward, config)?;
            Ok((ys, overlap.map(|f| f()).transpose()?))
        }
    }

    // the cached experts on the stream, the misses through a host function on the side stream, joined at the end
    fn forward_split(
        &self,
        forward: &MoEForward,
        config: MoEForwardConfig,
        overlap: Option<&mut dyn FnMut() -> Result<Tensor>>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let (tokens, hidden) = forward.shape.flat();
        let picks = tokens * self.shared.topk;
        let device = forward.xs.device();
        let main = self.device.cuda_stream();
        offload_activity();
        let ex = &self.shared.exchange;
        let ids = forward.topk_ids.to_dtype(DType::U32)?.contiguous()?;
        let weights = forward.topk_weights.to_dtype(DType::F32)?.contiguous()?;
        let xs = match forward.xs_flat.dtype() {
            DType::BF16 | DType::F32 => forward.xs_flat.contiguous()?,
            _ => forward.xs_flat.to_dtype(DType::F32)?.contiguous()?,
        };
        // SAFETY: the preparation below writes every slot id before the cached experts' kernel reads one
        let slot_ids = unsafe { Tensor::empty((tokens, self.shared.topk), DType::U32, device)? };
        let ys = Tensor::zeros((tokens, hidden), DType::F32, device)?;
        let int = |n: usize| i32::try_from(n).expect("a split batch fits i32");
        let mapped = |buffer: &CudaSlice<u32>| buffer.device_ptr(&main).0 as *mut std::ffi::c_void;
        let mapped_f32 =
            |buffer: &CudaSlice<f32>| buffer.device_ptr(&main).0 as *mut std::ffi::c_void;
        let xs_ptr = if xs.dtype() == DType::BF16 {
            tensor_ptr::<half::bf16>(&xs, &main)?
        } else {
            tensor_ptr::<f32>(&xs, &main)?
        };
        let table = {
            let (storage, _) = self.shared.table.storage_and_layout();
            let inference_tensor::Storage::Cuda(cuda) = &*storage else {
                unreachable!("the slot table is built on CUDA")
            };
            cuda.as_cuda_slice::<u32>()?.device_ptr(&main).0
        };
        // SAFETY: the pinned buffers hold the largest split batch; the preparation writes them before the event the
        // side stream waits on, the host function reads them after it, and the stream waits for the side stream's copy
        // of the CPU result before it moves on, so the next step cannot overwrite a buffer still in use
        unsafe {
            let status = crate::cuda::ffi::moe_split_prepare(
                tensor_ptr::<u32>(&ids, &main)? as *const std::ffi::c_void,
                tensor_ptr::<f32>(&weights, &main)? as *const std::ffi::c_void,
                xs_ptr as *const std::ffi::c_void,
                i32::from(xs.dtype() == DType::BF16),
                table as *const std::ffi::c_void,
                tensor_ptr::<u32>(&slot_ids, &main)? as *mut std::ffi::c_void,
                mapped(&ex.ids),
                mapped_f32(&ex.weights),
                mapped(&ex.slots),
                mapped_f32(&ex.xs),
                int(picks),
                int(tokens * hidden),
                main.cu_stream() as i64,
            );
            if status != 0 {
                inference_tensor::bail!(
                    "preparing the offloaded experts' split failed with CUDA error {status}"
                );
            }
            let status = crate::cuda::ffi::moe_doorbell_ring(
                mapped(&self.shared.counter),
                mapped(&self.shared.bell),
                int(tokens),
                main.cu_stream() as i64,
            );
            if status != 0 {
                inference_tensor::bail!("ringing the MoE doorbell failed with CUDA error {status}");
            }
        }

        let hits = MoEForward {
            xs: forward.xs,
            xs_flat: forward.xs_flat,
            topk_weights: forward.topk_weights,
            topk_ids: &slot_ids,
            original_dtype: forward.original_dtype,
            shape: forward.shape,
            lora: None,
        };
        let cache = &self.shared.cache;
        let cache = [&cache[0] as &dyn KernelWeight, &cache[1], &cache[2]];
        let hit_config = MoEForwardConfig {
            num_experts: cache[0].kernel_shape().dims()[0],
            ..config
        };
        let gpu = indexed_mmvq_with(cache, &hits, hit_config)?.ok_or_else(|| {
            inference_tensor::Error::msg("cached experts cannot run the indexed matvec")
        })?;
        let side = overlap.map(|f| f()).transpose()?;
        // SAFETY: the wait reads the answer only once the host thread has written it, and the next ring of this layer
        // comes after the stream has seen this one answered, so the host thread is done with the buffers by then
        let status = unsafe {
            crate::cuda::ffi::moe_doorbell_wait(
                mapped(&self.shared.counter),
                mapped(&self.shared.done),
                mapped_f32(&ex.ys),
                tensor_ptr::<f32>(&ys, &main)? as *mut std::ffi::c_void,
                int(tokens * hidden),
                mapped(&self.shared.timed_out),
                main.cu_stream() as i64,
            )
        };
        if status != 0 {
            inference_tensor::bail!("waiting on the MoE doorbell failed with CUDA error {status}");
        }
        let out = (gpu + ys.to_dtype(forward.original_dtype)?)?;
        Ok((out, side))
    }

    // the cached experts this batch routes to from their slots, then the rest over the bus a staging group at a time;
    // each pass parks the routes it does not run, and the passes' outputs add up
    fn forward_staged(&self, forward: &MoEForward, config: MoEForwardConfig) -> Result<Tensor> {
        let experts = self.experts();
        let staging = {
            let layouts = LAYOUTS.lock().expect("expert layouts poisoned").clone();
            staging(&self.device, &layouts, experts.min(self.staging_experts))?
        };
        // reading the routes waits for all earlier work, so no host function can still evict a slot read below
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
        let mut used = ids.iter().map(|&e| e as usize).collect::<Vec<_>>();
        used.sort_unstable();
        used.dedup();
        let main = self.device.cuda_stream();
        // a landed slot changes only through a host function, and none is queued (see above)
        let slot_of = self
            .shared
            .state
            .lock()
            .expect("expert cache poisoned")
            .slot_of
            .clone();
        let (cached, uncached): (Vec<usize>, Vec<usize>) =
            used.iter().partition(|&&e| slot_of[e].is_some());

        let mut out = None;
        if !cached.is_empty() {
            let cache = &self.shared.cache;
            let slots = cache[0].kernel_shape().dims()[0];
            let pass = self.staged_pass(forward, config, &ids, &weights, slots, |e| slot_of[e])?;
            let weights = [&cache[0] as &dyn KernelWeight, &cache[1], &cache[2]];
            out = Some(self.run_pass(weights, forward, &pass, config, slots)?);
        }
        for group in uncached.chunks(self.staging_experts) {
            let mut views = Vec::with_capacity(PROJECTIONS);
            for (host, stack) in self.shared.host.iter().zip(&staging.stacks) {
                let [_, rows, cols] = host.dims();
                let view = stack.view(host.ty(), group.len(), rows, cols)?;
                let size = host.expert_bytes();
                let mut p = 0;
                while p < group.len() {
                    // experts with consecutive ids are one copy
                    let mut end = p + 1;
                    while end < group.len() && group[end] == group[end - 1] + 1 {
                        end += 1;
                    }
                    let src = &host.bytes()[group[p] * size..group[end - 1] * size + size];
                    // SAFETY: host experts live as long as the layer; the stream orders this copy after the previous
                    // pass's reads of the staging and before this one's
                    if host.is_pageable() {
                        let mut ring = staging.ring.lock().expect("bounce ring poisoned");
                        unsafe { view.fill_bounced_async(p, src, &mut ring, &main)? };
                    } else {
                        unsafe { view.fill_async(p, src, &main)? };
                    }
                    p = end;
                }
                // SAFETY: as the copies above, ordered on `main` before this pass's reads
                unsafe { view.zero_padding_async(group.len(), &main)? };
                views.push(view);
            }
            let pass = self.staged_pass(forward, config, &ids, &weights, group.len(), |e| {
                group.binary_search(&e).ok()
            })?;
            let weights = [&views[0] as &dyn KernelWeight, &views[1], &views[2]];
            let partial = self.run_pass(weights, forward, &pass, config, group.len())?;
            out = Some(match out {
                Some(sum) => (sum + partial)?,
                None => partial,
            });
        }
        out.ok_or_else(|| inference_tensor::Error::msg("a staged batch routes to no expert"))
    }

    // the routes as one pass sees them: its experts' positions in its stack, the rest parked at `experts` with weight 0
    fn staged_pass(
        &self,
        forward: &MoEForward,
        config: MoEForwardConfig,
        ids: &[u32],
        weights: &[f32],
        experts: usize,
        position: impl Fn(usize) -> Option<usize>,
    ) -> Result<(Tensor, Tensor)> {
        let shape = (forward.shape.num_tokens, config.num_experts_per_tok);
        let device = forward.xs.device();
        let parked = u32::try_from(experts).expect("expert counts fit u32");
        let (pass_ids, pass_weights): (Vec<u32>, Vec<f32>) = ids
            .iter()
            .zip(weights)
            .map(|(&e, &w)| match position(e as usize) {
                Some(p) => (u32::try_from(p).expect("fits u32"), w),
                None => (parked, 0.0),
            })
            .unzip();
        Ok((
            Tensor::from_vec(pass_ids, shape, device)?,
            Tensor::from_vec(pass_weights, shape, device)?
                .to_dtype(forward.topk_weights.dtype())?,
        ))
    }

    fn run_pass(
        &self,
        weights: [&dyn KernelWeight; PROJECTIONS],
        forward: &MoEForward,
        (ids, routing): &(Tensor, Tensor),
        config: MoEForwardConfig,
        experts: usize,
    ) -> Result<Tensor> {
        let pass = MoEForward {
            xs: forward.xs,
            xs_flat: forward.xs_flat,
            topk_weights: routing,
            topk_ids: ids,
            original_dtype: forward.original_dtype,
            shape: forward.shape,
            lora: None,
        };
        let pass_config = MoEForwardConfig {
            num_experts: experts,
            ..config
        };
        grouped_parked_with(weights, &pass, pass_config)?.ok_or_else(|| {
            inference_tensor::Error::msg("staged experts cannot run the grouped kernels")
        })
    }
}

impl Drop for OffloadedExperts {
    fn drop(&mut self) {
        doorbell().unregister(&self.shared);
    }
}

fn zeroed_mapped(stream: &Arc<CudaStream>, len: usize) -> Result<CudaSlice<u32>> {
    let buffer = mapped_buffer(stream, len)?;
    mapped_host(&buffer, len).fill(0);
    Ok(buffer)
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

    use super::super::backends::grouped_with;
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

    // random blocks with a small leading f16 scale per block (or the row's f32 one), as GGUF lays them out
    fn random_blocks(
        ty: inference_quant::GgufType,
        rows: usize,
        cols: usize,
        salt: u32,
    ) -> Vec<u8> {
        let row_bytes = ty.row_bytes(cols).expect("whole blocks");
        let mut state = salt.wrapping_mul(2_654_435_761).wrapping_add(7);
        let mut bytes = (0..EXPERTS * rows * row_bytes)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state.to_le_bytes()[0]
            })
            .collect::<Vec<_>>();
        let meta = ty.row_scale_bytes();
        for row in bytes.chunks_exact_mut(row_bytes) {
            if meta > 0 {
                row[..meta].copy_from_slice(&(0.002f32 / 64.0).to_le_bytes());
                continue;
            }
            for block in row.chunks_exact_mut(ty.type_size()) {
                block[..2].copy_from_slice(&half::f16::from_f32(0.002).to_le_bytes());
            }
        }
        bytes
    }

    // Unsloth's Flash-Next mix: IQ3_S gate/up and IQ4_NL down over a width no Q8_K super-block fills; the staged
    // prompt path and the cached decode path against the CPU expert path, which tests check against dense math
    #[test]
    fn offloaded_32_wide_experts_match_the_cpu_path() -> Result<()> {
        crate::skip_without_cuda!();
        const INTER_32: usize = 640;
        let device = Device::new_cuda(0)?;
        let cuda = device.as_cuda_device()?;
        let types = [
            inference_quant::GgufType::Iq3S,
            inference_quant::GgufType::Iq3S,
            inference_quant::GgufType::Iq4Nl,
        ];
        let dims = [
            [EXPERTS, INTER_32, HIDDEN],
            [EXPERTS, INTER_32, HIDDEN],
            [EXPERTS, HIDDEN, INTER_32],
        ];
        let bytes = [0, 1, 2].map(|p| {
            random_blocks(
                types[p],
                dims[p][1],
                dims[p][2],
                u32::try_from(p).expect("three") + 1,
            )
        });
        let host = |p: usize| HostExpertStack::new(types[p], dims[p], &bytes[p], cuda);
        let offloaded = OffloadedExperts::new(
            [host(0)?, host(1)?, host(2)?],
            SLOTS,
            TOPK,
            cuda,
            GluActivationType::Silu,
        )?;
        let config = MoEForwardConfig {
            num_experts: EXPERTS,
            num_experts_per_tok: TOPK,
            act: Activation::Silu,
        };
        let projection = |p: usize| inference_quant::cpu_experts::CpuExpertProjection {
            ty: types[p],
            bytes: &bytes[p],
            rows: dims[p][1],
            cols: dims[p][2],
        };
        let cpu = inference_quant::cpu_experts::CpuExperts {
            gate: projection(0),
            up: projection(1),
            down: projection(2),
            act: GluActivationType::Silu,
        };
        // what another layer left in the shared staging past the experts a prompt stages: 0xFF reads as NaN scales
        let layouts = LAYOUTS.lock().expect("expert layouts poisoned").clone();
        for (p, stack) in staging(cuda, &layouts, EXPERTS)?.stacks.iter().enumerate() {
            let [_, rows, cols] = dims[p];
            let view = stack.view(types[p], EXPERTS, rows, cols)?;
            let poison =
                vec![0xFF_u8; EXPERTS * rows * types[p].row_bytes(cols).expect("whole blocks")];
            // SAFETY: `poison` outlives the synchronize below, and nothing reads the staging before it
            unsafe { view.fill_async(0, &poison, &cuda.cuda_stream())? };
        }
        cuda.cuda_stream()
            .synchronize()
            .map_err(inference_tensor::Error::wrap)?;
        for tokens in [1, 2, PROMPT] {
            let mut run = run(&device, tokens, tokens)?;
            if tokens == PROMPT {
                // two of the experts, so the staged stack ends short of the poisoned bytes
                let ids = (0..tokens * TOPK)
                    .map(|i| u32::try_from(i % TOPK).expect("few experts"))
                    .collect::<Vec<_>>();
                run.ids = Tensor::from_vec(ids, (tokens, TOPK), &device)?;
            }
            let (actual, _) = offloaded.forward(&run.forward(), config, None)?;
            let xs = run
                .xs_flat
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let ids = run
                .ids
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<u32>()?;
            let weights = run
                .weights
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let routes = ids
                .iter()
                .zip(&weights)
                .enumerate()
                .map(
                    |(i, (&expert, &weight))| inference_quant::cpu_experts::CpuExpertRoute {
                        token: i / TOPK,
                        expert: expert as usize,
                        weight,
                    },
                )
                .collect::<Vec<_>>();
            let mut expected = vec![0f32; xs.len()];
            cpu.forward(&xs, &routes, &mut expected)?;
            let expected = Tensor::from_vec(expected, (tokens, HIDDEN), &Device::Cpu)?;
            let actual = actual.to_device(&Device::Cpu)?.reshape((tokens, HIDDEN))?;
            let values = actual.flatten_all()?.to_vec1::<f32>()?;
            assert!(
                values.iter().all(|v| v.is_finite()),
                "{tokens} tokens: non-finite output"
            );
            let err = relative_error(&actual, &expected)?;
            // the CPU quantizes activations to Q8_K or Q8 blocks of 32, the GPU kernels to Q8_1
            assert!(err < 0.03, "{tokens} tokens: relative error {err}");
        }
        Ok(())
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
        let mut offloaded = OffloadedExperts::new(
            [host(0)?, host(1)?, host(2)?],
            SLOTS,
            TOPK,
            cuda,
            GluActivationType::Silu,
        )?;
        // a prompt's uncached experts (at least EXPERTS - SLOTS) stage in more than one group
        offloaded.staging_experts = 3;
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
            // work overlapped with the CPU's misses comes back beside the layer's output
            let marker = f32::from(u8::try_from(step).expect("few steps"));
            let mut side = || Tensor::full(marker, 3, &device);
            let (actual, side) = offloaded.forward(&run.forward(), config, Some(&mut side))?;
            assert_eq!(
                side.expect("the overlap ran").to_vec1::<f32>()?,
                vec![marker; 3]
            );
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
        let (actual, _) = offloaded.forward(&run.forward(), config, None)?;
        let expected =
            grouped_with(resident, &run.forward(), config)?.expect("Q4_K runs the grouped mmq");
        // the passes' sums add in another order than one reduce
        assert!(
            relative_error(&actual, &expected)? < 1e-5,
            "the staged prompt path runs the same kernels"
        );
        Ok(())
    }

    // The fixed cost of a decode step's host round trip: an offloaded layer whose experts are all cached (so the
    // CPU computes nothing) against the same cached-expert kernel resident, eager and replayed from a graph
    #[test]
    #[ignore = "benchmark"]
    fn split_round_trip_costs_a_fixed_latency() -> Result<()> {
        use crate::cuda::graph_capture::CudaGraphHandle;
        use inference_tensor::cuda::cudarc::driver::sys;
        const BENCH_EXPERTS: usize = 64;
        const BENCH_HIDDEN: usize = 2560;
        const BENCH_INTER: usize = 256;
        const BENCH_TOPK: usize = 10;
        const STEPS: u32 = 500;
        crate::skip_without_cuda!();
        let device = Device::new_cuda(0)?;
        let cuda = device.as_cuda_device()?;
        let q4k = |rows: usize, cols: usize, salt: usize| -> Result<Vec<u8>> {
            let weights = patterned(&[BENCH_EXPERTS, rows, cols], salt, 0.1)?;
            Ok(QTensor::quantize(&weights, GgmlDType::Q4K)?
                .data()?
                .into_owned())
        };
        let bytes = [
            q4k(BENCH_INTER, BENCH_HIDDEN, 1)?,
            q4k(BENCH_INTER, BENCH_HIDDEN, 2)?,
            q4k(BENCH_HIDDEN, BENCH_INTER, 3)?,
        ];
        let dims = [
            [BENCH_EXPERTS, BENCH_INTER, BENCH_HIDDEN],
            [BENCH_EXPERTS, BENCH_INTER, BENCH_HIDDEN],
            [BENCH_EXPERTS, BENCH_HIDDEN, BENCH_INTER],
        ];
        let host = |p: usize| {
            HostExpertStack::new(inference_quant::GgufType::Q4K, dims[p], &bytes[p], cuda)
        };
        let offloaded = OffloadedExperts::new(
            [host(0)?, host(1)?, host(2)?],
            BENCH_EXPERTS,
            BENCH_TOPK,
            cuda,
            GluActivationType::Silu,
        )?;
        let resident = (0..PROJECTIONS)
            .map(|p| {
                let [_, rows, cols] = dims[p];
                let stack = DeviceExpertStack::zeros(
                    &[(inference_quant::GgufType::Q4K, rows, cols)],
                    BENCH_EXPERTS,
                    cuda,
                )?;
                // SAFETY: `bytes` outlives the test, and the synchronize below lands the copy before any read
                unsafe { stack.fill_async(0, &bytes[p], &cuda.cuda_stream())? };
                Ok(stack)
            })
            .collect::<Result<Vec<_>>>()?;
        let resident = [
            &resident[0] as &dyn KernelWeight,
            &resident[1],
            &resident[2],
        ];
        let config = MoEForwardConfig {
            num_experts: BENCH_EXPERTS,
            num_experts_per_tok: BENCH_TOPK,
            act: Activation::Silu,
        };
        let xs = patterned(&[1, 1, BENCH_HIDDEN], 5, 1.0)?.to_device(&device)?;
        let xs_flat = xs.reshape((1, BENCH_HIDDEN))?;
        let ids = Tensor::from_vec(
            (0..u32::try_from(BENCH_TOPK).expect("fits u32")).collect::<Vec<_>>(),
            (1, BENCH_TOPK),
            &device,
        )?;
        let weights = Tensor::full(0.1f32, (1, BENCH_TOPK), &device)?;
        let forward = MoEForward {
            xs: &xs,
            xs_flat: &xs_flat,
            topk_weights: &weights,
            topk_ids: &ids,
            original_dtype: DType::F32,
            shape: MoEForwardShape::new(1, 1, BENCH_HIDDEN),
            lora: None,
        };
        let stream = cuda.cuda_stream();
        let sync = || stream.synchronize().map_err(inference_tensor::Error::wrap);
        // the routed experts' fills land before the timing
        for _ in 0..20 {
            offloaded.forward(&forward, config, None)?;
            sync()?;
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let time = |f: &dyn Fn() -> Result<()>| -> Result<f64> {
            f()?;
            sync()?;
            let t = std::time::Instant::now();
            for _ in 0..STEPS {
                f()?;
            }
            sync()?;
            Ok(t.elapsed().as_secs_f64() * 1e6 / f64::from(STEPS))
        };
        let graph = |f: &dyn Fn() -> Result<()>| -> Result<CudaGraphHandle> {
            sync()?;
            // as the engine captures: no pool reuse across the graph's allocations, uploads cached
            let _pool = crate::cuda::graph_capture::prepare_cuda_graph_memory_pool(&stream)?;
            let _htod = cuda.enable_cuda_graph_htod_cache();
            // the warm-up fills the upload cache the capture reads
            f()?;
            sync()?;
            let restore = crate::cuda::graph_capture::disable_event_tracking_for_capture(&stream);
            stream
                .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
                .map_err(inference_tensor::Error::wrap)?;
            f()?;
            let graph = CudaGraphHandle::end_capture(&stream)?.expect("a captured graph");
            crate::cuda::graph_capture::restore_event_tracking_after_capture(&stream, restore);
            Ok(graph)
        };
        let offloaded_step =
            || -> Result<()> { offloaded.forward(&forward, config, None).map(|_| ()) };
        let resident_step = || -> Result<()> {
            indexed_mmvq_with(resident, &forward, config)?.expect("Q4_K runs the indexed matvec");
            Ok(())
        };
        let eager_offloaded = time(&offloaded_step)?;
        let eager_resident = time(&resident_step)?;
        let offloaded_graph = graph(&offloaded_step)?;
        let resident_graph = graph(&resident_step)?;
        let graph_offloaded = time(&|| offloaded_graph.launch())?;
        let graph_resident = time(&|| resident_graph.launch())?;
        eprintln!(
            "eager: offloaded {eager_offloaded:.1} us, resident {eager_resident:.1} us; graph: offloaded \
             {graph_offloaded:.1} us, resident {graph_resident:.1} us"
        );
        Ok(())
    }
}
