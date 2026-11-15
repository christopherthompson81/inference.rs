//! A device cache of the hot experts of an EXL3 projection kept in host memory. The kernels read every expert through
//! the projection's address table, so a cached expert is just a table entry pointing into a device slot instead of the
//! host copy. Decode graphs capture only a copy of each forward's picks into a mapped record; `cache_step`, run as a
//! step starts (outside any capture), scores the last step's picks, opens the slots whose copies have landed and starts
//! the next copies on a stream of its own. An evicted expert's entry points back at the host before its slot is
//! overwritten, and a new one's points at the slot only once its copy has landed, so a kernel reads valid bytes either
//! way.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};

use inference_tensor::cuda_backend::cudarc::driver::{
    CudaEvent, CudaSlice, CudaStream, DevicePtr, result,
};
use inference_tensor::{CudaDevice, DType, Result, Storage, Tensor};

// a step's picks are added to scores decayed by this
const SCORE_DECAY: f32 = 0.9;
// copies a projection starts a step, and has in flight at most: 16 warmed a whole cache of 256 within a reply on
// Qwen3.6, against 1 and 4; at a quarter of the experts cached, 4 and 16 were level
const FILLS_PER_STEP: usize = 16;
const MAX_PENDING: usize = 16;
// an uncached expert replaces a cached one only past this multiple of its score, so the cache does not churn
const REPLACE_MARGIN: f32 = 1.5;
// picks the routing record holds (8 verify tokens of up to 128 experts each)
const ROUTING_PICKS: usize = 1024;

static CACHES: Mutex<Vec<Weak<Mutex<HostCache>>>> = Mutex::new(Vec::new());

pub(super) struct HostCache {
    stream: Arc<CudaStream>,
    copy: Arc<CudaStream>,
    // each expert's trellis, suh and svh at `offsets` within its `stride` bytes, from `host_base` (host memory mapped
    // into the device) and in `slots`
    host_base: u64,
    stride: usize,
    offsets: [usize; 3],
    experts: usize,
    slots: CudaSlice<u8>,
    occupant: Vec<Option<usize>>,
    // copies under way: slot, expert, landed
    pending: Vec<(usize, usize, CudaEvent)>,
    // the address table as the kernels should next read it, and the device table it is copied to
    mirror: Vec<i64>,
    tables: Tensor,
    score: Vec<f32>,
    routing: CudaSlice<u32>,
    routed: AtomicUsize,
}

impl std::fmt::Debug for HostCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostCache")
            .field("experts", &self.experts)
            .field("slots", &self.occupant.len())
            .finish()
    }
}

impl HostCache {
    /// A cache of `slots` experts for the projection whose `experts` live from `host_base`, `stride` bytes apart;
    /// `tables` is the device address table the kernels read (trellis, then suh, then svh entries).
    pub(super) fn new(
        device: &CudaDevice,
        host_base: u64,
        stride: usize,
        offsets: [usize; 3],
        experts: usize,
        slots: usize,
        tables: &Tensor,
    ) -> Result<Arc<Mutex<Self>>> {
        let stream = device.cuda_stream();
        let copy = stream.fork().map_err(inference_tensor::Error::wrap)?;
        let mirror = tables.to_vec1::<i64>()?;
        // SAFETY: a slot is written by its copy before any table entry points at it
        let slot_bytes = unsafe { stream.alloc::<u8>(slots.min(experts) * stride) }
            .map_err(inference_tensor::Error::wrap)?;
        // SAFETY: the record is written by a forward's copy before `step` reads the picks it counts
        let routing = unsafe { stream.alloc_mapped_host::<u32>(ROUTING_PICKS) }
            .map_err(inference_tensor::Error::wrap)?;
        let cache = Arc::new(Mutex::new(Self {
            stream,
            copy,
            host_base,
            stride,
            offsets,
            experts,
            slots: slot_bytes,
            occupant: vec![None; slots.min(experts)],
            pending: Vec::new(),
            mirror,
            tables: tables.clone(),
            score: vec![0.0; experts],
            routing,
            routed: AtomicUsize::new(0),
        }));
        let mut all = CACHES.lock().expect("expert caches poisoned");
        all.retain(|c| c.strong_count() > 0);
        all.push(Arc::downgrade(&cache));
        Ok(cache)
    }

    /// Copies this forward's picks (`ids`, any integer dtype, on this device) into the routing record, in stream order:
    /// a decode graph captures the copy, so each replay refreshes the record.
    pub(super) fn record(&self, ids: &Tensor) -> Result<()> {
        let ids = ids.to_dtype(DType::U32)?.flatten_all()?.contiguous()?;
        let picks = ids.elem_count().min(ROUTING_PICKS);
        let (s, l) = ids.storage_and_layout();
        let Storage::Cuda(c) = &*s else {
            inference_tensor::bail!("an EXL3 expert cache records CUDA picks");
        };
        let (src, _src) = crate::utils::slice_ptr(c.as_cuda_slice::<u32>()?, l.start_offset());
        let (dst, _dst) = self.routing.device_ptr(&self.stream);
        // SAFETY: both span `picks` u32 on this stream's device
        unsafe {
            result::memcpy_dtod_async(dst, src, picks * 4, self.stream.cu_stream())
                .map_err(inference_tensor::Error::wrap)?;
        }
        self.routed.store(picks, Ordering::Relaxed);
        Ok(())
    }

    // experts whose table entries point into a slot
    #[cfg(test)]
    pub(super) fn landed(&self) -> usize {
        self.occupant.iter().flatten().count() - self.pending.len()
    }

    // an expert's trellis, suh and svh addresses from `base`, its first byte (its host copy, or a slot)
    fn addresses(&self, base: u64) -> [i64; 3] {
        self.offsets.map(|o| (base + o as u64) as i64)
    }

    fn host_of(&self, expert: usize) -> u64 {
        self.host_base + (expert * self.stride) as u64
    }

    fn point(&mut self, expert: usize, at: [i64; 3]) {
        for (j, a) in at.into_iter().enumerate() {
            self.mirror[j * self.experts + expert] = a;
        }
    }

    fn flush(&self) -> Result<()> {
        let (s, l) = self.tables.storage_and_layout();
        let Storage::Cuda(c) = &*s else {
            inference_tensor::bail!("an EXL3 address table lives on CUDA");
        };
        let (dst, _dst) = crate::utils::slice_ptr(c.as_cuda_slice::<i64>()?, l.start_offset());
        // SAFETY: the table holds `mirror.len()` i64; a pageable source is staged before the call returns
        unsafe {
            result::memcpy_htod_async(dst, &self.mirror, self.stream.cu_stream())
                .map_err(inference_tensor::Error::wrap)
        }
    }

    // the last step's picks scored, landed copies opened, and the next copies started
    fn step(&mut self) -> Result<()> {
        let picks = self.routed.swap(0, Ordering::Relaxed);
        if picks == 0 && self.pending.is_empty() {
            return Ok(());
        }
        let host = self.routing.mapped_host_ptr().expect("mapped allocation");
        // SAFETY: the record is mapped host memory of ROUTING_PICKS u32, last written by a finished step
        let record = unsafe { std::slice::from_raw_parts(host, picks) };
        self.score.iter_mut().for_each(|s| *s *= SCORE_DECAY);
        for &e in record {
            if let Some(s) = self.score.get_mut(e as usize) {
                *s += 1.0;
            }
        }
        let mut dirty = false;
        let landed = self
            .pending
            .iter()
            .filter(|(_, _, done)| done.is_complete())
            .map(|&(slot, expert, _)| (slot, expert))
            .collect::<Vec<_>>();
        if !landed.is_empty() {
            self.pending.retain(|(_, _, done)| !done.is_complete());
            let base = self.slots.device_ptr(&self.stream).0;
            for (slot, expert) in landed {
                let at = self.addresses(base + (slot * self.stride) as u64);
                self.point(expert, at);
            }
            dirty = true;
        }
        for _ in 0..FILLS_PER_STEP {
            if self.pending.len() >= MAX_PENDING {
                break;
            }
            let cached = |e: usize| self.occupant.contains(&Some(e));
            let Some(want) = (0..self.experts)
                .filter(|&e| !cached(e) && self.score[e] > 0.0)
                .max_by(|&a, &b| self.score[a].total_cmp(&self.score[b]))
            else {
                break;
            };
            let busy = |slot: usize| self.pending.iter().any(|&(s, _, _)| s == slot);
            let victim = (0..self.occupant.len())
                .filter(|&s| !busy(s))
                .min_by(|&a, &b| {
                    let score =
                        |s: usize| self.occupant[s].map_or(f32::NEG_INFINITY, |e| self.score[e]);
                    score(a).total_cmp(&score(b))
                });
            let Some(slot) = victim else {
                break;
            };
            if let Some(old) = self.occupant[slot] {
                if self.score[want] <= self.score[old] * REPLACE_MARGIN {
                    break;
                }
                // the old expert's kernels read the host copy from here on, before its slot is overwritten
                let at = self.addresses(self.host_of(old));
                self.point(old, at);
                self.flush()?;
                dirty = false;
            }
            let freed = self
                .stream
                .record_event(None)
                .map_err(inference_tensor::Error::wrap)?;
            self.copy
                .wait(&freed)
                .map_err(inference_tensor::Error::wrap)?;
            let (dst, _dst) = self.slots.device_ptr(&self.copy);
            let src = self.host_of(want);
            // SAFETY: the slot and the expert's host bytes each span `stride` bytes, on the copy stream
            unsafe {
                result::memcpy_dtod_async(
                    dst + (slot * self.stride) as u64,
                    src,
                    self.stride,
                    self.copy.cu_stream(),
                )
                .map_err(inference_tensor::Error::wrap)?;
            }
            let done = self
                .copy
                .record_event(None)
                .map_err(inference_tensor::Error::wrap)?;
            self.occupant[slot] = Some(want);
            self.pending.push((slot, want, done));
        }
        if dirty {
            self.flush()?;
        }
        Ok(())
    }
}

/// Every live EXL3 expert cache's maintenance for the step about to start; outside any capture.
pub fn cache_step() -> Result<()> {
    let caches = CACHES
        .lock()
        .expect("expert caches poisoned")
        .iter()
        .filter_map(Weak::upgrade)
        .collect::<Vec<_>>();
    for cache in caches {
        cache.lock().expect("expert cache poisoned").step()?;
    }
    Ok(())
}
