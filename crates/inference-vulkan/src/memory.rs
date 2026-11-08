use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use ash::vk;

use crate::{Context, Error, Result};

/// A storage binding past this reads zeros, so no buffer here is larger.
pub(crate) const MAX_BINDING_BYTES: u64 = 4 << 30;
// Buffers up to this size are pooled in classes from the smallest, powers of two up to FINE_CLASS_BYTES and eighths of
// one past it (a power of two would round a large activation up by almost as much again); larger ones are allocated
// to size and freed when dropped
const MIN_POOLED_BYTES: u64 = 256;
const FINE_CLASS_BYTES: u64 = 1 << 20;
const MAX_POOLED_BYTES: u64 = 1 << 30;
// The most idle host memory the pool holds before a dropped buffer is freed instead (on an iGPU, the model's RAM):
// a default 1024-token prompt chunk's working set (Lightning: ~4.5 GB), so it is reused rather than freed each chunk.
// Idle device memory is not capped: a prompt's working set cycles through it, and an allocation that runs out of
// device memory frees it first.
const MAX_POOL_BYTES: u64 = 6 << 30;
// Whether a device allocation has fallen back to host memory, so the fallback is reported once
static SPILLED: AtomicBool = AtomicBool::new(false);
// The device memory every buffer together leaves free, short of the device's budget: allocating to the edge of a
// discrete GPU's memory stalls its driver for seconds
const ROOM_MARGIN_BYTES: u64 = 512 << 20;
// An idle device buffer none has reused for this long is an earlier phase's size (a speech model's subsampling, then
// its encoder, then a diarizer each take their own classes): a new allocation frees it first, so the pool holds one
// phase's working set rather than the sum of every phase's
const STALE_IDLE: std::time::Duration = std::time::Duration::from_millis(150);
// Idle device bytes the pool keeps however stale, so a model's whole working set (a prompt's, paused between
// requests) survives; past them a miss frees the stale ones, and an idle engine the rest
const STALE_KEEP_BYTES: u64 = 1 << 30;
/// The device memory a discrete GPU's long-lived buffers leave free, for activations, caches and scratch.
pub const EXACT_RESERVE_BYTES: u64 = 2 << 30;
// A host write to or read of device memory goes through staging buffers of up to this, a submit each
const STAGING_BYTES: usize = 64 << 20;

// an allocation back in the pool, and when it came back
type PoolEntry = (std::time::Instant, Arc<Held>);

/// Buffers nothing references any longer, by placement and capacity, for new ones to reuse: a Vulkan allocation costs
/// tens of microseconds, and a model step allocates hundreds of short-lived tensors.
#[derive(Default)]
pub(crate) struct BufferPool {
    // oldest first in each class
    free: HashMap<(bool, u64), Vec<PoolEntry>>,
    // idle host bytes (capped), and idle device bytes (counted, not capped)
    bytes: u64,
    device_bytes: u64,
}

// a buffer, its memory and the memory's mapping
#[derive(Clone, Copy)]
struct Allocation {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    ptr: *mut u8,
    size: u64,
}

/// A pooled allocation, shared by the buffer using it, the pool while it is idle, and the batches whose commands use
/// it (they retain this, not the buffer): freed when the last of them drops it. A device buffer goes back to the pool
/// when its last user drops it, before the work recorded on it finishes, and the work recorded after takes it: the
/// stream orders every command after the ones before (a barrier follows each), as CUDA's allocator reuses a block in
/// stream order. A mapped one taken so (`Buffer::early`) is written on the stream, not in place, while batches hold it.
pub(crate) struct Held {
    // weak, so the pool's entries do not keep their context alive
    context: std::sync::Weak<Context>,
    allocation: Allocation,
}

// SAFETY: the handles and mapping are only used by whoever holds the buffer the allocation backs
unsafe impl Send for Held {}
unsafe impl Sync for Held {}

impl Held {
    // whether nothing but the pool holds it: no buffer, and no batch whose commands may still run on it
    fn idle(held: &Arc<Held>) -> bool {
        Arc::strong_count(held) == 1
    }
}

impl Drop for Held {
    fn drop(&mut self) {
        // a context dropping or shut down frees (or leaks) its memory itself
        if let Some(context) = self.context.upgrade()
            && !context.is_shut()
        {
            // SAFETY: the last holder is gone, so no command still uses it
            unsafe { self.allocation.release(&context) };
        }
    }
}

impl BufferPool {
    pub(crate) fn device_bytes(&self) -> u64 {
        self.device_bytes
    }

    /// # Safety
    /// No submit may still use the buffers.
    pub(crate) unsafe fn destroy(&mut self, device: &ash::Device) {
        for (_, held) in self.free.drain().flat_map(|(_, allocations)| allocations) {
            // SAFETY: the caller's contract; the held allocation's own drop skips a dropping or shut context
            unsafe { held.allocation.free(device) };
        }
        self.bytes = 0;
        self.device_bytes = 0;
    }
}

impl Allocation {
    // frees it, and takes device memory off the context's count
    unsafe fn release(self, context: &Context) {
        if self.ptr.is_null() {
            context.device_bytes.fetch_sub(self.size, Ordering::Relaxed);
        }
        // SAFETY: the caller's contract
        unsafe { self.free(&context.device) };
    }

    unsafe fn free(self, device: &ash::Device) {
        // SAFETY: the caller's contract that the GPU no longer uses it
        unsafe {
            if !self.ptr.is_null() {
                device.unmap_memory(self.memory);
            }
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}

// the capacity a buffer of `len` bytes is allocated with
fn capacity(len: u64) -> u64 {
    if len > MAX_POOLED_BYTES {
        len
    } else if len > FINE_CLASS_BYTES {
        len.next_multiple_of(len.next_power_of_two() / 8)
    } else {
        len.next_power_of_two().max(MIN_POOLED_BYTES)
    }
}

/// Memory a kernel binds and reads from `offset`: a buffer, or an import.
pub trait Bound: std::any::Any + Send + Sync {
    fn vk_buffer(&self) -> vk::Buffer;
    fn offset(&self) -> u64;
    /// The bytes from `offset` it holds.
    fn bytes(&self) -> usize;
}

impl Bound for Buffer {
    fn vk_buffer(&self) -> vk::Buffer {
        self.buffer
    }

    fn offset(&self) -> u64 {
        0
    }

    fn bytes(&self) -> usize {
        self.len
    }
}

impl Bound for ImportedMemory {
    fn vk_buffer(&self) -> vk::Buffer {
        self.buffer
    }

    fn offset(&self) -> u64 {
        self.offset
    }

    fn bytes(&self) -> usize {
        self.len
    }
}

/// Memory a kernel reads and writes: in host RAM, mapped for the CPU as well (cached when the device offers it: an iGPU
/// reads cached host pages as fast as uncached ones), on a device that shares the host's memory; in the device's own
/// memory on a discrete one, which the host reaches through staging copies on the stream.
pub struct Buffer {
    context: Arc<Context>,
    pub(crate) buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    // null for device memory
    ptr: *mut u8,
    len: usize,
    capacity: u64,
    // a pooled buffer's allocation; a buffer without one frees its own when it drops
    held: Option<Arc<Held>>,
    // mapped and taken from the pool while batches recorded before it still held it: the CPU writes it on the stream
    early: AtomicBool,
}

// SAFETY: the mapping is plain memory, and the Vulkan handles are only used through the context's synchronized calls
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    /// A buffer where the device's kernels read fastest: the device's memory, or mapped host RAM on a unified device.
    pub fn new(context: &Arc<Context>, len: usize) -> Result<Self> {
        Self::placed(context, len, !context.unified, false)
    }

    /// A buffer as [`Self::new`] places it, for the CPU to write before the GPU uses it: where mapped, one no submitted
    /// work still uses, so the write lands in place.
    pub fn for_upload(context: &Arc<Context>, len: usize) -> Result<Self> {
        Self::placed(context, len, !context.unified, true)
    }

    /// A buffer in host RAM the CPU maps, wherever the device's own memory is, which no submitted work still uses (the
    /// CPU writes it in place).
    pub fn host(context: &Arc<Context>, len: usize) -> Result<Self> {
        Self::placed(context, len, false, true)
    }

    // `idle_only` keeps a mapped buffer from being taken in stream order
    fn placed(
        context: &Arc<Context>,
        len: usize,
        device_local: bool,
        idle_only: bool,
    ) -> Result<Self> {
        if len == 0 || len as u64 > MAX_BINDING_BYTES {
            return Err(Error::Invalid(format!("a buffer of {len} bytes")));
        }
        let capacity = capacity(len as u64);
        context
            .allocated
            .fetch_add(capacity, std::sync::atomic::Ordering::Relaxed);
        let take = || {
            context.pool.lock().expect("vulkan pool poisoned").take(
                device_local,
                capacity,
                idle_only,
            )
        };
        let mut pooled = take();
        // short of room, finished work's buffers come back before any idle one is freed: a prompt's temporaries
        // (recorded a forward ahead of the GPU) are the sizes the next ones take
        while pooled.is_none()
            && device_local
            && context.device_free() < capacity + ROOM_MARGIN_BYTES
            && context.try_wait_oldest()?
        {
            pooled = take();
        }
        if let Some(held) = pooled {
            let Allocation {
                buffer,
                memory,
                ptr,
                ..
            } = held.allocation;
            let early = !ptr.is_null() && !Held::idle(&held);
            return Ok(Self {
                context: context.clone(),
                buffer,
                memory,
                ptr,
                len,
                capacity,
                held: Some(held),
                early: AtomicBool::new(early),
            });
        }
        Self::allocate(context, len, capacity, device_local, true)
    }

    /// A buffer of `len` bytes placed as `new` places it, allocated to size and freed when dropped, for what lives as
    /// long as the model (weights), which a size class would round up by as much again. On a discrete GPU these stay
    /// in device memory while everything live there leaves [`EXACT_RESERVE_BYTES`] of it for the short-lived buffers,
    /// then go to host memory.
    pub fn exact(context: &Arc<Context>, len: usize) -> Result<Self> {
        if len == 0 || len as u64 > MAX_BINDING_BYTES {
            return Err(Error::Invalid(format!("a buffer of {len} bytes")));
        }
        // free now, as the driver counts it (the paged cache, the weights placed so far and other processes all
        // taken), plus idle pooled buffers, which an allocation frees first
        let idle = context
            .pool
            .lock()
            .expect("vulkan pool poisoned")
            .device_bytes;
        let fits = len as u64 + EXACT_RESERVE_BYTES <= context.device_free() + idle;
        Self::allocate(context, len, len as u64, !context.unified && fits, false)
    }

    // a fresh allocation; device memory a discrete GPU has run out of comes from the buffers finished work held (the
    // CPU records a whole forward ahead of the GPU, its dropped temporaries alive until their batch's fence) and the
    // pool's idle ones, freed, and past them from host RAM, which the GPU reads over the bus
    fn allocate(
        context: &Arc<Context>,
        len: usize,
        capacity: u64,
        device_local: bool,
        pooled: bool,
    ) -> Result<Self> {
        context.live()?;
        let out_of_memory =
            |r: &Result<Self>| matches!(r, Err(Error::Vk(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)));
        if device_local {
            let stale = context
                .pool
                .lock()
                .expect("vulkan pool poisoned")
                .drain_stale(std::time::Instant::now());
            drop(stale);
            make_room(context, capacity)?;
        }
        let first = Self::allocate_once(context, len, capacity, device_local, pooled);
        if !device_local || !out_of_memory(&first) {
            return first;
        }
        context.try_synchronize()?;
        let idle = context
            .pool
            .lock()
            .expect("vulkan pool poisoned")
            .drain(true);
        drop(idle);
        let retried = Self::allocate_once(context, len, capacity, true, pooled);
        if !out_of_memory(&retried) {
            return retried;
        }
        if !SPILLED.swap(true, Ordering::Relaxed) {
            let held = context.device_bytes.load(Ordering::Relaxed) >> 20;
            eprintln!(
                "vulkan: device memory is full ({} MiB asked, {held} MiB held, {} MiB free); further buffers live in \
                 host memory, read over the bus",
                capacity >> 20,
                context.device_free() >> 20
            );
        }
        Self::allocate_once(context, len, capacity, false, pooled)
    }

    fn allocate_once(
        context: &Arc<Context>,
        len: usize,
        capacity: u64,
        device_local: bool,
        pooled: bool,
    ) -> Result<Self> {
        let device = &context.device;
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        let info = vk::BufferCreateInfo::default()
            .size(capacity)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: the create info outlives the call
        let buffer = unsafe { device.create_buffer(&info, None)? };
        // SAFETY: the buffer was created on this device
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let bits = requirements.memory_type_bits;
        let visible =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let memory_type = if device_local {
            context.memory_type(bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
        } else {
            context
                .memory_type(bits, visible | vk::MemoryPropertyFlags::HOST_CACHED)
                .or_else(|| context.memory_type(bits, visible))
        };
        let Some(memory_type) = memory_type else {
            // SAFETY: the buffer is unused
            unsafe { device.destroy_buffer(buffer, None) };
            return Err(Error::NoMemoryType);
        };
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);
        // SAFETY: the allocation is bound once, and a host-visible one is mapped whole for the buffer's life
        unsafe {
            let memory = match device.allocate_memory(&alloc, None) {
                Ok(memory) => memory,
                Err(err) => {
                    device.destroy_buffer(buffer, None);
                    return Err(err.into());
                }
            };
            device.bind_buffer_memory(buffer, memory, 0)?;
            let ptr = if device_local {
                std::ptr::null_mut()
            } else {
                device
                    .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())?
                    .cast()
            };
            if device_local {
                context.device_bytes.fetch_add(capacity, Ordering::Relaxed);
            }
            let held = pooled.then(|| {
                Arc::new(Held {
                    context: Arc::downgrade(context),
                    allocation: Allocation {
                        buffer,
                        memory,
                        ptr,
                        size: capacity,
                    },
                })
            });
            Ok(Self {
                context: context.clone(),
                buffer,
                memory,
                ptr,
                len,
                capacity,
                held,
                early: AtomicBool::new(false),
            })
        }
    }

    // what a command using it retains: its allocation where pooled, else the buffer
    pub(crate) fn retained(self: Arc<Self>) -> crate::Retained {
        match &self.held {
            Some(held) => held.clone(),
            None => self,
        }
    }

    /// A buffer holding `bytes`.
    pub fn with_bytes(context: &Arc<Context>, bytes: &[u8]) -> Result<Self> {
        let mut buffer = Self::for_upload(context, bytes.len())?;
        buffer.write(0, bytes)?;
        Ok(buffer)
    }

    /// A buffer holding `values`.
    pub fn with_f32(context: &Arc<Context>, values: &[f32]) -> Result<Self> {
        Self::with_bytes(context, f32_bytes(values))
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether the CPU maps the buffer, so its bytes are written and read in place.
    pub(crate) fn context(&self) -> &Arc<Context> {
        &self.context
    }

    pub fn is_mapped(&self) -> bool {
        !self.ptr.is_null()
    }

    // whether the CPU may write the mapping in place: mapped, and no batch recorded before the buffer was taken may
    // still use it
    fn in_place(&self) -> bool {
        if self.ptr.is_null() {
            return false;
        }
        if !self.early.load(Ordering::Relaxed) {
            return true;
        }
        let free = self
            .held
            .as_ref()
            .is_none_or(|held| Arc::strong_count(held) == 1);
        if free {
            self.early.store(false, Ordering::Relaxed);
        }
        free
    }

    /// The mapping's first byte, when mapped and no earlier work may still use it; what the CPU writes there the GPU
    /// must not be using.
    pub fn mapped_ptr(&self) -> Option<std::ptr::NonNull<u8>> {
        if !self.in_place() {
            return None;
        }
        std::ptr::NonNull::new(self.ptr)
    }

    /// Writes `bytes` at `offset`: in place when mapped, which the GPU must not be using, else as a copy recorded on
    /// the stream, ordered after what is recorded before it.
    pub fn write(&mut self, offset: usize, bytes: &[u8]) -> Result<()> {
        self.context.forget_quantized(self.buffer);
        if offset + bytes.len() > self.len {
            return Err(Error::Invalid(format!(
                "{} bytes at {offset} of a {}-byte buffer",
                bytes.len(),
                self.len
            )));
        }
        if self.in_place() {
            // SAFETY: the range is in the mapping, which nothing else writes while `&mut self` is held
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.ptr.add(offset), bytes.len())
            };
            return Ok(());
        }
        for (i, chunk) in bytes.chunks(STAGING_BYTES).enumerate() {
            let mut staging = Self::host(&self.context, chunk.len())?;
            staging.write(0, chunk)?;
            let at = (offset + i * STAGING_BYTES) as u64;
            let staging = Arc::new(staging);
            self.context.record(|recorder| {
                recorder.copy(staging.buffer, 0, self.buffer, at, chunk.len() as u64);
                recorder.retain(staging.clone());
                self.retain_in(recorder);
                Ok(())
            })?;
            self.context.flush()?;
        }
        Ok(())
    }

    /// Fills the buffer with copies of `word`, ordered as [`Self::write`] is.
    pub fn fill(&mut self, word: u32) -> Result<()> {
        self.context.forget_quantized(self.buffer);
        if self.in_place() {
            let words = self.len / 4;
            // SAFETY: the mapping is page aligned and `len` bytes long, unique while `&mut self` is held
            unsafe { std::slice::from_raw_parts_mut(self.ptr.cast::<u32>(), words) }.fill(word);
            let tail = &word.to_le_bytes()[..self.len % 4];
            // SAFETY: as above
            unsafe {
                std::ptr::copy_nonoverlapping(tail.as_ptr(), self.ptr.add(words * 4), tail.len())
            };
            return Ok(());
        }
        let (buffer, size) = (self.buffer, self.capacity);
        self.context.record(|recorder| {
            recorder.fill(buffer, size, word);
            self.retain_in(recorder);
            Ok(())
        })
    }

    // a pooled buffer's allocation kept by the batch a command writing it is recorded in: dropped before the command
    // runs, it would otherwise look idle, and a mapped one be written in place under it
    fn retain_in(&self, recorder: &crate::Recorder<'_>) {
        if let Some(held) = &self.held {
            recorder.retain(held.clone());
        }
    }

    /// Reads the buffer's first `out.len()` bytes once the GPU has finished everything recorded so far.
    pub fn read_into(&self, out: &mut [u8]) -> Result<()> {
        let len = out.len().min(self.len);
        let out = &mut out[..len];
        if self.is_mapped() {
            self.context.synchronize()?;
            // SAFETY: the range is in the mapping, which the GPU no longer writes
            unsafe { std::ptr::copy_nonoverlapping(self.ptr, out.as_mut_ptr(), len) };
            return Ok(());
        }
        for (i, chunk) in out.chunks_mut(STAGING_BYTES).enumerate() {
            let staging = Self::host(&self.context, chunk.len())?;
            let at = (i * STAGING_BYTES) as u64;
            self.context.record(|recorder| {
                recorder.copy(self.buffer, at, staging.buffer, 0, chunk.len() as u64);
                Ok(())
            })?;
            staging.read_into(chunk)?;
        }
        Ok(())
    }

    /// The buffer's bytes once the GPU has finished everything recorded so far.
    pub fn read(&self) -> Result<Vec<u8>> {
        let mut out = vec![0u8; self.len];
        self.read_into(&mut out)?;
        Ok(out)
    }

    /// The buffer's bytes as `f32`s, as [`Self::read`].
    pub fn read_f32(&self) -> Result<Vec<f32>> {
        let mut out = vec![0f32; self.len / 4];
        // SAFETY: any bytes are a valid f32
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), out.len() * 4) };
        self.read_into(bytes)?;
        Ok(out)
    }
}

/// `values`' bytes.
pub fn f32_bytes(values: &[f32]) -> &[u8] {
    // SAFETY: an f32's bytes are initialized and u8 has no alignment
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

impl Drop for Buffer {
    // a recorded command retains what it uses (a pooled buffer's allocation, an unpooled buffer itself), so a freed
    // allocation is one nothing submitted uses
    fn drop(&mut self) {
        // the device is gone: its memory went with it
        if self.context.is_shut() {
            return;
        }
        match self.held.take() {
            Some(held) => {
                let refused = self
                    .context
                    .pool
                    .lock()
                    .expect("vulkan pool poisoned")
                    .give(self.ptr.is_null(), self.capacity, held);
                drop(refused);
            }
            None => {
                let allocation = Allocation {
                    buffer: self.buffer,
                    memory: self.memory,
                    ptr: self.ptr,
                    size: self.capacity,
                };
                // SAFETY: see above
                unsafe { allocation.release(&self.context) };
            }
        }
    }
}

impl Context {
    /// Frees the pool's idle buffers, for an idle engine whose next request may be hours off: on a unified-memory device
    /// all of them (host RAM held for reuse), on a discrete one the device memory past STALE_KEEP_BYTES, largest first
    /// (a long request's temporaries, GBs; a model's ordinary working set stays for its next request).
    pub fn release_idle(&self) {
        // the last request's batches finish first, so their temporaries are idle too
        if !self.try_synchronize().unwrap_or(false) {
            return;
        }
        let mut pool = self.pool.lock().expect("vulkan pool poisoned");
        let mut idle = pool.drain(false);
        while !self.is_unified() && pool.device_bytes > STALE_KEEP_BYTES {
            match pool.pop_largest(true) {
                Some(held) => idle.push(held),
                None => break,
            }
        }
        drop(pool);
        drop(idle);
    }
}

// frees idle device buffers, the largest first, until `capacity` more device memory leaves ROOM_MARGIN_BYTES free;
// once the pool is empty, the finished work's buffers come back to it first (once)
fn make_room(context: &Context, capacity: u64) -> Result<()> {
    let mut synchronized = false;
    while context.device_free() < capacity + ROOM_MARGIN_BYTES {
        let largest = context
            .pool
            .lock()
            .expect("vulkan pool poisoned")
            .pop_largest(true);
        match largest {
            Some(held) => drop(held),
            None if !synchronized => {
                synchronized = true;
                if !context.try_synchronize()? {
                    break;
                }
            }
            None => break,
        }
    }
    Ok(())
}

impl BufferPool {
    // the largest allocation of one placement that nothing but the pool holds, out of the pool: freed as it drops
    fn pop_largest(&mut self, device_local: bool) -> Option<Arc<Held>> {
        let (key, at) = self
            .free
            .iter()
            .filter(|(key, _)| key.0 == device_local)
            .filter_map(|(&key, entries)| {
                entries
                    .iter()
                    .rposition(|(_, held)| Held::idle(held))
                    .map(|at| (key, at))
            })
            .max_by_key(|&((_, capacity), _)| capacity)?;
        let (_, held) = self.free.get_mut(&key)?.remove(at);
        self.forget(key, 1);
        Some(held)
    }

    // idle bytes taken off the count as `n` entries of class `key` leave
    fn forget(&mut self, (local, capacity): (bool, u64), n: usize) {
        if local {
            self.device_bytes -= capacity * n as u64;
        } else {
            self.bytes -= capacity * n as u64;
        }
    }

    // past STALE_KEEP_BYTES idle, the device allocations none has taken for STALE_IDLE, out of the pool
    fn drain_stale(&mut self, now: std::time::Instant) -> Vec<Arc<Held>> {
        let mut out = Vec::new();
        if self.device_bytes <= STALE_KEEP_BYTES {
            return out;
        }
        let keys: Vec<_> = self.free.keys().copied().filter(|key| key.0).collect();
        for key in keys {
            let entries = self.free.get_mut(&key).expect("a listed class");
            let stale = entries
                .iter()
                .take_while(|(since, _)| now.duration_since(*since) >= STALE_IDLE)
                .count();
            out.extend(entries.drain(..stale).map(|(_, held)| held));
            self.forget(key, stale);
        }
        out
    }

    // every allocation of one placement, out of the pool
    fn drain(&mut self, device_local: bool) -> Vec<Arc<Held>> {
        let keys: Vec<_> = self
            .free
            .keys()
            .copied()
            .filter(|key| key.0 == device_local)
            .collect();
        let mut out = Vec::new();
        for key in keys {
            let entries = self.free.remove(&key).expect("a listed class");
            self.forget(key, entries.len());
            out.extend(entries.into_iter().map(|(_, held)| held));
        }
        out
    }

    // the newest allocation of a class; a mapped one that no batch holds first, and with `idle_only` only such a one
    fn take(&mut self, device_local: bool, capacity: u64, idle_only: bool) -> Option<Arc<Held>> {
        let key = (device_local, capacity);
        let entries = self.free.get_mut(&key)?;
        let newest = entries.len().checked_sub(1)?;
        let at = match entries.iter().rposition(|(_, held)| Held::idle(held)) {
            _ if device_local => newest,
            Some(at) => at,
            None if idle_only => return None,
            None => newest,
        };
        let (_, held) = entries.remove(at);
        self.forget(key, 1);
        Some(held)
    }

    // keeps `held` for reuse, or hands it back (to be freed as it drops) when it is not a pooled size or the pool is
    // full
    fn give(&mut self, device_local: bool, capacity: u64, held: Arc<Held>) -> Option<Arc<Held>> {
        if capacity > MAX_POOLED_BYTES || (!device_local && self.bytes + capacity > MAX_POOL_BYTES)
        {
            return Some(held);
        }
        if device_local {
            self.device_bytes += capacity;
        } else {
            self.bytes += capacity;
        }
        self.free
            .entry((device_local, capacity))
            .or_default()
            .push((std::time::Instant::now(), held));
        None
    }
}

/// A host allocation the device reads in place: its range rounded out to the import alignment, bound as one buffer,
/// with `offset` the allocation's first byte within it. Every submit while it lives validates its pages (~0.8 us a MB on
/// RADV), so it suits one large buffer streamed in long submits, not a model's weights.
pub struct ImportedMemory {
    context: Arc<Context>,
    pub(crate) buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    pub(crate) offset: u64,
    len: usize,
}

// SAFETY: as for `Buffer`; the imported pages belong to the caller, who keeps them alive
unsafe impl Send for ImportedMemory {}
unsafe impl Sync for ImportedMemory {}

impl ImportedMemory {
    /// # Safety
    /// `bytes`, rounded out to whole pages of the import alignment, must stay mapped and unchanged for as long as the
    /// import lives; a page past either end belongs to the import too, so it must be mapped memory as well.
    pub unsafe fn new(context: &Arc<Context>, bytes: &[u8]) -> Result<Self> {
        let align = context.import_alignment;
        let start = bytes.as_ptr() as u64;
        let base = start & !(align - 1);
        let size = (start + bytes.len() as u64).next_multiple_of(align) - base;
        if bytes.is_empty() || size > MAX_BINDING_BYTES {
            return Err(Error::Invalid(format!(
                "an import of {} bytes",
                bytes.len()
            )));
        }
        let handle = vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT;
        let device = &context.device;
        let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(handle);
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut external);
        // SAFETY: the pages are the caller's to keep mapped (this fn's contract); the create infos outlive their calls
        unsafe {
            let buffer = device.create_buffer(&info, None)?;
            let requirements = device.get_buffer_memory_requirements(buffer);
            let mut host = vk::MemoryHostPointerPropertiesEXT::default();
            (context
                .host_memory
                .fp()
                .get_memory_host_pointer_properties_ext)(
                context.host_memory.device(),
                handle,
                base as *const _,
                &mut host,
            )
            .result()?;
            let memory_type = context
                .memory_type(
                    requirements.memory_type_bits & host.memory_type_bits,
                    vk::MemoryPropertyFlags::empty(),
                )
                .ok_or(Error::NoMemoryType)?;
            let mut import = vk::ImportMemoryHostPointerInfoEXT::default()
                .handle_type(handle)
                .host_pointer(base as *mut _);
            let alloc = vk::MemoryAllocateInfo::default()
                .allocation_size(size)
                .memory_type_index(memory_type)
                .push_next(&mut import);
            let memory = device.allocate_memory(&alloc, None)?;
            device.bind_buffer_memory(buffer, memory, 0)?;
            Ok(Self {
                context: context.clone(),
                buffer,
                memory,
                offset: start - base,
                len: bytes.len(),
            })
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for ImportedMemory {
    fn drop(&mut self) {
        if self.context.is_shut() {
            return;
        }
        // SAFETY: every submit is waited on, so the GPU no longer reads the pages
        unsafe {
            self.context.device.destroy_buffer(self.buffer, None);
            self.context.device.free_memory(self.memory, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // past one staging buffer, so a device-memory write and read take several copies
    const LEN: usize = STAGING_BYTES + 4096 + 12;

    #[test]
    fn buffers_take_writes_fills_and_reads_wherever_they_live() -> Result<()> {
        let context = match Context::new() {
            Ok(context) => context,
            Err(err) => {
                eprintln!("skipping: {err}");
                return Ok(());
            }
        };
        let bytes: Vec<u8> = (0..LEN).map(|i| (i * 31 % 251) as u8).collect();
        for buffer in [Buffer::new(&context, LEN)?, Buffer::host(&context, LEN)?] {
            let mut buffer = buffer;
            buffer.fill(0x0302_0100)?;
            let filled = buffer.read()?;
            assert!(filled.iter().enumerate().all(|(i, &b)| b == (i % 4) as u8));
            buffer.write(5, &bytes[..LEN - 5])?;
            let got = buffer.read()?;
            assert_eq!(&got[..5], &[0, 1, 2, 3, 0]);
            assert_eq!(&got[5..], &bytes[..LEN - 5]);
        }
        assert_eq!(
            Buffer::new(&context, LEN)?.is_mapped(),
            context.is_unified()
        );
        Ok(())
    }

    // dropped buffers wait in the pool for reuse until the idle release hands them back: all of an iGPU's host RAM, a
    // discrete GPU's device memory past STALE_KEEP_BYTES
    #[test]
    fn an_idle_release_empties_the_pool() -> Result<()> {
        let context = match Context::new() {
            Ok(context) => context,
            Err(err) => {
                eprintln!("skipping: {err}");
                return Ok(());
            }
        };
        let idle = |context: &Context| {
            let pool = context.pool.lock().expect("vulkan pool poisoned");
            pool.bytes + pool.device_bytes
        };
        context.release_idle();
        let before = idle(&context);
        drop(Buffer::new(&context, 3 << 20)?);
        assert!(idle(&context) > before);
        context.release_idle();
        if context.is_unified() {
            assert_eq!(idle(&context), 0);
            return Ok(());
        }
        // a discrete GPU keeps up to STALE_KEEP_BYTES of it, and frees what is past them
        assert_eq!(idle(&context), before + capacity(3 << 20));
        let big: Vec<_> = (0..STALE_KEEP_BYTES / (256 << 20) + 2)
            .map(|_| Buffer::new(&context, 256 << 20))
            .collect::<Result<_>>()?;
        drop(big);
        context.release_idle();
        assert!(idle(&context) <= STALE_KEEP_BYTES);
        Ok(())
    }

    // a discrete GPU's allocation that misses, past STALE_KEEP_BYTES idle, frees the idle buffers none has reused for
    // STALE_IDLE (an earlier phase's sizes), and keeps those just given back
    #[test]
    fn a_missed_allocation_frees_stale_idle_buffers() -> Result<()> {
        let context = match Context::new() {
            Ok(context) => context,
            Err(err) => {
                eprintln!("skipping: {err}");
                return Ok(());
            }
        };
        if context.is_unified() {
            return Ok(());
        }
        context.release_idle();
        let idle = |context: &Context| context.pool.lock().unwrap().device_bytes;
        // under STALE_KEEP_BYTES idle, a stale buffer stays
        drop(Buffer::new(&context, 3 << 20)?);
        std::thread::sleep(STALE_IDLE + std::time::Duration::from_millis(20));
        drop(Buffer::new(&context, 5 << 20)?);
        assert_eq!(idle(&context), capacity(3 << 20) + capacity(5 << 20));
        context.release_idle();
        // past it, the stale classes go and the 64 MiB one just given back stays
        let big: Vec<_> = (0..STALE_KEEP_BYTES / (256 << 20) + 1)
            .map(|_| Buffer::new(&context, 256 << 20))
            .collect::<Result<_>>()?;
        drop(big);
        std::thread::sleep(STALE_IDLE + std::time::Duration::from_millis(20));
        drop(Buffer::new(&context, 64 << 20)?);
        assert_eq!(idle(&context), capacity(64 << 20));
        let _fresh = Buffer::new(&context, 7 << 20)?;
        assert_eq!(idle(&context), capacity(64 << 20));
        Ok(())
    }

    // a buffer its last user drops while a copy from it is pending is the next buffer of its size, before the copy
    // finishes (a mapped one too, written on the stream meanwhile); the stream orders that buffer's fill after the
    // copy, which still reads the old values
    #[test]
    fn a_dropped_buffer_is_reused_in_stream_order() -> Result<()> {
        let context = match Context::new() {
            Ok(context) => context,
            Err(err) => {
                eprintln!("skipping: {err}");
                return Ok(());
            }
        };
        const BYTES: usize = 8 << 20;
        let source = Arc::new(Buffer::new(&context, BYTES)?);
        let copy = Arc::new(Buffer::new(&context, BYTES)?);
        let reused_handle = source.buffer;
        context.record(|r| {
            r.fill(source.buffer, BYTES as u64, 1);
            r.copy(source.buffer, 0, copy.buffer, 0, BYTES as u64);
            r.retain(source.clone());
            r.retain(copy.clone());
            Ok(())
        })?;
        drop(source);
        let mut next = Buffer::new(&context, BYTES)?;
        assert_eq!(
            next.buffer, reused_handle,
            "the dropped allocation is taken again"
        );
        next.fill(2)?;
        let words = |bytes: Vec<u8>| -> Vec<u32> {
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|w| u32::from_le_bytes(*w))
                .collect()
        };
        assert!(words(copy.read()?).iter().all(|&w| w == 1));
        assert!(words(next.read()?).iter().all(|&w| w == 2));
        Ok(())
    }

    // a buffer taken in stream order, filled on the stream and dropped before the fill runs, is not an upload's once
    // the work before it finishes: the fill's batch holds it, so the upload's bytes are what it reads back
    #[test]
    fn a_pending_fill_keeps_its_buffer_from_an_upload() -> Result<()> {
        let context = match Context::new() {
            Ok(context) => context,
            Err(err) => {
                eprintln!("skipping: {err}");
                return Ok(());
            }
        };
        const BYTES: usize = 8 << 20;
        let source = Arc::new(Buffer::new(&context, BYTES)?);
        let copy = Arc::new(Buffer::new(&context, BYTES)?);
        context.record(|r| {
            r.copy(source.buffer, 0, copy.buffer, 0, BYTES as u64);
            r.retain(source.clone());
            r.retain(copy.clone());
            Ok(())
        })?;
        context.flush()?;
        drop(source);
        let mut next = Buffer::new(&context, BYTES)?;
        next.fill(2)?;
        drop(next);
        while context.try_wait_oldest()? && Arc::strong_count(&copy) > 1 {}
        let upload = Buffer::with_bytes(&context, &[3u8; BYTES])?;
        assert!(upload.read()?.iter().all(|&b| b == 3));
        Ok(())
    }

    // device memory is counted while a buffer lives and while the pool holds it, and released largest first
    #[test]
    fn device_bytes_follow_buffers_through_the_pool() -> Result<()> {
        let context = match Context::new() {
            Ok(context) => context,
            Err(err) => {
                eprintln!("skipping: {err}");
                return Ok(());
            }
        };
        let base = context.device_bytes.load(Ordering::Relaxed);
        let sizes = [3 << 20, 40 << 20, 5 << 20];
        let buffers: Vec<_> = sizes
            .iter()
            .map(|&len| Buffer::new(&context, len))
            .collect::<Result<_>>()?;
        let held: u64 = sizes.iter().map(|&len| capacity(len as u64)).sum();
        let device = !context.is_unified();
        let expect = |bytes| if device { base + bytes } else { base };
        assert_eq!(context.device_bytes.load(Ordering::Relaxed), expect(held));
        drop(buffers);
        assert_eq!(context.device_bytes.load(Ordering::Relaxed), expect(held));
        if device {
            let largest = context.pool.lock().unwrap().pop_largest(true).unwrap();
            assert_eq!(largest.allocation.size, capacity(40 << 20));
            // the pool's was the last hold, so it frees as it drops
            drop(largest);
            assert_eq!(
                context.device_bytes.load(Ordering::Relaxed),
                expect(held - capacity(40 << 20))
            );
        }
        Ok(())
    }

    // the copy engine's device-to-device rate, the ceiling a kernel's reads and writes are measured against
    #[test]
    #[ignore = "benchmark"]
    fn copy_bandwidth() -> Result<()> {
        let context = Context::new()?;
        const LEN: usize = 256 << 20;
        let (a, b) = (
            Arc::new(Buffer::new(&context, LEN)?),
            Arc::new(Buffer::new(&context, LEN)?),
        );
        for _ in 0..3 {
            let t0 = std::time::Instant::now();
            context.record(|r| {
                for _ in 0..20 {
                    r.copy(a.buffer, 0, b.buffer, 0, LEN as u64);
                }
                Ok(())
            })?;
            context.synchronize()?;
            eprintln!(
                "copy: {:.0} GB/s",
                20.0 * 2.0 * LEN as f64 / t0.elapsed().as_secs_f64() / 1e9
            );
        }
        Ok(())
    }
}
