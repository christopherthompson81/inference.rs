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
/// The device memory a discrete GPU's long-lived buffers leave free, for activations, caches and scratch.
pub const EXACT_RESERVE_BYTES: u64 = 2 << 30;
// A host write to or read of device memory goes through staging buffers of up to this, a submit each
const STAGING_BYTES: usize = 64 << 20;

/// Buffers nothing references any longer, by placement and capacity, for new ones to reuse: a Vulkan allocation costs
/// tens of microseconds, and a model step allocates hundreds of short-lived tensors.
#[derive(Default)]
pub(crate) struct BufferPool {
    free: HashMap<(bool, u64), Vec<Allocation>>,
    // idle host bytes (capped), and idle device bytes (counted, not capped)
    bytes: u64,
    device_bytes: u64,
}

// a buffer, its memory and the memory's mapping
struct Allocation {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    ptr: *mut u8,
    size: u64,
}

// SAFETY: the handles and mapping are only used by whoever takes the allocation out of the pool's lock
unsafe impl Send for Allocation {}

impl BufferPool {
    pub(crate) fn device_bytes(&self) -> u64 {
        self.device_bytes
    }

    /// # Safety
    /// No submit may still use the buffers.
    pub(crate) unsafe fn destroy(&mut self, device: &ash::Device) {
        for allocation in self.free.drain().flat_map(|(_, allocations)| allocations) {
            // SAFETY: the caller's contract
            unsafe { allocation.free(device) };
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
    pooled: bool,
}

// SAFETY: the mapping is plain memory, and the Vulkan handles are only used through the context's synchronized calls
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    /// A buffer where the device's kernels read fastest: the device's memory, or mapped host RAM on a unified device.
    pub fn new(context: &Arc<Context>, len: usize) -> Result<Self> {
        Self::placed(context, len, !context.unified)
    }

    /// A buffer in host RAM the CPU maps, wherever the device's own memory is.
    pub fn host(context: &Arc<Context>, len: usize) -> Result<Self> {
        Self::placed(context, len, false)
    }

    fn placed(context: &Arc<Context>, len: usize, device_local: bool) -> Result<Self> {
        if len == 0 || len as u64 > MAX_BINDING_BYTES {
            return Err(Error::Invalid(format!("a buffer of {len} bytes")));
        }
        let capacity = capacity(len as u64);
        let take = || {
            context
                .pool
                .lock()
                .expect("vulkan pool poisoned")
                .take(device_local, capacity)
        };
        let pooled = take();
        if let Some(Allocation {
            buffer,
            memory,
            ptr,
            ..
        }) = pooled
        {
            return Ok(Self {
                context: context.clone(),
                buffer,
                memory,
                ptr,
                len,
                capacity,
                pooled: true,
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
        for allocation in idle {
            // SAFETY: a pooled allocation is one no submit uses any longer
            unsafe { allocation.release(context) };
        }
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
            Ok(Self {
                context: context.clone(),
                buffer,
                memory,
                ptr,
                len,
                capacity,
                pooled,
            })
        }
    }

    /// A buffer holding `bytes`.
    pub fn with_bytes(context: &Arc<Context>, bytes: &[u8]) -> Result<Self> {
        let mut buffer = Self::new(context, bytes.len())?;
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

    /// The mapping's first byte, when mapped; what the CPU writes there the GPU must not be using.
    pub fn mapped_ptr(&self) -> Option<std::ptr::NonNull<u8>> {
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
        if self.is_mapped() {
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
                Ok(())
            })?;
            self.context.flush()?;
        }
        Ok(())
    }

    /// Fills the buffer with copies of `word`, ordered as [`Self::write`] is.
    pub fn fill(&mut self, word: u32) -> Result<()> {
        self.context.forget_quantized(self.buffer);
        if self.is_mapped() {
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
            Ok(())
        })
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
    // a recorded command retains what it uses, so nothing submitted reads a buffer once it drops
    fn drop(&mut self) {
        // the device is gone: its memory went with it
        if self.context.is_shut() {
            return;
        }
        let allocation = Allocation {
            buffer: self.buffer,
            memory: self.memory,
            ptr: self.ptr,
            size: self.capacity,
        };
        let refused = if self.pooled {
            self.context
                .pool
                .lock()
                .expect("vulkan pool poisoned")
                .give(self.ptr.is_null(), self.capacity, allocation)
        } else {
            Some(allocation)
        };
        if let Some(allocation) = refused {
            // SAFETY: see above
            unsafe { allocation.release(&self.context) };
        }
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
            // SAFETY: a pooled allocation is one no submit uses any longer
            Some(allocation) => unsafe { allocation.release(context) },
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
    // the largest idle allocation of one placement, out of the pool
    fn pop_largest(&mut self, device_local: bool) -> Option<Allocation> {
        let key = self
            .free
            .iter()
            .filter(|(key, allocations)| key.0 == device_local && !allocations.is_empty())
            .map(|(&key, _)| key)
            .max_by_key(|&(_, capacity)| capacity)?;
        let allocation = self.free.get_mut(&key)?.pop()?;
        if device_local {
            self.device_bytes -= key.1;
        } else {
            self.bytes -= key.1;
        }
        Some(allocation)
    }

    // every idle allocation of one placement, out of the pool
    fn drain(&mut self, device_local: bool) -> Vec<Allocation> {
        let mut out = Vec::new();
        self.free.retain(|&(local, capacity), allocations| {
            if local != device_local {
                return true;
            }
            if local {
                self.device_bytes -= capacity * allocations.len() as u64;
            } else {
                self.bytes -= capacity * allocations.len() as u64;
            }
            out.append(allocations);
            false
        });
        out
    }

    fn take(&mut self, device_local: bool, capacity: u64) -> Option<Allocation> {
        let allocation = self.free.get_mut(&(device_local, capacity))?.pop()?;
        if device_local {
            self.device_bytes -= capacity;
        } else {
            self.bytes -= capacity;
        }
        Some(allocation)
    }

    // keeps `allocation` for reuse, or hands it back to be freed when it is not a pooled size or the pool is full
    fn give(
        &mut self,
        device_local: bool,
        capacity: u64,
        allocation: Allocation,
    ) -> Option<Allocation> {
        if capacity > MAX_POOLED_BYTES || (!device_local && self.bytes + capacity > MAX_POOL_BYTES)
        {
            return Some(allocation);
        }
        if device_local {
            self.device_bytes += capacity;
        } else {
            self.bytes += capacity;
        }
        self.free
            .entry((device_local, capacity))
            .or_default()
            .push(allocation);
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
            assert_eq!(largest.size, capacity(40 << 20));
            // SAFETY: a pooled allocation is one no submit uses
            unsafe { largest.release(&context) };
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
