use std::collections::HashMap;
use std::sync::Arc;

use ash::vk;

use crate::{Context, Error, Result};

/// A storage binding past this reads zeros, so no buffer here is larger.
pub(crate) const MAX_BINDING_BYTES: u64 = 4 << 30;
// Buffers up to this size are pooled in power-of-two classes from the smallest; larger ones are allocated to size and
// freed when dropped
const MIN_POOLED_BYTES: u64 = 256;
const MAX_POOLED_BYTES: u64 = 256 << 20;
// The most idle memory the pool holds before a dropped buffer is freed instead
const MAX_POOL_BYTES: u64 = 2 << 30;
// A host write to or read of device memory goes through staging buffers of up to this, a submit each
const STAGING_BYTES: usize = 64 << 20;

/// Buffers nothing references any longer, by placement and capacity, for new ones to reuse: a Vulkan allocation costs
/// tens of microseconds, and a model step allocates hundreds of short-lived tensors.
#[derive(Default)]
pub(crate) struct BufferPool {
    free: HashMap<(bool, u64), Vec<Allocation>>,
    bytes: u64,
}

// a buffer, its memory and the memory's mapping
struct Allocation {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    ptr: *mut u8,
}

// SAFETY: the handles and mapping are only used by whoever takes the allocation out of the pool's lock
unsafe impl Send for Allocation {}

impl BufferPool {
    /// # Safety
    /// No submit may still use the buffers.
    pub(crate) unsafe fn destroy(&mut self, device: &ash::Device) {
        for allocation in self.free.drain().flat_map(|(_, allocations)| allocations) {
            // SAFETY: the caller's contract
            unsafe { allocation.free(device) };
        }
        self.bytes = 0;
    }
}

impl Allocation {
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
        let pooled = context
            .pool
            .lock()
            .expect("vulkan pool poisoned")
            .take(device_local, capacity);
        if let Some(Allocation {
            buffer,
            memory,
            ptr,
        }) = pooled
        {
            return Ok(Self {
                context: context.clone(),
                buffer,
                memory,
                ptr,
                len,
                capacity,
            });
        }
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
            Ok(Self {
                context: context.clone(),
                buffer,
                memory,
                ptr,
                len,
                capacity,
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
    pub fn is_mapped(&self) -> bool {
        !self.ptr.is_null()
    }

    /// Writes `bytes` at `offset`: in place when mapped, which the GPU must not be using, else as a copy recorded on
    /// the stream, ordered after what is recorded before it.
    pub fn write(&mut self, offset: usize, bytes: &[u8]) -> Result<()> {
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
        let allocation = Allocation {
            buffer: self.buffer,
            memory: self.memory,
            ptr: self.ptr,
        };
        let refused = self
            .context
            .pool
            .lock()
            .expect("vulkan pool poisoned")
            .give(self.ptr.is_null(), self.capacity, allocation);
        if let Some(allocation) = refused {
            // SAFETY: see above
            unsafe { allocation.free(&self.context.device) };
        }
    }
}

impl BufferPool {
    fn take(&mut self, device_local: bool, capacity: u64) -> Option<Allocation> {
        let allocation = self.free.get_mut(&(device_local, capacity))?.pop()?;
        self.bytes -= capacity;
        Some(allocation)
    }

    // keeps `allocation` for reuse, or hands it back to be freed when it is not a pooled size or the pool is full
    fn give(
        &mut self,
        device_local: bool,
        capacity: u64,
        allocation: Allocation,
    ) -> Option<Allocation> {
        if capacity > MAX_POOLED_BYTES || self.bytes + capacity > MAX_POOL_BYTES {
            return Some(allocation);
        }
        self.bytes += capacity;
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
}
