use std::sync::Arc;

use ash::vk;

use crate::{Context, Error, Result};

/// A storage binding past this reads zeros, so no buffer here is larger.
pub(crate) const MAX_BINDING_BYTES: u64 = 4 << 30;

/// Memory the device allocated in host RAM, mapped for the CPU's reads and writes as well (cached when the device
/// offers it: an iGPU reads cached host pages as fast as uncached ones).
pub struct HostBuffer {
    context: Arc<Context>,
    pub(crate) buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    ptr: *mut u8,
    len: usize,
}

// SAFETY: the mapping is plain memory, and the Vulkan handles are only used through the context's synchronized calls
unsafe impl Send for HostBuffer {}
unsafe impl Sync for HostBuffer {}

impl HostBuffer {
    pub fn new(context: &Arc<Context>, len: usize) -> Result<Self> {
        if len == 0 || len as u64 > MAX_BINDING_BYTES {
            return Err(Error::Invalid(format!("a host buffer of {len} bytes")));
        }
        let device = &context.device;
        let info = vk::BufferCreateInfo::default()
            .size(len as u64)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: the create info outlives the call
        let buffer = unsafe { device.create_buffer(&info, None)? };
        // SAFETY: the buffer was created on this device
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let visible =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let memory_type = context
            .memory_type(
                requirements.memory_type_bits,
                visible | vk::MemoryPropertyFlags::HOST_CACHED,
            )
            .or_else(|| context.memory_type(requirements.memory_type_bits, visible))
            .ok_or(Error::NoMemoryType)?;
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);
        // SAFETY: the allocation is host visible, bound once, and mapped whole for the buffer's life
        unsafe {
            let memory = device.allocate_memory(&alloc, None)?;
            device.bind_buffer_memory(buffer, memory, 0)?;
            let ptr = device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())?;
            Ok(Self {
                context: context.clone(),
                buffer,
                memory,
                ptr: ptr.cast(),
                len,
            })
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The buffer's bytes as `f32`s; the GPU must not be writing them.
    pub fn as_f32(&self) -> &[f32] {
        // SAFETY: the mapping is `len` bytes, page aligned, and lives as long as `self`
        unsafe { std::slice::from_raw_parts(self.ptr.cast(), self.len / 4) }
    }

    /// As [`Self::as_f32`], for writing; the GPU must not be reading them.
    pub fn as_f32_mut(&mut self) -> &mut [f32] {
        // SAFETY: as above, and `&mut self` keeps the slice unique on the host side
        unsafe { std::slice::from_raw_parts_mut(self.ptr.cast(), self.len / 4) }
    }
}

impl Drop for HostBuffer {
    fn drop(&mut self) {
        // SAFETY: every submit is waited on, so the GPU no longer uses the buffer
        unsafe {
            self.context.device.unmap_memory(self.memory);
            self.context.device.destroy_buffer(self.buffer, None);
            self.context.device.free_memory(self.memory, None);
        }
    }
}

/// A host allocation the device reads in place: its range rounded out to the import alignment, bound as one buffer,
/// with `offset` the allocation's first byte within it.
pub struct ImportedMemory {
    context: Arc<Context>,
    pub(crate) buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    pub(crate) offset: u64,
    len: usize,
}

// SAFETY: as for `HostBuffer`; the imported pages belong to the caller, who keeps them alive
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
