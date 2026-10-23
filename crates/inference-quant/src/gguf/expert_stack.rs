//! Expert matrices of one MoE projection, in pinned host memory or in device slots the GGUF kernels read in place.

use std::sync::{Arc, Mutex, Weak};

use inference_tensor::cuda_backend::cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, DevicePtr, SyncOnDrop, result, sys,
};
use inference_tensor::{CudaDevice, Device, Result, Shape};

use super::kernel::{GgufType, KernelWeight};

// Zeroed bytes past the last expert, as the raw tensors pad theirs: the matmul kernels read whole tiles of K
const ROW_PADDING_ELEMENTS: usize = 512;

/// One projection's experts as their GGUF blocks in page-locked host memory mapped into the device: the CPU reads
/// them in place, device copies from them run asynchronously, and the GPU kernels can read them over the bus.
pub struct HostExpertStack {
    ty: GgufType,
    shape: Shape,
    len: usize,
    storage: HostBlocks,
    device: CudaDevice,
}

enum HostBlocks {
    // copied into a pinned allocation, padding included
    Pinned(CudaSlice<u8>),
    // read in place from the GGUF mapping, whose shard is registered with the device
    Mapped {
        blocks: super::mapped::MappedBlocks,
        shard: Arc<ShardRegistration>,
    },
}

/// A GGUF shard's whole mapping page-locked and mapped into one device: a registration may not overlap another, and
/// neighbouring tensors share pages, so every stack in a shard shares one.
struct ShardRegistration {
    key: ((usize, usize), usize),
    host: *mut std::ffi::c_void,
    device_base: sys::CUdeviceptr,
    context: Arc<CudaContext>,
    // keeps the mapping alive while it is registered
    _blocks: super::mapped::MappedBlocks,
}

// SAFETY: the registered range is read-only, and the context handle is shareable
unsafe impl Send for ShardRegistration {}
unsafe impl Sync for ShardRegistration {}

impl Drop for ShardRegistration {
    fn drop(&mut self) {
        if self.context.bind_to_thread().is_ok() {
            // SAFETY: `host` was registered by `register` and nothing reads it once the last stack drops
            let _ = unsafe { sys::cuMemHostUnregister(self.host) };
        }
    }
}

static SHARD_REGISTRATIONS: Mutex<Vec<Weak<ShardRegistration>>> = Mutex::new(Vec::new());

impl ShardRegistration {
    // the shard holding `blocks`, registered with `device` once and shared after
    fn for_blocks(blocks: &super::mapped::MappedBlocks, device: &CudaDevice) -> Option<Arc<Self>> {
        let key = (blocks.shard_key(), device.cuda_stream().context().ordinal());
        let mut registry = SHARD_REGISTRATIONS.lock().ok()?;
        registry.retain(|weak| weak.strong_count() > 0);
        if let Some(found) = registry
            .iter()
            .filter_map(Weak::upgrade)
            .find(|reg| reg.key == key)
        {
            return Some(found);
        }
        let context = device.cuda_stream().context().clone();
        context.bind_to_thread().ok()?;
        let (mapping, _) = blocks.shard_mapping();
        let host = mapping.as_ptr().cast_mut().cast::<std::ffi::c_void>();
        // SAFETY: the mapping stays alive (held below) and is only read through the registration
        if let Err(err) = unsafe {
            sys::cuMemHostRegister_v2(
                host,
                mapping.len(),
                sys::CU_MEMHOSTREGISTER_DEVICEMAP | sys::CU_MEMHOSTREGISTER_READ_ONLY,
            )
        }
        .result()
        {
            crate::log::once_log_warn(format!(
                "GGUF experts are copied to pinned memory: registering the file mapping failed ({err:?})"
            ));
            return None;
        }
        let mut device_base: sys::CUdeviceptr = 0;
        // SAFETY: `host` was just registered with the device map flag
        if unsafe { sys::cuMemHostGetDevicePointer_v2(&mut device_base, host, 0) }
            .result()
            .is_err()
        {
            // SAFETY: registered just above
            let _ = unsafe { sys::cuMemHostUnregister(host) };
            return None;
        }
        let registration = Arc::new(Self {
            key,
            host,
            device_base,
            context,
            _blocks: blocks.clone(),
        });
        registry.push(Arc::downgrade(&registration));
        Some(registration)
    }
}

impl HostExpertStack {
    /// Copies `bytes` (`experts x rows x cols` blocks, expert-major) into mapped host memory on `device`.
    pub fn new(ty: GgufType, dims: [usize; 3], bytes: &[u8], device: &CudaDevice) -> Result<Self> {
        let [experts, rows, cols] = dims;
        let expected = ty
            .row_bytes(cols)
            .map(|row| row * rows * experts)
            .ok_or_else(|| {
                inference_tensor::Error::msg(format!("{ty:?} rows cannot hold {cols} elements"))
            })?;
        if bytes.len() != expected {
            inference_tensor::bail!(
                "{ty:?} {dims:?} experts take {expected} bytes, got {}",
                bytes.len()
            );
        }
        super::ffi::ensure_type_loaded(ty).map_err(inference_tensor::Error::msg)?;
        let padding = ROW_PADDING_ELEMENTS * ty.type_size() / ty.block_size();
        // SAFETY: the blocks and then the zeroed padding fill the allocation before anything reads it
        let mapped = unsafe {
            device
                .cuda_stream()
                .alloc_mapped_host::<u8>(bytes.len() + padding)
        }
        .map_err(inference_tensor::Error::wrap)?;
        let host = mapped.mapped_host_ptr().expect("mapped allocation");
        // SAFETY: `host` spans the allocation just made, which nothing else references yet
        let all = unsafe { std::slice::from_raw_parts_mut(host, bytes.len() + padding) };
        all[..bytes.len()].copy_from_slice(bytes);
        all[bytes.len()..].fill(0);
        Ok(Self {
            ty,
            shape: Shape::from((experts, rows, cols)),
            len: bytes.len(),
            storage: HostBlocks::Pinned(mapped),
            device: device.clone(),
        })
    }

    /// The experts read in place from their GGUF mapping, its shard page-locked and mapped into `device`; copied as
    /// `new` does when the shard cannot be registered or the blocks end too near the mapping's end for the kernels'
    /// whole-tile reads.
    pub fn from_mapped(blocks: super::mapped::MappedBlocks, device: &CudaDevice) -> Result<Self> {
        let Ok(dims) = <[usize; 3]>::try_from(blocks.dims.as_slice()) else {
            inference_tensor::bail!(
                "expert blocks must be [experts, rows, cols], got {:?}",
                blocks.dims
            );
        };
        let ty = blocks.ty;
        let padding = ROW_PADDING_ELEMENTS * ty.type_size() / ty.block_size();
        let shard = (blocks.trailing() >= padding)
            .then(|| ShardRegistration::for_blocks(&blocks, device))
            .flatten();
        let Some(shard) = shard else {
            return Self::new(ty, dims, blocks.bytes(), device);
        };
        super::ffi::ensure_type_loaded(ty).map_err(inference_tensor::Error::msg)?;
        let [experts, rows, cols] = dims;
        let expected = ty.row_bytes(cols).map(|row| row * rows * experts);
        if expected != Some(blocks.bytes().len()) {
            inference_tensor::bail!(
                "{ty:?} {dims:?} experts take {expected:?} bytes, got {}",
                blocks.bytes().len()
            );
        }
        crate::log::once_log_info(
            "GGUF experts are read in place from the file mapping, page-locked",
        );
        Ok(Self {
            ty,
            shape: Shape::from((experts, rows, cols)),
            len: blocks.bytes().len(),
            storage: HostBlocks::Mapped { blocks, shard },
            device: device.clone(),
        })
    }

    pub fn ty(&self) -> GgufType {
        self.ty
    }

    /// `[experts, rows, cols]`
    pub fn dims(&self) -> [usize; 3] {
        let d = self.shape.dims();
        [d[0], d[1], d[2]]
    }

    pub fn expert_bytes(&self) -> usize {
        self.len / self.dims()[0]
    }

    /// Every expert's blocks.
    pub fn bytes(&self) -> &[u8] {
        match &self.storage {
            HostBlocks::Pinned(pinned) => {
                let host = pinned.mapped_host_ptr().expect("mapped allocation");
                // SAFETY: filled at construction and never written again; the device only reads it
                unsafe { std::slice::from_raw_parts(host, self.len) }
            }
            HostBlocks::Mapped { blocks, .. } => blocks.bytes(),
        }
    }
}

impl KernelWeight for HostExpertStack {
    fn gguf_type(&self) -> GgufType {
        self.ty
    }

    fn kernel_shape(&self) -> &Shape {
        &self.shape
    }

    fn kernel_device(&self) -> Device {
        Device::Cuda(self.device.clone())
    }

    fn kernel_ptr<'a>(&'a self, stream: &'a CudaStream) -> Result<(*const u8, SyncOnDrop<'a>)> {
        match &self.storage {
            HostBlocks::Pinned(pinned) => {
                let (ptr, guard) = pinned.device_ptr(stream);
                Ok((ptr as *const u8, guard))
            }
            // read-only memory the device never writes, so nothing to order against
            HostBlocks::Mapped { blocks, shard } => {
                let (_, offset) = blocks.shard_mapping();
                Ok((
                    (shard.device_base + offset as sys::CUdeviceptr) as *const u8,
                    SyncOnDrop::Record(None),
                ))
            }
        }
    }
}

/// Expert matrices of one projection in a device buffer, laid out as `[slots, rows, cols]` blocks.
#[derive(Clone)]
pub struct DeviceExpertStack {
    ty: GgufType,
    shape: Shape,
    expert_bytes: usize,
    blocks: Arc<CudaSlice<u8>>,
    device: CudaDevice,
}

impl DeviceExpertStack {
    /// Room for `slots` experts of the largest of `layouts` (`(type, rows, cols)`), zeroed.
    pub fn zeros(
        layouts: &[(GgufType, usize, usize)],
        slots: usize,
        device: &CudaDevice,
    ) -> Result<Self> {
        let Some(&(ty, rows, cols)) = layouts.first() else {
            inference_tensor::bail!("an expert stack needs a layout");
        };
        let mut capacity = 0;
        for &(ty, rows, cols) in layouts {
            super::ffi::ensure_type_loaded(ty).map_err(inference_tensor::Error::msg)?;
            let row = ty.row_bytes(cols).ok_or_else(|| {
                inference_tensor::Error::msg(format!("{ty:?} rows cannot hold {cols} elements"))
            })?;
            let padding = ROW_PADDING_ELEMENTS * ty.type_size() / ty.block_size();
            capacity = capacity.max(slots * rows * row + padding);
        }
        let blocks = device.alloc_zeros::<u8>(capacity)?;
        Self {
            ty,
            shape: Shape::from((slots, rows, cols)),
            expert_bytes: 0,
            blocks: Arc::new(blocks),
            device: device.clone(),
        }
        .view(ty, slots, rows, cols)
    }

    /// The same buffer read as `slots` experts of `ty`, `rows x cols` each.
    pub fn view(&self, ty: GgufType, slots: usize, rows: usize, cols: usize) -> Result<Self> {
        let expert_bytes = ty.row_bytes(cols).map(|row| row * rows).ok_or_else(|| {
            inference_tensor::Error::msg(format!("{ty:?} rows cannot hold {cols} elements"))
        })?;
        let padding = ROW_PADDING_ELEMENTS * ty.type_size() / ty.block_size();
        if slots * expert_bytes + padding > self.blocks.len() {
            inference_tensor::bail!(
                "{slots} {ty:?} experts of {rows}x{cols} overflow the expert stack"
            );
        }
        Ok(Self {
            ty,
            shape: Shape::from((slots, rows, cols)),
            expert_bytes,
            blocks: self.blocks.clone(),
            device: self.device.clone(),
        })
    }

    pub fn slots(&self) -> usize {
        self.shape.dims()[0]
    }

    /// Queues a copy of host experts (whole ones, back to back) into the slots from `first_slot` on `stream`.
    /// # Safety
    /// `src` must stay alive and unchanged until the copy completes on `stream` (pinned memory makes it async), and
    /// nothing may read the written slots before then.
    pub unsafe fn fill_async(
        &self,
        first_slot: usize,
        src: &[u8],
        stream: &CudaStream,
    ) -> Result<()> {
        if !src.len().is_multiple_of(self.expert_bytes)
            || first_slot * self.expert_bytes + src.len() > self.slots() * self.expert_bytes
        {
            inference_tensor::bail!(
                "expert copy of {} bytes does not fit from slot {first_slot}",
                src.len()
            );
        }
        let (base, _guard) = self.blocks.device_ptr(stream);
        let dst = base + (first_slot * self.expert_bytes) as u64;
        unsafe { result::memcpy_htod_async(dst, src, stream.cu_stream()) }
            .map_err(inference_tensor::Error::wrap)
    }
}

impl DeviceExpertStack {
    /// Queues zeroing the kernels' read padding past this view's first `slots` experts on `stream`: the buffer is
    /// shared by views of other types, whose stale blocks read as this type could hold an Inf or NaN scale.
    /// # Safety
    /// Nothing may read those bytes until the memset completes on `stream`.
    pub unsafe fn zero_padding_async(&self, slots: usize, stream: &CudaStream) -> Result<()> {
        let padding = ROW_PADDING_ELEMENTS * self.ty.type_size() / self.ty.block_size();
        let start = slots * self.expert_bytes;
        if start + padding > self.blocks.len() {
            inference_tensor::bail!("{slots} experts and their padding overflow the expert stack");
        }
        let (base, _guard) = self.blocks.device_ptr(stream);
        unsafe { result::memset_d8_async(base + start as u64, 0, padding, stream.cu_stream()) }
            .map_err(inference_tensor::Error::wrap)
    }

    /// Queues a copy of `src`'s expert in `src_slot` into this stack's `dst_slot` on `stream`.
    /// # Safety
    /// Nothing may write the source slot or read the destination slot until the copy completes on `stream`.
    pub unsafe fn copy_slot_async(
        &self,
        dst_slot: usize,
        src: &DeviceExpertStack,
        src_slot: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        if src.expert_bytes != self.expert_bytes
            || dst_slot >= self.slots()
            || src_slot >= src.slots()
        {
            inference_tensor::bail!("expert slot copy between mismatched stacks");
        }
        let (dst, _dst_guard) = self.blocks.device_ptr(stream);
        let (src_base, _src_guard) = src.blocks.device_ptr(stream);
        let offset = |slot: usize| (slot * self.expert_bytes) as u64;
        unsafe {
            result::memcpy_dtod_async(
                dst + offset(dst_slot),
                src_base + offset(src_slot),
                self.expert_bytes,
                stream.cu_stream(),
            )
        }
        .map_err(inference_tensor::Error::wrap)
    }
}

impl KernelWeight for DeviceExpertStack {
    fn gguf_type(&self) -> GgufType {
        self.ty
    }

    fn kernel_shape(&self) -> &Shape {
        &self.shape
    }

    fn kernel_device(&self) -> Device {
        Device::Cuda(self.device.clone())
    }

    fn kernel_ptr<'a>(&'a self, stream: &'a CudaStream) -> Result<(*const u8, SyncOnDrop<'a>)> {
        let (ptr, guard) = self.blocks.device_ptr(stream);
        Ok((ptr as *const u8, guard))
    }
}
