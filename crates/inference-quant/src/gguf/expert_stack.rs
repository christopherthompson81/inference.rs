//! Expert matrices of one MoE projection, in pinned host memory or in device slots the GGUF kernels read in place.

use std::sync::Arc;

use inference_tensor::cuda_backend::cudarc::driver::{
    CudaSlice, CudaStream, DevicePtr, SyncOnDrop, result,
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
    bytes: CudaSlice<u8>,
    device: CudaDevice,
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
            bytes: mapped,
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
        let host = self.bytes.mapped_host_ptr().expect("mapped allocation");
        // SAFETY: filled at construction and never written again; the device only reads it
        unsafe { std::slice::from_raw_parts(host, self.len) }
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
        let (ptr, guard) = self.bytes.device_ptr(stream);
        Ok((ptr as *const u8, guard))
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
