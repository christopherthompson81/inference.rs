//! Expert matrices of one MoE projection, in host memory (pinned, or the GGUF mapping itself) or in device slots.

use std::sync::{Arc, Mutex, Weak};

use inference_tensor::cuda_backend::cudarc::driver::{
    CudaContext, CudaEvent, CudaSlice, CudaStream, DevicePtr, SyncOnDrop, result, sys,
};
use inference_tensor::{CudaDevice, Device, Result, Shape};

use super::kernel::{GgufType, KernelWeight};

// Zeroed bytes past the last expert, as the raw tensors pad theirs: the matmul kernels read whole tiles of K
const ROW_PADDING_ELEMENTS: usize = 512;
// A bounced upload's page-locked chunks: their size, how many are in flight, and the parts the pool splits each into
const BOUNCE_CHUNK_BYTES: usize = 16 << 20;
const BOUNCE_CHUNKS: usize = 4;
const BOUNCE_PARTS: usize = 64;

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
    // read in place from the GGUF mapping as ordinary pageable memory: a copy source for the device cache and the
    // CPU's input, never read by a kernel
    Pageable(super::mapped::MappedBlocks),
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
            crate::log::once_log_info(format!(
                "GGUF experts stay pageable in the file mapping: registering it with the device failed ({err:?})"
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

    /// The experts read in place from their GGUF mapping, its shard page-locked and mapped into `device` when it can
    /// be registered, else as pageable memory (copies from it to the device are synchronous, and nothing is pinned).
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
        super::ffi::ensure_type_loaded(ty).map_err(inference_tensor::Error::msg)?;
        let [experts, rows, cols] = dims;
        let expected = ty.row_bytes(cols).map(|row| row * rows * experts);
        if expected != Some(blocks.bytes().len()) {
            inference_tensor::bail!(
                "{ty:?} {dims:?} experts take {expected:?} bytes, got {}",
                blocks.bytes().len()
            );
        }
        let len = blocks.bytes().len();
        let storage = match shard {
            Some(shard) => {
                crate::log::once_log_info(
                    "GGUF experts are read in place from the file mapping, page-locked",
                );
                HostBlocks::Mapped { blocks, shard }
            }
            None => {
                crate::log::once_log_info(
                    "GGUF experts are read in place from the file mapping, pageable",
                );
                HostBlocks::Pageable(blocks)
            }
        };
        Ok(Self {
            ty,
            shape: Shape::from((experts, rows, cols)),
            len,
            storage,
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

    /// Whether the experts are ordinary pageable memory, so an upload from them is best bounced (see [`BounceRing`]).
    pub fn is_pageable(&self) -> bool {
        matches!(self.storage, HostBlocks::Pageable(_))
    }

    /// Every expert's blocks.
    pub fn bytes(&self) -> &[u8] {
        match &self.storage {
            HostBlocks::Pinned(pinned) => {
                let host = pinned.mapped_host_ptr().expect("mapped allocation");
                // SAFETY: filled at construction and never written again; the device only reads it
                unsafe { std::slice::from_raw_parts(host, self.len) }
            }
            HostBlocks::Mapped { blocks, .. } | HostBlocks::Pageable(blocks) => blocks.bytes(),
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
            HostBlocks::Pageable(_) => inference_tensor::bail!(
                "pageable host experts are no kernel weight; copy them to the device"
            ),
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
        self.check_fill(first_slot, src)?;
        let (base, _guard) = self.blocks.device_ptr(stream);
        let dst = base + (first_slot * self.expert_bytes) as u64;
        unsafe { result::memcpy_htod_async(dst, src, stream.cu_stream()) }
            .map_err(inference_tensor::Error::wrap)
    }

    /// As [`Self::fill_async`], with a pageable `src` copied through `ring` by the whole thread pool.
    /// # Safety
    /// As [`Self::fill_async`].
    pub unsafe fn fill_bounced_async(
        &self,
        first_slot: usize,
        src: &[u8],
        ring: &mut BounceRing,
        stream: &CudaStream,
    ) -> Result<()> {
        self.check_fill(first_slot, src)?;
        let (base, _guard) = self.blocks.device_ptr(stream);
        let dst = base + (first_slot * self.expert_bytes) as u64;
        unsafe { ring.upload(dst, src, stream) }
    }

    fn check_fill(&self, first_slot: usize, src: &[u8]) -> Result<()> {
        if !src.len().is_multiple_of(self.expert_bytes)
            || first_slot * self.expert_bytes + src.len() > self.slots() * self.expert_bytes
        {
            inference_tensor::bail!(
                "expert copy of {} bytes does not fit from slot {first_slot}",
                src.len()
            );
        }
        Ok(())
    }
}

/// Page-locked chunks a pageable upload goes through: one thread's memcpy out of the page cache runs at about half the
/// DMA's rate, so the thread pool fills each chunk while the one before it is on the bus.
pub struct BounceRing {
    chunks: Vec<(CudaSlice<u8>, CudaEvent)>,
    next: usize,
}

impl BounceRing {
    pub fn new(stream: &Arc<CudaStream>) -> Result<Self> {
        let chunks = (0..BOUNCE_CHUNKS)
            .map(|_| {
                // SAFETY: every upload writes a chunk's bytes before its copy reads them
                let chunk = unsafe { stream.alloc_mapped_host::<u8>(BOUNCE_CHUNK_BYTES) }
                    .map_err(inference_tensor::Error::wrap)?;
                let done = stream
                    .context()
                    .new_event(None)
                    .map_err(inference_tensor::Error::wrap)?;
                Ok((chunk, done))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { chunks, next: 0 })
    }

    /// Queues `src` into the device memory at `dst` on `stream`.
    /// # Safety
    /// `dst..dst + src.len()` is device memory nothing reads until the copies complete on `stream`.
    pub unsafe fn upload(&mut self, dst: u64, src: &[u8], stream: &CudaStream) -> Result<()> {
        let wrap = inference_tensor::Error::wrap;
        for (i, part) in src.chunks(BOUNCE_CHUNK_BYTES).enumerate() {
            let (chunk, done) = &self.chunks[self.next];
            self.next = (self.next + 1) % self.chunks.len();
            // the chunk's last upload has left it
            done.synchronize().map_err(wrap)?;
            let host = chunk.mapped_host_ptr().expect("mapped allocation") as usize;
            let piece = part.len().div_ceil(BOUNCE_PARTS);
            inference_tensor::utils::barrier_pool().execute_chunked(BOUNCE_PARTS, |range| {
                for p in range {
                    let start = (p * piece).min(part.len());
                    let end = (start + piece).min(part.len());
                    // SAFETY: the chunk holds BOUNCE_CHUNK_BYTES and each part writes its own range of it
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            part[start..end].as_ptr(),
                            (host as *mut u8).add(start),
                            end - start,
                        )
                    };
                }
            });
            // SAFETY: the chunk is page-locked and stays untouched until `done` passes this copy
            let staged = unsafe { std::slice::from_raw_parts(host as *const u8, part.len()) };
            let at = dst + (i * BOUNCE_CHUNK_BYTES) as u64;
            unsafe { result::memcpy_htod_async(at, staged, stream.cu_stream()) }.map_err(wrap)?;
            done.record(stream).map_err(wrap)?;
        }
        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::archive::GgufArchive;
    use crate::gguf::archive::tests::{TestTensor, write_test_gguf};

    const EXPERTS: usize = 4;
    const ROWS: usize = 32;
    const COLS: usize = 64;
    const NAME: &str = "blk.0.ffn_down_exps.weight";
    // GGUF's type id of IQ4_NL
    const IQ4_NL_ID: u32 = 20;

    // Experts read from their GGUF mapping (page-locked where the driver registers it, else pageable) fill device slots
    // with their file bytes.
    #[test]
    fn mapped_experts_fill_device_slots_with_their_file_bytes() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        let cuda = device.as_cuda_device()?;
        let ty = GgufType::Iq4Nl;
        let expert = ty.row_bytes(COLS).expect("whole blocks") * ROWS;
        let data = (0..EXPERTS * expert)
            .map(|i| u8::try_from(i * 31 % 251).expect("below 251"))
            .collect::<Vec<_>>();
        let file = write_test_gguf(
            &[],
            &[TestTensor {
                name: NAME,
                shape: [EXPERTS, ROWS, COLS].map(|d| d as u64).to_vec(),
                dtype: IQ4_NL_ID,
                data: data.clone(),
                offset: None,
            }],
            super::super::archive::DEFAULT_ALIGNMENT,
        );
        let archive = Arc::new(GgufArchive::open_file(file.path())?);
        let (shard, range) = archive.mapped_range(NAME).expect("a mapped tensor");
        let blocks = super::super::mapped::MappedBlocks::new(
            ty,
            vec![EXPERTS, ROWS, COLS],
            archive,
            shard,
            range,
        );
        let host = HostExpertStack::from_mapped(blocks, cuda)?;
        assert_eq!(host.bytes(), data.as_slice());
        let slots =
            DeviceExpertStack::zeros(&[(ty, ROWS, COLS)], 2, cuda)?.view(ty, 2, ROWS, COLS)?;
        let stream = cuda.cuda_stream();
        // SAFETY: `host` outlives the synchronize below, and nothing reads the slot before it
        unsafe { slots.fill_async(1, &host.bytes()[2 * expert..3 * expert], &stream)? };
        stream
            .synchronize()
            .map_err(inference_tensor::Error::wrap)?;
        let back = stream
            .clone_dtoh(&*slots.blocks)
            .map_err(inference_tensor::Error::wrap)?;
        assert_eq!(&back[expert..2 * expert], &data[2 * expert..3 * expert]);
        Ok(())
    }

    // host-to-device copies of one expert's size from pageable and from pinned memory, synchronously as the fill
    // worker waits on them
    #[test]
    #[ignore = "benchmark"]
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn expert_copy_bandwidth() -> Result<()> {
        const EXPERT_BYTES: usize = 720 * 1024;
        const COPIES: usize = 2000;
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        let cuda = device.as_cuda_device()?;
        let stream = cuda.cuda_stream();
        let dst = cuda.alloc_zeros::<u8>(EXPERT_BYTES)?;
        let pageable = (0..EXPERT_BYTES * COPIES / 8)
            .map(|i| i as u8)
            .collect::<Vec<_>>();
        // SAFETY: filled below before any copy reads it
        let pinned = unsafe { stream.alloc_mapped_host::<u8>(EXPERT_BYTES * 8) }
            .map_err(inference_tensor::Error::wrap)?;
        let pinned_host = pinned.mapped_host_ptr().expect("mapped allocation");
        // SAFETY: the allocation just made spans this
        unsafe { std::slice::from_raw_parts_mut(pinned_host, EXPERT_BYTES * 8).fill(1) };
        let (dst_ptr, _guard) = dst.device_ptr(&stream);
        let time = |label: &str, src: &dyn Fn(usize) -> *const u8| -> Result<()> {
            let start = std::time::Instant::now();
            for i in 0..COPIES {
                // SAFETY: each source spans EXPERT_BYTES live bytes, and the stream is synchronized per copy
                unsafe {
                    let src = std::slice::from_raw_parts(src(i), EXPERT_BYTES);
                    result::memcpy_htod_async(dst_ptr, src, stream.cu_stream())
                        .map_err(inference_tensor::Error::wrap)?;
                }
                stream
                    .synchronize()
                    .map_err(inference_tensor::Error::wrap)?;
            }
            let secs = start.elapsed().as_secs_f64();
            println!(
                "{label:<10} {:6.1} us a copy, {:5.2} GB/s",
                secs / COPIES as f64 * 1e6,
                (EXPERT_BYTES * COPIES) as f64 / secs / 1e9
            );
            Ok(())
        };
        let span = pageable.len() - EXPERT_BYTES;
        time("pageable", &|i| {
            pageable[(i * EXPERT_BYTES * 7) % span..].as_ptr()
        })?;
        time("pinned", &|i| unsafe {
            pinned_host.add((i % 8) * EXPERT_BYTES)
        })?;
        // a bounce through pinned memory: chunks memcpy'd into two pinned halves, each copy queued before the next memcpy
        for chunks in [2, 4, 8] {
            let chunk = EXPERT_BYTES / chunks;
            let start = std::time::Instant::now();
            for i in 0..COPIES {
                let src = &pageable[(i * EXPERT_BYTES * 7) % span..][..EXPERT_BYTES];
                let mut events: [Option<inference_tensor::cuda_backend::cudarc::driver::CudaEvent>;
                    2] = [None, None];
                for c in 0..chunks {
                    let half = c % 2;
                    if let Some(event) = events[half].take() {
                        event.synchronize().map_err(inference_tensor::Error::wrap)?;
                    }
                    // SAFETY: half `half` of the pinned buffer is no longer read (its last copy's event passed)
                    let staged = unsafe {
                        std::slice::from_raw_parts_mut(pinned_host.add(half * chunk), chunk)
                    };
                    staged.copy_from_slice(&src[c * chunk..][..chunk]);
                    // SAFETY: `staged` stays untouched until its event passes
                    unsafe {
                        result::memcpy_htod_async(
                            dst_ptr + (c * chunk) as u64,
                            &*staged,
                            stream.cu_stream(),
                        )
                        .map_err(inference_tensor::Error::wrap)?;
                    }
                    events[half] = Some(
                        stream
                            .record_event(None)
                            .map_err(inference_tensor::Error::wrap)?,
                    );
                }
                stream
                    .synchronize()
                    .map_err(inference_tensor::Error::wrap)?;
            }
            let secs = start.elapsed().as_secs_f64();
            println!(
                "bounce/{chunks:<3} {:6.1} us a copy, {:5.2} GB/s",
                secs / COPIES as f64 * 1e6,
                (EXPERT_BYTES * COPIES) as f64 / secs / 1e9
            );
        }
        Ok(())
    }
}
