//! Mapped host buffers that a host function launched on a stream shares with the device, for steps that hand work to
//! the CPU without waiting on it, so they can be captured in a CUDA graph.

use std::sync::Arc;

use inference_tensor::cuda::cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DeviceRepr};
use inference_tensor::{Result, Tensor};

/// `len` elements of mapped host memory; a step writes each buffer before it reads it.
pub fn mapped_buffer<T: DeviceRepr>(stream: &Arc<CudaStream>, len: usize) -> Result<CudaSlice<T>> {
    // SAFETY: every user writes the buffer before reading it
    unsafe { stream.alloc_mapped_host(len) }.map_err(inference_tensor::Error::wrap)
}

/// A mapped buffer's first `len` elements on the host side, which the stream orders every access to.
#[allow(clippy::mut_from_ref)]
pub fn mapped_host<T>(buffer: &CudaSlice<T>, len: usize) -> &mut [T] {
    let ptr = buffer.mapped_host_ptr().expect("mapped allocation");
    assert!(len <= buffer.len());
    // SAFETY: the buffer is pinned host memory of at least `len` elements, touched only in stream order
    unsafe { std::slice::from_raw_parts_mut(ptr, len) }
}

/// A contiguous CUDA tensor's first element, for copies queued on `stream`.
pub fn tensor_ptr<T: inference_tensor::cuda_backend::CudaDType>(
    tensor: &Tensor,
    stream: &CudaStream,
) -> Result<u64> {
    let (storage, layout) = tensor.storage_and_layout();
    let inference_tensor::Storage::Cuda(cuda) = &*storage else {
        inference_tensor::bail!("mapped host copies take CUDA tensors");
    };
    let slice = cuda.as_cuda_slice::<T>()?;
    let (ptr, _guard) = slice.device_ptr(stream);
    Ok(ptr + (layout.start_offset() * std::mem::size_of::<T>()) as u64)
}
