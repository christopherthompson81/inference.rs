/// RWKV-7's WKV recurrence on CUDA over [`crate::ops::Wkv7`]'s inputs (f32), `state` advanced in place.
#[cfg(feature = "cuda")]
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
pub fn rwkv7_wkv_cuda(
    wkv: crate::ops::Wkv7<'_>,
    state: &inference_tensor::Tensor,
) -> inference_tensor::Result<inference_tensor::Tensor> {
    use inference_tensor::cuda_backend::cudarc::driver::DevicePtr;
    use inference_tensor::{CudaStorage, Storage, Tensor};

    let inputs = [wkv.r, wkv.w, wkv.k, wkv.v, wkv.a, wkv.b].map(Tensor::contiguous);
    let [r, w, k, v, a, b] = inputs;
    let (r, w, k, v, a, b) = (r?, w?, k?, v?, a?, b?);
    let (batch, seq_len, heads, head_size) = r.dims4()?;
    let supported = unsafe { crate::cuda::ffi::rwkv7_head_size() } as usize;
    if head_size != supported {
        inference_tensor::bail!(
            "rwkv7_wkv_cuda: head size {head_size}, the kernel runs {supported}"
        );
    }
    if !state.is_contiguous() {
        inference_tensor::bail!("rwkv7_wkv_cuda: the state must be contiguous");
    }
    let dev = r.device().as_cuda_device()?;
    macro_rules! ptr {
        ($tensor:expr) => {{
            let (s, l) = $tensor.storage_and_layout();
            let s = match &*s {
                Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                _ => inference_tensor::bail!("rwkv7_wkv_cuda: tensors must be on CUDA"),
            };
            s.slice(l.start_offset()..).device_ptr(s.stream()).0
        }};
    }
    let y = unsafe { dev.alloc::<f32>(batch * seq_len * heads * head_size) }?;
    unsafe {
        crate::cuda::ffi::rwkv7_wkv_cuda(
            ptr!(r) as *const f32,
            ptr!(w) as *const f32,
            ptr!(k) as *const f32,
            ptr!(v) as *const f32,
            ptr!(a) as *const f32,
            ptr!(b) as *const f32,
            ptr!(state) as *mut f32,
            y.device_ptr(y.stream()).0 as *mut f32,
            batch as i32,
            seq_len as i32,
            heads as i32,
            dev.cuda_stream().cu_stream() as i64,
        );
    }
    Ok(Tensor::from((
        Storage::Cuda(CudaStorage::wrap_cuda_slice(y, dev.clone())),
        (batch, seq_len, heads, head_size),
    )))
}
