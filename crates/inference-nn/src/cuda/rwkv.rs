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

#[cfg(feature = "cuda")]
pub(crate) fn dtype_code(dtype: inference_tensor::DType) -> inference_tensor::Result<i32> {
    match dtype {
        inference_tensor::DType::F16 => Ok(0),
        inference_tensor::DType::BF16 => Ok(1),
        inference_tensor::DType::F32 => Ok(2),
        other => inference_tensor::bail!("rwkv7: no {other:?} kernels"),
    }
}

// a CUDA tensor's device pointer at its offset
#[cfg(feature = "cuda")]
pub(crate) fn device_ptr(t: &inference_tensor::Tensor) -> inference_tensor::Result<u64> {
    use inference_tensor::cuda_backend::cudarc::driver::DevicePtr;
    use inference_tensor::{DType, Storage};
    let (s, l) = t.storage_and_layout();
    let Storage::Cuda(c) = &*s else {
        inference_tensor::bail!("rwkv7: tensors must be on CUDA");
    };
    macro_rules! at {
        ($ty:ty) => {{
            let slice = c.as_cuda_slice::<$ty>()?;
            slice.slice(l.start_offset()..).device_ptr(slice.stream()).0
        }};
    }
    Ok(match t.dtype() {
        DType::U32 => at!(u32),
        DType::F16 => at!(half::f16),
        DType::BF16 => at!(half::bf16),
        DType::F32 => at!(f32),
        other => inference_tensor::bail!("rwkv7: no {other:?} tensors"),
    })
}

/// [`crate::ops::rwkv_token_shift`] on CUDA, one kernel for every mix.
#[cfg(feature = "cuda")]
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
pub fn rwkv_token_shift_cuda(
    x: &inference_tensor::Tensor,
    last: &inference_tensor::Tensor,
    mixes: &inference_tensor::Tensor,
    extra: Option<&inference_tensor::Tensor>,
) -> inference_tensor::Result<(inference_tensor::Tensor, inference_tensor::Tensor)> {
    use inference_tensor::Tensor;
    let x = x.contiguous()?;
    let (batch, seq_len, hidden) = x.dims3()?;
    let mixes_n = mixes.dim(0)?;
    let last = last.to_dtype(x.dtype())?.contiguous()?;
    let mixes = mixes.to_dtype(x.dtype())?.contiguous()?;
    let extra = extra
        .map(|e| e.to_dtype(x.dtype())?.contiguous())
        .transpose()?;
    let out = Tensor::zeros((mixes_n, batch, seq_len, hidden), x.dtype(), x.device())?;
    let new_last = Tensor::zeros((batch, hidden), x.dtype(), x.device())?;
    let stream = x.device().as_cuda_device()?.cuda_stream().cu_stream() as i64;
    unsafe {
        crate::cuda::ffi::rwkv7_token_shift_cuda(
            device_ptr(&x)? as *const _,
            device_ptr(&last)? as *const _,
            device_ptr(&mixes)? as *const _,
            match &extra {
                Some(e) => device_ptr(e)? as *const _,
                None => std::ptr::null(),
            },
            device_ptr(&out)? as *mut _,
            device_ptr(&new_last)? as *mut _,
            batch as i32,
            seq_len as i32,
            hidden as i32,
            mixes_n as i32,
            dtype_code(x.dtype())?,
            stream,
        );
    }
    Ok((out, new_last))
}

#[cfg(feature = "cuda")]
#[repr(C)]
struct TimeMixParams {
    inputs: [*const core::ffi::c_void; 8],
    vectors: [*const f32; 5],
    state: *mut f32,
    out: *mut core::ffi::c_void,
    seq_len: i32,
    heads: i32,
    norm_eps: f32,
}

/// [`crate::ops::rwkv7_time_mix`] on CUDA: everything between the projections and `o_proj` in one kernel.
#[cfg(feature = "cuda")]
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
pub fn rwkv7_time_mix_cuda(
    tm: crate::ops::TimeMix7<'_>,
    state: &inference_tensor::Tensor,
) -> inference_tensor::Result<inference_tensor::Tensor> {
    use inference_tensor::Tensor;
    let dtype = tm.r.dtype();
    let (batch, seq_len, hidden) = tm.r.dims3()?;
    let supported = unsafe { crate::cuda::ffi::rwkv7_head_size() } as usize;
    if tm.head_dim != supported || hidden != tm.heads * tm.head_dim {
        inference_tensor::bail!(
            "rwkv7_time_mix_cuda: {} heads of {}, the kernel runs heads of {supported}",
            tm.heads,
            tm.head_dim
        );
    }
    if !state.is_contiguous() {
        inference_tensor::bail!("rwkv7_time_mix_cuda: the state must be contiguous");
    }
    let same = |t: &Tensor| -> inference_tensor::Result<Tensor> { t.to_dtype(dtype)?.contiguous() };
    let [r, w, k, v, a] = [tm.r, tm.w, tm.k, tm.v, tm.a].map(same);
    let (r, w, k, v, a) = (r?, w?, k?, v?, a?);
    let g = tm.g.map(same).transpose()?;
    let mix = tm
        .v_mix
        .map(
            |(first, mix)| -> inference_tensor::Result<(Tensor, Tensor)> {
                Ok((same(first)?, same(mix)?))
            },
        )
        .transpose()?;
    let f32 = |t: &Tensor| -> inference_tensor::Result<Tensor> {
        t.to_dtype(inference_tensor::DType::F32)?
            .flatten_all()?
            .contiguous()
    };
    let [k_k, k_a, r_k] = [tm.k_k, tm.k_a, tm.r_k].map(f32);
    let vectors = [k_k?, k_a?, r_k?];
    let norm = tm
        .norm
        .map(|(w, b)| -> inference_tensor::Result<(Tensor, Tensor)> { Ok((f32(w)?, f32(b)?)) })
        .transpose()?;
    let out = Tensor::zeros((batch, seq_len, hidden), dtype, r.device())?;
    let null = std::ptr::null::<core::ffi::c_void>();
    let optional =
        |t: Option<&Tensor>| t.map_or(Ok(null), |t| device_ptr(t).map(|p| p as *const _));
    let params = TimeMixParams {
        inputs: [
            device_ptr(&r)? as *const _,
            device_ptr(&w)? as *const _,
            device_ptr(&k)? as *const _,
            device_ptr(&v)? as *const _,
            device_ptr(&a)? as *const _,
            optional(g.as_ref())?,
            mix.as_ref()
                .map_or(Ok(null), |(f, _)| device_ptr(f).map(|p| p as *const _))?,
            mix.as_ref()
                .map_or(Ok(null), |(_, m)| device_ptr(m).map(|p| p as *const _))?,
        ],
        vectors: [
            device_ptr(&vectors[0])? as *const f32,
            device_ptr(&vectors[1])? as *const f32,
            device_ptr(&vectors[2])? as *const f32,
            optional(norm.as_ref().map(|n| &n.0))?.cast(),
            optional(norm.as_ref().map(|n| &n.1))?.cast(),
        ],
        state: device_ptr(state)? as *mut f32,
        out: device_ptr(&out)? as *mut _,
        seq_len: seq_len as i32,
        heads: tm.heads as i32,
        norm_eps: tm.norm_eps as f32,
    };
    let stream = r.device().as_cuda_device()?.cuda_stream().cu_stream() as i64;
    unsafe {
        crate::cuda::ffi::rwkv7_time_mix_cuda(
            (&raw const params).cast(),
            batch as i32,
            dtype_code(dtype)?,
            stream,
        );
    }
    Ok(out)
}

#[cfg(feature = "cuda")]
#[repr(C)]
struct Rwkv6TimeMixParams {
    inputs: [*const core::ffi::c_void; 5],
    vectors: [*const f32; 3],
    state: *mut f32,
    out: *mut core::ffi::c_void,
    seq_len: i32,
    heads: i32,
    norm_eps: f32,
}

/// [`crate::ops::rwkv6_time_mix`] on CUDA, one kernel.
#[cfg(feature = "cuda")]
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
pub fn rwkv6_time_mix_cuda(
    tm: crate::ops::TimeMix6<'_>,
    state: &inference_tensor::Tensor,
) -> inference_tensor::Result<inference_tensor::Tensor> {
    use inference_tensor::Tensor;
    let dtype = tm.r.dtype();
    let (batch, seq_len, hidden) = tm.r.dims3()?;
    let supported = unsafe { crate::cuda::ffi::rwkv7_head_size() } as usize;
    if tm.head_dim != supported || hidden != tm.heads * tm.head_dim {
        inference_tensor::bail!(
            "rwkv6_time_mix_cuda: {} heads of {}, the kernel runs heads of {supported}",
            tm.heads,
            tm.head_dim
        );
    }
    if !state.is_contiguous() {
        inference_tensor::bail!("rwkv6_time_mix_cuda: the state must be contiguous");
    }
    let same = |t: &Tensor| -> inference_tensor::Result<Tensor> { t.to_dtype(dtype)?.contiguous() };
    let [r, k, v, w, g] = [tm.r, tm.k, tm.v, tm.w, tm.g].map(same);
    let (r, k, v, w, g) = (r?, k?, v?, w?, g?);
    let f32 = |t: &Tensor| -> inference_tensor::Result<Tensor> {
        t.to_dtype(inference_tensor::DType::F32)?
            .flatten_all()?
            .contiguous()
    };
    let [u, norm_w, norm_b] = [tm.u, tm.norm_w, tm.norm_b].map(f32);
    let (u, norm_w, norm_b) = (u?, norm_w?, norm_b?);
    let out = Tensor::zeros((batch, seq_len, hidden), dtype, r.device())?;
    let params = Rwkv6TimeMixParams {
        inputs: [
            device_ptr(&r)? as *const _,
            device_ptr(&k)? as *const _,
            device_ptr(&v)? as *const _,
            device_ptr(&w)? as *const _,
            device_ptr(&g)? as *const _,
        ],
        vectors: [
            device_ptr(&u)? as *const f32,
            device_ptr(&norm_w)? as *const f32,
            device_ptr(&norm_b)? as *const f32,
        ],
        state: device_ptr(state)? as *mut f32,
        out: device_ptr(&out)? as *mut _,
        seq_len: seq_len as i32,
        heads: tm.heads as i32,
        norm_eps: tm.norm_eps as f32,
    };
    let stream = r.device().as_cuda_device()?.cuda_stream().cu_stream() as i64;
    unsafe {
        crate::cuda::ffi::rwkv6_time_mix_cuda(
            (&raw const params).cast(),
            batch as i32,
            dtype_code(dtype)?,
            stream,
        );
    }
    Ok(out)
}

#[cfg(feature = "cuda")]
#[repr(C)]
struct Qrwkv6TimeMixParams {
    inputs: [*const core::ffi::c_void; 5],
    state: *mut f32,
    out: *mut core::ffi::c_void,
    seq_len: i32,
    heads: i32,
    kv_heads: i32,
    scale: f32,
}

/// [`crate::ops::qrwkv6_time_mix`] on CUDA, one kernel.
#[cfg(feature = "cuda")]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss
)]
pub fn qrwkv6_time_mix_cuda(
    tm: crate::ops::TimeMixQrwkv6<'_>,
    state: &inference_tensor::Tensor,
) -> inference_tensor::Result<inference_tensor::Tensor> {
    use inference_tensor::Tensor;
    let dtype = tm.r.dtype();
    let (batch, seq_len, hidden) = tm.r.dims3()?;
    if hidden != tm.heads * tm.head_dim || tm.kv_heads == 0 || !tm.heads.is_multiple_of(tm.kv_heads)
    {
        inference_tensor::bail!(
            "qrwkv6_time_mix_cuda: {hidden} wide is not {} heads of {} over {} key/value heads",
            tm.heads,
            tm.head_dim,
            tm.kv_heads
        );
    }
    if !state.is_contiguous() {
        inference_tensor::bail!("qrwkv6_time_mix_cuda: the state must be contiguous");
    }
    let same = |t: &Tensor| -> inference_tensor::Result<Tensor> { t.to_dtype(dtype)?.contiguous() };
    let [r, k, v, w, g] = [tm.r, tm.k, tm.v, tm.w, tm.g].map(same);
    let (r, k, v, w, g) = (r?, k?, v?, w?, g?);
    let out = Tensor::zeros((batch, seq_len, hidden), dtype, r.device())?;
    let params = Qrwkv6TimeMixParams {
        inputs: [
            device_ptr(&r)? as *const _,
            device_ptr(&k)? as *const _,
            device_ptr(&v)? as *const _,
            device_ptr(&w)? as *const _,
            device_ptr(&g)? as *const _,
        ],
        state: device_ptr(state)? as *mut f32,
        out: device_ptr(&out)? as *mut _,
        seq_len: seq_len as i32,
        heads: tm.heads as i32,
        kv_heads: tm.kv_heads as i32,
        scale: (tm.head_dim as f32).powf(-0.5),
    };
    let stream = r.device().as_cuda_device()?.cuda_stream().cu_stream() as i64;
    let ran = unsafe {
        crate::cuda::ffi::qrwkv6_time_mix_cuda(
            (&raw const params).cast(),
            batch as i32,
            tm.head_dim as i32,
            dtype_code(dtype)?,
            stream,
        )
    };
    if ran == 0 {
        inference_tensor::bail!(
            "qrwkv6_time_mix_cuda: no kernel for heads of {}",
            tm.head_dim
        );
    }
    Ok(out)
}
