/// [`crate::ops::kda`] on CUDA, one kernel.
#[cfg(feature = "cuda")]
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
pub fn kda_cuda(
    kda: crate::ops::Kda<'_>,
    state: &inference_tensor::Tensor,
) -> inference_tensor::Result<inference_tensor::Tensor> {
    use inference_tensor::{DType, Tensor};
    let dtype = kda.q.dtype();
    let (batch, seq_len, hidden) = kda.q.dims3()?;
    if hidden != kda.heads * kda.head_dim {
        inference_tensor::bail!(
            "kda_cuda: {hidden} wide is not {} heads of {}",
            kda.heads,
            kda.head_dim
        );
    }
    if !state.is_contiguous() || state.dtype() != DType::F32 {
        inference_tensor::bail!("kda_cuda: the state must be contiguous f32");
    }
    let same = |t: &Tensor| -> inference_tensor::Result<Tensor> { t.to_dtype(dtype)?.contiguous() };
    let [q, k, v, f, gate, b] = [kda.q, kda.k, kda.v, kda.f, kda.gate, kda.b].map(same);
    let (q, k, v, f, gate, b) = (q?, k?, v?, f?, gate?, b?);
    let f32 = |t: &Tensor| -> inference_tensor::Result<Tensor> {
        t.to_dtype(DType::F32)?.flatten_all()?.contiguous()
    };
    let [dt_bias, a_log, norm_w] = [kda.dt_bias, kda.a_log, kda.norm_w].map(f32);
    let (dt_bias, a_log, norm_w) = (dt_bias?, a_log?, norm_w?);
    let slots = kda
        .slots
        .map(|s| s.to_dtype(DType::U32)?.contiguous())
        .transpose()?;
    let out = Tensor::zeros((batch, seq_len, hidden), dtype, q.device())?;
    let ptr = super::rwkv::device_ptr;
    let params = KdaParams {
        inputs: [
            ptr(&q)? as *const _,
            ptr(&k)? as *const _,
            ptr(&v)? as *const _,
            ptr(&f)? as *const _,
            ptr(&gate)? as *const _,
            ptr(&b)? as *const _,
        ],
        vectors: [
            ptr(&dt_bias)? as *const f32,
            ptr(&a_log)? as *const f32,
            ptr(&norm_w)? as *const f32,
        ],
        state: ptr(state)? as *mut f32,
        slots: match &slots {
            Some(s) => ptr(s)? as *const u32,
            None => std::ptr::null(),
        },
        out: ptr(&out)? as *mut _,
        seq_len: seq_len as i32,
        heads: kda.heads as i32,
        norm_eps: kda.norm_eps as f32,
    };
    let stream = q.device().as_cuda_device()?.cuda_stream().cu_stream() as i64;
    let ran = unsafe {
        crate::cuda::ffi::kda_cuda(
            (&raw const params).cast(),
            batch as i32,
            kda.head_dim as i32,
            super::rwkv::dtype_code(dtype)?,
            stream,
        )
    };
    if ran == 0 {
        inference_tensor::bail!("kda_cuda: no kernel for heads of {}", kda.head_dim);
    }
    Ok(out)
}

#[cfg(feature = "cuda")]
#[repr(C)]
struct KdaParams {
    inputs: [*const core::ffi::c_void; 6],
    vectors: [*const f32; 3],
    state: *mut f32,
    slots: *const u32,
    out: *mut core::ffi::c_void,
    seq_len: i32,
    heads: i32,
    norm_eps: f32,
}
