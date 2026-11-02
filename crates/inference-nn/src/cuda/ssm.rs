/// Mamba-2's selective scan on CUDA in one kernel launch, over the inputs [`crate::ops::SelectiveScan`] describes
/// (f32, B and C by group). `state` is `(batch, heads, head_dim, d_state)`, or with `slots` (u32, one per batch row) a
/// recurrent pool whose row `slots[b]` it updates in place, a slot past `i32::MAX` marking a padding row that leaves
/// the pool and scores 0. Returns `y` `(batch, seq_len, heads, head_dim)`, f32.
#[cfg(feature = "cuda")]
pub fn selective_scan_cuda(
    scan: crate::ops::SelectiveScan<'_>,
    state: &mut inference_tensor::Tensor,
    slots: Option<&inference_tensor::Tensor>,
) -> inference_tensor::Result<inference_tensor::Tensor> {
    use inference_tensor::cuda_backend::cudarc::driver::DevicePtr;

    let [x, dt, a, b, c, d, dt_bias] = [
        scan.x,
        scan.dt,
        scan.a,
        scan.b,
        scan.c,
        scan.d,
        scan.dt_bias,
    ]
    .map(inference_tensor::Tensor::contiguous);
    let (x, dt, a, b, c, d, dt_bias) = (x?, dt?, a?, b?, c?, d?, dt_bias?);
    let (batch_size, seq_len, n_heads, head_dim) = x.dims4()?;
    let (_, _, n_groups, d_state) = b.dims4()?;
    if !n_heads.is_multiple_of(n_groups) {
        inference_tensor::bail!(
            "selective_scan_cuda: {n_heads} heads do not split over {n_groups} groups"
        );
    }
    let dev = x.device().as_cuda_device()?;

    macro_rules! get_ptr {
        ($tensor:expr, $ty:ty) => {{
            let (s, l) = $tensor.storage_and_layout();
            let s = match &*s {
                inference_tensor::Storage::Cuda(c) => c.as_cuda_slice::<$ty>()?,
                _ => inference_tensor::bail!("selective_scan_cuda: tensor must be on CUDA"),
            };
            s.slice(l.start_offset()..).device_ptr(s.stream()).0
        }};
    }

    let state_ptr = get_ptr!(state, f32) as *mut f32;
    let slots = slots
        .map(inference_tensor::Tensor::contiguous)
        .transpose()?;
    let slot_ptr = match &slots {
        Some(slots) => get_ptr!(slots, u32) as *const i32,
        None => std::ptr::null(),
    };

    let y_buf = unsafe { dev.alloc::<f32>(batch_size * seq_len * n_heads * head_dim) }?;
    let stream = dev.cuda_stream().cu_stream() as i64;

    #[allow(clippy::cast_possible_truncation)]
    unsafe {
        crate::cuda::ffi::selective_scan_cuda(
            get_ptr!(x, f32) as *const f32,
            get_ptr!(dt, f32) as *const f32,
            get_ptr!(a, f32) as *const f32,
            get_ptr!(b, f32) as *const f32,
            get_ptr!(c, f32) as *const f32,
            get_ptr!(d, f32) as *const f32,
            get_ptr!(dt_bias, f32) as *const f32,
            state_ptr,
            y_buf.device_ptr(y_buf.stream()).0 as *mut f32,
            slot_ptr,
            batch_size as i32,
            n_heads as i32,
            n_groups as i32,
            head_dim as i32,
            d_state as i32,
            seq_len as i32,
            scan.dt_min,
            scan.dt_max,
            stream,
        );
    }

    let y_storage = inference_tensor::CudaStorage::wrap_cuda_slice(y_buf, dev.clone());
    Ok(inference_tensor::Tensor::from((
        inference_tensor::Storage::Cuda(y_storage),
        (batch_size, seq_len, n_heads, head_dim),
    )))
}

#[cfg(not(feature = "cuda"))]
pub fn selective_scan_cuda(
    _scan: crate::ops::SelectiveScan<'_>,
    _state: &mut inference_tensor::Tensor,
    _slots: Option<&inference_tensor::Tensor>,
) -> inference_tensor::Result<inference_tensor::Tensor> {
    inference_tensor::bail!("selective_scan_cuda requires the cuda feature")
}
