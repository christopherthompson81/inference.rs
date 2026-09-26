use super::*;

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn try_cuda_qk_rms_norm_rope(
    q: &Tensor,
    k: Option<&Tensor>,
    q_weight: &Tensor,
    k_weight: Option<&Tensor>,
    q_eps: f32,
    k_eps: f32,
    cos: &Tensor,
    sin: &Tensor,
    is_neox: bool,
    output_layout: QkRopeOutputLayout,
) -> Result<Option<(Tensor, Option<Tensor>)>> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice};
    use std::ffi::c_void;

    if !q.device().is_cuda() {
        return Ok(None);
    }

    let dtype = q.dtype();
    if !matches!(dtype, DType::BF16 | DType::F16 | DType::F32)
        || q_weight.dtype() != dtype
        || k_weight.is_some_and(|weight| weight.dtype() != dtype)
        || cos.dtype() != dtype
        || sin.dtype() != dtype
    {
        return Ok(None);
    }

    if !q_weight.device().same_device(q.device())
        || !cos.device().same_device(q.device())
        || !sin.device().same_device(q.device())
        || k.is_some_and(|k| !k.device().same_device(q.device()) || k.dtype() != dtype)
        || k_weight.is_some_and(|weight| !weight.device().same_device(q.device()))
    {
        return Ok(None);
    }

    let (batch, q_heads, seq_len, head_dim) = q.dims4()?;
    if seq_len == 1 && q.is_contiguous() && k.is_none_or(Tensor::is_contiguous) {
        return Ok(None);
    }

    let (k_heads, k_elem_count) = if let Some(k) = k {
        let (k_batch, k_heads, k_seq_len, k_head_dim) = k.dims4()?;
        if (k_batch, k_seq_len, k_head_dim) != (batch, seq_len, head_dim) {
            candle_core::bail!(
                "q/k shape mismatch for fused qk norm rope: {:?} vs {:?}",
                q.shape(),
                k.shape()
            );
        }
        let Some(k_weight) = k_weight else {
            candle_core::bail!("missing k norm weight for fused qk norm rope");
        };
        if k_weight.dims1()? != head_dim {
            candle_core::bail!(
                "k norm weight size {} does not match head dim {head_dim}",
                k_weight.dims1()?
            );
        }
        (k_heads, k.elem_count())
    } else {
        (0, 0)
    };

    if q_weight.dims1()? != head_dim {
        candle_core::bail!(
            "q norm weight size {} does not match head dim {head_dim}",
            q_weight.dims1()?
        );
    }

    let (cos_rows, rot_dim) = cos.dims2()?;
    if sin.dims2()? != (cos_rows, rot_dim) {
        candle_core::bail!(
            "cos/sin shape mismatch for fused qk norm rope: {:?} vs {:?}",
            cos.shape(),
            sin.shape()
        );
    }
    if rot_dim == 0 || rot_dim * 2 > head_dim {
        return Ok(None);
    }

    let cos_batch_stride = if cos_rows == seq_len {
        0
    } else if cos_rows == batch * seq_len {
        seq_len
    } else {
        candle_core::bail!(
            "cos/sin rows {cos_rows} do not match seq_len {seq_len} or batch*seq_len {}",
            batch * seq_len
        );
    };

    for (name, value) in [
        ("batch", batch),
        ("q_heads", q_heads),
        ("k_heads", k_heads),
        ("seq_len", seq_len),
        ("head_dim", head_dim),
        ("rot_dim", rot_dim),
        ("cos_batch_stride", cos_batch_stride),
    ] {
        if value > i32::MAX as usize {
            candle_core::bail!("fused qk norm rope {name} is too large: {value}");
        }
    }
    let batch_i32 = i32::try_from(batch).map_err(candle_core::Error::wrap)?;
    let q_heads_i32 = i32::try_from(q_heads).map_err(candle_core::Error::wrap)?;
    let k_heads_i32 = i32::try_from(k_heads).map_err(candle_core::Error::wrap)?;
    let seq_len_i32 = i32::try_from(seq_len).map_err(candle_core::Error::wrap)?;
    let head_dim_i32 = i32::try_from(head_dim).map_err(candle_core::Error::wrap)?;
    let rot_dim_i32 = i32::try_from(rot_dim).map_err(candle_core::Error::wrap)?;
    let cos_batch_stride_i32 = i32::try_from(cos_batch_stride).map_err(candle_core::Error::wrap)?;

    let cos = cos.contiguous()?;
    let sin = sin.contiguous()?;
    let q_weight = q_weight.contiguous()?;
    let k_weight = k_weight.map(Tensor::contiguous).transpose()?;

    let (q_storage, q_layout) = q.storage_and_layout();
    let q_storage = match &*q_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let k_storage_and_layout = k.map(Tensor::storage_and_layout);
    let (q_weight_storage, q_weight_layout) = q_weight.storage_and_layout();
    let q_weight_storage = match &*q_weight_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let k_weight_storage_and_layout = k_weight.as_ref().map(Tensor::storage_and_layout);
    let (cos_storage, cos_layout) = cos.storage_and_layout();
    let cos_storage = match &*cos_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let (sin_storage, sin_layout) = sin.storage_and_layout();
    let sin_storage = match &*sin_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };

    let dev = q_storage.device();
    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as i64;
    let q_shape = match output_layout {
        QkRopeOutputLayout::HeadsFirst => Shape::from_dims(&[batch, q_heads, seq_len, head_dim]),
        QkRopeOutputLayout::TokensFirst => Shape::from_dims(&[batch, seq_len, q_heads, head_dim]),
    };
    let k_shape = match output_layout {
        QkRopeOutputLayout::HeadsFirst => Shape::from_dims(&[batch, k_heads, seq_len, head_dim]),
        QkRopeOutputLayout::TokensFirst => Shape::from_dims(&[batch, seq_len, k_heads, head_dim]),
    };
    let q_elem_count = q.elem_count();

    let q_stride = q_layout.stride();
    let k_stride = k_storage_and_layout
        .as_ref()
        .map(|(_, layout)| layout.stride())
        .unwrap_or(&[0, 0, 0, 0]);

    macro_rules! launch {
        ($variant:ident, $ty:ty, $dtype_id:expr) => {{
            let CudaStorageSlice::$variant(q_src) = &q_storage.slice else {
                candle_core::bail!("fused qk norm rope q dtype mismatch");
            };
            let CudaStorageSlice::$variant(q_weight_src) = &q_weight_storage.slice else {
                candle_core::bail!("fused qk norm rope q weight dtype mismatch");
            };
            let CudaStorageSlice::$variant(cos_src) = &cos_storage.slice else {
                candle_core::bail!("fused qk norm rope cos dtype mismatch");
            };
            let CudaStorageSlice::$variant(sin_src) = &sin_storage.slice else {
                candle_core::bail!("fused qk norm rope sin dtype mismatch");
            };

            let mut q_out_buf = unsafe { dev.alloc::<$ty>(q_elem_count) }?;
            let mut k_out_buf = if k_elem_count == 0 {
                None
            } else {
                Some(unsafe { dev.alloc::<$ty>(k_elem_count) }?)
            };

            let (q_ptr, q_guard) = q_src.device_ptr(&stream);
            let q_ptr = unsafe { (q_ptr as *const $ty).add(q_layout.start_offset()) };
            let (q_weight_ptr, q_weight_guard) = q_weight_src.device_ptr(&stream);
            let q_weight_ptr =
                unsafe { (q_weight_ptr as *const $ty).add(q_weight_layout.start_offset()) };
            let (cos_ptr, cos_guard) = cos_src.device_ptr(&stream);
            let cos_ptr = unsafe { (cos_ptr as *const $ty).add(cos_layout.start_offset()) };
            let (sin_ptr, sin_guard) = sin_src.device_ptr(&stream);
            let sin_ptr = unsafe { (sin_ptr as *const $ty).add(sin_layout.start_offset()) };

            let mut k_guard = None;
            let k_ptr = if let Some((k_storage, k_layout)) = &k_storage_and_layout {
                let k_storage = match &**k_storage {
                    candle_core::Storage::Cuda(s) => s,
                    _ => return Ok(None),
                };
                let CudaStorageSlice::$variant(k_src) = &k_storage.slice else {
                    candle_core::bail!("fused qk norm rope k dtype mismatch");
                };
                let (ptr, guard) = k_src.device_ptr(&stream);
                k_guard = Some(guard);
                unsafe { (ptr as *const $ty).add(k_layout.start_offset()) }
            } else {
                std::ptr::null()
            };

            let mut k_weight_guard = None;
            let k_weight_ptr =
                if let Some((k_weight_storage, k_weight_layout)) = &k_weight_storage_and_layout {
                    let k_weight_storage = match &**k_weight_storage {
                        candle_core::Storage::Cuda(s) => s,
                        _ => return Ok(None),
                    };
                    let CudaStorageSlice::$variant(k_weight_src) = &k_weight_storage.slice else {
                        candle_core::bail!("fused qk norm rope k weight dtype mismatch");
                    };
                    let (ptr, guard) = k_weight_src.device_ptr(&stream);
                    k_weight_guard = Some(guard);
                    unsafe { (ptr as *const $ty).add(k_weight_layout.start_offset()) }
                } else {
                    q_weight_ptr
                };

            let (q_out_ptr, q_out_guard) = q_out_buf.device_ptr_mut(&stream);
            let mut k_out_guard = None;
            let k_out_ptr = if let Some(k_out_buf) = &mut k_out_buf {
                let (ptr, guard) = k_out_buf.device_ptr_mut(&stream);
                k_out_guard = Some(guard);
                ptr as *mut $ty
            } else {
                std::ptr::null_mut()
            };

            unsafe {
                ffi::qk_rms_norm_rope(
                    q_ptr as *const c_void,
                    k_ptr as *const c_void,
                    q_weight_ptr as *const c_void,
                    k_weight_ptr as *const c_void,
                    cos_ptr as *const c_void,
                    sin_ptr as *const c_void,
                    q_out_ptr as *mut c_void,
                    k_out_ptr as *mut c_void,
                    q_stride[0] as i64,
                    q_stride[1] as i64,
                    q_stride[2] as i64,
                    q_stride[3] as i64,
                    k_stride[0] as i64,
                    k_stride[1] as i64,
                    k_stride[2] as i64,
                    k_stride[3] as i64,
                    batch_i32,
                    q_heads_i32,
                    k_heads_i32,
                    seq_len_i32,
                    head_dim_i32,
                    rot_dim_i32,
                    cos_batch_stride_i32,
                    q_eps,
                    k_eps,
                    i32::from(is_neox),
                    $dtype_id,
                    i32::from(output_layout == QkRopeOutputLayout::TokensFirst),
                    stream_ptr,
                );
            }

            drop(q_guard);
            drop(q_weight_guard);
            drop(cos_guard);
            drop(sin_guard);
            drop(k_guard);
            drop(k_weight_guard);
            drop(q_out_guard);
            drop(k_out_guard);

            let q_storage = CudaStorage {
                slice: CudaStorageSlice::$variant(q_out_buf),
                device: dev.clone(),
            };
            let q_tensor = Tensor::from((candle_core::Storage::Cuda(q_storage), q_shape));

            let k_tensor = if let Some(k_out_buf) = k_out_buf {
                let k_storage = CudaStorage {
                    slice: CudaStorageSlice::$variant(k_out_buf),
                    device: dev.clone(),
                };
                Some(Tensor::from((
                    candle_core::Storage::Cuda(k_storage),
                    k_shape,
                )))
            } else {
                None
            };
            Ok(Some((q_tensor, k_tensor)))
        }};
    }

    match dtype {
        DType::BF16 => launch!(BF16, half::bf16, 1),
        DType::F16 => launch!(F16, half::f16, 0),
        DType::F32 => launch!(F32, f32, 2),
        _ => Ok(None),
    }
}

#[cfg(feature = "cuda")]
pub fn try_cuda_rope_sincos_positions(
    positions: &Tensor,
    inv_freq: &Tensor,
    dtype: DType,
) -> Result<Option<(Tensor, Tensor)>> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice};
    use std::ffi::c_void;

    if !positions.device().is_cuda()
        || positions.dtype() != DType::U32
        || inv_freq.dtype() != DType::F32
        || !inv_freq.device().same_device(positions.device())
        || !matches!(dtype, DType::BF16 | DType::F16 | DType::F32)
    {
        return Ok(None);
    }

    let rows = positions.dims1()?;
    let width = inv_freq.dims1()?;
    if rows == 0 || width == 0 {
        return Ok(None);
    }
    let rows_i32 = i32::try_from(rows).map_err(candle_core::Error::wrap)?;
    let width_i32 = i32::try_from(width).map_err(candle_core::Error::wrap)?;
    let elements = rows
        .checked_mul(width)
        .ok_or_else(|| candle_core::Error::msg("RoPE sincos output size overflow"))?;

    let positions = positions.contiguous()?;
    let inv_freq = inv_freq.contiguous()?;
    let (positions_storage, positions_layout) = positions.storage_and_layout();
    let positions_storage = match &*positions_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => return Ok(None),
    };
    let (inv_freq_storage, inv_freq_layout) = inv_freq.storage_and_layout();
    let inv_freq_storage = match &*inv_freq_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => return Ok(None),
    };
    let CudaStorageSlice::U32(positions_src) = &positions_storage.slice else {
        candle_core::bail!("RoPE sincos positions dtype mismatch");
    };
    let CudaStorageSlice::F32(inv_freq_src) = &inv_freq_storage.slice else {
        candle_core::bail!("RoPE sincos inverse frequency dtype mismatch");
    };

    let dev = positions_storage.device();
    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as i64;
    let (positions_ptr, positions_guard) = positions_src.device_ptr(&stream);
    let positions_ptr =
        unsafe { (positions_ptr as *const u32).add(positions_layout.start_offset()) };
    let (inv_freq_ptr, inv_freq_guard) = inv_freq_src.device_ptr(&stream);
    let inv_freq_ptr = unsafe { (inv_freq_ptr as *const f32).add(inv_freq_layout.start_offset()) };
    let output_shape = Shape::from_dims(&[rows, width]);

    macro_rules! launch {
        ($variant:ident, $ty:ty, $dtype_id:expr) => {{
            let mut cos_buf = unsafe { dev.alloc::<$ty>(elements) }?;
            let mut sin_buf = unsafe { dev.alloc::<$ty>(elements) }?;
            let (cos_ptr, cos_guard) = cos_buf.device_ptr_mut(&stream);
            let (sin_ptr, sin_guard) = sin_buf.device_ptr_mut(&stream);
            unsafe {
                ffi::rope_sincos_positions(
                    positions_ptr as *const c_void,
                    inv_freq_ptr as *const c_void,
                    cos_ptr as *mut c_void,
                    sin_ptr as *mut c_void,
                    rows_i32,
                    width_i32,
                    $dtype_id,
                    stream_ptr,
                );
            }
            drop(cos_guard);
            drop(sin_guard);

            let cos_storage = CudaStorage {
                slice: CudaStorageSlice::$variant(cos_buf),
                device: dev.clone(),
            };
            let sin_storage = CudaStorage {
                slice: CudaStorageSlice::$variant(sin_buf),
                device: dev.clone(),
            };
            let cos = Tensor::from((
                candle_core::Storage::Cuda(cos_storage),
                output_shape.clone(),
            ));
            let sin = Tensor::from((
                candle_core::Storage::Cuda(sin_storage),
                output_shape.clone(),
            ));
            Ok(Some((cos, sin)))
        }};
    }

    let result = match dtype {
        DType::BF16 => launch!(BF16, half::bf16, 1),
        DType::F16 => launch!(F16, half::f16, 0),
        DType::F32 => launch!(F32, f32, 2),
        _ => unreachable!(),
    };
    drop(positions_guard);
    drop(inv_freq_guard);
    result
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn try_cuda_qk_rms_norm_rope_positions(
    q: &Tensor,
    k: Option<&Tensor>,
    q_weight: &Tensor,
    k_weight: Option<&Tensor>,
    q_eps: f32,
    k_eps: f32,
    cos: &Tensor,
    sin: &Tensor,
    positions: &Tensor,
    is_neox: bool,
) -> Result<Option<(Tensor, Option<Tensor>)>> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice};
    use std::ffi::c_void;

    if !q.device().is_cuda() {
        return Ok(None);
    }

    let dtype = q.dtype();
    if !matches!(dtype, DType::BF16 | DType::F16 | DType::F32)
        || q_weight.dtype() != dtype
        || k_weight.is_some_and(|weight| weight.dtype() != dtype)
        || cos.dtype() != dtype
        || sin.dtype() != dtype
        || positions.dtype() != DType::U32
    {
        return Ok(None);
    }

    if !q_weight.device().same_device(q.device())
        || !cos.device().same_device(q.device())
        || !sin.device().same_device(q.device())
        || !positions.device().same_device(q.device())
        || k.is_some_and(|k| !k.device().same_device(q.device()) || k.dtype() != dtype)
        || k_weight.is_some_and(|weight| !weight.device().same_device(q.device()))
    {
        return Ok(None);
    }

    let (batch, q_heads, seq_len, head_dim) = q.dims4()?;
    let expected_positions = batch * seq_len;
    if positions.dims1()? != expected_positions {
        candle_core::bail!(
            "positions length {} does not match token count {expected_positions}",
            positions.dims1()?
        );
    }

    let (k_heads, k_elem_count) = if let Some(k) = k {
        let (k_batch, k_heads, k_seq_len, k_head_dim) = k.dims4()?;
        if (k_batch, k_seq_len, k_head_dim) != (batch, seq_len, head_dim) {
            candle_core::bail!(
                "q/k shape mismatch for fused qk norm rope positions: {:?} vs {:?}",
                q.shape(),
                k.shape()
            );
        }
        let Some(k_weight) = k_weight else {
            candle_core::bail!("missing k norm weight for fused qk norm rope positions");
        };
        if k_weight.dims1()? != head_dim {
            candle_core::bail!(
                "k norm weight size {} does not match head dim {head_dim}",
                k_weight.dims1()?
            );
        }
        (k_heads, k.elem_count())
    } else {
        (0, 0)
    };

    if q_weight.dims1()? != head_dim {
        candle_core::bail!(
            "q norm weight size {} does not match head dim {head_dim}",
            q_weight.dims1()?
        );
    }

    let (cos_rows, rot_dim) = cos.dims2()?;
    if sin.dims2()? != (cos_rows, rot_dim) {
        candle_core::bail!(
            "cos/sin shape mismatch for fused qk norm rope positions: {:?} vs {:?}",
            cos.shape(),
            sin.shape()
        );
    }
    if rot_dim == 0 || rot_dim * 2 > head_dim {
        return Ok(None);
    }
    for (name, value) in [
        ("batch", batch),
        ("q_heads", q_heads),
        ("k_heads", k_heads),
        ("seq_len", seq_len),
        ("head_dim", head_dim),
        ("rot_dim", rot_dim),
    ] {
        if value > i32::MAX as usize {
            candle_core::bail!("fused qk norm rope positions {name} is too large: {value}");
        }
    }
    let batch_i32 = i32::try_from(batch).map_err(candle_core::Error::wrap)?;
    let q_heads_i32 = i32::try_from(q_heads).map_err(candle_core::Error::wrap)?;
    let k_heads_i32 = i32::try_from(k_heads).map_err(candle_core::Error::wrap)?;
    let seq_len_i32 = i32::try_from(seq_len).map_err(candle_core::Error::wrap)?;
    let head_dim_i32 = i32::try_from(head_dim).map_err(candle_core::Error::wrap)?;
    let rot_dim_i32 = i32::try_from(rot_dim).map_err(candle_core::Error::wrap)?;

    let cos = cos.contiguous()?;
    let sin = sin.contiguous()?;
    let positions = positions.contiguous()?;
    let q_weight = q_weight.contiguous()?;
    let k_weight = k_weight.map(Tensor::contiguous).transpose()?;

    let (q_storage, q_layout) = q.storage_and_layout();
    let q_storage = match &*q_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let k_storage_and_layout = k.map(Tensor::storage_and_layout);
    let (q_weight_storage, q_weight_layout) = q_weight.storage_and_layout();
    let q_weight_storage = match &*q_weight_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let k_weight_storage_and_layout = k_weight.as_ref().map(Tensor::storage_and_layout);
    let (cos_storage, cos_layout) = cos.storage_and_layout();
    let cos_storage = match &*cos_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let (sin_storage, sin_layout) = sin.storage_and_layout();
    let sin_storage = match &*sin_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let (positions_storage, positions_layout) = positions.storage_and_layout();
    let positions_storage = match &*positions_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };

    let dev = q_storage.device();
    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as i64;
    let output_token_major = q.stride()[1] == head_dim && q.stride()[3] == 1;
    let output_shape = |heads| {
        Shape::from_dims(&if output_token_major {
            [batch, seq_len, heads, head_dim]
        } else {
            [batch, heads, seq_len, head_dim]
        })
    };
    let q_shape = output_shape(q_heads);
    let k_shape = output_shape(k_heads);
    let q_elem_count = q.elem_count();

    let q_stride = q_layout.stride();
    let k_stride = k_storage_and_layout
        .as_ref()
        .map(|(_, layout)| layout.stride())
        .unwrap_or(&[0, 0, 0, 0]);

    macro_rules! launch {
        ($variant:ident, $ty:ty, $dtype_id:expr) => {{
            let CudaStorageSlice::$variant(q_src) = &q_storage.slice else {
                candle_core::bail!("fused qk norm rope positions q dtype mismatch");
            };
            let CudaStorageSlice::$variant(q_weight_src) = &q_weight_storage.slice else {
                candle_core::bail!("fused qk norm rope positions q weight dtype mismatch");
            };
            let CudaStorageSlice::$variant(cos_src) = &cos_storage.slice else {
                candle_core::bail!("fused qk norm rope positions cos dtype mismatch");
            };
            let CudaStorageSlice::$variant(sin_src) = &sin_storage.slice else {
                candle_core::bail!("fused qk norm rope positions sin dtype mismatch");
            };
            let CudaStorageSlice::U32(positions_src) = &positions_storage.slice else {
                candle_core::bail!("fused qk norm rope positions dtype mismatch");
            };

            let mut q_out_buf = unsafe { dev.alloc::<$ty>(q_elem_count) }?;
            let mut k_out_buf = if k_elem_count == 0 {
                None
            } else {
                Some(unsafe { dev.alloc::<$ty>(k_elem_count) }?)
            };

            let (q_ptr, q_guard) = q_src.device_ptr(&stream);
            let q_ptr = unsafe { (q_ptr as *const $ty).add(q_layout.start_offset()) };
            let (q_weight_ptr, q_weight_guard) = q_weight_src.device_ptr(&stream);
            let q_weight_ptr =
                unsafe { (q_weight_ptr as *const $ty).add(q_weight_layout.start_offset()) };
            let (cos_ptr, cos_guard) = cos_src.device_ptr(&stream);
            let cos_ptr = unsafe { (cos_ptr as *const $ty).add(cos_layout.start_offset()) };
            let (sin_ptr, sin_guard) = sin_src.device_ptr(&stream);
            let sin_ptr = unsafe { (sin_ptr as *const $ty).add(sin_layout.start_offset()) };
            let (positions_ptr, positions_guard) = positions_src.device_ptr(&stream);
            let positions_ptr =
                unsafe { (positions_ptr as *const u32).add(positions_layout.start_offset()) };

            let mut k_guard = None;
            let k_ptr = if let Some((k_storage, k_layout)) = &k_storage_and_layout {
                let k_storage = match &**k_storage {
                    candle_core::Storage::Cuda(s) => s,
                    _ => return Ok(None),
                };
                let CudaStorageSlice::$variant(k_src) = &k_storage.slice else {
                    candle_core::bail!("fused qk norm rope positions k dtype mismatch");
                };
                let (ptr, guard) = k_src.device_ptr(&stream);
                k_guard = Some(guard);
                unsafe { (ptr as *const $ty).add(k_layout.start_offset()) }
            } else {
                std::ptr::null()
            };

            let mut k_weight_guard = None;
            let k_weight_ptr =
                if let Some((k_weight_storage, k_weight_layout)) = &k_weight_storage_and_layout {
                    let k_weight_storage = match &**k_weight_storage {
                        candle_core::Storage::Cuda(s) => s,
                        _ => return Ok(None),
                    };
                    let CudaStorageSlice::$variant(k_weight_src) = &k_weight_storage.slice else {
                        candle_core::bail!("fused qk norm rope positions k weight dtype mismatch");
                    };
                    let (ptr, guard) = k_weight_src.device_ptr(&stream);
                    k_weight_guard = Some(guard);
                    unsafe { (ptr as *const $ty).add(k_weight_layout.start_offset()) }
                } else {
                    q_weight_ptr
                };

            let (q_out_ptr, q_out_guard) = q_out_buf.device_ptr_mut(&stream);
            let mut k_out_guard = None;
            let k_out_ptr = if let Some(k_out_buf) = &mut k_out_buf {
                let (ptr, guard) = k_out_buf.device_ptr_mut(&stream);
                k_out_guard = Some(guard);
                ptr as *mut $ty
            } else {
                std::ptr::null_mut()
            };

            unsafe {
                ffi::qk_rms_norm_rope_positions(
                    q_ptr as *const c_void,
                    k_ptr as *const c_void,
                    q_weight_ptr as *const c_void,
                    k_weight_ptr as *const c_void,
                    cos_ptr as *const c_void,
                    sin_ptr as *const c_void,
                    positions_ptr as *const c_void,
                    q_out_ptr as *mut c_void,
                    k_out_ptr as *mut c_void,
                    q_stride[0] as i64,
                    q_stride[1] as i64,
                    q_stride[2] as i64,
                    q_stride[3] as i64,
                    k_stride[0] as i64,
                    k_stride[1] as i64,
                    k_stride[2] as i64,
                    k_stride[3] as i64,
                    batch_i32,
                    q_heads_i32,
                    k_heads_i32,
                    seq_len_i32,
                    head_dim_i32,
                    rot_dim_i32,
                    q_eps,
                    k_eps,
                    i32::from(is_neox),
                    $dtype_id,
                    i32::from(output_token_major),
                    stream_ptr,
                );
            }

            drop(q_guard);
            drop(q_weight_guard);
            drop(cos_guard);
            drop(sin_guard);
            drop(positions_guard);
            drop(k_guard);
            drop(k_weight_guard);
            drop(q_out_guard);
            drop(k_out_guard);

            let q_storage = CudaStorage {
                slice: CudaStorageSlice::$variant(q_out_buf),
                device: dev.clone(),
            };
            let q_tensor = Tensor::from((candle_core::Storage::Cuda(q_storage), q_shape));

            let k_tensor = if let Some(k_out_buf) = k_out_buf {
                let k_storage = CudaStorage {
                    slice: CudaStorageSlice::$variant(k_out_buf),
                    device: dev.clone(),
                };
                Some(Tensor::from((
                    candle_core::Storage::Cuda(k_storage),
                    k_shape,
                )))
            } else {
                None
            };

            let (q_tensor, k_tensor) = if output_token_major {
                (
                    q_tensor.transpose(1, 2)?,
                    k_tensor.map(|tensor| tensor.transpose(1, 2)).transpose()?,
                )
            } else {
                (q_tensor, k_tensor)
            };
            Ok(Some((q_tensor, k_tensor)))
        }};
    }

    match dtype {
        DType::BF16 => launch!(BF16, half::bf16, 1),
        DType::F16 => launch!(F16, half::f16, 0),
        DType::F32 => launch!(F32, f32, 2),
        _ => Ok(None),
    }
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn try_cuda_qkv_rms_norm_rope_positions(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    q_weight: &Tensor,
    k_weight: &Tensor,
    v_weight: &Tensor,
    q_eps: f32,
    k_eps: f32,
    v_eps: f32,
    cos: &Tensor,
    sin: &Tensor,
    positions: &Tensor,
    is_neox: bool,
) -> Result<Option<(Tensor, Tensor, Tensor)>> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice};
    use std::ffi::c_void;

    if !q.device().is_cuda() {
        return Ok(None);
    }

    let dtype = q.dtype();
    if !matches!(dtype, DType::BF16 | DType::F16 | DType::F32)
        || k.dtype() != dtype
        || v.dtype() != dtype
        || q_weight.dtype() != dtype
        || k_weight.dtype() != dtype
        || v_weight.dtype() != dtype
        || cos.dtype() != dtype
        || sin.dtype() != dtype
        || positions.dtype() != DType::U32
    {
        return Ok(None);
    }

    if !q_weight.device().same_device(q.device())
        || !k_weight.device().same_device(q.device())
        || !v_weight.device().same_device(q.device())
        || !cos.device().same_device(q.device())
        || !sin.device().same_device(q.device())
        || !positions.device().same_device(q.device())
        || !k.device().same_device(q.device())
        || !v.device().same_device(q.device())
    {
        return Ok(None);
    }

    let (batch, q_heads, seq_len, head_dim) = q.dims4()?;
    let (k_batch, k_heads, k_seq_len, k_head_dim) = k.dims4()?;
    let (v_batch, v_heads, v_seq_len, v_head_dim) = v.dims4()?;
    if (k_batch, k_seq_len, k_head_dim) != (batch, seq_len, head_dim)
        || (v_batch, v_heads, v_seq_len, v_head_dim) != (batch, k_heads, seq_len, head_dim)
    {
        candle_core::bail!(
            "q/k/v shape mismatch for fused qkv norm rope positions: {:?}, {:?}, {:?}",
            q.shape(),
            k.shape(),
            v.shape()
        );
    }
    let expected_positions = batch * seq_len;
    if positions.dims1()? != expected_positions {
        candle_core::bail!(
            "positions length {} does not match token count {expected_positions}",
            positions.dims1()?
        );
    }
    if q_weight.dims1()? != head_dim
        || k_weight.dims1()? != head_dim
        || v_weight.dims1()? != head_dim
    {
        candle_core::bail!("qkv norm weight size does not match head dim {head_dim}");
    }

    let (cos_rows, rot_dim) = cos.dims2()?;
    if sin.dims2()? != (cos_rows, rot_dim) {
        candle_core::bail!(
            "cos/sin shape mismatch for fused qkv norm rope positions: {:?} vs {:?}",
            cos.shape(),
            sin.shape()
        );
    }
    if rot_dim == 0 || rot_dim * 2 > head_dim {
        return Ok(None);
    }
    for (name, value) in [
        ("batch", batch),
        ("q_heads", q_heads),
        ("k_heads", k_heads),
        ("seq_len", seq_len),
        ("head_dim", head_dim),
        ("rot_dim", rot_dim),
    ] {
        if value > i32::MAX as usize {
            candle_core::bail!("fused qkv norm rope positions {name} is too large: {value}");
        }
    }
    let batch_i32 = i32::try_from(batch).map_err(candle_core::Error::wrap)?;
    let q_heads_i32 = i32::try_from(q_heads).map_err(candle_core::Error::wrap)?;
    let k_heads_i32 = i32::try_from(k_heads).map_err(candle_core::Error::wrap)?;
    let seq_len_i32 = i32::try_from(seq_len).map_err(candle_core::Error::wrap)?;
    let head_dim_i32 = i32::try_from(head_dim).map_err(candle_core::Error::wrap)?;
    let rot_dim_i32 = i32::try_from(rot_dim).map_err(candle_core::Error::wrap)?;

    let cos = cos.contiguous()?;
    let sin = sin.contiguous()?;
    let positions = positions.contiguous()?;
    let q_weight = q_weight.contiguous()?;
    let k_weight = k_weight.contiguous()?;
    let v_weight = v_weight.contiguous()?;

    let (q_storage, q_layout) = q.storage_and_layout();
    let q_storage = match &*q_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let (k_storage, k_layout) = k.storage_and_layout();
    let k_storage = match &*k_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let (v_storage, v_layout) = v.storage_and_layout();
    let v_storage = match &*v_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let (q_weight_storage, q_weight_layout) = q_weight.storage_and_layout();
    let q_weight_storage = match &*q_weight_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let (k_weight_storage, k_weight_layout) = k_weight.storage_and_layout();
    let k_weight_storage = match &*k_weight_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let (v_weight_storage, v_weight_layout) = v_weight.storage_and_layout();
    let v_weight_storage = match &*v_weight_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let (cos_storage, cos_layout) = cos.storage_and_layout();
    let cos_storage = match &*cos_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let (sin_storage, sin_layout) = sin.storage_and_layout();
    let sin_storage = match &*sin_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let (positions_storage, positions_layout) = positions.storage_and_layout();
    let positions_storage = match &*positions_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };

    let dev = q_storage.device();
    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as i64;
    let q_shape = Shape::from_dims(&[batch, q_heads, seq_len, head_dim]);
    let kv_shape = Shape::from_dims(&[batch, k_heads, seq_len, head_dim]);
    let q_elem_count = q.elem_count();
    let kv_elem_count = k.elem_count();

    let q_stride = q_layout.stride();
    let k_stride = k_layout.stride();
    let v_stride = v_layout.stride();

    macro_rules! launch {
        ($variant:ident, $ty:ty, $dtype_id:expr) => {{
            let CudaStorageSlice::$variant(q_src) = &q_storage.slice else {
                candle_core::bail!("fused qkv norm rope positions q dtype mismatch");
            };
            let CudaStorageSlice::$variant(k_src) = &k_storage.slice else {
                candle_core::bail!("fused qkv norm rope positions k dtype mismatch");
            };
            let CudaStorageSlice::$variant(v_src) = &v_storage.slice else {
                candle_core::bail!("fused qkv norm rope positions v dtype mismatch");
            };
            let CudaStorageSlice::$variant(q_weight_src) = &q_weight_storage.slice else {
                candle_core::bail!("fused qkv norm rope positions q weight dtype mismatch");
            };
            let CudaStorageSlice::$variant(k_weight_src) = &k_weight_storage.slice else {
                candle_core::bail!("fused qkv norm rope positions k weight dtype mismatch");
            };
            let CudaStorageSlice::$variant(v_weight_src) = &v_weight_storage.slice else {
                candle_core::bail!("fused qkv norm rope positions v weight dtype mismatch");
            };
            let CudaStorageSlice::$variant(cos_src) = &cos_storage.slice else {
                candle_core::bail!("fused qkv norm rope positions cos dtype mismatch");
            };
            let CudaStorageSlice::$variant(sin_src) = &sin_storage.slice else {
                candle_core::bail!("fused qkv norm rope positions sin dtype mismatch");
            };
            let CudaStorageSlice::U32(positions_src) = &positions_storage.slice else {
                candle_core::bail!("fused qkv norm rope positions dtype mismatch");
            };

            let mut q_out_buf = unsafe { dev.alloc::<$ty>(q_elem_count) }?;
            let mut k_out_buf = unsafe { dev.alloc::<$ty>(kv_elem_count) }?;
            let mut v_out_buf = unsafe { dev.alloc::<$ty>(kv_elem_count) }?;

            let (q_ptr, q_guard) = q_src.device_ptr(&stream);
            let q_ptr = unsafe { (q_ptr as *const $ty).add(q_layout.start_offset()) };
            let (k_ptr, k_guard) = k_src.device_ptr(&stream);
            let k_ptr = unsafe { (k_ptr as *const $ty).add(k_layout.start_offset()) };
            let (v_ptr, v_guard) = v_src.device_ptr(&stream);
            let v_ptr = unsafe { (v_ptr as *const $ty).add(v_layout.start_offset()) };
            let (q_weight_ptr, q_weight_guard) = q_weight_src.device_ptr(&stream);
            let q_weight_ptr =
                unsafe { (q_weight_ptr as *const $ty).add(q_weight_layout.start_offset()) };
            let (k_weight_ptr, k_weight_guard) = k_weight_src.device_ptr(&stream);
            let k_weight_ptr =
                unsafe { (k_weight_ptr as *const $ty).add(k_weight_layout.start_offset()) };
            let (v_weight_ptr, v_weight_guard) = v_weight_src.device_ptr(&stream);
            let v_weight_ptr =
                unsafe { (v_weight_ptr as *const $ty).add(v_weight_layout.start_offset()) };
            let (cos_ptr, cos_guard) = cos_src.device_ptr(&stream);
            let cos_ptr = unsafe { (cos_ptr as *const $ty).add(cos_layout.start_offset()) };
            let (sin_ptr, sin_guard) = sin_src.device_ptr(&stream);
            let sin_ptr = unsafe { (sin_ptr as *const $ty).add(sin_layout.start_offset()) };
            let (positions_ptr, positions_guard) = positions_src.device_ptr(&stream);
            let positions_ptr =
                unsafe { (positions_ptr as *const u32).add(positions_layout.start_offset()) };

            let (q_out_ptr, q_out_guard) = q_out_buf.device_ptr_mut(&stream);
            let (k_out_ptr, k_out_guard) = k_out_buf.device_ptr_mut(&stream);
            let (v_out_ptr, v_out_guard) = v_out_buf.device_ptr_mut(&stream);

            unsafe {
                ffi::qkv_rms_norm_rope_positions(
                    q_ptr as *const c_void,
                    k_ptr as *const c_void,
                    v_ptr as *const c_void,
                    q_weight_ptr as *const c_void,
                    k_weight_ptr as *const c_void,
                    v_weight_ptr as *const c_void,
                    cos_ptr as *const c_void,
                    sin_ptr as *const c_void,
                    positions_ptr as *const c_void,
                    q_out_ptr as *mut c_void,
                    k_out_ptr as *mut c_void,
                    v_out_ptr as *mut c_void,
                    q_stride[0] as i64,
                    q_stride[1] as i64,
                    q_stride[2] as i64,
                    q_stride[3] as i64,
                    k_stride[0] as i64,
                    k_stride[1] as i64,
                    k_stride[2] as i64,
                    k_stride[3] as i64,
                    v_stride[0] as i64,
                    v_stride[1] as i64,
                    v_stride[2] as i64,
                    v_stride[3] as i64,
                    batch_i32,
                    q_heads_i32,
                    k_heads_i32,
                    seq_len_i32,
                    head_dim_i32,
                    rot_dim_i32,
                    q_eps,
                    k_eps,
                    v_eps,
                    i32::from(is_neox),
                    $dtype_id,
                    stream_ptr,
                );
            }

            drop(q_guard);
            drop(k_guard);
            drop(v_guard);
            drop(q_weight_guard);
            drop(k_weight_guard);
            drop(v_weight_guard);
            drop(cos_guard);
            drop(sin_guard);
            drop(positions_guard);
            drop(q_out_guard);
            drop(k_out_guard);
            drop(v_out_guard);

            let q_storage = CudaStorage {
                slice: CudaStorageSlice::$variant(q_out_buf),
                device: dev.clone(),
            };
            let k_storage = CudaStorage {
                slice: CudaStorageSlice::$variant(k_out_buf),
                device: dev.clone(),
            };
            let v_storage = CudaStorage {
                slice: CudaStorageSlice::$variant(v_out_buf),
                device: dev.clone(),
            };
            Ok(Some((
                Tensor::from((candle_core::Storage::Cuda(q_storage), q_shape)),
                Tensor::from((candle_core::Storage::Cuda(k_storage), kv_shape.clone())),
                Tensor::from((candle_core::Storage::Cuda(v_storage), kv_shape)),
            )))
        }};
    }

    match dtype {
        DType::BF16 => launch!(BF16, half::bf16, 1),
        DType::F16 => launch!(F16, half::f16, 0),
        DType::F32 => launch!(F32, f32, 2),
        _ => Ok(None),
    }
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QkRopeOutputLayout {
    HeadsFirst,
    TokensFirst,
}
