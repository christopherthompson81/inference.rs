use super::*;

#[cfg(feature = "cuda")]
pub fn try_cuda_rms_norm_strided_4d(
    input: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<Option<Tensor>> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice};
    use std::ffi::c_void;

    if !input.device().is_cuda() || input.rank() != 4 {
        return Ok(None);
    }
    let dtype = input.dtype();
    if !matches!(dtype, DType::BF16 | DType::F16 | DType::F32) || weight.dtype() != dtype {
        return Ok(None);
    }
    if !weight.device().same_device(input.device()) {
        return Ok(None);
    }

    let (batch, heads, seq_len, head_dim) = input.dims4()?;
    if weight.dims1()? != head_dim {
        candle_core::bail!(
            "cuda_rms_norm_strided_4d weight size {} does not match head dim {head_dim}",
            weight.dims1()?
        );
    }
    if input.elem_count() == 0 {
        return Ok(None);
    }
    for (name, value) in [
        ("batch", batch),
        ("heads", heads),
        ("seq_len", seq_len),
        ("head_dim", head_dim),
    ] {
        if value > i32::MAX as usize {
            candle_core::bail!("cuda_rms_norm_strided_4d {name} is too large: {value}");
        }
    }

    let (input_storage, input_layout) = input.storage_and_layout();
    if input_layout.is_contiguous() {
        return Ok(None);
    }
    let input_storage = match &*input_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let weight = weight.contiguous()?;
    let (weight_storage, weight_layout) = weight.storage_and_layout();
    let weight_storage = match &*weight_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let dev = input_storage.device();
    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as i64;
    let shape = input.shape().clone();
    let elem_count = input.elem_count();
    let stride = input_layout.stride();
    let batch_i32 = i32::try_from(batch).map_err(candle_core::Error::wrap)?;
    let heads_i32 = i32::try_from(heads).map_err(candle_core::Error::wrap)?;
    let seq_len_i32 = i32::try_from(seq_len).map_err(candle_core::Error::wrap)?;
    let head_dim_i32 = i32::try_from(head_dim).map_err(candle_core::Error::wrap)?;

    macro_rules! launch {
        ($variant:ident, $ty:ty, $ffi_fn:ident) => {{
            let CudaStorageSlice::$variant(src) = &input_storage.slice else {
                candle_core::bail!("cuda_rms_norm_strided_4d input dtype mismatch");
            };
            let CudaStorageSlice::$variant(weight_src) = &weight_storage.slice else {
                candle_core::bail!("cuda_rms_norm_strided_4d weight dtype mismatch");
            };
            let mut out = unsafe { dev.alloc::<$ty>(elem_count) }?;
            let (src_ptr, src_guard) = src.device_ptr(&stream);
            let (weight_ptr, weight_guard) = weight_src.device_ptr(&stream);
            let (out_ptr, out_guard) = out.device_ptr_mut(&stream);
            let src_ptr = unsafe { (src_ptr as *const $ty).add(input_layout.start_offset()) };
            let weight_ptr =
                unsafe { (weight_ptr as *const $ty).add(weight_layout.start_offset()) };

            unsafe {
                ffi::$ffi_fn(
                    src_ptr as *const c_void,
                    weight_ptr as *const c_void,
                    out_ptr as *mut c_void,
                    stride[0] as i64,
                    stride[1] as i64,
                    stride[2] as i64,
                    stride[3] as i64,
                    batch_i32,
                    heads_i32,
                    seq_len_i32,
                    head_dim_i32,
                    eps,
                    stream_ptr,
                );
            }

            drop(src_guard);
            drop(weight_guard);
            drop(out_guard);

            let out_storage = CudaStorage {
                slice: CudaStorageSlice::$variant(out),
                device: dev.clone(),
            };
            Ok(Some(Tensor::from((
                candle_core::Storage::Cuda(out_storage),
                shape,
            ))))
        }};
    }

    match dtype {
        DType::BF16 => launch!(BF16, half::bf16, rms_norm_strided_4d_bf16),
        DType::F16 => launch!(F16, half::f16, rms_norm_strided_4d_f16),
        DType::F32 => launch!(F32, f32, rms_norm_strided_4d_f32),
        _ => Ok(None),
    }
}

#[cfg(feature = "cuda")]
pub fn cuda_rms_norm_residual(
    input: &Tensor,
    residual: &Tensor,
    weight: &Tensor,
    scale: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice};
    use std::ffi::c_void;

    if input.shape() != residual.shape() {
        candle_core::bail!(
            "cuda_rms_norm_residual input/residual shape mismatch: {:?} vs {:?}",
            input.shape(),
            residual.shape()
        );
    }
    if input.dtype() != residual.dtype() || input.dtype() != weight.dtype() {
        candle_core::bail!(
            "cuda_rms_norm_residual dtype mismatch: input {:?}, residual {:?}, weight {:?}",
            input.dtype(),
            residual.dtype(),
            weight.dtype()
        );
    }
    if !matches!(input.dtype(), DType::BF16 | DType::F16 | DType::F32) {
        candle_core::bail!(
            "cuda_rms_norm_residual only supports BF16/F16/F32, got {:?}",
            input.dtype()
        );
    }
    if !residual.device().same_device(input.device())
        || !weight.device().same_device(input.device())
    {
        candle_core::bail!("cuda_rms_norm_residual tensors must be on the same CUDA device");
    }
    if let Some(scale) = scale {
        if scale.elem_count() != 1 {
            candle_core::bail!(
                "cuda_rms_norm_residual scale must have one element, got {}",
                scale.elem_count()
            );
        }
        if scale.dtype() != input.dtype() {
            candle_core::bail!(
                "cuda_rms_norm_residual scale dtype mismatch: input {:?}, scale {:?}",
                input.dtype(),
                scale.dtype()
            );
        }
        if !scale.device().same_device(input.device()) {
            candle_core::bail!("cuda_rms_norm_residual scale must be on the same CUDA device");
        }
    }

    let ncols = input.dim(D::Minus1)?;
    if weight.dims1()? != ncols {
        candle_core::bail!(
            "cuda_rms_norm_residual weight size {} does not match last dim {ncols}",
            weight.dims1()?
        );
    }
    let elem_count = input.elem_count();
    if elem_count == 0 {
        candle_core::bail!("cuda_rms_norm_residual got empty input");
    }
    let nrows = elem_count / ncols;
    if nrows > i32::MAX as usize || ncols > i32::MAX as usize {
        candle_core::bail!(
            "cuda_rms_norm_residual input is too large: nrows={nrows}, ncols={ncols}"
        );
    }
    let nrows_i32 = i32::try_from(nrows).map_err(candle_core::Error::wrap)?;
    let ncols_i32 = i32::try_from(ncols).map_err(candle_core::Error::wrap)?;

    let input = input.contiguous()?;
    let residual = residual.contiguous()?;
    let weight = weight.contiguous()?;
    let scale = scale.map(Tensor::contiguous).transpose()?;

    let (input_storage, input_layout) = input.storage_and_layout();
    let input_storage = match &*input_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_rms_norm_residual requires CUDA input"),
    };
    let (residual_storage, residual_layout) = residual.storage_and_layout();
    let residual_storage = match &*residual_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_rms_norm_residual requires CUDA residual"),
    };
    let (weight_storage, weight_layout) = weight.storage_and_layout();
    let weight_storage = match &*weight_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_rms_norm_residual requires CUDA weight"),
    };
    let scale_storage_and_layout = scale.as_ref().map(|scale| scale.storage_and_layout());

    let dev = input_storage.device();
    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as i64;
    let shape = input.shape().clone();

    macro_rules! launch {
        ($variant:ident, $ty:ty, $ffi_fn:ident) => {{
            let CudaStorageSlice::$variant(src) = &input_storage.slice else {
                candle_core::bail!("cuda_rms_norm_residual input dtype mismatch");
            };
            let CudaStorageSlice::$variant(residual_src) = &residual_storage.slice else {
                candle_core::bail!("cuda_rms_norm_residual residual dtype mismatch");
            };
            let CudaStorageSlice::$variant(weight_src) = &weight_storage.slice else {
                candle_core::bail!("cuda_rms_norm_residual weight dtype mismatch");
            };
            let (scale_ptr, scale_guard) =
                if let Some((scale_storage, scale_layout)) = &scale_storage_and_layout {
                    let scale_storage = match &**scale_storage {
                        candle_core::Storage::Cuda(s) => s,
                        _ => candle_core::bail!("cuda_rms_norm_residual requires CUDA scale"),
                    };
                    let CudaStorageSlice::$variant(scale_src) = &scale_storage.slice else {
                        candle_core::bail!("cuda_rms_norm_residual scale dtype mismatch");
                    };
                    let (scale_ptr, scale_guard) = scale_src.device_ptr(&stream);
                    (
                        unsafe { (scale_ptr as *const $ty).add(scale_layout.start_offset()) }
                            as *const c_void,
                        Some(scale_guard),
                    )
                } else {
                    (std::ptr::null(), None)
                };

            let mut out = unsafe { dev.alloc::<$ty>(elem_count) }?;
            let (src_ptr, src_guard) = src.device_ptr(&stream);
            let (residual_ptr, residual_guard) = residual_src.device_ptr(&stream);
            let (weight_ptr, weight_guard) = weight_src.device_ptr(&stream);
            let (out_ptr, out_guard) = out.device_ptr_mut(&stream);
            let src_ptr = unsafe { (src_ptr as *const $ty).add(input_layout.start_offset()) };
            let residual_ptr =
                unsafe { (residual_ptr as *const $ty).add(residual_layout.start_offset()) };
            let weight_ptr =
                unsafe { (weight_ptr as *const $ty).add(weight_layout.start_offset()) };

            unsafe {
                ffi::$ffi_fn(
                    src_ptr as *const c_void,
                    residual_ptr as *const c_void,
                    weight_ptr as *const c_void,
                    scale_ptr,
                    out_ptr as *mut c_void,
                    nrows_i32,
                    ncols_i32,
                    eps,
                    stream_ptr,
                );
            }

            drop(src_guard);
            drop(residual_guard);
            drop(weight_guard);
            drop(scale_guard);
            drop(out_guard);

            let out_storage = CudaStorage {
                slice: CudaStorageSlice::$variant(out),
                device: dev.clone(),
            };
            Ok(Tensor::from((
                candle_core::Storage::Cuda(out_storage),
                shape,
            )))
        }};
    }
    match input.dtype() {
        DType::BF16 => launch!(BF16, half::bf16, rms_norm_residual_bf16),
        DType::F16 => launch!(F16, half::f16, rms_norm_residual_f16),
        DType::F32 => launch!(F32, f32, rms_norm_residual_f32),
        dtype => candle_core::bail!("cuda_rms_norm_residual unsupported dtype {dtype:?}"),
    }
}

#[cfg(feature = "cuda")]
pub fn cuda_add_rms_norm(
    input: &Tensor,
    residual: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice};
    use std::ffi::c_void;

    if input.shape() != residual.shape() {
        candle_core::bail!(
            "cuda_add_rms_norm input/residual shape mismatch: {:?} vs {:?}",
            input.shape(),
            residual.shape()
        );
    }
    if input.dtype() != residual.dtype() || input.dtype() != weight.dtype() {
        candle_core::bail!(
            "cuda_add_rms_norm dtype mismatch: input {:?}, residual {:?}, weight {:?}",
            input.dtype(),
            residual.dtype(),
            weight.dtype()
        );
    }
    if !matches!(input.dtype(), DType::BF16 | DType::F16 | DType::F32) {
        candle_core::bail!(
            "cuda_add_rms_norm only supports BF16/F16/F32, got {:?}",
            input.dtype()
        );
    }
    if !residual.device().same_device(input.device())
        || !weight.device().same_device(input.device())
    {
        candle_core::bail!("cuda_add_rms_norm tensors must be on the same CUDA device");
    }

    let ncols = input.dim(D::Minus1)?;
    if weight.dims1()? != ncols {
        candle_core::bail!(
            "cuda_add_rms_norm weight size {} does not match last dim {ncols}",
            weight.dims1()?
        );
    }
    let elem_count = input.elem_count();
    if ncols == 0 || elem_count == 0 {
        candle_core::bail!("cuda_add_rms_norm got empty input");
    }
    let nrows = elem_count / ncols;
    if nrows > i32::MAX as usize || ncols > i32::MAX as usize {
        candle_core::bail!("cuda_add_rms_norm input is too large: nrows={nrows}, ncols={ncols}");
    }
    let nrows_i32 = i32::try_from(nrows).map_err(candle_core::Error::wrap)?;
    let ncols_i32 = i32::try_from(ncols).map_err(candle_core::Error::wrap)?;

    let input = input.contiguous()?;
    let residual = residual.contiguous()?;
    let weight = weight.contiguous()?;

    let (input_storage, input_layout) = input.storage_and_layout();
    let input_storage = match &*input_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("cuda_add_rms_norm requires CUDA input"),
    };
    let (residual_storage, residual_layout) = residual.storage_and_layout();
    let residual_storage = match &*residual_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("cuda_add_rms_norm requires CUDA residual"),
    };
    let (weight_storage, weight_layout) = weight.storage_and_layout();
    let weight_storage = match &*weight_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("cuda_add_rms_norm requires CUDA weight"),
    };
    let dev = input_storage.device();
    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as i64;
    let shape = input.shape().clone();

    macro_rules! launch {
        ($variant:ident, $ty:ty, $ffi_fn:ident) => {{
            let CudaStorageSlice::$variant(src) = &input_storage.slice else {
                candle_core::bail!("cuda_add_rms_norm input dtype mismatch");
            };
            let CudaStorageSlice::$variant(residual_src) = &residual_storage.slice else {
                candle_core::bail!("cuda_add_rms_norm residual dtype mismatch");
            };
            let CudaStorageSlice::$variant(weight_src) = &weight_storage.slice else {
                candle_core::bail!("cuda_add_rms_norm weight dtype mismatch");
            };

            let mut residual_out = unsafe { dev.alloc::<$ty>(elem_count) }?;
            let mut norm_out = unsafe { dev.alloc::<$ty>(elem_count) }?;
            let (src_ptr, src_guard) = src.device_ptr(&stream);
            let (residual_ptr, residual_guard) = residual_src.device_ptr(&stream);
            let (weight_ptr, weight_guard) = weight_src.device_ptr(&stream);
            let (residual_out_ptr, residual_out_guard) = residual_out.device_ptr_mut(&stream);
            let (norm_out_ptr, norm_out_guard) = norm_out.device_ptr_mut(&stream);

            let src_ptr = unsafe { (src_ptr as *const $ty).add(input_layout.start_offset()) };
            let residual_ptr =
                unsafe { (residual_ptr as *const $ty).add(residual_layout.start_offset()) };
            let weight_ptr =
                unsafe { (weight_ptr as *const $ty).add(weight_layout.start_offset()) };

            unsafe {
                ffi::$ffi_fn(
                    src_ptr as *const c_void,
                    residual_ptr as *const c_void,
                    weight_ptr as *const c_void,
                    residual_out_ptr as *mut c_void,
                    norm_out_ptr as *mut c_void,
                    nrows_i32,
                    ncols_i32,
                    eps,
                    stream_ptr,
                );
            }

            drop(src_guard);
            drop(residual_guard);
            drop(weight_guard);
            drop(residual_out_guard);
            drop(norm_out_guard);

            let residual_storage = CudaStorage {
                slice: CudaStorageSlice::$variant(residual_out),
                device: dev.clone(),
            };
            let norm_storage = CudaStorage {
                slice: CudaStorageSlice::$variant(norm_out),
                device: dev.clone(),
            };
            Ok((
                Tensor::from((candle_core::Storage::Cuda(residual_storage), shape.clone())),
                Tensor::from((candle_core::Storage::Cuda(norm_storage), shape)),
            ))
        }};
    }
    match input.dtype() {
        DType::BF16 => launch!(BF16, half::bf16, add_rms_norm_bf16),
        DType::F16 => launch!(F16, half::f16, add_rms_norm_f16),
        DType::F32 => launch!(F32, f32, add_rms_norm_f32),
        dtype => candle_core::bail!("cuda_add_rms_norm unsupported dtype {dtype:?}"),
    }
}

#[cfg(feature = "metal")]
pub fn metal_rms_norm_residual(
    input: &Tensor,
    residual: &Tensor,
    weight: &Tensor,
    scale: Option<&Tensor>,
    eps: f32,
) -> Result<Option<Tensor>> {
    use candle_core::{backend::BackendStorage, MetalStorage, Shape, Storage};

    if input.shape() != residual.shape() {
        return Ok(None);
    }
    let n_cols = input.dim(D::Minus1)?;
    if weight.dims1()? != n_cols {
        return Ok(None);
    }
    let n_rows = input.elem_count() / n_cols;
    if n_rows == 0 {
        return Ok(None);
    }
    if let Some(scale) = scale {
        if scale.elem_count() != 1 {
            return Ok(None);
        }
    }
    let input = input.contiguous()?;
    let residual = residual.contiguous()?;
    let weight = weight.contiguous()?;
    let scale_t = scale.map(Tensor::contiguous).transpose()?;

    let (input_storage, input_layout) = input.storage_and_layout();
    let Storage::Metal(input_storage) = &*input_storage else {
        return Ok(None);
    };
    let (residual_storage, residual_layout) = residual.storage_and_layout();
    let Storage::Metal(residual_storage) = &*residual_storage else {
        return Ok(None);
    };
    let (weight_storage, weight_layout) = weight.storage_and_layout();
    let Storage::Metal(weight_storage) = &*weight_storage else {
        return Ok(None);
    };
    let scale_storage_and_layout = scale_t.as_ref().map(|s| s.storage_and_layout());
    let scale_metal = match scale_storage_and_layout.as_ref() {
        Some((s, l)) => {
            let Storage::Metal(s) = &**s else {
                return Ok(None);
            };
            Some((s, l))
        }
        None => None,
    };

    let device = input_storage.device().clone();
    let dtype = input.dtype();
    let out_buf = device.new_buffer(input.elem_count(), dtype, "rmsnorm-residual-out")?;

    let encoder = device.command_encoder()?;
    encoder.set_label("rmsnorm-residual");

    let x_offset = input_layout.start_offset() * dtype.size_in_bytes();
    let res_offset = residual_layout.start_offset() * dtype.size_in_bytes();
    let w_offset = weight_layout.start_offset() * dtype.size_in_bytes();
    let scale_arg = scale_metal
        .as_ref()
        .map(|(s, l)| (s.buffer(), l.start_offset() * dtype.size_in_bytes()));

    inference_quant::metal_kernels::call_rmsnorm_residual(
        device.device(),
        &encoder,
        &inference_quant::metal_kernels::Kernels::new(),
        dtype,
        (input_storage.buffer(), x_offset),
        (residual_storage.buffer(), res_offset),
        (weight_storage.buffer(), w_offset),
        scale_arg,
        &out_buf,
        n_cols,
        n_rows,
        eps,
    )
    .map_err(candle_core::Error::wrap)?;

    let out = Tensor::from((
        Storage::Metal(MetalStorage::new(
            out_buf,
            device.clone(),
            input.elem_count(),
            dtype,
        )),
        Shape::from(input.dims()),
    ));
    Ok(Some(out))
}

#[cfg(feature = "cuda")]
pub fn cuda_rms_norm_residual_then_rms_norm(
    input: &Tensor,
    residual: &Tensor,
    residual_weight: &Tensor,
    scale: Option<&Tensor>,
    norm_weight: &Tensor,
    residual_eps: f32,
    norm_eps: f32,
) -> Result<(Tensor, Tensor)> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice};
    use std::ffi::c_void;

    if input.shape() != residual.shape() {
        candle_core::bail!(
            "cuda_rms_norm_residual_then_rms_norm input/residual shape mismatch: {:?} vs {:?}",
            input.shape(),
            residual.shape()
        );
    }
    if input.dtype() != residual.dtype()
        || input.dtype() != residual_weight.dtype()
        || input.dtype() != norm_weight.dtype()
    {
        candle_core::bail!(
            "cuda_rms_norm_residual_then_rms_norm dtype mismatch: input {:?}, residual {:?}, residual_weight {:?}, norm_weight {:?}",
            input.dtype(),
            residual.dtype(),
            residual_weight.dtype(),
            norm_weight.dtype()
        );
    }
    if !matches!(input.dtype(), DType::BF16 | DType::F16 | DType::F32) {
        candle_core::bail!(
            "cuda_rms_norm_residual_then_rms_norm only supports BF16/F16/F32, got {:?}",
            input.dtype()
        );
    }
    if !residual.device().same_device(input.device())
        || !residual_weight.device().same_device(input.device())
        || !norm_weight.device().same_device(input.device())
    {
        candle_core::bail!(
            "cuda_rms_norm_residual_then_rms_norm tensors must be on the same CUDA device"
        );
    }
    if let Some(scale) = scale {
        if scale.elem_count() != 1 {
            candle_core::bail!(
                "cuda_rms_norm_residual_then_rms_norm scale must have one element, got {}",
                scale.elem_count()
            );
        }
        if scale.dtype() != input.dtype() {
            candle_core::bail!(
                "cuda_rms_norm_residual_then_rms_norm scale dtype mismatch: input {:?}, scale {:?}",
                input.dtype(),
                scale.dtype()
            );
        }
        if !scale.device().same_device(input.device()) {
            candle_core::bail!(
                "cuda_rms_norm_residual_then_rms_norm scale must be on the same CUDA device"
            );
        }
    }

    let ncols = input.dim(D::Minus1)?;
    if residual_weight.dims1()? != ncols {
        candle_core::bail!(
            "cuda_rms_norm_residual_then_rms_norm residual weight size {} does not match last dim {ncols}",
            residual_weight.dims1()?
        );
    }
    if norm_weight.dims1()? != ncols {
        candle_core::bail!(
            "cuda_rms_norm_residual_then_rms_norm norm weight size {} does not match last dim {ncols}",
            norm_weight.dims1()?
        );
    }
    let elem_count = input.elem_count();
    if elem_count == 0 {
        candle_core::bail!("cuda_rms_norm_residual_then_rms_norm got empty input");
    }
    let nrows = elem_count / ncols;
    if nrows > i32::MAX as usize || ncols > i32::MAX as usize {
        candle_core::bail!(
            "cuda_rms_norm_residual_then_rms_norm input is too large: nrows={nrows}, ncols={ncols}"
        );
    }
    let nrows_i32 = i32::try_from(nrows).map_err(candle_core::Error::wrap)?;
    let ncols_i32 = i32::try_from(ncols).map_err(candle_core::Error::wrap)?;

    let input = input.contiguous()?;
    let residual = residual.contiguous()?;
    let residual_weight = residual_weight.contiguous()?;
    let norm_weight = norm_weight.contiguous()?;
    let scale = scale.map(Tensor::contiguous).transpose()?;

    let (input_storage, input_layout) = input.storage_and_layout();
    let input_storage = match &*input_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_rms_norm_residual_then_rms_norm requires CUDA input"),
    };
    let (residual_storage, residual_layout) = residual.storage_and_layout();
    let residual_storage = match &*residual_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_rms_norm_residual_then_rms_norm requires CUDA residual"),
    };
    let (residual_weight_storage, residual_weight_layout) = residual_weight.storage_and_layout();
    let residual_weight_storage = match &*residual_weight_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => {
            candle_core::bail!("cuda_rms_norm_residual_then_rms_norm requires CUDA residual weight")
        }
    };
    let (norm_weight_storage, norm_weight_layout) = norm_weight.storage_and_layout();
    let norm_weight_storage = match &*norm_weight_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_rms_norm_residual_then_rms_norm requires CUDA norm weight"),
    };
    let scale_storage_and_layout = scale.as_ref().map(|scale| scale.storage_and_layout());

    let dev = input_storage.device();
    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as i64;
    let shape = input.shape().clone();

    macro_rules! launch {
        ($variant:ident, $ty:ty, $ffi_fn:ident) => {{
            let CudaStorageSlice::$variant(src) = &input_storage.slice else {
                candle_core::bail!("cuda_rms_norm_residual_then_rms_norm input dtype mismatch");
            };
            let CudaStorageSlice::$variant(residual_src) = &residual_storage.slice else {
                candle_core::bail!("cuda_rms_norm_residual_then_rms_norm residual dtype mismatch");
            };
            let CudaStorageSlice::$variant(residual_weight_src) = &residual_weight_storage.slice
            else {
                candle_core::bail!(
                    "cuda_rms_norm_residual_then_rms_norm residual weight dtype mismatch"
                );
            };
            let CudaStorageSlice::$variant(norm_weight_src) = &norm_weight_storage.slice else {
                candle_core::bail!(
                    "cuda_rms_norm_residual_then_rms_norm norm weight dtype mismatch"
                );
            };
            let (scale_ptr, scale_guard) = if let Some((scale_storage, scale_layout)) =
                &scale_storage_and_layout
            {
                let scale_storage = match &**scale_storage {
                    candle_core::Storage::Cuda(s) => s,
                    _ => candle_core::bail!(
                        "cuda_rms_norm_residual_then_rms_norm requires CUDA scale"
                    ),
                };
                let CudaStorageSlice::$variant(scale_src) = &scale_storage.slice else {
                    candle_core::bail!("cuda_rms_norm_residual_then_rms_norm scale dtype mismatch");
                };
                let (scale_ptr, scale_guard) = scale_src.device_ptr(&stream);
                (
                    unsafe { (scale_ptr as *const $ty).add(scale_layout.start_offset()) }
                        as *const c_void,
                    Some(scale_guard),
                )
            } else {
                (std::ptr::null(), None)
            };

            let mut residual_out = unsafe { dev.alloc::<$ty>(elem_count) }?;
            let mut norm_out = unsafe { dev.alloc::<$ty>(elem_count) }?;
            let (src_ptr, src_guard) = src.device_ptr(&stream);
            let (residual_ptr, residual_guard) = residual_src.device_ptr(&stream);
            let (residual_weight_ptr, residual_weight_guard) =
                residual_weight_src.device_ptr(&stream);
            let (norm_weight_ptr, norm_weight_guard) = norm_weight_src.device_ptr(&stream);
            let (residual_out_ptr, residual_out_guard) = residual_out.device_ptr_mut(&stream);
            let (norm_out_ptr, norm_out_guard) = norm_out.device_ptr_mut(&stream);

            let src_ptr = unsafe { (src_ptr as *const $ty).add(input_layout.start_offset()) };
            let residual_ptr =
                unsafe { (residual_ptr as *const $ty).add(residual_layout.start_offset()) };
            let residual_weight_ptr = unsafe {
                (residual_weight_ptr as *const $ty).add(residual_weight_layout.start_offset())
            };
            let norm_weight_ptr =
                unsafe { (norm_weight_ptr as *const $ty).add(norm_weight_layout.start_offset()) };

            unsafe {
                ffi::$ffi_fn(
                    src_ptr as *const c_void,
                    residual_ptr as *const c_void,
                    residual_weight_ptr as *const c_void,
                    scale_ptr,
                    norm_weight_ptr as *const c_void,
                    residual_out_ptr as *mut c_void,
                    norm_out_ptr as *mut c_void,
                    nrows_i32,
                    ncols_i32,
                    residual_eps,
                    norm_eps,
                    stream_ptr,
                );
            }

            drop(src_guard);
            drop(residual_guard);
            drop(residual_weight_guard);
            drop(norm_weight_guard);
            drop(scale_guard);
            drop(residual_out_guard);
            drop(norm_out_guard);

            let residual_storage = CudaStorage {
                slice: CudaStorageSlice::$variant(residual_out),
                device: dev.clone(),
            };
            let norm_storage = CudaStorage {
                slice: CudaStorageSlice::$variant(norm_out),
                device: dev.clone(),
            };
            Ok((
                Tensor::from((candle_core::Storage::Cuda(residual_storage), shape.clone())),
                Tensor::from((candle_core::Storage::Cuda(norm_storage), shape)),
            ))
        }};
    }
    match input.dtype() {
        DType::BF16 => launch!(BF16, half::bf16, rms_norm_residual_then_rms_norm_bf16),
        DType::F16 => launch!(F16, half::f16, rms_norm_residual_then_rms_norm_f16),
        DType::F32 => launch!(F32, f32, rms_norm_residual_then_rms_norm_f32),
        dtype => {
            candle_core::bail!("cuda_rms_norm_residual_then_rms_norm unsupported dtype {dtype:?}")
        }
    }
}
