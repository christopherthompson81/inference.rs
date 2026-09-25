use super::*;

#[cfg(feature = "cuda")]
pub fn cuda_apply_sparse_penalties_f32(
    input: &Tensor,
    token_ids: &Tensor,
    counts: &Tensor,
    frequency_penalty: f32,
    presence_penalty: f32,
    repetition_penalty: f32,
) -> Result<Tensor> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice};
    use std::ffi::c_void;

    if input.dtype() != DType::F32 {
        candle_core::bail!("cuda_apply_sparse_penalties_f32 requires F32 logits");
    }
    if token_ids.dtype() != DType::U32 {
        candle_core::bail!("cuda_apply_sparse_penalties_f32 requires U32 token ids");
    }
    if counts.dtype() != DType::F32 {
        candle_core::bail!("cuda_apply_sparse_penalties_f32 requires F32 counts");
    }
    if token_ids.elem_count() != counts.elem_count() {
        candle_core::bail!(
            "cuda_apply_sparse_penalties_f32 token ids/counts length mismatch: {} vs {}",
            token_ids.elem_count(),
            counts.elem_count()
        );
    }
    if !token_ids.device().same_device(input.device())
        || !counts.device().same_device(input.device())
    {
        candle_core::bail!(
            "cuda_apply_sparse_penalties_f32 tensors must be on the same CUDA device"
        );
    }

    let input = input.contiguous()?;
    let token_ids = token_ids.contiguous()?;
    let counts = counts.contiguous()?;

    let elem_count = input.elem_count();
    let n_tokens = token_ids.elem_count();
    if elem_count == 0 {
        candle_core::bail!("cuda_apply_sparse_penalties_f32 got empty logits");
    }
    if elem_count > i32::MAX as usize {
        candle_core::bail!(
            "cuda_apply_sparse_penalties_f32 input is too large: {elem_count} elements"
        );
    }
    if n_tokens > i32::MAX as usize {
        candle_core::bail!(
            "cuda_apply_sparse_penalties_f32 token list is too large: {n_tokens} elements"
        );
    }
    let elem_count_i32 = i32::try_from(elem_count).map_err(candle_core::Error::wrap)?;
    let n_tokens_i32 = i32::try_from(n_tokens).map_err(candle_core::Error::wrap)?;

    let (input_storage, input_layout) = input.storage_and_layout();
    let input_storage = match &*input_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_apply_sparse_penalties_f32 requires CUDA logits"),
    };
    let CudaStorageSlice::F32(src) = &input_storage.slice else {
        candle_core::bail!("cuda_apply_sparse_penalties_f32 only supports F32 logits");
    };

    let (token_storage, token_layout) = token_ids.storage_and_layout();
    let token_storage = match &*token_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_apply_sparse_penalties_f32 requires CUDA token ids"),
    };
    let CudaStorageSlice::U32(token_src) = &token_storage.slice else {
        candle_core::bail!("cuda_apply_sparse_penalties_f32 only supports U32 token ids");
    };

    let (count_storage, count_layout) = counts.storage_and_layout();
    let count_storage = match &*count_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_apply_sparse_penalties_f32 requires CUDA counts"),
    };
    let CudaStorageSlice::F32(count_src) = &count_storage.slice else {
        candle_core::bail!("cuda_apply_sparse_penalties_f32 only supports F32 counts");
    };

    let dev = input_storage.device();
    let stream = dev.cuda_stream();
    let mut out = unsafe { dev.alloc::<f32>(elem_count) }?;

    let (src_ptr, src_guard) = src.device_ptr(&stream);
    let (token_ptr, token_guard) = token_src.device_ptr(&stream);
    let (count_ptr, count_guard) = count_src.device_ptr(&stream);
    let (out_ptr, out_guard) = out.device_ptr_mut(&stream);

    let src_ptr = unsafe { (src_ptr as *const f32).add(input_layout.start_offset()) };
    let token_ptr = unsafe { (token_ptr as *const u32).add(token_layout.start_offset()) };
    let count_ptr = unsafe { (count_ptr as *const f32).add(count_layout.start_offset()) };

    unsafe {
        ffi::apply_sparse_penalties_f32(
            src_ptr as *const c_void,
            out_ptr as *mut c_void,
            token_ptr,
            count_ptr,
            elem_count_i32,
            n_tokens_i32,
            frequency_penalty,
            presence_penalty,
            repetition_penalty,
            stream.cu_stream() as i64,
        );
    }

    drop(src_guard);
    drop(token_guard);
    drop(count_guard);
    drop(out_guard);

    let out_storage = CudaStorage {
        slice: CudaStorageSlice::F32(out),
        device: dev.clone(),
    };
    Ok(Tensor::from((
        candle_core::Storage::Cuda(out_storage),
        input.shape().clone(),
    )))
}

#[cfg(feature = "cuda")]
pub fn cuda_apply_sparse_logits_bias_f32(
    input: &Tensor,
    token_ids: &Tensor,
    biases: &Tensor,
) -> Result<Tensor> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice};
    use std::ffi::c_void;

    if input.dtype() != DType::F32 {
        candle_core::bail!("cuda_apply_sparse_logits_bias_f32 requires F32 logits");
    }
    if token_ids.dtype() != DType::U32 {
        candle_core::bail!("cuda_apply_sparse_logits_bias_f32 requires U32 token ids");
    }
    if biases.dtype() != DType::F32 {
        candle_core::bail!("cuda_apply_sparse_logits_bias_f32 requires F32 biases");
    }
    if token_ids.elem_count() != biases.elem_count() {
        candle_core::bail!(
            "cuda_apply_sparse_logits_bias_f32 token ids/biases length mismatch: {} vs {}",
            token_ids.elem_count(),
            biases.elem_count()
        );
    }
    if !token_ids.device().same_device(input.device())
        || !biases.device().same_device(input.device())
    {
        candle_core::bail!(
            "cuda_apply_sparse_logits_bias_f32 tensors must be on the same CUDA device"
        );
    }

    let input = input.contiguous()?;
    let token_ids = token_ids.contiguous()?;
    let biases = biases.contiguous()?;

    let elem_count = input.elem_count();
    let n_tokens = token_ids.elem_count();
    if elem_count == 0 {
        candle_core::bail!("cuda_apply_sparse_logits_bias_f32 got empty logits");
    }
    if elem_count > i32::MAX as usize {
        candle_core::bail!(
            "cuda_apply_sparse_logits_bias_f32 input is too large: {elem_count} elements"
        );
    }
    if n_tokens > i32::MAX as usize {
        candle_core::bail!(
            "cuda_apply_sparse_logits_bias_f32 token list is too large: {n_tokens} elements"
        );
    }
    let elem_count_i32 = i32::try_from(elem_count).map_err(candle_core::Error::wrap)?;
    let n_tokens_i32 = i32::try_from(n_tokens).map_err(candle_core::Error::wrap)?;

    let (input_storage, input_layout) = input.storage_and_layout();
    let input_storage = match &*input_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_apply_sparse_logits_bias_f32 requires CUDA logits"),
    };
    let CudaStorageSlice::F32(src) = &input_storage.slice else {
        candle_core::bail!("cuda_apply_sparse_logits_bias_f32 only supports F32 logits");
    };

    let (token_storage, token_layout) = token_ids.storage_and_layout();
    let token_storage = match &*token_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_apply_sparse_logits_bias_f32 requires CUDA token ids"),
    };
    let CudaStorageSlice::U32(token_src) = &token_storage.slice else {
        candle_core::bail!("cuda_apply_sparse_logits_bias_f32 only supports U32 token ids");
    };

    let (bias_storage, bias_layout) = biases.storage_and_layout();
    let bias_storage = match &*bias_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_apply_sparse_logits_bias_f32 requires CUDA biases"),
    };
    let CudaStorageSlice::F32(bias_src) = &bias_storage.slice else {
        candle_core::bail!("cuda_apply_sparse_logits_bias_f32 only supports F32 biases");
    };

    let dev = input_storage.device();
    let stream = dev.cuda_stream();
    let mut out = unsafe { dev.alloc::<f32>(elem_count) }?;

    let (src_ptr, src_guard) = src.device_ptr(&stream);
    let (token_ptr, token_guard) = token_src.device_ptr(&stream);
    let (bias_ptr, bias_guard) = bias_src.device_ptr(&stream);
    let (out_ptr, out_guard) = out.device_ptr_mut(&stream);

    let src_ptr = unsafe { (src_ptr as *const f32).add(input_layout.start_offset()) };
    let token_ptr = unsafe { (token_ptr as *const u32).add(token_layout.start_offset()) };
    let bias_ptr = unsafe { (bias_ptr as *const f32).add(bias_layout.start_offset()) };

    unsafe {
        ffi::apply_sparse_logits_bias_f32(
            src_ptr as *const c_void,
            out_ptr as *mut c_void,
            token_ptr,
            bias_ptr,
            elem_count_i32,
            n_tokens_i32,
            stream.cu_stream() as i64,
        );
    }

    drop(src_guard);
    drop(token_guard);
    drop(bias_guard);
    drop(out_guard);

    let out_storage = CudaStorage {
        slice: CudaStorageSlice::F32(out),
        device: dev.clone(),
    };
    Ok(Tensor::from((
        candle_core::Storage::Cuda(out_storage),
        input.shape().clone(),
    )))
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_apply_causal_mask_f32(
    scores: &Tensor,
    q_offset: usize,
    prefix_len: usize,
) -> Result<()> {
    struct CausalMaskF32 {
        q_offset: usize,
        prefix_len: usize,
    }

    impl candle_core::InplaceOp1 for CausalMaskF32 {
        fn name(&self) -> &'static str {
            "causal-mask-f32"
        }

        fn cpu_fwd(
            &self,
            _storage: &mut candle_core::CpuStorage,
            _layout: &candle_core::Layout,
        ) -> Result<()> {
            candle_core::bail!("causal-mask-f32 requires CUDA storage")
        }

        fn cuda_fwd(
            &self,
            storage: &mut candle_core::CudaStorage,
            layout: &candle_core::Layout,
        ) -> Result<()> {
            use candle_core::backend::BackendStorage;
            use candle_core::cuda_backend::cudarc::driver::DevicePtrMut;
            use candle_core::cuda_backend::CudaStorageSlice;
            use std::ffi::c_void;

            let (batch_heads, q_len, kv_len) = layout.shape().dims3()?;
            let batch_heads = i32::try_from(batch_heads).map_err(candle_core::Error::wrap)?;
            let q_len = i32::try_from(q_len).map_err(candle_core::Error::wrap)?;
            let kv_len = i32::try_from(kv_len).map_err(candle_core::Error::wrap)?;
            let q_offset = i32::try_from(self.q_offset).map_err(candle_core::Error::wrap)?;
            let prefix_len = i32::try_from(self.prefix_len).map_err(candle_core::Error::wrap)?;
            if !layout.is_contiguous() {
                candle_core::bail!("causal-mask-f32 requires contiguous scores")
            }
            let dev = storage.device();
            let stream = dev.cuda_stream();
            let CudaStorageSlice::F32(scores) = &mut storage.slice else {
                candle_core::bail!("causal-mask-f32 requires F32 scores")
            };
            let (scores_ptr, scores_guard) = scores.device_ptr_mut(&stream);
            let scores_ptr =
                unsafe { (scores_ptr as *mut f32).add(layout.start_offset()) as *mut c_void };
            unsafe {
                ffi::apply_causal_mask_f32(
                    scores_ptr,
                    batch_heads,
                    q_len,
                    kv_len,
                    q_offset,
                    prefix_len,
                    stream.cu_stream() as i64,
                );
            }
            drop(scores_guard);
            Ok(())
        }
    }

    scores.inplace_op1(&CausalMaskF32 {
        q_offset,
        prefix_len,
    })
}

#[cfg(feature = "metal")]
pub fn metal_apply_sparse_penalties(
    input: &Tensor,
    token_ids: &Tensor,
    counts: &Tensor,
    frequency_penalty: f32,
    presence_penalty: f32,
    repetition_penalty: f32,
) -> Result<Tensor> {
    use candle_core::{backend::BackendStorage, MetalStorage, Shape, Storage};

    if !matches!(input.dtype(), DType::F32 | DType::F16 | DType::BF16) {
        candle_core::bail!("metal_apply_sparse_penalties requires F32/F16/BF16 logits");
    }
    if token_ids.dtype() != DType::U32 || counts.dtype() != DType::F32 {
        candle_core::bail!("metal_apply_sparse_penalties token_ids must be u32, counts f32");
    }
    let dtype = input.dtype();
    let n = input.elem_count();
    let n_tokens = token_ids.elem_count();
    if counts.elem_count() != n_tokens {
        candle_core::bail!("token_ids and counts length mismatch");
    }

    let input = input.contiguous()?;
    let token_ids = token_ids.contiguous()?;
    let counts = counts.contiguous()?;

    let (input_s, input_l) = input.storage_and_layout();
    let (tok_s, tok_l) = token_ids.storage_and_layout();
    let (cnt_s, cnt_l) = counts.storage_and_layout();
    let (Storage::Metal(input_s), Storage::Metal(tok_s), Storage::Metal(cnt_s)) =
        (&*input_s, &*tok_s, &*cnt_s)
    else {
        candle_core::bail!("metal_apply_sparse_penalties requires Metal tensors");
    };
    let device = input_s.device().clone();

    let out_buf = device.new_buffer(n, dtype, "penalties-out")?;
    let encoder = device.command_encoder()?;
    encoder.set_label("penalties-copy");
    {
        use inference_quant::metal_kernels::Kernels;
        inference_quant::metal_kernels::call_copy_logits(
            device.device(),
            &encoder,
            &Kernels::new(),
            dtype,
            input_s.buffer(),
            input_l.start_offset() * input.dtype().size_in_bytes(),
            &out_buf,
            n,
        )
        .map_err(|e| candle_core::Error::Msg(format!("metal copy: {e}")))?;
    }
    encoder.set_label("penalties-apply");
    inference_quant::metal_kernels::call_apply_sparse_penalties(
        device.device(),
        &encoder,
        &inference_quant::metal_kernels::Kernels::new(),
        dtype,
        &out_buf,
        tok_s.buffer(),
        cnt_s.buffer(),
        n,
        n_tokens,
        frequency_penalty,
        presence_penalty,
        repetition_penalty,
    )
    .map_err(|e| candle_core::Error::Msg(format!("metal penalties: {e}")))?;
    let _ = (tok_l, cnt_l);
    Ok(Tensor::from((
        Storage::Metal(MetalStorage::new(out_buf, device.clone(), n, dtype)),
        Shape::from(input.dims()),
    )))
}
