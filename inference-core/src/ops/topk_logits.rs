use super::*;

#[cfg(feature = "cuda")]
#[allow(dead_code)]
#[allow(clippy::cast_possible_truncation)]
pub fn cuda_topk_logits_f32(
    input: &Tensor,
    k: usize,
    temperature: f64,
) -> Result<TopKLogitsOutput> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::CudaStorageSlice;

    if temperature <= 0.0 || !temperature.is_finite() {
        candle_core::bail!("cuda_topk_logits_f32 requires a positive finite temperature");
    }

    let input = input.contiguous()?;
    if input.dtype() != DType::F32 {
        candle_core::bail!("cuda_topk_logits_f32 requires F32 logits");
    }

    let ncols = input.elem_count();
    if ncols == 0 {
        candle_core::bail!("cuda_topk_logits_f32 got empty logits");
    }
    let k = k.min(ncols);
    if k == 0 || k > CUDA_TOPK_MAX_K {
        candle_core::bail!(
            "cuda_topk_logits_f32 k={} must be in [1, {}]",
            k,
            CUDA_TOPK_MAX_K
        );
    }

    let nblocks = ncols.div_ceil(CUDA_TOPK_CHUNK_SIZE);
    let stage2_candidates = nblocks * k;
    if stage2_candidates > CUDA_TOPK_MAX_STAGE2_CANDIDATES {
        candle_core::bail!(
            "cuda_topk_logits_f32 workspace too large: {} candidates",
            stage2_candidates
        );
    }

    let (storage, layout) = input.storage_and_layout();
    let storage = match &*storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_topk_logits_f32 requires CUDA tensor"),
    };

    let dev = storage.device();
    let stream = dev.cuda_stream();
    let stream_raw = stream.cu_stream() as i64;

    let (src_ptr, _src_guard) = match &storage.slice {
        CudaStorageSlice::F32(inp) => inp.device_ptr(&stream),
        _ => candle_core::bail!("cuda_topk_logits_f32 only supports F32"),
    };
    let src_ptr = unsafe { (src_ptr as *const f32).add(layout.start_offset()) };

    let workspace_elems = nblocks * k;
    let mut block_values = unsafe { dev.alloc::<f32>(workspace_elems) }?;
    let mut block_indices = unsafe { dev.alloc::<u32>(workspace_elems) }?;
    let mut block_maxes = unsafe { dev.alloc::<f32>(nblocks) }?;
    let mut block_sums = unsafe { dev.alloc::<f32>(nblocks) }?;
    let mut values_dst = unsafe { dev.alloc::<f32>(k) }?;
    let mut indices_dst = unsafe { dev.alloc::<u32>(k) }?;
    let mut softmax_info_dst = unsafe { dev.alloc::<f32>(2) }?;

    let (block_values_ptr, block_values_guard) = block_values.device_ptr_mut(&stream);
    let (block_indices_ptr, block_indices_guard) = block_indices.device_ptr_mut(&stream);
    let (block_maxes_ptr, block_maxes_guard) = block_maxes.device_ptr_mut(&stream);
    let (block_sums_ptr, block_sums_guard) = block_sums.device_ptr_mut(&stream);
    let (values_ptr, values_guard) = values_dst.device_ptr_mut(&stream);
    let (indices_ptr, indices_guard) = indices_dst.device_ptr_mut(&stream);
    let (softmax_info_ptr, softmax_info_guard) = softmax_info_dst.device_ptr_mut(&stream);

    unsafe {
        ffi::topk_large_f32(
            src_ptr,
            block_values_ptr as *mut f32,
            block_indices_ptr as *mut u32,
            block_maxes_ptr as *mut f32,
            block_sums_ptr as *mut f32,
            values_ptr as *mut f32,
            indices_ptr as *mut u32,
            softmax_info_ptr as *mut f32,
            ncols as i32,
            k as i32,
            CUDA_TOPK_CHUNK_SIZE as i32,
            nblocks as i32,
            (1.0 / temperature) as f32,
            stream_raw,
        );
    }

    drop(block_values_guard);
    drop(block_indices_guard);
    drop(block_maxes_guard);
    drop(block_sums_guard);
    drop(values_guard);
    drop(indices_guard);
    drop(softmax_info_guard);

    let values_storage = candle_core::cuda_backend::CudaStorage {
        slice: CudaStorageSlice::F32(values_dst),
        device: dev.clone(),
    };
    let indices_storage = candle_core::cuda_backend::CudaStorage {
        slice: CudaStorageSlice::U32(indices_dst),
        device: dev.clone(),
    };
    let softmax_info_storage = candle_core::cuda_backend::CudaStorage {
        slice: CudaStorageSlice::F32(softmax_info_dst),
        device: dev.clone(),
    };
    let workspace = vec![
        Tensor::from((
            candle_core::Storage::Cuda(candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::F32(block_values),
                device: dev.clone(),
            }),
            Shape::from_dims(&[workspace_elems]),
        )),
        Tensor::from((
            candle_core::Storage::Cuda(candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::U32(block_indices),
                device: dev.clone(),
            }),
            Shape::from_dims(&[workspace_elems]),
        )),
        Tensor::from((
            candle_core::Storage::Cuda(candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::F32(block_maxes),
                device: dev.clone(),
            }),
            Shape::from_dims(&[nblocks]),
        )),
        Tensor::from((
            candle_core::Storage::Cuda(candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::F32(block_sums),
                device: dev.clone(),
            }),
            Shape::from_dims(&[nblocks]),
        )),
    ];

    Ok(TopKLogitsOutput {
        values: Tensor::from((
            candle_core::Storage::Cuda(values_storage),
            Shape::from_dims(&[k]),
        )),
        indices: Tensor::from((
            candle_core::Storage::Cuda(indices_storage),
            Shape::from_dims(&[k]),
        )),
        softmax_info: Tensor::from((
            candle_core::Storage::Cuda(softmax_info_storage),
            Shape::from_dims(&[2]),
        )),
        _workspace: workspace,
    })
}

#[cfg(feature = "cuda")]
#[allow(clippy::cast_possible_truncation)]
pub fn cuda_topk_logits_f32_packed(
    input: &Tensor,
    k: usize,
    temperature: f64,
) -> Result<TopKLogitsPackedOutput> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::CudaStorageSlice;

    if temperature <= 0.0 || !temperature.is_finite() {
        candle_core::bail!("cuda_topk_logits_f32_packed requires a positive finite temperature");
    }

    let input = input.contiguous()?;
    if input.dtype() != DType::F32 {
        candle_core::bail!("cuda_topk_logits_f32_packed requires F32 logits");
    }

    let ncols = input.elem_count();
    if ncols == 0 {
        candle_core::bail!("cuda_topk_logits_f32_packed got empty logits");
    }
    let k = k.min(ncols);
    if k == 0 || k > CUDA_TOPK_MAX_K {
        candle_core::bail!(
            "cuda_topk_logits_f32_packed k={} must be in [1, {}]",
            k,
            CUDA_TOPK_MAX_K
        );
    }

    let nblocks = ncols.div_ceil(CUDA_TOPK_CHUNK_SIZE);
    let stage2_candidates = nblocks * k;
    if stage2_candidates > CUDA_TOPK_MAX_STAGE2_CANDIDATES {
        candle_core::bail!(
            "cuda_topk_logits_f32_packed workspace too large: {} candidates",
            stage2_candidates
        );
    }

    let (storage, layout) = input.storage_and_layout();
    let storage = match &*storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_topk_logits_f32_packed requires CUDA tensor"),
    };

    let dev = storage.device();
    let stream = dev.cuda_stream();
    let stream_raw = stream.cu_stream() as i64;

    let (src_ptr, src_guard) = match &storage.slice {
        CudaStorageSlice::F32(inp) => inp.device_ptr(&stream),
        _ => candle_core::bail!("cuda_topk_logits_f32_packed only supports F32"),
    };
    let src_ptr = unsafe { (src_ptr as *const f32).add(layout.start_offset()) };

    let workspace_elems = nblocks * k;
    let mut block_values = unsafe { dev.alloc::<f32>(workspace_elems) }?;
    let mut block_indices = unsafe { dev.alloc::<u32>(workspace_elems) }?;
    let mut block_maxes = unsafe { dev.alloc::<f32>(nblocks) }?;
    let mut block_sums = unsafe { dev.alloc::<f32>(nblocks) }?;
    let mut packed_dst = unsafe { dev.alloc::<f32>(2 * k + 2) }?;

    let (block_values_ptr, block_values_guard) = block_values.device_ptr_mut(&stream);
    let (block_indices_ptr, block_indices_guard) = block_indices.device_ptr_mut(&stream);
    let (block_maxes_ptr, block_maxes_guard) = block_maxes.device_ptr_mut(&stream);
    let (block_sums_ptr, block_sums_guard) = block_sums.device_ptr_mut(&stream);
    let (packed_ptr, packed_guard) = packed_dst.device_ptr_mut(&stream);

    unsafe {
        ffi::topk_large_f32_packed(
            src_ptr,
            block_values_ptr as *mut f32,
            block_indices_ptr as *mut u32,
            block_maxes_ptr as *mut f32,
            block_sums_ptr as *mut f32,
            packed_ptr as *mut f32,
            ncols as i32,
            k as i32,
            CUDA_TOPK_CHUNK_SIZE as i32,
            nblocks as i32,
            (1.0 / temperature) as f32,
            stream_raw,
        );
    }

    drop(src_guard);
    drop(block_values_guard);
    drop(block_indices_guard);
    drop(block_maxes_guard);
    drop(block_sums_guard);
    drop(packed_guard);

    let packed_storage = candle_core::cuda_backend::CudaStorage {
        slice: CudaStorageSlice::F32(packed_dst),
        device: dev.clone(),
    };
    let workspace = vec![
        Tensor::from((
            candle_core::Storage::Cuda(candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::F32(block_values),
                device: dev.clone(),
            }),
            Shape::from_dims(&[workspace_elems]),
        )),
        Tensor::from((
            candle_core::Storage::Cuda(candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::U32(block_indices),
                device: dev.clone(),
            }),
            Shape::from_dims(&[workspace_elems]),
        )),
        Tensor::from((
            candle_core::Storage::Cuda(candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::F32(block_maxes),
                device: dev.clone(),
            }),
            Shape::from_dims(&[nblocks]),
        )),
        Tensor::from((
            candle_core::Storage::Cuda(candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::F32(block_sums),
                device: dev.clone(),
            }),
            Shape::from_dims(&[nblocks]),
        )),
    ];

    Ok(TopKLogitsPackedOutput {
        packed: Tensor::from((
            candle_core::Storage::Cuda(packed_storage),
            Shape::from_dims(&[2 * k + 2]),
        )),
        k,
        _workspace: workspace,
    })
}

#[cfg(feature = "cuda")]
pub(crate) struct CudaTopKLogitsPackedWorkspace {
    location: candle_core::DeviceLocation,
    stream: Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
    pub(super) capacity_rows: usize,
    vocab: usize,
    pub(super) capacity_k: usize,
    nblocks: usize,
    #[cfg(test)]
    pub(super) id: u64,
    block_values: Tensor,
    block_indices: Tensor,
    block_maxes: Tensor,
    block_sums: Tensor,
    packed: Tensor,
}

#[cfg(all(feature = "cuda", test))]
fn cuda_topk_logits_packed_workspace_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_ID: AtomicU64 = AtomicU64::new(1);
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

#[cfg(feature = "cuda")]
impl CudaTopKLogitsPackedWorkspace {
    fn new(
        dev: &candle_core::CudaDevice,
        rows: usize,
        vocab: usize,
        k: usize,
        nblocks: usize,
    ) -> Result<Self> {
        use candle_core::backend::BackendDevice;

        let capacity_rows = rows
            .checked_next_power_of_two()
            .ok_or_else(|| candle_core::Error::msg("CUDA top-k row capacity overflow"))?;
        let capacity_k = k
            .checked_next_power_of_two()
            .ok_or_else(|| candle_core::Error::msg("CUDA top-k width capacity overflow"))?;
        let workspace_elems = capacity_rows
            .checked_mul(nblocks)
            .and_then(|elems| elems.checked_mul(capacity_k))
            .ok_or_else(|| candle_core::Error::msg("CUDA top-k workspace overflow"))?;
        let block_elems = capacity_rows
            .checked_mul(nblocks)
            .ok_or_else(|| candle_core::Error::msg("CUDA top-k block workspace overflow"))?;
        let packed_width = capacity_k
            .checked_mul(2)
            .and_then(|width| width.checked_add(2))
            .ok_or_else(|| candle_core::Error::msg("CUDA top-k packed width overflow"))?;
        let packed_elems = capacity_rows
            .checked_mul(packed_width)
            .ok_or_else(|| candle_core::Error::msg("CUDA top-k packed workspace overflow"))?;
        let device = candle_core::Device::Cuda(dev.clone());
        Ok(Self {
            location: dev.location(),
            stream: dev.cuda_stream(),
            capacity_rows,
            vocab,
            capacity_k,
            nblocks,
            #[cfg(test)]
            id: cuda_topk_logits_packed_workspace_id(),
            block_values: Tensor::zeros(workspace_elems, DType::F32, &device)?,
            block_indices: Tensor::zeros(workspace_elems, DType::U32, &device)?,
            block_maxes: Tensor::zeros(block_elems, DType::F32, &device)?,
            block_sums: Tensor::zeros(block_elems, DType::F32, &device)?,
            packed: Tensor::zeros(packed_elems, DType::F32, &device)?,
        })
    }

    fn can_hold(
        &self,
        dev: &candle_core::CudaDevice,
        rows: usize,
        vocab: usize,
        k: usize,
        nblocks: usize,
    ) -> bool {
        use candle_core::backend::BackendDevice;

        let stream = dev.cuda_stream();
        self.location == dev.location()
            && Arc::ptr_eq(self.stream.context(), stream.context())
            && self.stream.cu_stream() == stream.cu_stream()
            && self.capacity_rows >= rows
            && self.vocab == vocab
            && self.capacity_k >= k
            && self.nblocks == nblocks
    }
}

#[cfg(feature = "cuda")]
pub(crate) struct CudaRankedTopKPackedWorkspace {
    location: candle_core::DeviceLocation,
    stream: Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
    capacity_rows: usize,
    vocab: usize,
    capacity_k: usize,
    nblocks: usize,
    block_values: Tensor,
    block_indices: Tensor,
    radix_state: Tensor,
    packed: Tensor,
}

#[cfg(feature = "cuda")]
impl CudaRankedTopKPackedWorkspace {
    fn new(
        dev: &candle_core::CudaDevice,
        rows: usize,
        vocab: usize,
        k: usize,
        nblocks: usize,
    ) -> Result<Self> {
        use candle_core::backend::BackendDevice;

        let capacity_rows = rows
            .checked_next_power_of_two()
            .ok_or_else(|| candle_core::Error::msg("CUDA ranked top-k row capacity overflow"))?;
        let capacity_k = k
            .checked_next_power_of_two()
            .ok_or_else(|| candle_core::Error::msg("CUDA ranked top-k width capacity overflow"))?;
        let candidate_elems = capacity_rows
            .checked_mul(nblocks)
            .and_then(|elems| elems.checked_mul(capacity_k))
            .ok_or_else(|| candle_core::Error::msg("CUDA ranked top-k workspace overflow"))?;
        let radix_state_elems = capacity_rows
            .checked_mul(unsafe { ffi::topk_large_ranked_state_words_per_row() })
            .ok_or_else(|| candle_core::Error::msg("CUDA ranked top-k radix overflow"))?;
        let index_elems = capacity_rows
            .checked_mul(nblocks)
            .and_then(|elems| elems.checked_mul(capacity_k))
            .ok_or_else(|| candle_core::Error::msg("CUDA ranked top-k index overflow"))?;
        let packed_elems = capacity_rows
            .checked_mul(capacity_k)
            .and_then(|elems| elems.checked_mul(2))
            .ok_or_else(|| candle_core::Error::msg("CUDA ranked top-k packed overflow"))?;
        let device = candle_core::Device::Cuda(dev.clone());
        Ok(Self {
            location: dev.location(),
            stream: dev.cuda_stream(),
            capacity_rows,
            vocab,
            capacity_k,
            nblocks,
            block_values: Tensor::zeros(candidate_elems, DType::F32, &device)?,
            block_indices: Tensor::zeros(index_elems, DType::U32, &device)?,
            radix_state: Tensor::zeros(radix_state_elems, DType::U32, &device)?,
            packed: Tensor::zeros(packed_elems, DType::F32, &device)?,
        })
    }

    fn can_hold(
        &self,
        dev: &candle_core::CudaDevice,
        rows: usize,
        vocab: usize,
        k: usize,
        nblocks: usize,
    ) -> bool {
        use candle_core::backend::BackendDevice;

        let stream = dev.cuda_stream();
        self.location == dev.location()
            && Arc::ptr_eq(self.stream.context(), stream.context())
            && self.stream.cu_stream() == stream.cu_stream()
            && self.capacity_rows >= rows
            && self.vocab == vocab
            && self.capacity_k >= k
            && self.nblocks == nblocks
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_topk_logits_packed_batched(
    input: &Tensor,
    k: usize,
    inverse_temperatures: &Tensor,
) -> Result<TopKLogitsPackedOutput> {
    let mut workspace = None;
    cuda_topk_logits_packed_batched_with_workspace(input, k, inverse_temperatures, &mut workspace)
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_topk_logits_packed_batched_with_workspace(
    input: &Tensor,
    k: usize,
    inverse_temperatures: &Tensor,
    cache: &mut Option<CudaTopKLogitsPackedWorkspace>,
) -> Result<TopKLogitsPackedOutput> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::cuda_backend::CudaStorageSlice;
    use std::ffi::c_void;

    const OP: &str = "cuda_topk_logits_packed_batched";

    if !matches!(input.dtype(), DType::BF16 | DType::F16 | DType::F32) {
        candle_core::bail!("{OP} requires BF16, F16, or F32 logits");
    }
    if inverse_temperatures.dtype() != DType::F32 {
        candle_core::bail!("{OP} requires F32 inverse temperatures");
    }
    if !input.is_contiguous() || !inverse_temperatures.is_contiguous() {
        return Err(candle_core::Error::RequiresContiguous { op: OP });
    }
    if !input.device().same_device(inverse_temperatures.device()) {
        candle_core::bail!("{OP} tensors must be on the same CUDA device");
    }

    let vocab =
        input.dims().last().copied().ok_or_else(|| {
            candle_core::Error::Msg(format!("{OP} requires logits with rank >= 1"))
        })?;
    if vocab == 0 {
        candle_core::bail!("{OP} got an empty vocabulary");
    }
    let batch = input.elem_count() / vocab;
    if batch == 0 {
        candle_core::bail!("{OP} got an empty batch");
    }
    if inverse_temperatures.dims() != [batch] {
        candle_core::bail!(
            "{OP} expected inverse temperatures with shape [{batch}], got {:?}",
            inverse_temperatures.dims()
        );
    }
    let k = k.min(vocab);
    if k == 0 || k > CUDA_TOPK_MAX_K {
        candle_core::bail!("{OP} k={k} must be in [1, {}]", CUDA_TOPK_MAX_K.min(vocab));
    }
    if vocab > CUDA_TOPK_MAX_EXACT_PACKED_VOCAB {
        candle_core::bail!(
            "{OP} vocabulary size {vocab} cannot be represented exactly by packed F32 indices"
        );
    }
    if vocab > i32::MAX as usize {
        candle_core::bail!("{OP} vocabulary is too large: {vocab}");
    }
    if batch > CUDA_TOPK_MAX_GRID_Y {
        candle_core::bail!("{OP} batch is too large for a 2D CUDA launch: {batch}");
    }

    let nblocks = vocab.div_ceil(CUDA_TOPK_CHUNK_SIZE);
    let candidates_per_row = nblocks
        .checked_mul(k)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} candidate count overflow")))?;
    if candidates_per_row > CUDA_TOPK_MAX_STAGE2_CANDIDATES {
        candle_core::bail!("{OP} workspace too large: {candidates_per_row} candidates per row");
    }
    let workspace_elems = batch
        .checked_mul(candidates_per_row)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} candidate workspace overflow")))?;
    let block_elems = batch
        .checked_mul(nblocks)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} block workspace overflow")))?;
    let packed_width = k
        .checked_mul(2)
        .and_then(|width| width.checked_add(2))
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} packed width overflow")))?;
    let packed_elems = batch
        .checked_mul(packed_width)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} packed output overflow")))?;

    let nrows_i32 = i32::try_from(batch).map_err(candle_core::Error::wrap)?;
    let ncols_i32 = i32::try_from(vocab).map_err(candle_core::Error::wrap)?;
    let k_i32 = i32::try_from(k).map_err(candle_core::Error::wrap)?;
    let chunk_size_i32 = i32::try_from(CUDA_TOPK_CHUNK_SIZE).map_err(candle_core::Error::wrap)?;
    let nblocks_i32 = i32::try_from(nblocks).map_err(candle_core::Error::wrap)?;

    let (input_storage, input_layout) = input.storage_and_layout();
    let input_storage = match &*input_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA logits"),
    };
    let (temperature_storage, temperature_layout) = inverse_temperatures.storage_and_layout();
    let temperature_storage = match &*temperature_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA inverse temperatures"),
    };
    let CudaStorageSlice::F32(temperature_slice) = &temperature_storage.slice else {
        candle_core::bail!("{OP} only supports F32 inverse temperatures");
    };

    let dev = input_storage.device();
    let stream = dev.cuda_stream();
    let needs_alloc = cache
        .as_ref()
        .is_none_or(|workspace| !workspace.can_hold(dev, batch, vocab, k, nblocks));
    if needs_alloc {
        *cache = Some(CudaTopKLogitsPackedWorkspace::new(
            dev, batch, vocab, k, nblocks,
        )?);
    }
    let workspace = cache
        .as_ref()
        .expect("CUDA top-k workspace was allocated above");
    let block_values = workspace.block_values.narrow(0, 0, workspace_elems)?;
    let block_indices = workspace.block_indices.narrow(0, 0, workspace_elems)?;
    let block_maxes = workspace.block_maxes.narrow(0, 0, block_elems)?;
    let block_sums = workspace.block_sums.narrow(0, 0, block_elems)?;
    let packed_dst = workspace.packed.narrow(0, 0, packed_elems)?;

    macro_rules! input_ptr {
        ($slice:expr, $ty:ty) => {{
            let (ptr, guard) = $slice.device_ptr(&stream);
            let ptr =
                unsafe { (ptr as *const $ty).add(input_layout.start_offset()) as *const c_void };
            (ptr, guard)
        }};
    }
    let (input_ptr, input_guard) = match &input_storage.slice {
        CudaStorageSlice::F32(slice) => input_ptr!(slice, f32),
        CudaStorageSlice::BF16(slice) => input_ptr!(slice, half::bf16),
        CudaStorageSlice::F16(slice) => input_ptr!(slice, half::f16),
        _ => candle_core::bail!("{OP} logits dtype mismatch"),
    };
    let (temperature_ptr, temperature_guard) = temperature_slice.device_ptr(&stream);
    let (block_values_storage_guard, block_values_layout) = block_values.storage_and_layout();
    let candle_core::Storage::Cuda(block_values_storage) = &*block_values_storage_guard else {
        unreachable!("CUDA top-k workspace values are CUDA")
    };
    let CudaStorageSlice::F32(block_values_slice) = &block_values_storage.slice else {
        unreachable!("CUDA top-k workspace values are F32")
    };
    let (block_values_ptr, block_values_guard) = block_values_slice.device_ptr(&stream);
    let block_values_ptr =
        unsafe { (block_values_ptr as *mut f32).add(block_values_layout.start_offset()) };
    let (block_indices_storage_guard, block_indices_layout) = block_indices.storage_and_layout();
    let candle_core::Storage::Cuda(block_indices_storage) = &*block_indices_storage_guard else {
        unreachable!("CUDA top-k workspace indices are CUDA")
    };
    let CudaStorageSlice::U32(block_indices_slice) = &block_indices_storage.slice else {
        unreachable!("CUDA top-k workspace indices are U32")
    };
    let (block_indices_ptr, block_indices_guard) = block_indices_slice.device_ptr(&stream);
    let block_indices_ptr =
        unsafe { (block_indices_ptr as *mut u32).add(block_indices_layout.start_offset()) };
    let (block_maxes_storage, block_maxes_layout) = block_maxes.storage_and_layout();
    let candle_core::Storage::Cuda(block_maxes_storage) = &*block_maxes_storage else {
        unreachable!("CUDA top-k workspace maxima are CUDA")
    };
    let CudaStorageSlice::F32(block_maxes_slice) = &block_maxes_storage.slice else {
        unreachable!("CUDA top-k workspace maxima are F32")
    };
    let (block_maxes_ptr, block_maxes_guard) = block_maxes_slice.device_ptr(&stream);
    let block_maxes_ptr =
        unsafe { (block_maxes_ptr as *mut f32).add(block_maxes_layout.start_offset()) };
    let (block_sums_storage, block_sums_layout) = block_sums.storage_and_layout();
    let candle_core::Storage::Cuda(block_sums_storage) = &*block_sums_storage else {
        unreachable!("CUDA top-k workspace sums are CUDA")
    };
    let CudaStorageSlice::F32(block_sums_slice) = &block_sums_storage.slice else {
        unreachable!("CUDA top-k workspace sums are F32")
    };
    let (block_sums_ptr, block_sums_guard) = block_sums_slice.device_ptr(&stream);
    let block_sums_ptr =
        unsafe { (block_sums_ptr as *mut f32).add(block_sums_layout.start_offset()) };
    let (packed_storage_guard, packed_layout) = packed_dst.storage_and_layout();
    let candle_core::Storage::Cuda(packed_storage) = &*packed_storage_guard else {
        unreachable!("CUDA top-k packed workspace is CUDA")
    };
    let CudaStorageSlice::F32(packed_slice) = &packed_storage.slice else {
        unreachable!("CUDA top-k packed workspace is F32")
    };
    let (packed_ptr, packed_guard) = packed_slice.device_ptr(&stream);
    let packed_ptr = unsafe { (packed_ptr as *mut f32).add(packed_layout.start_offset()) };
    let temperature_ptr =
        unsafe { (temperature_ptr as *const f32).add(temperature_layout.start_offset()) };

    macro_rules! launch {
        ($kernel:path, $input:expr) => {{
            unsafe {
                $kernel(
                    $input,
                    temperature_ptr,
                    block_values_ptr,
                    block_indices_ptr,
                    block_maxes_ptr,
                    block_sums_ptr,
                    packed_ptr,
                    nrows_i32,
                    ncols_i32,
                    k_i32,
                    chunk_size_i32,
                    nblocks_i32,
                    stream.cu_stream() as i64,
                );
            }
        }};
    }
    match input.dtype() {
        DType::F32 => launch!(ffi::topk_large_f32_packed_batched, input_ptr.cast::<f32>()),
        DType::BF16 => launch!(ffi::topk_large_bf16_packed_batched, input_ptr),
        DType::F16 => launch!(ffi::topk_large_f16_packed_batched, input_ptr),
        _ => unreachable!(),
    }

    drop(input_guard);
    drop(temperature_guard);
    drop(block_values_guard);
    drop(block_indices_guard);
    drop(block_maxes_guard);
    drop(block_sums_guard);
    drop(packed_guard);
    Ok(TopKLogitsPackedOutput {
        packed: packed_dst.reshape((batch, packed_width))?,
        k,
        _workspace: vec![
            block_values.clone(),
            block_indices.clone(),
            block_maxes.clone(),
            block_sums.clone(),
        ],
    })
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_topk_ranked_packed_batched(
    input: &Tensor,
    k: usize,
) -> Result<RankedTopKPackedOutput> {
    let mut workspace = None;
    cuda_topk_ranked_packed_batched_with_workspace(input, k, &mut workspace)
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_topk_ranked_packed_batched_with_workspace(
    input: &Tensor,
    k: usize,
    cache: &mut Option<CudaRankedTopKPackedWorkspace>,
) -> Result<RankedTopKPackedOutput> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::cuda_backend::CudaStorageSlice;
    use std::ffi::c_void;

    const OP: &str = "cuda_topk_ranked_packed_batched";

    if !matches!(input.dtype(), DType::BF16 | DType::F16 | DType::F32) {
        candle_core::bail!("{OP} requires BF16, F16, or F32 logits");
    }
    if !input.is_contiguous() {
        return Err(candle_core::Error::RequiresContiguous { op: OP });
    }
    let vocab =
        input.dims().last().copied().ok_or_else(|| {
            candle_core::Error::Msg(format!("{OP} requires logits with rank >= 1"))
        })?;
    if vocab == 0 {
        candle_core::bail!("{OP} got an empty vocabulary");
    }
    let batch = input.elem_count() / vocab;
    if batch == 0 {
        candle_core::bail!("{OP} got an empty batch");
    }
    let k = k.min(vocab);
    if k == 0 || k > CUDA_TOPK_MAX_K {
        candle_core::bail!("{OP} k={k} must be in [1, {}]", CUDA_TOPK_MAX_K.min(vocab));
    }
    if vocab > CUDA_TOPK_MAX_EXACT_PACKED_VOCAB {
        candle_core::bail!(
            "{OP} vocabulary size {vocab} cannot be represented exactly by packed F32 indices"
        );
    }
    if vocab > i32::MAX as usize {
        candle_core::bail!("{OP} vocabulary is too large: {vocab}");
    }
    if batch > CUDA_TOPK_MAX_GRID_Y {
        candle_core::bail!("{OP} batch is too large for a 2D CUDA launch: {batch}");
    }

    let nblocks = vocab.div_ceil(CUDA_TOPK_CHUNK_SIZE);
    let candidates_per_row = nblocks
        .checked_mul(k)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} candidate count overflow")))?;
    if candidates_per_row > CUDA_TOPK_MAX_STAGE2_CANDIDATES {
        candle_core::bail!("{OP} workspace too large: {candidates_per_row} candidates per row");
    }
    let workspace_elems = batch
        .checked_mul(candidates_per_row)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} candidate workspace overflow")))?;
    let radix_state_words_per_row = unsafe { ffi::topk_large_ranked_state_words_per_row() };
    let radix_state_elems = batch
        .checked_mul(radix_state_words_per_row)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} radix workspace overflow")))?;
    let packed_width = k
        .checked_mul(2)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} packed width overflow")))?;
    let packed_elems = batch
        .checked_mul(packed_width)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} packed output overflow")))?;

    let nrows_i32 = i32::try_from(batch).map_err(candle_core::Error::wrap)?;
    let ncols_i32 = i32::try_from(vocab).map_err(candle_core::Error::wrap)?;
    let k_i32 = i32::try_from(k).map_err(candle_core::Error::wrap)?;
    let chunk_size_i32 = i32::try_from(CUDA_TOPK_CHUNK_SIZE).map_err(candle_core::Error::wrap)?;
    let nblocks_i32 = i32::try_from(nblocks).map_err(candle_core::Error::wrap)?;

    let (input_storage, input_layout) = input.storage_and_layout();
    let input_storage = match &*input_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA logits"),
    };
    let dev = input_storage.device();
    let stream = dev.cuda_stream();
    let needs_alloc = cache
        .as_ref()
        .is_none_or(|workspace| !workspace.can_hold(dev, batch, vocab, k, nblocks));
    if needs_alloc {
        *cache = Some(CudaRankedTopKPackedWorkspace::new(
            dev, batch, vocab, k, nblocks,
        )?);
    }
    let workspace = cache
        .as_ref()
        .expect("CUDA ranked top-k workspace was allocated above");
    let block_values = workspace.block_values.narrow(0, 0, workspace_elems)?;
    let block_indices = workspace.block_indices.narrow(0, 0, workspace_elems)?;
    let radix_state = workspace.radix_state.narrow(0, 0, radix_state_elems)?;
    let packed_dst = workspace.packed.narrow(0, 0, packed_elems)?;

    macro_rules! input_ptr {
        ($slice:expr, $ty:ty) => {{
            let (ptr, guard) = $slice.device_ptr(&stream);
            let ptr =
                unsafe { (ptr as *const $ty).add(input_layout.start_offset()) as *const c_void };
            (ptr, guard)
        }};
    }
    let (input_ptr, input_guard) = match &input_storage.slice {
        CudaStorageSlice::F32(slice) => input_ptr!(slice, f32),
        CudaStorageSlice::BF16(slice) => input_ptr!(slice, half::bf16),
        CudaStorageSlice::F16(slice) => input_ptr!(slice, half::f16),
        _ => candle_core::bail!("{OP} logits dtype mismatch"),
    };
    let (block_values_storage_guard, block_values_layout) = block_values.storage_and_layout();
    let candle_core::Storage::Cuda(block_values_storage) = &*block_values_storage_guard else {
        unreachable!("CUDA ranked top-k workspace values are CUDA")
    };
    let CudaStorageSlice::F32(block_values_slice) = &block_values_storage.slice else {
        unreachable!("CUDA ranked top-k workspace values are F32")
    };
    let (block_values_ptr, block_values_guard) = block_values_slice.device_ptr(&stream);
    let block_values_ptr =
        unsafe { (block_values_ptr as *mut f32).add(block_values_layout.start_offset()) };
    let (block_indices_storage_guard, block_indices_layout) = block_indices.storage_and_layout();
    let candle_core::Storage::Cuda(block_indices_storage) = &*block_indices_storage_guard else {
        unreachable!("CUDA ranked top-k workspace indices are CUDA")
    };
    let CudaStorageSlice::U32(block_indices_slice) = &block_indices_storage.slice else {
        unreachable!("CUDA ranked top-k workspace indices are U32")
    };
    let (block_indices_ptr, block_indices_guard) = block_indices_slice.device_ptr(&stream);
    let block_indices_ptr =
        unsafe { (block_indices_ptr as *mut u32).add(block_indices_layout.start_offset()) };
    let (packed_storage_guard, packed_layout) = packed_dst.storage_and_layout();
    let candle_core::Storage::Cuda(packed_storage) = &*packed_storage_guard else {
        unreachable!("CUDA ranked top-k packed workspace is CUDA")
    };
    let CudaStorageSlice::F32(packed_slice) = &packed_storage.slice else {
        unreachable!("CUDA ranked top-k packed workspace is F32")
    };
    let (packed_ptr, packed_guard) = packed_slice.device_ptr(&stream);
    let packed_ptr = unsafe { (packed_ptr as *mut f32).add(packed_layout.start_offset()) };
    let (radix_storage_guard, radix_layout) = radix_state.storage_and_layout();
    let candle_core::Storage::Cuda(radix_storage) = &*radix_storage_guard else {
        unreachable!("CUDA ranked top-k radix workspace is CUDA")
    };
    let CudaStorageSlice::U32(radix_slice) = &radix_storage.slice else {
        unreachable!("CUDA ranked top-k radix workspace is U32")
    };
    let (radix_ptr, radix_guard) = radix_slice.device_ptr(&stream);
    let radix_ptr = unsafe { (radix_ptr as *mut u32).add(radix_layout.start_offset()) };

    macro_rules! launch {
        ($kernel:path, $input:expr) => {{
            unsafe {
                $kernel(
                    $input,
                    block_values_ptr,
                    block_indices_ptr,
                    packed_ptr,
                    radix_ptr.cast::<c_void>(),
                    nrows_i32,
                    ncols_i32,
                    k_i32,
                    chunk_size_i32,
                    nblocks_i32,
                    stream.cu_stream() as i64,
                )
            }
        }};
    }
    let status = match input.dtype() {
        DType::F32 => launch!(
            ffi::topk_large_ranked_f32_packed_batched,
            input_ptr.cast::<f32>()
        ),
        DType::BF16 => launch!(ffi::topk_large_ranked_bf16_packed_batched, input_ptr),
        DType::F16 => launch!(ffi::topk_large_ranked_f16_packed_batched, input_ptr),
        _ => unreachable!(),
    };

    drop(input_guard);
    drop(block_values_guard);
    drop(block_indices_guard);
    drop(packed_guard);
    drop(radix_guard);
    drop(block_values_storage_guard);
    drop(block_indices_storage_guard);
    drop(packed_storage_guard);
    drop(radix_storage_guard);
    if status != 0 {
        candle_core::bail!("{OP} CUDA launch failed with status {status}");
    }

    Ok(RankedTopKPackedOutput {
        packed: packed_dst.reshape((batch, packed_width))?,
        k,
        _workspace: vec![block_values, block_indices, radix_state],
    })
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_topk_logits_f32_packed_batched(
    input: &Tensor,
    k: usize,
    inverse_temperatures: &Tensor,
) -> Result<TopKLogitsPackedOutput> {
    if input.dtype() != DType::F32 {
        candle_core::bail!("cuda_topk_logits_f32_packed_batched requires F32 logits");
    }
    cuda_topk_logits_packed_batched(input, k, inverse_temperatures)
}

#[cfg(feature = "metal")]
#[allow(clippy::cast_possible_truncation)]
pub fn metal_topk_logits_packed(
    input: &Tensor,
    k: usize,
    temperature: f64,
) -> Result<TopKLogitsPackedOutput> {
    use candle_core::{backend::BackendStorage, MetalStorage, Shape, Storage};

    const MAX_K: usize = 128;
    const CHUNK_SIZE: usize = 2048;

    if temperature <= 0.0 || !temperature.is_finite() {
        candle_core::bail!("metal_topk_logits_packed requires a positive finite temperature");
    }
    let input = input.contiguous()?;
    if !matches!(input.dtype(), DType::F32 | DType::F16 | DType::BF16) {
        candle_core::bail!("metal_topk_logits_packed requires F32/F16/BF16 logits");
    }
    let dtype = input.dtype();
    let ncols = input.elem_count();
    if ncols == 0 {
        candle_core::bail!("metal_topk_logits_packed got empty logits");
    }
    let k = k.min(ncols);
    if k == 0 || k > MAX_K {
        candle_core::bail!("metal_topk_logits_packed k={k} must be in [1, {MAX_K}]");
    }
    let nblocks = ncols.div_ceil(CHUNK_SIZE);

    let (input_s, input_l) = input.storage_and_layout();
    let Storage::Metal(input_s) = &*input_s else {
        candle_core::bail!("metal_topk_logits_packed requires Metal tensor");
    };
    let device = input_s.device().clone();

    let block_values_buf = device.new_buffer(nblocks * k, DType::F32, "topk-block-values")?;
    let block_indices_buf = device.new_buffer(nblocks * k, DType::U32, "topk-block-indices")?;
    let block_maxes_buf = device.new_buffer(nblocks, DType::F32, "topk-block-maxes")?;
    let block_sums_buf = device.new_buffer(nblocks, DType::F32, "topk-block-sums")?;
    let packed_buf = device.new_buffer(2 * k + 2, DType::F32, "topk-packed")?;

    let encoder = device.command_encoder()?;
    encoder.set_label("topk-logits-packed");

    let inv_temp = (1.0_f64 / temperature) as f32;
    let input_offset = input_l.start_offset() * input.dtype().size_in_bytes();

    inference_quant::metal_kernels::call_topk_logits_packed(
        device.device(),
        &encoder,
        &inference_quant::metal_kernels::Kernels::new(),
        dtype,
        input_s.buffer(),
        &block_values_buf,
        &block_indices_buf,
        &block_maxes_buf,
        &block_sums_buf,
        &packed_buf,
        ncols,
        k,
        CHUNK_SIZE,
        inv_temp,
    )
    .map_err(|e| candle_core::Error::Msg(format!("metal_topk_logits_packed kernel error: {e}")))?;
    let _ = (
        input_offset,
        &block_values_buf,
        &block_indices_buf,
        &block_maxes_buf,
        &block_sums_buf,
    );

    let packed = Tensor::from((
        Storage::Metal(MetalStorage::new(
            packed_buf,
            device.clone(),
            2 * k + 2,
            DType::F32,
        )),
        Shape::from(vec![2 * k + 2]),
    ));
    Ok(TopKLogitsPackedOutput {
        packed,
        k,
        _workspace: vec![],
    })
}
