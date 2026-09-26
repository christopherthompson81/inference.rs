use super::*;

#[cfg(feature = "cuda")]
pub fn cuda_dflash_greedy_select(
    topk: &RankedTopKPackedOutput,
    projected_hidden: &Tensor,
    predecessor_codebook: &Tensor,
    successor_codebook: &Tensor,
    anchors: &Tensor,
) -> Result<Tensor> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::CudaStorageSlice;
    use std::ffi::c_void;

    const OP: &str = "cuda_dflash_greedy_select";
    let packed_topk = &topk.packed;
    let k = topk.k;

    let [rows, packed_width] = packed_topk.dims() else {
        candle_core::bail!("{OP} expected packed top-k with shape [batch * positions, 2 * k]");
    };
    let [hidden_rows, rank] = projected_hidden.dims() else {
        candle_core::bail!(
            "{OP} expected projected hidden states with shape [batch * positions, rank]"
        );
    };
    let [predecessor_vocab, predecessor_rank] = predecessor_codebook.dims() else {
        candle_core::bail!("{OP} expected predecessor codebook with shape [vocab, rank]");
    };
    let [successor_vocab, successor_rank] = successor_codebook.dims() else {
        candle_core::bail!("{OP} expected successor codebook with shape [vocab, rank]");
    };
    let [batch] = anchors.dims() else {
        candle_core::bail!("{OP} expected anchors with shape [batch]");
    };
    let (rows, packed_width, hidden_rows, rank) = (*rows, *packed_width, *hidden_rows, *rank);
    let (predecessor_vocab, predecessor_rank) = (*predecessor_vocab, *predecessor_rank);
    let (successor_vocab, successor_rank, batch) = (*successor_vocab, *successor_rank, *batch);

    if rows == 0 || batch == 0 || rank == 0 || predecessor_vocab == 0 {
        candle_core::bail!("{OP} does not support empty inputs");
    }
    if rows % batch != 0 {
        candle_core::bail!("{OP} row count {rows} is not divisible by batch size {batch}");
    }
    if hidden_rows != rows {
        candle_core::bail!("{OP} expected {rows} projected hidden rows, got {hidden_rows}");
    }
    if predecessor_vocab != successor_vocab || predecessor_rank != rank || successor_rank != rank {
        candle_core::bail!(
            "{OP} codebook shapes {:?} and {:?} do not match hidden rank {rank}",
            predecessor_codebook.dims(),
            successor_codebook.dims()
        );
    }
    let expected_packed_width = k
        .checked_mul(2)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} packed width overflow")))?;
    if packed_width != expected_packed_width {
        candle_core::bail!(
            "{OP} expected rank-only packed top-k width {expected_packed_width}, got {packed_width}"
        );
    }
    if k == 0 || k > CUDA_DFLASH_SELECTOR_MAX_K {
        candle_core::bail!("{OP} k={k} must be in [1, {CUDA_DFLASH_SELECTOR_MAX_K}]");
    }
    if predecessor_vocab > CUDA_TOPK_MAX_EXACT_PACKED_VOCAB {
        candle_core::bail!(
            "{OP} vocabulary size {predecessor_vocab} cannot be represented exactly by packed F32 indices"
        );
    }
    if packed_topk.dtype() != DType::F32 {
        candle_core::bail!("{OP} requires F32 packed top-k values");
    }
    if anchors.dtype() != DType::U32 {
        candle_core::bail!("{OP} requires U32 anchors");
    }
    for (name, tensor) in [
        ("projected hidden states", projected_hidden),
        ("predecessor codebook", predecessor_codebook),
        ("successor codebook", successor_codebook),
    ] {
        if !matches!(tensor.dtype(), DType::BF16 | DType::F32) {
            candle_core::bail!("{OP} requires BF16 or F32 {name}");
        }
    }
    for tensor in [
        packed_topk,
        projected_hidden,
        predecessor_codebook,
        successor_codebook,
        anchors,
    ] {
        if !tensor.is_contiguous() {
            return Err(candle_core::Error::RequiresContiguous { op: OP });
        }
        if !packed_topk.device().same_device(tensor.device()) {
            candle_core::bail!("{OP} tensors must be on the same CUDA device");
        }
    }
    let positions = rows / batch;
    let batch_i32 = i32::try_from(batch).map_err(candle_core::Error::wrap)?;
    let positions_i32 = i32::try_from(positions).map_err(candle_core::Error::wrap)?;
    let rank_i32 = i32::try_from(rank).map_err(candle_core::Error::wrap)?;
    let vocab_i32 = i32::try_from(predecessor_vocab).map_err(candle_core::Error::wrap)?;
    let k_i32 = i32::try_from(k).map_err(candle_core::Error::wrap)?;
    let packed_width_i32 = i32::try_from(packed_width).map_err(candle_core::Error::wrap)?;

    let (packed_storage, packed_layout) = packed_topk.storage_and_layout();
    let packed_storage = match &*packed_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA tensors"),
    };
    let (hidden_storage, hidden_layout) = projected_hidden.storage_and_layout();
    let hidden_storage = match &*hidden_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA tensors"),
    };
    let (predecessor_storage, predecessor_layout) = predecessor_codebook.storage_and_layout();
    let predecessor_storage = match &*predecessor_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA tensors"),
    };
    let (successor_storage, successor_layout) = successor_codebook.storage_and_layout();
    let successor_storage = match &*successor_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA tensors"),
    };
    let (anchor_storage, anchor_layout) = anchors.storage_and_layout();
    let anchor_storage = match &*anchor_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA tensors"),
    };

    let dev = packed_storage.device();
    let stream = dev.cuda_stream();
    let CudaStorageSlice::F32(packed_slice) = &packed_storage.slice else {
        candle_core::bail!("{OP} packed top-k dtype mismatch");
    };
    let CudaStorageSlice::U32(anchor_slice) = &anchor_storage.slice else {
        candle_core::bail!("{OP} anchor dtype mismatch");
    };
    let (packed_ptr, packed_guard) = packed_slice.device_ptr(&stream);
    let packed_ptr = unsafe { (packed_ptr as *const f32).add(packed_layout.start_offset()) };
    let (anchor_ptr, anchor_guard) = anchor_slice.device_ptr(&stream);
    let anchor_ptr = unsafe { (anchor_ptr as *const u32).add(anchor_layout.start_offset()) };

    macro_rules! data_ptr {
        ($storage:expr, $layout:expr, $name:expr) => {{
            match &$storage.slice {
                CudaStorageSlice::F32(slice) => {
                    let (ptr, guard) = slice.device_ptr(&stream);
                    let ptr =
                        unsafe { (ptr as *const f32).add($layout.start_offset()) as *const c_void };
                    (ptr, CUDA_DFLASH_SELECTOR_F32, guard)
                }
                CudaStorageSlice::BF16(slice) => {
                    let (ptr, guard) = slice.device_ptr(&stream);
                    let ptr = unsafe {
                        (ptr as *const half::bf16).add($layout.start_offset()) as *const c_void
                    };
                    (ptr, CUDA_DFLASH_SELECTOR_BF16, guard)
                }
                _ => candle_core::bail!("{OP} {} dtype mismatch", $name),
            }
        }};
    }

    let (hidden_ptr, hidden_dtype, hidden_guard) =
        data_ptr!(hidden_storage, hidden_layout, "projected hidden states");
    let (predecessor_ptr, predecessor_dtype, predecessor_guard) = data_ptr!(
        predecessor_storage,
        predecessor_layout,
        "predecessor codebook"
    );
    let (successor_ptr, successor_dtype, successor_guard) =
        data_ptr!(successor_storage, successor_layout, "successor codebook");

    let mut selected = unsafe { dev.alloc::<u32>(rows) }?;
    let (selected_ptr, selected_guard) = selected.device_ptr_mut(&stream);
    unsafe {
        ffi::dflash_greedy_select(
            packed_ptr,
            hidden_ptr,
            predecessor_ptr,
            successor_ptr,
            anchor_ptr,
            selected_ptr as *mut u32,
            batch_i32,
            positions_i32,
            rank_i32,
            vocab_i32,
            k_i32,
            packed_width_i32,
            hidden_dtype,
            predecessor_dtype,
            successor_dtype,
            stream.cu_stream() as i64,
        );
    }

    drop(packed_guard);
    drop(hidden_guard);
    drop(predecessor_guard);
    drop(successor_guard);
    drop(anchor_guard);
    drop(selected_guard);

    Ok(Tensor::from((
        candle_core::Storage::Cuda(candle_core::cuda_backend::CudaStorage {
            slice: CudaStorageSlice::U32(selected),
            device: dev.clone(),
        }),
        Shape::from_dims(&[batch, positions]),
    )))
}

#[cfg(feature = "cuda")]
pub fn cuda_dflash_sample_select(
    input: DFlashSelectorSampleInput<'_>,
) -> Result<DFlashSelectorSampleOutput> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::CudaStorageSlice;
    use std::ffi::c_void;

    const OP: &str = "cuda_dflash_sample_select";
    let DFlashSelectorSampleInput {
        topk,
        projected_hidden,
        predecessor_codebook,
        successor_codebook,
        anchors,
        inverse_temperatures,
        uniforms,
    } = input;
    let packed_topk = &topk.packed;
    let k = topk.k;

    let [rows, packed_width] = packed_topk.dims() else {
        candle_core::bail!("{OP} expected packed top-k with shape [batch * positions, 2 * k]");
    };
    let [hidden_rows, rank] = projected_hidden.dims() else {
        candle_core::bail!(
            "{OP} expected projected hidden states with shape [batch * positions, rank]"
        );
    };
    let [predecessor_vocab, predecessor_rank] = predecessor_codebook.dims() else {
        candle_core::bail!("{OP} expected predecessor codebook with shape [vocab, rank]");
    };
    let [successor_vocab, successor_rank] = successor_codebook.dims() else {
        candle_core::bail!("{OP} expected successor codebook with shape [vocab, rank]");
    };
    let [batch] = anchors.dims() else {
        candle_core::bail!("{OP} expected anchors with shape [batch]");
    };
    let (rows, packed_width, hidden_rows, rank) = (*rows, *packed_width, *hidden_rows, *rank);
    let (predecessor_vocab, predecessor_rank) = (*predecessor_vocab, *predecessor_rank);
    let (successor_vocab, successor_rank, batch) = (*successor_vocab, *successor_rank, *batch);

    if rows == 0 || batch == 0 || rank == 0 || predecessor_vocab == 0 {
        candle_core::bail!("{OP} does not support empty inputs");
    }
    if rows % batch != 0 {
        candle_core::bail!("{OP} row count {rows} is not divisible by batch size {batch}");
    }
    if hidden_rows != rows {
        candle_core::bail!("{OP} expected {rows} projected hidden rows, got {hidden_rows}");
    }
    if predecessor_vocab != successor_vocab || predecessor_rank != rank || successor_rank != rank {
        candle_core::bail!(
            "{OP} codebook shapes {:?} and {:?} do not match hidden rank {rank}",
            predecessor_codebook.dims(),
            successor_codebook.dims()
        );
    }
    let expected_packed_width = k
        .checked_mul(2)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} packed width overflow")))?;
    if packed_width != expected_packed_width {
        candle_core::bail!(
            "{OP} expected rank-only packed top-k width {expected_packed_width}, got {packed_width}"
        );
    }
    if k == 0 || k > CUDA_DFLASH_SELECTOR_MAX_K {
        candle_core::bail!("{OP} k={k} must be in [1, {CUDA_DFLASH_SELECTOR_MAX_K}]");
    }
    if predecessor_vocab > CUDA_TOPK_MAX_EXACT_PACKED_VOCAB {
        candle_core::bail!(
            "{OP} vocabulary size {predecessor_vocab} cannot be represented exactly by packed F32 indices"
        );
    }
    let positions = rows / batch;
    if inverse_temperatures.dims() != [batch] {
        candle_core::bail!(
            "{OP} expected inverse temperatures with shape [{batch}], got {:?}",
            inverse_temperatures.dims()
        );
    }
    if uniforms.dims() != [batch, positions] {
        candle_core::bail!(
            "{OP} expected uniforms with shape [{batch}, {positions}], got {:?}",
            uniforms.dims()
        );
    }
    if packed_topk.dtype() != DType::F32
        || inverse_temperatures.dtype() != DType::F32
        || uniforms.dtype() != DType::F32
    {
        candle_core::bail!("{OP} requires F32 packed top-k, inverse temperatures, and uniforms");
    }
    if anchors.dtype() != DType::U32 {
        candle_core::bail!("{OP} requires U32 anchors");
    }
    for (name, tensor) in [
        ("projected hidden states", projected_hidden),
        ("predecessor codebook", predecessor_codebook),
        ("successor codebook", successor_codebook),
    ] {
        if !matches!(tensor.dtype(), DType::BF16 | DType::F32) {
            candle_core::bail!("{OP} requires BF16 or F32 {name}");
        }
    }
    for tensor in [
        packed_topk,
        projected_hidden,
        predecessor_codebook,
        successor_codebook,
        anchors,
        inverse_temperatures,
        uniforms,
    ] {
        if !tensor.is_contiguous() {
            return Err(candle_core::Error::RequiresContiguous { op: OP });
        }
        if !packed_topk.device().same_device(tensor.device()) {
            candle_core::bail!("{OP} tensors must be on the same CUDA device");
        }
    }

    let sparse_elems = rows
        .checked_mul(k)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} output overflow")))?;
    let batch_i32 = i32::try_from(batch).map_err(candle_core::Error::wrap)?;
    let positions_i32 = i32::try_from(positions).map_err(candle_core::Error::wrap)?;
    let rank_i32 = i32::try_from(rank).map_err(candle_core::Error::wrap)?;
    let vocab_i32 = i32::try_from(predecessor_vocab).map_err(candle_core::Error::wrap)?;
    let k_i32 = i32::try_from(k).map_err(candle_core::Error::wrap)?;
    let packed_width_i32 = i32::try_from(packed_width).map_err(candle_core::Error::wrap)?;

    let (packed_storage, packed_layout) = packed_topk.storage_and_layout();
    let packed_storage = match &*packed_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA tensors"),
    };
    let (hidden_storage, hidden_layout) = projected_hidden.storage_and_layout();
    let hidden_storage = match &*hidden_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA tensors"),
    };
    let (predecessor_storage, predecessor_layout) = predecessor_codebook.storage_and_layout();
    let predecessor_storage = match &*predecessor_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA tensors"),
    };
    let (successor_storage, successor_layout) = successor_codebook.storage_and_layout();
    let successor_storage = match &*successor_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA tensors"),
    };
    let (anchor_storage, anchor_layout) = anchors.storage_and_layout();
    let anchor_storage = match &*anchor_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA tensors"),
    };
    let (temperature_storage, temperature_layout) = inverse_temperatures.storage_and_layout();
    let temperature_storage = match &*temperature_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA tensors"),
    };
    let (uniform_storage, uniform_layout) = uniforms.storage_and_layout();
    let uniform_storage = match &*uniform_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA tensors"),
    };

    let dev = packed_storage.device();
    let stream = dev.cuda_stream();
    let CudaStorageSlice::F32(packed_slice) = &packed_storage.slice else {
        candle_core::bail!("{OP} packed top-k dtype mismatch");
    };
    let CudaStorageSlice::U32(anchor_slice) = &anchor_storage.slice else {
        candle_core::bail!("{OP} anchor dtype mismatch");
    };
    let CudaStorageSlice::F32(temperature_slice) = &temperature_storage.slice else {
        candle_core::bail!("{OP} inverse temperature dtype mismatch");
    };
    let CudaStorageSlice::F32(uniform_slice) = &uniform_storage.slice else {
        candle_core::bail!("{OP} uniform dtype mismatch");
    };
    let (packed_ptr, packed_guard) = packed_slice.device_ptr(&stream);
    let packed_ptr = unsafe { (packed_ptr as *const f32).add(packed_layout.start_offset()) };
    let (anchor_ptr, anchor_guard) = anchor_slice.device_ptr(&stream);
    let anchor_ptr = unsafe { (anchor_ptr as *const u32).add(anchor_layout.start_offset()) };
    let (temperature_ptr, temperature_guard) = temperature_slice.device_ptr(&stream);
    let temperature_ptr =
        unsafe { (temperature_ptr as *const f32).add(temperature_layout.start_offset()) };
    let (uniform_ptr, uniform_guard) = uniform_slice.device_ptr(&stream);
    let uniform_ptr = unsafe { (uniform_ptr as *const f32).add(uniform_layout.start_offset()) };

    macro_rules! data_ptr {
        ($storage:expr, $layout:expr, $name:expr) => {{
            match &$storage.slice {
                CudaStorageSlice::F32(slice) => {
                    let (ptr, guard) = slice.device_ptr(&stream);
                    let ptr =
                        unsafe { (ptr as *const f32).add($layout.start_offset()) as *const c_void };
                    (ptr, CUDA_DFLASH_SELECTOR_F32, guard)
                }
                CudaStorageSlice::BF16(slice) => {
                    let (ptr, guard) = slice.device_ptr(&stream);
                    let ptr = unsafe {
                        (ptr as *const half::bf16).add($layout.start_offset()) as *const c_void
                    };
                    (ptr, CUDA_DFLASH_SELECTOR_BF16, guard)
                }
                _ => candle_core::bail!("{OP} {} dtype mismatch", $name),
            }
        }};
    }

    let (hidden_ptr, hidden_dtype, hidden_guard) =
        data_ptr!(hidden_storage, hidden_layout, "projected hidden states");
    let (predecessor_ptr, predecessor_dtype, predecessor_guard) = data_ptr!(
        predecessor_storage,
        predecessor_layout,
        "predecessor codebook"
    );
    let (successor_ptr, successor_dtype, successor_guard) =
        data_ptr!(successor_storage, successor_layout, "successor codebook");

    let mut selected = unsafe { dev.alloc::<u32>(rows) }?;
    let mut candidate_ids = unsafe { dev.alloc::<u32>(sparse_elems) }?;
    let mut candidate_probs = unsafe { dev.alloc::<f32>(sparse_elems) }?;
    let (selected_ptr, selected_guard) = selected.device_ptr_mut(&stream);
    let (candidate_ids_ptr, candidate_ids_guard) = candidate_ids.device_ptr_mut(&stream);
    let (candidate_probs_ptr, candidate_probs_guard) = candidate_probs.device_ptr_mut(&stream);
    unsafe {
        ffi::dflash_sample_select(
            packed_ptr,
            hidden_ptr,
            predecessor_ptr,
            successor_ptr,
            anchor_ptr,
            temperature_ptr,
            uniform_ptr,
            selected_ptr as *mut u32,
            candidate_ids_ptr as *mut u32,
            candidate_probs_ptr as *mut f32,
            batch_i32,
            positions_i32,
            rank_i32,
            vocab_i32,
            k_i32,
            packed_width_i32,
            hidden_dtype,
            predecessor_dtype,
            successor_dtype,
            stream.cu_stream() as i64,
        );
    }

    drop(packed_guard);
    drop(hidden_guard);
    drop(predecessor_guard);
    drop(successor_guard);
    drop(anchor_guard);
    drop(temperature_guard);
    drop(uniform_guard);
    drop(selected_guard);
    drop(candidate_ids_guard);
    drop(candidate_probs_guard);

    let tokens = Tensor::from((
        candle_core::Storage::Cuda(candle_core::cuda_backend::CudaStorage {
            slice: CudaStorageSlice::U32(selected),
            device: dev.clone(),
        }),
        Shape::from_dims(&[batch, positions]),
    ));
    let candidate_ids = Tensor::from((
        candle_core::Storage::Cuda(candle_core::cuda_backend::CudaStorage {
            slice: CudaStorageSlice::U32(candidate_ids),
            device: dev.clone(),
        }),
        Shape::from_dims(&[batch, positions, k]),
    ));
    let candidate_probs = Tensor::from((
        candle_core::Storage::Cuda(candle_core::cuda_backend::CudaStorage {
            slice: CudaStorageSlice::F32(candidate_probs),
            device: dev.clone(),
        }),
        Shape::from_dims(&[batch, positions, k]),
    ));

    Ok(DFlashSelectorSampleOutput {
        tokens,
        candidate_ids,
        candidate_probs,
    })
}

#[cfg(feature = "cuda")]
pub struct DFlashSelectorSampleInput<'a> {
    pub topk: &'a RankedTopKPackedOutput,
    pub projected_hidden: &'a Tensor,
    pub predecessor_codebook: &'a Tensor,
    pub successor_codebook: &'a Tensor,
    pub anchors: &'a Tensor,
    pub inverse_temperatures: &'a Tensor,
    pub uniforms: &'a Tensor,
}

#[cfg(feature = "cuda")]
pub struct DFlashSelectorSampleOutput {
    pub tokens: Tensor,
    pub candidate_ids: Tensor,
    pub candidate_probs: Tensor,
}
