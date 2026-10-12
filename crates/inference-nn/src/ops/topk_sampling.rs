use super::*;

/// Logits dtype codes of the nucleus and normalizer kernels (`sort.cu`).
#[cfg(feature = "cuda")]
pub const CUDA_LOGITS_DTYPE_F16: i32 = 0;
#[cfg(feature = "cuda")]
pub const CUDA_LOGITS_DTYPE_BF16: i32 = 1;
#[cfg(feature = "cuda")]
pub const CUDA_LOGITS_DTYPE_F32: i32 = 2;

// (pointer, stream guard, dtype code) of CUDA F16 / BF16 / F32 logits; the caller has checked the dtype.
#[cfg(feature = "cuda")]
macro_rules! cuda_logits_ptr {
    ($storage:expr, $layout:expr, $stream:expr) => {{
        use inference_tensor::cuda_backend::CudaStorageSlice;
        use inference_tensor::cuda_backend::cudarc::driver::DevicePtr;
        macro_rules! at_offset {
            ($slice:expr, $ty:ty, $dtype:expr) => {{
                let (ptr, guard) = $slice.device_ptr($stream);
                let ptr = unsafe {
                    (ptr as *const $ty).add($layout.start_offset()) as *const std::ffi::c_void
                };
                (ptr, guard, $dtype)
            }};
        }
        match &$storage.slice {
            CudaStorageSlice::F16(slice) => at_offset!(slice, half::f16, CUDA_LOGITS_DTYPE_F16),
            CudaStorageSlice::BF16(slice) => at_offset!(slice, half::bf16, CUDA_LOGITS_DTYPE_BF16),
            CudaStorageSlice::F32(slice) => at_offset!(slice, f32, CUDA_LOGITS_DTYPE_F32),
            _ => unreachable!("logits dtype was checked to be F16, BF16 or F32"),
        }
    }};
}

#[cfg(feature = "cuda")]
pub struct CudaTopKSamplingWorkspace {
    capacity_rows: usize,
    capacity_k: usize,
    vocab: usize,
    location: inference_tensor::DeviceLocation,
    ranked: Option<CudaRankedTopKPackedWorkspace>,
    token_ring: CudaAsyncTokenRing,
    slots: Vec<CudaTopKSamplingSlot>,
}

#[cfg(feature = "cuda")]
struct CudaTopKSamplingSlot {
    params: inference_tensor::cuda_backend::cudarc::driver::CudaSlice<f32>,
    params_host: inference_tensor::cuda_backend::cudarc::driver::PinnedHostSlice<f32>,
}

#[cfg(feature = "cuda")]
pub struct CudaTopKSamplingSubmission {
    pub(super) token: CudaAsyncTokenSubmission,
}

#[cfg(feature = "cuda")]
impl CudaTopKSamplingSubmission {
    pub fn batch_size(&self) -> usize {
        self.token.batch_size()
    }

    pub fn wait(&self) -> Result<()> {
        self.token.wait()
    }
}

#[cfg(feature = "cuda")]
pub struct CudaTopKSamplingCompletion<'a> {
    token_ids: &'a [u32],
}

#[cfg(feature = "cuda")]
impl<'a> CudaTopKSamplingCompletion<'a> {
    pub fn token_ids(&self) -> &'a [u32] {
        self.token_ids
    }
}

#[cfg(feature = "cuda")]
fn new_cuda_topk_sampling_slot(
    dev: &inference_tensor::CudaDevice,
    capacity_rows: usize,
) -> Result<CudaTopKSamplingSlot> {
    let stream = dev.cuda_stream();
    let context = stream.context();
    let param_elems = capacity_rows
        .checked_mul(CUDA_TOPK_SAMPLING_PARAM_WIDTH)
        .ok_or_else(|| inference_tensor::Error::msg("CUDA top-k sampling parameter overflow"))?;
    let mut params_host = unsafe { context.alloc_pinned::<f32>(param_elems) }
        .map_err(inference_tensor::Error::wrap)?;
    params_host
        .as_mut_slice()
        .map_err(inference_tensor::Error::wrap)?
        .fill(0.0);

    Ok(CudaTopKSamplingSlot {
        params: unsafe { dev.alloc::<f32>(param_elems) }?,
        params_host,
    })
}

#[cfg(feature = "cuda")]
fn new_cuda_topk_sampling_workspace(
    dev: &inference_tensor::CudaDevice,
    rows: usize,
    vocab: usize,
    k: usize,
) -> Result<CudaTopKSamplingWorkspace> {
    use inference_tensor::backend::BackendDevice;

    let capacity_rows = rows
        .checked_next_power_of_two()
        .ok_or_else(|| inference_tensor::Error::msg("CUDA top-k sampling row capacity overflow"))?;
    let capacity_k = k.checked_next_power_of_two().ok_or_else(|| {
        inference_tensor::Error::msg("CUDA top-k sampling width capacity overflow")
    })?;
    let mut slots = Vec::with_capacity(CUDA_ASYNC_TOKEN_RING_SLOTS);
    for _ in 0..CUDA_ASYNC_TOKEN_RING_SLOTS {
        slots.push(new_cuda_topk_sampling_slot(dev, capacity_rows)?);
    }
    Ok(CudaTopKSamplingWorkspace {
        capacity_rows,
        capacity_k,
        vocab,
        location: dev.location(),
        ranked: None,
        token_ring: new_cuda_async_token_ring(dev, capacity_rows)?,
        slots,
    })
}

#[cfg(feature = "cuda")]
fn validate_cuda_topk_sampling_params(
    params: &[CudaTopKSamplingParams],
    rows: usize,
    vocab: usize,
    op: &'static str,
) -> Result<usize> {
    if params.len() != rows {
        inference_tensor::bail!("{op} expected {rows} sampling parameter rows");
    }
    let mut max_k = 0usize;
    for params in params {
        if !params.inverse_temperature.is_finite() || params.inverse_temperature <= 0.0 {
            inference_tensor::bail!("{op} requires positive finite inverse temperatures");
        }
        if params.top_k > CUDA_TOPK_MAX_K {
            inference_tensor::bail!("{op} top-k must be at most {CUDA_TOPK_MAX_K}");
        }
        if !params.top_p.is_finite() || !params.min_p.is_finite() {
            inference_tensor::bail!("{op} requires finite top-p and min-p values");
        }
        if !(0.0..1.0).contains(&params.uniform) {
            inference_tensor::bail!("{op} requires uniforms in [0, 1)");
        }
        let row_k = if params.is_nucleus() {
            cuda_topk_ranked_packed_max_k(vocab).ok_or_else(|| {
                inference_tensor::Error::msg(format!("{op} cannot rank a {vocab}-token vocabulary"))
            })?
        } else if params.top_k == 0 {
            inference_tensor::bail!("{op} top-k 0 needs a top-p or min-p filter");
        } else {
            params.top_k.min(vocab)
        };
        max_k = max_k.max(row_k);
    }
    Ok(max_k)
}

#[cfg(feature = "cuda")]
fn copy_cuda_topk_sampling_params(
    dev: &inference_tensor::CudaDevice,
    slot: &mut CudaTopKSamplingSlot,
    params: &[CudaTopKSamplingParams],
) -> Result<()> {
    let host = slot
        .params_host
        .as_mut_slice()
        .map_err(inference_tensor::Error::wrap)?;
    let (rows, _) = host.as_chunks_mut::<CUDA_TOPK_SAMPLING_PARAM_WIDTH>();
    for (row, params) in rows.iter_mut().zip(params) {
        *row = params.packed_row()?;
    }
    dev.memcpy_htod(&slot.params_host, &mut slot.params)?;
    Ok(())
}

#[cfg(feature = "cuda")]
fn cuda_topk_sampling_submit_inner(
    input: &Tensor,
    token_ids_dst: Option<&Tensor>,
    params: &[CudaTopKSamplingParams],
    cache: &mut Option<CudaTopKSamplingWorkspace>,
    op: &'static str,
) -> Result<CudaTopKSamplingSubmission> {
    use inference_tensor::backend::{BackendDevice, BackendStorage};
    use inference_tensor::cuda_backend::CudaStorageSlice;
    use inference_tensor::cuda_backend::cudarc::driver::DevicePtr;

    if !matches!(input.dtype(), DType::BF16 | DType::F16 | DType::F32) {
        inference_tensor::bail!("{op} requires BF16, F16, or F32 logits");
    }
    if !input.is_contiguous() {
        return Err(inference_tensor::Error::RequiresContiguous { op });
    }
    let [rows, vocab] = input.dims() else {
        inference_tensor::bail!("{op} requires logits with shape [batch, vocab]");
    };
    if *rows == 0 || *vocab == 0 {
        inference_tensor::bail!("{op} requires non-empty logits");
    }
    let max_k = validate_cuda_topk_sampling_params(params, *rows, *vocab, op)?;
    let (storage, _) = input.storage_and_layout();
    let storage = match &*storage {
        inference_tensor::Storage::Cuda(storage) => storage,
        _ => inference_tensor::bail!("{op} requires CUDA logits"),
    };
    let dev = storage.device();
    let needs_alloc = cache.as_ref().is_none_or(|workspace| {
        workspace.capacity_rows < *rows
            || workspace.capacity_k < max_k
            || workspace.vocab != *vocab
            || workspace.location != dev.location()
    });
    if needs_alloc {
        if cache
            .as_ref()
            .is_some_and(|workspace| workspace.token_ring.has_pending())
        {
            inference_tensor::bail!("{op} cannot resize while submissions are pending");
        }
        *cache = Some(new_cuda_topk_sampling_workspace(dev, *rows, *vocab, max_k)?);
    }

    let stream = dev.cuda_stream();
    let workspace = cache
        .as_mut()
        .expect("CUDA top-k sampling workspace was allocated above");
    let reservation = workspace
        .token_ring
        .reserve(input, token_ids_dst, *rows, op)?;
    let slot_index = reservation.slot;
    let result = (|| {
        let ranked =
            cuda_topk_ranked_packed_batched_with_workspace(input, max_k, &mut workspace.ranked)?;
        let slot = &mut workspace.slots[slot_index];
        copy_cuda_topk_sampling_params(dev, slot, params)?;

        let (packed_storage, packed_layout) = ranked.packed.storage_and_layout();
        let inference_tensor::Storage::Cuda(packed_storage) = &*packed_storage else {
            unreachable!("ranked top-k output is CUDA")
        };
        let CudaStorageSlice::F32(packed_slice) = &packed_storage.slice else {
            unreachable!("ranked top-k output is F32")
        };
        let (token_storage, token_layout) = reservation.device_tokens.storage_and_layout();
        let inference_tensor::Storage::Cuda(token_storage) = &*token_storage else {
            unreachable!("reserved token destination is CUDA")
        };
        let CudaStorageSlice::U32(token_slice) = &token_storage.slice else {
            unreachable!("reserved token destination is U32")
        };
        let (packed_ptr, packed_guard) = packed_slice.device_ptr(&stream);
        let (params_ptr, params_guard) = slot.params.device_ptr(&stream);
        let (tokens_ptr, tokens_guard) = token_slice.device_ptr(&stream);
        let packed_ptr = unsafe { (packed_ptr as *const f32).add(packed_layout.start_offset()) };
        let params_ptr = params_ptr as *const f32;
        let tokens_ptr = unsafe { (tokens_ptr as *mut u32).add(token_layout.start_offset()) };
        let rows_i32 = i32::try_from(*rows).map_err(inference_tensor::Error::wrap)?;
        let packed_k_i32 = i32::try_from(ranked.k).map_err(inference_tensor::Error::wrap)?;
        unsafe {
            ffi::sample_ranked_topk(
                packed_ptr,
                params_ptr,
                tokens_ptr,
                rows_i32,
                packed_k_i32,
                stream.cu_stream() as i64,
            );
        }
        if params.iter().any(CudaTopKSamplingParams::is_nucleus) {
            let (logits_storage, logits_layout) = input.storage_and_layout();
            let inference_tensor::Storage::Cuda(logits_storage) = &*logits_storage else {
                unreachable!("logits were checked to be CUDA above")
            };
            let (logits_ptr, logits_guard, dtype) =
                cuda_logits_ptr!(logits_storage, logits_layout, &stream);
            unsafe {
                ffi::sample_nucleus(
                    logits_ptr,
                    packed_ptr,
                    params_ptr,
                    tokens_ptr,
                    rows_i32,
                    i32::try_from(*vocab).map_err(inference_tensor::Error::wrap)?,
                    packed_k_i32,
                    dtype,
                    stream.cu_stream() as i64,
                );
            }
            drop(logits_guard);
        }
        drop(packed_guard);
        drop(params_guard);
        drop(tokens_guard);
        Result::<()>::Ok(())
    })();
    if let Err(error) = result {
        let _ = stream.synchronize();
        workspace.token_ring.abort(&reservation);
        return Err(error);
    }
    let token = match workspace.token_ring.finish_launch(reservation) {
        Ok(token) => token,
        Err(error) => {
            let _ = stream.synchronize();
            return Err(error);
        }
    };
    Ok(CudaTopKSamplingSubmission { token })
}

#[cfg(feature = "cuda")]
pub fn cuda_topk_sampling_submit_batched(
    input: &Tensor,
    params: &[CudaTopKSamplingParams],
    cache: &mut Option<CudaTopKSamplingWorkspace>,
) -> Result<CudaTopKSamplingSubmission> {
    cuda_topk_sampling_submit_inner(
        input,
        None,
        params,
        cache,
        "cuda_topk_sampling_submit_batched",
    )
}

#[cfg(feature = "cuda")]
pub fn cuda_topk_sampling_submit_batched_into(
    input: &Tensor,
    token_ids_dst: &Tensor,
    params: &[CudaTopKSamplingParams],
    cache: &mut Option<CudaTopKSamplingWorkspace>,
) -> Result<CudaTopKSamplingSubmission> {
    cuda_topk_sampling_submit_inner(
        input,
        Some(token_ids_dst),
        params,
        cache,
        "cuda_topk_sampling_submit_batched_into",
    )
}

#[cfg(feature = "cuda")]
pub fn cuda_topk_sampling_device_tokens_wait_on(
    workspace: &mut CudaTopKSamplingWorkspace,
    submission: &CudaTopKSamplingSubmission,
    consumer_stream: &Arc<inference_tensor::cuda_backend::cudarc::driver::CudaStream>,
) -> Result<()> {
    workspace.token_ring.wait_on(
        &submission.token,
        consumer_stream,
        "cuda_topk_sampling_device_tokens_wait_on",
    )
}

#[cfg(feature = "cuda")]
pub fn cuda_topk_sampling_device_tokens_release_after(
    workspace: &mut CudaTopKSamplingWorkspace,
    submission: &CudaTopKSamplingSubmission,
    consumer_stream: &Arc<inference_tensor::cuda_backend::cudarc::driver::CudaStream>,
) -> Result<()> {
    workspace.token_ring.release_after(
        &submission.token,
        consumer_stream,
        "cuda_topk_sampling_device_tokens_release_after",
    )
}

#[cfg(feature = "cuda")]
pub fn cuda_topk_sampling_submission_complete<'a>(
    workspace: &'a mut CudaTopKSamplingWorkspace,
    submission: &CudaTopKSamplingSubmission,
) -> Result<CudaTopKSamplingCompletion<'a>> {
    let token_ids = workspace
        .token_ring
        .complete(&submission.token, "cuda_topk_sampling_submission_complete")?;
    if token_ids.contains(&CUDA_TOP1_INVALID_TOKEN) {
        inference_tensor::bail!("invalid CUDA top-k sampling output");
    }
    Ok(CudaTopKSamplingCompletion { token_ids })
}

#[cfg(feature = "cuda")]
pub fn cuda_topk_sampling_submission_cancel(
    workspace: &mut CudaTopKSamplingWorkspace,
    submission: &CudaTopKSamplingSubmission,
) -> Result<()> {
    workspace
        .token_ring
        .cancel(&submission.token, "cuda_topk_sampling_submission_cancel")
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
pub struct CudaTopKSamplingParams {
    pub inverse_temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub uniform: f32,
}

#[cfg(feature = "cuda")]
impl CudaTopKSamplingParams {
    /// Top-p or min-p over the whole vocabulary: no top-k bound, sampled by the nucleus kernel.
    pub fn is_nucleus(&self) -> bool {
        self.top_k == 0
            && (crate::sampler::is_active_filter(self.top_p)
                || crate::sampler::is_active_filter(self.min_p))
    }

    fn packed_row(&self) -> Result<[f32; CUDA_TOPK_SAMPLING_PARAM_WIDTH]> {
        let top_k = u16::try_from(self.top_k).map_err(inference_tensor::Error::wrap)?;
        Ok([
            self.inverse_temperature,
            f32::from(top_k),
            self.top_p,
            self.min_p,
            self.uniform,
        ])
    }
}

/// Draft tokens sampled on the GPU with the sparse distribution (`ids`, `probs`, `[rows, k]`) each was drawn from.
#[cfg(feature = "cuda")]
pub struct CudaDraftSample {
    pub tokens: Tensor,
    pub ids: Tensor,
    pub probs: Tensor,
}

/// Flattens per-row sampling parameters into the `[rows, 5]` layout the ranked samplers read.
#[cfg(feature = "cuda")]
pub fn cuda_topk_sampling_param_rows(params: &[CudaTopKSamplingParams]) -> Result<Vec<f32>> {
    let mut rows = Vec::with_capacity(params.len() * CUDA_TOPK_SAMPLING_PARAM_WIDTH);
    for row in params {
        rows.extend(row.packed_row()?);
    }
    Ok(rows)
}

/// One draft token per `logits` row from its ranked top-k, with `params` from [`cuda_topk_sampling_param_rows`].
#[cfg(feature = "cuda")]
pub fn cuda_ranked_topk_draft_sample(
    logits: &Tensor,
    params: &Tensor,
    top_ks: &[usize],
) -> Result<CudaDraftSample> {
    use inference_tensor::backend::BackendStorage;
    use inference_tensor::cuda_backend::CudaStorageSlice;
    use inference_tensor::cuda_backend::cudarc::driver::DevicePtr;

    const OP: &str = "cuda_ranked_topk_draft_sample";
    let [rows, vocab] = *logits.dims() else {
        inference_tensor::bail!("{OP} requires logits with shape [rows, vocab]");
    };
    if params.dims() != [rows, CUDA_TOPK_SAMPLING_PARAM_WIDTH] || params.dtype() != DType::F32 {
        inference_tensor::bail!(
            "{OP} requires F32 parameters with shape [{rows}, {CUDA_TOPK_SAMPLING_PARAM_WIDTH}]"
        );
    }
    if top_ks.len() != rows {
        inference_tensor::bail!("{OP} expected {rows} top-k values");
    }
    let widest = cuda_topk_ranked_packed_max_k(vocab)
        .ok_or_else(|| inference_tensor::Error::msg(format!("{OP} cannot rank {vocab} tokens")))?;
    let packed_k = top_ks
        .iter()
        .map(|&k| if k == 0 { widest } else { k.min(widest) })
        .max()
        .unwrap_or(widest);
    let ranked = cuda_topk_ranked_packed_batched(&logits.contiguous()?, packed_k)?;
    let device = logits.device();
    let tokens = Tensor::zeros(rows, DType::U32, device)?;
    let ids = Tensor::zeros((rows, ranked.k), DType::U32, device)?;
    let probs = Tensor::zeros((rows, ranked.k), DType::F32, device)?;
    let params = params.contiguous()?;
    {
        let (packed_storage, packed_layout) = ranked.packed.storage_and_layout();
        let (params_storage, params_layout) = params.storage_and_layout();
        let (tokens_storage, _) = tokens.storage_and_layout();
        let (ids_storage, _) = ids.storage_and_layout();
        let (probs_storage, _) = probs.storage_and_layout();
        let (
            inference_tensor::Storage::Cuda(packed_storage),
            inference_tensor::Storage::Cuda(params_storage),
            inference_tensor::Storage::Cuda(tokens_storage),
            inference_tensor::Storage::Cuda(ids_storage),
            inference_tensor::Storage::Cuda(probs_storage),
        ) = (
            &*packed_storage,
            &*params_storage,
            &*tokens_storage,
            &*ids_storage,
            &*probs_storage,
        )
        else {
            inference_tensor::bail!("{OP} requires CUDA tensors");
        };
        let (
            CudaStorageSlice::F32(packed_slice),
            CudaStorageSlice::F32(params_slice),
            CudaStorageSlice::U32(tokens_slice),
            CudaStorageSlice::U32(ids_slice),
            CudaStorageSlice::F32(probs_slice),
        ) = (
            &packed_storage.slice,
            &params_storage.slice,
            &tokens_storage.slice,
            &ids_storage.slice,
            &probs_storage.slice,
        )
        else {
            unreachable!("draft sampling tensors were created with these dtypes")
        };
        let stream = packed_storage.device().cuda_stream();
        let (packed_ptr, _packed_guard) = packed_slice.device_ptr(&stream);
        let (params_ptr, _params_guard) = params_slice.device_ptr(&stream);
        let (tokens_ptr, _tokens_guard) = tokens_slice.device_ptr(&stream);
        let (ids_ptr, _ids_guard) = ids_slice.device_ptr(&stream);
        let (probs_ptr, _probs_guard) = probs_slice.device_ptr(&stream);
        unsafe {
            ffi::sample_ranked_topk_distribution(
                (packed_ptr as *const f32).add(packed_layout.start_offset()),
                (params_ptr as *const f32).add(params_layout.start_offset()),
                tokens_ptr as *mut u32,
                ids_ptr as *mut u32,
                probs_ptr as *mut f32,
                i32::try_from(rows).map_err(inference_tensor::Error::wrap)?,
                i32::try_from(ranked.k).map_err(inference_tensor::Error::wrap)?,
                stream.cu_stream() as i64,
            );
        }
    }
    Ok(CudaDraftSample { tokens, ids, probs })
}

/// Each ranked row's full-vocabulary normalizer, the sum of `exp((logit - max) * t)` with `t` the inverse temperature
/// of its sequence; `logits` is `[rows, vocab]` and `inverse_temperatures` `[sequences]`, rows grouped by sequence.
#[cfg(feature = "cuda")]
pub fn cuda_ranked_row_normalizers(
    logits: &Tensor,
    ranked: &RankedTopKPackedOutput,
    inverse_temperatures: &Tensor,
) -> Result<Tensor> {
    use inference_tensor::backend::BackendStorage;
    use inference_tensor::cuda_backend::CudaStorageSlice;
    use inference_tensor::cuda_backend::cudarc::driver::DevicePtr;

    const OP: &str = "cuda_ranked_row_normalizers";
    let [rows, vocab] = *logits.dims() else {
        inference_tensor::bail!("{OP} requires logits with shape [rows, vocab]");
    };
    let sequences = inverse_temperatures.dims1()?;
    if !matches!(logits.dtype(), DType::BF16 | DType::F16 | DType::F32)
        || !logits.is_contiguous()
        || inverse_temperatures.dtype() != DType::F32
        || sequences == 0
        || rows % sequences != 0
        || ranked.packed.dims() != [rows, 2 * ranked.k]
    {
        inference_tensor::bail!("{OP} got mismatched logits, ranked rows or inverse temperatures");
    }
    let normalizers = Tensor::zeros(rows, DType::F32, logits.device())?;
    let inverse_temperatures = inverse_temperatures.contiguous()?;
    {
        let (logits_storage, logits_layout) = logits.storage_and_layout();
        let (packed_storage, packed_layout) = ranked.packed.storage_and_layout();
        let (temperature_storage, temperature_layout) = inverse_temperatures.storage_and_layout();
        let (normalizer_storage, _) = normalizers.storage_and_layout();
        let (
            inference_tensor::Storage::Cuda(logits_storage),
            inference_tensor::Storage::Cuda(packed_storage),
            inference_tensor::Storage::Cuda(temperature_storage),
            inference_tensor::Storage::Cuda(normalizer_storage),
        ) = (
            &*logits_storage,
            &*packed_storage,
            &*temperature_storage,
            &*normalizer_storage,
        )
        else {
            inference_tensor::bail!("{OP} requires CUDA tensors");
        };
        let (
            CudaStorageSlice::F32(packed_slice),
            CudaStorageSlice::F32(temperature_slice),
            CudaStorageSlice::F32(normalizer_slice),
        ) = (
            &packed_storage.slice,
            &temperature_storage.slice,
            &normalizer_storage.slice,
        )
        else {
            inference_tensor::bail!("{OP} requires F32 ranked rows and temperatures");
        };
        let stream = logits_storage.device().cuda_stream();
        let (logits_ptr, _logits_guard, dtype) =
            cuda_logits_ptr!(logits_storage, logits_layout, &stream);
        let (packed_ptr, _packed_guard) = packed_slice.device_ptr(&stream);
        let (temperature_ptr, _temperature_guard) = temperature_slice.device_ptr(&stream);
        let (normalizer_ptr, _normalizer_guard) = normalizer_slice.device_ptr(&stream);
        unsafe {
            ffi::ranked_row_normalizers(
                logits_ptr,
                (packed_ptr as *const f32).add(packed_layout.start_offset()),
                (temperature_ptr as *const f32).add(temperature_layout.start_offset()),
                normalizer_ptr as *mut f32,
                i32::try_from(rows).map_err(inference_tensor::Error::wrap)?,
                i32::try_from(vocab).map_err(inference_tensor::Error::wrap)?,
                i32::try_from(ranked.k).map_err(inference_tensor::Error::wrap)?,
                i32::try_from(rows / sequences).map_err(inference_tensor::Error::wrap)?,
                dtype,
                stream.cu_stream() as i64,
            );
        }
    }
    Ok(normalizers)
}
