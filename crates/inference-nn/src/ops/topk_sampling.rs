use super::*;

#[cfg(feature = "cuda")]
pub struct CudaTopKSamplingWorkspace {
    capacity_rows: usize,
    capacity_k: usize,
    vocab: usize,
    location: candle_core::DeviceLocation,
    ranked: Option<CudaRankedTopKPackedWorkspace>,
    token_ring: CudaAsyncTokenRing,
    slots: Vec<CudaTopKSamplingSlot>,
}

#[cfg(feature = "cuda")]
struct CudaTopKSamplingSlot {
    params: candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>,
    params_host: candle_core::cuda_backend::cudarc::driver::PinnedHostSlice<f32>,
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
    dev: &candle_core::CudaDevice,
    capacity_rows: usize,
) -> Result<CudaTopKSamplingSlot> {
    let stream = dev.cuda_stream();
    let context = stream.context();
    let param_elems = capacity_rows
        .checked_mul(CUDA_TOPK_SAMPLING_PARAM_WIDTH)
        .ok_or_else(|| candle_core::Error::msg("CUDA top-k sampling parameter overflow"))?;
    let mut params_host =
        unsafe { context.alloc_pinned::<f32>(param_elems) }.map_err(candle_core::Error::wrap)?;
    params_host
        .as_mut_slice()
        .map_err(candle_core::Error::wrap)?
        .fill(0.0);

    Ok(CudaTopKSamplingSlot {
        params: unsafe { dev.alloc::<f32>(param_elems) }?,
        params_host,
    })
}

#[cfg(feature = "cuda")]
fn new_cuda_topk_sampling_workspace(
    dev: &candle_core::CudaDevice,
    rows: usize,
    vocab: usize,
    k: usize,
) -> Result<CudaTopKSamplingWorkspace> {
    use candle_core::backend::BackendDevice;

    let capacity_rows = rows
        .checked_next_power_of_two()
        .ok_or_else(|| candle_core::Error::msg("CUDA top-k sampling row capacity overflow"))?;
    let capacity_k = k
        .checked_next_power_of_two()
        .ok_or_else(|| candle_core::Error::msg("CUDA top-k sampling width capacity overflow"))?;
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
        candle_core::bail!("{op} expected {rows} sampling parameter rows");
    }
    let mut max_k = 0usize;
    for params in params {
        if !params.inverse_temperature.is_finite() || params.inverse_temperature <= 0.0 {
            candle_core::bail!("{op} requires positive finite inverse temperatures");
        }
        if params.top_k == 0 || params.top_k > CUDA_TOPK_MAX_K {
            candle_core::bail!("{op} top-k must be in [1, {CUDA_TOPK_MAX_K}]");
        }
        if !params.top_p.is_finite() || !params.min_p.is_finite() {
            candle_core::bail!("{op} requires finite top-p and min-p values");
        }
        if !(0.0..1.0).contains(&params.uniform) {
            candle_core::bail!("{op} requires uniforms in [0, 1)");
        }
        max_k = max_k.max(params.top_k.min(vocab));
    }
    Ok(max_k)
}

#[cfg(feature = "cuda")]
fn copy_cuda_topk_sampling_params(
    dev: &candle_core::CudaDevice,
    slot: &mut CudaTopKSamplingSlot,
    params: &[CudaTopKSamplingParams],
) -> Result<()> {
    let host = slot
        .params_host
        .as_mut_slice()
        .map_err(candle_core::Error::wrap)?;
    for (row, params) in params.iter().enumerate() {
        let start = row * CUDA_TOPK_SAMPLING_PARAM_WIDTH;
        let top_k = u16::try_from(params.top_k).map_err(candle_core::Error::wrap)?;
        host[start] = params.inverse_temperature;
        host[start + 1] = f32::from(top_k);
        host[start + 2] = params.top_p;
        host[start + 3] = params.min_p;
        host[start + 4] = params.uniform;
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
    use candle_core::backend::{BackendDevice, BackendStorage};
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::cuda_backend::CudaStorageSlice;

    if !matches!(input.dtype(), DType::BF16 | DType::F16 | DType::F32) {
        candle_core::bail!("{op} requires BF16, F16, or F32 logits");
    }
    if !input.is_contiguous() {
        return Err(candle_core::Error::RequiresContiguous { op });
    }
    let [rows, vocab] = input.dims() else {
        candle_core::bail!("{op} requires logits with shape [batch, vocab]");
    };
    if *rows == 0 || *vocab == 0 {
        candle_core::bail!("{op} requires non-empty logits");
    }
    let max_k = validate_cuda_topk_sampling_params(params, *rows, *vocab, op)?;
    let (storage, _) = input.storage_and_layout();
    let storage = match &*storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{op} requires CUDA logits"),
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
            candle_core::bail!("{op} cannot resize while submissions are pending");
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
        let candle_core::Storage::Cuda(packed_storage) = &*packed_storage else {
            unreachable!("ranked top-k output is CUDA")
        };
        let CudaStorageSlice::F32(packed_slice) = &packed_storage.slice else {
            unreachable!("ranked top-k output is F32")
        };
        let (token_storage, token_layout) = reservation.device_tokens.storage_and_layout();
        let candle_core::Storage::Cuda(token_storage) = &*token_storage else {
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
        unsafe {
            ffi::sample_ranked_topk(
                packed_ptr,
                params_ptr,
                tokens_ptr,
                i32::try_from(*rows).map_err(candle_core::Error::wrap)?,
                i32::try_from(ranked.k).map_err(candle_core::Error::wrap)?,
                stream.cu_stream() as i64,
            );
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
    consumer_stream: &Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
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
    consumer_stream: &Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
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
        candle_core::bail!("invalid CUDA top-k sampling output");
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
