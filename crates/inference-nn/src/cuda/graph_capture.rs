use std::{
    collections::HashMap,
    ptr::NonNull,
    sync::{Arc, Mutex, OnceLock},
};

use candle_core::cuda_backend::cudarc::driver::{
    CudaEvent, CudaStream, DevicePtr, PinnedHostSlice, sys,
};
use candle_core::{DType, Device, DeviceLocation, Storage, Tensor, Var};

const CUDA_GRAPH_INSTANTIATE_FLAGS: u64 =
    sys::CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH.0 as u64;

const CUDA_GRAPH_BATCH_BUCKET_GRANULARITY: usize = 8;

const CUDA_GRAPH_COARSE_EXACT_BATCH_BUCKETS: usize = 8;

const TARGET_SINGLE_TOKEN_EXACT_BATCH_BUCKETS: usize = 16;

pub const CUDA_GRAPH_MAX_BATCH_BUCKET: usize = 128;

pub const CUDA_GRAPH_PRECAPTURE_MAX_BATCH: usize = CUDA_GRAPH_MAX_BATCH_BUCKET;

const CUDA_GRAPH_EVENTS_METRIC: &str = "inference_cuda_graph_events_total";

const CUDA_GRAPH_DISPATCH_METRIC: &str = "inference_cuda_graph_dispatch_total";

const CUDA_GRAPH_EVICTIONS_METRIC: &str = "inference_cuda_graph_evictions_total";

const CUDA_GRAPH_RESIDENT_ENTRIES_METRIC: &str = "inference_cuda_graph_resident_entries";

static CUDA_GRAPH_MEMORY_POOL_SCOPES: OnceLock<Mutex<HashMap<usize, MemoryPoolScopeState>>> =
    OnceLock::new();

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CudaGraphComponent {
    Target,
    DFlash,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CudaGraphEvictionReason {
    Capacity,
    MemoryPressure,
    SpecStateBudget,
}

impl CudaGraphEvictionReason {
    const fn label(self) -> &'static str {
        match self {
            Self::Capacity => "capacity",
            Self::MemoryPressure => "memory_pressure",
            Self::SpecStateBudget => "spec_state_budget",
        }
    }
}

impl CudaGraphComponent {
    const fn label(self) -> &'static str {
        match self {
            Self::Target => "target",
            Self::DFlash => "dflash",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CudaGraphEvent {
    Capture,
    Replay,
    EagerFallback,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CudaGraphDispatchMode {
    Replay,
    Eager,
    Skipped,
}

impl CudaGraphDispatchMode {
    const fn label(self) -> &'static str {
        match self {
            Self::Replay => "replay",
            Self::Eager => "eager",
            Self::Skipped => "skipped",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CudaGraphDispatchReason {
    CacheHit,
    Disabled,
    ModelUnsupported,
    SpeculativeConflict,
    PagedAttentionUnavailable,
    Prefill,
    IncompatibleShape,
    BatchUnsupported,
    CacheConfigUnavailable,
    RuntimeDisabled,
    PaddingUnavailable,
    CachePopulation,
    Fallback,
}

impl CudaGraphDispatchReason {
    const fn label(self) -> &'static str {
        match self {
            Self::CacheHit => "cache_hit",
            Self::Disabled => "disabled",
            Self::ModelUnsupported => "model_unsupported",
            Self::SpeculativeConflict => "speculative_conflict",
            Self::PagedAttentionUnavailable => "paged_attention_unavailable",
            Self::Prefill => "prefill",
            Self::IncompatibleShape => "incompatible_shape",
            Self::BatchUnsupported => "batch_unsupported",
            Self::CacheConfigUnavailable => "cache_config_unavailable",
            Self::RuntimeDisabled => "runtime_disabled",
            Self::PaddingUnavailable => "padding_unavailable",
            Self::CachePopulation => "cache_population",
            Self::Fallback => "fallback",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct CudaGraphDispatchLabels {
    component: &'static str,
    mode: &'static str,
    reason: &'static str,
}

pub fn record_cuda_graph_dispatch(
    component: CudaGraphComponent,
    mode: CudaGraphDispatchMode,
    reason: CudaGraphDispatchReason,
) {
    let labels = cuda_graph_dispatch_labels(component, mode, reason);
    metrics::counter!(
        CUDA_GRAPH_DISPATCH_METRIC,
        "component" => labels.component,
        "mode" => labels.mode,
        "reason" => labels.reason,
    )
    .increment(1);
}

impl CudaGraphEvent {
    const fn label(self) -> &'static str {
        match self {
            Self::Capture => "capture",
            Self::Replay => "replay",
            Self::EagerFallback => "eager_fallback",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum CudaGraphOutcome {
    Success,
    Failure,
}

impl CudaGraphOutcome {
    const fn label(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Failure => "failure",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct CudaGraphEventLabels {
    component: &'static str,
    event: &'static str,
    outcome: &'static str,
}

fn record_cuda_graph_event(
    component: CudaGraphComponent,
    event: CudaGraphEvent,
    outcome: CudaGraphOutcome,
) {
    let labels = cuda_graph_event_labels(component, event, outcome);
    metrics::counter!(
        CUDA_GRAPH_EVENTS_METRIC,
        "component" => labels.component,
        "event" => labels.event,
        "outcome" => labels.outcome,
    )
    .increment(1);
}

pub fn record_cuda_graph_evictions(
    component: CudaGraphComponent,
    reason: CudaGraphEvictionReason,
    count: usize,
) {
    metrics::counter!(
        CUDA_GRAPH_EVICTIONS_METRIC,
        "component" => component.label(),
        "reason" => reason.label()
    )
    .increment(u64::try_from(count).unwrap_or(u64::MAX));
}

pub fn record_cuda_graph_resident_entries(component: CudaGraphComponent, count: usize) {
    let count = u32::try_from(count).unwrap_or(u32::MAX);
    metrics::gauge!(
        CUDA_GRAPH_RESIDENT_ENTRIES_METRIC,
        "component" => component.label()
    )
    .set(f64::from(count));
}

pub fn take_cuda_graph_capacity_eviction<T>(entries: &mut Vec<T>, capacity: usize) -> Option<T> {
    assert!(capacity > 0, "CUDA graph cache capacity must be nonzero");
    (entries.len() >= capacity).then(|| entries.remove(0))
}

pub fn reclaim_cuda_graph_entries(
    max_entries: usize,
    reclaim_target: impl FnOnce(usize) -> usize,
    reclaim_speculative: impl FnOnce(usize) -> usize,
) -> usize {
    if max_entries == 0 {
        return 0;
    }
    let target = reclaim_target(max_entries);
    debug_assert!(target <= max_entries);
    let remaining = max_entries - target;
    if remaining == 0 {
        return target;
    }
    let speculative = reclaim_speculative(remaining);
    debug_assert!(speculative <= remaining);
    target + speculative
}

#[must_use]
pub struct CudaGraphEventGuard {
    component: CudaGraphComponent,
    event: CudaGraphEvent,
    outcome: CudaGraphOutcome,
}

impl CudaGraphEventGuard {
    pub const fn new(component: CudaGraphComponent, event: CudaGraphEvent) -> Self {
        Self {
            component,
            event,
            outcome: CudaGraphOutcome::Failure,
        }
    }

    pub fn success(mut self) {
        self.outcome = CudaGraphOutcome::Success;
    }
}

impl Drop for CudaGraphEventGuard {
    fn drop(&mut self) {
        record_cuda_graph_event(self.component, self.event, self.outcome);
        if self.outcome == CudaGraphOutcome::Success && self.event == CudaGraphEvent::EagerFallback
        {
            record_cuda_graph_dispatch(
                self.component,
                CudaGraphDispatchMode::Eager,
                CudaGraphDispatchReason::Fallback,
            );
        }
    }
}

struct MemoryPoolScopeState {
    graph_guards: usize,
    engine_retainers: usize,
    release_threshold: u64,
}

/// Graph batch bucket a decode batch pads up to, or None when it is too large to graph.
pub fn cuda_graph_batch_bucket(
    component: CudaGraphComponent,
    q_len: usize,
    batch: usize,
) -> Option<usize> {
    if batch == 0 {
        None
    } else if batch <= cuda_graph_exact_batch_buckets(component, q_len) {
        Some(batch)
    } else {
        let bucket = batch
            .div_ceil(CUDA_GRAPH_BATCH_BUCKET_GRANULARITY)
            .saturating_mul(CUDA_GRAPH_BATCH_BUCKET_GRANULARITY);
        (bucket <= CUDA_GRAPH_MAX_BATCH_BUCKET).then_some(bucket)
    }
}

pub fn cuda_graph_precapture_max_batch(
    component: CudaGraphComponent,
    q_len: usize,
    configured_max: usize,
) -> usize {
    if configured_max == 0 {
        0
    } else {
        cuda_graph_batch_bucket(component, q_len, configured_max)
            .unwrap_or(CUDA_GRAPH_MAX_BATCH_BUCKET)
    }
}

/// The batch buckets captured ahead of time at load.
pub fn cuda_graph_precapture_batches(
    component: CudaGraphComponent,
    q_len: usize,
) -> impl Iterator<Item = usize> {
    let exact = cuda_graph_exact_batch_buckets(component, q_len);
    (1..=exact).chain(
        (exact + CUDA_GRAPH_BATCH_BUCKET_GRANULARITY..=CUDA_GRAPH_PRECAPTURE_MAX_BATCH)
            .step_by(CUDA_GRAPH_BATCH_BUCKET_GRANULARITY),
    )
}

pub struct CudaGraphHandle {
    graph: sys::CUgraph,
    exec: sys::CUgraphExec,
    stream: Arc<CudaStream>,
}

impl Drop for CudaGraphHandle {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
        let _ = self.stream.context().bind_to_thread();
        if !self.exec.is_null() {
            let _ = unsafe { sys::cuGraphExecDestroy(self.exec) };
            self.exec = std::ptr::null_mut();
        }
        if !self.graph.is_null() {
            let _ = unsafe { sys::cuGraphDestroy(self.graph) };
            self.graph = std::ptr::null_mut();
        }
    }
}

impl CudaGraphHandle {
    pub fn end_capture(stream: &Arc<CudaStream>) -> candle_core::Result<Option<Self>> {
        let mut graph = std::ptr::null_mut();
        let result = unsafe { sys::cuStreamEndCapture(stream.cu_stream(), &mut graph) };
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(candle_core::Error::msg(format!("{result:?}"))
                .context("CUDA graph stream end capture failed"));
        }
        if graph.is_null() {
            return Ok(None);
        }

        let mut exec = std::ptr::null_mut();
        let result = unsafe {
            sys::cuGraphInstantiateWithFlags(&mut exec, graph, CUDA_GRAPH_INSTANTIATE_FLAGS)
        };
        if result != sys::CUresult::CUDA_SUCCESS {
            let _ = unsafe { sys::cuGraphDestroy(graph) };
            return Err(candle_core::Error::msg(format!("{result:?}"))
                .context("CUDA graph instantiate failed"));
        }

        Ok(Some(Self {
            graph,
            exec,
            stream: stream.clone(),
        }))
    }

    pub fn upload(&self) -> candle_core::Result<()> {
        let result = unsafe { sys::cuGraphUpload(self.exec, self.stream.cu_stream()) };
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(
                candle_core::Error::msg(format!("{result:?}")).context("CUDA graph upload failed")
            );
        }
        let _ = self.stream.context().check_err();
        Ok(())
    }

    pub fn launch(&self) -> candle_core::Result<()> {
        let result = unsafe { sys::cuGraphLaunch(self.exec, self.stream.cu_stream()) };
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(
                candle_core::Error::msg(format!("{result:?}")).context("CUDA graph launch failed")
            );
        }
        let _ = self.stream.context().check_err();
        Ok(())
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

struct CudaGraphPinnedAllocation<T> {
    allocation: PinnedHostSlice<T>,
    ptr: NonNull<T>,
}

impl<T> CudaGraphPinnedAllocation<T> {
    fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.allocation.len()) }
    }

    fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }
}

enum CudaGraphPinnedData {
    U8(CudaGraphPinnedAllocation<u8>),
    U32(CudaGraphPinnedAllocation<u32>),
    I32(CudaGraphPinnedAllocation<i32>),
    I64(CudaGraphPinnedAllocation<i64>),
    F32(CudaGraphPinnedAllocation<f32>),
}

struct CudaGraphPinnedBuffer {
    data: CudaGraphPinnedData,
    initialized: bool,
}

struct CudaGraphCopyCompletion {
    event: CudaEvent,
    stream: Arc<CudaStream>,
    pending: bool,
    active: bool,
    ordered_after_graph: bool,
}

pub struct CudaGraphHostStaging {
    buffers: HashMap<(&'static str, DeviceLocation), CudaGraphPinnedBuffer>,
    completions: HashMap<DeviceLocation, CudaGraphCopyCompletion>,
    graph_complete: CudaEvent,
    graph_stream: Arc<CudaStream>,
    graph_pending: bool,
}

fn same_cuda_stream(left: &CudaStream, right: &CudaStream) -> bool {
    Arc::ptr_eq(left.context(), right.context()) && left.cu_stream() == right.cu_stream()
}

impl Drop for CudaGraphHostStaging {
    fn drop(&mut self) {
        for completion in self.completions.values() {
            if completion.pending {
                let _ = completion.event.synchronize();
            } else if completion.active {
                let _ = completion.stream.synchronize();
            }
        }
    }
}

pub fn cuda_decode_graphs_enabled() -> bool {
    crate::perf_flags::cuda_graphs_enabled()
}

#[must_use]
pub struct CudaGraphMemoryPoolGuard {
    stream: Arc<CudaStream>,
    pool: Option<usize>,
}

impl Drop for CudaGraphMemoryPoolGuard {
    fn drop(&mut self) {
        let Some(pool) = self.pool else { return };
        if let Err(err) = self.stream.context().bind_to_thread() {
            tracing::warn!("Failed to bind CUDA context while restoring graph memory pool: {err}");
        }
        let scopes = CUDA_GRAPH_MEMORY_POOL_SCOPES.get_or_init(Default::default);
        let mut scopes = scopes
            .lock()
            .expect("CUDA graph memory pool scopes poisoned");
        let Some(scope) = scopes.get_mut(&pool) else {
            tracing::warn!("CUDA graph memory pool scope disappeared before restoration");
            return;
        };
        scope.graph_guards = scope
            .graph_guards
            .checked_sub(1)
            .expect("CUDA graph memory pool guard underflow");
        if scope.graph_guards != 0 {
            return;
        }
        let restore_threshold = scope.engine_retainers == 0;
        if restore_threshold {
            let pool_ptr = pool as sys::CUmemoryPool;
            if let Err(err) = set_memory_pool_release_threshold(pool_ptr, scope.release_threshold) {
                tracing::warn!("Failed to restore CUDA graph memory pool threshold: {err:?}");
            }
        }
        let _ = self.stream.synchronize();
        if let Err(err) = trim_cuda_graph_memory_bound(&self.stream) {
            tracing::warn!("Failed to trim CUDA graph memory after capture: {err:?}");
        }
        if restore_threshold {
            scopes.remove(&pool);
        }
    }
}

#[must_use]
pub struct CudaMemoryPoolRetention {
    stream: Arc<CudaStream>,
    pool: Option<usize>,
}

impl Drop for CudaMemoryPoolRetention {
    fn drop(&mut self) {
        let Some(pool) = self.pool else { return };
        if let Err(err) = self.stream.context().bind_to_thread() {
            tracing::warn!(
                "Failed to bind CUDA context while releasing memory pool retention: {err}"
            );
        }
        let scopes = CUDA_GRAPH_MEMORY_POOL_SCOPES.get_or_init(Default::default);
        let mut scopes = scopes
            .lock()
            .expect("CUDA graph memory pool scopes poisoned");
        let Some(scope) = scopes.get_mut(&pool) else {
            tracing::warn!("CUDA memory pool retention disappeared before restoration");
            return;
        };
        scope.engine_retainers = scope
            .engine_retainers
            .checked_sub(1)
            .expect("CUDA memory pool retention underflow");
        if scope.engine_retainers != 0 || scope.graph_guards != 0 {
            return;
        }
        if let Err(err) =
            set_memory_pool_release_threshold(pool as sys::CUmemoryPool, scope.release_threshold)
        {
            tracing::warn!("Failed to restore CUDA memory pool retention threshold: {err:?}");
        }
        scopes.remove(&pool);
    }
}

fn memory_pool_scope(
    scopes: &mut HashMap<usize, MemoryPoolScopeState>,
    pool: sys::CUmemoryPool,
) -> candle_core::Result<&mut MemoryPoolScopeState> {
    use std::collections::hash_map::Entry;

    Ok(match scopes.entry(pool as usize) {
        Entry::Occupied(scope) => scope.into_mut(),
        Entry::Vacant(entry) => {
            let release_threshold = memory_pool_release_threshold(pool)?;
            set_memory_pool_release_threshold(pool, u64::MAX)?;
            entry.insert(MemoryPoolScopeState {
                graph_guards: 0,
                engine_retainers: 0,
                release_threshold,
            })
        }
    })
}

pub fn retain_cuda_memory_pool(
    stream: &Arc<CudaStream>,
) -> candle_core::Result<CudaMemoryPoolRetention> {
    let pool = if stream.context().has_async_alloc() {
        let pool = cuda_memory_pool(stream)?;
        let scopes = CUDA_GRAPH_MEMORY_POOL_SCOPES.get_or_init(Default::default);
        let mut scopes = scopes
            .lock()
            .expect("CUDA graph memory pool scopes poisoned");
        let scope = memory_pool_scope(&mut scopes, pool)?;
        scope.engine_retainers = scope
            .engine_retainers
            .checked_add(1)
            .expect("CUDA memory pool retention overflow");
        Some(pool as usize)
    } else {
        None
    };
    Ok(CudaMemoryPoolRetention {
        stream: stream.clone(),
        pool,
    })
}

fn memory_pool_release_threshold(pool: sys::CUmemoryPool) -> candle_core::Result<u64> {
    let mut value = 0u64;
    let result = unsafe {
        sys::cuMemPoolGetAttribute(
            pool,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
            (&mut value as *mut u64).cast(),
        )
    };
    if result != sys::CUresult::CUDA_SUCCESS {
        return Err(candle_core::Error::msg(format!("{result:?}"))
            .context("CUDA graph mempool release threshold lookup failed"));
    }
    Ok(value)
}

fn set_memory_pool_release_threshold(
    pool: sys::CUmemoryPool,
    mut value: u64,
) -> candle_core::Result<()> {
    let result = unsafe {
        sys::cuMemPoolSetAttribute(
            pool,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
            (&mut value as *mut u64).cast(),
        )
    };
    if result != sys::CUresult::CUDA_SUCCESS {
        return Err(candle_core::Error::msg(format!("{result:?}"))
            .context("CUDA graph mempool release threshold setup failed"));
    }
    Ok(())
}

fn cuda_memory_pool(stream: &Arc<CudaStream>) -> candle_core::Result<sys::CUmemoryPool> {
    stream
        .context()
        .bind_to_thread()
        .map_err(candle_core::Error::wrap)?;
    let mut pool = std::ptr::null_mut();
    let result = unsafe { sys::cuDeviceGetMemPool(&mut pool, stream.context().cu_device()) };
    if result != sys::CUresult::CUDA_SUCCESS {
        return Err(candle_core::Error::msg(format!("{result:?}"))
            .context("CUDA graph mempool lookup failed"));
    }
    Ok(pool)
}

pub fn cuda_graph_memory_pool_scope_active(device: &Device) -> candle_core::Result<bool> {
    let Device::Cuda(device) = device else {
        return Ok(false);
    };
    let stream = device.cuda_stream();
    if !stream.context().has_async_alloc() {
        return Ok(false);
    }
    let pool = cuda_memory_pool(&stream)? as usize;
    let scopes = CUDA_GRAPH_MEMORY_POOL_SCOPES.get_or_init(Default::default);
    let scopes = scopes
        .lock()
        .expect("CUDA graph memory pool scopes poisoned");
    Ok(scopes
        .get(&pool)
        .is_some_and(|scope| scope.graph_guards != 0))
}

fn trim_cuda_graph_memory_bound(stream: &Arc<CudaStream>) -> candle_core::Result<()> {
    let result = unsafe { sys::cuDeviceGraphMemTrim(stream.context().cu_device()) };
    if result != sys::CUresult::CUDA_SUCCESS {
        return Err(
            candle_core::Error::msg(format!("{result:?}")).context("CUDA graph memory trim failed")
        );
    }
    Ok(())
}

pub fn cuda_graph_memory_attribute(
    stream: &Arc<CudaStream>,
    attribute: sys::CUgraphMem_attribute,
) -> candle_core::Result<usize> {
    stream
        .context()
        .bind_to_thread()
        .map_err(candle_core::Error::wrap)?;
    let mut value = 0usize;
    let result = unsafe {
        sys::cuDeviceGetGraphMemAttribute(
            stream.context().cu_device(),
            attribute,
            (&mut value as *mut usize).cast(),
        )
    };
    if result != sys::CUresult::CUDA_SUCCESS {
        return Err(candle_core::Error::msg(format!("{result:?}"))
            .context("CUDA graph memory attribute lookup failed"));
    }
    Ok(value)
}

pub fn trim_cuda_graph_memory(stream: &Arc<CudaStream>) -> candle_core::Result<()> {
    stream
        .context()
        .bind_to_thread()
        .map_err(candle_core::Error::wrap)
        .map_err(|err| err.context("CUDA graph memory trim context bind failed"))?;
    let pool = cuda_memory_pool(stream)? as usize;
    let scopes = CUDA_GRAPH_MEMORY_POOL_SCOPES.get_or_init(Default::default);
    let scopes = scopes
        .lock()
        .expect("CUDA graph memory pool scopes poisoned");
    if scopes
        .get(&pool)
        .is_some_and(|scope| scope.graph_guards != 0)
    {
        return Ok(());
    }
    stream
        .synchronize()
        .map_err(candle_core::Error::wrap)
        .map_err(|err| err.context("CUDA graph memory trim synchronization failed"))?;
    trim_cuda_graph_memory_bound(stream)
}

pub fn prepare_cuda_graph_memory_pool(
    stream: &Arc<CudaStream>,
) -> candle_core::Result<CudaGraphMemoryPoolGuard> {
    if !stream.context().has_async_alloc() {
        return Ok(CudaGraphMemoryPoolGuard {
            stream: stream.clone(),
            pool: None,
        });
    }

    let pool = cuda_memory_pool(stream)?;
    let pool_key = pool as usize;
    let scopes = CUDA_GRAPH_MEMORY_POOL_SCOPES.get_or_init(Default::default);
    let mut scopes = scopes
        .lock()
        .expect("CUDA graph memory pool scopes poisoned");
    let scope = memory_pool_scope(&mut scopes, pool)?;
    scope.graph_guards = scope
        .graph_guards
        .checked_add(1)
        .expect("CUDA graph memory pool guard overflow");
    drop(scopes);

    let guard = CudaGraphMemoryPoolGuard {
        stream: stream.clone(),
        pool: Some(pool_key),
    };
    for attr in [
        sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES,
        sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC,
        sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES,
    ] {
        let mut enabled = 1i32;
        let result =
            unsafe { sys::cuMemPoolSetAttribute(pool, attr, (&mut enabled as *mut i32).cast()) };
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(candle_core::Error::msg(format!("{result:?}"))
                .context("CUDA graph mempool reuse setup failed"));
        }
    }

    Ok(guard)
}

pub fn disable_event_tracking_for_capture(stream: &Arc<CudaStream>) -> bool {
    let restore = stream.context().is_event_tracking();
    if restore {
        unsafe { stream.context().disable_event_tracking() };
    }
    restore
}

pub fn restore_event_tracking_after_capture(stream: &Arc<CudaStream>, restore: bool) {
    if restore {
        unsafe { stream.context().enable_event_tracking() };
    }
}

pub fn end_cuda_capture_discard(stream: &Arc<CudaStream>) {
    if matches!(
        stream.capture_status(),
        Ok(status) if status != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
    ) {
        let mut graph = std::ptr::null_mut();
        let result = unsafe { sys::cuStreamEndCapture(stream.cu_stream(), &mut graph) };
        if result == sys::CUresult::CUDA_SUCCESS && !graph.is_null() {
            let _ = unsafe { sys::cuGraphDestroy(graph) };
        }
    }
}

impl CudaGraphPinnedBuffer {
    fn new(dst: &Var) -> candle_core::Result<Self> {
        let Device::Cuda(device) = dst.device() else {
            candle_core::bail!("CUDA graph host staging requires a CUDA destination");
        };
        let stream = device.cuda_stream();
        let context = stream.context();
        let len = dst.elem_count();
        macro_rules! allocate {
            ($variant:ident, $ty:ty) => {{
                let mut allocation = unsafe {
                    context
                        .alloc_pinned::<$ty>(len)
                        .map_err(candle_core::Error::wrap)?
                };
                let ptr = NonNull::new(allocation.as_mut_ptr().map_err(candle_core::Error::wrap)?)
                    .ok_or_else(|| {
                        candle_core::Error::msg("CUDA returned a null pinned pointer")
                    })?;
                CudaGraphPinnedData::$variant(CudaGraphPinnedAllocation { allocation, ptr })
            }};
        }
        let data = match dst.dtype() {
            DType::U8 => allocate!(U8, u8),
            DType::U32 => allocate!(U32, u32),
            DType::I32 => allocate!(I32, i32),
            DType::I64 => allocate!(I64, i64),
            DType::F32 => allocate!(F32, f32),
            dtype => candle_core::bail!(
                "CUDA graph host staging does not support metadata dtype {dtype:?}"
            ),
        };
        Ok(Self {
            data,
            initialized: false,
        })
    }

    pub fn copy_from(
        &mut self,
        src: &Tensor,
        dst: &Var,
        stream: &Arc<CudaStream>,
    ) -> candle_core::Result<()> {
        if src.shape() != dst.shape() || src.dtype() != dst.dtype() {
            candle_core::bail!("CUDA graph host staging expected matching tensors");
        }
        let (src_storage, src_layout) = src.storage_and_layout();
        let Storage::Cpu(src_storage) = &*src_storage else {
            candle_core::bail!("CUDA graph host staging expected CPU source metadata");
        };
        if !src_layout.is_contiguous() {
            candle_core::bail!("CUDA graph host staging expected contiguous source metadata");
        }
        let (dst_storage, dst_layout) = dst.storage_and_layout();
        let Storage::Cuda(dst_storage) = &*dst_storage else {
            candle_core::bail!("CUDA graph host staging expected CUDA destination metadata");
        };
        if !dst_layout.is_contiguous() {
            candle_core::bail!("CUDA graph host staging expected contiguous destination metadata");
        }
        let len = src.elem_count();
        let src_offset = src_layout.start_offset();
        let dst_offset = dst_layout.start_offset();

        macro_rules! stage_and_copy {
            ($variant:ident, $ty:ty) => {{
                let CudaGraphPinnedData::$variant(host) = &mut self.data else {
                    candle_core::bail!("CUDA graph host staging dtype changed");
                };
                let src = src_storage.as_slice::<$ty>()?;
                let src = &src[src_offset..src_offset + len];
                let host_slice = host.as_mut_slice();
                if self.initialized && host_slice == src {
                    return Ok(());
                }
                host_slice.copy_from_slice(src);
                self.initialized = true;
                let dst = dst_storage.as_cuda_slice::<$ty>()?;
                let dst = dst.slice(dst_offset..dst_offset + len);
                let (dst_ptr, _dst_guard) = dst.device_ptr(stream);
                let result = unsafe {
                    sys::cuMemcpyHtoDAsync_v2(
                        dst_ptr,
                        host.as_ptr().cast(),
                        len * std::mem::size_of::<$ty>(),
                        stream.cu_stream(),
                    )
                };
                if result != sys::CUresult::CUDA_SUCCESS {
                    return Err(candle_core::Error::msg(format!("{result:?}"))
                        .context("CUDA graph metadata H2D copy failed"));
                }
            }};
        }

        match src.dtype() {
            DType::U8 => stage_and_copy!(U8, u8),
            DType::U32 => stage_and_copy!(U32, u32),
            DType::I32 => stage_and_copy!(I32, i32),
            DType::I64 => stage_and_copy!(I64, i64),
            DType::F32 => stage_and_copy!(F32, f32),
            dtype => candle_core::bail!(
                "CUDA graph host staging does not support metadata dtype {dtype:?}"
            ),
        }
        Ok(())
    }

    fn copy_from_u32_slice(
        &mut self,
        src: &[u32],
        dst: &Var,
        stream: &Arc<CudaStream>,
    ) -> candle_core::Result<()> {
        if dst.dtype() != DType::U32 || dst.elem_count() != src.len() {
            candle_core::bail!("CUDA graph host staging expected matching u32 state indices");
        }
        let (dst_storage, dst_layout) = dst.storage_and_layout();
        let Storage::Cuda(dst_storage) = &*dst_storage else {
            candle_core::bail!("CUDA graph host staging expected CUDA state indices");
        };
        if !dst_layout.is_contiguous() {
            candle_core::bail!("CUDA graph host staging expected contiguous state indices");
        }
        let CudaGraphPinnedData::U32(host) = &mut self.data else {
            candle_core::bail!("CUDA graph host staging state index dtype changed");
        };
        let host_slice = host.as_mut_slice();
        if self.initialized && host_slice == src {
            return Ok(());
        }
        host_slice.copy_from_slice(src);
        self.initialized = true;
        let dst = dst_storage.as_cuda_slice::<u32>()?;
        let dst_offset = dst_layout.start_offset();
        let dst = dst.slice(dst_offset..dst_offset + src.len());
        let (dst_ptr, _dst_guard) = dst.device_ptr(stream);
        let result = unsafe {
            sys::cuMemcpyHtoDAsync_v2(
                dst_ptr,
                host.as_ptr().cast(),
                std::mem::size_of_val(src),
                stream.cu_stream(),
            )
        };
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(candle_core::Error::msg(format!("{result:?}"))
                .context("CUDA graph state index H2D copy failed"));
        }
        Ok(())
    }

    #[cfg(all(feature = "flash-attn", target_family = "unix"))]
    fn copy_from_f32_slice(
        &mut self,
        src: &[f32],
        dst: &Var,
        stream: &Arc<CudaStream>,
    ) -> candle_core::Result<()> {
        if dst.dtype() != DType::F32 || dst.elem_count() != src.len() {
            candle_core::bail!("CUDA graph host staging expected matching f32 metadata");
        }
        let (dst_storage, dst_layout) = dst.storage_and_layout();
        let Storage::Cuda(dst_storage) = &*dst_storage else {
            candle_core::bail!("CUDA graph host staging expected CUDA f32 metadata");
        };
        if !dst_layout.is_contiguous() {
            candle_core::bail!("CUDA graph host staging expected contiguous f32 metadata");
        }
        let CudaGraphPinnedData::F32(host) = &mut self.data else {
            candle_core::bail!("CUDA graph host staging f32 metadata dtype changed");
        };
        let host_slice = host.as_mut_slice();
        if self.initialized && host_slice == src {
            return Ok(());
        }
        host_slice.copy_from_slice(src);
        self.initialized = true;
        let dst = dst_storage.as_cuda_slice::<f32>()?;
        let dst_offset = dst_layout.start_offset();
        let dst = dst.slice(dst_offset..dst_offset + src.len());
        let (dst_ptr, _dst_guard) = dst.device_ptr(stream);
        let result = unsafe {
            sys::cuMemcpyHtoDAsync_v2(
                dst_ptr,
                host.as_ptr().cast(),
                std::mem::size_of_val(src),
                stream.cu_stream(),
            )
        };
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(candle_core::Error::msg(format!("{result:?}"))
                .context("CUDA graph f32 metadata H2D copy failed"));
        }
        Ok(())
    }

    #[cfg(all(feature = "flash-attn", target_family = "unix"))]
    fn copy_from_i64_slice(
        &mut self,
        src: &[i64],
        dst: &Var,
        stream: &Arc<CudaStream>,
    ) -> candle_core::Result<()> {
        if dst.dtype() != DType::I64 || dst.elem_count() != src.len() {
            candle_core::bail!("CUDA graph host staging expected matching i64 metadata");
        }
        let (dst_storage, dst_layout) = dst.storage_and_layout();
        let Storage::Cuda(dst_storage) = &*dst_storage else {
            candle_core::bail!("CUDA graph host staging expected CUDA i64 metadata");
        };
        if !dst_layout.is_contiguous() {
            candle_core::bail!("CUDA graph host staging expected contiguous i64 metadata");
        }
        let CudaGraphPinnedData::I64(host) = &mut self.data else {
            candle_core::bail!("CUDA graph host staging i64 metadata dtype changed");
        };
        let host_slice = host.as_mut_slice();
        if self.initialized && host_slice == src {
            return Ok(());
        }
        host_slice.copy_from_slice(src);
        self.initialized = true;
        let dst = dst_storage.as_cuda_slice::<i64>()?;
        let dst_offset = dst_layout.start_offset();
        let dst = dst.slice(dst_offset..dst_offset + src.len());
        let (dst_ptr, _dst_guard) = dst.device_ptr(stream);
        let result = unsafe {
            sys::cuMemcpyHtoDAsync_v2(
                dst_ptr,
                host.as_ptr().cast(),
                std::mem::size_of_val(src),
                stream.cu_stream(),
            )
        };
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(candle_core::Error::msg(format!("{result:?}"))
                .context("CUDA graph i64 metadata H2D copy failed"));
        }
        Ok(())
    }
}

impl CudaGraphHostStaging {
    pub fn staged_buffer_count(&self) -> usize {
        self.buffers.len()
    }

    pub fn pending_completion_count(&self) -> usize {
        self.completions.len()
    }

    pub fn new(graph_stream: Arc<CudaStream>) -> candle_core::Result<Self> {
        let graph_complete = graph_stream
            .context()
            .new_event(None)
            .map_err(candle_core::Error::wrap)?;
        Ok(Self {
            buffers: HashMap::new(),
            completions: HashMap::new(),
            graph_complete,
            graph_stream,
            graph_pending: false,
        })
    }

    pub fn update(
        &mut self,
        copy: impl FnOnce(&mut Self) -> candle_core::Result<()>,
    ) -> candle_core::Result<()> {
        self.begin_update()?;
        let copy_result = copy(self);
        let finish_result = self.finish_update();
        copy_result.and(finish_result)
    }

    fn begin_update(&mut self) -> candle_core::Result<()> {
        for completion in self.completions.values_mut() {
            completion.ordered_after_graph = false;
            if completion.active {
                completion
                    .stream
                    .synchronize()
                    .map_err(candle_core::Error::wrap)?;
                completion.active = false;
            }
            if completion.pending {
                completion
                    .event
                    .synchronize()
                    .map_err(candle_core::Error::wrap)?;
                completion.pending = false;
            }
        }
        Ok(())
    }

    fn finish_update(&mut self) -> candle_core::Result<()> {
        let mut result = Ok(());
        for completion in self.completions.values_mut() {
            if !completion.active {
                continue;
            }
            match completion.event.record(&completion.stream) {
                Ok(()) => {
                    completion.pending = true;
                    completion.active = false;
                }
                Err(err) => {
                    completion.pending = false;
                    if completion.stream.synchronize().is_ok() {
                        completion.active = false;
                    }
                    if result.is_ok() {
                        result = Err(candle_core::Error::wrap(err));
                    }
                }
            }
        }
        result
    }

    pub fn order_before_graph(&self) -> candle_core::Result<()> {
        for completion in self.completions.values() {
            if completion.pending && !same_cuda_stream(&completion.stream, &self.graph_stream) {
                self.graph_stream
                    .wait(&completion.event)
                    .map_err(candle_core::Error::wrap)?;
            }
        }
        Ok(())
    }

    pub fn record_graph_complete(&mut self) -> candle_core::Result<()> {
        self.graph_complete
            .record(&self.graph_stream)
            .map_err(candle_core::Error::wrap)?;
        self.graph_pending = true;
        Ok(())
    }

    fn prepare_copy(
        &mut self,
        location: DeviceLocation,
        stream: &Arc<CudaStream>,
    ) -> candle_core::Result<()> {
        let completion = match self.completions.entry(location) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                let event = stream
                    .context()
                    .new_event(Some(sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC))
                    .map_err(candle_core::Error::wrap)?;
                entry.insert(CudaGraphCopyCompletion {
                    event,
                    stream: stream.clone(),
                    pending: false,
                    active: false,
                    ordered_after_graph: false,
                })
            }
        };
        if !same_cuda_stream(&completion.stream, stream) {
            candle_core::bail!("CUDA graph metadata stream changed during replay");
        }
        if self.graph_pending && !completion.ordered_after_graph {
            if !same_cuda_stream(&completion.stream, &self.graph_stream) {
                completion
                    .stream
                    .wait(&self.graph_complete)
                    .map_err(candle_core::Error::wrap)?;
            }
            completion.ordered_after_graph = true;
        }
        completion.active = true;
        Ok(())
    }

    pub fn copy_from(
        &mut self,
        name: &'static str,
        location: DeviceLocation,
        src: &Tensor,
        dst: &Var,
    ) -> candle_core::Result<()> {
        let stream = dst.device().as_cuda_device()?.cuda_stream();
        self.prepare_copy(location, &stream)?;
        let buffer = match self.buffers.entry((name, location)) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(CudaGraphPinnedBuffer::new(dst)?)
            }
        };
        buffer.copy_from(src, dst, &stream)
    }

    pub fn copy_from_u32_slice(
        &mut self,
        name: &'static str,
        location: DeviceLocation,
        src: &[u32],
        dst: &Var,
    ) -> candle_core::Result<()> {
        let stream = dst.device().as_cuda_device()?.cuda_stream();
        self.prepare_copy(location, &stream)?;
        let buffer = match self.buffers.entry((name, location)) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(CudaGraphPinnedBuffer::new(dst)?)
            }
        };
        buffer.copy_from_u32_slice(src, dst, &stream)
    }

    #[cfg(all(feature = "flash-attn", target_family = "unix"))]
    pub fn copy_from_f32_slice(
        &mut self,
        name: &'static str,
        location: DeviceLocation,
        src: &[f32],
        dst: &Var,
    ) -> candle_core::Result<()> {
        let stream = dst.device().as_cuda_device()?.cuda_stream();
        self.prepare_copy(location, &stream)?;
        let buffer = match self.buffers.entry((name, location)) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(CudaGraphPinnedBuffer::new(dst)?)
            }
        };
        buffer.copy_from_f32_slice(src, dst, &stream)
    }

    #[cfg(all(feature = "flash-attn", target_family = "unix"))]
    pub fn copy_from_i64_slice(
        &mut self,
        name: &'static str,
        location: DeviceLocation,
        src: &[i64],
        dst: &Var,
    ) -> candle_core::Result<()> {
        let stream = dst.device().as_cuda_device()?.cuda_stream();
        self.prepare_copy(location, &stream)?;
        let buffer = match self.buffers.entry((name, location)) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(CudaGraphPinnedBuffer::new(dst)?)
            }
        };
        buffer.copy_from_i64_slice(src, dst, &stream)
    }
}

const fn cuda_graph_dispatch_labels(
    component: CudaGraphComponent,
    mode: CudaGraphDispatchMode,
    reason: CudaGraphDispatchReason,
) -> CudaGraphDispatchLabels {
    CudaGraphDispatchLabels {
        component: component.label(),
        mode: mode.label(),
        reason: reason.label(),
    }
}

const fn cuda_graph_event_labels(
    component: CudaGraphComponent,
    event: CudaGraphEvent,
    outcome: CudaGraphOutcome,
) -> CudaGraphEventLabels {
    CudaGraphEventLabels {
        component: component.label(),
        event: event.label(),
        outcome: outcome.label(),
    }
}

const fn cuda_graph_exact_batch_buckets(component: CudaGraphComponent, q_len: usize) -> usize {
    if matches!(component, CudaGraphComponent::Target) && q_len == 1 {
        TARGET_SINGLE_TOKEN_EXACT_BATCH_BUCKETS
    } else {
        CUDA_GRAPH_COARSE_EXACT_BATCH_BUCKETS
    }
}

unsafe impl Send for CudaGraphHandle {}

// The allocation owns the pointer and all accesses require an exclusive borrow.
unsafe impl<T: Send> Send for CudaGraphPinnedAllocation<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_event_labels_have_fixed_cardinality() {
        use std::collections::HashSet;

        let mut labels = HashSet::new();
        for component in [CudaGraphComponent::Target, CudaGraphComponent::DFlash] {
            for event in [
                CudaGraphEvent::Capture,
                CudaGraphEvent::Replay,
                CudaGraphEvent::EagerFallback,
            ] {
                for outcome in [CudaGraphOutcome::Success, CudaGraphOutcome::Failure] {
                    assert!(labels.insert(cuda_graph_event_labels(component, event, outcome)));
                }
            }
        }

        assert_eq!(
            CUDA_GRAPH_EVENTS_METRIC,
            "inference_cuda_graph_events_total"
        );
        assert_eq!(labels.len(), 12);
        assert_eq!(
            labels
                .iter()
                .map(|labels| labels.component)
                .collect::<HashSet<_>>(),
            HashSet::from(["target", "dflash"])
        );
        assert_eq!(
            labels
                .iter()
                .map(|labels| labels.event)
                .collect::<HashSet<_>>(),
            HashSet::from(["capture", "replay", "eager_fallback"])
        );
        assert_eq!(
            labels
                .iter()
                .map(|labels| labels.outcome)
                .collect::<HashSet<_>>(),
            HashSet::from(["success", "failure"])
        );
    }

    #[test]
    fn graph_dispatch_labels_have_fixed_cardinality() {
        use std::collections::HashSet;

        let reasons = [
            CudaGraphDispatchReason::CacheHit,
            CudaGraphDispatchReason::Disabled,
            CudaGraphDispatchReason::ModelUnsupported,
            CudaGraphDispatchReason::SpeculativeConflict,
            CudaGraphDispatchReason::PagedAttentionUnavailable,
            CudaGraphDispatchReason::Prefill,
            CudaGraphDispatchReason::IncompatibleShape,
            CudaGraphDispatchReason::BatchUnsupported,
            CudaGraphDispatchReason::CacheConfigUnavailable,
            CudaGraphDispatchReason::RuntimeDisabled,
            CudaGraphDispatchReason::PaddingUnavailable,
            CudaGraphDispatchReason::CachePopulation,
            CudaGraphDispatchReason::Fallback,
        ];
        let mut labels = HashSet::new();
        for component in [CudaGraphComponent::Target, CudaGraphComponent::DFlash] {
            for mode in [
                CudaGraphDispatchMode::Replay,
                CudaGraphDispatchMode::Eager,
                CudaGraphDispatchMode::Skipped,
            ] {
                for reason in reasons {
                    assert!(labels.insert(cuda_graph_dispatch_labels(component, mode, reason)));
                }
            }
        }

        assert_eq!(
            CUDA_GRAPH_DISPATCH_METRIC,
            "inference_cuda_graph_dispatch_total"
        );
        assert_eq!(labels.len(), 78);
        assert_eq!(
            labels
                .iter()
                .map(|labels| labels.mode)
                .collect::<HashSet<_>>(),
            HashSet::from(["replay", "eager", "skipped"])
        );
        assert_eq!(
            labels
                .iter()
                .map(|labels| labels.reason)
                .collect::<HashSet<_>>()
                .len(),
            reasons.len()
        );
    }

    #[test]
    fn graph_eviction_labels_have_fixed_cardinality() {
        use std::collections::HashSet;

        let labels = [
            (
                CudaGraphComponent::Target,
                CudaGraphEvictionReason::Capacity,
            ),
            (
                CudaGraphComponent::Target,
                CudaGraphEvictionReason::MemoryPressure,
            ),
            (
                CudaGraphComponent::Target,
                CudaGraphEvictionReason::SpecStateBudget,
            ),
            (
                CudaGraphComponent::DFlash,
                CudaGraphEvictionReason::Capacity,
            ),
            (
                CudaGraphComponent::DFlash,
                CudaGraphEvictionReason::MemoryPressure,
            ),
        ]
        .into_iter()
        .map(|(component, reason)| (component.label(), reason.label()))
        .collect::<HashSet<_>>();

        assert_eq!(
            CUDA_GRAPH_EVICTIONS_METRIC,
            "inference_cuda_graph_evictions_total"
        );
        assert_eq!(
            CUDA_GRAPH_RESIDENT_ENTRIES_METRIC,
            "inference_cuda_graph_resident_entries"
        );
        assert_eq!(labels.len(), 5);
        assert!(labels.contains(&("target", "capacity")));
        assert!(labels.contains(&("dflash", "capacity")));
        assert!(labels.contains(&("target", "memory_pressure")));
        assert!(labels.contains(&("dflash", "memory_pressure")));
        assert!(labels.contains(&("target", "spec_state_budget")));
    }

    #[test]
    fn target_single_token_batch_buckets_are_exact_through_16() {
        assert_eq!(
            cuda_graph_batch_bucket(CudaGraphComponent::Target, 1, 0),
            None
        );
        for batch in 1..=TARGET_SINGLE_TOKEN_EXACT_BATCH_BUCKETS {
            assert_eq!(
                cuda_graph_batch_bucket(CudaGraphComponent::Target, 1, batch),
                Some(batch)
            );
        }
        assert_eq!(
            cuda_graph_batch_bucket(CudaGraphComponent::Target, 1, 17),
            Some(24)
        );
        assert_eq!(
            cuda_graph_batch_bucket(CudaGraphComponent::Target, 1, 31),
            Some(32)
        );
        assert_eq!(
            cuda_graph_batch_bucket(CudaGraphComponent::Target, 1, CUDA_GRAPH_MAX_BATCH_BUCKET),
            Some(CUDA_GRAPH_MAX_BATCH_BUCKET)
        );
        assert_eq!(
            cuda_graph_batch_bucket(
                CudaGraphComponent::Target,
                1,
                CUDA_GRAPH_MAX_BATCH_BUCKET + 1
            ),
            None
        );
    }

    #[test]
    fn multi_token_and_dflash_batches_keep_coarse_buckets() {
        for (component, q_len) in [
            (CudaGraphComponent::Target, 4),
            (CudaGraphComponent::DFlash, 1),
            (CudaGraphComponent::DFlash, 8),
        ] {
            assert_eq!(cuda_graph_batch_bucket(component, q_len, 8), Some(8));
            assert_eq!(cuda_graph_batch_bucket(component, q_len, 9), Some(16));
            assert_eq!(cuda_graph_batch_bucket(component, q_len, 17), Some(24));
        }
    }

    #[test]
    fn precapture_batches_follow_component_and_query_width() {
        assert_eq!(
            cuda_graph_precapture_batches(CudaGraphComponent::Target, 1).collect::<Vec<_>>(),
            vec![
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 24, 32, 40, 48, 56, 64, 72,
                80, 88, 96, 104, 112, 120, 128,
            ]
        );
        assert_eq!(
            cuda_graph_precapture_batches(CudaGraphComponent::Target, 4).collect::<Vec<_>>(),
            vec![
                1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120,
                128,
            ]
        );
        assert_eq!(
            cuda_graph_precapture_batches(CudaGraphComponent::DFlash, 1).collect::<Vec<_>>(),
            cuda_graph_precapture_batches(CudaGraphComponent::Target, 4).collect::<Vec<_>>()
        );
    }

    #[test]
    fn precapture_max_batch_uses_the_shape_policy() {
        assert_eq!(
            cuda_graph_precapture_max_batch(CudaGraphComponent::Target, 1, 0),
            0
        );
        assert_eq!(
            cuda_graph_precapture_max_batch(CudaGraphComponent::Target, 1, 9),
            9
        );
        assert_eq!(
            cuda_graph_precapture_max_batch(CudaGraphComponent::Target, 4, 9),
            16
        );
        assert_eq!(
            cuda_graph_precapture_max_batch(CudaGraphComponent::DFlash, 8, 9),
            16
        );
        assert_eq!(
            cuda_graph_precapture_max_batch(CudaGraphComponent::Target, 1, 17),
            24
        );
        assert_eq!(
            cuda_graph_precapture_max_batch(CudaGraphComponent::Target, 1, 129),
            128
        );
    }

    #[test]
    fn graph_staging_orders_secondary_streams_both_directions() -> anyhow::Result<()> {
        skip_without_cuda!();
        let device = Device::new_cuda(0)?;
        let graph_stream = device.as_cuda_device()?.cuda_stream();
        let copy_stream = graph_stream.fork()?;
        assert!(!same_cuda_stream(&graph_stream, &copy_stream));
        let location = device.location();
        let mut staging = CudaGraphHostStaging::new(graph_stream.clone())?;

        staging.record_graph_complete()?;
        staging.update(|staging| staging.prepare_copy(location, &copy_stream))?;
        assert!(staging.completions[&location].ordered_after_graph);
        staging.order_before_graph()?;

        staging.record_graph_complete()?;
        staging.update(|staging| staging.prepare_copy(location, &copy_stream))?;
        assert!(staging.completions[&location].ordered_after_graph);
        staging.order_before_graph()?;
        graph_stream.synchronize()?;
        Ok(())
    }

    #[test]
    fn graph_memory_pool_scope_restores_release_threshold() -> anyhow::Result<()> {
        skip_without_cuda!();
        let device = Device::new_cuda(0)?;
        let stream = device.as_cuda_device()?.cuda_stream();
        let pool = cuda_memory_pool(&stream)?;
        let original = memory_pool_release_threshold(pool)?;

        let first = prepare_cuda_graph_memory_pool(&stream)?;
        let second = prepare_cuda_graph_memory_pool(&stream)?;
        assert_eq!(memory_pool_release_threshold(pool)?, u64::MAX);
        drop(first);
        assert_eq!(memory_pool_release_threshold(pool)?, u64::MAX);
        drop(second);
        assert_eq!(memory_pool_release_threshold(pool)?, original);
        Ok(())
    }

    #[test]
    fn engine_memory_pool_retention_coordinates_nested_graph_scopes() -> anyhow::Result<()> {
        skip_without_cuda!();
        let device = Device::new_cuda(0)?;
        let alias = Device::new_cuda(0)?;
        let stream = device.as_cuda_device()?.cuda_stream();
        let alias_stream = alias.as_cuda_device()?.cuda_stream();
        let pool = cuda_memory_pool(&stream)?;
        assert_eq!(cuda_memory_pool(&alias_stream)?, pool);
        let original = memory_pool_release_threshold(pool)?;

        for graph_first in [false, true] {
            for engines_drop_first in [false, true] {
                let prior_graph = graph_first
                    .then(|| prepare_cuda_graph_memory_pool(&stream))
                    .transpose()?;
                let first = retain_cuda_memory_pool(&stream)?;
                let second = retain_cuda_memory_pool(&alias_stream)?;
                assert_eq!(memory_pool_release_threshold(pool)?, u64::MAX);
                assert_eq!(cuda_graph_memory_pool_scope_active(&device)?, graph_first);
                let outer = match prior_graph {
                    Some(guard) => guard,
                    None => prepare_cuda_graph_memory_pool(&stream)?,
                };
                let inner = prepare_cuda_graph_memory_pool(&alias_stream)?;
                assert!(cuda_graph_memory_pool_scope_active(&alias)?);

                if engines_drop_first {
                    drop(second);
                    assert_eq!(memory_pool_release_threshold(pool)?, u64::MAX);
                    drop(first);
                    assert_eq!(memory_pool_release_threshold(pool)?, u64::MAX);
                    drop(outer);
                    assert!(cuda_graph_memory_pool_scope_active(&device)?);
                    assert_eq!(memory_pool_release_threshold(pool)?, u64::MAX);
                    drop(inner);
                } else {
                    drop(outer);
                    assert!(cuda_graph_memory_pool_scope_active(&device)?);
                    drop(inner);
                    assert!(!cuda_graph_memory_pool_scope_active(&device)?);
                    assert_eq!(memory_pool_release_threshold(pool)?, u64::MAX);
                    drop(first);
                    assert_eq!(memory_pool_release_threshold(pool)?, u64::MAX);
                    drop(second);
                }
                assert!(!cuda_graph_memory_pool_scope_active(&device)?);
                assert_eq!(memory_pool_release_threshold(pool)?, original);
            }
        }
        Ok(())
    }

    #[test]
    fn graph_memory_cleanup_returns_allocator_to_baseline() -> anyhow::Result<()> {
        skip_without_cuda!();
        let device = Device::new_cuda(0)?;
        let stream = device.as_cuda_device()?.cuda_stream();
        let used_attribute = sys::CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT;
        let reserved_attribute = sys::CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT;
        trim_cuda_graph_memory(&stream)?;
        let used_before = cuda_graph_memory_attribute(&stream, used_attribute)?;
        let reserved_before = cuda_graph_memory_attribute(&stream, reserved_attribute)?;

        let pool = cuda_memory_pool(&stream)?;
        let original = memory_pool_release_threshold(pool)?;
        let retention = retain_cuda_memory_pool(&stream)?;
        let guard = prepare_cuda_graph_memory_pool(&stream)?;
        let input = Var::from_tensor(&Tensor::from_vec(vec![1f32, 2.0], 2, &device)?)?;
        let restore_event_tracking = disable_event_tracking_for_capture(&stream);
        stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;
        let output = input.as_detached_tensor().affine(2.0, 1.0)?;
        let graph = CudaGraphHandle::end_capture(&stream)?
            .ok_or_else(|| anyhow::anyhow!("CUDA graph capture returned no graph"))?;
        restore_event_tracking_after_capture(&stream, restore_event_tracking);
        graph.upload()?;
        graph.launch()?;
        stream.synchronize()?;
        drop(output);
        stream.synchronize()?;
        drop(graph);
        drop(guard);
        assert!(!cuda_graph_memory_pool_scope_active(&device)?);
        assert_eq!(memory_pool_release_threshold(pool)?, u64::MAX);

        trim_cuda_graph_memory(&stream)?;
        assert_eq!(
            cuda_graph_memory_attribute(&stream, used_attribute)?,
            used_before
        );
        assert_eq!(
            cuda_graph_memory_attribute(&stream, reserved_attribute)?,
            reserved_before
        );
        drop(retention);
        assert_eq!(memory_pool_release_threshold(pool)?, original);
        Ok(())
    }
}
