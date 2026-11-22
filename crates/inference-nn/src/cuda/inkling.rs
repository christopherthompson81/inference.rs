//! Inkling 2's per-slot elementwise chains, each one launch (see `kernels/cuda/inkling.cu`): the rows' L2 clip, unit
//! RMS, the depth mix and the short causal convolution with its state. Each takes f32 or bf16 and returns f32, or
//! `None` off CUDA or for a layout it does not take.

use inference_tensor::backend::BackendStorage;
use inference_tensor::cuda_backend::CudaStorage;
use inference_tensor::cuda_backend::cudarc::driver::{
    CudaStream, DevicePtr, DevicePtrMut, SyncOnDrop,
};
use inference_tensor::{DType, Device, Layout, Result, Shape, Storage, Tensor};

// the kernels' dtype codes
const F32_CODE: i32 = 0;
const BF16_CODE: i32 = 1;
// the kernel's bounds: states and queries a mix takes
const MAX_STATES: usize = 16;
const MAX_QUERIES: usize = 4;

fn status(code: i32, what: &str) -> Result<()> {
    if code != 0 {
        inference_tensor::bail!("Inkling {what} failed with CUDA status {code}");
    }
    Ok(())
}

fn i32_of(v: usize) -> Result<i32> {
    i32::try_from(v).map_err(inference_tensor::Error::wrap)
}

fn i64_of(v: usize) -> Result<i64> {
    i64::try_from(v).map_err(inference_tensor::Error::wrap)
}

fn takes(x: &Tensor) -> bool {
    x.device().is_cuda() && matches!(x.dtype(), DType::F32 | DType::BF16)
}

// a tensor's first element on the device, its dtype code, and the guard to hold until the launch is queued
fn device_ptr<'a>(
    storage: &'a Storage,
    layout: &Layout,
    stream: &'a CudaStream,
) -> Result<(u64, i32, SyncOnDrop<'a>)> {
    let Storage::Cuda(s) = storage else {
        inference_tensor::bail!("Inkling kernels need CUDA storage");
    };
    let offset = layout.start_offset() as u64;
    Ok(match s.dtype() {
        DType::F32 => {
            let (p, g) = s.as_cuda_slice::<f32>()?.device_ptr(stream);
            (p + offset * 4, F32_CODE, g)
        }
        DType::BF16 => {
            let (p, g) = s.as_cuda_slice::<half::bf16>()?.device_ptr(stream);
            (p + offset * 2, BF16_CODE, g)
        }
        dtype => inference_tensor::bail!("Inkling kernels take f32 or bf16, not {dtype:?}"),
    })
}

fn wrap(
    out: inference_tensor::cuda_backend::cudarc::driver::CudaSlice<f32>,
    dev: &inference_tensor::CudaDevice,
    shape: Shape,
) -> Tensor {
    Tensor::from((
        Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone())),
        shape,
    ))
}

/// Each last-axis row times `scale[row % scale rows]` (when given, f32 `(scale rows, n)`), its L2 norm brought down
/// to at most `threshold` (the norm plus `eps` divides).
pub fn l2_clip(
    x: &Tensor,
    scale: Option<&Tensor>,
    threshold: f32,
    eps: f32,
) -> Result<Option<Tensor>> {
    let Device::Cuda(dev) = x.device() else {
        return Ok(None);
    };
    if !takes(x) || scale.is_some_and(|s| s.dtype() != DType::F32) {
        return Ok(None);
    }
    let x = x.contiguous()?;
    let n = x.dim(x.rank() - 1)?;
    let rows = x.elem_count() / n.max(1);
    let scale = scale.map(Tensor::contiguous).transpose()?;
    let scale_rows = match &scale {
        Some(s) if s.elem_count() % n == 0 => s.elem_count() / n,
        Some(_) => return Ok(None),
        None => 1,
    };
    let stream = dev.cuda_stream();
    // SAFETY: the kernel writes every element
    let mut out = unsafe { dev.alloc::<f32>(x.elem_count())? };
    {
        let (xg, xl) = x.storage_and_layout();
        let (xp, dtype, _x) = device_ptr(&xg, xl, &stream)?;
        let scale_guard = scale.as_ref().map(|s| s.storage_and_layout());
        let scale_ptr = scale_guard
            .as_ref()
            .map(|(g, l)| device_ptr(g, l, &stream))
            .transpose()?;
        let (op, _o) = out.device_ptr_mut(&stream);
        // SAFETY: the buffers cover rows x n (the scale its rows x n) on this stream
        status(
            unsafe {
                super::ffi::inkling_l2_clip(
                    xp as *const core::ffi::c_void,
                    dtype,
                    scale_ptr
                        .as_ref()
                        .map_or(std::ptr::null(), |(p, _, _)| *p as *const f32),
                    op as *mut f32,
                    i64_of(rows)?,
                    i32_of(n)?,
                    i32_of(scale_rows)?,
                    threshold,
                    eps,
                    stream.cu_stream() as i64,
                )
            },
            "L2 clip",
        )?;
    }
    Ok(Some(wrap(out, dev, x.shape().clone())))
}

/// Each last-axis row over `sqrt(mean(x^2) + eps)`.
pub fn unit_rms(x: &Tensor, eps: f32) -> Result<Option<Tensor>> {
    let Device::Cuda(dev) = x.device() else {
        return Ok(None);
    };
    if !takes(x) {
        return Ok(None);
    }
    let x = x.contiguous()?;
    let n = x.dim(x.rank() - 1)?;
    let rows = x.elem_count() / n.max(1);
    let stream = dev.cuda_stream();
    // SAFETY: the kernel writes every element
    let mut out = unsafe { dev.alloc::<f32>(x.elem_count())? };
    {
        let (xg, xl) = x.storage_and_layout();
        let (xp, dtype, _x) = device_ptr(&xg, xl, &stream)?;
        let (op, _o) = out.device_ptr_mut(&stream);
        // SAFETY: the buffers cover rows x n on this stream
        status(
            unsafe {
                super::ffi::inkling_unit_rms(
                    xp as *const core::ffi::c_void,
                    dtype,
                    op as *mut f32,
                    i64_of(rows)?,
                    i32_of(n)?,
                    eps,
                    stream.cu_stream() as i64,
                )
            },
            "unit RMS",
        )?;
    }
    Ok(Some(wrap(out, dev, x.shape().clone())))
}

/// The states (the blocks stacked `(s - 1, b, t, hidden)`, when any, then `current` `(b, t, hidden)`, one dtype)
/// unit-normed (`eps`), and each of `queries` (f32 `(q, hidden)`) softmax-attending over them (scores over
/// `sqrt(hidden)`): `(b, t, q, hidden)` in `out_dtype` (f32 or bf16), summed in f32.
pub fn depth_mix(
    blocks: Option<&Tensor>,
    current: &Tensor,
    queries: &Tensor,
    eps: f32,
    out_dtype: DType,
) -> Result<Option<Tensor>> {
    let Device::Cuda(dev) = current.device() else {
        return Ok(None);
    };
    let (b, t, hidden) = current.dims3()?;
    let q_count = queries.dim(0)?;
    let n_blocks = blocks.map_or(Ok(0), |blocks| blocks.dim(0))?;
    let n_states = n_blocks + 1;
    if !takes(current)
        || queries.dtype() != DType::F32
        || queries.dim(1)? != hidden
        || n_states > MAX_STATES
        || q_count > MAX_QUERIES
        || !matches!(out_dtype, DType::F32 | DType::BF16)
        || blocks.is_some_and(|blocks| {
            blocks.dtype() != current.dtype()
                || blocks.dims() != [n_blocks, b, t, hidden]
                || !blocks.is_contiguous()
        })
    {
        return Ok(None);
    }
    let current = current.contiguous()?;
    let queries = queries.contiguous()?;
    let rows = b * t;
    let stream = dev.cuda_stream();
    let (cg, cl) = current.storage_and_layout();
    let (cp, dtype, _c) = device_ptr(&cg, cl, &stream)?;
    let block_guard = blocks.as_ref().map(|b| b.storage_and_layout());
    let block_ptr = block_guard
        .as_ref()
        .map(|(g, l)| device_ptr(g, l, &stream))
        .transpose()?;
    let (qg, ql) = queries.storage_and_layout();
    let (qp, _, _q) = device_ptr(&qg, ql, &stream)?;
    let shape = Shape::from((b, t, q_count, hidden));
    // SAFETY: a pure size computation
    let scratch = unsafe { super::ffi::inkling_depth_mix_scratch(i64_of(rows)?, i32_of(hidden)?) };
    // SAFETY: the kernels write every partial before reading it
    let scratch = usize::try_from(scratch).map_err(inference_tensor::Error::wrap)?;
    let mut partials = (scratch > 0)
        .then(|| unsafe { dev.alloc::<f32>(scratch) })
        .transpose()?;
    let partials_guard = partials.as_mut().map(|p| p.device_ptr_mut(&stream));
    let partials_ptr = partials_guard
        .as_ref()
        .map_or(std::ptr::null_mut(), |(p, _)| *p as *mut f32);
    macro_rules! run {
        ($ty:ty, $code:expr) => {{
            // SAFETY: the kernel writes every element
            let mut out = unsafe { dev.alloc::<$ty>(rows * q_count * hidden)? };
            {
                let (op, _o) = out.device_ptr_mut(&stream);
                // SAFETY: the buffers cover the states, queries and output shapes on this stream
                status(
                    unsafe {
                        super::ffi::inkling_depth_mix(
                            block_ptr.as_ref().map_or(std::ptr::null(), |(p, _, _)| {
                                *p as *const core::ffi::c_void
                            }),
                            cp as *const core::ffi::c_void,
                            dtype,
                            qp as *const f32,
                            op as *mut core::ffi::c_void,
                            $code,
                            partials_ptr,
                            i32_of(n_states)?,
                            i64_of(rows)?,
                            i32_of(q_count)?,
                            i32_of(hidden)?,
                            eps,
                            stream.cu_stream() as i64,
                        )
                    },
                    "depth mix",
                )?;
            }
            Tensor::from((
                Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone())),
                shape,
            ))
        }};
    }
    Ok(Some(if out_dtype == DType::F32 {
        run!(f32, F32_CODE)
    } else {
        run!(half::bf16, BF16_CODE)
    }))
}

/// `x` `(b, t, c)` through a causal depthwise convolution (f32 `weight` `(c, width)`, the last tap on the current
/// input) with a residual, its last `width - 1` inputs carried in `state` `(b, c, width - 1)` (a narrowed view, its
/// channels' rows contiguous); `taps` (f32 `(b, t, heads, width)`, `heads`) adds a weight a lag shared within a head.
/// The sum over `sqrt(terms)`, summed in f32 and written in `x`'s dtype when `out_like_x` (else f32), and the new
/// state `(b, c, width - 1)` f32.
pub fn short_conv(
    x: &Tensor,
    state: &Tensor,
    weight: &Tensor,
    taps: Option<(&Tensor, usize)>,
    out_like_x: bool,
) -> Result<Option<(Tensor, Tensor)>> {
    let Device::Cuda(dev) = x.device() else {
        return Ok(None);
    };
    let (b, t, c) = x.dims3()?;
    let width = weight.dim(1)?;
    let state_l = state.layout();
    let stride = state_l.stride();
    if !takes(x)
        || !takes(state)
        || weight.dtype() != DType::F32
        || width < 2
        || state.dims3()? != (b, c, width - 1)
        || stride[2] != 1
        || stride[1] != width - 1
        || taps.is_some_and(|(taps, heads)| {
            taps.dtype() != DType::F32 || heads == 0 || !c.is_multiple_of(heads)
        })
    {
        return Ok(None);
    }
    let state_batch = stride[0];
    let x = x.contiguous()?;
    let weight = weight.contiguous()?;
    let taps = taps
        .map(|(taps, heads)| -> Result<_> { Ok((taps.contiguous()?, heads)) })
        .transpose()?;
    let stream = dev.cuda_stream();
    // SAFETY: the kernel writes every element
    let mut new_state = unsafe { dev.alloc::<f32>(b * c * (width - 1))? };
    let (xg, xl) = x.storage_and_layout();
    let (xp, x_dtype, _x) = device_ptr(&xg, xl, &stream)?;
    let (sg, sl) = state.storage_and_layout();
    let (sp, s_dtype, _s) = device_ptr(&sg, sl, &stream)?;
    let (wg, wl) = weight.storage_and_layout();
    let (wp, _, _w) = device_ptr(&wg, wl, &stream)?;
    let tap_guard = taps.as_ref().map(|(t, _)| t.storage_and_layout());
    let tap_ptr = tap_guard
        .as_ref()
        .map(|(g, l)| device_ptr(g, l, &stream))
        .transpose()?;
    let heads = taps.as_ref().map_or(1, |(_, h)| *h);
    let shape = Shape::from((b, t, c));
    macro_rules! run {
        ($ty:ty) => {{
            // SAFETY: the kernel writes every element
            let mut out = unsafe { dev.alloc::<$ty>(b * t * c)? };
            {
                let (op, _o) = out.device_ptr_mut(&stream);
                let (np, _n) = new_state.device_ptr_mut(&stream);
                // SAFETY: the buffers cover the shapes and the state's strides checked above, on this stream
                status(
                    unsafe {
                        super::ffi::inkling_short_conv(
                            xp as *const core::ffi::c_void,
                            x_dtype,
                            sp as *const core::ffi::c_void,
                            s_dtype,
                            i64_of(state_batch)?,
                            wp as *const f32,
                            tap_ptr
                                .as_ref()
                                .map_or(std::ptr::null(), |(p, _, _)| *p as *const f32),
                            i32_of(heads)?,
                            op as *mut core::ffi::c_void,
                            i32::from(out_like_x),
                            np as *mut f32,
                            i32_of(b)?,
                            i32_of(t)?,
                            i32_of(c)?,
                            i32_of(width)?,
                            stream.cu_stream() as i64,
                        )
                    },
                    "short convolution",
                )?;
            }
            Tensor::from((
                Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone())),
                shape,
            ))
        }};
    }
    let out = if out_like_x && x_dtype == BF16_CODE {
        run!(half::bf16)
    } else {
        run!(f32)
    };
    Ok(Some((
        out,
        wrap(new_state, dev, Shape::from((b, c, width - 1))),
    )))
}

/// Each row's greedy pick under a recurrent Markov head: the argmax over `v < vocab` of
/// `logits[v] + fold_b[v] + fold_w[v] . latent` (`logits` `(b, >= vocab)` f32 or bf16 with its last axis contiguous,
/// `fold_w` `(>= vocab, rank)` BF16, `fold_b` F32, `latent` `(b, rank)` F32), as U32 `(b,)`; `None` for a layout the
/// kernel does not take.
pub fn markov_pick(
    logits: &Tensor,
    fold_w: &Tensor,
    fold_b: &Tensor,
    latent: &Tensor,
    vocab: usize,
) -> Result<Option<Tensor>> {
    let Device::Cuda(dev) = logits.device() else {
        return Ok(None);
    };
    let (b, width) = logits.dims2()?;
    let (rows, rank) = fold_w.dims2()?;
    let stride = logits.stride();
    if !takes(logits)
        || stride[1] != 1
        || fold_w.dtype() != DType::BF16
        || !fold_w.is_contiguous()
        || !fold_w.layout().start_offset().is_multiple_of(8)
        || !rank.is_multiple_of(8)
        || rows < vocab
        || width < vocab
        || fold_b.dtype() != DType::F32
        || fold_b.dims1()? < vocab
        || latent.dtype() != DType::F32
        || latent.dims2()? != (b, rank)
    {
        return Ok(None);
    }
    let fold_b = fold_b.contiguous()?;
    let latent = latent.contiguous()?;
    let stream = dev.cuda_stream();
    // SAFETY: a size query
    let parts = usize::try_from(unsafe { super::ffi::inkling_markov_pick_scratch(i32_of(b)?) })
        .map_err(inference_tensor::Error::wrap)?;
    // SAFETY: the partial kernel writes every partial before the final one reads them
    let mut part_v = unsafe { dev.alloc::<f32>(parts)? };
    let mut part_i = unsafe { dev.alloc::<i32>(parts)? };
    let mut out = unsafe { dev.alloc::<u32>(b)? };
    let (lg, ll) = logits.storage_and_layout();
    let (lp, l_dtype, _l) = device_ptr(&lg, ll, &stream)?;
    let (wg, wl) = fold_w.storage_and_layout();
    let (wp, _, _w) = device_ptr(&wg, wl, &stream)?;
    let (bg, bl) = fold_b.storage_and_layout();
    let (bp, _, _b) = device_ptr(&bg, bl, &stream)?;
    let (ag, al) = latent.storage_and_layout();
    let (ap, _, _a) = device_ptr(&ag, al, &stream)?;
    {
        let (vp, _v) = part_v.device_ptr_mut(&stream);
        let (ip, _i) = part_i.device_ptr_mut(&stream);
        let (op, _o) = out.device_ptr_mut(&stream);
        // SAFETY: the buffers cover the shapes checked above, on this stream
        status(
            unsafe {
                super::ffi::inkling_markov_pick(
                    lp as *const core::ffi::c_void,
                    l_dtype,
                    i64_of(stride[0])?,
                    wp as *const core::ffi::c_void,
                    bp as *const f32,
                    ap as *const f32,
                    i32_of(b)?,
                    i32_of(rank)?,
                    i32_of(vocab)?,
                    vp as *mut f32,
                    ip as *mut i32,
                    op as *mut u32,
                    stream.cu_stream() as i64,
                )
            },
            "Markov pick",
        )?;
    }
    Ok(Some(Tensor::from((
        Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone())),
        Shape::from(b),
    ))))
}

/// [`short_conv`] over a slot pool in place: one conv's channels of each batch row's slot.
pub struct ShortConvPool<'a> {
    // `(b, t, c)` f32 or bf16
    pub x: &'a Tensor,
    // `[pool rows, pool channels, width - 1]` F32, updated in place
    pub pool: &'a Tensor,
    // each batch row's slot `[b]` U32; a slot past the pool reads zeros and writes nothing
    pub slots: &'a Tensor,
    // this conv's first channel in the pool
    pub channel_off: usize,
    pub weight: &'a Tensor,
    pub taps: Option<(&'a Tensor, usize)>,
    pub out_like_x: bool,
    // a verify's rollback: the states as read `(b, pool channels, width - 1)` F32 and the inputs
    // `(b, t, pool channels)` in x's dtype, this conv's channels of each
    pub before: Option<&'a Tensor>,
    pub stash: Option<&'a Tensor>,
}

/// The convolution's output, its state updated in the pool; `None` where the layout is not the kernel's.
pub fn short_conv_pool(op: &ShortConvPool<'_>) -> Result<Option<Tensor>> {
    let Device::Cuda(dev) = op.x.device() else {
        return Ok(None);
    };
    let (b, t, c) = op.x.dims3()?;
    let width = op.weight.dim(1)?;
    let (pool_rows, pool_c, hist) = op.pool.dims3()?;
    // SAFETY: a constant query
    let max_hist = unsafe { super::ffi::inkling_short_conv_pool_max_hist() } as usize;
    if !takes(op.x)
        || op.pool.dtype() != DType::F32
        || !op.pool.is_contiguous()
        || op.slots.dtype() != DType::U32
        || op.slots.dims1()? != b
        || op.weight.dtype() != DType::F32
        || width < 2
        || hist != width - 1
        || hist > max_hist
        || op.channel_off + c > pool_c
        || op.taps.is_some_and(|(taps, heads)| {
            taps.dtype() != DType::F32 || heads == 0 || !c.is_multiple_of(heads)
        })
        || op.before.is_some_and(|before| {
            before.dtype() != DType::F32
                || !before.is_contiguous()
                || before.dims() != [b, pool_c, hist]
        })
        || op.stash.is_some_and(|stash| {
            stash.dtype() != op.x.dtype()
                || !stash.is_contiguous()
                || stash.dims() != [b, t, pool_c]
        })
    {
        return Ok(None);
    }
    let x = op.x.contiguous()?;
    let slots = op.slots.contiguous()?;
    let weight = op.weight.contiguous()?;
    let taps = op
        .taps
        .map(|(taps, heads)| -> Result<_> { Ok((taps.contiguous()?, heads)) })
        .transpose()?;
    let stream = dev.cuda_stream();
    let (xg, xl) = x.storage_and_layout();
    let (xp, x_dtype, _x) = device_ptr(&xg, xl, &stream)?;
    let (pg, pl) = op.pool.storage_and_layout();
    let (pp, _, _p) = device_ptr(&pg, pl, &stream)?;
    let (sg, sl) = slots.storage_and_layout();
    let Storage::Cuda(slot_storage) = &*sg else {
        inference_tensor::bail!("Inkling kernels need CUDA storage");
    };
    let (slot_base, _s) = slot_storage.as_cuda_slice::<u32>()?.device_ptr(&stream);
    let slot_ptr = slot_base + sl.start_offset() as u64 * 4;
    let (wg, wl) = weight.storage_and_layout();
    let (wp, _, _w) = device_ptr(&wg, wl, &stream)?;
    let tap_guard = taps.as_ref().map(|(t, _)| t.storage_and_layout());
    let tap_ptr = tap_guard
        .as_ref()
        .map(|(g, l)| device_ptr(g, l, &stream))
        .transpose()?;
    let heads = taps.as_ref().map_or(1, |(_, h)| *h);
    let before_guard = op.before.map(|t| t.storage_and_layout());
    let before_ptr = before_guard
        .as_ref()
        .map(|(g, l)| device_ptr(g, l, &stream))
        .transpose()?;
    let stash_guard = op.stash.map(|t| t.storage_and_layout());
    let stash_ptr = stash_guard
        .as_ref()
        .map(|(g, l)| device_ptr(g, l, &stream))
        .transpose()?;
    let shape = Shape::from((b, t, c));
    macro_rules! run {
        ($ty:ty) => {{
            // SAFETY: the kernel writes every element
            let mut out = unsafe { dev.alloc::<$ty>(b * t * c)? };
            {
                let (op_ptr, _o) = out.device_ptr_mut(&stream);
                // SAFETY: the buffers cover the shapes checked above, on this stream; the pool's rows are written
                // only by this launch's threads, one (row, channel) each
                status(
                    unsafe {
                        super::ffi::inkling_short_conv_pool(
                            xp as *const core::ffi::c_void,
                            x_dtype,
                            pp as *mut f32,
                            slot_ptr as *const u32,
                            i32_of(pool_rows)?,
                            i32_of(pool_c)?,
                            i32_of(op.channel_off)?,
                            wp as *const f32,
                            tap_ptr
                                .as_ref()
                                .map_or(std::ptr::null(), |(p, _, _)| *p as *const f32),
                            i32_of(heads)?,
                            op_ptr as *mut core::ffi::c_void,
                            i32::from(op.out_like_x),
                            before_ptr
                                .as_ref()
                                .map_or(std::ptr::null_mut(), |(p, _, _)| *p as *mut f32),
                            stash_ptr
                                .as_ref()
                                .map_or(std::ptr::null_mut(), |(p, _, _)| {
                                    *p as *mut core::ffi::c_void
                                }),
                            i32_of(b)?,
                            i32_of(t)?,
                            i32_of(c)?,
                            i32_of(width)?,
                            stream.cu_stream() as i64,
                        )
                    },
                    "short convolution over the pool",
                )?;
            }
            Tensor::from((
                Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone())),
                shape,
            ))
        }};
    }
    Ok(Some(if op.out_like_x && x_dtype == BF16_CODE {
        run!(half::bf16)
    } else {
        run!(f32)
    }))
}

/// A loop controller's input: `x` and `message` (`(.., hidden)`) each unit-normed (`eps`), side by side
/// on the last axis, f32.
pub fn controller_in(x: &Tensor, message: &Tensor, eps: f32) -> Result<Option<Tensor>> {
    let Device::Cuda(dev) = x.device() else {
        return Ok(None);
    };
    if !takes(x) || !takes(message) || message.shape() != x.shape() {
        return Ok(None);
    }
    let (x, message) = (x.contiguous()?, message.contiguous()?);
    let hidden = x.dim(x.rank() - 1)?;
    let rows = x.elem_count() / hidden.max(1);
    let mut dims = x.dims().to_vec();
    *dims.last_mut().expect("a controller input has a last axis") *= 2;
    let stream = dev.cuda_stream();
    // SAFETY: the kernel writes every element
    let mut out = unsafe { dev.alloc::<f32>(2 * x.elem_count())? };
    {
        let (xg, xl) = x.storage_and_layout();
        let (xp, x_dtype, _x) = device_ptr(&xg, xl, &stream)?;
        let (mg, ml) = message.storage_and_layout();
        let (mp, m_dtype, _m) = device_ptr(&mg, ml, &stream)?;
        let (op, _o) = out.device_ptr_mut(&stream);
        // SAFETY: the buffers cover rows x hidden (the output twice that) on this stream
        status(
            unsafe {
                super::ffi::inkling_controller_in(
                    xp as *const core::ffi::c_void,
                    x_dtype,
                    mp as *const core::ffi::c_void,
                    m_dtype,
                    op as *mut f32,
                    i64_of(rows)?,
                    i32_of(hidden)?,
                    eps,
                    stream.cu_stream() as i64,
                )
            },
            "controller input",
        )?;
    }
    Ok(Some(wrap(out, dev, Shape::from(dims))))
}

/// A whole loop controller over a few rows (decode, verify) in two launches: `x` and `message` unit-normed into f32
/// `down` `[rank, 2 hidden]`, the bottleneck through f32 `gain_forget` `[2 hidden, rank]` (the gain's rows over the
/// forget's), `(1 - tanh(forget)) x + tanh(gain) message` in `x`'s dtype. `None` past the rows or rank it fuses.
pub fn controller(
    x: &Tensor,
    message: &Tensor,
    down: &Tensor,
    gain_forget: &Tensor,
    eps: f32,
) -> Result<Option<Tensor>> {
    let Device::Cuda(dev) = x.device() else {
        return Ok(None);
    };
    let hidden = x.dim(x.rank() - 1)?;
    let rank = down.dim(0)?;
    let rows = x.elem_count() / hidden.max(1);
    // SAFETY: a pure size computation
    let scratch = unsafe { super::ffi::inkling_controller_scratch(i64_of(rows)?, i32_of(rank)?) };
    if scratch == 0
        || !takes(x)
        || !takes(message)
        || message.shape() != x.shape()
        || [down, gain_forget]
            .iter()
            .any(|w| w.dtype() != DType::F32 || !w.is_contiguous())
        || down.dims() != [rank, 2 * hidden]
        || gain_forget.dims() != [2 * hidden, rank]
    {
        return Ok(None);
    }
    let (x, message) = (x.contiguous()?, message.contiguous()?);
    let stream = dev.cuda_stream();
    let shape = x.shape().clone();
    let scratch = usize::try_from(scratch).map_err(inference_tensor::Error::wrap)?;
    // SAFETY: the first kernel writes every element before the second reads it
    let mut z = unsafe { dev.alloc::<f32>(scratch)? };
    let (xg, xl) = x.storage_and_layout();
    let (xp, dtype, _x) = device_ptr(&xg, xl, &stream)?;
    let (mg, ml) = message.storage_and_layout();
    let (mp, m_dtype, _m) = device_ptr(&mg, ml, &stream)?;
    let (dg, dl) = down.storage_and_layout();
    let (dp, _, _d) = device_ptr(&dg, dl, &stream)?;
    let (gg, gl) = gain_forget.storage_and_layout();
    let (gp, _, _g) = device_ptr(&gg, gl, &stream)?;
    let (zp, _z) = z.device_ptr_mut(&stream);
    macro_rules! run {
        ($ty:ty) => {{
            // SAFETY: the kernel writes every element
            let mut out = unsafe { dev.alloc::<$ty>(x.elem_count())? };
            {
                let (op, _o) = out.device_ptr_mut(&stream);
                // SAFETY: the buffers cover the rows, the weights their shapes and z the scratch, on this stream
                status(
                    unsafe {
                        super::ffi::inkling_controller(
                            xp as *const core::ffi::c_void,
                            dtype,
                            mp as *const core::ffi::c_void,
                            m_dtype,
                            dp as *const f32,
                            gp as *const f32,
                            zp as *mut f32,
                            op as *mut core::ffi::c_void,
                            i64_of(rows)?,
                            i32_of(hidden)?,
                            i32_of(rank)?,
                            eps,
                            stream.cu_stream() as i64,
                        )
                    },
                    "controller",
                )?;
            }
            Tensor::from((
                Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone())),
                shape,
            ))
        }};
    }
    Ok(Some(if dtype == F32_CODE {
        run!(f32)
    } else {
        run!(half::bf16)
    }))
}

/// A loop controller's output, `(1 - tanh(forget)) x + tanh(gain) message`, in `x`'s dtype (the message f32 or bf16) (`gain` and `forget` f32
/// of `x`'s shape).
pub fn controller_out(
    x: &Tensor,
    message: &Tensor,
    gain: &Tensor,
    forget: &Tensor,
) -> Result<Option<Tensor>> {
    let Device::Cuda(dev) = x.device() else {
        return Ok(None);
    };
    if !takes(x)
        || !takes(message)
        || gain.dtype() != DType::F32
        || forget.dtype() != DType::F32
        || [message, gain, forget]
            .iter()
            .any(|t| t.shape() != x.shape())
    {
        return Ok(None);
    }
    let (x, message, gain, forget) = (
        x.contiguous()?,
        message.contiguous()?,
        gain.contiguous()?,
        forget.contiguous()?,
    );
    let n = x.elem_count();
    let stream = dev.cuda_stream();
    let shape = x.shape().clone();
    let (xg, xl) = x.storage_and_layout();
    let (xp, dtype, _x) = device_ptr(&xg, xl, &stream)?;
    let (mg, ml) = message.storage_and_layout();
    let (mp, m_dtype, _m) = device_ptr(&mg, ml, &stream)?;
    let (gg, gl) = gain.storage_and_layout();
    let (gp, _, _g) = device_ptr(&gg, gl, &stream)?;
    let (fg, fl) = forget.storage_and_layout();
    let (fp, _, _f) = device_ptr(&fg, fl, &stream)?;
    macro_rules! run {
        ($ty:ty) => {{
            // SAFETY: the kernel writes every element
            let mut out = unsafe { dev.alloc::<$ty>(n)? };
            {
                let (op, _o) = out.device_ptr_mut(&stream);
                // SAFETY: every buffer covers n elements on this stream
                status(
                    unsafe {
                        super::ffi::inkling_controller_out(
                            xp as *const core::ffi::c_void,
                            dtype,
                            mp as *const core::ffi::c_void,
                            m_dtype,
                            gp as *const f32,
                            fp as *const f32,
                            op as *mut core::ffi::c_void,
                            i64_of(n)?,
                            stream.cu_stream() as i64,
                        )
                    },
                    "controller output",
                )?;
            }
            Tensor::from((
                Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone())),
                shape,
            ))
        }};
    }
    Ok(Some(if dtype == F32_CODE {
        run!(f32)
    } else {
        run!(half::bf16)
    }))
}

// a view as (rows, cols) rows apart by its returned stride: the last axis contiguous, the rows evenly spaced
fn row_view(x: &Tensor) -> Option<(usize, usize, usize)> {
    let dims = x.dims();
    let stride = x.layout().stride();
    let cols = *dims.last()?;
    if stride.last() != Some(&1) {
        return None;
    }
    if dims.len() == 1 {
        return Some((1, cols, cols));
    }
    let row = stride[dims.len() - 2];
    // the leading axes merge into rows when each steps over the ones after it
    for i in 0..dims.len() - 2 {
        if dims[i] > 1 && stride[i] != stride[i + 1] * dims[i + 1] {
            return None;
        }
    }
    Some((x.elem_count() / cols.max(1), cols, row))
}

/// The softcapped SwiGLU, `gate tanh(a / gate) sigmoid(a) * up tanh(b / up)`, in `a`'s dtype (a and b of one shape
/// and dtype, f32 or bf16; each row-strided, such as two halves of one projection).
pub fn softcap_silu_mul(a: &Tensor, b: &Tensor, gate: f32, up: f32) -> Result<Option<Tensor>> {
    let Device::Cuda(dev) = a.device() else {
        return Ok(None);
    };
    if !takes(a) || b.dtype() != a.dtype() || b.shape() != a.shape() {
        return Ok(None);
    }
    let (Some((rows, cols, a_row)), Some((_, _, b_row))) = (row_view(a), row_view(b)) else {
        return Ok(None);
    };
    let stream = dev.cuda_stream();
    let shape = a.shape().clone();
    let (ag, al) = a.storage_and_layout();
    let (ap, dtype, _a) = device_ptr(&ag, al, &stream)?;
    let (bg, bl) = b.storage_and_layout();
    let (bp, _, _b) = device_ptr(&bg, bl, &stream)?;
    macro_rules! run {
        ($ty:ty) => {{
            // SAFETY: the kernel writes every element
            let mut out = unsafe { dev.alloc::<$ty>(rows * cols)? };
            {
                let (op, _o) = out.device_ptr_mut(&stream);
                // SAFETY: a and b cover rows of cols at their strides, the output rows x cols, on this stream
                status(
                    unsafe {
                        super::ffi::inkling_softcap_silu_mul(
                            ap as *const core::ffi::c_void,
                            i64_of(a_row)?,
                            bp as *const core::ffi::c_void,
                            i64_of(b_row)?,
                            dtype,
                            op as *mut core::ffi::c_void,
                            i64_of(rows)?,
                            i32_of(cols)?,
                            gate,
                            up,
                            stream.cu_stream() as i64,
                        )
                    },
                    "softcapped SwiGLU",
                )?;
            }
            Tensor::from((
                Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone())),
                shape,
            ))
        }};
    }
    Ok(Some(if dtype == F32_CODE {
        run!(f32)
    } else {
        run!(half::bf16)
    }))
}

/// `x` (f32) times `sigmoid(gate)` (f32 or bf16, `x`'s shape), in `out_dtype` (f32 or bf16).
pub fn sigmoid_gate(x: &Tensor, gate: &Tensor, out_dtype: DType) -> Result<Option<Tensor>> {
    let Device::Cuda(dev) = x.device() else {
        return Ok(None);
    };
    if x.dtype() != DType::F32
        || !takes(gate)
        || gate.shape() != x.shape()
        || !matches!(out_dtype, DType::F32 | DType::BF16)
    {
        return Ok(None);
    }
    let (x, gate) = (x.contiguous()?, gate.contiguous()?);
    let n = x.elem_count();
    let stream = dev.cuda_stream();
    let shape = x.shape().clone();
    let (xg, xl) = x.storage_and_layout();
    let (xp, _, _x) = device_ptr(&xg, xl, &stream)?;
    let (gg, gl) = gate.storage_and_layout();
    let (gp, g_dtype, _g) = device_ptr(&gg, gl, &stream)?;
    macro_rules! run {
        ($ty:ty, $code:expr) => {{
            // SAFETY: the kernel writes every element
            let mut out = unsafe { dev.alloc::<$ty>(n)? };
            {
                let (op, _o) = out.device_ptr_mut(&stream);
                // SAFETY: every buffer covers n elements on this stream
                status(
                    unsafe {
                        super::ffi::inkling_sigmoid_gate(
                            xp as *const f32,
                            gp as *const core::ffi::c_void,
                            g_dtype,
                            op as *mut core::ffi::c_void,
                            $code,
                            i64_of(n)?,
                            stream.cu_stream() as i64,
                        )
                    },
                    "sigmoid gate",
                )?;
            }
            Tensor::from((
                Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone())),
                shape,
            ))
        }};
    }
    Ok(Some(if out_dtype == DType::F32 {
        run!(f32, F32_CODE)
    } else {
        run!(half::bf16, BF16_CODE)
    }))
}

/// One cache slot of a verify step's convolution rollback (see [`conv_rollback`]).
pub struct ConvRollbackSlot<'a> {
    // `[slots, c, hist]` F32
    pub pool: &'a Tensor,
    // the batch rows' pool rows `[b]` U32
    pub rows: &'a Tensor,
    // `[b, c, hist]` F32 and `[b, t, c]` F32 or BF16
    pub state: &'a Tensor,
    pub inputs: &'a Tensor,
}

/// Every slot's pool rows rolled back to the window of `[state, inputs]` ending at each batch row's last kept row
/// (`keep[row]` rows kept; negative leaves the row), one launch; `false` where a slot's layout is not the kernel's.
pub fn conv_rollback(slots: &[ConvRollbackSlot<'_>], keep: &[i32]) -> Result<bool> {
    let Some(first) = slots.first() else {
        return Ok(true);
    };
    let Device::Cuda(dev) = first.pool.device() else {
        return Ok(false);
    };
    let (b, t, c) = first.inputs.dims3()?;
    let hist = first.state.dim(2)?;
    let inputs_dtype = first.inputs.dtype();
    let fits = |s: &ConvRollbackSlot<'_>| -> Result<bool> {
        Ok(s.pool.dtype() == DType::F32
            && s.pool.is_contiguous()
            && s.pool.dims3()?.1 == c
            && s.pool.dims3()?.2 == hist
            && s.rows.dtype() == DType::U32
            && s.rows.is_contiguous()
            && s.state.dtype() == DType::F32
            && s.state.is_contiguous()
            && s.state.dims3()? == (b, c, hist)
            && s.inputs.dtype() == inputs_dtype
            && matches!(inputs_dtype, DType::F32 | DType::BF16)
            && s.inputs.is_contiguous()
            && s.inputs.dims3()? == (b, t, c))
    };
    for slot in slots {
        if !fits(slot)? {
            return Ok(false);
        }
    }
    if keep.len() != b {
        inference_tensor::bail!("conv rollback keeps {} rows of {b}", keep.len());
    }
    let stream = dev.cuda_stream();
    let storages = slots
        .iter()
        .map(|s| [s.pool, s.rows, s.state, s.inputs].map(|t| t.storage_and_layout()))
        .collect::<Vec<_>>();
    let mut guards = Vec::with_capacity(slots.len() * 4);
    let mut table = Vec::with_capacity(slots.len() * 4);

    for parts in &storages {
        for (storage, layout) in parts {
            let Storage::Cuda(cuda) = &**storage else {
                return Ok(false);
            };
            let (ptr, guard) = match cuda.dtype() {
                DType::F32 => {
                    let (p, g) = cuda.as_cuda_slice::<f32>()?.device_ptr(&stream);
                    (p + layout.start_offset() as u64 * 4, g)
                }
                DType::U32 => {
                    let (p, g) = cuda.as_cuda_slice::<u32>()?.device_ptr(&stream);
                    (p + layout.start_offset() as u64 * 4, g)
                }
                DType::BF16 => {
                    let (p, g) = cuda.as_cuda_slice::<half::bf16>()?.device_ptr(&stream);
                    (p + layout.start_offset() as u64 * 2, g)
                }
                _ => return Ok(false),
            };
            table.push(ptr);
            guards.push(guard);
        }
    }
    let table = stream
        .clone_htod(&table)
        .map_err(inference_tensor::Error::wrap)?;
    let keep = stream
        .clone_htod(keep)
        .map_err(inference_tensor::Error::wrap)?;
    let (tp, _t) = table.device_ptr(&stream);
    let (kp, _k) = keep.device_ptr(&stream);
    // SAFETY: the table holds each slot's pool, rows, state and inputs, checked above, alive until the launch is queued
    status(
        unsafe {
            super::ffi::inkling_conv_rollback(
                tp as *const core::ffi::c_void,
                kp as *const i32,
                i32_of(slots.len())?,
                i32_of(b)?,
                i32_of(t)?,
                i32_of(c)?,
                i32_of(hist)?,
                i32::from(inputs_dtype == DType::BF16),
                stream.cu_stream() as i64,
            )
        },
        "conv rollback",
    )?;
    drop(guards);
    Ok(true)
}

#[cfg(test)]
mod tests {
    use inference_tensor::{D, DType, Device, IndexOp, Result, Tensor};

    const HIDDEN: usize = 576;
    // sqrt(HIDDEN), the scores' scale
    const HIDDEN_SQRT: f64 = 24.0;
    const EPS: f32 = 1e-6;
    const TOLERANCE: f32 = 1e-4;

    // the mix on the CPU: each state unit-normed, each query's softmax over the states' scores
    fn reference(states: &[Tensor], queries: &Tensor) -> Result<Tensor> {
        let stacked = Tensor::stack(states, 2)?;
        let rms = (stacked.sqr()?.mean_keepdim(D::Minus1)? + f64::from(EPS))?.sqrt()?;
        let normed = stacked.broadcast_div(&rms)?;
        let (b, t, s, h) = normed.dims4()?;
        let flat = normed.reshape((b * t, s, h))?;
        let scores = (flat.broadcast_matmul(&queries.t()?)? / HIDDEN_SQRT)?;
        let weights = inference_tensor::nn::ops::softmax(&scores, 1)?;
        weights
            .transpose(1, 2)?
            .contiguous()?
            .matmul(&flat)?
            .reshape((b, t, queries.dim(0)?, h))
    }

    // a decode's few rows split over blocks, a prefill's many one block a row: both give the CPU's mix
    fn cuda_depth_mix_matches_the_reference() -> Result<()> {
        let Ok(cuda) = Device::new_cuda(0) else {
            return Ok(());
        };
        for (rows, n_states, q_count) in [(1, 13, 3), (5, 2, 1), (40, 7, 4)] {
            let states = (0..n_states)
                .map(|_| Tensor::randn(0f32, 1f32, (1, rows, HIDDEN), &Device::Cpu))
                .collect::<Result<Vec<_>>>()?;
            let queries = Tensor::randn(0f32, 0.2f32, (q_count, HIDDEN), &Device::Cpu)?;
            let expected = reference(&states, &queries)?;
            let blocks = Tensor::stack(&states[..n_states - 1], 0)?.to_device(&cuda)?;
            let current = states[n_states - 1].to_device(&cuda)?;
            let mixed = super::depth_mix(
                (n_states > 1).then_some(&blocks),
                &current,
                &queries.to_device(&cuda)?,
                EPS,
                DType::F32,
            )?
            .expect("the kernel takes f32 states");
            let error = (mixed.to_device(&Device::Cpu)? - &expected)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            assert!(
                error < TOLERANCE,
                "{rows} rows of {n_states} states: error {error}"
            );
        }
        Ok(())
    }

    // the fused pick against the bias through a matmul, the sum's argmax over the unpadded vocabulary; logits a
    // strided step of `[b, steps, padded]`, and a padded row larger than every real one
    fn cuda_markov_pick_takes_the_biased_argmax() -> Result<()> {
        const B: usize = 3;
        const STEPS: usize = 2;
        const VOCAB: usize = 5000;
        const PADDED: usize = 5056;
        const RANK: usize = 64;
        let Ok(cuda) = Device::new_cuda(0) else {
            return Ok(());
        };
        let fold_w = Tensor::randn(0f32, 1f32, (PADDED, RANK), &cuda)?.to_dtype(DType::BF16)?;
        let fold_b = Tensor::randn(0f32, 1f32, PADDED, &cuda)?;
        let latent = Tensor::randn(0f32, 1f32, (B, RANK), &cuda)?;
        let padding = Tensor::full(1e4f32, (B, STEPS, PADDED - VOCAB), &cuda)?;
        for dtype in [DType::F32, DType::BF16] {
            let logits = Tensor::cat(
                &[
                    &Tensor::randn(0f32, 4f32, (B, STEPS, VOCAB), &cuda)?,
                    &padding,
                ],
                2,
            )?
            .to_dtype(dtype)?;
            let step = logits.i((.., 1))?;
            let bias = latent
                .matmul(&fold_w.to_dtype(DType::F32)?.t()?)?
                .broadcast_add(&fold_b)?;
            let expected = (step.to_dtype(DType::F32)? + bias)?
                .narrow(1, 0, VOCAB)?
                .argmax(D::Minus1)?
                .to_vec1::<u32>()?;
            let picked = super::markov_pick(&step, &fold_w, &fold_b, &latent, VOCAB)?
                .expect("the kernel takes the layout")
                .to_vec1::<u32>()?;
            assert_eq!(picked, expected, "{dtype:?}");
        }
        Ok(())
    }

    // the in-place pool conv against the gathered rows' kernel: outputs, the slots' new states, the states as read and
    // the inputs it stashes, a padded row's slot left; with and without taps, a decode's row and a verify's
    fn cuda_pool_conv_matches_the_gathered_rows() -> Result<()> {
        const C: usize = 48;
        const POOL_C: usize = 80;
        const OFF: usize = 16;
        const WIDTH: usize = 4;
        const HIST: usize = WIDTH - 1;
        const POOL_ROWS: usize = 5;
        const HEADS: usize = 4;
        // past the pool: a padded row
        const PAD: u32 = u32::MAX;
        let Ok(cuda) = Device::new_cuda(0) else {
            return Ok(());
        };
        let host_slots = [3u32, PAD, 1];
        let b = host_slots.len();
        for (t, with_taps) in [(1, false), (1, true), (4, true)] {
            let pool = Tensor::randn(0f32, 1f32, (POOL_ROWS, POOL_C, HIST), &cuda)?;
            let x = Tensor::randn(0f32, 1f32, (b, t, C), &cuda)?.to_dtype(DType::BF16)?;
            let weight = Tensor::randn(0f32, 0.5f32, (C, WIDTH), &cuda)?;
            let taps = Tensor::randn(0f32, 0.5f32, (b, t, HEADS, WIDTH), &cuda)?;
            let taps = with_taps.then_some((&taps, HEADS));
            let state = Tensor::stack(
                &host_slots
                    .iter()
                    .map(|&s| match s {
                        PAD => Tensor::zeros((C, HIST), DType::F32, &cuda),
                        s => pool.get(s as usize)?.narrow(0, OFF, C),
                    })
                    .collect::<Result<Vec<_>>>()?,
                0,
            )?;
            let (expected, new_state) =
                super::short_conv(&x, &state, &weight, taps, true)?.expect("the rows kernel");
            let in_place = pool.copy()?;
            let before = Tensor::zeros((b, POOL_C, HIST), DType::F32, &cuda)?;
            let stash = Tensor::zeros((b, t, POOL_C), DType::BF16, &cuda)?;
            let out = super::short_conv_pool(&super::ShortConvPool {
                x: &x,
                pool: &in_place,
                slots: &Tensor::new(&host_slots, &cuda)?,
                channel_off: OFF,
                weight: &weight,
                taps,
                out_like_x: true,
                before: Some(&before),
                stash: Some(&stash),
            })?
            .expect("the pool kernel");
            let max = |a: &Tensor, e: &Tensor| -> Result<f32> {
                (a.to_dtype(DType::F32)? - e.to_dtype(DType::F32)?)?
                    .abs()?
                    .max_all()?
                    .to_scalar::<f32>()
            };
            let label = format!("t {t} taps {with_taps}");
            assert!(max(&out, &expected)? < TOLERANCE, "{label}: out");
            assert_eq!(max(&stash.narrow(2, OFF, C)?, &x)?, 0.0, "{label}: stash");
            for (bi, &s) in host_slots.iter().enumerate() {
                if s == PAD {
                    continue;
                }
                let row = in_place.get(s as usize)?;
                assert!(
                    max(&row.narrow(0, OFF, C)?, &new_state.get(bi)?)? < TOLERANCE,
                    "{label}: state"
                );
                assert_eq!(
                    max(&before.get(bi)?.narrow(0, OFF, C)?, &state.get(bi)?)?,
                    0.0,
                    "{label}: before"
                );
                let others = |p: &Tensor| -> Result<Tensor> {
                    Tensor::cat(
                        &[
                            p.narrow(0, 0, OFF)?,
                            p.narrow(0, OFF + C, POOL_C - OFF - C)?,
                        ],
                        0,
                    )
                };
                assert_eq!(
                    max(&others(&row)?, &others(&pool.get(s as usize)?)?)?,
                    0.0,
                    "{label}: others"
                );
            }
            let untouched = [0usize, 2, 4];
            for r in untouched {
                assert_eq!(
                    max(&in_place.get(r)?, &pool.get(r)?)?,
                    0.0,
                    "{label}: row {r}"
                );
            }
        }
        Ok(())
    }

    // each kept row's pool row is the window of `[state, inputs]` ending at it, from a BF16 stash as from its F32
    // widening; a row that kept everything is left
    fn cuda_conv_rollback_takes_either_stash_dtype() -> Result<()> {
        const B: usize = 3;
        const T: usize = 4;
        const C: usize = 40;
        const HIST: usize = 3;
        const POOL_ROWS: usize = 6;
        let Ok(cuda) = Device::new_cuda(0) else {
            return Ok(());
        };
        let state = Tensor::randn(0f32, 1f32, (B, C, HIST), &cuda)?;
        let inputs = Tensor::randn(0f32, 1f32, (B, T, C), &cuda)?.to_dtype(DType::BF16)?;
        let rows = Tensor::new(&[4u32, 0, 2], &cuda)?;
        let keep = [1i32, -1, 3];
        let window = Tensor::cat(&[&state.transpose(1, 2)?, &inputs.to_dtype(DType::F32)?], 1)?;
        let mut pools = Vec::new();
        for stash in [inputs.clone(), inputs.to_dtype(DType::F32)?] {
            let pool = Tensor::zeros((POOL_ROWS, C, HIST), DType::F32, &cuda)?;
            let slot = super::ConvRollbackSlot {
                pool: &pool,
                rows: &rows,
                state: &state,
                inputs: &stash,
            };
            assert!(super::conv_rollback(&[slot], &keep)?);
            pools.push(pool.to_device(&Device::Cpu)?);
        }
        for (bi, (&row, &kept)) in [4usize, 0, 2].iter().zip(&keep).enumerate() {
            let expected = if kept < 0 {
                Tensor::zeros((C, HIST), DType::F32, &Device::Cpu)?
            } else {
                window
                    .get(bi)?
                    .narrow(0, kept as usize, HIST)?
                    .t()?
                    .to_device(&Device::Cpu)?
            };
            for pool in &pools {
                let error = (pool.get(row)? - &expected)?
                    .abs()?
                    .max_all()?
                    .to_scalar::<f32>()?;
                assert!(error == 0.0, "row {bi}: error {error}");
            }
        }
        Ok(())
    }

    // the fused controller of a few rows: the unit-normed pair through the bottleneck, the gated sum, in x's dtype
    fn cuda_fused_controller_matches_the_reference() -> Result<()> {
        const RANK: usize = 24;
        let Ok(cuda) = Device::new_cuda(0) else {
            return Ok(());
        };
        let unit = |t: &Tensor| -> Result<Tensor> {
            let rms = (t.sqr()?.mean_keepdim(D::Minus1)? + f64::from(EPS))?.sqrt()?;
            t.broadcast_div(&rms)
        };
        // a BF16 output within two of its steps (2^-8 of the value each) of the f32 reference, past 1e-4
        const BF16_ROUNDING: f64 = 2.0 / 256.0;
        for (rows, dtype, relative) in [
            (1, DType::F32, 0.0),
            (4, DType::BF16, BF16_ROUNDING),
            (7, DType::F32, 0.0),
        ] {
            let x = Tensor::randn(0f32, 1f32, (1, rows, HIDDEN), &Device::Cpu)?.to_dtype(dtype)?;
            let message = Tensor::randn(0f32, 1f32, (1, rows, HIDDEN), &Device::Cpu)?;
            let down = Tensor::randn(0f32, 0.05f32, (RANK, 2 * HIDDEN), &Device::Cpu)?;
            let gain_forget = Tensor::randn(0f32, 0.3f32, (2 * HIDDEN, RANK), &Device::Cpu)?;
            let xf = x.to_dtype(DType::F32)?;
            let gate = Tensor::cat(&[&unit(&xf)?, &unit(&message)?], D::Minus1)?;
            let both = gate
                .broadcast_matmul(&down.t()?)?
                .broadcast_matmul(&gain_forget.t()?)?;
            let gain = both.narrow(D::Minus1, 0, HIDDEN)?.tanh()?;
            let forget = both.narrow(D::Minus1, HIDDEN, HIDDEN)?.tanh()?;
            let expected = ((&xf - (forget * &xf)?)? + (gain * &message)?)?;
            let fused = super::controller(
                &x.to_device(&cuda)?,
                &message.to_device(&cuda)?,
                &down.to_device(&cuda)?,
                &gain_forget.to_device(&cuda)?,
                EPS,
            )?
            .expect("a few rows of a narrow rank fuse");
            assert_eq!(fused.dtype(), dtype);
            let error = ((fused.to_device(&Device::Cpu)?.to_dtype(DType::F32)? - &expected)?
                .abs()?
                - (expected.abs()? * relative)?)?
                .max_all()?
                .to_scalar::<f32>()?;
            assert!(
                error < 1e-4,
                "{rows} rows {dtype:?}: error {error} past the rounding"
            );
        }
        Ok(())
    }

    inference_tensor::gpu_cases!(
        cuda_cases,
        [
            cuda_depth_mix_matches_the_reference,
            cuda_fused_controller_matches_the_reference,
            cuda_conv_rollback_takes_either_stash_dtype,
            cuda_pool_conv_matches_the_gathered_rows,
            cuda_markov_pick_takes_the_biased_argmax
        ]
    );
}
