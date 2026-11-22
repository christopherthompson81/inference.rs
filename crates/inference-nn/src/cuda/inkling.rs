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
    // `[b, c, hist]` and `[b, t, c]` F32
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
            && s.inputs.dtype() == DType::F32
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
    use inference_tensor::{D, DType, Device, Result, Tensor};

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
        for (rows, dtype, tolerance) in [(1, DType::F32, 1e-4f32), (4, DType::BF16, 2e-2)] {
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
            let error = (fused.to_device(&Device::Cpu)?.to_dtype(DType::F32)? - &expected)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            assert!(error < tolerance, "{rows} rows {dtype:?}: error {error}");
        }
        Ok(())
    }

    inference_tensor::gpu_cases!(
        cuda_cases,
        [
            cuda_depth_mix_matches_the_reference,
            cuda_fused_controller_matches_the_reference
        ]
    );
}
