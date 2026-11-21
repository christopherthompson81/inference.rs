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
