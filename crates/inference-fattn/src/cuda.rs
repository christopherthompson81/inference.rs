use std::sync::Arc;

use candle_core::cuda_backend::cudarc::driver::{CudaStream, DevicePtr, DevicePtrMut};
use candle_core::{
    CpuStorage, CudaStorage, DType, Device, Layout, Result, Shape, Storage, Tensor,
    backend::BackendStorage,
};

use crate::FattnOptions;

// ggml_type ids
const GGML_TYPE_F32: i32 = 0;
const GGML_TYPE_F16: i32 = 1;
const GGML_TYPE_BF16: i32 = 30;
// GGML_CUDA_MAX_DEVICES in ggml-cuda.h; fattn keeps per-device state in arrays of this size
const MAX_DEVICES: usize = 16;
// The MLA head dim: fattn reads V out of K's tiles for it (`V_is_K_view = DKQ == 576`, fattn-mma-f16.cuh)
const MLA_HEAD_DIM: usize = 576;

mod ffi {
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct Tensor {
        pub data: *const core::ffi::c_void,
        pub ty: i32,
        pub ne: [i64; 4],
        pub nb: [i64; 4],
    }

    #[repr(C)]
    pub struct Args {
        pub q: Tensor,
        pub k: Tensor,
        pub v: Tensor,
        pub mask: Tensor,
        pub sinks: Tensor,
        pub dst: *mut core::ffi::c_void,
        pub scale: f32,
        pub max_bias: f32,
        pub softcap: f32,
        pub device: i32,
        pub stream: *mut core::ffi::c_void,
    }

    unsafe extern "C" {
        pub fn inference_fattn_supported(args: *const Args) -> bool;
        pub fn inference_fattn_alloc_size(args: *const Args) -> usize;
        pub fn inference_fattn_forward(args: *const Args) -> i32;
    }
}

const ABSENT: ffi::Tensor = ffi::Tensor {
    data: std::ptr::null(),
    ty: GGML_TYPE_F32,
    ne: [1; 4],
    nb: [0; 4],
};

// cudarc's device-pointer guards record stream use on drop, so they stay alive until the launch is enqueued.
trait Held {}
impl<T> Held for T {}
type Guards<'a> = Vec<Box<dyn Held + 'a>>;

fn ggml_type(dtype: DType) -> Result<i32> {
    Ok(match dtype {
        DType::F32 => GGML_TYPE_F32,
        DType::F16 => GGML_TYPE_F16,
        DType::BF16 => GGML_TYPE_BF16,
        dt => candle_core::bail!("fattn does not take {dt:?}"),
    })
}

// fattn's process-aborting GGML_ASSERT checks become errors here: operands are checked before any descriptor is built.
fn validate(q: &Tensor, k: &Tensor, v: &Tensor, opts: &FattnOptions) -> Result<()> {
    let (b, _, h, d_qk) = q.dims4()?;
    let (kb, s_kv, h_kv, kd) = k.dims4()?;
    let (vb, vs, vh, _) = v.dims4()?;
    if kd != d_qk || (kb, s_kv, h_kv) != (vb, vs, vh) || kb != b {
        candle_core::bail!(
            "fattn operands disagree: q {:?} k {:?} v {:?}",
            q.shape(),
            k.shape(),
            v.shape()
        );
    }
    if d_qk == MLA_HEAD_DIM {
        let (ks, kl) = k.storage_and_layout();
        let (vs, vl) = v.storage_and_layout();
        let aliases = std::ptr::eq(&*ks, &*vs)
            && kl.start_offset() == vl.start_offset()
            && kl.stride()[..3] == vl.stride()[..3];
        if !aliases {
            candle_core::bail!(
                "fattn at head dim {MLA_HEAD_DIM} needs v to be a view of k's leading dims"
            );
        }
    }
    if h_kv == 0 || h % h_kv != 0 {
        candle_core::bail!("fattn needs n_head ({h}) to be a multiple of n_head_kv ({h_kv})");
    }
    if !matches!(k.dtype(), DType::F16 | DType::BF16) || k.dtype() != v.dtype() {
        candle_core::bail!(
            "fattn takes f16 or bf16 K and V of one dtype, got {:?} and {:?}",
            k.dtype(),
            v.dtype()
        );
    }
    if let Some(mask) = &opts.mask {
        let (mb, mq, mkv) = mask.dims3()?;
        if mask.dtype() != DType::F16 || mask.layout().stride()[2] != 1 {
            candle_core::bail!(
                "fattn needs an f16 mask with a contiguous last dim, got {:?}",
                mask.dtype()
            );
        }
        if (mq, mkv) != (q.dim(1)?, s_kv) || mb == 0 || b % mb != 0 {
            candle_core::bail!(
                "fattn mask {:?} does not fit q {:?} and k {:?}",
                mask.shape(),
                q.shape(),
                k.shape()
            );
        }
    }
    if let Some(sinks) = &opts.sinks
        && (sinks.dtype() != DType::F32 || sinks.dims1()? != h || !sinks.is_contiguous())
    {
        candle_core::bail!(
            "fattn needs contiguous f32 sinks of length {h}, got {:?}",
            sinks.shape()
        );
    }
    if let Device::Cuda(dev) = q.device()
        && dev.cuda_stream().context().ordinal() >= MAX_DEVICES
    {
        candle_core::bail!("fattn supports CUDA devices 0..{MAX_DEVICES}");
    }
    Ok(())
}

// A `(batch, seq, heads, dim)` operand as ggml's `[dim, seq, heads, batch]` with byte strides.
fn bhsd(ptr: u64, dtype: DType, layout: &Layout) -> Result<ffi::Tensor> {
    let (b, s, h, d) = layout.shape().dims4()?;
    let st = layout.stride();
    if st[3] != 1 {
        candle_core::bail!("fattn needs the head dim contiguous, got strides {st:?}");
    }
    let es = dtype.size_in_bytes() as i64;
    Ok(ffi::Tensor {
        data: ptr as *const _,
        ty: ggml_type(dtype)?,
        ne: [d as i64, s as i64, h as i64, b as i64],
        nb: [es, st[1] as i64 * es, st[2] as i64 * es, st[0] as i64 * es],
    })
}

// A `(batch | 1, seq_q, seq_kv)` mask as ggml's `[n_kv, n_q, 1, ne33]`.
fn mask_descriptor(ptr: u64, layout: &Layout) -> Result<ffi::Tensor> {
    let (mb, sq, skv) = layout.shape().dims3()?;
    let st = layout.stride();
    let es = DType::F16.size_in_bytes() as i64;
    Ok(ffi::Tensor {
        data: ptr as *const _,
        ty: GGML_TYPE_F16,
        ne: [skv as i64, sq as i64, 1, mb as i64],
        nb: [
            es,
            st[1] as i64 * es,
            (st[1] * sq) as i64 * es,
            st[0] as i64 * es,
        ],
    })
}

fn sinks_descriptor(ptr: u64, layout: &Layout) -> Result<ffi::Tensor> {
    let h = layout.shape().dims1()? as i64;
    let es = DType::F32.size_in_bytes() as i64;
    Ok(ffi::Tensor {
        data: ptr as *const _,
        ty: GGML_TYPE_F32,
        ne: [h, 1, 1, 1],
        nb: [es, es * h, es * h, es * h],
    })
}

fn device_ptr<'a>(
    storage: &'a CudaStorage,
    layout: &Layout,
    stream: &'a Arc<CudaStream>,
    guards: &mut Guards<'a>,
) -> Result<u64> {
    let (ptr, guard): (u64, Box<dyn Held + 'a>) = match storage.dtype() {
        DType::F32 => {
            let (p, g) = storage.as_cuda_slice::<f32>()?.device_ptr(stream);
            (p, Box::new(g))
        }
        DType::F16 => {
            let (p, g) = storage.as_cuda_slice::<half::f16>()?.device_ptr(stream);
            (p, Box::new(g))
        }
        DType::BF16 => {
            let (p, g) = storage.as_cuda_slice::<half::bf16>()?.device_ptr(stream);
            (p, Box::new(g))
        }
        dt => candle_core::bail!("fattn does not take {dt:?}"),
    };
    guards.push(guard);
    Ok(ptr + (layout.start_offset() * storage.dtype().size_in_bytes()) as u64)
}

fn args(
    q: ffi::Tensor,
    k: ffi::Tensor,
    v: ffi::Tensor,
    mask: ffi::Tensor,
    sinks: ffi::Tensor,
    opts: &FattnOptions,
    stream: &CudaStream,
) -> ffi::Args {
    ffi::Args {
        q,
        k,
        v,
        mask,
        sinks,
        dst: std::ptr::null_mut(),
        scale: opts.scale,
        max_bias: 0.,
        softcap: opts.softcap,
        device: stream.context().ordinal() as i32,
        stream: stream.cu_stream() as *mut _,
    }
}

fn extra_operand<'a>(
    held: &'a Option<(candle_core::StorageRef<'_>, &Layout)>,
    descriptor: fn(u64, &Layout) -> Result<ffi::Tensor>,
    stream: &'a Arc<CudaStream>,
    guards: &mut Guards<'a>,
) -> Result<ffi::Tensor> {
    let Some((storage, layout)) = held else {
        return Ok(ABSENT);
    };
    let Storage::Cuda(storage) = &**storage else {
        candle_core::bail!("fattn operands must be on CUDA")
    };
    descriptor(device_ptr(storage, layout, stream, guards)?, layout)
}

struct Fattn<'a> {
    opts: &'a FattnOptions,
}

impl candle_core::CustomOp3 for Fattn<'_> {
    fn name(&self) -> &'static str {
        "fattn"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("fattn is CUDA only")
    }

    fn cuda_fwd(
        &self,
        q: &CudaStorage,
        q_l: &Layout,
        k: &CudaStorage,
        k_l: &Layout,
        v: &CudaStorage,
        v_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let dev = q.device();
        let stream = dev.cuda_stream();
        if stream.cu_stream().is_null() {
            candle_core::bail!("fattn needs a non-default CUDA stream");
        }
        let mask = self.opts.mask.as_ref().map(|t| t.storage_and_layout());
        let sinks = self.opts.sinks.as_ref().map(|t| t.storage_and_layout());
        let mut guards = Guards::new();
        let q_t = bhsd(device_ptr(q, q_l, &stream, &mut guards)?, q.dtype(), q_l)?;
        let k_t = bhsd(device_ptr(k, k_l, &stream, &mut guards)?, k.dtype(), k_l)?;
        let v_t = bhsd(device_ptr(v, v_l, &stream, &mut guards)?, v.dtype(), v_l)?;
        let mask_t = extra_operand(&mask, mask_descriptor, &stream, &mut guards)?;
        let sinks_t = extra_operand(&sinks, sinks_descriptor, &stream, &mut guards)?;
        let mut args = args(q_t, k_t, v_t, mask_t, sinks_t, self.opts, &stream);
        if !unsafe { ffi::inference_fattn_supported(&args) } {
            candle_core::bail!(
                "fattn has no kernel for q {:?} k {:?} v {:?}",
                q_l.shape(),
                k_l.shape(),
                v_l.shape()
            );
        }
        let (b, sq, h, _) = q_l.shape().dims4()?;
        let dv = v_l.shape().dims4()?.3;
        let bytes = unsafe { ffi::inference_fattn_alloc_size(&args) };
        let mut dst = unsafe { dev.alloc::<f32>(bytes.div_ceil(DType::F32.size_in_bytes()))? };
        {
            let (ptr, _dst_guard) = dst.device_ptr_mut(&stream);
            args.dst = ptr as *mut _;
            let err = unsafe { ffi::inference_fattn_forward(&args) };
            if err != 0 {
                candle_core::bail!("fattn launch failed with CUDA error {err}");
            }
        }
        drop(guards);
        Ok((
            CudaStorage::wrap_cuda_slice(dst, dev.clone()),
            Shape::from((b, sq, h, dv)),
        ))
    }
}

/// Attention over `q (b, seq_q, n_head, d_qk)`, `k (b, seq_kv, n_head_kv, d_qk)`, `v (b, seq_kv, n_head_kv, d_v)`.
pub fn flash_attn(q: &Tensor, k: &Tensor, v: &Tensor, opts: &FattnOptions) -> Result<Tensor> {
    validate(q, k, v, opts)?;
    let out = q
        .to_dtype(DType::F32)?
        .apply_op3_no_bwd(k, v, &Fattn { opts })?;
    // the output buffer also holds fattn's f16 K/V scratch; a copy releases it with the result
    if q.dtype() == DType::F32 {
        out.copy()
    } else {
        out.to_dtype(q.dtype())
    }
}

/// Whether a kernel exists for these operands on this device; reads only shapes, dtypes and strides.
pub fn supported(q: &Tensor, k: &Tensor, v: &Tensor, opts: &FattnOptions) -> Result<bool> {
    let Device::Cuda(dev) = q.device() else {
        return Ok(false);
    };
    validate(q, k, v, opts)?;
    let layout = |t: &Tensor| t.layout().clone();
    let q_t = bhsd(0, DType::F32, &Layout::contiguous(q.shape()))?;
    let k_t = bhsd(0, k.dtype(), &layout(k))?;
    let v_t = bhsd(0, v.dtype(), &layout(v))?;
    let mask_t = match &opts.mask {
        Some(m) => mask_descriptor(0, &layout(m))?,
        None => ABSENT,
    };
    let sinks_t = match &opts.sinks {
        Some(s) => sinks_descriptor(0, &layout(s))?,
        None => ABSENT,
    };
    let args = args(q_t, k_t, v_t, mask_t, sinks_t, opts, &dev.cuda_stream());
    Ok(unsafe { ffi::inference_fattn_supported(&args) })
}

/// Additive f16 causal mask `(1, seq_q, seq_kv)`, with the queries aligned to the end of the keys.
pub fn causal_mask(seq_q: usize, seq_kv: usize, device: &Device) -> Result<Tensor> {
    let Some(offset) = seq_kv.checked_sub(seq_q) else {
        candle_core::bail!("a causal mask needs seq_q ({seq_q}) <= seq_kv ({seq_kv})");
    };
    let mask: Vec<f32> = (0..seq_q)
        .flat_map(|i| {
            (0..seq_kv).map(move |j| {
                if j <= i + offset {
                    0.
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect();
    Tensor::from_vec(mask, (1, seq_q, seq_kv), device)?.to_dtype(DType::F16)
}
