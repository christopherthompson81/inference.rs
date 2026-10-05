use std::sync::Arc;

use candle_core::cuda_backend::cudarc::driver::{CudaStream, DevicePtr, DevicePtrMut};
use candle_core::{
    CpuStorage, CudaStorage, D, DType, Device, Layout, Result, Shape, Storage, Tensor,
    backend::BackendStorage,
};

use crate::{FattnOptions, KvScales, Packed, PagedKv};

// ggml_type ids
const GGML_TYPE_F32: i32 = 0;
const GGML_TYPE_F16: i32 = 1;
const GGML_TYPE_BF16: i32 = 30;
// fp8 e4m3 K/V travel as a one-byte type; fattn_layout::fp8 says how to read them
const GGML_TYPE_I8: i32 = 24;
// GGML_CUDA_MAX_DEVICES in ggml-cuda.h; fattn keeps per-device state in arrays of this size
const MAX_DEVICES: usize = 16;
// The MLA head dim: fattn reads V out of K's tiles for it (`V_is_K_view = DKQ == 576`, fattn-mma-f16.cuh)
const MLA_HEAD_DIM: usize = 576;
// ggml_cuda_get_max_cpy_bytes: f32 Q rows load in 16-byte chunks, and gqa_opt_applies wants 16-byte Q strides
const Q_LOAD_ALIGN: usize = 16;
// the kernels load K/V rows in place in chunks of this many bytes; only bf16 K/V for the tile kernel is copied first
const KV_LOAD_ALIGN: usize = 16;
// supported() never launches, but fattn reads a null mask or sinks pointer as absent, so it describes operands at this
const PROBE_PTR: u64 = 256;
// e4m3's largest value times this reaches f16's largest (65504 / 448), where dequantized K/V would overflow
const FP8_MAX_SCALE: f32 = 146.;
// FATTN_KQ_STRIDE: a paged call's K/V length is padded to it, which the GQA-batched kernels and tile skipping need
const PAGED_KV_PAD: usize = 256;

mod ffi {
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct Tensor {
        pub data: *const core::ffi::c_void,
        pub ty: i32,
        pub ne: [i64; 4],
        pub nb: [i64; 4],
    }

    // fattn_layout in fattn-common.cuh
    #[repr(C)]
    pub struct FattnLayout {
        pub block_table: *const core::ffi::c_void,
        pub seq_lens: *const core::ffi::c_void,
        pub max_blocks: i32,
        pub block_size_log2: i32,
        pub block_stride_k: i64,
        pub block_stride_v: i64,
        pub fp8: i32,
        pub k_scale: f32,
        pub v_scale: f32,
        pub cu_q: *const core::ffi::c_void,
        pub cu_kv: *const core::ffi::c_void,
    }

    #[repr(C)]
    pub struct Args {
        pub q: Tensor,
        pub k: Tensor,
        pub v: Tensor,
        pub mask: Tensor,
        pub sinks: Tensor,
        pub dst: *mut core::ffi::c_void,
        pub dst_type: i32,
        pub scale: f32,
        pub max_bias: f32,
        pub softcap: f32,
        pub device: i32,
        pub stream: *mut core::ffi::c_void,
        pub lay: *const FattnLayout,
    }

    unsafe extern "C" {
        pub fn inference_fattn_supported(args: *const Args) -> bool;
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
        DType::F8E4M3 => GGML_TYPE_I8,
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
    if let Some(mask) = &opts.mask {
        let (mb, mq, mkv) = mask.dims3()?;
        if (mq, mkv) != (q.dim(1)?, s_kv) || mb == 0 || b % mb != 0 {
            candle_core::bail!(
                "fattn mask {:?} does not fit q {:?} and k {:?}",
                mask.shape(),
                q.shape(),
                k.shape()
            );
        }
    }
    validate_operands(q, k, v, opts)
}

// Checks shared by every call: dtypes, alignment, mask and sinks layout, the device.
fn validate_operands(q: &Tensor, k: &Tensor, v: &Tensor, opts: &FattnOptions) -> Result<()> {
    let h = q.dim(D::Minus2)?;
    let rows_aligned = |t: &Tensor| {
        let l = t.layout();
        let es = t.dtype().size_in_bytes();
        (l.start_offset() * es).is_multiple_of(KV_LOAD_ALIGN)
            && l.stride().split_last().is_some_and(|(_, outer)| {
                outer
                    .iter()
                    .all(|&s| (s * es).is_multiple_of(KV_LOAD_ALIGN))
            })
    };
    if !rows_aligned(k) || !rows_aligned(v) {
        candle_core::bail!(
            "fattn reads K/V rows in {KV_LOAD_ALIGN}-byte chunks; their offsets and strides must align"
        );
    }
    if !matches!(k.dtype(), DType::F16 | DType::BF16 | DType::F8E4M3) || k.dtype() != v.dtype() {
        candle_core::bail!(
            "fattn takes f16, bf16 or fp8 e4m3 K and V of one dtype, got {:?} and {:?}",
            k.dtype(),
            v.dtype()
        );
    }
    let fp8 = k.dtype() == DType::F8E4M3;
    if opts.kv_scales.is_some() && !fp8 {
        candle_core::bail!(
            "fattn's kv_scales dequantize fp8 K/V; these are {:?}",
            k.dtype()
        );
    }
    if let Some(KvScales { k: ks, v: vs }) = opts.kv_scales
        && ![ks, vs]
            .iter()
            .all(|s| s.is_finite() && *s > 0. && *s <= FP8_MAX_SCALE)
    {
        candle_core::bail!(
            "fattn's fp8 scales must lie in (0, {FP8_MAX_SCALE}], got k {ks} v {vs}"
        );
    }
    // at 576 V is read out of K's tiles, which are dequantized with the K scale
    if fp8 && q.dim(D::Minus1)? == MLA_HEAD_DIM {
        candle_core::bail!("fattn does not take fp8 K/V at head dim {MLA_HEAD_DIM}");
    }
    if let Some(mask) = &opts.mask
        && (mask.dtype() != DType::F16 || mask.layout().stride()[2] != 1)
    {
        candle_core::bail!(
            "fattn needs an f16 mask with a contiguous last dim, got {:?}",
            mask.dtype()
        );
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
        DType::U32 => {
            let (p, g) = storage.as_cuda_slice::<u32>()?.device_ptr(stream);
            (p, Box::new(g))
        }
        DType::F8E4M3 => {
            let (p, g) = storage
                .as_cuda_slice::<float8::F8E4M3>()?
                .device_ptr(stream);
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
        dst_type: q.ty,
        scale: opts.scale,
        max_bias: 0.,
        softcap: opts.softcap,
        device: stream.context().ordinal() as i32,
        stream: stream.cu_stream() as *mut _,
        lay: std::ptr::null(),
    }
}

// A `(num_blocks, n_head_kv, block_size, dim)` cache as ggml's `[dim, n_kv, n_head_kv, batch]`; the table picks blocks.
fn cache_descriptor(
    ptr: u64,
    dtype: DType,
    layout: &Layout,
    b: usize,
    n_kv: usize,
) -> Result<ffi::Tensor> {
    let (_, h_kv, _, d) = layout.shape().dims4()?;
    let st = layout.stride();
    let es = dtype.size_in_bytes() as i64;
    Ok(ffi::Tensor {
        data: ptr as *const _,
        ty: ggml_type(dtype)?,
        ne: [d as i64, n_kv as i64, h_kv as i64, b as i64],
        nb: [es, st[2] as i64 * es, st[1] as i64 * es, 0],
    })
}

// A packed `(total, heads, dim)` operand as ggml's `[dim, n_rows, heads, batch]`; cu_seqlens picks each sequence's rows.
fn packed_descriptor(
    ptr: u64,
    dtype: DType,
    layout: &Layout,
    n_rows: usize,
    b: usize,
) -> Result<ffi::Tensor> {
    let (_, h, d) = layout.shape().dims3()?;
    let st = layout.stride();
    if st[2] != 1 {
        candle_core::bail!("fattn needs the head dim contiguous, got strides {st:?}");
    }
    let es = dtype.size_in_bytes() as i64;
    Ok(ffi::Tensor {
        data: ptr as *const _,
        ty: ggml_type(dtype)?,
        ne: [d as i64, n_rows as i64, h as i64, b as i64],
        nb: [es, st[0] as i64 * es, st[1] as i64 * es, 0],
    })
}

fn optional_ptr<'a>(
    held: &'a Option<(candle_core::StorageRef<'_>, &Layout)>,
    stream: &'a Arc<CudaStream>,
    guards: &mut Guards<'a>,
) -> Result<*const core::ffi::c_void> {
    let Some((storage, layout)) = held else {
        return Ok(std::ptr::null());
    };
    let Storage::Cuda(storage) = &**storage else {
        candle_core::bail!("fattn operands must be on CUDA")
    };
    Ok(device_ptr(storage, layout, stream, guards)? as *const _)
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
    paged: Option<PagedKv<'a>>,
    // packed sequences of Q (and dst), and of dense K/V
    q_seqs: Option<Packed<'a>>,
    kv_seqs: Option<Packed<'a>>,
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
        let tables = self.paged.map(|p| {
            (
                p.block_table.storage_and_layout(),
                p.seq_lens.storage_and_layout(),
            )
        });
        let cu_q = self.q_seqs.map(|p| p.cu_seqlens.storage_and_layout());
        let cu_kv = self.kv_seqs.map(|p| p.cu_seqlens.storage_and_layout());
        let mut guards = Guards::new();
        let q_ptr = device_ptr(q, q_l, &stream, &mut guards)?;
        let (b, q_t, out_rows, h, sq) = match &self.q_seqs {
            Some(p) => {
                let (total, h, _) = q_l.shape().dims3()?;
                let b = p.cu_seqlens.dim(0)? - 1;
                (
                    b,
                    packed_descriptor(q_ptr, q.dtype(), q_l, p.max_len, b)?,
                    total,
                    h,
                    p.max_len,
                )
            }
            None => {
                let (b, sq, h, _) = q_l.shape().dims4()?;
                (b, bhsd(q_ptr, q.dtype(), q_l)?, b * sq, h, sq)
            }
        };
        let k_ptr = device_ptr(k, k_l, &stream, &mut guards)?;
        let v_ptr = device_ptr(v, v_l, &stream, &mut guards)?;
        let fp8 = k.dtype() == DType::F8E4M3;
        let mut lay = dense_layout(k.dtype(), self.opts);
        if let (Some(p), Some(((table, table_l), (lens, lens_l)))) = (&self.paged, &tables) {
            let (Storage::Cuda(table), Storage::Cuda(lens)) = (&**table, &**lens) else {
                candle_core::bail!("fattn operands must be on CUDA")
            };
            let es = k.dtype().size_in_bytes() as i64;
            lay.block_table = device_ptr(table, table_l, &stream, &mut guards)? as *const _;
            lay.seq_lens = device_ptr(lens, lens_l, &stream, &mut guards)? as *const _;
            lay.max_blocks = p.block_table.dim(1)? as i32;
            lay.block_size_log2 = k_l.dims()[2].trailing_zeros() as i32;
            lay.block_stride_k = k_l.stride()[0] as i64 * es;
            lay.block_stride_v = v_l.stride()[0] as i64 * es;
        }
        lay.cu_q = optional_ptr(&cu_q, &stream, &mut guards)?;
        lay.cu_kv = optional_ptr(&cu_kv, &stream, &mut guards)?;
        let (k_t, v_t) = match (&self.paged, &self.kv_seqs) {
            (Some(p), _) => {
                let n_kv = paged_kv_len(p)?;
                (
                    cache_descriptor(k_ptr, k.dtype(), k_l, b, n_kv)?,
                    cache_descriptor(v_ptr, v.dtype(), v_l, b, n_kv)?,
                )
            }
            (None, Some(p)) => {
                let n_kv = varlen_kv_len(p.max_len);
                (
                    packed_descriptor(k_ptr, k.dtype(), k_l, n_kv, b)?,
                    packed_descriptor(v_ptr, v.dtype(), v_l, n_kv, b)?,
                )
            }
            (None, None) => (bhsd(k_ptr, k.dtype(), k_l)?, bhsd(v_ptr, v.dtype(), v_l)?),
        };
        let mask_t = extra_operand(&mask, mask_descriptor, &stream, &mut guards)?;
        let sinks_t = extra_operand(&sinks, sinks_descriptor, &stream, &mut guards)?;
        let mut args = args(q_t, k_t, v_t, mask_t, sinks_t, self.opts, &stream);
        if self.paged.is_some() || fp8 || self.q_seqs.is_some() || self.kv_seqs.is_some() {
            args.lay = &lay;
        }
        if !unsafe { ffi::inference_fattn_supported(&args) } {
            candle_core::bail!(
                "fattn has no kernel for q {:?} k {:?} v {:?}",
                q_l.shape(),
                k_l.shape(),
                v_l.shape()
            );
        }
        let dv = v_l.shape().dim(D::Minus1)?;
        let n = out_rows * h * dv;
        let launch = |args: &mut ffi::Args, ptr: u64| -> Result<()> {
            args.dst = ptr as *mut _;
            match unsafe { ffi::inference_fattn_forward(args) } {
                0 => Ok(()),
                err => candle_core::bail!("fattn launch failed with CUDA error {err}"),
            }
        };
        let out = match q.dtype() {
            DType::BF16 => {
                let mut dst = unsafe { dev.alloc::<half::bf16>(n)? };
                launch(&mut args, dst.device_ptr_mut(&stream).0)?;
                CudaStorage::wrap_cuda_slice(dst, dev.clone())
            }
            _ => {
                let mut dst = unsafe { dev.alloc::<f32>(n)? };
                launch(&mut args, dst.device_ptr_mut(&stream).0)?;
                CudaStorage::wrap_cuda_slice(dst, dev.clone())
            }
        };
        drop(guards);
        let shape = match &self.q_seqs {
            Some(_) => Shape::from((out_rows, h, dv)),
            None => Shape::from((b, sq, h, dv)),
        };
        Ok((out, shape))
    }
}

// fattn reads Q and writes its result in f32 or bf16; f16 queries go through f32.
fn native_q_dtype(dtype: DType) -> DType {
    match dtype {
        DType::BF16 => DType::BF16,
        _ => DType::F32,
    }
}

// Q passes through as is when fattn takes its dtype, its rows meet the 16-byte loads and its strides fit the i32 nb0x.
fn q_passes_through(q: &Tensor) -> bool {
    let l = q.layout();
    let es = q.dtype().size_in_bytes();
    let aligned = |n: usize| (n * es).is_multiple_of(Q_LOAD_ALIGN);
    let Some((&inner, outer)) = l.stride().split_last() else {
        return false;
    };
    native_q_dtype(q.dtype()) == q.dtype()
        && inner == 1
        && aligned(l.start_offset())
        && outer
            .iter()
            .all(|&s| aligned(s) && s * es <= i32::MAX as usize)
}

// Anything else becomes a fresh offset-0 copy; `contiguous()` would keep a contiguous view at an unaligned offset.
fn kernel_q(q: &Tensor) -> Result<Tensor> {
    if q_passes_through(q) {
        Ok(q.clone())
    } else if native_q_dtype(q.dtype()) == q.dtype() {
        q.force_contiguous()
    } else {
        q.to_dtype(native_q_dtype(q.dtype()))
    }
}

// The layout of a batched dense call: only fp8 storage and its scales; callers fill in paging and packing.
fn dense_layout(k_dtype: DType, opts: &FattnOptions) -> ffi::FattnLayout {
    let scales = opts.kv_scales.unwrap_or_default();
    ffi::FattnLayout {
        block_table: std::ptr::null(),
        seq_lens: std::ptr::null(),
        max_blocks: 0,
        block_size_log2: 0,
        block_stride_k: 0,
        block_stride_v: 0,
        fp8: (k_dtype == DType::F8E4M3) as i32,
        k_scale: scales.k,
        v_scale: scales.v,
        cu_q: std::ptr::null(),
        cu_kv: std::ptr::null(),
    }
}

/// Attention over `q (b, seq_q, n_head, d_qk)`, `k (b, seq_kv, n_head_kv, d_qk)`, `v (b, seq_kv, n_head_kv, d_v)`.
pub fn flash_attn(q: &Tensor, k: &Tensor, v: &Tensor, opts: &FattnOptions) -> Result<Tensor> {
    validate(q, k, v, opts)?;
    kernel_q(q)?
        .apply_op3_no_bwd(
            k,
            v,
            &Fattn {
                opts,
                paged: None,
                q_seqs: None,
                kv_seqs: None,
            },
        )?
        .to_dtype(q.dtype())
}

/// Whether a kernel exists for these operands on this device; reads only shapes, dtypes and strides.
pub fn supported(q: &Tensor, k: &Tensor, v: &Tensor, opts: &FattnOptions) -> Result<bool> {
    let Device::Cuda(dev) = q.device() else {
        return Ok(false);
    };
    validate(q, k, v, opts)?;
    let layout = |t: &Tensor| t.layout().clone();
    let q_layout = if q_passes_through(q) {
        layout(q)
    } else {
        Layout::contiguous(q.shape())
    };
    let q_t = bhsd(PROBE_PTR, native_q_dtype(q.dtype()), &q_layout)?;
    let k_t = bhsd(PROBE_PTR, k.dtype(), &layout(k))?;
    let v_t = bhsd(PROBE_PTR, v.dtype(), &layout(v))?;
    let mask_t = match &opts.mask {
        Some(m) => mask_descriptor(PROBE_PTR, &layout(m))?,
        None => ABSENT,
    };
    let sinks_t = match &opts.sinks {
        Some(s) => sinks_descriptor(PROBE_PTR, &layout(s))?,
        None => ABSENT,
    };
    let mut args = args(q_t, k_t, v_t, mask_t, sinks_t, opts, &dev.cuda_stream());
    let lay = dense_layout(k.dtype(), opts);
    if lay.fp8 != 0 {
        args.lay = &lay;
    }
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

/// The K/V length a paged call attends over (the mask's last dim): the block table's span, padded to fattn's KV tile.
pub fn paged_kv_len(kv: &PagedKv) -> Result<usize> {
    let block_size = kv.k_cache.dim(2)?;
    Ok((kv.block_table.dim(1)? * block_size).next_multiple_of(PAGED_KV_PAD))
}

// `(b, sq, h, d)`: the batch, Q rows per sequence (the most, when packed), heads and head dim of Q.
fn validate_paged(
    q: &Tensor,
    (b, sq, h, d): (usize, usize, usize, usize),
    kv: &PagedKv,
    opts: &FattnOptions,
) -> Result<()> {
    let (nb, h_kv, bs, kd) = kv.k_cache.dims4()?;
    let (vnb, vh, vbs, _) = kv.v_cache.dims4()?;
    if !bs.is_power_of_two() {
        candle_core::bail!("paged fattn needs a power-of-two block size, got {bs}");
    }
    if kd != d || (nb, h_kv, bs) != (vnb, vh, vbs) {
        candle_core::bail!(
            "fattn paged operands disagree: q {:?} k cache {:?} v cache {:?}",
            q.shape(),
            kv.k_cache.shape(),
            kv.v_cache.shape()
        );
    }
    if d == MLA_HEAD_DIM {
        candle_core::bail!(
            "fattn reads V out of K at head dim {MLA_HEAD_DIM}, which separate paged caches cannot give"
        );
    }
    if kv.block_table.dim(1)? == 0 {
        candle_core::bail!("paged fattn needs at least one block per sequence");
    }
    let tables_ok = kv.block_table.dtype() == DType::U32
        && kv.block_table.dims2()?.0 == b
        && kv.block_table.is_contiguous()
        && kv.seq_lens.dtype() == DType::U32
        && kv.seq_lens.dims1()? == b
        && kv.seq_lens.is_contiguous();
    if !tables_ok {
        candle_core::bail!(
            "fattn needs contiguous u32 block tables (batch, max_blocks) and seq lens (batch,), got {:?} and {:?}",
            kv.block_table.shape(),
            kv.seq_lens.shape()
        );
    }
    // rows past a sequence's length are read from its first row, so only the mask keeps them out
    let n_kv = paged_kv_len(kv)?;
    match &opts.mask {
        Some(mask) if mask.dims3()? == (b, sq, n_kv) => {}
        _ => candle_core::bail!(
            "paged fattn needs a (batch, seq_q, {n_kv}) mask hiding each sequence's unused rows"
        ),
    }
    if kv.k_cache.layout().stride()[3] != 1 || kv.v_cache.layout().stride()[3] != 1 {
        candle_core::bail!("fattn needs the paged caches' head dim contiguous");
    }
    if h_kv == 0 || h % h_kv != 0 {
        candle_core::bail!("fattn needs n_head ({h}) to be a multiple of n_head_kv ({h_kv})");
    }
    Ok(())
}

/// Attention of `q (b, seq_q, n_head, d)` over each sequence's rows in a paged cache; the mask spans `paged_kv_len`.
pub fn flash_attn_paged(q: &Tensor, kv: &PagedKv, opts: &FattnOptions) -> Result<Tensor> {
    validate_paged(q, q.dims4()?, kv, opts)?;
    validate_operands(q, kv.k_cache, kv.v_cache, opts)?;
    kernel_q(q)?
        .apply_op3_no_bwd(
            kv.k_cache,
            kv.v_cache,
            &Fattn {
                opts,
                paged: Some(*kv),
                q_seqs: None,
                kv_seqs: None,
            },
        )?
        .to_dtype(q.dtype())
}

/// Additive f16 causal mask `(b, seq_q, n_kv)`; sequence `i`'s queries are the last `seq_q` of its `seq_lens[i]` rows.
pub fn paged_causal_mask(
    seq_lens: &[usize],
    seq_q: usize,
    n_kv: usize,
    device: &Device,
) -> Result<Tensor> {
    varlen_causal_mask(&vec![seq_q; seq_lens.len()], seq_lens, n_kv, device)
}

/// Additive f16 causal mask `(b, max q_lens, n_kv)` for packed or paged sequences: sequence `i`'s queries are the
/// last `q_lens[i]` of its `kv_lens[i]` rows; its rows past `q_lens[i]` are fully masked.
pub fn varlen_causal_mask(
    q_lens: &[usize],
    kv_lens: &[usize],
    n_kv: usize,
    device: &Device,
) -> Result<Tensor> {
    if q_lens.len() != kv_lens.len() {
        candle_core::bail!(
            "{} query lengths for {} sequences",
            q_lens.len(),
            kv_lens.len()
        );
    }
    let max_q = q_lens.iter().copied().max().unwrap_or(0);
    let mut mask = Vec::with_capacity(q_lens.len() * max_q * n_kv);
    for (&q_len, &kv_len) in q_lens.iter().zip(kv_lens) {
        let Some(offset) = kv_len.checked_sub(q_len) else {
            candle_core::bail!(
                "a causal mask needs q_len ({q_len}) <= the sequence's length ({kv_len})"
            );
        };
        for i in 0..max_q {
            mask.extend((0..n_kv).map(|j| {
                if i < q_len && j <= i + offset {
                    0f32
                } else {
                    f32::NEG_INFINITY
                }
            }));
        }
    }
    Tensor::from_vec(mask, (q_lens.len(), max_q, n_kv), device)?.to_dtype(DType::F16)
}

/// The K/V length a dense varlen call attends over (the mask's last dim): the longest sequence, padded to the KV tile.
pub fn varlen_kv_len(max_len: usize) -> usize {
    max_len.next_multiple_of(PAGED_KV_PAD)
}

// The batch `seqs` describes; its offsets and longest length cannot be checked against the device-side values.
fn validate_packed(seqs: &Packed, rows: usize, what: &str) -> Result<usize> {
    let cu = seqs.cu_seqlens;
    if cu.dtype() != DType::U32 || cu.rank() != 1 || cu.dim(0)? < 2 || !cu.is_contiguous() {
        candle_core::bail!(
            "fattn needs contiguous u32 {what} cu_seqlens (batch + 1,), got {:?}",
            cu.shape()
        );
    }
    if seqs.max_len == 0 || seqs.max_len > rows {
        candle_core::bail!(
            "fattn {what} max_len {} must lie in 1..={rows}",
            seqs.max_len
        );
    }
    // the kernels read cu_seqlens and row indices as i32
    if rows > i32::MAX as usize {
        candle_core::bail!(
            "fattn takes at most {} packed {what} rows, got {rows}",
            i32::MAX
        );
    }
    Ok(cu.dim(0)? - 1)
}

fn check_mask(opts: &FattnOptions, dims: (usize, usize, usize)) -> Result<()> {
    match &opts.mask {
        Some(mask) if mask.dims3()? == dims => Ok(()),
        _ => candle_core::bail!(
            "varlen fattn needs a {dims:?} mask hiding each sequence's unused rows"
        ),
    }
}

/// Attention over sequences packed along dim 0: `q (total_q, n_head, d)`, `k (total_kv, n_head_kv, d)`,
/// `v (total_kv, n_head_kv, d_v)`, delimited by `q_seqs` and `kv_seqs`. The mask is `(b, q max_len, varlen_kv_len)`.
pub fn flash_attn_varlen(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    q_seqs: &Packed,
    kv_seqs: &Packed,
    opts: &FattnOptions,
) -> Result<Tensor> {
    let (tq, h, d) = q.dims3()?;
    let (tk, h_kv, kd) = k.dims3()?;
    let (vt, vh, _) = v.dims3()?;
    let b = validate_packed(q_seqs, tq, "q")?;
    if validate_packed(kv_seqs, tk, "kv")? != b || kd != d || (vt, vh) != (tk, h_kv) {
        candle_core::bail!(
            "fattn varlen operands disagree: q {:?} k {:?} v {:?}, q cu {:?} kv cu {:?}",
            q.shape(),
            k.shape(),
            v.shape(),
            q_seqs.cu_seqlens.shape(),
            kv_seqs.cu_seqlens.shape()
        );
    }
    if d == MLA_HEAD_DIM {
        candle_core::bail!("fattn varlen does not take head dim {MLA_HEAD_DIM}");
    }
    if h_kv == 0 || h % h_kv != 0 {
        candle_core::bail!("fattn needs n_head ({h}) to be a multiple of n_head_kv ({h_kv})");
    }
    check_mask(opts, (b, q_seqs.max_len, varlen_kv_len(kv_seqs.max_len)))?;
    validate_operands(q, k, v, opts)?;
    kernel_q(q)?
        .apply_op3_no_bwd(
            k,
            v,
            &Fattn {
                opts,
                paged: None,
                q_seqs: Some(*q_seqs),
                kv_seqs: Some(*kv_seqs),
            },
        )?
        .to_dtype(q.dtype())
}

/// Attention of packed `q (total_q, n_head, d)`, delimited by `q_seqs`, over each sequence's rows in a paged cache;
/// the mask is `(b, q max_len, paged_kv_len)`.
pub fn flash_attn_paged_varlen(
    q: &Tensor,
    q_seqs: &Packed,
    kv: &PagedKv,
    opts: &FattnOptions,
) -> Result<Tensor> {
    let (tq, h, d) = q.dims3()?;
    let b = validate_packed(q_seqs, tq, "q")?;
    validate_paged(q, (b, q_seqs.max_len, h, d), kv, opts)?;
    validate_operands(q, kv.k_cache, kv.v_cache, opts)?;
    kernel_q(q)?
        .apply_op3_no_bwd(
            kv.k_cache,
            kv.v_cache,
            &Fattn {
                opts,
                paged: Some(*kv),
                q_seqs: Some(*q_seqs),
                kv_seqs: None,
            },
        )?
        .to_dtype(q.dtype())
}
