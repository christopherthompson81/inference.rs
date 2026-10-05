use std::sync::{Arc, OnceLock};

use candle_core::cuda_backend::cudarc::driver::{CudaStream, DevicePtr, DevicePtrMut};
use candle_core::{
    CpuStorage, CudaStorage, D, DType, Device, Layout, Result, Shape, Storage, Tensor,
    backend::BackendStorage,
};

use crate::{FattnOptions, KvScales, Packed, PagedKv};

// Turing: causal, packed and paged calls run on the mma kernel alone (ggml_cuda_get_best_fattn_kernel)
const MMA_MIN_COMPUTE: (i32, i32) = (7, 5);

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
// fattn.cu runs these head dims only with the GQA optimisation, which needs a mask (ggml_cuda_get_best_fattn_kernel)
const GQA_ONLY_HEAD_DIMS: [usize; 4] = [192, 320, 512, 576];
// ggml_cuda_get_max_cpy_bytes: f32 Q rows load in 16-byte chunks, and gqa_opt_applies wants 16-byte Q strides
const Q_LOAD_ALIGN: usize = 16;
// the kernels load K/V rows in place in chunks of this many bytes; only bf16 K/V for the tile kernel is copied first
const KV_LOAD_ALIGN: usize = 16;
// a never-read address: supported() describes operands at it (null reads as absent), and implicit masks point at it
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
        pub implicit_mask: i32,
        pub causal: i32,
        pub window_left: i32,
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
    if opts.causal && q.dim(1)? > s_kv {
        candle_core::bail!(
            "causal fattn needs seq_q ({}) <= seq_kv ({s_kv})",
            q.dim(1)?
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

fn fp8_scale_limit(opts: &FattnOptions) -> Option<String> {
    let KvScales { k, v } = opts.kv_scales?;
    let in_range = |s: f32| s.is_finite() && s > 0. && s <= FP8_MAX_SCALE;
    (!in_range(k) || !in_range(v))
        .then(|| format!("fattn's fp8 scales must lie in (0, {FP8_MAX_SCALE}], got k {k} v {v}"))
}

// What paged calls cannot do, as opposed to malformed operands: `supported_paged` answers false for these.
fn paged_limit(q: &Tensor, kv: &PagedKv, opts: &FattnOptions) -> Result<Option<String>> {
    let (block_size, head_dim) = (kv.k_cache.dim(2)?, q.dim(D::Minus1)?);
    if let Some(limit) = operand_limit(q, kv.k_cache, kv.v_cache) {
        return Ok(Some(limit));
    }
    Ok(if !block_size.is_power_of_two() {
        Some(format!(
            "paged fattn needs a power-of-two block size, got {block_size}"
        ))
    } else if head_dim == MLA_HEAD_DIM {
        Some(format!(
            "fattn reads V out of K at head dim {MLA_HEAD_DIM}, which separate paged caches cannot give"
        ))
    } else {
        fp8_scale_limit(opts)
    })
}

fn rows_aligned(t: &Tensor) -> bool {
    let l = t.layout();
    let es = t.dtype().size_in_bytes();
    (l.start_offset() * es).is_multiple_of(KV_LOAD_ALIGN)
        && l.stride().split_last().is_some_and(|(_, outer)| {
            outer
                .iter()
                .all(|&s| (s * es).is_multiple_of(KV_LOAD_ALIGN))
        })
}

// K/V dtypes, alignment and the device: what fattn cannot take, whatever the call.
fn operand_limit(q: &Tensor, k: &Tensor, v: &Tensor) -> Option<String> {
    if !rows_aligned(k) || !rows_aligned(v) {
        return Some(format!(
            "fattn reads K/V rows in {KV_LOAD_ALIGN}-byte chunks; their offsets and strides must align"
        ));
    }
    if !matches!(k.dtype(), DType::F16 | DType::BF16 | DType::F8E4M3) || k.dtype() != v.dtype() {
        return Some(format!(
            "fattn takes f16, bf16 or fp8 e4m3 K and V of one dtype, got {:?} and {:?}",
            k.dtype(),
            v.dtype()
        ));
    }
    if let Device::Cuda(dev) = q.device()
        && dev.cuda_stream().context().ordinal() >= MAX_DEVICES
    {
        return Some(format!("fattn supports CUDA devices 0..{MAX_DEVICES}"));
    }
    None
}

// Checks shared by every call: dtypes, alignment, mask and sinks layout, the device.
fn validate_operands(q: &Tensor, k: &Tensor, v: &Tensor, opts: &FattnOptions) -> Result<()> {
    let h = q.dim(D::Minus2)?;
    if let Some(limit) = operand_limit(q, k, v) {
        candle_core::bail!("{limit}");
    }
    let fp8 = k.dtype() == DType::F8E4M3;
    if opts.kv_scales.is_some() && !fp8 {
        candle_core::bail!(
            "fattn's kv_scales dequantize fp8 K/V; these are {:?}",
            k.dtype()
        );
    }
    if let Some(limit) = fp8_scale_limit(opts) {
        candle_core::bail!("{limit}");
    }
    // at 576 V is read out of K's tiles, which are dequantized with the K scale
    if fp8 && q.dim(D::Minus1)? == MLA_HEAD_DIM {
        candle_core::bail!("fattn does not take fp8 K/V at head dim {MLA_HEAD_DIM}");
    }
    if opts.mask.is_some() && opts.causal {
        candle_core::bail!("fattn takes a mask tensor or causal masking, not both");
    }
    if opts
        .window_left
        .is_some_and(|w| !opts.causal || w > i32::MAX as usize)
    {
        candle_core::bail!("fattn's window_left needs causal masking and must fit an i32");
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

fn held_ptr<'a>(
    held: &'a Option<(candle_core::StorageRef<'_>, &Layout)>,
    stream: &'a Arc<CudaStream>,
    guards: &mut Guards<'a>,
) -> Result<Option<u64>> {
    let Some((storage, layout)) = held else {
        return Ok(None);
    };
    let Storage::Cuda(storage) = &**storage else {
        candle_core::bail!("fattn operands must be on CUDA")
    };
    device_ptr(storage, layout, stream, guards).map(Some)
}

fn held(t: Option<&Tensor>) -> Option<(candle_core::StorageRef<'_>, &Layout)> {
    t.map(Tensor::storage_and_layout)
}

fn null_or(ptr: Option<u64>) -> *const core::ffi::c_void {
    ptr.map_or(std::ptr::null(), |p| p as *const _)
}

// Device addresses of a call's operands; a probe puts PROBE_PTR at each one present.
struct Addrs {
    q: u64,
    k: u64,
    v: u64,
    mask: Option<u64>,
    sinks: Option<u64>,
    block_table: Option<u64>,
    seq_lens: Option<u64>,
    cu_q: Option<u64>,
    cu_kv: Option<u64>,
}

// A call's descriptors; the args point at `lay`, so they must not outlive it.
struct Call {
    q: ffi::Tensor,
    k: ffi::Tensor,
    v: ffi::Tensor,
    mask: ffi::Tensor,
    sinks: ffi::Tensor,
    lay: ffi::FattnLayout,
    uses_lay: bool,
    out_shape: Shape,
}

impl Call {
    fn args(&self, opts: &FattnOptions, stream: &CudaStream) -> ffi::Args {
        let mut args = args(self.q, self.k, self.v, self.mask, self.sinks, opts, stream);
        if self.uses_lay {
            args.lay = &self.lay;
        }
        args
    }
}

struct Fattn<'a> {
    opts: &'a FattnOptions,
    paged: Option<PagedKv<'a>>,
    // packed sequences of Q (and dst), and of dense K/V
    q_seqs: Option<Packed<'a>>,
    kv_seqs: Option<Packed<'a>>,
}

impl Fattn<'_> {
    fn describe(
        &self,
        a: &Addrs,
        (q_dt, q_l): (DType, &Layout),
        (k_dt, k_l): (DType, &Layout),
        (v_dt, v_l): (DType, &Layout),
    ) -> Result<Call> {
        let (b, q_t, out_rows, h, sq) = match &self.q_seqs {
            Some(p) => {
                let (total, h, _) = q_l.shape().dims3()?;
                let b = p.cu_seqlens.dim(0)? - 1;
                (
                    b,
                    packed_descriptor(a.q, q_dt, q_l, p.max_len, b)?,
                    total,
                    h,
                    p.max_len,
                )
            }
            None => {
                let (b, sq, h, _) = q_l.shape().dims4()?;
                (b, bhsd(a.q, q_dt, q_l)?, b * sq, h, sq)
            }
        };
        let mut lay = dense_layout(k_dt, self.opts, sq, q_l.dims()[q_l.dims().len() - 1]);
        if let Some(p) = &self.paged {
            let es = k_dt.size_in_bytes() as i64;
            lay.block_table = null_or(a.block_table);
            lay.seq_lens = null_or(a.seq_lens);
            lay.max_blocks = p.block_table.dim(1)? as i32;
            lay.block_size_log2 = k_l.dims()[2].trailing_zeros() as i32;
            lay.block_stride_k = k_l.stride()[0] as i64 * es;
            lay.block_stride_v = v_l.stride()[0] as i64 * es;
        }
        lay.cu_q = null_or(a.cu_q);
        lay.cu_kv = null_or(a.cu_kv);
        let (k_t, v_t) = match (&self.paged, &self.kv_seqs) {
            (Some(p), _) => {
                let n_kv = paged_kv_len(p)?;
                (
                    cache_descriptor(a.k, k_dt, k_l, b, n_kv)?,
                    cache_descriptor(a.v, v_dt, v_l, b, n_kv)?,
                )
            }
            (None, Some(p)) => {
                let n_kv = varlen_kv_len(p.max_len);
                (
                    packed_descriptor(a.k, k_dt, k_l, n_kv, b)?,
                    packed_descriptor(a.v, v_dt, v_l, n_kv, b)?,
                )
            }
            (None, None) => (bhsd(a.k, k_dt, k_l)?, bhsd(a.v, v_dt, v_l)?),
        };
        let sequences = self.paged.is_some() || self.q_seqs.is_some() || self.kv_seqs.is_some();
        // sequences of their own lengths need a mask; without a tensor the kernel builds it from the lengths
        if self.opts.mask.is_none() && sequences {
            lay.implicit_mask = 1;
        }
        let mask = match (&self.opts.mask, a.mask) {
            _ if lay.implicit_mask != 0 => implicit_mask_descriptor(sq, k_t.ne[1] as usize),
            (Some(m), Some(ptr)) => mask_descriptor(ptr, m.layout())?,
            _ => ABSENT,
        };
        let sinks = match (&self.opts.sinks, a.sinks) {
            (Some(s), Some(ptr)) => sinks_descriptor(ptr, s.layout())?,
            _ => ABSENT,
        };
        let dv = v_l.shape().dim(D::Minus1)?;
        let out_shape = match &self.q_seqs {
            Some(_) => Shape::from((out_rows, h, dv)),
            None => Shape::from((b, sq, h, dv)),
        };
        Ok(Call {
            q: q_t,
            k: k_t,
            v: v_t,
            mask,
            sinks,
            uses_lay: sequences || lay.fp8 != 0 || lay.implicit_mask != 0,
            lay,
            out_shape,
        })
    }

    // Whether fattn has a kernel for this call, from shapes, dtypes and strides alone; operands are validated.
    fn probe(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<bool> {
        let Device::Cuda(dev) = q.device() else {
            return Ok(false);
        };
        let q_layout = if q_passes_through(q) {
            q.layout().clone()
        } else {
            Layout::contiguous(q.shape())
        };
        let present = |t: Option<&Tensor>| t.map(|_| PROBE_PTR);
        let addrs = Addrs {
            q: PROBE_PTR,
            k: PROBE_PTR,
            v: PROBE_PTR,
            mask: present(self.opts.mask.as_ref()),
            sinks: present(self.opts.sinks.as_ref()),
            block_table: present(self.paged.map(|p| p.block_table)),
            seq_lens: present(self.paged.map(|p| p.seq_lens)),
            cu_q: present(self.q_seqs.map(|p| p.cu_seqlens)),
            cu_kv: present(self.kv_seqs.map(|p| p.cu_seqlens)),
        };
        let call = self.describe(
            &addrs,
            (native_q_dtype(q.dtype()), &q_layout),
            (k.dtype(), k.layout()),
            (v.dtype(), v.layout()),
        )?;
        Ok(unsafe { ffi::inference_fattn_supported(&call.args(self.opts, &dev.cuda_stream())) })
    }
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
        let mask = held(self.opts.mask.as_ref());
        let sinks = held(self.opts.sinks.as_ref());
        let block_table = held(self.paged.map(|p| p.block_table));
        let seq_lens = held(self.paged.map(|p| p.seq_lens));
        let cu_q = held(self.q_seqs.map(|p| p.cu_seqlens));
        let cu_kv = held(self.kv_seqs.map(|p| p.cu_seqlens));
        let mut guards = Guards::new();
        let addrs = Addrs {
            q: device_ptr(q, q_l, &stream, &mut guards)?,
            k: device_ptr(k, k_l, &stream, &mut guards)?,
            v: device_ptr(v, v_l, &stream, &mut guards)?,
            mask: held_ptr(&mask, &stream, &mut guards)?,
            sinks: held_ptr(&sinks, &stream, &mut guards)?,
            block_table: held_ptr(&block_table, &stream, &mut guards)?,
            seq_lens: held_ptr(&seq_lens, &stream, &mut guards)?,
            cu_q: held_ptr(&cu_q, &stream, &mut guards)?,
            cu_kv: held_ptr(&cu_kv, &stream, &mut guards)?,
        };
        let call = self.describe(&addrs, (q.dtype(), q_l), (k.dtype(), k_l), (v.dtype(), v_l))?;
        let mut args = call.args(self.opts, &stream);
        if !unsafe { ffi::inference_fattn_supported(&args) } {
            candle_core::bail!(
                "fattn has no kernel for q {:?} k {:?} v {:?}",
                q_l.shape(),
                k_l.shape(),
                v_l.shape()
            );
        }
        let n = call.out_shape.elem_count();
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
        Ok((out, call.out_shape))
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

// The layout of a batched dense call (fp8 storage, scales, causal masking); callers fill in paging and packing.
fn dense_layout(
    k_dtype: DType,
    opts: &FattnOptions,
    seq_q: usize,
    head_dim: usize,
) -> ffi::FattnLayout {
    let scales = opts.kv_scales.unwrap_or_default();
    // one query with no window sees every key, so causal masks nothing and the call can keep the vec kernel; the
    // head dims that only run GQA-batched still need a mask (real or implicit) to be selected at all
    let masks_nothing =
        seq_q == 1 && opts.window_left.is_none() && !GQA_ONLY_HEAD_DIMS.contains(&head_dim);
    let causal = opts.causal && !masks_nothing;
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
        implicit_mask: (opts.mask.is_none() && causal) as i32,
        causal: causal as i32,
        window_left: opts.window_left.map_or(-1, |w| w as i32),
    }
}

// An implicit mask's descriptor: never read, but its presence and shape select the GQA-batched kernels as a mask would.
fn implicit_mask_descriptor(n_q: usize, n_kv: usize) -> ffi::Tensor {
    let es = DType::F16.size_in_bytes() as i64;
    let (n_q, n_kv) = (n_q as i64, n_kv as i64);
    ffi::Tensor {
        data: PROBE_PTR as *const _,
        ty: GGML_TYPE_F16,
        ne: [n_kv, n_q, 1, 1],
        nb: [es, n_kv * es, n_kv * n_q * es, n_kv * n_q * es],
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

/// Whether every visible CUDA device runs fattn's mma kernel, which causal, packed and paged calls need.
pub fn mma_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        use candle_core::cuda_backend::cudarc::driver::{result, sys::CUdevice_attribute};
        let all_turing = || -> std::result::Result<bool, result::DriverError> {
            result::init()?;
            let count = result::device::get_count()?;
            (0..count).try_fold(count > 0, |ok, ordinal| {
                let dev = result::device::get(ordinal)?;
                // SAFETY: dev comes from device::get
                let compute = unsafe {
                    (
                        result::device::get_attribute(
                            dev,
                            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                        )?,
                        result::device::get_attribute(
                            dev,
                            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                        )?,
                    )
                };
                Ok(ok && compute >= MMA_MIN_COMPUTE)
            })
        };
        all_turing().unwrap_or(false)
    })
}

/// Whether a kernel exists for these operands on this device (false for limits such as f32 K/V); reads no data.
pub fn supported(q: &Tensor, k: &Tensor, v: &Tensor, opts: &FattnOptions) -> Result<bool> {
    if operand_limit(q, k, v).is_some() || fp8_scale_limit(opts).is_some() {
        return Ok(false);
    }
    validate(q, k, v, opts)?;
    Fattn {
        opts,
        paged: None,
        q_seqs: None,
        kv_seqs: None,
    }
    .probe(q, k, v)
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
    if let Some(limit) = paged_limit(q, kv, opts)? {
        candle_core::bail!("{limit}");
    }
    if kd != d || (nb, h_kv, bs) != (vnb, vh, vbs) {
        candle_core::bail!(
            "fattn paged operands disagree: q {:?} k cache {:?} v cache {:?}",
            q.shape(),
            kv.k_cache.shape(),
            kv.v_cache.shape()
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
    check_mask(opts, (b, sq, n_kv))?;
    if kv.k_cache.layout().stride()[3] != 1 || kv.v_cache.layout().stride()[3] != 1 {
        candle_core::bail!("fattn needs the paged caches' head dim contiguous");
    }
    if h_kv == 0 || h % h_kv != 0 {
        candle_core::bail!("fattn needs n_head ({h}) to be a multiple of n_head_kv ({h_kv})");
    }
    Ok(())
}

/// Whether `flash_attn_paged` has a kernel for these operands; false for limits such as f32 K/V or odd block sizes.
pub fn supported_paged(q: &Tensor, kv: &PagedKv, opts: &FattnOptions) -> Result<bool> {
    if paged_limit(q, kv, opts)?.is_some() {
        return Ok(false);
    }
    validate_paged(q, q.dims4()?, kv, opts)?;
    validate_operands(q, kv.k_cache, kv.v_cache, opts)?;
    Fattn {
        opts,
        paged: Some(*kv),
        q_seqs: None,
        kv_seqs: None,
    }
    .probe(q, kv.k_cache, kv.v_cache)
}

/// Attention of `q (b, seq_q, n_head, d)` over each sequence's rows in a paged cache; a mask spans `paged_kv_len`.
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
    // without one the kernel masks each sequence's unused rows (and causally, if asked) itself
    match &opts.mask {
        Some(mask) if mask.dims3()? != dims => {
            candle_core::bail!("fattn mask {:?} does not fit {dims:?}", mask.shape())
        }
        _ => Ok(()),
    }
}

/// Attention over sequences packed along dim 0: `q (total_q, n_head, d)`, `k (total_kv, n_head_kv, d)`,
/// `v (total_kv, n_head_kv, d_v)`, delimited by `q_seqs` and `kv_seqs`. A mask, if given, is
/// `(b, q max_len, varlen_kv_len)`; without one the kernel masks from the lengths (and causally, if `causal`).
pub fn flash_attn_varlen(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    q_seqs: &Packed,
    kv_seqs: &Packed,
    opts: &FattnOptions,
) -> Result<Tensor> {
    validate_varlen(q, k, v, (q_seqs, kv_seqs), opts)?;
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

/// Whether `flash_attn_varlen` has a kernel for these operands; false for limits such as f32 K/V.
pub fn supported_varlen(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    q_seqs: &Packed,
    kv_seqs: &Packed,
    opts: &FattnOptions,
) -> Result<bool> {
    if operand_limit(q, k, v).is_some()
        || fp8_scale_limit(opts).is_some()
        || q.dim(D::Minus1)? == MLA_HEAD_DIM
    {
        return Ok(false);
    }
    validate_varlen(q, k, v, (q_seqs, kv_seqs), opts)?;
    Fattn {
        opts,
        paged: None,
        q_seqs: Some(*q_seqs),
        kv_seqs: Some(*kv_seqs),
    }
    .probe(q, k, v)
}

fn validate_varlen(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    (q_seqs, kv_seqs): (&Packed, &Packed),
    opts: &FattnOptions,
) -> Result<()> {
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
    validate_operands(q, k, v, opts)
}

/// Whether `flash_attn_paged_varlen` has a kernel for these operands; false for limits as in `supported_paged`.
pub fn supported_paged_varlen(
    q: &Tensor,
    q_seqs: &Packed,
    kv: &PagedKv,
    opts: &FattnOptions,
) -> Result<bool> {
    if paged_limit(q, kv, opts)?.is_some() {
        return Ok(false);
    }
    let (tq, h, d) = q.dims3()?;
    let b = validate_packed(q_seqs, tq, "q")?;
    validate_paged(q, (b, q_seqs.max_len, h, d), kv, opts)?;
    validate_operands(q, kv.k_cache, kv.v_cache, opts)?;
    Fattn {
        opts,
        paged: Some(*kv),
        q_seqs: Some(*q_seqs),
        kv_seqs: None,
    }
    .probe(q, kv.k_cache, kv.v_cache)
}

/// Attention of packed `q (total_q, n_head, d)`, delimited by `q_seqs`, over each sequence's rows in a paged cache;
/// a mask, if given, is `(b, q max_len, paged_kv_len)`, else the kernel masks from the lengths.
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
