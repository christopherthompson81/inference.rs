//! Causal attention with a relative bias looked up by distance and a sink logit per query (Inkling 2), fused: the
//! bias and the mask are never materialized (see `kernels/cuda/biased_attention.cu`). Keys come from a contiguous
//! view or a paged cache, positions from one past length or a device array (so a captured graph reads them).

use inference_tensor::backend::BackendStorage;
use inference_tensor::cuda_backend::CudaStorage;
use inference_tensor::cuda_backend::cudarc::driver::{
    CudaStream, DevicePtr, DevicePtrMut, SyncOnDrop,
};
use inference_tensor::{DType, Device, Layout, Result, Shape, Storage, Tensor};

/// Head sizes the kernel is built for
pub const HEAD_DIMS: [usize; 3] = [32, 64, 128];
// queries up to this many take the kernel that splits the keys across warps (decode): a speculative verify's rows,
// up to 7 drafts and the token they follow (the scalar prefill kernel walks a long past one block a row)
const DECODE_ROWS: i32 = 8;
// the paged kernel's cache dtype codes
const CACHE_F32: i32 = 0;
const CACHE_F16: i32 = 1;
const CACHE_BF16: i32 = 2;
const CACHE_E4M3: i32 = 3;
const CACHE_Q8: i32 = 4;
const CACHE_PACKED: i32 = 5;
// the q8 cache's bits, read byte by byte; the packed widths below it, read through their 8-value chunks
const Q8_BITS: u32 = 8;
const PACKED_BITS: std::ops::RangeInclusive<u32> = 2..=6;
// rows (batch x queries x heads) a split decode counts its finished splits for; more take one block a row
const MAX_SPLIT_ROWS: usize = 4096;

// each device's split-decode counts, zeroed once and left zeroed by the kernel, kept for the process (a captured graph
// holds the pointer)
static ARRIVED: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<inference_tensor::cuda_backend::DeviceId, u64>>,
> = std::sync::OnceLock::new();

// the device's counts, made on its first eager use; none while a stream captures (the allocation would be the graph's)
fn arrived_counts(dev: &inference_tensor::CudaDevice, stream: &CudaStream) -> Result<Option<u64>> {
    let mut map = ARRIVED
        .get_or_init(Default::default)
        .lock()
        .expect("biased attention counts poisoned");
    if let Some(&ptr) = map.get(&dev.id()) {
        return Ok(Some(ptr));
    }
    if stream.capture_status()
        == Ok(inference_tensor::cuda_backend::cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE)
    {
        return Ok(None);
    }
    let mut counts = unsafe { dev.alloc_process_lifetime::<u32>(MAX_SPLIT_ROWS)? };
    dev.cuda_stream()
        .memset_zeros(&mut counts)
        .map_err(inference_tensor::Error::wrap)?;
    let ptr = {
        let (ptr, _guard) = counts.device_ptr(stream);
        ptr
    };
    // the process keeps the counts: graphs captured later read them
    std::mem::forget(counts);
    map.insert(dev.id(), ptr);
    Ok(Some(ptr))
}

/// Where the keys and values are.
pub enum Keys<'a> {
    /// f32 `(b, kv_heads, past + t, head_dim)`, rows contiguous (a cache's narrowed view)
    View { k: &'a Tensor, v: &'a Tensor },
    /// a paged cache `[blocks, kv_heads, block_size, head_dim]` (f32, f16 or bf16) and each sequence's block table
    /// row (u32 `(b, width)`)
    Paged {
        k_cache: &'a Tensor,
        v_cache: &'a Tensor,
        block_tables: &'a Tensor,
        // with a ring of this many blocks a sequence: each table row `(b, 1)` holds its ring's first block, and key
        // block k is block `first + k % ring` (a window within the ring); 0 for whole tables
        ring: usize,
    },
}

/// Each sequence's past: the positions before its first query.
pub enum Past<'a> {
    Fixed(usize),
    /// `lens[b] - minus`, `lens` u32 `(b,)` on the device (a decode step's context lengths, minus its queries)
    Lens {
        lens: &'a Tensor,
        minus: usize,
    },
}

/// The attention inputs on one CUDA device.
pub struct BiasedAttention<'a> {
    /// f32 `(b, t, heads, head_dim)`, query `i` at position `past + i`
    pub q: &'a Tensor,
    pub keys: Keys<'a>,
    pub past: Past<'a>,
    /// f32 `(b, t, heads, extent + 1)`: each query's bias at distances `0..extent` (none past them; the last column
    /// unread)
    pub table: &'a Tensor,
    /// f32 `(b, t, heads)`: each query's sink logit, a weight with a zero value
    pub sinks: &'a Tensor,
    /// keys this many or more back are hidden; 0 for none
    pub window: usize,
    pub scale: f32,
    /// `(alpha, floor)`: each query's score, the sink's aside, times `1 + alpha ln(max(1, (pos + 1) / floor))`
    pub log_scale: Option<(f32, f32)>,
}

// the kernel's shape argument, as `BiasedAttentionShape` lays it out
#[repr(C)]
struct LaunchShape {
    batch: i32,
    heads: i32,
    kv_heads: i32,
    t: i32,
    head_dim: i32,
    extent: i32,
    window: i32,
    scale: f32,
    log_alpha: f32,
    log_floor: f32,
    past_lens: *const u32,
    past_sub: i32,
    past: i32,
    decode_rows: i32,
    // a split decode's partials and its rows' counts, or null
    partials: *mut f32,
    arrived: *mut u32,
}

fn i32_of(v: usize) -> Result<i32> {
    i32::try_from(v).map_err(inference_tensor::Error::wrap)
}

fn i64_of(v: usize) -> Result<i64> {
    i64::try_from(v).map_err(inference_tensor::Error::wrap)
}

// a K or V view's batch and head strides, its rows contiguous
fn kv_strides(layout: &Layout, head_dim: usize) -> Option<(usize, usize)> {
    let stride = layout.stride();
    (stride.len() == 4
        && stride[3] == 1
        && stride[2] == head_dim
        && layout.start_offset().is_multiple_of(4)
        && stride[1].is_multiple_of(4)
        && stride[0].is_multiple_of(4))
    .then(|| (stride[0], stride[1]))
}

// a tensor's first element on the device and the guard to hold until the launch is queued
fn device_ptr<'a>(
    storage: &'a Storage,
    layout: &Layout,
    stream: &'a CudaStream,
) -> Result<(u64, SyncOnDrop<'a>)> {
    let Storage::Cuda(s) = storage else {
        inference_tensor::bail!("biased attention needs CUDA storage");
    };
    let offset = layout.start_offset() as u64;
    Ok(match s.dtype() {
        DType::F32 => {
            let (p, g) = s.as_cuda_slice::<f32>()?.device_ptr(stream);
            (p + offset * 4, g)
        }
        DType::F16 => {
            let (p, g) = s.as_cuda_slice::<half::f16>()?.device_ptr(stream);
            (p + offset * 2, g)
        }
        DType::BF16 => {
            let (p, g) = s.as_cuda_slice::<half::bf16>()?.device_ptr(stream);
            (p + offset * 2, g)
        }
        DType::U32 => {
            let (p, g) = s.as_cuda_slice::<u32>()?.device_ptr(stream);
            (p + offset * 4, g)
        }
        DType::F8E4M3 => {
            let (p, g) = s.as_cuda_slice::<float8::F8E4M3>()?.device_ptr(stream);
            (p + offset, g)
        }
        DType::U8 => {
            let (p, g) = s.as_cuda_slice::<u8>()?.device_ptr(stream);
            (p + offset, g)
        }
        dtype => inference_tensor::bail!("biased attention does not take {dtype:?}"),
    })
}

/// `(b, t, heads * head_dim)` f32, the layout the output gate and projection take; `None` where the kernel does not
/// take the head size, the dtypes or the K/V layouts.
pub fn biased_attention(op: &BiasedAttention<'_>) -> Result<Option<Tensor>> {
    // a q8 cache holds K and V Hadamard-rotated: Q rotated as K was keeps the scores, and the output rotated as V was
    // undoes it
    if let Keys::Paged {
        k_cache,
        v_cache,
        block_tables,
        ring,
    } = &op.keys
        && k_cache.dtype() == DType::U8
    {
        let d = op.q.dim(3)?;
        if !inference_paged_attn::quant_cache_bits(k_cache, v_cache, d)
            .is_some_and(|(k, v)| k == v && (k == Q8_BITS || PACKED_BITS.contains(&k)))
        {
            return Ok(None);
        }
        let q =
            inference_paged_attn::hadamard(op.q, inference_paged_attn::kv_quant::k_rotation(d))?;
        let past = match &op.past {
            Past::Fixed(past) => Past::Fixed(*past),
            Past::Lens { lens, minus } => Past::Lens {
                lens,
                minus: *minus,
            },
        };
        let rotated = BiasedAttention {
            q: &q,
            keys: Keys::Paged {
                k_cache,
                v_cache,
                block_tables,
                ring: *ring,
            },
            past,
            table: op.table,
            sinks: op.sinks,
            window: op.window,
            scale: op.scale,
            log_scale: op.log_scale,
        };
        return biased_attention_rows(&rotated)?
            .map(|out| {
                inference_paged_attn::hadamard(&out, inference_paged_attn::kv_quant::v_rotation(d))
            })
            .transpose();
    }
    biased_attention_rows(op)
}

fn biased_attention_rows(op: &BiasedAttention<'_>) -> Result<Option<Tensor>> {
    let Device::Cuda(dev) = op.q.device() else {
        return Ok(None);
    };
    let (b, t, heads, d) = op.q.dims4()?;
    let extent = op.table.dim(3)? - 1;
    let kv_heads = match &op.keys {
        Keys::View { k, .. } => k.dim(1)?,
        Keys::Paged { k_cache, .. } => k_cache.dim(1)?,
    };
    if op.table.dims4()? != (b, t, heads, extent + 1)
        || op.sinks.dims3()? != (b, t, heads)
        || kv_heads == 0
        || !heads.is_multiple_of(kv_heads)
    {
        inference_tensor::bail!(
            "biased attention shapes q {:?} table {:?} sinks {:?} kv heads {kv_heads}",
            op.q.shape(),
            op.table.shape(),
            op.sinks.shape()
        );
    }
    let f32_inputs = [op.q, op.table, op.sinks]
        .iter()
        .all(|x| x.dtype() == DType::F32);
    if !HEAD_DIMS.contains(&d) || !f32_inputs {
        return Ok(None);
    }
    let (past_lens, past_sub, past) = match &op.past {
        Past::Fixed(past) => (None, 0, *past),
        Past::Lens { lens, minus } => {
            if lens.dtype() != DType::U32 || lens.elem_count() != b {
                inference_tensor::bail!(
                    "biased attention takes {b} u32 lengths, not {:?}",
                    lens.shape()
                );
            }
            (Some(lens.contiguous()?), *minus, 0)
        }
    };
    let (q, table, sinks) = (
        op.q.contiguous()?,
        op.table.contiguous()?,
        op.sinks.contiguous()?,
    );
    let stream = dev.cuda_stream();
    // SAFETY: the kernel writes every output element
    let mut out = unsafe { dev.alloc::<f32>(b * t * heads * d)? };
    {
        let (qg, ql) = q.storage_and_layout();
        let (q_ptr, _q) = device_ptr(&qg, ql, &stream)?;
        let (tg, tl) = table.storage_and_layout();
        let (t_ptr, _t) = device_ptr(&tg, tl, &stream)?;
        let (sg, sl) = sinks.storage_and_layout();
        let (s_ptr, _s) = device_ptr(&sg, sl, &stream)?;
        let lens_guard = past_lens.as_ref().map(|l| l.storage_and_layout());
        let lens_ptr = lens_guard
            .as_ref()
            .map(|(g, l)| device_ptr(g, l, &stream))
            .transpose()?;
        let arrived = if t <= DECODE_ROWS as usize && b * t * heads <= MAX_SPLIT_ROWS {
            arrived_counts(dev, &stream)?
        } else {
            None
        };
        // SAFETY: a size query
        let partial_len = if arrived.is_some() {
            usize::try_from(unsafe {
                super::ffi::biased_attention_decode_partials(
                    i32_of(b)?,
                    i32_of(t)?,
                    i32_of(heads)?,
                    i32_of(kv_heads)?,
                    i32_of(d)?,
                )
            })
            .map_err(inference_tensor::Error::wrap)?
        } else {
            0
        };
        // SAFETY: each split writes its partials before the combine reads them
        let mut partials = (partial_len > 0)
            .then(|| unsafe { dev.alloc::<f32>(partial_len) })
            .transpose()?;
        let partials_guard = partials.as_mut().map(|p| p.device_ptr_mut(&stream));
        let (o_ptr, _o) = out.device_ptr_mut(&stream);
        let shape = LaunchShape {
            batch: i32_of(b)?,
            heads: i32_of(heads)?,
            kv_heads: i32_of(kv_heads)?,
            t: i32_of(t)?,
            head_dim: i32_of(d)?,
            extent: i32_of(extent)?,
            window: i32_of(op.window)?,
            scale: op.scale,
            log_alpha: op.log_scale.map_or(0.0, |(alpha, _)| alpha),
            log_floor: op.log_scale.map_or(1.0, |(_, floor)| floor),
            past_lens: lens_ptr
                .as_ref()
                .map_or(std::ptr::null(), |(p, _)| *p as *const u32),
            past_sub: i32_of(past_sub)?,
            past: i32_of(past)?,
            decode_rows: DECODE_ROWS,
            partials: partials_guard
                .as_ref()
                .map_or(std::ptr::null_mut(), |(p, _)| *p as *mut f32),
            arrived: arrived.map_or(std::ptr::null_mut(), |p| p as *mut u32),
        };
        let cu = stream.cu_stream() as i64;
        let code = match &op.keys {
            Keys::View { k, v } => {
                let keys = past + t;
                if k.dtype() != DType::F32
                    || v.dtype() != DType::F32
                    || k.dims4()? != (b, kv_heads, keys, d)
                    || v.dims4()? != (b, kv_heads, keys, d)
                    || past_lens.is_some()
                {
                    return Ok(None);
                }
                let (kg, kl) = k.storage_and_layout();
                let (vg, vl) = v.storage_and_layout();
                let (Some((k_batch, k_head)), Some((v_batch, v_head))) =
                    (kv_strides(kl, d), kv_strides(vl, d))
                else {
                    return Ok(None);
                };
                let (k_ptr, _k) = device_ptr(&kg, kl, &stream)?;
                let (v_ptr, _v) = device_ptr(&vg, vl, &stream)?;
                // SAFETY: the pointers cover the shapes and strides checked above, all on this stream
                unsafe {
                    super::ffi::biased_attention_f32(
                        q_ptr as *const f32,
                        k_ptr as *const f32,
                        v_ptr as *const f32,
                        t_ptr as *const f32,
                        s_ptr as *const f32,
                        o_ptr as *mut f32,
                        (&raw const shape).cast(),
                        i64_of(k_batch)?,
                        i64_of(k_head)?,
                        i64_of(v_batch)?,
                        i64_of(v_head)?,
                        cu,
                    )
                }
            }
            Keys::Paged {
                k_cache,
                v_cache,
                block_tables,
                ring,
            } => {
                let (_, cache_heads, block_size, cache_d) = k_cache.dims4()?;
                let (cache_dtype, cache_bits, row) = match k_cache.dtype() {
                    DType::F32 => (CACHE_F32, 0, d),
                    DType::F16 => (CACHE_F16, 0, d),
                    DType::BF16 => (CACHE_BF16, 0, d),
                    DType::F8E4M3 => (CACHE_E4M3, 0, d),
                    DType::U8 => {
                        let Some((bits, _)) =
                            inference_paged_attn::quant_cache_bits(k_cache, v_cache, d)
                        else {
                            return Ok(None);
                        };
                        let code = if bits == Q8_BITS {
                            CACHE_Q8
                        } else {
                            CACHE_PACKED
                        };
                        (
                            code,
                            bits as i32,
                            inference_paged_attn::kv_quant::row_bytes(d, bits),
                        )
                    }
                    _ => return Ok(None),
                };
                let tables_l = block_tables.layout();
                let (rows, width) = block_tables.dims2()?;
                if (cache_heads, cache_d) != (kv_heads, row)
                    || v_cache.shape() != k_cache.shape()
                    || v_cache.dtype() != k_cache.dtype()
                    || !k_cache.is_contiguous()
                    || !v_cache.is_contiguous()
                    || block_tables.dtype() != DType::U32
                    || rows != b
                    || width == 0
                    || tables_l.stride()[1] != 1
                {
                    return Ok(None);
                }
                let table_stride = tables_l.stride()[0];
                let (kg, kl) = k_cache.storage_and_layout();
                let (vg, vl) = v_cache.storage_and_layout();
                let (bg, bl) = block_tables.storage_and_layout();
                let (k_ptr, _k) = device_ptr(&kg, kl, &stream)?;
                let (v_ptr, _v) = device_ptr(&vg, vl, &stream)?;
                let (b_ptr, _b) = device_ptr(&bg, bl, &stream)?;
                // SAFETY: the pointers cover the caches, the block tables and the shapes checked above, all on
                // this stream
                unsafe {
                    super::ffi::biased_attention_paged(
                        q_ptr as *const f32,
                        k_ptr as *const core::ffi::c_void,
                        v_ptr as *const core::ffi::c_void,
                        cache_dtype,
                        cache_bits,
                        b_ptr as *const u32,
                        i64_of(table_stride)?,
                        i32_of(width)?,
                        i32_of(block_size)?,
                        i32_of(*ring)?,
                        t_ptr as *const f32,
                        s_ptr as *const f32,
                        o_ptr as *mut f32,
                        (&raw const shape).cast(),
                        cu,
                    )
                }
            }
        };
        if code != 0 {
            inference_tensor::bail!("biased attention failed with CUDA status {code}");
        }
    }
    Ok(Some(Tensor::from((
        Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone())),
        Shape::from_dims(&[b, t, heads * d]),
    ))))
}

// test shapes are small counts, exact in the casts
#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
mod tests {
    use super::*;

    // shuffled blocks a paged case's sequences take, as the engine's block tables hand them out
    const TEST_BLOCK_SIZE: usize = 16;
    // biased_attention.cu's VERIFY_MIN_KEYS: a sequence's verify rows from this many keys take the tensor-core kernel
    const VERIFY_MIN_KEYS: usize = 8192;
    // biased_attention.cu's MMA_MIN_QUERIES: prompts from this many queries run on tensor cores (sm80+)
    const TENSOR_CORE_QUERIES: usize = 32;
    const F32_TOLERANCE: f32 = 1e-4;
    const TENSOR_CORE_TOLERANCE: f32 = 1e-3;

    struct Case {
        b: usize,
        h: usize,
        hkv: usize,
        t: usize,
        past: usize,
        d: usize,
        extent: usize,
        window: usize,
        log_scale: Option<(f32, f32)>,
        // q, K and V's spread, the scores' kept fixed: activations of small or large magnitude
        magnitude: f32,
        // a paged cache's window ring of this many blocks a sequence (0: the whole sequence in its block table)
        ring: usize,
    }

    // the materialized form: scores plus the bias gathered by distance, scaled, masked, a sink column, softmax,
    // values; k and v `(b, hkv, past + t, d)` f32
    fn reference(
        c: &Case,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        table: &Tensor,
        sinks: &Tensor,
    ) -> Result<Tensor> {
        let (b, h, t, d) = (c.b, c.h, c.t, c.d);
        let s = c.past + t;
        let q = q.transpose(1, 2)?.contiguous()?;
        let rows = table
            .transpose(1, 2)?
            .contiguous()?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mut bias = vec![0f32; b * h * t * s];
        let mut taus = vec![1f32; b * h * t * s];
        for bh in 0..b * h {
            for i in 0..t {
                let pos = c.past + i;
                let tau = c.log_scale.map_or(1.0, |(alpha, floor)| {
                    1.0 + alpha * ((pos + 1) as f32 / floor).max(1.0).ln()
                });
                for j in 0..s {
                    let visible = j <= pos && (c.window == 0 || pos - j < c.window);
                    let at = (bh * t + i) * s + j;
                    taus[at] = tau;
                    bias[at] = if !visible {
                        f32::NEG_INFINITY
                    } else if pos - j < c.extent {
                        rows[(bh * t + i) * (c.extent + 1) + pos - j]
                    } else {
                        0.0
                    };
                }
            }
        }
        let device = q.device();
        let bias = Tensor::from_vec(bias, (b, h, t, s), device)?;
        let taus = Tensor::from_vec(taus, (b, h, t, s), device)?;
        let group = h / c.hkv;
        let k = k.repeat_interleave_for_test(group)?;
        let v = v.repeat_interleave_for_test(group)?;
        let scale = 1.0 / (d as f64).sqrt();
        let scores = (((q.matmul(&k.t()?)? * scale)? + bias)? * taus)?;
        let sinks = sinks.transpose(1, 2)?.contiguous()?.unsqueeze(3)?;
        let probs =
            inference_tensor::nn::ops::softmax_last_dim(&Tensor::cat(&[&scores, &sinks], 3)?)?;
        probs
            .narrow(3, 0, s)?
            .contiguous()?
            .matmul(&v.contiguous()?)?
            .transpose(1, 2)?
            .reshape((b, t, h * d))
    }

    trait RepeatHeads {
        fn repeat_interleave_for_test(&self, group: usize) -> Result<Tensor>;
    }

    impl RepeatHeads for Tensor {
        fn repeat_interleave_for_test(&self, group: usize) -> Result<Tensor> {
            let (b, hkv, s, d) = self.dims4()?;
            self.unsqueeze(2)?
                .broadcast_as((b, hkv, group, s, d))?
                .contiguous()?
                .reshape((b, hkv * group, s, d))
        }
    }

    // `x` (b, hkv, s, d) laid into a paged cache of `dtype` through shuffled blocks: the cache and the block tables
    fn paged(x: &Tensor, dtype: DType) -> Result<(Tensor, Tensor)> {
        let (b, hkv, s, d) = x.dims4()?;
        let per_seq = s.div_ceil(TEST_BLOCK_SIZE);
        let blocks = b * per_seq + 1;
        // a fixed shuffle: sequence-major order reversed, block 0 left unused
        let ids = (0..b * per_seq)
            .map(|i| (b * per_seq - i) as u32)
            .collect::<Vec<_>>();
        let values = x.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        let mut cache = vec![0f32; blocks * hkv * TEST_BLOCK_SIZE * d];
        for bi in 0..b {
            for h in 0..hkv {
                for key in 0..s {
                    let block = ids[bi * per_seq + key / TEST_BLOCK_SIZE] as usize;
                    let dst = ((block * hkv + h) * TEST_BLOCK_SIZE + key % TEST_BLOCK_SIZE) * d;
                    let src = ((bi * hkv + h) * s + key) * d;
                    cache[dst..dst + d].copy_from_slice(&values[src..src + d]);
                }
            }
        }
        let device = x.device();
        let cache =
            Tensor::from_vec(cache, (blocks, hkv, TEST_BLOCK_SIZE, d), device)?.to_dtype(dtype)?;
        let tables = Tensor::from_vec(ids, (b, per_seq), device)?;
        Ok((cache, tables))
    }

    // `x` (b, hkv, s, d) written in key order into a window ring of `ring` blocks a sequence (key block k in ring block
    // k % ring, later keys over earlier ones), each sequence's ring after the last: the cache and each ring's first block
    fn ringed(x: &Tensor, dtype: DType, ring: usize) -> Result<(Tensor, Tensor)> {
        let (b, hkv, s, d) = x.dims4()?;
        let blocks = b * ring + 1;
        let values = x.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        let mut cache = vec![0f32; blocks * hkv * TEST_BLOCK_SIZE * d];
        for bi in 0..b {
            for h in 0..hkv {
                for key in 0..s {
                    let block = 1 + bi * ring + key / TEST_BLOCK_SIZE % ring;
                    let dst = ((block * hkv + h) * TEST_BLOCK_SIZE + key % TEST_BLOCK_SIZE) * d;
                    let src = ((bi * hkv + h) * s + key) * d;
                    cache[dst..dst + d].copy_from_slice(&values[src..src + d]);
                }
            }
        }
        let device = x.device();
        let cache =
            Tensor::from_vec(cache, (blocks, hkv, TEST_BLOCK_SIZE, d), device)?.to_dtype(dtype)?;
        let firsts = (0..b).map(|bi| (1 + bi * ring) as u32).collect::<Vec<_>>();
        Ok((cache, Tensor::from_vec(firsts, (b, 1), device)?))
    }

    // a case's inputs over a quantized cache that paged-attn's writer filled (rotated, `bits` a value), through shuffled
    // blocks
    struct QuantInputs {
        q: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        tables: Tensor,
        lens: Tensor,
        table: Tensor,
        sinks: Tensor,
    }

    impl QuantInputs {
        fn new(c: &Case, bits: u32, device: &Device) -> Result<(Self, Tensor, Tensor)> {
            let s = c.past + c.t;
            let rand = |dims: &[usize]| Tensor::randn(0f32, 1.0, dims, device);
            let spread = |dims: &[usize], by: f32| rand(dims)? * f64::from(by);
            let q = spread(&[c.b, c.t, c.h, c.d], 1.0 / c.magnitude)?;
            let (k, v) = (
                spread(&[c.b, c.hkv, s, c.d], c.magnitude)?,
                spread(&[c.b, c.hkv, s, c.d], c.magnitude)?,
            );
            let table = rand(&[c.b, c.t, c.h, c.extent + 1])?;
            let sinks = rand(&[c.b, c.t, c.h])?;
            let per_seq = s.div_ceil(TEST_BLOCK_SIZE);
            let blocks = c.b * per_seq + 1;
            // the same shuffle as `paged`: block 0 unused
            let ids = (0..c.b * per_seq)
                .map(|i| (c.b * per_seq - i) as u32)
                .collect::<Vec<_>>();
            let slots = (0..c.b)
                .flat_map(|bi| (0..s).map(move |key| (bi, key)))
                .map(|(bi, key)| {
                    i64::from(ids[bi * per_seq + key / TEST_BLOCK_SIZE]) * TEST_BLOCK_SIZE as i64
                        + (key % TEST_BLOCK_SIZE) as i64
                })
                .collect::<Vec<_>>();
            let row = inference_paged_attn::kv_quant::row_bytes(c.d, bits);
            let cache = || Tensor::zeros((blocks, c.hkv, TEST_BLOCK_SIZE, row), DType::U8, device);
            let (k_cache, v_cache) = (cache()?, cache()?);
            let tokens = |x: &Tensor| -> Result<Tensor> {
                x.permute((0, 2, 1, 3))?
                    .contiguous()?
                    .reshape((c.b * s, c.hkv, c.d))
            };
            inference_paged_attn::reshape_and_cache_quant(
                &tokens(&k)?,
                &tokens(&v)?,
                &k_cache,
                &v_cache,
                &Tensor::from_vec(slots, c.b * s, device)?,
            )?;
            let tables = Tensor::from_vec(ids, (c.b, per_seq), device)?;
            let lens = Tensor::from_vec(vec![s as u32; c.b], c.b, device)?;
            let inputs = Self {
                q,
                k_cache,
                v_cache,
                tables,
                lens,
                table,
                sinks,
            };
            Ok((inputs, k, v))
        }

        fn op<'a>(&'a self, c: &Case) -> BiasedAttention<'a> {
            BiasedAttention {
                q: &self.q,
                keys: Keys::Paged {
                    k_cache: &self.k_cache,
                    v_cache: &self.v_cache,
                    block_tables: &self.tables,
                    ring: 0,
                },
                past: Past::Lens {
                    lens: &self.lens,
                    minus: c.t,
                },
                table: &self.table,
                sinks: &self.sinks,
                window: c.window,
                scale: 1.0 / (c.d as f32).sqrt(),
                log_scale: c.log_scale,
            }
        }
    }

    // the kernel over a quantized cache of `bits` against the reference over the values its gather reads back
    fn run_quant(c: &Case, bits: u32) -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        let s = c.past + c.t;
        let (inputs, _, _) = QuantInputs::new(c, bits, &device)?;
        let cu = Tensor::from_vec(
            (0..=c.b).map(|i| (i * s) as u32).collect::<Vec<_>>(),
            c.b + 1,
            &device,
        )?;
        let (kg, vg) = inference_paged_attn::gather_kv_cache_quant(
            &inputs.k_cache,
            &inputs.v_cache,
            &inputs.tables,
            &cu,
            c.b * s,
            c.d,
            DType::F32,
        )?;
        let heads_first = |x: Tensor| -> Result<Tensor> {
            x.reshape((c.b, s, c.hkv, c.d))?
                .permute((0, 2, 1, 3))?
                .contiguous()
        };
        let (kg, vg) = (heads_first(kg)?, heads_first(vg)?);
        let got = biased_attention(&inputs.op(c))?.expect("the kernel takes the quantized cache");
        let want = reference(c, &inputs.q, &kg, &vg, &inputs.table, &inputs.sinks)?;
        let err = (got - &want)?.abs()?.max_all()?.to_scalar::<f32>()?;
        let mag = want.abs()?.max_all()?.to_scalar::<f32>()?;
        let tolerance = if c.t >= TENSOR_CORE_QUERIES {
            TENSOR_CORE_TOLERANCE
        } else {
            F32_TOLERANCE
        };
        assert!(
            err <= tolerance * mag.max(c.magnitude),
            "t {} past {}: q{bits} max error {err} of {mag}",
            c.t,
            c.past
        );
        Ok(())
    }

    // an Inkling 2 full-context decode over a q8 cache at long context: a decode row and verify rows up to 7 drafts,
    // microseconds a call and the K/V bytes a second; the cache bytes are drawn directly (no f32 copies to write)
    #[test]
    #[ignore = "benchmark"]
    fn bench_q8_long_context_decode() -> Result<()> {
        const PAST: usize = 102_400;
        const WARMUP: usize = 5;
        const CALLS: usize = 50;
        // an f16 scale of about 0.01 for every group
        const SCALE_BITS: u16 = 0x211E;
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        let (h, hkv, d) = (24, 8, 128);
        let row = inference_paged_attn::kv_quant::row_bytes(d, Q8_BITS);
        for t in [1, 2, 4, DECODE_ROWS as usize] {
            let s = PAST + t;
            let blocks = s.div_ceil(TEST_BLOCK_SIZE);
            let mut seed = 0x2545_f491_u64;
            let mut bytes = vec![0u8; blocks * hkv * TEST_BLOCK_SIZE * row];
            for (i, b) in bytes.iter_mut().enumerate() {
                let at = i % row;
                *b = if at < d {
                    seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    (seed >> 56) as u8
                } else {
                    SCALE_BITS.to_le_bytes()[(at - d) % 2]
                };
            }
            let shape4 = (blocks, hkv, TEST_BLOCK_SIZE, row);
            let k_cache = Tensor::from_vec(bytes.clone(), shape4, &device)?;
            let v_cache = Tensor::from_vec(bytes, shape4, &device)?;
            let inputs = QuantInputs {
                q: Tensor::randn(0f32, 1.0, (1, t, h, d), &device)?,
                k_cache,
                v_cache,
                tables: Tensor::from_vec(
                    (0..blocks as u32).collect::<Vec<_>>(),
                    (1, blocks),
                    &device,
                )?,
                lens: Tensor::from_vec(vec![s as u32], 1, &device)?,
                table: Tensor::randn(0f32, 1.0, (1, t, h, 257), &device)?,
                sinks: Tensor::randn(0f32, 1.0, (1, t, h), &device)?,
            };
            let mut c = shape((1, h, hkv, t, PAST, d, 256, 0));
            c.log_scale = Some((0.1, 64.0));
            let op = inputs.op(&c);
            for _ in 0..WARMUP {
                biased_attention(&op)?;
            }
            device.synchronize()?;
            let start = std::time::Instant::now();
            for _ in 0..CALLS {
                biased_attention(&op)?;
            }
            device.synchronize()?;
            let us = start.elapsed().as_secs_f64() * 1e6 / CALLS as f64;
            let read = 2 * hkv * s * row;
            println!(
                "t {t}: {us:.1} us a call, {:.0} GB/s",
                read as f64 / us / 1e3
            );
        }
        Ok(())
    }

    // a q8 cache in every kernel: decode, a split long-past decode, the scalar and the tensor-core prefill
    fn cuda_paged_q8_matches_the_reference() -> Result<()> {
        let mut long = shape((2, 8, 2, 1, 2000, 128, 256, 1500));
        long.log_scale = Some((0.1, 64.0));
        for c in [
            shape((2, 8, 2, 1, 700, 128, 256, 0)),
            long,
            shape((2, 8, 2, 4, 300, 64, 32, 0)),
            shape((1, 8, 2, 100, 50, 64, 40, 0)),
        ] {
            run_quant(&c, Q8_BITS)?;
        }
        Ok(())
    }

    // the packed widths in every kernel: decode, a split long-past decode, grouped heads, the tensor-core verify rows,
    // the scalar and the tensor-core prefill
    fn cuda_paged_packed_widths_match_the_reference() -> Result<()> {
        let cases = || {
            let mut long = shape((2, 8, 2, 1, 2000, 128, 256, 1500));
            long.log_scale = Some((0.1, 64.0));
            [
                shape((2, 8, 2, 1, 700, 128, 256, 0)),
                long,
                shape((2, 8, 2, 4, 300, 64, 32, 0)),
                shape((1, 8, 2, 100, 50, 64, 40, 0)),
                shape((2, 24, 8, 1, 2000, 128, 256, 0)),
                shape((
                    1,
                    24,
                    8,
                    DECODE_ROWS as usize,
                    VERIFY_MIN_KEYS + 100,
                    128,
                    256,
                    0,
                )),
            ]
        };
        for bits in PACKED_BITS {
            for c in cases() {
                run_quant(&c, bits)?;
            }
        }
        Ok(())
    }

    // the kernel against the reference, on a contiguous view (K/V narrowed from a longer cache) or a paged cache of
    // `paged_dtype` with the past from a device array
    fn run(c: &Case, paged_dtype: Option<DType>) -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        let s = c.past + c.t;
        let rand = |dims: &[usize]| Tensor::randn(0f32, 1.0, dims, &device);
        let spread = |dims: &[usize], by: f32| rand(dims)? * f64::from(by);
        let q = spread(&[c.b, c.t, c.h, c.d], 1.0 / c.magnitude)?;
        let k = spread(&[c.b, c.hkv, s + 16, c.d], c.magnitude)?.narrow(2, 0, s)?;
        let v = spread(&[c.b, c.hkv, s + 16, c.d], c.magnitude)?.narrow(2, 0, s)?;
        let table = rand(&[c.b, c.t, c.h, c.extent + 1])?;
        let sinks = rand(&[c.b, c.t, c.h])?;
        let scale = 1.0 / (c.d as f32).sqrt();
        let lens = Tensor::from_vec(vec![s as u32; c.b], c.b, &device)?;
        let (got, want) = match paged_dtype {
            None => {
                let op = BiasedAttention {
                    q: &q,
                    keys: Keys::View { k: &k, v: &v },
                    past: Past::Fixed(c.past),
                    table: &table,
                    sinks: &sinks,
                    window: c.window,
                    scale,
                    log_scale: c.log_scale,
                };
                let got = biased_attention(&op)?.expect("the kernel takes these shapes");
                (got, reference(c, &q, &k, &v, &table, &sinks)?)
            }
            Some(dtype) => {
                // the reference reads the values the cache holds
                let (k, v) = (
                    k.to_dtype(dtype)?.to_dtype(DType::F32)?,
                    v.to_dtype(dtype)?.to_dtype(DType::F32)?,
                );
                let layout = |x: &Tensor| {
                    if c.ring > 0 {
                        ringed(x, dtype, c.ring)
                    } else {
                        paged(x, dtype)
                    }
                };
                let (k_cache, block_tables) = layout(&k)?;
                let (v_cache, _) = layout(&v)?;
                let op = BiasedAttention {
                    q: &q,
                    keys: Keys::Paged {
                        k_cache: &k_cache,
                        v_cache: &v_cache,
                        block_tables: &block_tables,
                        ring: c.ring,
                    },
                    past: Past::Lens {
                        lens: &lens,
                        minus: c.t,
                    },
                    table: &table,
                    sinks: &sinks,
                    window: c.window,
                    scale,
                    log_scale: c.log_scale,
                };
                let got = biased_attention(&op)?.expect("the kernel takes these shapes");
                (got, reference(c, &q, &k, &v, &table, &sinks)?)
            }
        };
        let err = (got - &want)?.abs()?.max_all()?.to_scalar::<f32>()?;
        let mag = want.abs()?.max_all()?.to_scalar::<f32>()?;
        // the tensor-core prefill (half-precision caches) sums f16 products of q and the probabilities in two parts
        let tolerance =
            if c.t >= TENSOR_CORE_QUERIES && paged_dtype.is_some_and(|d| d != DType::F32) {
                TENSOR_CORE_TOLERANCE
            } else {
                F32_TOLERANCE
            };
        assert!(
            err <= tolerance * mag.max(c.magnitude),
            "b {} h {} t {} past {} d {} paged {paged_dtype:?}: max error {err} of {mag}",
            c.b,
            c.h,
            c.t,
            c.past,
            c.d
        );
        Ok(())
    }

    fn shape(
        (b, h, hkv, t, past, d, extent, window): (
            usize,
            usize,
            usize,
            usize,
            usize,
            usize,
            usize,
            usize,
        ),
    ) -> Case {
        Case {
            b,
            h,
            hkv,
            t,
            past,
            d,
            extent,
            window,
            log_scale: None,
            magnitude: 1.0,
            ring: 0,
        }
    }

    fn cuda_prefill_matches_the_reference() -> Result<()> {
        run(&shape((2, 8, 2, 37, 0, 128, 16, 0)), None)
    }

    fn cuda_prefill_after_a_past_matches_the_reference() -> Result<()> {
        run(&shape((1, 4, 4, 21, 50, 64, 40, 0)), None)
    }

    fn cuda_windowed_prefill_matches_the_reference() -> Result<()> {
        run(&shape((1, 8, 2, 70, 30, 128, 64, 48)), None)
    }

    fn cuda_decode_matches_the_reference() -> Result<()> {
        run(&shape((2, 8, 2, 1, 700, 128, 256, 0)), None)
    }

    // rows over several DECODE_SPLIT_KEYS of keys split them over blocks: verify rows and a decode row, windowed, from
    // a view and a paged cache
    fn cuda_split_decode_over_long_pasts_matches_the_reference() -> Result<()> {
        // a 7-draft verify's rows, a 3-draft one's, a decode row
        for t in [DECODE_ROWS as usize, 4, 1] {
            let mut c = shape((2, 8, 2, t, 2000, 128, 256, 1500));
            c.log_scale = Some((0.1, 64.0));
            run(&c, None)?;
            run(&c, Some(DType::BF16))?;
        }
        Ok(())
    }

    // an E4M3 cache's bytes widened in every kernel: decode, a split long-past decode, the tensor-core prefill
    fn cuda_paged_e4m3_matches_the_reference() -> Result<()> {
        let mut long = shape((2, 8, 2, 1, 2000, 128, 256, 1500));
        long.log_scale = Some((0.1, 64.0));
        for c in [
            shape((2, 8, 2, 1, 700, 128, 256, 0)),
            long,
            shape((1, 8, 2, 100, 50, 64, 40, 0)),
            shape((2, 8, 2, 23, 41, 128, 32, 0)),
        ] {
            run(&c, Some(DType::F8E4M3))?;
        }
        Ok(())
    }

    fn cuda_windowed_decode_matches_the_reference() -> Result<()> {
        run(&shape((1, 4, 1, 1, 300, 32, 64, 100)), None)
    }

    fn cuda_short_decode_rows_match_the_reference() -> Result<()> {
        run(&shape((1, 4, 2, 3, 9, 64, 8, 0)), None)
    }

    fn cuda_log_scaled_queries_match_the_reference() -> Result<()> {
        let mut c = shape((1, 8, 2, 5, 40, 128, 16, 0));
        c.log_scale = Some((0.1, 8.0));
        run(&c, None)?;
        c.t = 1;
        run(&c, None)
    }

    fn cuda_paged_prefill_after_a_past_matches_the_reference() -> Result<()> {
        run(&shape((2, 8, 2, 23, 41, 128, 32, 0)), Some(DType::F32))
    }

    fn cuda_paged_decode_matches_the_reference() -> Result<()> {
        let mut c = shape((3, 8, 2, 1, 333, 128, 256, 100));
        c.log_scale = Some((0.1, 64.0));
        run(&c, Some(DType::F32))
    }

    fn cuda_tensor_core_prefill_after_a_past_matches_the_reference() -> Result<()> {
        let mut c = shape((1, 8, 2, 100, 50, 64, 40, 0));
        c.log_scale = Some((0.1, 8.0));
        run(&c, Some(DType::BF16))?;
        run(&shape((2, 4, 1, 40, 7, 32, 16, 0)), Some(DType::BF16))
    }

    fn cuda_tensor_core_paged_bf16_prefill_matches_the_reference() -> Result<()> {
        run(&shape((2, 8, 2, 77, 33, 128, 64, 60)), Some(DType::BF16))
    }

    fn cuda_tensor_core_tiny_shapes_match_the_reference() -> Result<()> {
        for window in [0, 4] {
            let mut c = shape((3, 2, 1, 40, 0, 64, 8, window));
            c.log_scale = Some((0.1, 8.0));
            run(&c, Some(DType::BF16))?;
            run(&c, None)?;
        }
        Ok(())
    }

    fn cuda_paged_bf16_decode_matches_the_reference() -> Result<()> {
        run(&shape((2, 4, 1, 1, 77, 64, 32, 0)), Some(DType::BF16))
    }

    // a block takes every head of a KV head (Inkling 2's 3 a KV head, on a view and a q8 cache, split long pasts
    // and verify rows); 8 heads a KV head take the per-head kernel
    fn cuda_grouped_decode_matches_the_reference() -> Result<()> {
        for t in [1, DECODE_ROWS as usize] {
            let mut c = shape((2, 24, 8, t, 2000, 128, 256, 0));
            c.log_scale = Some((0.1, 64.0));
            run(&c, None)?;
            run_quant(&c, Q8_BITS)?;
        }
        run(&shape((1, 8, 1, 1, 700, 64, 32, 0)), None)?;
        // past VERIFY_MIN_KEYS the verify rows take the tensor-core kernel
        let mut long = shape((
            2,
            24,
            8,
            DECODE_ROWS as usize,
            VERIFY_MIN_KEYS + 100,
            128,
            256,
            0,
        ));
        long.log_scale = Some((0.1, 64.0));
        run(&long, Some(DType::BF16))?;
        run_quant(&long, Q8_BITS)?;
        // a window shorter than the verify's rows, and a bias reaching past it; K and V far below and above 1, q
        // the other way
        for t in [2, DECODE_ROWS as usize] {
            for magnitude in [1.0, 1e-4, 1e3] {
                let mut c = shape((1, 2, 1, t, VERIFY_MIN_KEYS + 20, 64, 8, 4));
                c.magnitude = magnitude;
                run(&c, Some(DType::BF16))?;
                run(&c, Some(DType::F16))?;
                run_quant(&c, Q8_BITS)?;
            }
        }
        // the window in a ring of two blocks a sequence that wrapped over the first keys: a decode row and verify rows
        for (b, t) in [
            (1, 1),
            (1, 2),
            (1, DECODE_ROWS as usize),
            (2, DECODE_ROWS as usize),
        ] {
            let mut c = shape((b, 2, 1, t, VERIFY_MIN_KEYS + 40, 64, 8, 4));
            c.ring = 2;
            run(&c, Some(DType::BF16))?;
            run(&c, Some(DType::F16))?;
        }
        Ok(())
    }

    // under VERIFY_MIN_KEYS a verify's rows sum as decode rows do, bit for bit, so drafting keeps the plain greedy reply
    fn cuda_short_verify_rows_are_decode_rows_bit_for_bit() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        let c = shape((2, 24, 8, DECODE_ROWS as usize, 2000, 128, 256, 0));
        let s = c.past + c.t;
        let rand = |dims: &[usize]| Tensor::randn(0f32, 1.0, dims, &device);
        let q = rand(&[c.b, c.t, c.h, c.d])?;
        let (k_cache, tables) = paged(&rand(&[c.b, c.hkv, s, c.d])?, DType::BF16)?;
        let (v_cache, _) = paged(&rand(&[c.b, c.hkv, s, c.d])?, DType::BF16)?;
        let table = rand(&[c.b, c.t, c.h, c.extent + 1])?;
        let sinks = rand(&[c.b, c.t, c.h])?;
        let lens = Tensor::from_vec(vec![s as u32; c.b], c.b, &device)?;
        let attend = |rows: std::ops::Range<usize>| -> Result<Vec<u32>> {
            let n = rows.len();
            let (q, table, sinks) = (
                q.narrow(1, rows.start, n)?.contiguous()?,
                table.narrow(1, rows.start, n)?.contiguous()?,
                sinks.narrow(1, rows.start, n)?.contiguous()?,
            );
            let op = BiasedAttention {
                q: &q,
                keys: Keys::Paged {
                    k_cache: &k_cache,
                    v_cache: &v_cache,
                    block_tables: &tables,
                    ring: 0,
                },
                past: Past::Lens {
                    lens: &lens,
                    minus: c.t - rows.start,
                },
                table: &table,
                sinks: &sinks,
                window: 0,
                scale: 1.0 / (c.d as f32).sqrt(),
                log_scale: None,
            };
            let out = biased_attention(&op)?.expect("the kernel takes these shapes");
            Ok(out
                .flatten_all()?
                .to_vec1::<f32>()?
                .into_iter()
                .map(f32::to_bits)
                .collect())
        };
        let verify = attend(0..c.t)?;
        let per_row = c.h * c.d;
        for i in 0..c.t {
            let row = attend(i..i + 1)?;
            for bi in 0..c.b {
                let at = (bi * c.t + i) * per_row;
                assert!(
                    verify[at..at + per_row] == row[bi * per_row..(bi + 1) * per_row],
                    "batch {bi} row {i} differs from its decode"
                );
            }
        }
        Ok(())
    }

    inference_tensor::gpu_cases!(
        cuda_cases,
        [
            cuda_decode_matches_the_reference,
            cuda_short_verify_rows_are_decode_rows_bit_for_bit,
            cuda_grouped_decode_matches_the_reference,
            cuda_log_scaled_queries_match_the_reference,
            cuda_paged_bf16_decode_matches_the_reference,
            cuda_paged_decode_matches_the_reference,
            cuda_paged_prefill_after_a_past_matches_the_reference,
            cuda_prefill_after_a_past_matches_the_reference,
            cuda_prefill_matches_the_reference,
            cuda_short_decode_rows_match_the_reference,
            cuda_split_decode_over_long_pasts_matches_the_reference,
            cuda_paged_e4m3_matches_the_reference,
            cuda_paged_q8_matches_the_reference,
            cuda_paged_packed_widths_match_the_reference,
            cuda_tensor_core_paged_bf16_prefill_matches_the_reference,
            cuda_tensor_core_tiny_shapes_match_the_reference,
            cuda_tensor_core_prefill_after_a_past_matches_the_reference,
            cuda_windowed_decode_matches_the_reference,
            cuda_windowed_prefill_matches_the_reference
        ]
    );
}
