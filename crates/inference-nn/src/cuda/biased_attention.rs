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
// queries up to this many take the kernel that splits the keys across warps (decode)
const DECODE_ROWS: i32 = 4;
// the paged kernel's cache dtype codes
const CACHE_F32: i32 = 0;
const CACHE_F16: i32 = 1;
const CACHE_BF16: i32 = 2;

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
        dtype => inference_tensor::bail!("biased attention does not take {dtype:?}"),
    })
}

/// `(b, t, heads * head_dim)` f32, the layout the output gate and projection take; `None` where the kernel does not
/// take the head size, the dtypes or the K/V layouts.
pub fn biased_attention(op: &BiasedAttention<'_>) -> Result<Option<Tensor>> {
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
            } => {
                let (_, cache_heads, block_size, cache_d) = k_cache.dims4()?;
                let cache_dtype = match k_cache.dtype() {
                    DType::F32 => CACHE_F32,
                    DType::F16 => CACHE_F16,
                    DType::BF16 => CACHE_BF16,
                    _ => return Ok(None),
                };
                let tables_l = block_tables.layout();
                let (rows, width) = block_tables.dims2()?;
                if (cache_heads, cache_d) != (kv_heads, d)
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
                        b_ptr as *const u32,
                        i64_of(table_stride)?,
                        i32_of(width)?,
                        i32_of(block_size)?,
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

    // the kernel against the reference, on a contiguous view (K/V narrowed from a longer cache) or a paged cache of
    // `paged_dtype` with the past from a device array
    fn run(c: &Case, paged_dtype: Option<DType>) -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        let s = c.past + c.t;
        let rand = |dims: &[usize]| Tensor::randn(0f32, 1.0, dims, &device);
        let q = rand(&[c.b, c.t, c.h, c.d])?;
        let k = rand(&[c.b, c.hkv, s + 16, c.d])?.narrow(2, 0, s)?;
        let v = rand(&[c.b, c.hkv, s + 16, c.d])?.narrow(2, 0, s)?;
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
                let (k_cache, block_tables) = paged(&k, dtype)?;
                let (v_cache, _) = paged(&v, dtype)?;
                let op = BiasedAttention {
                    q: &q,
                    keys: Keys::Paged {
                        k_cache: &k_cache,
                        v_cache: &v_cache,
                        block_tables: &block_tables,
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
        assert!(
            err <= 1e-4 * mag.max(1.0),
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

    fn cuda_paged_bf16_decode_matches_the_reference() -> Result<()> {
        run(&shape((2, 4, 1, 1, 77, 64, 32, 0)), Some(DType::BF16))
    }

    inference_tensor::gpu_cases!(
        cuda_cases,
        [
            cuda_decode_matches_the_reference,
            cuda_log_scaled_queries_match_the_reference,
            cuda_paged_bf16_decode_matches_the_reference,
            cuda_paged_decode_matches_the_reference,
            cuda_paged_prefill_after_a_past_matches_the_reference,
            cuda_prefill_after_a_past_matches_the_reference,
            cuda_prefill_matches_the_reference,
            cuda_short_decode_rows_match_the_reference,
            cuda_windowed_decode_matches_the_reference,
            cuda_windowed_prefill_matches_the_reference
        ]
    );
}
