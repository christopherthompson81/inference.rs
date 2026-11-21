//! Causal attention with a relative bias looked up by distance and a sink logit per query (Inkling 2), fused: the
//! bias and the mask are never materialized (see `kernels/cuda/biased_attention.cu`).

use inference_tensor::cuda_backend::CudaStorage;
use inference_tensor::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
use inference_tensor::{DType, Device, Layout, Result, Shape, Storage, Tensor};

/// Head sizes the kernel is built for
pub const HEAD_DIMS: [usize; 3] = [32, 64, 128];
// queries up to this many take the kernel that splits the keys across warps (decode)
const DECODE_ROWS: i32 = 4;

/// The attention inputs, all f32 on one CUDA device.
pub struct BiasedAttention<'a> {
    /// `(b, heads, t, head_dim)`, query `i` at position `past + i`
    pub q: &'a Tensor,
    /// `(b, kv_heads, past + t, head_dim)`, rows contiguous (a cache's narrowed view)
    pub k: &'a Tensor,
    pub v: &'a Tensor,
    /// `(b, heads, t, extent + 1)`: each query's bias at distances `0..extent` (none past them; the last column unread)
    pub table: &'a Tensor,
    /// `(b, heads, t)`: each query's sink logit, a weight with a zero value
    pub sinks: &'a Tensor,
    /// keys this many or more back are hidden; 0 for none
    pub window: usize,
    pub scale: f32,
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

/// `(b, t, heads * head_dim)` f32, the layout the output gate and projection take; `None` where the kernel does not
/// take the head size or the K/V layouts.
pub fn biased_attention(op: &BiasedAttention<'_>) -> Result<Option<Tensor>> {
    let (b, heads, t, d) = op.q.dims4()?;
    let (kb, kv_heads, keys, kd) = op.k.dims4()?;
    let extent = op.table.dim(3)? - 1;
    if (kb, kd) != (b, d)
        || op.v.dims4()? != (b, kv_heads, keys, d)
        || op.table.dims4()? != (b, heads, t, extent + 1)
        || op.sinks.dims3()? != (b, heads, t)
        || keys < t
        || kv_heads == 0
        || !heads.is_multiple_of(kv_heads)
    {
        inference_tensor::bail!(
            "biased attention shapes q {:?} k {:?} v {:?} table {:?} sinks {:?}",
            op.q.shape(),
            op.k.shape(),
            op.v.shape(),
            op.table.shape(),
            op.sinks.shape()
        );
    }
    let all_f32 = [op.q, op.k, op.v, op.table, op.sinks]
        .iter()
        .all(|x| x.dtype() == DType::F32);
    let Device::Cuda(dev) = op.q.device() else {
        return Ok(None);
    };
    if !HEAD_DIMS.contains(&d) || !all_f32 {
        return Ok(None);
    }
    let (q, table, sinks) = (
        op.q.contiguous()?,
        op.table.contiguous()?,
        op.sinks.contiguous()?,
    );
    let (kg, kl) = op.k.storage_and_layout();
    let (vg, vl) = op.v.storage_and_layout();
    let (Some((k_batch, k_head)), Some((v_batch, v_head))) = (kv_strides(kl, d), kv_strides(vl, d))
    else {
        return Ok(None);
    };
    let stream = dev.cuda_stream();
    // SAFETY: the kernel writes every output element
    let mut out = unsafe { dev.alloc::<f32>(b * t * heads * d)? };
    {
        let (qg, ql) = q.storage_and_layout();
        let (tg, tl) = table.storage_and_layout();
        let (sg, sl) = sinks.storage_and_layout();
        let (
            Storage::Cuda(qs),
            Storage::Cuda(ks),
            Storage::Cuda(vs),
            Storage::Cuda(ts),
            Storage::Cuda(ss),
        ) = (&*qg, &*kg, &*vg, &*tg, &*sg)
        else {
            inference_tensor::bail!("biased attention needs CUDA storage");
        };
        fn view<'s>(
            s: &'s CudaStorage,
            l: &Layout,
        ) -> Result<inference_tensor::cuda_backend::cudarc::driver::CudaView<'s, f32>> {
            Ok(s.as_cuda_slice::<f32>()?.slice(l.start_offset()..))
        }
        let (qv, kv, vv, tv, sv) = (
            view(qs, ql)?,
            view(ks, kl)?,
            view(vs, vl)?,
            view(ts, tl)?,
            view(ss, sl)?,
        );
        let (q_ptr, _q) = qv.device_ptr(&stream);
        let (k_ptr, _k) = kv.device_ptr(&stream);
        let (v_ptr, _v) = vv.device_ptr(&stream);
        let (t_ptr, _t) = tv.device_ptr(&stream);
        let (s_ptr, _s) = sv.device_ptr(&stream);
        let (o_ptr, _o) = out.device_ptr_mut(&stream);
        // SAFETY: the pointers cover the shapes and strides checked above, all on this stream
        let code = unsafe {
            super::ffi::biased_attention_f32(
                q_ptr as *const f32,
                k_ptr as *const f32,
                v_ptr as *const f32,
                t_ptr as *const f32,
                s_ptr as *const f32,
                o_ptr as *mut f32,
                i32_of(b)?,
                i32_of(heads)?,
                i32_of(kv_heads)?,
                i32_of(t)?,
                i32_of(keys)?,
                i32_of(keys - t)?,
                i32_of(d)?,
                i32_of(extent)?,
                i32_of(op.window)?,
                op.scale,
                i64_of(k_batch)?,
                i64_of(k_head)?,
                i64_of(v_batch)?,
                i64_of(v_head)?,
                DECODE_ROWS,
                stream.cu_stream() as i64,
            )
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

#[cfg(test)]
mod tests {
    use super::*;

    // the materialized form: scores plus the bias gathered by distance, masked, a sink column, softmax, values
    fn reference(op: &BiasedAttention<'_>) -> Result<Tensor> {
        let (b, h, t, d) = op.q.dims4()?;
        let (_, hkv, s, _) = op.k.dims4()?;
        let extent = op.table.dim(3)? - 1;
        let past = s - t;
        let mut bias = vec![0f32; b * h * t * s];
        let rows = op
            .table
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for bh in 0..b * h {
            for i in 0..t {
                let pos = past + i;
                for j in 0..s {
                    let visible = j <= pos && (op.window == 0 || pos - j < op.window);
                    let at = (bh * t + i) * s + j;
                    bias[at] = if !visible {
                        f32::NEG_INFINITY
                    } else if pos - j < extent {
                        rows[(bh * t + i) * (extent + 1) + pos - j]
                    } else {
                        0.0
                    };
                }
            }
        }
        let device = op.q.device();
        let bias = Tensor::from_vec(bias, (b, h, t, s), device)?;
        let group = h / hkv;
        let k = op.k.repeat_interleave_for_test(group)?;
        let v = op.v.repeat_interleave_for_test(group)?;
        let scores = ((op.q.matmul(&k.t()?)? * f64::from(op.scale))? + bias)?;
        let sinks = op.sinks.unsqueeze(3)?;
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

    // (batch, heads, kv heads, t, past, head_dim, extent, window), K/V narrowed from a longer cache
    #[allow(clippy::cast_precision_loss)]
    fn case(shape: (usize, usize, usize, usize, usize, usize, usize, usize)) -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        let (b, h, hkv, t, past, d, extent, window) = shape;
        let s = past + t;
        let capacity = s + 16;
        let rand = |dims: &[usize]| Tensor::randn(0f32, 1.0, dims, &device);
        let q = rand(&[b, h, t, d])?;
        let k = rand(&[b, hkv, capacity, d])?.narrow(2, 0, s)?;
        let v = rand(&[b, hkv, capacity, d])?.narrow(2, 0, s)?;
        let table = rand(&[b, h, t, extent + 1])?;
        let sinks = rand(&[b, h, t])?;
        let op = BiasedAttention {
            q: &q,
            k: &k,
            v: &v,
            table: &table,
            sinks: &sinks,
            window,
            scale: 1.0 / (d as f32).sqrt(),
        };
        let got = biased_attention(&op)?.expect("the kernel takes these shapes");
        let want = reference(&op)?;
        let err = (got - &want)?.abs()?.max_all()?.to_scalar::<f32>()?;
        let scale = want.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(
            err <= 1e-4 * scale.max(1.0),
            "{shape:?}: max error {err} of {scale}"
        );
        Ok(())
    }

    fn cuda_prefill_matches_the_reference() -> Result<()> {
        case((2, 8, 2, 37, 0, 128, 16, 0))
    }

    fn cuda_prefill_after_a_past_matches_the_reference() -> Result<()> {
        case((1, 4, 4, 21, 50, 64, 40, 0))
    }

    fn cuda_windowed_prefill_matches_the_reference() -> Result<()> {
        case((1, 8, 2, 70, 30, 128, 64, 48))
    }

    fn cuda_decode_matches_the_reference() -> Result<()> {
        case((2, 8, 2, 1, 700, 128, 256, 0))
    }

    fn cuda_windowed_decode_matches_the_reference() -> Result<()> {
        case((1, 4, 1, 1, 300, 32, 64, 100))
    }

    fn cuda_short_decode_rows_match_the_reference() -> Result<()> {
        case((1, 4, 2, 3, 9, 64, 8, 0))
    }

    inference_tensor::gpu_cases!(
        cuda_cases,
        [
            cuda_decode_matches_the_reference,
            cuda_prefill_after_a_past_matches_the_reference,
            cuda_prefill_matches_the_reference,
            cuda_short_decode_rows_match_the_reference,
            cuda_windowed_decode_matches_the_reference,
            cuda_windowed_prefill_matches_the_reference
        ]
    );
}
