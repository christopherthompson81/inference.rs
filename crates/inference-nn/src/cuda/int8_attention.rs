//! Bidirectional attention with an int8 Q.K^T and an fp16 P.V (see `kernels/cuda/int8_attention.cu`), for long
//! sequences where attention dominates: video diffusion transformers' joint self-attention.

use inference_tensor::cuda_backend::CudaStorage;
use inference_tensor::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
use inference_tensor::{DType, Device, Result, Shape, Storage, Tensor};

/// Head sizes the kernel is built for
pub const HEAD_DIMS: [usize; 2] = [64, 128];

fn status(code: i32, what: &str) -> Result<()> {
    if code != 0 {
        inference_tensor::bail!("{what} failed with CUDA status {code}");
    }
    Ok(())
}

fn i32_of(v: usize) -> Result<i32> {
    i32::try_from(v).map_err(inference_tensor::Error::wrap)
}

/// Attention over `q` `[b, heads, n_q, d]` and `k`, `v` `[b, heads, n_kv, d]` (q and k bf16, d one of [`HEAD_DIMS`]),
/// no mask; returns `[b, n_q, heads * d]` bf16, the layout the output projection takes.
pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor, softmax_scale: f32) -> Result<Tensor> {
    let (b, heads, n_q, d) = q.dims4()?;
    let (kb, kh, n_kv, kd) = k.dims4()?;
    if (kb, kh, kd) != (b, heads, d) || v.dims4()? != (b, heads, n_kv, d) {
        inference_tensor::bail!(
            "int8 attention shapes q {:?} k {:?} v {:?}",
            q.shape(),
            k.shape(),
            v.shape()
        );
    }
    if !HEAD_DIMS.contains(&d) || q.dtype() != DType::BF16 || k.dtype() != DType::BF16 {
        inference_tensor::bail!("int8 attention takes bf16 q and k with heads of {HEAD_DIMS:?}");
    }
    let Device::Cuda(dev) = q.device() else {
        inference_tensor::bail!("int8 attention needs CUDA");
    };
    let (q, k) = (q.contiguous()?, k.contiguous()?);
    let v = v.to_dtype(DType::F16)?.contiguous()?;
    let stream = dev.cuda_stream();
    let s = stream.cu_stream() as i64;
    let bh = b * heads;
    let (bh_i, d_i, nq_i, nkv_i) = (i32_of(bh)?, i32_of(d)?, i32_of(n_q)?, i32_of(n_kv)?);
    // SAFETY: each buffer is written in full before it is read
    let (mut q8, mut qs, mut k8, mut ks, mut mean, mut out) = unsafe {
        (
            dev.alloc::<u8>(bh * n_q * d)?,
            dev.alloc::<f32>(bh * n_q)?,
            dev.alloc::<u8>(bh * n_kv * d)?,
            dev.alloc::<f32>(bh * n_kv)?,
            dev.alloc::<f32>(bh * d)?,
            dev.alloc::<half::bf16>(b * n_q * heads * d)?,
        )
    };
    {
        let (qg, ql) = q.storage_and_layout();
        let (kg, kl) = k.storage_and_layout();
        let (vg, vl) = v.storage_and_layout();
        let (Storage::Cuda(qst), Storage::Cuda(kst), Storage::Cuda(vst)) = (&*qg, &*kg, &*vg)
        else {
            inference_tensor::bail!("int8 attention needs CUDA storage");
        };
        let qv = qst
            .as_cuda_slice::<half::bf16>()?
            .slice(ql.start_offset()..);
        let kv = kst
            .as_cuda_slice::<half::bf16>()?
            .slice(kl.start_offset()..);
        let vv = vst.as_cuda_slice::<half::f16>()?.slice(vl.start_offset()..);
        let (q_ptr, _q) = qv.device_ptr(&stream);
        let (k_ptr, _k) = kv.device_ptr(&stream);
        let (v_ptr, _v) = vv.device_ptr(&stream);
        let (q8_ptr, _q8) = q8.device_ptr_mut(&stream);
        let (qs_ptr, _qs) = qs.device_ptr_mut(&stream);
        let (k8_ptr, _k8) = k8.device_ptr_mut(&stream);
        let (ks_ptr, _ks) = ks.device_ptr_mut(&stream);
        let (mean_ptr, _mean) = mean.device_ptr_mut(&stream);
        let (out_ptr, _out) = out.device_ptr_mut(&stream);
        let p = |ptr: u64| ptr as *const std::ffi::c_void;
        let m = |ptr: u64| ptr as *mut std::ffi::c_void;
        // SAFETY: the pointers cover the shapes checked above, all on this stream
        unsafe {
            use super::ffi::{sa_attention, sa_mean_bf16, sa_quantize_bf16};
            status(
                sa_mean_bf16(p(k_ptr), m(mean_ptr), bh_i, nkv_i, d_i, s),
                "K mean",
            )?;
            let rows = (bh * n_q) as i64;
            status(
                sa_quantize_bf16(
                    p(q_ptr),
                    std::ptr::null(),
                    m(q8_ptr),
                    m(qs_ptr),
                    rows,
                    nq_i,
                    d_i,
                    s,
                ),
                "Q quantize",
            )?;
            let rows = (bh * n_kv) as i64;
            status(
                sa_quantize_bf16(
                    p(k_ptr),
                    p(mean_ptr),
                    m(k8_ptr),
                    m(ks_ptr),
                    rows,
                    nkv_i,
                    d_i,
                    s,
                ),
                "K quantize",
            )?;
            let scale_log2 = softmax_scale * std::f32::consts::LOG2_E;
            status(
                sa_attention(
                    p(q8_ptr),
                    p(qs_ptr),
                    p(k8_ptr),
                    p(ks_ptr),
                    p(v_ptr),
                    m(out_ptr),
                    i32_of(b)?,
                    i32_of(heads)?,
                    nq_i,
                    nkv_i,
                    d_i,
                    scale_log2,
                    s,
                ),
                "int8 attention",
            )?;
        }
    }
    let storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
    Ok(Tensor::from((
        Storage::Cuda(storage),
        Shape::from_dims(&[b, n_q, heads * d]),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use inference_tensor::D;

    // softmax(q k^T * scale) v in f32, as [b, n_q, heads * d]
    fn reference(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor> {
        let (b, h, n, d) = q.dims4()?;
        let (q, k, v) = (
            q.to_dtype(DType::F32)?,
            k.to_dtype(DType::F32)?,
            v.to_dtype(DType::F32)?,
        );
        let w = inference_tensor::nn::ops::softmax_last_dim(&(q.matmul(&k.t()?)? * scale)?)?;
        w.matmul(&v)?.transpose(1, 2)?.reshape((b, n, h * d))
    }

    fn bf16_flash(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor> {
        let (b, h, n, d) = q.dims4()?;
        let params = crate::attention::SdpaParams {
            n_kv_groups: 1,
            softcap: None,
            softmax_scale: scale,
            sliding_window: None,
            sinks: None,
            chunk: None,
        };
        crate::attention::Sdpa
            .run_attention(
                q,
                k,
                &v.contiguous()?,
                &crate::attention::AttentionMask::None,
                Some(&crate::attention::FlashParams::empty(false)),
                &params,
            )?
            .transpose(1, 2)?
            .reshape((b, n, h * d))?
            .to_dtype(DType::F32)
    }

    // a video diffusion step's joint self-attention: 45K tokens, 40 heads of 128
    #[test]
    #[ignore = "benchmark"]
    #[allow(clippy::cast_precision_loss)]
    fn long_sequence_throughput() -> Result<()> {
        const TOKENS: usize = 45_240;
        const HEADS: usize = 40;
        const HEAD_DIM: usize = 128;
        const REPEATS: usize = 3;
        let dev = Device::new_cuda(0)?;
        let make = || -> Result<Tensor> {
            Tensor::randn(0f32, 1., (1, HEADS, TOKENS, HEAD_DIM), &dev)?.to_dtype(DType::BF16)
        };
        let (q, k, v) = (make()?, make()?, make()?);
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let flop = 4.0 * (TOKENS * TOKENS * HEAD_DIM * HEADS) as f64;
        let time = |f: &dyn Fn() -> Result<Tensor>| -> Result<f64> {
            f()?;
            dev.synchronize()?;
            let start = std::time::Instant::now();
            for _ in 0..REPEATS {
                f()?;
            }
            dev.synchronize()?;
            Ok(start.elapsed().as_secs_f64() / REPEATS as f64)
        };
        let s = time(&|| bf16_flash(&q, &k, &v, scale))?;
        println!(
            "bf16 flash   {:7.1} ms  {:6.1} TFLOP/s",
            s * 1e3,
            flop / s / 1e12
        );
        let s = time(&|| attention(&q, &k, &v, scale))?;
        println!(
            "int8 qk      {:7.1} ms  {:6.1} TFLOP/s",
            s * 1e3,
            flop / s / 1e12
        );
        Ok(())
    }

    // tails on both sequences, both head sizes, and keys with a large shared offset the mean removes
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn matches_f32_attention() -> Result<()> {
        skip_without_cuda!();
        const TOLERANCE: f32 = 0.03;
        let dev = Device::new_cuda(0)?;
        for (n_q, n_kv, d) in [(37, 53, 64), (300, 1000, 128), (129, 64, 128)] {
            let heads = 3;
            let q = (Tensor::randn(0f32, 1., (1, heads, n_q, d), &dev)? * 2.0)?
                .to_dtype(DType::BF16)?;
            let offset = Tensor::randn(0f32, 4., (1, heads, 1, d), &dev)?;
            let k = (Tensor::randn(0f32, 1., (1, heads, n_kv, d), &dev)? * 2.0)?
                .broadcast_add(&offset)?
                .to_dtype(DType::BF16)?;
            let v = Tensor::randn(0f32, 1., (1, heads, n_kv, d), &dev)?.to_dtype(DType::BF16)?;
            let scale = 1.0 / (d as f32).sqrt();
            let want = reference(&q, &k, &v, f64::from(scale))?;
            let got = attention(&q, &k, &v, scale)?.to_dtype(DType::F32)?;
            let err = (&got - &want)?
                .sqr()?
                .mean_all()?
                .sqrt()?
                .to_scalar::<f32>()?;
            let rms = want.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
            let max = (&got - &want)?
                .abs()?
                .max_keepdim(D::Minus1)?
                .max_all()?
                .to_scalar::<f32>()?;
            let flash = bf16_flash(&q, &k, &v, scale)?;
            let flash_err = (&flash - &want)?
                .sqr()?
                .mean_all()?
                .sqrt()?
                .to_scalar::<f32>()?;
            println!(
                "{n_q}x{n_kv} d{d}: rel rms {:.2e} (bf16 flash {:.2e}), max {max:.3e}",
                err / rms,
                flash_err / rms
            );
            assert!(
                err / rms < TOLERANCE,
                "{n_q}x{n_kv} d{d}: rel rms {}",
                err / rms
            );
        }
        Ok(())
    }
}
