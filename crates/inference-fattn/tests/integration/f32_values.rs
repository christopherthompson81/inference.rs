//! The mma kernel with f32 value sums (`FattnOptions::f32_values`) against the f32 reference, and against the f16 sums
//! on rows spread over many keys.

use anyhow::Result;
use inference_fattn::{FattnOptions, causal_mask, flash_attn, supported};
use inference_tensor::{DType, Tensor};

use super::parity::{Case, N_HEAD, QLayout, case, cuda, reference};

const HEAD_DIM: usize = 128;
// Max |fattn - reference| / max |reference| with f32 sums: the f16 inputs and the rounded result remain
const TOLERANCE: [(DType, f32); 2] = [(DType::F16, 2e-3), (DType::BF16, 8e-3)];
// keys an even row spreads over, and the values' mean: a 3.3M sum, past f16's 65504 however the kernel splits it,
// while one KV tile's (summed in f16) stays inside it
const EVEN_KEYS: usize = 8192;
const LARGE_VALUE: f64 = 400.0;

inference_tensor::gpu_cases!(
    cuda_cases,
    [
        matches_the_reference,
        even_rows_over_large_values_stay_exact,
        diffuse_rows_beat_the_f16_sums,
        unbuilt_shapes_are_not_supported
    ]
);

fn f32_opts(scale: f32) -> FattnOptions {
    FattnOptions {
        scale,
        f32_values: true,
        ..Default::default()
    }
}

fn worst_over_peak(got: &Tensor, want: &Tensor) -> Result<f32> {
    let diff = (got.to_dtype(DType::F32)? - want)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let peak = want.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
    Ok(diff / peak)
}

fn matches_the_reference() -> Result<()> {
    let Some(dev) = cuda() else { return Ok(()) };
    let scale = 1. / (HEAD_DIM as f32).sqrt();
    // short and long Q (16 to 64 columns), batch 1 for the stream-k fixups, causal through a mask and implicitly
    for (batch, seq_q, seq_kv, causal) in [
        (2, 5, 300, false),
        (2, 100, 100, true),
        (1, 300, 300, true),
        (1, 1000, 1000, false),
        (2, 640, 1024, true),
    ] {
        let c = Case {
            batch,
            causal,
            q_layout: QLayout::Contiguous,
            ..case(HEAD_DIM, N_HEAD, seq_q, seq_kv)
        };
        for (dtype, tolerance) in TOLERANCE {
            let rand = |s: usize| Tensor::randn(0f32, 1., (batch, s, N_HEAD, HEAD_DIM), &dev);
            let (q, k, v) = (
                rand(seq_q)?.to_dtype(dtype)?,
                rand(seq_kv)?.to_dtype(dtype)?,
                rand(seq_kv)?.to_dtype(dtype)?,
            );
            let mask = if causal {
                Some(causal_mask(seq_q, seq_kv, &dev)?)
            } else {
                None
            };
            let opts = FattnOptions {
                mask: mask.clone(),
                ..f32_opts(scale)
            };
            assert!(supported(&q, &k, &v, &opts)?, "q {seq_q} kv {seq_kv}");
            let got = flash_attn(&q, &k, &v, &opts)?;
            let want = reference(&q, &k, &v, mask.as_ref(), None, &c, scale)?;
            let err = worst_over_peak(&got, &want)?;
            assert!(err <= tolerance, "q {seq_q} kv {seq_kv} {dtype:?}: {err}");
        }
    }
    Ok(())
}

fn even_rows_over_large_values_stay_exact() -> Result<()> {
    let Some(dev) = cuda() else { return Ok(()) };
    let shape = (1, EVEN_KEYS, 2, HEAD_DIM);
    let q = Tensor::zeros(shape, DType::F16, &dev)?;
    let k = Tensor::randn(0f32, 1., shape, &dev)?.to_dtype(DType::F16)?;
    let v = (Tensor::randn(0f32, 4., shape, &dev)? + LARGE_VALUE)?.to_dtype(DType::F16)?;
    let mean = v.to_dtype(DType::F32)?.mean_keepdim(1)?;
    let scale = 1. / (HEAD_DIM as f32).sqrt();
    let got = flash_attn(&q, &k, &v, &f32_opts(scale))?.to_dtype(DType::F32)?;
    assert!(
        got.flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|x| x.is_finite()),
        "non-finite with f32 sums"
    );
    let err = worst_over_peak(&got, &mean.broadcast_as(got.shape())?.contiguous()?)?;
    // the result's own f16 rounding at 1000 is 2^-11 relative
    assert!(err <= 1e-3, "off the mean by {err}");
    // the f16 sums overflow here: what the f32 ones are for
    let f16 = flash_attn(
        &q,
        &k,
        &v,
        &FattnOptions {
            scale,
            ..Default::default()
        },
    )?;
    assert!(
        f16.to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .any(|x| !x.is_finite()),
        "the f16 sums no longer overflow on this case"
    );
    Ok(())
}

fn diffuse_rows_beat_the_f16_sums() -> Result<()> {
    let Some(dev) = cuda() else { return Ok(()) };
    let (seq, heads) = (EVEN_KEYS, 2);
    // small queries spread each row over most keys; values offset from zero so the sums grow
    let shape = (1, seq, heads, HEAD_DIM);
    let q = (Tensor::randn(0f32, 0.1, shape, &dev)?).to_dtype(DType::BF16)?;
    let k = Tensor::randn(0f32, 1., shape, &dev)?.to_dtype(DType::BF16)?;
    let v = (Tensor::randn(0f32, 1., shape, &dev)? + 4.0)?.to_dtype(DType::BF16)?;
    let scale = 1. / (HEAD_DIM as f32).sqrt();
    let want = plain_reference(&q, &k, &v, scale)?;
    let f32_err = worst_over_peak(&flash_attn(&q, &k, &v, &f32_opts(scale))?, &want)?;
    let f16_err = worst_over_peak(
        &flash_attn(
            &q,
            &k,
            &v,
            &FattnOptions {
                scale,
                ..Default::default()
            },
        )?,
        &want,
    )?;
    assert!(
        f32_err * 4.0 < f16_err,
        "f32 sums {f32_err} against f16 sums {f16_err}"
    );
    Ok(())
}

// softmax(scale q k^T) v in f32 without masks or GQA, for any head count (the parity reference assumes N_HEAD)
fn plain_reference(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor> {
    let heads_first = |t: &Tensor| t.to_dtype(DType::F32)?.transpose(1, 2)?.contiguous();
    let att = (heads_first(q)?.matmul(&heads_first(k)?.t()?)? * f64::from(scale))?;
    let probs = inference_tensor::nn::ops::softmax(&att, inference_tensor::D::Minus1)?;
    Ok(probs.matmul(&heads_first(v)?)?.transpose(1, 2)?)
}

fn unbuilt_shapes_are_not_supported() -> Result<()> {
    let Some(dev) = cuda() else { return Ok(()) };
    let opts = f32_opts(0.1);
    let qkv = |d: usize, heads_kv: usize| -> Result<(Tensor, Tensor, Tensor)> {
        let t = |h: usize| Tensor::randn(0f32, 1., (1, 64, h, d), &dev)?.to_dtype(DType::F16);
        Ok((t(N_HEAD)?, t(heads_kv)?, t(heads_kv)?))
    };
    let (q, k, v) = qkv(64, N_HEAD)?;
    assert!(!supported(&q, &k, &v, &opts)?, "head dim 64");
    let (q, k, v) = qkv(HEAD_DIM, 2)?;
    assert!(!supported(&q, &k, &v, &opts)?, "GQA");
    let (q, k, v) = qkv(HEAD_DIM, N_HEAD)?;
    let softcap = FattnOptions {
        softcap: 30.0,
        ..opts.clone()
    };
    assert!(!supported(&q, &k, &v, &softcap)?, "softcap");
    assert!(supported(&q, &k, &v, &opts)?, "head 128 without GQA");
    Ok(())
}
