use super::*;
use candle_core::{D, DType, Device, Result as CandleResult, Tensor};
use candle_nn::ops::softmax;

const EPS: f32 = 1e-4;

fn sdpa(softcap: Option<f32>) -> SdpaParams {
    SdpaParams {
        softmax_scale: 1.0,
        softcap,
        n_kv_groups: 1,
        sliding_window: None,
        sinks: None,
    }
}

fn assert_close(lhs: &Tensor, rhs: &Tensor) -> CandleResult<()> {
    assert_within(lhs, rhs, EPS)
}

fn assert_within(lhs: &Tensor, rhs: &Tensor, tol: f32) -> CandleResult<()> {
    let lhs = lhs.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let rhs = rhs.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    for (lhs, rhs) in lhs.iter().zip(rhs.iter()) {
        assert!((lhs - rhs).abs() < tol, "{lhs} vs {rhs}");
    }
    Ok(())
}

fn naive_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    softcap: Option<f32>,
) -> CandleResult<Tensor> {
    let (b, q_len, h, d) = q.dims4()?;
    let kv_len = k.dim(1)?;
    let dv = v.dim(3)?;
    let q = q
        .clone()
        .permute((0, 2, 1, 3))?
        .reshape(&[b * h, q_len, d])?;
    let k = k
        .clone()
        .permute((0, 2, 1, 3))?
        .reshape(&[b * h, kv_len, d])?;
    let v = v
        .clone()
        .permute((0, 2, 1, 3))?
        .reshape(&[b * h, kv_len, dv])?;

    let mut logits = q.matmul(&k.transpose(1, 2)?)?;
    if let Some(softcap) = softcap {
        logits = (logits / softcap as f64)?.tanh()?;
        logits = (logits * softcap as f64)?;
    }
    if let Some(mask) = mask {
        logits = logits.broadcast_add(mask)?;
    }
    let weights = softmax(&logits, D::Minus1)?;
    weights.matmul(&v)?.reshape(&[b, h, q_len, dv])
}

// Distinct values, so a kernel that reads value rows at the query width gets them wrong.
fn ramp(dims: (usize, usize, usize, usize), scale: f32) -> CandleResult<Tensor> {
    let n = dims.0 * dims.1 * dims.2 * dims.3;
    let values = (0..n).map(|i| ((i % 7) as f32 - 3.0) * scale).collect();
    Tensor::from_vec(values, dims, &Device::Cpu)
}

// MLA checkpoints (DeepSeek-V2/V3) have value heads narrower than their query and key heads.
const QK_DIM: usize = 6;
const NARROW_V_DIM: usize = 4;
const HALF_EPS: f32 = 3e-2;

#[test]
fn test_flash_attn_cpu_narrower_value_heads() -> CandleResult<()> {
    let (b, h, kv_len) = (1, 2, 3);
    for q_len in [1, 3] {
        let q = ramp((b, q_len, h, QK_DIM), 0.1)?;
        let k = ramp((b, kv_len, h, QK_DIM), 0.2)?;
        let v = ramp((b, kv_len, h, NARROW_V_DIM), 0.3)?;
        let out = run_flash_attn_cpu::<f32>(&q, &k, &v, None, &sdpa(None))?;
        assert_eq!(out.shape().dims(), &[b, h, q_len, NARROW_V_DIM]);
        assert_close(&out, &naive_attention(&q, &k, &v, None, None)?)?;
    }
    Ok(())
}

#[test]
fn test_flash_attn_cpu_narrower_value_heads_masked_and_softcapped() -> CandleResult<()> {
    let (b, h, kv_len) = (1, 2, 3);
    for q_len in [1, 3] {
        let q = ramp((b, q_len, h, QK_DIM), 0.1)?;
        let k = ramp((b, kv_len, h, QK_DIM), 0.2)?;
        let v = ramp((b, kv_len, h, NARROW_V_DIM), 0.3)?;
        let causal = (0..q_len)
            .flat_map(|i| {
                (0..kv_len).map(move |j| {
                    if j + q_len > i + kv_len {
                        f32::MIN
                    } else {
                        0.0
                    }
                })
            })
            .collect();
        let mask = Tensor::from_vec(causal, (1, q_len, kv_len), &Device::Cpu)?;
        for (mask, softcap) in [
            (Some(&mask), None),
            (None, Some(0.5)),
            (Some(&mask), Some(0.5)),
        ] {
            let out = run_flash_attn_cpu::<f32>(&q, &k, &v, mask, &sdpa(softcap))?;
            assert_close(&out, &naive_attention(&q, &k, &v, mask, softcap)?)?;
        }
    }
    Ok(())
}

#[test]
fn test_flash_attn_cpu_narrower_value_heads_half_precision() -> CandleResult<()> {
    let (b, h, kv_len) = (1, 2, 3);
    for q_len in [1, 3] {
        let q = ramp((b, q_len, h, QK_DIM), 0.1)?;
        let k = ramp((b, kv_len, h, QK_DIM), 0.2)?;
        let v = ramp((b, kv_len, h, NARROW_V_DIM), 0.3)?;
        let expected = naive_attention(&q, &k, &v, None, None)?;
        let [qb, kb, vb] = [&q, &k, &v].map(|t| t.to_dtype(DType::BF16));
        let out = run_flash_attn_cpu::<half::bf16>(&qb?, &kb?, &vb?, None, &sdpa(None))?;
        assert_within(&out, &expected, HALF_EPS)?;
        let [qh, kh, vh] = [&q, &k, &v].map(|t| t.to_dtype(DType::F16));
        let out = run_flash_attn_cpu::<half::f16>(&qh?, &kh?, &vh?, None, &sdpa(None))?;
        assert_within(&out, &expected, HALF_EPS)?;
    }
    Ok(())
}

#[test]
fn test_flash_attn_cpu_single_q() -> CandleResult<()> {
    let (b, h, d, kv_len) = (1, 2, 4, 2);
    let q = Tensor::from_vec(vec![1.0f32; b * h * d], (b, 1, h, d), &Device::Cpu)?;
    let k = Tensor::from_vec(
        vec![1.0f32; b * kv_len * h * d],
        (b, kv_len, h, d),
        &Device::Cpu,
    )?;
    let v = Tensor::from_vec(
        vec![1.0f32; b * kv_len * h * d],
        (b, kv_len, h, d),
        &Device::Cpu,
    )?;

    let out = run_flash_attn_cpu::<f32>(&q, &k, &v, None, &sdpa(None))?;
    assert_eq!(out.shape().dims(), &[b, h, 1, d]);
    assert_close(&out, &naive_attention(&q, &k, &v, None, None)?)
}

#[test]
fn test_flash_attn_cpu_single_q_with_leading_masked_keys() -> CandleResult<()> {
    let (b, h, d, kv_len, hidden) = (1, 2, 4, 5, 3);
    let q = ramp((b, 1, h, d), 0.1)?;
    let k = ramp((b, kv_len, h, d), 0.2)?;
    let v = ramp((b, kv_len, h, d), 0.3)?;
    let mask = (0..kv_len)
        .map(|key| if key < hidden { f32::NEG_INFINITY } else { 0.0 })
        .collect::<Vec<_>>();
    let mask = Tensor::from_vec(mask, (1, kv_len), &Device::Cpu)?;
    let out = run_flash_attn_cpu::<f32>(&q, &k, &v, Some(&mask), &sdpa(None))?;
    assert_close(&out, &naive_attention(&q, &k, &v, Some(&mask), None)?)
}

#[test]
fn test_flash_attn_cpu_full_q_with_a_masked_leading_tile() -> CandleResult<()> {
    // a whole leading KV tile masked, and a bias after it so the mask rows take the general path
    let (b, q_len, h, d, kv_len, hidden) = (1, 3, 2, 4, 300, 160);
    let q = ramp((b, q_len, h, d), 0.1)?;
    let k = ramp((b, kv_len, h, d), 0.2)?;
    let v = ramp((b, kv_len, h, d), 0.3)?;
    let mask = (0..q_len * kv_len)
        .map(|i| match i % kv_len {
            key if key < hidden => f32::NEG_INFINITY,
            key => (key % 5) as f32 * 0.1,
        })
        .collect::<Vec<_>>();
    let mask = Tensor::from_vec(mask, (q_len, kv_len), &Device::Cpu)?;
    let out = run_flash_attn_cpu::<f32>(&q, &k, &v, Some(&mask), &sdpa(None))?;
    assert_close(&out, &naive_attention(&q, &k, &v, Some(&mask), None)?)
}

#[test]
fn test_flash_attn_cpu_single_q_multiple_kv_chunks() -> CandleResult<()> {
    let (b, h, d, kv_len) = (1, 4, 8, 1024);
    let q = Tensor::from_vec(
        (0..b * h * d)
            .map(|x| (x % 17) as f32 / 17.0)
            .collect::<Vec<_>>(),
        (b, 1, h, d),
        &Device::Cpu,
    )?;
    let k = Tensor::from_vec(
        (0..b * kv_len * h * d)
            .map(|x| (x % 19) as f32 / 19.0)
            .collect::<Vec<_>>(),
        (b, kv_len, h, d),
        &Device::Cpu,
    )?;
    let v = Tensor::from_vec(
        (0..b * kv_len * h * d)
            .map(|x| (x % 23) as f32 / 23.0)
            .collect::<Vec<_>>(),
        (b, kv_len, h, d),
        &Device::Cpu,
    )?;

    let out = run_flash_attn_cpu::<f32>(&q, &k, &v, None, &sdpa(None))?;
    assert_close(&out, &naive_attention(&q, &k, &v, None, None)?)
}

#[test]
fn test_flash_attn_cpu_single_q_half_masks() -> CandleResult<()> {
    let (b, h, d, kv_len) = (1, 2, 4, 3);
    let q = Tensor::from_vec(
        (0..b * h * d).map(|x| x as f32 / 17.0).collect::<Vec<_>>(),
        (b, 1, h, d),
        &Device::Cpu,
    )?;
    let k = Tensor::from_vec(
        (0..b * kv_len * h * d)
            .map(|x| x as f32 / 19.0)
            .collect::<Vec<_>>(),
        (b, kv_len, h, d),
        &Device::Cpu,
    )?;
    let v = Tensor::from_vec(
        (0..b * kv_len * h * d)
            .map(|x| x as f32 / 23.0)
            .collect::<Vec<_>>(),
        (b, kv_len, h, d),
        &Device::Cpu,
    )?;
    let mask_f32 = Tensor::from_vec(
        vec![0.0f32, f32::NEG_INFINITY, 0.0],
        (1, kv_len),
        &Device::Cpu,
    )?;
    let expected = naive_attention(&q, &k, &v, Some(&mask_f32), None)?;

    for dtype in [DType::F16, DType::BF16] {
        let mask = mask_f32.to_dtype(dtype)?;
        let out = run_flash_attn_cpu::<f32>(&q, &k, &v, Some(&mask), &sdpa(None))?;
        assert_close(&out, &expected)?;
    }

    Ok(())
}

#[test]
fn test_flash_attn_cpu_full_q() -> CandleResult<()> {
    let (b, q_len, h, d, kv_len) = (1, 2, 2, 4, 2);
    let q = Tensor::from_vec(
        vec![1.0f32; b * q_len * h * d],
        (b, q_len, h, d),
        &Device::Cpu,
    )?;
    let k = Tensor::from_vec(
        vec![1.0f32; b * kv_len * h * d],
        (b, kv_len, h, d),
        &Device::Cpu,
    )?;
    let v = Tensor::from_vec(
        vec![1.0f32; b * kv_len * h * d],
        (b, kv_len, h, d),
        &Device::Cpu,
    )?;

    let out = run_flash_attn_cpu::<f32>(&q, &k, &v, None, &sdpa(None))?;
    assert_eq!(out.shape().dims(), &[b, h, q_len, d]);
    assert_close(&out, &naive_attention(&q, &k, &v, None, None)?)
}

#[test]
fn test_flash_attn_cpu_full_q_respects_finite_mask_values() -> CandleResult<()> {
    let (b, q_len, h, d, kv_len) = (1, 2, 2, 4, 4);
    let q = Tensor::from_vec(
        (0..b * q_len * h * d)
            .map(|x| (x % 11) as f32 / 11.0)
            .collect::<Vec<_>>(),
        (b, q_len, h, d),
        &Device::Cpu,
    )?;
    let k = Tensor::from_vec(
        (0..b * kv_len * h * d)
            .map(|x| (x % 13) as f32 / 13.0)
            .collect::<Vec<_>>(),
        (b, kv_len, h, d),
        &Device::Cpu,
    )?;
    let v = Tensor::from_vec(
        (0..b * kv_len * h * d)
            .map(|x| (x % 17) as f32 / 17.0)
            .collect::<Vec<_>>(),
        (b, kv_len, h, d),
        &Device::Cpu,
    )?;
    let mask = Tensor::from_vec(
        vec![0.0f32, f32::MIN, 0.0, 0.0, 0.0, f32::MIN, 0.0, 0.0],
        (1, q_len, kv_len),
        &Device::Cpu,
    )?;

    let out = run_flash_attn_cpu::<f32>(&q, &k, &v, Some(&mask), &sdpa(None))?;
    assert_close(&out, &naive_attention(&q, &k, &v, Some(&mask), None)?)
}

#[test]
fn test_flash_attn_cpu_full_q_respects_noncontiguous_mask() -> CandleResult<()> {
    let (b, q_len, h, d, kv_len) = (1, 2, 2, 4, 4);
    let q = Tensor::from_vec(
        (0..b * q_len * h * d)
            .map(|x| (x % 11) as f32 / 11.0)
            .collect::<Vec<_>>(),
        (b, q_len, h, d),
        &Device::Cpu,
    )?;
    let k = Tensor::from_vec(
        (0..b * kv_len * h * d)
            .map(|x| (x % 13) as f32 / 13.0)
            .collect::<Vec<_>>(),
        (b, kv_len, h, d),
        &Device::Cpu,
    )?;
    let v = Tensor::from_vec(
        (0..b * kv_len * h * d)
            .map(|x| (x % 17) as f32 / 17.0)
            .collect::<Vec<_>>(),
        (b, kv_len, h, d),
        &Device::Cpu,
    )?;
    let mask = Tensor::from_vec(
        vec![
            0.0f32,
            f32::NEG_INFINITY,
            0.0,
            0.0,
            0.0,
            f32::NEG_INFINITY,
            0.0,
            0.0,
        ],
        (1, q_len, kv_len),
        &Device::Cpu,
    )?;

    let out = run_flash_attn_cpu::<f32>(&q, &k, &v, Some(&mask), &sdpa(None))?;
    assert_close(&out, &naive_attention(&q, &k, &v, Some(&mask), None)?)
}

#[test]
fn test_flash_attn_cpu_full_q_respects_head_specific_mask_classes() -> CandleResult<()> {
    let (b, q_len, h, d, kv_len) = (1, 2, 2, 4, 4);
    let q = Tensor::from_vec(
        (0..b * q_len * h * d)
            .map(|x| (x % 11) as f32 / 11.0)
            .collect::<Vec<_>>(),
        (b, q_len, h, d),
        &Device::Cpu,
    )?;
    let k = Tensor::from_vec(
        (0..b * kv_len * h * d)
            .map(|x| (x % 13) as f32 / 13.0)
            .collect::<Vec<_>>(),
        (b, kv_len, h, d),
        &Device::Cpu,
    )?;
    let v = Tensor::from_vec(
        (0..b * kv_len * h * d)
            .map(|x| (x % 17) as f32 / 17.0)
            .collect::<Vec<_>>(),
        (b, kv_len, h, d),
        &Device::Cpu,
    )?;
    let mask = Tensor::from_vec(
        vec![
            0.0,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            0.0,
            0.0,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            0.0,
            f32::MIN,
            0.0,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            0.0,
            f32::NEG_INFINITY,
            0.0,
        ],
        (b, h, q_len, kv_len),
        &Device::Cpu,
    )?;
    let expected_mask = mask.reshape((b * h, q_len, kv_len))?;

    let out = run_flash_attn_cpu::<f32>(&q, &k, &v, Some(&mask), &sdpa(None))?;
    assert_close(
        &out,
        &naive_attention(&q, &k, &v, Some(&expected_mask), None)?,
    )
}

#[test]
fn test_flash_attn_cpu_full_q_with_more_queries_than_keys() -> CandleResult<()> {
    let (b, q_len, h, d, kv_len) = (1, 4, 2, 4, 2);
    let q = Tensor::from_vec(
        (0..b * q_len * h * d)
            .map(|x| (x % 11) as f32 / 11.0)
            .collect::<Vec<_>>(),
        (b, q_len, h, d),
        &Device::Cpu,
    )?;
    let k = Tensor::from_vec(
        (0..b * kv_len * h * d)
            .map(|x| (x % 13) as f32 / 13.0)
            .collect::<Vec<_>>(),
        (b, kv_len, h, d),
        &Device::Cpu,
    )?;
    let v = Tensor::from_vec(
        (0..b * kv_len * h * d)
            .map(|x| (x % 17) as f32 / 17.0)
            .collect::<Vec<_>>(),
        (b, kv_len, h, d),
        &Device::Cpu,
    )?;
    let mask = Tensor::zeros((1, q_len, kv_len), DType::F32, &Device::Cpu)?;

    let out = run_flash_attn_cpu::<f32>(&q, &k, &v, Some(&mask), &sdpa(None))?;
    assert_close(&out, &naive_attention(&q, &k, &v, Some(&mask), None)?)
}

#[test]
fn test_flash_attn_cpu_single_q_softcap() -> CandleResult<()> {
    let (b, h, d, kv_len) = (1, 2, 4, 2);
    let q = Tensor::from_vec(vec![1.0f32; b * h * d], (b, 1, h, d), &Device::Cpu)?;
    let k = Tensor::from_vec(
        vec![1.0f32; b * kv_len * h * d],
        (b, kv_len, h, d),
        &Device::Cpu,
    )?;
    let v = Tensor::from_vec(
        vec![1.0f32; b * kv_len * h * d],
        (b, kv_len, h, d),
        &Device::Cpu,
    )?;

    let out = run_flash_attn_cpu::<f32>(&q, &k, &v, None, &sdpa(Some(0.5)))?;
    assert_eq!(out.shape().dims(), &[b, h, 1, d]);
    assert_close(&out, &naive_attention(&q, &k, &v, None, Some(0.5))?)
}

#[test]
fn test_flash_attn_cpu_full_q_softcap() -> CandleResult<()> {
    let (b, q_len, h, d, kv_len) = (1, 2, 2, 4, 2);
    let q = Tensor::from_vec(
        vec![1.0f32; b * q_len * h * d],
        (b, q_len, h, d),
        &Device::Cpu,
    )?;
    let k = Tensor::from_vec(
        vec![1.0f32; b * kv_len * h * d],
        (b, kv_len, h, d),
        &Device::Cpu,
    )?;
    let v = Tensor::from_vec(
        vec![1.0f32; b * kv_len * h * d],
        (b, kv_len, h, d),
        &Device::Cpu,
    )?;

    let out = run_flash_attn_cpu::<f32>(&q, &k, &v, None, &sdpa(Some(0.5)))?;
    assert_eq!(out.shape().dims(), &[b, h, q_len, d]);
    assert_close(&out, &naive_attention(&q, &k, &v, None, Some(0.5))?)
}
