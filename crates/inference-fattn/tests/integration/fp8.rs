//! fp8 e4m3 K/V against the reference over the values they dequantize to.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use inference_fattn::{FattnOptions, KvScales, causal_mask, flash_attn, supported};

use crate::parity::{Case, N_HEAD, case, cuda, reference};

// Non-unit scales, unequal for K and V, so a swapped or dropped scale shows up.
pub(super) const FP8_SCALES: KvScales = KvScales { k: 0.05, v: 0.2 };
// The kernel rounds each scale to f16 (2^-11) and computes in f16; Q and the result are bf16.
pub(super) const FP8_TOLERANCE: f32 = 1e-2;
// e4m3 codes that are NaN: S.1111.111
const FP8_NAN_CODES: [u8; 2] = [0x7f, 0xff];
const ALL_CODES_HEAD_DIM: usize = 256;

/// `x` stored as `dtype` (fp8 as `x / scale`), and the f32 values the stored tensor stands for.
pub(super) fn store(x: &Tensor, dtype: DType, scale: f32) -> Result<(Tensor, Tensor)> {
    if dtype != DType::F8E4M3 {
        let stored = x.to_dtype(dtype)?;
        return Ok((stored.clone(), stored.to_dtype(DType::F32)?));
    }
    let stored = (x / scale as f64)?.to_dtype(DType::F8E4M3)?;
    let values = (stored.to_dtype(DType::F32)? * scale as f64)?;
    Ok((stored, values))
}

struct Fp8Case {
    head_dim: usize,
    n_head_kv: usize,
    seq_q: usize,
    seq_kv: usize,
    q_dtype: DType,
    softcap: f32,
    sinks: bool,
}

fn fp8_case(head_dim: usize, n_head_kv: usize, seq_q: usize, seq_kv: usize) -> Fp8Case {
    Fp8Case {
        head_dim,
        n_head_kv,
        seq_q,
        seq_kv,
        q_dtype: DType::BF16,
        softcap: 0.,
        sinks: false,
    }
}

fn check(c: Fp8Case) -> Result<()> {
    let Some(dev) = cuda() else { return Ok(()) };
    let batch = 2;
    let d = c.head_dim;
    let rand = |s: usize, h: usize| Tensor::randn(0f32, 1., (batch, s, h, d), &Device::Cpu);
    let (k, k_ref) = store(&rand(c.seq_kv, c.n_head_kv)?, DType::F8E4M3, FP8_SCALES.k)?;
    let (v, v_ref) = store(&rand(c.seq_kv, c.n_head_kv)?, DType::F8E4M3, FP8_SCALES.v)?;
    let (k, v) = (k.to_device(&dev)?, v.to_device(&dev)?);
    let q = rand(c.seq_q, N_HEAD)?
        .to_dtype(c.q_dtype)?
        .to_device(&dev)?;
    let scale = 1. / (d as f32).sqrt();
    let mask = causal_mask(c.seq_q, c.seq_kv, &dev)?;
    let sinks = c
        .sinks
        .then(|| Tensor::randn(0f32, 1., N_HEAD, &dev))
        .transpose()?;
    let opts = FattnOptions {
        scale,
        softcap: c.softcap,
        mask: Some(mask.clone()),
        sinks: sinks.clone(),
        kv_scales: Some(FP8_SCALES),
    };
    assert!(supported(&q, &k, &v, &opts)?);
    let got = flash_attn(&q, &k, &v, &opts)?;
    let reference_case = Case {
        softcap: c.softcap,
        sinks: c.sinks,
        ..case(d, c.n_head_kv, c.seq_q, c.seq_kv)
    };
    let (k_ref, v_ref) = (k_ref.to_device(&dev)?, v_ref.to_device(&dev)?);
    let want = reference(
        &q,
        &k_ref,
        &v_ref,
        Some(&mask),
        sinks.as_ref(),
        &reference_case,
        scale,
    )?;
    let diff = (got.to_dtype(DType::F32)? - &want)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let peak = want.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
    assert!(
        diff <= FP8_TOLERANCE * peak,
        "fp8 hd {d} q {} kv {} {:?}: max diff {diff}, peak {peak}",
        c.seq_q,
        c.seq_kv,
        c.q_dtype
    );
    Ok(())
}

#[test]
fn decode_and_prefill() -> Result<()> {
    for hd in [64, 112, 128, 256] {
        check(fp8_case(hd, 2, 1, 512))?;
        check(fp8_case(hd, 2, 200, 200))?;
    }
    // past 512 Q rows: bf16 would convert first, fp8 still dequantizes in the loads
    check(fp8_case(128, 8, 600, 600))
}

#[test]
fn head_dim_512_loads_k_and_v_in_column_slices() -> Result<()> {
    // at 512 the tile loads split each row into column slices, so the fp8 byte offset of a slice is exercised
    check(fp8_case(512, 2, 64, 512))?;
    check(fp8_case(512, 2, 1, 512))
}

#[test]
fn f32_queries_softcap_and_sinks() -> Result<()> {
    check(Fp8Case {
        q_dtype: DType::F32,
        ..fp8_case(128, 2, 64, 256)
    })?;
    check(Fp8Case {
        softcap: 30.,
        sinks: true,
        ..fp8_case(128, 2, 64, 256)
    })
}

#[test]
fn every_code_decodes() -> Result<()> {
    // with every V row equal, the output is that row whatever the weights; the row holds every non-NaN code
    let Some(dev) = cuda() else { return Ok(()) };
    let codes: Vec<u8> = (0..=u8::MAX)
        .filter(|c| !FP8_NAN_CODES.contains(c))
        .cycle()
        .take(ALL_CODES_HEAD_DIM)
        .collect();
    let seq_kv = 256;
    let row = Tensor::from_vec(codes, ALL_CODES_HEAD_DIM, &Device::Cpu)?;
    let as_fp8 = |t: &Tensor| -> Result<Tensor> {
        let bytes = t.to_vec1::<u8>()?;
        let values: Vec<float8::F8E4M3> =
            bytes.into_iter().map(float8::F8E4M3::from_bits).collect();
        Ok(Tensor::from_vec(values, t.dims(), &Device::Cpu)?)
    };
    let v_row = as_fp8(&row)?;
    let v = v_row
        .reshape((1, 1, 1, ALL_CODES_HEAD_DIM))?
        .broadcast_as((1, seq_kv, 2, ALL_CODES_HEAD_DIM))?
        .contiguous()?;
    let k = Tensor::randn(0f32, 1., (1, seq_kv, 2, ALL_CODES_HEAD_DIM), &Device::Cpu)?
        .to_dtype(DType::F8E4M3)?;
    let q =
        Tensor::randn(0f32, 1., (1, 1, N_HEAD, ALL_CODES_HEAD_DIM), &dev)?.to_dtype(DType::BF16)?;
    let scales = KvScales { k: 1., v: 0.5 };
    let opts = FattnOptions {
        scale: 1. / (ALL_CODES_HEAD_DIM as f32).sqrt(),
        mask: Some(causal_mask(1, seq_kv, &dev)?),
        kv_scales: Some(scales),
        ..Default::default()
    };
    let got = flash_attn(&q, &k.to_device(&dev)?, &v.to_device(&dev)?, &opts)?
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?;
    let want = (v_row.to_dtype(DType::F32)? * scales.v as f64)?.to_vec1::<f32>()?;
    for h in 0..N_HEAD {
        let out = got.narrow(2, h, 1)?.flatten_all()?.to_vec1::<f32>()?;
        for (e, (g, w)) in out.iter().zip(&want).enumerate() {
            // relative to the value, with a floor for the subnormals near zero
            assert!(
                (g - w).abs() <= 1e-2 * w.abs().max(1e-3),
                "head {h} element {e}: got {g}, want {w}"
            );
        }
    }
    Ok(())
}

#[test]
fn rejects_scales_without_fp8_out_of_range_scales_and_fp8_mla() -> Result<()> {
    let Some(dev) = cuda() else { return Ok(()) };
    let t = |h: usize, d: usize| Tensor::zeros((1, 8, h, d), DType::BF16, &dev);
    let opts = FattnOptions {
        scale: 1.,
        kv_scales: Some(FP8_SCALES),
        ..Default::default()
    };
    assert!(flash_attn(&t(8, 64)?, &t(2, 64)?, &t(2, 64)?, &opts).is_err());
    let fp8 = |h: usize, d: usize| {
        Tensor::zeros((1, 8, h, d), DType::F32, &Device::Cpu)?
            .to_dtype(DType::F8E4M3)?
            .to_device(&dev)
    };
    for bad in [0., -1., f32::NAN, 1000.] {
        let bad_scale = FattnOptions {
            kv_scales: Some(KvScales { k: bad, v: 1. }),
            ..opts.clone()
        };
        assert!(flash_attn(&t(8, 64)?, &fp8(2, 64)?, &fp8(2, 64)?, &bad_scale).is_err());
    }
    let k = fp8(1, 576)?;
    let mla = FattnOptions {
        mask: Some(Tensor::zeros((1, 8, 8), DType::F16, &dev)?),
        ..opts
    };
    assert!(flash_attn(&t(8, 576)?, &k, &k.narrow(3, 0, 512)?, &mla).is_err());
    Ok(())
}
