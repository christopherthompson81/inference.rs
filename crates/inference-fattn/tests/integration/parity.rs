//! fattn against an F32 reference across its kernels (mma, vec, tile), head dims, masks, softcap and sinks.

use anyhow::Result;
use candle_core::{D, DType, Device, Tensor};
use inference_fattn::{FattnOptions, causal_mask, flash_attn};

const BATCH: usize = 2;
pub(super) const N_HEAD: usize = 8;
// MLA's absorbed head dim, where V is the leading 512 dims of K
const MLA_HEAD_DIM: usize = 576;
// Max |fattn - reference| / max |reference| per input dtype. The mma path computes in f16 either way, and a bf16
// result is rounded to bf16 on the way out.
pub(super) const TOLERANCE: [(DType, f32); 2] = [(DType::F16, 4e-3), (DType::BF16, 1e-2)];

#[derive(Clone, Copy, PartialEq)]
pub(super) enum QLayout {
    Contiguous,
    // a transposed (non-contiguous) view
    Transposed,
    // an odd offset and row stride, which the kernels' vector loads cannot take as is
    Misaligned,
    // contiguous, but starting at an odd element
    OffsetContiguous,
}

#[derive(Clone, Copy)]
pub(super) struct Case {
    pub(super) batch: usize,
    pub(super) head_dim: usize,
    pub(super) head_dim_v: usize,
    pub(super) n_head_kv: usize,
    pub(super) seq_q: usize,
    pub(super) seq_kv: usize,
    pub(super) causal: bool,
    pub(super) softcap: f32,
    pub(super) sinks: bool,
    pub(super) q_layout: QLayout,
    pub(super) f32_q: bool,
}

pub(super) fn case(head_dim: usize, n_head_kv: usize, seq_q: usize, seq_kv: usize) -> Case {
    Case {
        batch: BATCH,
        head_dim,
        head_dim_v: head_dim,
        n_head_kv,
        seq_q,
        seq_kv,
        causal: true,
        softcap: 0.,
        sinks: false,
        q_layout: QLayout::Contiguous,
        f32_q: false,
    }
}

pub(super) fn cuda() -> Option<Device> {
    let dev = Device::new_cuda(0).ok();
    if dev.is_none() {
        eprintln!("SKIP: no CUDA device");
    }
    dev
}

// Repeat K/V heads for GQA, then softmax(softcap(scale * q k^T) + mask) v, with sinks as extra per-head logits.
pub(super) fn reference(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    sinks: Option<&Tensor>,
    case: &Case,
    scale: f32,
) -> Result<Tensor> {
    let groups = N_HEAD / case.n_head_kv;
    let expand = |t: &Tensor| -> Result<Tensor> {
        let (b, s, h, d) = t.dims4()?;
        Ok(t.to_dtype(DType::F32)?
            .unsqueeze(3)?
            .expand((b, s, h, groups, d))?
            .reshape((b, s, h * groups, d))?)
    };
    let q = q.to_dtype(DType::F32)?.transpose(1, 2)?.contiguous()?;
    let k = expand(k)?.transpose(1, 2)?.contiguous()?;
    let v = expand(v)?.transpose(1, 2)?.contiguous()?;
    let mut att = (q.matmul(&k.t()?)? * scale as f64)?;
    if case.softcap > 0. {
        att = ((att / case.softcap as f64)?.tanh()? * case.softcap as f64)?;
    }
    if let Some(mask) = mask {
        att = att.broadcast_add(&mask.to_dtype(DType::F32)?.unsqueeze(1)?)?;
    }
    let att = match sinks {
        Some(sinks) => {
            let (b, h, sq, _) = att.dims4()?;
            let sink = sinks.reshape((1, h, 1, 1))?.broadcast_as((b, h, sq, 1))?;
            let logits = Tensor::cat(&[&att, &sink.contiguous()?], D::Minus1)?;
            let probs = candle_nn::ops::softmax(&logits, D::Minus1)?;
            probs.narrow(D::Minus1, 0, case.seq_kv)?.contiguous()?
        }
        None => candle_nn::ops::softmax(&att, D::Minus1)?,
    };
    Ok(att.matmul(&v)?.transpose(1, 2)?)
}

fn check(case: Case) -> Result<()> {
    let Some(dev) = cuda() else { return Ok(()) };
    let scale = 1. / (case.head_dim as f32).sqrt();
    for (dtype, tolerance) in TOLERANCE {
        let rand =
            |s: usize, h: usize, d: usize| Tensor::randn(0f32, 1., (case.batch, s, h, d), &dev);
        let q_dtype = if case.f32_q { DType::F32 } else { dtype };
        // views are taken after the cast, which would otherwise make q contiguous
        let q = match case.q_layout {
            QLayout::Contiguous => rand(case.seq_q, N_HEAD, case.head_dim)?.to_dtype(q_dtype)?,
            QLayout::Transposed => Tensor::randn(
                0f32,
                1.,
                (case.batch, N_HEAD, case.seq_q, case.head_dim),
                &dev,
            )?
            .to_dtype(q_dtype)?
            .transpose(1, 2)?,
            QLayout::Misaligned => rand(case.seq_q, N_HEAD, case.head_dim + 1)?
                .to_dtype(q_dtype)?
                .narrow(3, 1, case.head_dim)?,
            QLayout::OffsetContiguous => {
                let n = case.batch * case.seq_q * N_HEAD * case.head_dim;
                Tensor::randn(0f32, 1., n + 1, &dev)?
                    .to_dtype(q_dtype)?
                    .narrow(0, 1, n)?
                    .reshape((case.batch, case.seq_q, N_HEAD, case.head_dim))?
            }
        };
        let k = rand(case.seq_kv, case.n_head_kv, case.head_dim)?.to_dtype(dtype)?;
        let v = if case.head_dim == MLA_HEAD_DIM {
            k.narrow(3, 0, case.head_dim_v)?
        } else {
            rand(case.seq_kv, case.n_head_kv, case.head_dim_v)?.to_dtype(dtype)?
        };
        let mask = case
            .causal
            .then(|| causal_mask(case.seq_q, case.seq_kv, &dev))
            .transpose()?;
        let sinks = case
            .sinks
            .then(|| Tensor::randn(0f32, 1., N_HEAD, &dev))
            .transpose()?;
        let opts = FattnOptions {
            scale,
            softcap: case.softcap,
            mask: mask.clone(),
            sinks: sinks.clone(),
            ..Default::default()
        };
        assert!(inference_fattn::supported(&q, &k, &v, &opts)?);
        let got = flash_attn(&q, &k, &v, &opts)?;
        assert_eq!(got.dtype(), q.dtype());
        assert_eq!(
            got.dims4()?,
            (case.batch, case.seq_q, N_HEAD, case.head_dim_v)
        );
        let want = reference(&q, &k, &v, mask.as_ref(), sinks.as_ref(), &case, scale)?;
        let diff = (got.to_dtype(DType::F32)? - &want)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        let peak = want.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        assert!(
            diff <= tolerance * peak,
            "hd {}/{} kv_heads {} q {} kv {} {dtype:?}: max diff {diff}, peak {peak}",
            case.head_dim,
            case.head_dim_v,
            case.n_head_kv,
            case.seq_q,
            case.seq_kv
        );
    }
    Ok(())
}

#[test]
fn decode_mma() -> Result<()> {
    for hd in [64, 128, 256] {
        check(case(hd, 2, 1, 512))?;
        check(case(hd, 2, 1, 300))?;
    }
    Ok(())
}

#[test]
fn decode_vec() -> Result<()> {
    // batch 1 and GQA 1: no GQA optimisation, so Ampere picks the vector kernel
    for hd in [64, 128, 256] {
        check(Case {
            batch: 1,
            ..case(hd, N_HEAD, 1, 512)
        })?;
    }
    for (q_layout, f32_q) in [(QLayout::Transposed, false), (QLayout::Contiguous, true)] {
        check(Case {
            batch: 1,
            q_layout,
            f32_q,
            ..case(128, N_HEAD, 1, 512)
        })?;
    }
    Ok(())
}

#[test]
fn prefill() -> Result<()> {
    for hd in [64, 80, 96, 112, 128, 256] {
        check(case(hd, 2, 200, 200))?;
    }
    Ok(())
}

#[test]
fn prefill_tile() -> Result<()> {
    // head dims 40 and 72 (SigLIP) run the tile kernel even with tensor cores
    for hd in [40, 72] {
        check(Case {
            causal: false,
            ..case(hd, N_HEAD, 128, 128)
        })?;
    }
    for (q_layout, f32_q) in [(QLayout::Transposed, false), (QLayout::Contiguous, true)] {
        check(Case {
            causal: false,
            q_layout,
            f32_q,
            ..case(72, N_HEAD, 128, 128)
        })?;
    }
    Ok(())
}

#[test]
fn prefill_after_a_cached_prefix() -> Result<()> {
    check(case(128, 8, 37, 512))
}

#[test]
fn encoder_without_a_mask() -> Result<()> {
    check(Case {
        causal: false,
        ..case(64, N_HEAD, 100, 100)
    })
}

#[test]
fn head_dim_512_with_gqa() -> Result<()> {
    check(case(512, 2, 64, 512))?;
    check(case(512, 2, 1, 512))
}

#[test]
fn mla_shaped_head_dims() -> Result<()> {
    // 192/128 and 576/512 need the GQA optimisation (mask, KV padded to 256, GQA >= 8); 576 takes v as a view of k
    check(Case {
        head_dim_v: 128,
        ..case(192, 1, 32, 512)
    })?;
    check(Case {
        head_dim_v: 512,
        ..case(576, 1, 16, 512)
    })?;
    check(Case {
        head_dim_v: 512,
        ..case(576, 1, 1, 512)
    })
}

#[test]
fn softcap() -> Result<()> {
    check(Case {
        softcap: 30.,
        ..case(128, 2, 100, 256)
    })?;
    check(Case {
        softcap: 30.,
        ..case(256, 2, 1, 256)
    })
}

#[test]
fn sinks() -> Result<()> {
    check(Case {
        sinks: true,
        ..case(64, 8, 50, 256)
    })?;
    check(Case {
        sinks: true,
        ..case(64, 8, 1, 256)
    })
}

#[test]
fn stream_k_fixups() -> Result<()> {
    // batch 1 with few output tiles: stream-k splits tiles across blocks and a fixup kernel writes the result
    for (hd, seq) in [(256, 512), (128, 256), (64, 1024)] {
        check(Case {
            batch: 1,
            ..case(hd, 2, seq, seq)
        })?;
    }
    Ok(())
}

#[test]
fn long_prefill_converts_bf16_kv_first() -> Result<()> {
    // past 512 Q rows the mma kernel takes f16 copies of bf16 K/V rather than converting in its tile loads
    check(case(128, 2, 600, 600))?;
    check(case(256, 8, 640, 1024))
}

#[test]
fn strided_misaligned_and_f32_queries() -> Result<()> {
    check(Case {
        q_layout: QLayout::Transposed,
        ..case(128, 2, 64, 256)
    })?;
    check(Case {
        q_layout: QLayout::Misaligned,
        ..case(128, 2, 64, 256)
    })?;
    check(Case {
        q_layout: QLayout::OffsetContiguous,
        ..case(128, 2, 64, 256)
    })?;
    check(Case {
        q_layout: QLayout::OffsetContiguous,
        f32_q: true,
        ..case(64, N_HEAD, 1, 256)
    })?;
    check(Case {
        f32_q: true,
        ..case(128, 2, 64, 256)
    })
}

#[test]
fn rejects_mismatched_operands() -> Result<()> {
    let Some(dev) = cuda() else { return Ok(()) };
    let t = |h: usize, d: usize| Tensor::zeros((1, 8, h, d), DType::F16, &dev);
    let opts = FattnOptions {
        scale: 1.,
        ..Default::default()
    };
    // n_head 8 over 3 KV heads would trip a process-aborting GGML_ASSERT without the Rust-side check
    assert!(flash_attn(&t(8, 64)?, &t(3, 64)?, &t(3, 64)?, &opts).is_err());
    assert!(flash_attn(&t(8, 64)?, &t(2, 128)?, &t(2, 64)?, &opts).is_err());
    let bad_mask = FattnOptions {
        mask: Some(Tensor::zeros((1, 8, 7), DType::F16, &dev)?),
        ..opts.clone()
    };
    assert!(flash_attn(&t(8, 64)?, &t(2, 64)?, &t(2, 64)?, &bad_mask).is_err());
    let mla = |d: usize| Tensor::zeros((1, 8, 1, d), DType::F16, &dev);
    let separate_v = flash_attn(&t(8, MLA_HEAD_DIM)?, &mla(MLA_HEAD_DIM)?, &mla(512)?, &opts);
    assert!(separate_v.is_err());
    let odd_k = Tensor::zeros((1, 8, 2, 65), DType::F16, &dev)?.narrow(3, 1, 64)?;
    assert!(flash_attn(&t(8, 64)?, &odd_k, &t(2, 64)?, &opts).is_err());
    let f32_mask = FattnOptions {
        mask: Some(Tensor::zeros((1, 8, 8), DType::F32, &dev)?),
        ..opts
    };
    assert!(flash_attn(&t(8, 64)?, &t(2, 64)?, &t(2, 64)?, &f32_mask).is_err());
    Ok(())
}
