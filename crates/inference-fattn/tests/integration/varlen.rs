//! Varlen fattn (sequences packed along dim 0) against the dense reference per sequence, over packed K/V with NaN
//! past the last sequence and over a paged cache.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use inference_fattn::{
    FattnOptions, KvScales, Packed, PagedKv, flash_attn_paged_varlen, flash_attn_varlen,
    paged_kv_len, varlen_causal_mask, varlen_kv_len,
};

use crate::fp8::{FP8_SCALES, FP8_TOLERANCE, store};
use crate::parity::{Case, N_HEAD, TOLERANCE, case, cuda, reference};

// Rows of NaN after the packed K/V, so a read past the last sequence shows up.
const NAN_TAIL: usize = 300;
const BLOCK_SIZE: usize = 32;
const Q_LENS: &[usize] = &[5, 37, 1, 64];
// the last K/V length leaves its final KV tile partial, so the NaN tail would be read without the redirect
const KV_LENS: &[usize] = &[5, 100, 40, 70];

#[derive(Clone, Copy)]
struct VarlenCase {
    head_dim: usize,
    n_head_kv: usize,
    q_lens: &'static [usize],
    kv_lens: &'static [usize],
}

// Options the dense varlen check layers on top of a case.
#[derive(Clone, Copy, Default)]
struct Extras {
    fp8: bool,
    softcap: f32,
    sinks: bool,
    // added to the true longest Q length, as a caller sizing the grid generously would
    max_q_slack: usize,
    // implicit-mask-only modes: a causal sliding window, or no causality at all (an encoder over packed sequences)
    window_left: Option<usize>,
    bidirectional: bool,
}

impl Extras {
    fn implicit_only(&self) -> bool {
        self.window_left.is_some() || self.bidirectional
    }
}

// The reference's mask for one sequence: causal (optionally windowed) with the queries last, or none.
fn reference_mask(
    q_len: usize,
    kv_len: usize,
    extras: Extras,
    dev: &Device,
) -> Result<Option<Tensor>> {
    if extras.bidirectional {
        return Ok(None);
    }
    let offset = kv_len - q_len;
    let window = extras.window_left.unwrap_or(usize::MAX);
    let mask: Vec<f32> = (0..q_len)
        .flat_map(|i| {
            (0..kv_len).map(move |j| {
                let qp = i + offset;
                if j <= qp && qp - j <= window {
                    0.
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect();
    Ok(Some(
        Tensor::from_vec(mask, (1, q_len, kv_len), dev)?.to_dtype(DType::F16)?,
    ))
}

// The kernel's own mask for these modes (causal by default).
fn implicit_opts(opts: &FattnOptions, extras: Extras) -> FattnOptions {
    FattnOptions {
        causal: !extras.bidirectional,
        window_left: extras.window_left,
        mask: None,
        ..opts.clone()
    }
}

fn cu(lens: &[usize], dev: &Device) -> Result<Tensor> {
    let mut acc = vec![0u32];
    for l in lens {
        acc.push(acc.last().unwrap() + *l as u32);
    }
    Ok(Tensor::from_vec(acc, lens.len() + 1, dev)?)
}

fn starts(lens: &[usize]) -> Vec<usize> {
    lens.iter()
        .scan(0, |at, l| Some(std::mem::replace(at, *at + l)))
        .collect()
}

// Packed K/V of `total` rows as the head of a buffer whose tail is NaN, and the f32 values those rows stand for.
// fp8 keeps no NaN through dequantization, so its tail is only large.
fn packed_kv(
    total: usize,
    h_kv: usize,
    d: usize,
    dtype: DType,
    scale: f32,
    dev: &Device,
) -> Result<(Tensor, Tensor)> {
    let rows = Tensor::randn(0f32, 1., (total, h_kv, d), &Device::Cpu)?;
    let tail = Tensor::full(f32::NAN, (NAN_TAIL, h_kv, d), &Device::Cpu)?;
    let (stored, values) = store(&Tensor::cat(&[&rows, &tail], 0)?, dtype, scale)?;
    Ok((
        stored.to_device(dev)?.narrow(0, 0, total)?,
        values.to_device(dev)?.narrow(0, 0, total)?,
    ))
}

fn compare(
    got: &Tensor,
    (q, k, v): (&Tensor, &Tensor, &Tensor),
    c: &VarlenCase,
    extras: Extras,
    sinks: Option<&Tensor>,
    tolerance: f32,
) -> Result<()> {
    let dev = q.device();
    let d = c.head_dim;
    let scale = 1. / (d as f32).sqrt();
    let (q_starts, kv_starts) = (starts(c.q_lens), starts(c.kv_lens));
    for s in 0..c.q_lens.len() {
        let (q_len, kv_len) = (c.q_lens[s], c.kv_lens[s]);
        let seq = |t: &Tensor, at: usize, len: usize| t.narrow(0, at, len)?.unsqueeze(0);
        let q_s = seq(q, q_starts[s], q_len)?;
        let (k_s, v_s) = (seq(k, kv_starts[s], kv_len)?, seq(v, kv_starts[s], kv_len)?);
        let mask = reference_mask(q_len, kv_len, extras, dev)?;
        let reference_case = Case {
            softcap: extras.softcap,
            sinks: extras.sinks,
            ..case(d, c.n_head_kv, q_len, kv_len)
        };
        let want = reference(
            &q_s,
            &k_s,
            &v_s,
            mask.as_ref(),
            sinks,
            &reference_case,
            scale,
        )?
        .squeeze(0)?;
        let got_s = got.narrow(0, q_starts[s], q_len)?.to_dtype(DType::F32)?;
        let diff = (got_s - &want)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        let peak = want.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        assert!(
            diff <= tolerance * peak,
            "hd {d} seq {s} (q {q_len} kv {kv_len}) {:?}: max diff {diff}, peak {peak}",
            q.dtype()
        );
    }
    Ok(())
}

fn check_dense(c: VarlenCase, extras: Extras) -> Result<()> {
    let Some(dev) = cuda() else { return Ok(()) };
    let (tq, tk): (usize, usize) = (c.q_lens.iter().sum(), c.kv_lens.iter().sum());
    let d = c.head_dim;
    // (q dtype, K/V dtype, tolerance)
    let modes: Vec<(DType, DType, f32)> = if extras.fp8 {
        vec![(DType::BF16, DType::F8E4M3, FP8_TOLERANCE)]
    } else {
        TOLERANCE.iter().map(|&(dt, tol)| (dt, dt, tol)).collect()
    };
    let scales = if extras.fp8 {
        FP8_SCALES
    } else {
        KvScales::default()
    };
    for (dtype, kv_dtype, tolerance) in modes {
        let q = Tensor::randn(0f32, 1., (tq, N_HEAD, d), &dev)?.to_dtype(dtype)?;
        let (k, k_ref) = packed_kv(tk, c.n_head_kv, d, kv_dtype, scales.k, &dev)?;
        let (v, v_ref) = packed_kv(tk, c.n_head_kv, d, kv_dtype, scales.v, &dev)?;
        let (cu_q, cu_kv) = (cu(c.q_lens, &dev)?, cu(c.kv_lens, &dev)?);
        let q_seqs = Packed {
            cu_seqlens: &cu_q,
            max_len: (c.q_lens.iter().max().unwrap() + extras.max_q_slack).min(tq),
        };
        let kv_seqs = Packed {
            cu_seqlens: &cu_kv,
            max_len: *c.kv_lens.iter().max().unwrap(),
        };
        let n_kv = varlen_kv_len(kv_seqs.max_len);
        let mut mask = varlen_causal_mask(c.q_lens, c.kv_lens, n_kv, &dev)?;
        // a generous max_len needs mask rows up to it
        if q_seqs.max_len > mask.dim(1)? {
            let rows = q_seqs.max_len - mask.dim(1)?;
            let pad = Tensor::full(f32::NEG_INFINITY, (c.q_lens.len(), rows, n_kv), &dev)?;
            mask = Tensor::cat(&[&mask, &pad.to_dtype(DType::F16)?], 1)?;
        }
        let sinks = extras
            .sinks
            .then(|| Tensor::randn(0f32, 1., N_HEAD, &dev))
            .transpose()?;
        let opts = FattnOptions {
            scale: 1. / (d as f32).sqrt(),
            softcap: extras.softcap,
            mask: Some(mask),
            sinks: sinks.clone(),
            kv_scales: extras.fp8.then_some(scales),
            ..Default::default()
        };
        let implicit = implicit_opts(&opts, extras);
        let runs: &[&FattnOptions] = if extras.implicit_only() {
            &[&implicit]
        } else {
            &[&opts, &implicit]
        };
        for opts in runs {
            let got = flash_attn_varlen(&q, &k, &v, &q_seqs, &kv_seqs, opts)?;
            assert_eq!(got.dims3()?, (tq, N_HEAD, d));
            compare(
                &got,
                (&q, &k_ref, &v_ref),
                &c,
                extras,
                sinks.as_ref(),
                tolerance,
            )?;
        }
    }
    Ok(())
}

// The same sequences in a paged cache: sequence s's blocks are laid out in reverse order of the pool.
fn check_paged(c: VarlenCase, extras: Extras) -> Result<()> {
    let Some(dev) = cuda() else { return Ok(()) };
    let (tq, tk): (usize, usize) = (c.q_lens.iter().sum(), c.kv_lens.iter().sum());
    let d = c.head_dim;
    let h_kv = c.n_head_kv;
    let b = c.q_lens.len();
    let blocks_of = |l: &usize| l.div_ceil(BLOCK_SIZE);
    let max_blocks = c.kv_lens.iter().map(blocks_of).max().unwrap();
    let num_blocks: usize = c.kv_lens.iter().map(blocks_of).sum();
    let mut table = vec![0u32; b * max_blocks];
    let mut next = num_blocks;
    for (s, len) in c.kv_lens.iter().enumerate() {
        for j in 0..blocks_of(len) {
            next -= 1;
            table[s * max_blocks + j] = next as u32;
        }
    }
    for (dtype, tolerance) in TOLERANCE {
        let q = Tensor::randn(0f32, 1., (tq, N_HEAD, d), &dev)?.to_dtype(dtype)?;
        let k = Tensor::randn(0f32, 1., (tk, h_kv, d), &dev)?.to_dtype(dtype)?;
        let v = Tensor::randn(0f32, 1., (tk, h_kv, d), &dev)?.to_dtype(dtype)?;
        let nan = |rows: usize| Tensor::full(f32::NAN, (h_kv, rows, d), &dev)?.to_dtype(dtype);
        let cache = |t: &Tensor| -> Result<Tensor> {
            let mut blocks = vec![nan(BLOCK_SIZE)?; num_blocks];
            for (s, at) in starts(c.kv_lens).into_iter().enumerate() {
                for j in 0..blocks_of(&c.kv_lens[s]) {
                    let rows = (c.kv_lens[s] - j * BLOCK_SIZE).min(BLOCK_SIZE);
                    let part = t.narrow(0, at + j * BLOCK_SIZE, rows)?.transpose(0, 1)?;
                    // a full block has no padding, and a zero-row tensor cannot be allocated on CUDA
                    blocks[table[s * max_blocks + j] as usize] = if rows == BLOCK_SIZE {
                        part.contiguous()?
                    } else {
                        Tensor::cat(&[&part, &nan(BLOCK_SIZE - rows)?], 1)?
                    };
                }
            }
            Ok(Tensor::stack(&blocks, 0)?.contiguous()?)
        };
        let (k_cache, v_cache) = (cache(&k)?, cache(&v)?);
        let block_table = Tensor::from_vec(table.clone(), (b, max_blocks), &dev)?;
        let lens: Vec<u32> = c.kv_lens.iter().map(|&l| l as u32).collect();
        let seq_lens = Tensor::from_vec(lens, b, &dev)?;
        let kv = PagedKv {
            k_cache: &k_cache,
            v_cache: &v_cache,
            block_table: &block_table,
            seq_lens: &seq_lens,
            full_lens: None,
        };
        let cu_q = cu(c.q_lens, &dev)?;
        let q_seqs = Packed {
            cu_seqlens: &cu_q,
            max_len: *c.q_lens.iter().max().unwrap(),
        };
        let opts = FattnOptions {
            scale: 1. / (d as f32).sqrt(),
            mask: Some(varlen_causal_mask(
                c.q_lens,
                c.kv_lens,
                paged_kv_len(&kv)?,
                &dev,
            )?),
            ..Default::default()
        };
        let implicit = implicit_opts(&opts, extras);
        let runs: &[&FattnOptions] = if extras.implicit_only() {
            &[&implicit]
        } else {
            &[&opts, &implicit]
        };
        for opts in runs {
            let got = flash_attn_paged_varlen(&q, &q_seqs, &kv, opts)?;
            compare(&got, (&q, &k, &v), &c, extras, None, tolerance)?;
        }
    }
    Ok(())
}

#[test]
fn packed_prefill_and_chunks() -> Result<()> {
    for (d, h_kv) in [(64, 4), (128, 2), (256, 2)] {
        let c = VarlenCase {
            head_dim: d,
            n_head_kv: h_kv,
            q_lens: Q_LENS,
            kv_lens: KV_LENS,
        };
        check_dense(c, Extras::default())?;
    }
    // no GQA, and one long sequence beside short ones (whole Q tiles of the short ones are skipped)
    let long_beside_short = VarlenCase {
        head_dim: 128,
        n_head_kv: N_HEAD,
        q_lens: &[300, 3, 17],
        kv_lens: &[300, 3, 600],
    };
    check_dense(long_beside_short, Extras::default())
}

#[test]
fn fp8_softcap_sinks_one_sequence_and_a_generous_max_len() -> Result<()> {
    let c = VarlenCase {
        head_dim: 128,
        n_head_kv: 2,
        q_lens: Q_LENS,
        kv_lens: KV_LENS,
    };
    check_dense(
        c,
        Extras {
            fp8: true,
            ..Default::default()
        },
    )?;
    check_dense(
        c,
        Extras {
            softcap: 30.,
            sinks: true,
            ..Default::default()
        },
    )?;
    check_dense(
        c,
        Extras {
            max_q_slack: 20,
            ..Default::default()
        },
    )?;
    let one = VarlenCase {
        head_dim: 64,
        n_head_kv: 2,
        q_lens: &[33],
        kv_lens: &[90],
    };
    check_dense(one, Extras::default())
}

#[test]
fn packed_queries_over_a_paged_cache() -> Result<()> {
    for d in [64, 128] {
        check_paged(
            VarlenCase {
                head_dim: d,
                n_head_kv: 2,
                q_lens: Q_LENS,
                kv_lens: KV_LENS,
            },
            Extras::default(),
        )?;
    }
    Ok(())
}

#[test]
fn rejects_bad_sequences_and_masks() -> Result<()> {
    let Some(dev) = cuda() else { return Ok(()) };
    let q = Tensor::zeros((10, 8, 64), DType::BF16, &dev)?;
    let k = Tensor::zeros((12, 2, 64), DType::BF16, &dev)?;
    let cu_q = cu(&[4, 6], &dev)?;
    let cu_kv = cu(&[5, 7], &dev)?;
    let q_seqs = Packed {
        cu_seqlens: &cu_q,
        max_len: 6,
    };
    let kv_seqs = Packed {
        cu_seqlens: &cu_kv,
        max_len: 7,
    };
    let mut opts = FattnOptions {
        scale: 1.,
        ..Default::default()
    };
    // no mask (the kernel masks from the lengths), then a mask of the wrong width, then the right one
    assert!(flash_attn_varlen(&q, &k, &k, &q_seqs, &kv_seqs, &opts).is_ok());
    opts.mask = Some(Tensor::zeros((2, 6, 7), DType::F16, &dev)?);
    assert!(flash_attn_varlen(&q, &k, &k, &q_seqs, &kv_seqs, &opts).is_err());
    opts.mask = Some(Tensor::zeros((2, 6, varlen_kv_len(7)), DType::F16, &dev)?);
    assert!(flash_attn_varlen(&q, &k, &k, &q_seqs, &kv_seqs, &opts).is_ok());
    // a batch mismatch between Q and K/V, and a max_len past the rows
    let cu_three = cu(&[4, 4, 4], &dev)?;
    let three = Packed {
        cu_seqlens: &cu_three,
        max_len: 4,
    };
    assert!(flash_attn_varlen(&q, &k, &k, &q_seqs, &three, &opts).is_err());
    let too_long = Packed {
        max_len: 11,
        ..q_seqs
    };
    assert!(flash_attn_varlen(&q, &k, &k, &too_long, &kv_seqs, &opts).is_err());
    Ok(())
}

#[test]
fn implicit_windows_and_bidirectional_packed_sequences() -> Result<()> {
    let c = VarlenCase {
        head_dim: 128,
        n_head_kv: 2,
        q_lens: Q_LENS,
        kv_lens: KV_LENS,
    };
    let window = Extras {
        window_left: Some(30),
        ..Default::default()
    };
    check_dense(c, window)?;
    check_paged(c, window)?;
    check_dense(
        c,
        Extras {
            fp8: true,
            ..window
        },
    )?;
    // an encoder over packed sequences: every query sees its whole sequence
    let encoder = VarlenCase {
        head_dim: 64,
        n_head_kv: N_HEAD,
        q_lens: &[20, 77, 5],
        kv_lens: &[20, 77, 5],
    };
    check_dense(
        encoder,
        Extras {
            bidirectional: true,
            ..Default::default()
        },
    )
}
