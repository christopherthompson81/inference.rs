//! Paged fattn against the dense reference per sequence, over shuffled block tables with NaN in every unused slot.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use inference_fattn::{
    FattnOptions, KvScales, PagedKv, causal_mask, flash_attn_paged, paged_causal_mask,
    paged_kv_len, supported_paged,
};

use crate::fp8::{FP8_SCALES, FP8_TOLERANCE, store};
use crate::parity::{Case, N_HEAD, QLayout, TOLERANCE, case, cuda, reference};

const NUM_BLOCKS: usize = 48;
// Multiplier of a full-period LCG over block ids, so each sequence gets scattered, distinct blocks.
const BLOCK_SHUFFLE: usize = 29;

struct PagedCase {
    head_dim: usize,
    n_head_kv: usize,
    block_size: usize,
    seq_lens: &'static [usize],
    seq_q: usize,
    // table entries past what the longest sequence needs, as an over-allocated table has
    spare_blocks: usize,
}

fn check(c: PagedCase) -> Result<()> {
    let Some(dev) = cuda() else { return Ok(()) };
    let PagedCase {
        head_dim: d,
        n_head_kv: h_kv,
        block_size: bs,
        seq_lens,
        seq_q,
        spare_blocks,
    } = c;
    let b = seq_lens.len();
    let max_blocks = seq_lens.iter().map(|l| l.div_ceil(bs)).max().unwrap() + spare_blocks;
    let mut tables = vec![0u32; b * max_blocks];
    let mut next = 0;
    for (s, len) in seq_lens.iter().enumerate() {
        for j in 0..len.div_ceil(bs) {
            tables[s * max_blocks + j] = ((next * BLOCK_SHUFFLE + 7) % NUM_BLOCKS) as u32;
            next += 1;
        }
    }
    assert!(
        next <= NUM_BLOCKS,
        "the case needs more than {NUM_BLOCKS} blocks"
    );
    let scale = 1. / (d as f32).sqrt();
    // (q dtype, cache dtype, tolerance, fp8 scales)
    let modes = TOLERANCE
        .map(|(dtype, tol)| (dtype, dtype, tol, None))
        .into_iter()
        .chain([(DType::BF16, DType::F8E4M3, FP8_TOLERANCE, Some(FP8_SCALES))]);
    for (dtype, kv_dtype, tolerance, kv_scales) in modes {
        // per sequence: the stored K/V, then the values they stand for (dequantized for fp8)
        let rows: Vec<((Tensor, Tensor), (Tensor, Tensor))> = seq_lens
            .iter()
            .map(|&len| -> Result<_> {
                let scales = kv_scales.unwrap_or_default();
                let r = |s: f32| -> Result<(Tensor, Tensor)> {
                    store(
                        &Tensor::randn(0f32, 1., (len, h_kv, d), &Device::Cpu)?,
                        kv_dtype,
                        s,
                    )
                };
                let ((k, k_ref), (v, v_ref)) = (r(scales.k)?, r(scales.v)?);
                Ok(((k, v), (k_ref, v_ref)))
            })
            .collect::<Result<_>>()?;
        type Rows = ((Tensor, Tensor), (Tensor, Tensor));
        let cache = |pick: fn(&Rows) -> &Tensor| -> Result<Tensor> {
            let mut data = vec![f32::NAN; NUM_BLOCKS * h_kv * bs * d];
            for (s, len) in seq_lens.iter().enumerate() {
                let src = pick(&rows[s]).to_dtype(DType::F32)?.to_vec3::<f32>()?;
                for p in 0..*len {
                    let blk = tables[s * max_blocks + p / bs] as usize;
                    for (hd, row) in src[p].iter().enumerate() {
                        let at = ((blk * h_kv + hd) * bs + p % bs) * d;
                        data[at..at + d].copy_from_slice(row);
                    }
                }
            }
            Ok(
                Tensor::from_vec(data, (NUM_BLOCKS, h_kv, bs, d), &Device::Cpu)?
                    .to_dtype(kv_dtype)?
                    .to_device(&dev)?,
            )
        };
        let (k_cache, v_cache) = (cache(|r| &r.0.0)?, cache(|r| &r.0.1)?);
        let block_table = Tensor::from_vec(tables.clone(), (b, max_blocks), &dev)?;
        let lens: Vec<u32> = seq_lens.iter().map(|&l| l as u32).collect();
        let seq_lens_t = Tensor::from_vec(lens, b, &dev)?;
        let kv = PagedKv {
            k_cache: &k_cache,
            v_cache: &v_cache,
            block_table: &block_table,
            seq_lens: &seq_lens_t,
            full_lens: None,
        };
        let q = Tensor::randn(0f32, 1., (b, seq_q, N_HEAD, d), &dev)?.to_dtype(dtype)?;
        let opts = FattnOptions {
            scale,
            mask: Some(paged_causal_mask(
                seq_lens,
                seq_q,
                paged_kv_len(&kv)?,
                &dev,
            )?),
            kv_scales,
            ..Default::default()
        };
        let implicit = FattnOptions {
            causal: true,
            mask: None,
            ..opts.clone()
        };
        for opts in [&opts, &implicit] {
            let got = flash_attn_paged(&q, &kv, opts)?;
            assert_eq!(got.dims4()?, (b, seq_q, N_HEAD, d));
            for (s, &len) in seq_lens.iter().enumerate() {
                let seq = Case {
                    batch: 1,
                    q_layout: QLayout::Contiguous,
                    ..case(d, h_kv, seq_q, len)
                };
                let k = rows[s].1.0.to_device(&dev)?.unsqueeze(0)?;
                let v = rows[s].1.1.to_device(&dev)?.unsqueeze(0)?;
                let q_s = q.narrow(0, s, 1)?;
                let mask = causal_mask(seq_q, len, &dev)?;
                let want = reference(&q_s, &k, &v, Some(&mask), None, &seq, scale)?;
                let got_s = got.narrow(0, s, 1)?.to_dtype(DType::F32)?;
                let diff = (got_s - &want)?
                    .abs()?
                    .flatten_all()?
                    .max(0)?
                    .to_scalar::<f32>()?;
                let peak = want.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
                assert!(
                    diff <= tolerance * peak,
                    "hd {d} bs {bs} seq {s} (len {len}) q {seq_q} {kv_dtype:?}: max diff {diff}, peak {peak}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn decode() -> Result<()> {
    for d in [64, 128, 256] {
        check(PagedCase {
            head_dim: d,
            n_head_kv: 2,
            block_size: 32,
            seq_lens: &[100, 333, 37],
            seq_q: 1,
            spare_blocks: 0,
        })?;
    }
    Ok(())
}

#[test]
fn chunked_prefill_over_a_cached_prefix() -> Result<()> {
    check(PagedCase {
        head_dim: 128,
        n_head_kv: 2,
        block_size: 32,
        seq_lens: &[100, 333, 37],
        seq_q: 17,
        spare_blocks: 0,
    })
}

#[test]
fn small_blocks_and_no_gqa() -> Result<()> {
    check(PagedCase {
        head_dim: 128,
        n_head_kv: N_HEAD,
        block_size: 16,
        seq_lens: &[50, 129],
        seq_q: 1,
        spare_blocks: 0,
    })?;
    check(PagedCase {
        head_dim: 64,
        n_head_kv: 4,
        block_size: 16,
        seq_lens: &[64, 16, 200],
        seq_q: 9,
        spare_blocks: 0,
    })
}

#[test]
fn one_sequence_with_spare_table_entries() -> Result<()> {
    // batch 1 with blocks to spare: whole KV tiles past the sequence, which the mask scan must skip or the mask hide
    check(PagedCase {
        head_dim: 128,
        n_head_kv: 2,
        block_size: 32,
        seq_lens: &[70],
        seq_q: 1,
        spare_blocks: 20,
    })
}

#[test]
fn graph_padded_tables_split_by_live_length() -> Result<()> {
    // a CUDA graph's table spans a power-of-two bucket: stream-k splits the live rows only, so most blocks get none
    // 80 and 112 leave the fixups a half warp
    for (d, seq_q) in [(128, 1), (256, 1), (128, 4), (80, 1), (112, 2)] {
        check(PagedCase {
            head_dim: d,
            n_head_kv: 2,
            block_size: 32,
            seq_lens: &[40, 5, 300],
            seq_q,
            spare_blocks: 120,
        })?;
    }
    check(PagedCase {
        head_dim: 128,
        n_head_kv: 2,
        block_size: 16,
        seq_lens: &[9],
        seq_q: 1,
        spare_blocks: 250,
    })
}

#[test]
fn long_prefill_converts_in_the_loads() -> Result<()> {
    // past 512 Q rows dense bf16 K/V is converted first, which a paged cache cannot take
    check(PagedCase {
        head_dim: 128,
        n_head_kv: 2,
        block_size: 32,
        seq_lens: &[700],
        seq_q: 600,
        spare_blocks: 0,
    })
}

#[test]
fn rejects_a_short_mask_and_odd_blocks() -> Result<()> {
    let Some(dev) = cuda() else { return Ok(()) };
    let cache = Tensor::zeros((4, 2, 32, 64), DType::BF16, &dev)?;
    let block_table = Tensor::zeros((1, 2), DType::U32, &dev)?;
    let seq_lens = Tensor::ones(1, DType::U32, &dev)?;
    let kv = PagedKv {
        k_cache: &cache,
        v_cache: &cache,
        block_table: &block_table,
        seq_lens: &seq_lens,
        full_lens: None,
    };
    let q = Tensor::zeros((1, 1, 8, 64), DType::BF16, &dev)?;
    let mut opts = FattnOptions {
        scale: 1.,
        ..Default::default()
    };
    // no mask is fine (the kernel masks from the lengths); a mask of the wrong width is not
    assert!(flash_attn_paged(&q, &kv, &opts).is_ok());
    opts.mask = Some(Tensor::zeros((1, 1, 64), DType::F16, &dev)?);
    assert!(flash_attn_paged(&q, &kv, &opts).is_err());
    let odd_blocks = Tensor::zeros((4, 2, 24, 64), DType::BF16, &dev)?;
    let odd = PagedKv {
        k_cache: &odd_blocks,
        v_cache: &odd_blocks,
        ..kv
    };
    opts.mask = Some(Tensor::zeros(
        (1, 1, paged_kv_len(&odd)?),
        DType::F16,
        &dev,
    )?);
    assert!(flash_attn_paged(&q, &odd, &opts).is_err());
    Ok(())
}

#[test]
fn supported_paged_answers_for_kernel_limits() -> Result<()> {
    let Some(dev) = cuda() else { return Ok(()) };
    let block_table = Tensor::zeros((2, 3), DType::U32, &dev)?;
    let seq_lens = Tensor::ones(2, DType::U32, &dev)?;
    let probe = |dtype: DType, bs: usize, d: usize, opts: &FattnOptions| -> Result<bool> {
        let cache = Tensor::zeros((4, 2, bs, d), dtype, &dev)?;
        let kv = PagedKv {
            k_cache: &cache,
            v_cache: &cache,
            block_table: &block_table,
            seq_lens: &seq_lens,
            full_lens: None,
        };
        let q = Tensor::zeros((2, 1, 8, d), DType::BF16, &dev)?;
        let supported = supported_paged(&q, &kv, opts)?;
        // the probe must agree with the call it stands for
        assert_eq!(supported, flash_attn_paged(&q, &kv, opts).is_ok());
        Ok(supported)
    };
    let opts = FattnOptions {
        scale: 1.,
        causal: true,
        ..Default::default()
    };
    let windowed = FattnOptions {
        window_left: Some(7),
        softcap: 30.,
        ..opts.clone()
    };
    let fp8 = |k: f32| FattnOptions {
        kv_scales: Some(KvScales { k, v: 0.5 }),
        ..opts.clone()
    };
    for d in [64, 128, 256, 512] {
        assert!(probe(DType::BF16, 32, d, &opts)?, "head dim {d}");
        assert!(probe(DType::F8E4M3, 32, d, &fp8(0.25))?, "head dim {d}");
        // no kernel instantiates softcap at other head dims
        assert_eq!(
            probe(DType::F16, 16, d, &windowed)?,
            d != 64,
            "head dim {d}"
        );
    }
    assert!(!probe(DType::BF16, 24, 128, &opts)?);
    assert!(!probe(DType::F32, 32, 128, &opts)?);
    assert!(!probe(DType::F8E4M3, 32, 128, &fp8(200.))?);
    assert!(!probe(DType::BF16, 32, 576, &opts)?);
    // a malformed call is an error, not an unsupported one
    let cache = Tensor::zeros((4, 2, 32, 64), DType::BF16, &dev)?;
    let kv = PagedKv {
        k_cache: &cache,
        v_cache: &cache,
        block_table: &Tensor::zeros((3, 3), DType::U32, &dev)?,
        seq_lens: &seq_lens,
        full_lens: None,
    };
    assert!(
        supported_paged(
            &Tensor::zeros((2, 1, 8, 64), DType::BF16, &dev)?,
            &kv,
            &opts
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn chunks_count_from_full_lens() -> Result<()> {
    // tables that hold only each sequence's last rows (a window's), so chunk edges fall at full_lens-based positions
    let Some(dev) = cuda() else { return Ok(()) };
    let (d, h_kv, bs, chunk) = (128, 2, 16, 24);
    // (rows held, full length): the first row held sits mid-chunk, at a chunk edge, and at position 0
    let seqs = [(40usize, 70usize), (33, 81), (48, 48)];
    let scale = 1. / (d as f32).sqrt();
    for seq_q in [1, 3] {
        let b = seqs.len();
        let max_blocks = seqs.iter().map(|(l, _)| l.div_ceil(bs)).max().unwrap();
        let mut tables = vec![0u32; b * max_blocks];
        let mut next = 0;
        for (s, (len, _)) in seqs.iter().enumerate() {
            for j in 0..len.div_ceil(bs) {
                tables[s * max_blocks + j] = ((next * BLOCK_SHUFFLE + 7) % NUM_BLOCKS) as u32;
                next += 1;
            }
        }
        let rows: Vec<(Tensor, Tensor)> = seqs
            .iter()
            .map(|&(len, _)| -> Result<_> {
                let r = || Tensor::randn(0f32, 1., (len, h_kv, d), &Device::Cpu);
                Ok((r()?, r()?))
            })
            .collect::<Result<_>>()?;
        let cache = |pick: fn(&(Tensor, Tensor)) -> &Tensor| -> Result<Tensor> {
            let mut data = vec![f32::NAN; NUM_BLOCKS * h_kv * bs * d];
            for (s, (len, _)) in seqs.iter().enumerate() {
                let src = pick(&rows[s]).to_vec3::<f32>()?;
                for p in 0..*len {
                    let blk = tables[s * max_blocks + p / bs] as usize;
                    for (hd, row) in src[p].iter().enumerate() {
                        let at = ((blk * h_kv + hd) * bs + p % bs) * d;
                        data[at..at + d].copy_from_slice(row);
                    }
                }
            }
            Ok(
                Tensor::from_vec(data, (NUM_BLOCKS, h_kv, bs, d), &Device::Cpu)?
                    .to_dtype(DType::BF16)?
                    .to_device(&dev)?,
            )
        };
        let (k_cache, v_cache) = (cache(|r| &r.0)?, cache(|r| &r.1)?);
        let block_table = Tensor::from_vec(tables, (b, max_blocks), &dev)?;
        let lens = |pick: fn(&(usize, usize)) -> usize| -> Result<Tensor> {
            Ok(Tensor::from_vec(
                seqs.iter().map(|s| pick(s) as u32).collect::<Vec<_>>(),
                b,
                &dev,
            )?)
        };
        let (seq_lens, full_lens) = (lens(|s| s.0)?, lens(|s| s.1)?);
        let kv = PagedKv {
            k_cache: &k_cache,
            v_cache: &v_cache,
            block_table: &block_table,
            seq_lens: &seq_lens,
            full_lens: Some(&full_lens),
        };
        let q = Tensor::randn(0f32, 1., (b, seq_q, N_HEAD, d), &dev)?.to_dtype(DType::BF16)?;
        let opts = FattnOptions {
            scale,
            causal: true,
            chunk: Some(chunk),
            ..Default::default()
        };
        let got = flash_attn_paged(&q, &kv, &opts)?;
        for (s, &(len, full)) in seqs.iter().enumerate() {
            let start = full - len;
            let mask: Vec<f32> = (0..seq_q)
                .flat_map(|i| {
                    (0..len).map(move |j| {
                        let qp = len - seq_q + i;
                        if j <= qp && (start + j) / chunk == (start + qp) / chunk {
                            0.
                        } else {
                            f32::NEG_INFINITY
                        }
                    })
                })
                .collect();
            let mask = Tensor::from_vec(mask, (1, seq_q, len), &dev)?.to_dtype(DType::F16)?;
            let seq = Case {
                batch: 1,
                q_layout: QLayout::Contiguous,
                ..case(d, h_kv, seq_q, len)
            };
            let bf16 = |t: &Tensor| -> Result<Tensor> {
                Ok(t.to_dtype(DType::BF16)?.to_device(&dev)?.unsqueeze(0)?)
            };
            let (k, v) = (bf16(&rows[s].0)?, bf16(&rows[s].1)?);
            let want = reference(&q.narrow(0, s, 1)?, &k, &v, Some(&mask), None, &seq, scale)?;
            let diff = (got.narrow(0, s, 1)?.to_dtype(DType::F32)? - &want)?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_scalar::<f32>()?;
            let peak = want.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
            assert!(
                diff <= 1e-2 * peak,
                "seq {s} (rows {len} of {full}) q {seq_q}: max diff {diff}, peak {peak}"
            );
        }
    }
    Ok(())
}
