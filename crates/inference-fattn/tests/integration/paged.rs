//! Paged fattn against the dense reference per sequence, over shuffled block tables with NaN in every unused slot.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use inference_fattn::{
    FattnOptions, PagedKv, causal_mask, flash_attn_paged, paged_causal_mask, paged_kv_len,
};

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
    for (dtype, tolerance) in TOLERANCE {
        let rows: Vec<(Tensor, Tensor)> = seq_lens
            .iter()
            .map(|&len| -> Result<_> {
                let r = || Tensor::randn(0f32, 1., (len, h_kv, d), &Device::Cpu);
                Ok((r()?.to_dtype(dtype)?, r()?.to_dtype(dtype)?))
            })
            .collect::<Result<_>>()?;
        let cache = |pick: fn(&(Tensor, Tensor)) -> &Tensor| -> Result<Tensor> {
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
            Ok(Tensor::from_vec(data, (NUM_BLOCKS, h_kv, bs, d), &dev)?.to_dtype(dtype)?)
        };
        let (k_cache, v_cache) = (cache(|r| &r.0)?, cache(|r| &r.1)?);
        let block_table = Tensor::from_vec(tables.clone(), (b, max_blocks), &dev)?;
        let lens: Vec<u32> = seq_lens.iter().map(|&l| l as u32).collect();
        let seq_lens_t = Tensor::from_vec(lens, b, &dev)?;
        let kv = PagedKv {
            k_cache: &k_cache,
            v_cache: &v_cache,
            block_table: &block_table,
            seq_lens: &seq_lens_t,
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
            ..Default::default()
        };
        let got = flash_attn_paged(&q, &kv, &opts)?;
        assert_eq!(got.dims4()?, (b, seq_q, N_HEAD, d));
        for (s, &len) in seq_lens.iter().enumerate() {
            let seq = Case {
                batch: 1,
                q_layout: QLayout::Contiguous,
                ..case(d, h_kv, seq_q, len)
            };
            let k = rows[s].0.to_device(&dev)?.unsqueeze(0)?;
            let v = rows[s].1.to_device(&dev)?.unsqueeze(0)?;
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
                "hd {d} bs {bs} seq {s} (len {len}) q {seq_q} {dtype:?}: max diff {diff}, peak {peak}"
            );
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
fn rejects_a_missing_or_short_mask_and_odd_blocks() -> Result<()> {
    let Some(dev) = cuda() else { return Ok(()) };
    let cache = Tensor::zeros((4, 2, 32, 64), DType::BF16, &dev)?;
    let block_table = Tensor::zeros((1, 2), DType::U32, &dev)?;
    let seq_lens = Tensor::ones(1, DType::U32, &dev)?;
    let kv = PagedKv {
        k_cache: &cache,
        v_cache: &cache,
        block_table: &block_table,
        seq_lens: &seq_lens,
    };
    let q = Tensor::zeros((1, 1, 8, 64), DType::BF16, &dev)?;
    let mut opts = FattnOptions {
        scale: 1.,
        ..Default::default()
    };
    assert!(flash_attn_paged(&q, &kv, &opts).is_err());
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
