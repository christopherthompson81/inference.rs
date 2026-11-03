use std::collections::HashMap;

use inference_tensor::{DType, Device, IndexOp, Result, Tensor};

use super::{copy_blocks, gather_kv_cache, paged_attention, reshape_and_cache};

const KV_HEADS: usize = 2;
const HEADS: usize = 4;
const HEAD_SIZE: usize = 32;
const BLOCK_SIZE: usize = 4;
const BLOCKS: usize = 8;

// softcap, ALiBi slopes, sinks
type DecodeCase<'a> = (f32, &'a [f32], Option<&'a [f32]>);

struct Fixture {
    key_cache: Tensor,
    value_cache: Tensor,
    keys: Tensor,
    values: Tensor,
    // each sequence's blocks, out of order so a wrong table read shows
    tables: Vec<Vec<u32>>,
    lens: Vec<usize>,
}

fn rand(shape: &[usize], seed: u64) -> Result<Tensor> {
    let n = shape.iter().product::<usize>();
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let data = (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
        })
        .collect::<Vec<_>>();
    Tensor::from_vec(data, shape, &Device::Cpu)
}

fn fixture(dtype: DType) -> Result<Fixture> {
    let x = 16 / dtype.size_in_bytes();
    let key_cache = Tensor::zeros(
        (BLOCKS, KV_HEADS, HEAD_SIZE / x, BLOCK_SIZE, x),
        dtype,
        &Device::Cpu,
    )?;
    let value_cache = Tensor::zeros(
        (BLOCKS, KV_HEADS, HEAD_SIZE, BLOCK_SIZE),
        dtype,
        &Device::Cpu,
    )?;
    let tables = vec![vec![5, 2], vec![7, 0, 3]];
    let lens = vec![6, 9];
    let tokens = lens.iter().sum::<usize>();
    let keys = rand(&[tokens, KV_HEADS, HEAD_SIZE], 1)?.to_dtype(dtype)?;
    let values = rand(&[tokens, KV_HEADS, HEAD_SIZE], 2)?.to_dtype(dtype)?;
    let mut slots = Vec::new();
    for (table, &len) in tables.iter().zip(&lens) {
        slots.extend(
            (0..len).map(|t| (table[t / BLOCK_SIZE] as usize * BLOCK_SIZE + t % BLOCK_SIZE) as i64),
        );
    }
    // a padding row past the real ones must leave the cache alone
    let pad_keys = Tensor::cat(&[&keys, &keys.i(0..1)?], 0)?;
    let pad_values = Tensor::cat(&[&values, &values.i(0..1)?], 0)?;
    slots.push(-1);
    let slot_mapping = Tensor::new(slots.as_slice(), &Device::Cpu)?;
    reshape_and_cache(
        &pad_keys,
        &pad_values,
        None,
        None,
        &key_cache,
        &value_cache,
        &slot_mapping,
    )?;
    Ok(Fixture {
        key_cache,
        value_cache,
        keys,
        values,
        tables,
        lens,
    })
}

fn table_tensor(tables: &[Vec<u32>]) -> Result<Tensor> {
    let width = tables.iter().map(Vec::len).max().unwrap_or(0);
    let flat = tables
        .iter()
        .flat_map(|t| t.iter().copied().chain(std::iter::repeat(0)).take(width))
        .collect::<Vec<_>>();
    Tensor::from_vec(flat, (tables.len(), width), &Device::Cpu)
}

fn max_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    (a.to_dtype(DType::F32)? - b.to_dtype(DType::F32)?)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()
}

#[test]
fn gather_recovers_the_written_rows() -> Result<()> {
    for dtype in [DType::F32, DType::BF16, DType::F16] {
        let f = fixture(dtype)?;
        let mut cu = vec![0u32];
        for &len in &f.lens {
            cu.push(cu.last().unwrap() + len as u32);
        }
        let tokens = *cu.last().unwrap() as usize;
        let (k, v) = gather_kv_cache(
            &f.key_cache,
            &f.value_cache,
            None,
            None,
            &table_tensor(&f.tables)?,
            &Tensor::new(cu.as_slice(), &Device::Cpu)?,
            tokens,
            DType::F32,
        )?;
        assert_eq!(max_diff(&k, &f.keys)?, 0.0, "{dtype:?} keys");
        assert_eq!(max_diff(&v, &f.values)?, 0.0, "{dtype:?} values");
    }
    Ok(())
}

// softmax(scale * q k^T, capped, sloped, with a sink) v over each sequence's rows, one query row per sequence
fn reference_decode(
    f: &Fixture,
    q: &Tensor,
    scale: f32,
    softcap: f32,
    slopes: &[f32],
    sinks: &[f32],
) -> Result<Tensor> {
    let mut rows = Vec::new();
    let mut start = 0;
    for (seq, &len) in f.lens.iter().enumerate() {
        let keys = f.keys.i(start..start + len)?.to_dtype(DType::F32)?;
        let values = f.values.i(start..start + len)?.to_dtype(DType::F32)?;
        for head in 0..HEADS {
            let kv = head / (HEADS / KV_HEADS);
            let qh = q.i((seq, head))?.to_dtype(DType::F32)?;
            let k = keys.i((.., kv))?;
            let v = values.i((.., kv))?;
            let mut s = (k.matmul(&qh.unsqueeze(1)?)?.squeeze(1)? * scale as f64)?;
            if softcap != 1.0 {
                s = ((s / softcap as f64)?.tanh()? * softcap as f64)?;
            }
            let pos = Tensor::arange(0f32, len as f32, &Device::Cpu)?;
            s = (s + ((pos - (len as f64 - 1.0))? * slopes[head] as f64)?)?;
            let s = s.to_vec1::<f32>()?;
            let max = s.iter().copied().fold(sinks[head], f32::max);
            let e = s.iter().map(|x| (x - max).exp()).collect::<Vec<_>>();
            let sum = e.iter().sum::<f32>() + (sinks[head] - max).exp();
            let p = Tensor::new(
                e.iter().map(|x| x / sum).collect::<Vec<_>>().as_slice(),
                &Device::Cpu,
            )?;
            rows.push(p.unsqueeze(0)?.matmul(&v)?.squeeze(0)?);
        }
        start += len;
    }
    Tensor::stack(&rows, 0)?.reshape((f.lens.len(), HEADS, HEAD_SIZE))
}

#[test]
fn paged_decode_matches_softmax_attention() -> Result<()> {
    let scale = 1.0 / (HEAD_SIZE as f32).sqrt();
    let no_slopes = vec![0f32; HEADS];
    let slopes = vec![0.5f32, 0.25, 0.125, 0.0625];
    // a sink far below every score adds nothing
    let no_sinks = vec![-1e30f32; HEADS];
    let sinks = vec![0.3f32, -0.2, 1.0, 0.0];
    for dtype in [DType::F32, DType::BF16] {
        let f = fixture(dtype)?;
        let q = rand(&[f.lens.len(), HEADS, HEAD_SIZE], 3)?.to_dtype(dtype)?;
        let lens = Tensor::new(
            f.lens
                .iter()
                .map(|&l| l as u32)
                .collect::<Vec<_>>()
                .as_slice(),
            &Device::Cpu,
        )?;
        let tables = table_tensor(&f.tables)?;
        let cases: [DecodeCase<'_>; 4] = [
            (1.0, &no_slopes, None),
            (5.0, &no_slopes, None),
            (1.0, &slopes, None),
            (1.0, &no_slopes, Some(&sinks)),
        ];
        for (softcap, case_slopes, case_sinks) in cases {
            let alibi = Tensor::new(case_slopes, &Device::Cpu)?;
            let sink_t = case_sinks
                .map(|s| Tensor::new(s, &Device::Cpu))
                .transpose()?;
            let got = paged_attention(
                &q,
                None,
                None,
                &f.key_cache,
                &f.value_cache,
                &tables,
                &lens,
                Some(&alibi),
                *f.lens.iter().max().unwrap(),
                scale,
                softcap,
                sink_t.as_ref(),
            )?;
            assert_eq!(got.dtype(), dtype);
            let want = reference_decode(
                &f,
                &q,
                scale,
                softcap,
                case_slopes,
                case_sinks.unwrap_or(&no_sinks),
            )?;
            let tol = if dtype == DType::F32 { 1e-5 } else { 2e-2 };
            let diff = max_diff(&got, &want)?;
            assert!(
                diff < tol,
                "{dtype:?} softcap {softcap} sinks {}: {diff}",
                case_sinks.is_some()
            );
        }
    }
    Ok(())
}

#[test]
fn copy_blocks_duplicates_the_source_block() -> Result<()> {
    let f = fixture(DType::F32)?;
    let mut key_cache = f.key_cache.copy()?;
    let mut value_cache = f.value_cache.copy()?;
    copy_blocks(
        vec![&mut key_cache],
        vec![&mut value_cache],
        &HashMap::from([(5, vec![1, 4])]),
    )?;
    for dst in [1, 4] {
        assert_eq!(max_diff(&key_cache.i(dst)?, &f.key_cache.i(5)?)?, 0.0);
        assert_eq!(max_diff(&value_cache.i(dst)?, &f.value_cache.i(5)?)?, 0.0);
    }
    assert_eq!(max_diff(&key_cache.i(2)?, &f.key_cache.i(2)?)?, 0.0);
    Ok(())
}
