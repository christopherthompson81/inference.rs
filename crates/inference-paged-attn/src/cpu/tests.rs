use std::collections::HashMap;

use inference_tensor::{DType, Device, IndexOp, Result, Tensor};

use super::{NO_SOFTCAP, copy_blocks, gather_kv_cache, paged_attention, reshape_and_cache};

const KV_HEADS: usize = 2;
const HEADS: usize = 4;
const HEAD_SIZE: usize = 32;
const BLOCK_SIZE: usize = 4;
const BLOCKS: usize = 8;
// a Vulkan prompt row against the CPU, relative: the f32 kernel, and matrix cores (f16 Q, K, V and P; 1.0e-4 seen)
const VULKAN_PREFILL_TOLERANCE: f32 = 1e-4;
const VULKAN_MATRIX_CORE_PREFILL_TOLERANCE: f32 = 1e-3;

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

// Vulkan's cache writes and decode attention against the CPU's: the same rows written to both caches, then one query
// a sequence through block tables that skip around the cache
#[test]
fn vulkan_paged_attention_matches_the_cpu() -> Result<()> {
    let Ok(vk) = Device::new_vulkan(0) else {
        return Ok(());
    };
    const BLOCKS: usize = 6;
    const BLOCK: usize = 4;
    const KV_HEADS: usize = 2;
    const HEADS: usize = 4;
    const HEAD: usize = 64;
    const X: usize = 4;
    const TOKENS: usize = 13;
    let rows = |scale: f64| -> Result<Tensor> {
        Tensor::randn(0f32, 1., (TOKENS, KV_HEADS, HEAD), &Device::Cpu)? * scale
    };
    let (key, value) = (rows(1.0)?, rows(0.5)?);
    // seq 0 fills blocks 4 and 1 (tokens 0..7), seq 1 blocks 2 and 5 (6 tokens), one slot skipped
    let slots: Vec<i64> = (0..7)
        .map(|i| [4, 1][i / BLOCK] * BLOCK as i64 + (i % BLOCK) as i64)
        .chain([-1])
        .chain((0..5).map(|i| [2, 5][i / BLOCK] * BLOCK as i64 + (i % BLOCK) as i64))
        .collect();
    let slots = Tensor::new(slots.as_slice(), &Device::Cpu)?;
    let caches = |device: &Device| -> Result<(Tensor, Tensor)> {
        Ok((
            Tensor::zeros((BLOCKS, KV_HEADS, HEAD / X, BLOCK, X), DType::F32, device)?,
            Tensor::zeros((BLOCKS, KV_HEADS, HEAD, BLOCK), DType::F32, device)?,
        ))
    };
    let (kc, vc) = caches(&Device::Cpu)?;
    let (vkc, vvc) = caches(&vk)?;
    reshape_and_cache(&key, &value, None, None, &kc, &vc, &slots)?;
    reshape_and_cache(
        &key.to_device(&vk)?,
        &value.to_device(&vk)?,
        None,
        None,
        &vkc,
        &vvc,
        &slots.to_device(&vk)?,
    )?;
    assert_eq!(
        vkc.flatten_all()?.to_vec1::<f32>()?,
        kc.flatten_all()?.to_vec1::<f32>()?
    );
    assert_eq!(
        vvc.flatten_all()?.to_vec1::<f32>()?,
        vc.flatten_all()?.to_vec1::<f32>()?
    );
    let q = Tensor::randn(0f32, 1., (2, HEADS, HEAD), &Device::Cpu)?;
    let tables = Tensor::new(&[[4u32, 1], [2, 5]], &Device::Cpu)?;
    let lens = Tensor::new(&[7u32, 5], &Device::Cpu)?;
    let scale = 1.0 / (HEAD as f32).sqrt();
    let want = paged_attention(
        &q, None, None, &kc, &vc, &tables, &lens, None, 7, scale, NO_SOFTCAP, None,
    )?;
    let got = paged_attention(
        &q.to_device(&vk)?,
        None,
        None,
        &vkc,
        &vvc,
        &tables.to_device(&vk)?,
        &lens.to_device(&vk)?,
        None,
        7,
        scale,
        NO_SOFTCAP,
        None,
    )?;
    for (g, w) in got
        .flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .zip(want.flatten_all()?.to_vec1::<f32>()?)
    {
        assert!((g - w).abs() <= 1e-4 * w.abs().max(1.0), "{g} vs {w}");
    }
    Ok(())
}

// Nemotron-H's attention shape (32 heads over 2 kv heads, head 128) over contexts spanning several of the Vulkan
// kernel's chunks, one ending partway through a block, through a shuffled block table
#[test]
fn vulkan_paged_attention_splits_long_contexts() -> Result<()> {
    let Ok(vk) = Device::new_vulkan(0) else {
        return Ok(());
    };
    const BLOCK: usize = 16;
    const KV_HEADS: usize = 2;
    const HEADS: usize = 32;
    const HEAD: usize = 128;
    const X: usize = 4;
    const LENS: [u32; 2] = [700, 257];
    const MAX_BLOCKS: usize = 44;
    let blocks = 2 * MAX_BLOCKS;
    let kc = Tensor::randn(
        0f32,
        1.,
        (blocks, KV_HEADS, HEAD / X, BLOCK, X),
        &Device::Cpu,
    )?;
    let vc = Tensor::randn(0f32, 1., (blocks, KV_HEADS, HEAD, BLOCK), &Device::Cpu)?;
    // sequence s takes blocks s, s + 2, ..., in reverse
    let tables: Vec<u32> = (0..2u32)
        .flat_map(|s| (0..MAX_BLOCKS as u32).rev().map(move |b| 2 * b + s))
        .collect();
    let tables = Tensor::from_vec(tables, (2, MAX_BLOCKS), &Device::Cpu)?;
    let lens = Tensor::new(&LENS, &Device::Cpu)?;
    let q = Tensor::randn(0f32, 1., (2, HEADS, HEAD), &Device::Cpu)?;
    let scale = 1.0 / (HEAD as f32).sqrt();
    let max_context = LENS[0] as usize;
    let want = paged_attention(
        &q,
        None,
        None,
        &kc,
        &vc,
        &tables,
        &lens,
        None,
        max_context,
        scale,
        NO_SOFTCAP,
        None,
    )?;
    let on = |t: &Tensor| t.to_device(&vk);
    let got = paged_attention(
        &on(&q)?,
        None,
        None,
        &on(&kc)?,
        &on(&vc)?,
        &on(&tables)?,
        &on(&lens)?,
        None,
        max_context,
        scale,
        NO_SOFTCAP,
        None,
    )?;
    for (g, w) in got
        .flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .zip(want.flatten_all()?.to_vec1::<f32>()?)
    {
        assert!((g - w).abs() <= 1e-4 * w.abs().max(1.0), "{g} vs {w}");
    }
    Ok(())
}

// a prompt chunk's rows against a long cached prefix on Vulkan (f32 and f16 caches), each row causal over its own
// length, against the CPU kernel scoring each row as its own query: Nemotron-H's shape, a context over several of the kernel's splits ending
// partway through a block, rows off its workgroup's 8, a shuffled block table
#[test]
fn vulkan_paged_prefill_matches_each_row_scored_alone() -> Result<()> {
    // the f32 kernel; its matrix-core twin is the next test
    // SAFETY: nextest runs each test in its own process, and nothing has read the environment yet
    unsafe { std::env::set_var(inference_tensor::vulkan::NO_MATRIX_CORES_ENV, "1") };
    paged_prefill_rows_score_alone(VULKAN_PREFILL_TOLERANCE)
}

// on matrix cores where the device has them: Q, K, V and P in f16, as CUDA's flash kernels take them
#[test]
fn vulkan_matrix_core_paged_prefill_matches_each_row_scored_alone() -> Result<()> {
    paged_prefill_rows_score_alone(VULKAN_MATRIX_CORE_PREFILL_TOLERANCE)
}

fn paged_prefill_rows_score_alone(tolerance: f32) -> Result<()> {
    let Ok(vk) = Device::new_vulkan(0) else {
        return Ok(());
    };
    const BLOCK: usize = 16;
    const KV_HEADS: usize = 2;
    const HEADS: usize = 32;
    const HEAD: usize = 128;
    const CONTEXT: usize = 2500;
    const ROWS: usize = 70;
    let max_blocks = CONTEXT.div_ceil(BLOCK);
    let blocks = max_blocks + 3;
    // f32 caches hold 4 dimensions a key row, f16 ones 8 (16 bytes either way)
    for (dtype, x) in [(DType::F32, 4usize), (DType::F16, 8)] {
        let kc = Tensor::randn(
            0f32,
            1.,
            (blocks, KV_HEADS, HEAD / x, BLOCK, x),
            &Device::Cpu,
        )?
        .to_dtype(dtype)?;
        let vc = Tensor::randn(0f32, 1., (blocks, KV_HEADS, HEAD, BLOCK), &Device::Cpu)?
            .to_dtype(dtype)?;
        let table: Vec<u32> = (0..max_blocks as u32)
            .map(|b| (b * 7 + 3) % blocks as u32)
            .rev()
            .collect();
        let mut seen = table.clone();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), max_blocks, "a block a position range");
        let q = Tensor::randn(0f32, 1., (ROWS, HEADS, HEAD), &Device::Cpu)?;
        let scale = 1.0 / (HEAD as f32).sqrt();
        let tables = Tensor::from_vec(table.repeat(ROWS), (ROWS, max_blocks), &Device::Cpu)?;
        let lens: Vec<u32> = (0..ROWS).map(|r| (CONTEXT - ROWS + r + 1) as u32).collect();
        let want = paged_attention(
            &q,
            None,
            None,
            &kc,
            &vc,
            &tables,
            &Tensor::new(lens.as_slice(), &Device::Cpu)?,
            None,
            CONTEXT,
            scale,
            NO_SOFTCAP,
            None,
        )?;
        let on = |t: &Tensor| t.to_device(&vk);
        let got =
            inference_tensor::vulkan::paged_prefill(&inference_tensor::vulkan::PagedPrefill {
                q: &on(&q)?,
                key_cache: &on(&kc)?,
                value_cache: &on(&vc)?,
                block_table: &Tensor::new(table.as_slice(), &vk)?,
                context: CONTEXT,
                scale,
            })?;
        for (g, w) in got
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .zip(want.flatten_all()?.to_vec1::<f32>()?)
        {
            assert!(
                (g - w).abs() <= tolerance * w.abs().max(1.0),
                "{dtype:?}: {g} vs {w}"
            );
        }
    }
    Ok(())
}

// kv_quant's round trip on the host: each 32-group rotated, quantized at `bits` against an f16 scale, and rotated back
fn fake_quantize(rows: &[f32], bits: u32) -> Vec<f32> {
    fn hadamard(v: &mut [f32]) {
        let mut s = 1;
        while s < v.len() {
            for i in 0..v.len() {
                if i & s == 0 {
                    let (a, b) = (v[i], v[i | s]);
                    v[i] = a + b;
                    v[i | s] = a - b;
                }
            }
            s <<= 1;
        }
        let norm = 1.0 / (v.len() as f32).sqrt();
        v.iter_mut().for_each(|x| *x *= norm);
    }
    let mid = (1u32 << (bits - 1)) as f32;
    let qmax = ((1u32 << bits) - 1) as f32;
    let mut out = rows.to_vec();
    for group in out.chunks_mut(32) {
        hadamard(group);
        let scale = group.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-10);
        let stored = half::f16::from_f32(scale).to_f32();
        for x in group.iter_mut() {
            let q = (*x / scale).mul_add(mid, mid).floor().clamp(0.0, qmax);
            *x = (q + 0.5 - mid) / mid * stored;
        }
        hadamard(group);
    }
    out
}

#[test]
fn vulkan_quantized_cache_scores_as_its_dequantized_rows() -> Result<()> {
    let Ok(vk) = Device::new_vulkan(0) else {
        return Ok(());
    };
    const BLOCK: usize = 16;
    const KV_HEADS: usize = 2;
    const HEADS: usize = 16;
    const HEAD: usize = 128;
    const CONTEXT: usize = 1500;
    const ROWS: usize = 40;
    let max_blocks = CONTEXT.div_ceil(BLOCK);
    let blocks = max_blocks + 3;
    let table: Vec<u32> = (0..max_blocks as u32)
        .map(|b| (b * 7 + 3) % blocks as u32)
        .collect();
    let slots: Vec<i64> = (0..CONTEXT)
        .map(|t| (table[t / BLOCK] as usize * BLOCK + t % BLOCK) as i64)
        .collect();
    let k = Tensor::randn(0f32, 1., (CONTEXT, KV_HEADS, HEAD), &Device::Cpu)?;
    let v = Tensor::randn(0f32, 1., (CONTEXT, KV_HEADS, HEAD), &Device::Cpu)?;
    let q = Tensor::randn(0f32, 1., (ROWS, HEADS, HEAD), &Device::Cpu)?;
    let scale = 1.0 / (HEAD as f32).sqrt();
    let tables = Tensor::from_vec(table.repeat(ROWS), (ROWS, max_blocks), &Device::Cpu)?;
    let lens: Vec<u32> = (0..ROWS).map(|r| (CONTEXT - ROWS + r + 1) as u32).collect();
    let lens = Tensor::new(lens.as_slice(), &Device::Cpu)?;
    // odd widths cross word boundaries, and K and V may differ
    for (k_bits, v_bits) in [(8u32, 8u32), (4, 4), (3, 5)] {
        let fake = |t: &Tensor, bits| -> Result<Tensor> {
            Tensor::from_vec(
                fake_quantize(&t.flatten_all()?.to_vec1::<f32>()?, bits),
                t.shape(),
                &Device::Cpu,
            )
        };
        let (kc, vc) = (
            Tensor::zeros(
                (blocks, KV_HEADS, HEAD / 4, BLOCK, 4),
                DType::F32,
                &Device::Cpu,
            )?,
            Tensor::zeros((blocks, KV_HEADS, HEAD, BLOCK), DType::F32, &Device::Cpu)?,
        );
        let cpu_slots = Tensor::new(slots.as_slice(), &Device::Cpu)?;
        reshape_and_cache(
            &fake(&k, k_bits)?,
            &fake(&v, v_bits)?,
            None,
            None,
            &kc,
            &vc,
            &cpu_slots,
        )?;
        let want = paged_attention(
            &q, None, None, &kc, &vc, &tables, &lens, None, CONTEXT, scale, NO_SOFTCAP, None,
        )?;

        let rows = |bits| {
            Tensor::zeros(
                (
                    blocks,
                    KV_HEADS,
                    BLOCK,
                    crate::kv_quant::row_bytes(HEAD, bits),
                ),
                DType::U8,
                &vk,
            )
        };
        let (qk, qv) = (rows(k_bits)?, rows(v_bits)?);
        let on = |t: &Tensor| t.to_device(&vk);
        reshape_and_cache(&on(&k)?, &on(&v)?, None, None, &qk, &qv, &on(&cpu_slots)?)?;
        let prefill =
            inference_tensor::vulkan::paged_prefill(&inference_tensor::vulkan::PagedPrefill {
                q: &on(&q)?,
                key_cache: &qk,
                value_cache: &qv,
                block_table: &Tensor::new(table.as_slice(), &vk)?,
                context: CONTEXT,
                scale,
            })?;
        let decode = paged_attention(
            &on(&q)?,
            None,
            None,
            &qk,
            &qv,
            &on(&tables)?,
            &on(&lens)?,
            None,
            CONTEXT,
            scale,
            NO_SOFTCAP,
            None,
        )?;
        for (name, got) in [("prefill", prefill), ("decode", decode)] {
            let diff = max_diff(&got.to_device(&Device::Cpu)?, &want)?;
            assert!(diff <= 1e-3, "q{k_bits}/q{v_bits} {name}: max diff {diff}");
        }
    }
    Ok(())
}
