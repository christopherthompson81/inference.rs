//! fattn at model attention shapes: `--features cuda --run-ignored only -E 'test(/bench::/)'` (one test thread).

use std::time::Instant;

use anyhow::Result;
use inference_fattn::{
    FattnOptions, PagedKv, causal_mask, flash_attn, flash_attn_paged, paged_causal_mask,
    paged_kv_len,
};
use inference_tensor::{DType, Device, Tensor};

const WARMUP: usize = 5;
const ITERS: usize = 50;
// Runs only the rows whose label contains this, e.g. to profile one shape under nsys
const FILTER_ENV: &str = "FATTN_BENCH_FILTER";
// Set to run every row in f16 instead of bf16
const F16_ENV: &str = "FATTN_BENCH_F16";

struct Shape {
    name: &'static str,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
}

const SHAPES: [Shape; 2] = [
    Shape {
        name: "qwen3.5-0.8b full attn",
        n_head: 8,
        n_head_kv: 2,
        head_dim: 256,
    },
    Shape {
        name: "llama-8b",
        n_head: 32,
        n_head_kv: 8,
        head_dim: 128,
    },
];
// (batch, seq_q, seq_kv): prefill without a cache, then decode. fattn skips fully masked KV tiles only for
// seq_q >= 1024 or batch > 1 (fattn-common.cuh), hence the batch-2 prefill rows.
const RUNS: [(usize, usize, usize); 13] = [
    (1, 16, 4096),
    (1, 64, 4096),
    (1, 256, 4096),
    (1, 512, 4096),
    (1, 1024, 4096),
    (1, 512, 512),
    (2, 512, 512),
    (1, 2048, 2048),
    (2, 2048, 2048),
    (1, 8192, 8192),
    (1, 1, 512),
    (1, 1, 4096),
    (1, 1, 16384),
];

fn time(dev: &Device, f: impl Fn() -> inference_tensor::Result<Tensor>) -> Result<f64> {
    for _ in 0..WARMUP {
        f()?;
    }
    dev.synchronize()?;
    let start = Instant::now();
    for _ in 0..ITERS {
        f()?;
    }
    dev.synchronize()?;
    Ok(start.elapsed().as_secs_f64() * 1e6 / ITERS as f64)
}

#[test]
#[ignore]
fn prefill_and_decode() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let filter = std::env::var(FILTER_ENV).unwrap_or_default();
    let dtype = if std::env::var_os(F16_ENV).is_some() {
        DType::F16
    } else {
        DType::BF16
    };
    for shape in SHAPES {
        for (batch, seq_q, seq_kv) in RUNS {
            let label = format!("{:<24} b {batch} q {seq_q:>5} kv {seq_kv:>5}", shape.name);
            if !label.contains(&filter) {
                continue;
            }
            let rand = |s: usize, h: usize| -> Result<Tensor> {
                Ok(
                    Tensor::randn(0f32, 1., (batch, s, h, shape.head_dim), &dev)?
                        .to_dtype(dtype)?,
                )
            };
            let (q, k, v) = (
                rand(seq_q, shape.n_head)?,
                rand(seq_kv, shape.n_head_kv)?,
                rand(seq_kv, shape.n_head_kv)?,
            );
            let scale = 1. / (shape.head_dim as f32).sqrt();
            let opts = FattnOptions {
                scale,
                causal: true,
                ..Default::default()
            };
            let fattn_us = time(&dev, || flash_attn(&q, &k, &v, &opts))?;
            println!("{label}: fattn {fattn_us:9.1} us");
        }
    }
    Ok(())
}

// (batch, seq_q, seq_kv) for paged against dense: decode, a decode batch, and a prefill chunk over a cache
const PAGED_RUNS: [(usize, usize, usize); 8] = [
    (1, 1, 128),
    (1, 1, 2048),
    (1, 1, 4096),
    (1, 1, 16384),
    (8, 1, 4096),
    (8, 1, 16384),
    (1, 256, 4096),
    (1, 1024, 4096),
];
const PAGED_BLOCK_SIZE: usize = 32;

#[test]
#[ignore]
fn paged_vs_dense() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let filter = std::env::var(FILTER_ENV).unwrap_or_default();
    for shape in SHAPES {
        for (batch, seq_q, seq_kv) in PAGED_RUNS {
            let label = format!("{:<24} b {batch} q {seq_q:>5} kv {seq_kv:>5}", shape.name);
            if !label.contains(&filter) {
                continue;
            }
            let d = shape.head_dim;
            let h_kv = shape.n_head_kv;
            let blocks = seq_kv / PAGED_BLOCK_SIZE;
            let rand = |dims: &[usize]| -> Result<Tensor> {
                Ok(Tensor::randn(0f32, 1., dims, &dev)?.to_dtype(DType::BF16)?)
            };
            let q = rand(&[batch, seq_q, shape.n_head, d])?;
            let k_cache = rand(&[batch * blocks, h_kv, PAGED_BLOCK_SIZE, d])?;
            let v_cache = rand(&[batch * blocks, h_kv, PAGED_BLOCK_SIZE, d])?;
            // sequence s owns blocks s, s + batch, s + 2 * batch, ...: interleaved, as a shared pool hands them out
            let table: Vec<u32> = (0..batch)
                .flat_map(|s| (0..blocks).map(move |j| (j * batch + s) as u32))
                .collect();
            let block_table = Tensor::from_vec(table, (batch, blocks), &dev)?;
            let seq_lens = Tensor::from_vec(vec![seq_kv as u32; batch], batch, &dev)?;
            let kv = PagedKv {
                k_cache: &k_cache,
                v_cache: &v_cache,
                block_table: &block_table,
                seq_lens: &seq_lens,
                full_lens: None,
            };
            let lens = vec![seq_kv; batch];
            let scale = 1. / (d as f32).sqrt();
            let paged_opts = FattnOptions {
                scale,
                mask: Some(paged_causal_mask(&lens, seq_q, paged_kv_len(&kv)?, &dev)?),
                ..Default::default()
            };
            let dense = |c: &Tensor| -> Result<Tensor> {
                Ok(c.reshape((blocks, batch, h_kv, PAGED_BLOCK_SIZE, d))?
                    .permute((1, 0, 3, 2, 4))?
                    .reshape((batch, seq_kv, h_kv, d))?
                    .contiguous()?)
            };
            let (k, v) = (dense(&k_cache)?, dense(&v_cache)?);
            let dense_opts = FattnOptions {
                scale,
                mask: Some(paged_causal_mask(&lens, seq_q, seq_kv, &dev)?),
                ..Default::default()
            };
            let paged_us = time(&dev, || flash_attn_paged(&q, &kv, &paged_opts))?;
            let dense_us = time(&dev, || flash_attn(&q, &k, &v, &dense_opts))?;
            println!(
                "{label}: paged {paged_us:9.1} us  dense {dense_us:9.1} us  ratio {:.2}",
                paged_us / dense_us
            );
        }
    }
    Ok(())
}

// (batch, seq_q, seq_kv) for fp8 against bf16 K/V
const FP8_RUNS: [(usize, usize, usize); 5] = [
    (1, 1, 4096),
    (1, 1, 16384),
    (8, 1, 16384),
    (1, 256, 4096),
    (1, 1024, 4096),
];

#[test]
#[ignore]
fn fp8_vs_bf16() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let filter = std::env::var(FILTER_ENV).unwrap_or_default();
    for shape in SHAPES {
        for (batch, seq_q, seq_kv) in FP8_RUNS {
            let label = format!("{:<24} b {batch} q {seq_q:>5} kv {seq_kv:>5}", shape.name);
            if !label.contains(&filter) {
                continue;
            }
            let rand = |s: usize, h: usize| {
                Tensor::randn(0f32, 1., (batch, s, h, shape.head_dim), &dev)?.to_dtype(DType::BF16)
            };
            let q = rand(seq_q, shape.n_head)?;
            let (k, v) = (
                rand(seq_kv, shape.n_head_kv)?,
                rand(seq_kv, shape.n_head_kv)?,
            );
            let fp8 = |t: &Tensor| {
                t.to_device(&Device::Cpu)?
                    .to_dtype(DType::F8E4M3)?
                    .to_device(&dev)
            };
            let (k8, v8) = (fp8(&k)?, fp8(&v)?);
            let opts = FattnOptions {
                scale: 1. / (shape.head_dim as f32).sqrt(),
                mask: Some(causal_mask(seq_q, seq_kv, &dev)?),
                ..Default::default()
            };
            let fp8_opts = FattnOptions {
                kv_scales: Some(Default::default()),
                ..opts.clone()
            };
            let fp8_us = time(&dev, || flash_attn(&q, &k8, &v8, &fp8_opts))?;
            let bf16_us = time(&dev, || flash_attn(&q, &k, &v, &opts))?;
            println!(
                "{label}: fp8 {fp8_us:9.1} us  bf16 {bf16_us:9.1} us  ratio {:.2}",
                fp8_us / bf16_us
            );
        }
    }
    Ok(())
}
