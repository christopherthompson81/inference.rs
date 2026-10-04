//! fattn vs Dao FA2 at model attention shapes: `--features bench-fa2 --run-ignored only -E 'test(/bench::/)'`.

use std::time::Instant;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use inference_fattn::{FattnOptions, causal_mask, flash_attn};

const WARMUP: usize = 5;
const ITERS: usize = 50;

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
const RUNS: [(usize, usize, usize); 8] = [
    (1, 512, 512),
    (2, 512, 512),
    (1, 2048, 2048),
    (2, 2048, 2048),
    (1, 8192, 8192),
    (1, 1, 512),
    (1, 1, 4096),
    (1, 1, 16384),
];

fn time(dev: &Device, f: impl Fn() -> candle_core::Result<Tensor>) -> Result<f64> {
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
fn fattn_vs_fa2() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    for shape in SHAPES {
        for (batch, seq_q, seq_kv) in RUNS {
            let rand = |s: usize, h: usize| -> Result<Tensor> {
                Ok(
                    Tensor::randn(0f32, 1., (batch, s, h, shape.head_dim), &dev)?
                        .to_dtype(DType::BF16)?,
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
                mask: Some(causal_mask(seq_q, seq_kv, &dev)?),
                ..Default::default()
            };
            let fattn_us = time(&dev, || flash_attn(&q, &k, &v, &opts))?;
            let fa2_us = time(&dev, || {
                inference_flash_attn::flash_attn(&q, &k, &v, scale, seq_q > 1)
            })?;
            println!(
                "{:<24} b {batch} q {seq_q:>5} kv {seq_kv:>5}: fattn {fattn_us:9.1} us  fa2 {fa2_us:9.1} us  ratio {:.2}",
                shape.name,
                fattn_us / fa2_us
            );
        }
    }
    Ok(())
}
