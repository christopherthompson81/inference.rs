//! Time PP-DocLayoutV3 forward passes (model only, preprocessing excluded).

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use clap::Parser;
use inference_layout::pp_doclayout_v3::PPDocLayoutV3Detector;

#[derive(Parser)]
struct Args {
    /// HF `PP-DocLayoutV3_safetensors` directory.
    #[arg(long)]
    model: String,
    #[arg(long, default_value_t = 1)]
    batch: usize,
    #[arg(long, default_value_t = 3)]
    warmup: usize,
    #[arg(long, default_value_t = 10)]
    iters: usize,
    #[arg(long)]
    cpu: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let dev = if args.cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };
    let det = PPDocLayoutV3Detector::load(&args.model, &dev)?;
    let (h, w) = det.model().input_hw();
    let pv = Tensor::rand(0f32, 1., (args.batch, 3, h, w), &dev)?;

    let run = || -> Result<f64> {
        let t0 = std::time::Instant::now();
        let out = det.install(|| det.model().forward(&pv, false))?;
        out.logits.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        Ok(t0.elapsed().as_secs_f64() * 1e3)
    };
    for _ in 0..args.warmup {
        run()?;
    }
    let times = (0..args.iters).map(|_| run()).collect::<Result<Vec<_>>>()?;
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let min = times.iter().copied().fold(f64::MAX, f64::min);
    println!(
        "batch={} mean={mean:.2}ms min={min:.2}ms per_image={:.2}ms",
        args.batch,
        mean / args.batch as f64
    );
    Ok(())
}
