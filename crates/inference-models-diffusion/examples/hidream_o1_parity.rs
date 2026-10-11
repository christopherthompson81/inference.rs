//! Replays `scripts/hidream_o1_parity.sh`'s dumps: per-forward cosine from the reference's inputs, then a pixel diff.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use inference_models_diffusion::hidream::model::PATCH_SIZE;
use inference_models_diffusion::hidream::sampling::NUM_TRAIN_TIMESTEPS;
use inference_models_diffusion::hidream::{
    HiDreamO1Pipeline, Noise, UNCONDITIONAL_PROMPT, Variant, patches_to_image,
};
use inference_tensor::{D, DType, Device, Tensor};

// how much further from the higher-precision run than the reference a forward may land
const AGAINST_SLACK: f64 = 1e-3;
// the reference's flow times are F32 renderings of the same schedule
const FLOW_TIME_TOLERANCE: f32 = 1e-6;

#[derive(Parser)]
struct Args {
    /// Directory with `config.json`, `tokenizer.json` and the `model*.safetensors` shards.
    #[arg(long)]
    model: PathBuf,
    /// Our weights from this GGUF instead, the config and tokenizer still from `--model`.
    #[arg(long)]
    gguf: Option<PathBuf>,
    /// How much further from the F32 run than the reference a forward may land.
    #[arg(long, default_value_t = AGAINST_SLACK)]
    slack: f64,
    /// `reference.safetensors` and `reference.json` from the reference.
    #[arg(long)]
    dumps: PathBuf,
    #[arg(long, default_value_t = 0.999)]
    min_cosine: f64,
    /// `f32.<forward>` x_pred of an F32 run: those forwards must sit at least as close to it as the BF16 reference
    /// does, in place of `--min-cosine`, which still holds every other forward.
    #[arg(long)]
    against: Option<PathBuf>,
}

/// The reference's noise: its initial draw, then the draw of each Dev step.
struct ReplayNoise<'a> {
    tensors: &'a std::collections::HashMap<String, Tensor>,
    drawn: Option<usize>,
}

impl Noise for ReplayNoise<'_> {
    fn sample(&mut self, _patches: usize, device: &Device) -> inference_tensor::Result<Tensor> {
        let name = match self.drawn {
            None => "noise.initial".to_string(),
            Some(i) => format!("noise.{i}"),
        };
        self.drawn = Some(self.drawn.map_or(0, |i| i + 1));
        self.tensors
            .get(&name)
            .ok_or_else(|| inference_tensor::Error::msg(format!("the dumps have no {name}")))?
            .unsqueeze(0)?
            .to_device(device)
    }
}

fn cosine(a: &Tensor, b: &Tensor) -> Result<f64> {
    let (a, b) = (
        a.flatten_all()?.to_dtype(DType::F64)?,
        b.flatten_all()?.to_dtype(DType::F64)?,
    );
    let dot = (&a * &b)?.sum_all()?.to_scalar::<f64>()?;
    let norm = |v: &Tensor| -> Result<f64> { Ok(v.sqr()?.sum_all()?.to_scalar::<f64>()?.sqrt()) };
    Ok(dot / (norm(&a)? * norm(&b)?))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = Device::cuda_if_available(0)?;
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(args.dumps.join("reference.json"))?)?;
    let tensors = inference_tensor::safetensors::load(
        args.dumps.join("reference.safetensors"),
        &Device::Cpu,
    )?;
    let get = |name: &str| {
        tensors
            .get(name)
            .with_context(|| format!("the dumps have no {name}"))
    };
    let variant = match meta["variant"].as_str() {
        Some("dev") => Variant::Dev,
        _ => Variant::Full,
    };
    let prompt = meta["prompt"].as_str().context("no prompt")?;
    let (width, height) = (
        meta["width"].as_u64().context("no width")? as usize,
        meta["height"].as_u64().context("no height")? as usize,
    );
    let guidance = meta["guidance"].as_bool().unwrap_or(false);
    let calls = meta["calls"].as_array().context("no calls")?;

    let start = std::time::Instant::now();
    let pipeline = match &args.gguf {
        Some(gguf) => {
            HiDreamO1Pipeline::load_gguf(gguf, &args.model, variant, &device, DType::BF16)?
        }
        None => HiDreamO1Pipeline::load(&args.model, variant, &device, DType::BF16)?,
    };
    println!("loaded in {:.1}s", start.elapsed().as_secs_f64());
    let model = pipeline.model();
    let (rows, cols) = (height / PATCH_SIZE, width / PATCH_SIZE);
    let cond = model.sequence(&pipeline.prompt_tokens(prompt)?, rows, cols)?;
    let uncond = guidance
        .then(|| model.sequence(&pipeline.prompt_tokens(UNCONDITIONAL_PROMPT)?, rows, cols))
        .transpose()?;
    let steps = match variant {
        Variant::Full => Some(meta["steps"].as_u64().context("no steps")? as usize),
        Variant::Dev => None,
    };
    let settings = variant.settings(steps, None)?;

    let against = match &args.against {
        Some(path) => inference_tensor::safetensors::load(path, &Device::Cpu)?,
        None => Default::default(),
    };
    let mut failed = false;
    let mut cosines = Vec::with_capacity(calls.len());
    let (mut gated, mut gated_worst) = (f64::INFINITY, 0);
    let per_step = if guidance { 2 } else { 1 };
    for (k, call) in calls.iter().enumerate() {
        let t = call["timestep"].as_f64().context("no timestep")? as f32;
        let step = k / per_step;
        let want_t = 1.0 - settings.schedule.timesteps[step] / NUM_TRAIN_TIMESTEPS as f32;
        if (t - want_t).abs() > FLOW_TIME_TOLERANCE {
            println!("FAIL step {step}: the reference's flow time {t} is not ours {want_t}");
            failed = true;
        }
        let seq = match &uncond {
            Some(uncond) if k % 2 == 1 => uncond,
            _ => &cond,
        };
        let z = get(&format!("z_in.{k}"))?
            .unsqueeze(0)?
            .to_device(&device)?
            .to_dtype(DType::BF16)?;
        let ours = model
            .forward(seq, &z, &model.timestep_embeddings(&[t])?)?
            .squeeze(0)?
            .to_device(&Device::Cpu)?;
        let want = get(&format!("x_pred.{k}"))?;
        let cos = cosine(&ours, want)?;
        cosines.push(cos);
        match against.get(&format!("f32.{k}")) {
            Some(truth) => {
                let (ours_cos, reference_cos) = (cosine(&ours, truth)?, cosine(want, truth)?);
                // the reference's own BF16 noise sets the bar: ours must sit at least as close to the truth
                let closer = ours_cos >= reference_cos - args.slack;
                failed |= !closer;
                println!(
                    "{} forward {k}: ours vs f32 {ours_cos:.6}, reference vs f32 {reference_cos:.6}",
                    if closer { "ok  " } else { "FAIL" }
                );
            }
            None if cos < gated => (gated, gated_worst) = (cos, k),
            None => {}
        }
    }
    let min = cosines.iter().copied().fold(f64::INFINITY, f64::min);
    let mean = cosines.iter().sum::<f64>() / cosines.len().max(1) as f64;
    let worst = cosines
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(k, _)| k);
    let ok = gated >= args.min_cosine;
    failed |= !ok;
    println!(
        "{} teacher forced: {} forwards, x_pred cos min {min:.6} (forward {worst}) mean {mean:.6}; without an F32 run, min \
         {gated:.6} (forward {gated_worst})",
        if ok { "ok  " } else { "FAIL" },
        cosines.len()
    );

    let mut noise = ReplayNoise {
        tensors: &tensors,
        drawn: None,
    };
    let mut drift = Vec::new();
    let start = std::time::Instant::now();
    let z = pipeline.generate(
        prompt,
        (width, height),
        &settings,
        &mut noise,
        &mut |i, x_pred| {
            let want = get(&format!("x_pred.{}", i * per_step)).ok();
            let got = x_pred.squeeze(0).and_then(|x| x.to_device(&Device::Cpu));
            if let (Some(want), Ok(got)) = (want, got) {
                drift.push(cosine(&got, want).unwrap_or(f64::NAN));
            }
        },
    )?;
    let elapsed = start.elapsed().as_secs_f64();
    let ours = patches_to_image(&z, width, height)?
        .squeeze(0)?
        .permute((1, 2, 0))?
        .to_device(&Device::Cpu)?;
    let want = get("image")?.to_dtype(DType::F32)?;
    let diff = (&ours - &want)?.abs()?;
    let mean_abs = diff.mean_all()?.to_scalar::<f32>()?;
    let mse = diff.sqr()?.mean_all()?.to_scalar::<f32>()?;
    let psnr = 10.0 * (255.0f32 * 255.0 / mse).log10();
    let max = diff
        .max_keepdim(D::Minus1)?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    println!(
        "sampled {width}x{height} in {elapsed:.1}s: mean |diff| {mean_abs:.3}, max {max}, PSNR {psnr:.2} dB; x_pred cos by step {}",
        drift
            .iter()
            .map(|c| format!("{c:.4}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    let pixels: Vec<u8> = ours
        .flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .map(|v| v as u8)
        .collect();
    let mut ppm = format!("P6\n{width} {height}\n255\n").into_bytes();
    ppm.extend(pixels);
    std::fs::write(args.dumps.join("ours.ppm"), ppm)?;
    println!("ours: {}", args.dumps.join("ours.ppm").display());
    anyhow::ensure!(!failed, "parity failed");
    Ok(())
}
