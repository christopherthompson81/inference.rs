//! Replays `scripts/voxlingua_parity.sh`'s SpeechBrain dumps: the fbank, the embedding's cosine and the posterior's
//! largest difference, with the top language, for one 16 kHz clip.

use std::path::PathBuf;

use anyhow::{Result, bail};
use clap::Parser;
use inference_models_speech::voxlingua::{VoxLingua, VoxLinguaFiles, fbank};
use inference_tensor::Device;

#[derive(Parser)]
struct Args {
    /// The SpeechBrain checkpoint directory.
    #[arg(long)]
    model: PathBuf,
    /// A 16 kHz mono clip.
    #[arg(long)]
    wav: PathBuf,
    /// The reference's `ref.safetensors`.
    #[arg(long)]
    dump: PathBuf,
    #[arg(long, default_value_t = 0.9999)]
    min_cosine: f64,
    #[arg(long, default_value_t = 1e-3)]
    max_probability_diff: f32,
    #[arg(long)]
    cpu: bool,
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    let norm = |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    dot / (norm(a) * norm(b))
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };
    let model = VoxLingua::load(&VoxLinguaFiles::in_dir(&args.model), &device)?;
    let mut reader = hound::WavReader::open(&args.wav)?;
    let rate = reader.spec().sample_rate;
    let pcm: Vec<f32> = reader.samples::<f32>().collect::<Result<_, _>>()?;
    let dump = inference_tensor::safetensors::load(&args.dump, &Device::Cpu)?;
    let get = |name: &str| -> Result<Vec<f32>> {
        Ok(dump
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("dump has no `{name}`"))?
            .flatten_all()?
            .to_vec1()?)
    };
    let mut failed = false;

    let ours: Vec<f32> = fbank(&pcm, 60).into_iter().flatten().collect();
    let reference = get("fbank")?;
    println!(
        "fbank: {} vs {} values, max diff {:.2e}",
        ours.len(),
        reference.len(),
        max_diff(&ours, &reference)
    );
    failed |= ours.len() != reference.len();

    let start = std::time::Instant::now();
    let identified = model.identify(&pcm, rate)?;
    let elapsed = start.elapsed().as_secs_f64();
    let cos = cosine(&identified.embedding, &get("embedding")?);
    println!("embedding: cosine {cos:.6}");
    failed |= cos < args.min_cosine;

    let mut probabilities = vec![0f32; model.languages().len()];
    for scored in &identified.languages {
        let index = model
            .languages()
            .iter()
            .position(|l| *l == scored.language)
            .expect("a known language");
        probabilities[index] = scored.probability;
    }
    let reference = get("probabilities")?;
    let diff = max_diff(&probabilities, &reference);
    let top_reference = reference
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| &model.languages()[i])
        .expect("classes");
    let top = &identified.languages[0];
    println!(
        "posterior: max diff {diff:.2e}; ours {} {:.4}, reference {} ({elapsed:.2} s for {:.1} s)",
        top.language.code, top.probability, top_reference.code, identified.duration
    );
    failed |= diff > args.max_probability_diff || top.language != *top_reference;
    if failed {
        bail!("VoxLingua107 parity failed");
    }
    Ok(())
}
