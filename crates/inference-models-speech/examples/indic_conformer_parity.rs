//! Replays `scripts/indic_conformer_parity.sh`'s ONNX dumps for one 16 kHz clip: the features, the encoder output and
//! the greedy RNN-T token ids in the clip's language.

use std::path::PathBuf;

use anyhow::{Result, bail};
use clap::Parser;
use inference_models_speech::TranscriptionOptions;
use inference_models_speech::parakeet::Parakeet;
use inference_tensor::{DType, Device};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    nemo: PathBuf,
    #[arg(long)]
    language: String,
    /// A 16 kHz mono clip.
    #[arg(long)]
    wav: PathBuf,
    /// The reference's `ref.safetensors`.
    #[arg(long)]
    dump: PathBuf,
    #[arg(long, default_value_t = 0.9999)]
    min_cosine: f64,
    #[arg(long)]
    cuda: bool,
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

fn main() -> Result<()> {
    let args = Args::parse();
    let device = if args.cuda {
        Device::cuda_if_available(0)?
    } else {
        Device::Cpu
    };
    let model = Parakeet::load_nemo(&args.nemo, &device, DType::F32)?;
    let mut reader = hound::WavReader::open(&args.wav)?;
    let pcm: Vec<f32> = reader.samples::<f32>().collect::<Result<_, _>>()?;
    let dump = inference_tensor::safetensors::load(&args.dump, &Device::Cpu)?;
    let get = |name: &str| -> Result<inference_tensor::Tensor> {
        Ok(dump
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("dump has no `{name}`"))?
            .clone())
    };
    let mut failed = false;

    let (features, frames) = model.features(&pcm)?;
    let reference: Vec<f32> = get("features")?.flatten_all()?.to_vec1()?;
    let n = features.len().min(reference.len());
    let diff = features[..n]
        .iter()
        .zip(&reference[..n])
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!(
        "features: {frames} frames ours, {} reference, max diff {diff:.2e}",
        get("features")?.dim(0)?
    );

    let encoded = model.encode(&pcm, Some(&args.language))?;
    let ours: Vec<f32> = encoded.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let reference = get("encoded")?;
    let valid = reference.narrow(0, 0, encoded.dim(1)?.min(reference.dim(0)?))?;
    let cos = cosine(&ours, &valid.flatten_all()?.to_vec1()?);
    println!(
        "encoder: {} frames ours, {} reference, cosine {cos:.6}",
        encoded.dim(1)?,
        reference.dim(0)?
    );
    failed |= cos < args.min_cosine;

    let joint = model
        .languages()
        .iter()
        .position(|l| *l == args.language)
        .ok_or_else(|| anyhow::anyhow!("not a language of this model"))?;
    let ids: Vec<u32> = model
        .emissions_in(&encoded, None, joint)?
        .iter()
        .map(|e| e.token)
        .collect();
    let reference: Vec<u32> = get("rnnt")?
        .to_vec1::<i64>()?
        .into_iter()
        .map(|t| t as u32)
        .collect();
    let equal = ids
        .iter()
        .zip(&reference)
        .take_while(|(a, b)| a == b)
        .count();
    println!(
        "rnnt: {} ids ours, {} reference, first {equal} equal",
        ids.len(),
        reference.len()
    );
    failed |= ids != reference;

    let options = TranscriptionOptions {
        language: Some(args.language.clone()),
        ..TranscriptionOptions::default()
    };
    let start = std::time::Instant::now();
    let transcript = model.transcribe(&pcm, 16_000, &options)?;
    println!(
        "ours ({:.2} s): {}",
        start.elapsed().as_secs_f64(),
        transcript.text
    );
    if failed {
        bail!("IndicConformer parity failed");
    }
    Ok(())
}
