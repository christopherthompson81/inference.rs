//! Replays `scripts/cohere_asr_parity.sh`'s transformers dumps for one 16 kHz clip: the first chunk's features and
//! encoder output, and every chunk's greedy token ids.

use std::path::PathBuf;

use anyhow::{Result, bail};
use clap::Parser;
use inference_models_speech::TranscriptionOptions;
use inference_models_speech::cohere_asr::{CohereAsr, files_in};
use inference_tensor::{DType, Device};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
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
    cpu: bool,
    /// f32, bf16 or f16.
    #[arg(long, default_value = "f32")]
    dtype: String,
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
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };
    let dtype = match args.dtype.as_str() {
        "bf16" => DType::BF16,
        "f16" => DType::F16,
        _ => DType::F32,
    };
    let model = CohereAsr::load(&files_in(&args.model), &device, dtype)?;
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
    let spans = model.chunks(&pcm);
    let (first_start, first_end) = spans[0];
    let first = &pcm[first_start..first_end];

    let (features, frames) = model.features(first)?;
    let reference = get("features")?;
    let reference_values: Vec<f32> = reference.flatten_all()?.to_vec1()?;
    let diff = features
        .iter()
        .zip(&reference_values)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!(
        "features: {frames} vs {} frames, max diff {diff:.2e}",
        reference.dim(0)?
    );
    failed |= frames != reference.dim(0)?;

    let encoded: Vec<f32> = model
        .encode(first)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1()?;
    // the reference pads features by a frame (and a batch to its longest chunk), which its mask leaves out
    let reference_encoded = get("encoded")?;
    let hidden = reference_encoded.dim(1)?;
    let ours_frames = encoded.len() / hidden;
    let valid = reference_encoded.narrow(0, 0, ours_frames.min(reference_encoded.dim(0)?))?;
    let cos = cosine(&encoded, &valid.flatten_all()?.to_vec1()?);
    println!(
        "encoder: {ours_frames} frames ours, {} reference (padded), cosine {cos:.6} over ours",
        reference_encoded.dim(0)?
    );
    failed |= ours_frames > reference_encoded.dim(0)? || cos < args.min_cosine;

    let options = TranscriptionOptions {
        language: Some(args.language.clone()),
        ..TranscriptionOptions::default()
    };
    let prompt = model.prompt(&args.language)?;
    for (i, (start, end)) in spans.iter().enumerate() {
        let ours = model.decode_chunk(&pcm[*start..*end], &prompt)?;
        let reference: Vec<i64> = get(&format!("ids.{i}"))?.to_vec1()?;
        let reference: Vec<u32> = reference.into_iter().map(|t| t as u32).collect();
        let equal = ours
            .iter()
            .zip(&reference)
            .take_while(|(a, b)| a == b)
            .count();
        println!(
            "chunk {i} ({:.1}-{:.1} s): {} ids ours, {} reference, first {equal} equal",
            *start as f64 / 16000.0,
            *end as f64 / 16000.0,
            ours.len(),
            reference.len()
        );
        failed |= ours != reference;
    }
    let start = std::time::Instant::now();
    let transcript = model.transcribe(&pcm, 16_000, &options)?;
    println!(
        "ours ({:.2} s): {}",
        start.elapsed().as_secs_f64(),
        transcript.text
    );
    if failed {
        bail!("Cohere Transcribe parity failed");
    }
    Ok(())
}
