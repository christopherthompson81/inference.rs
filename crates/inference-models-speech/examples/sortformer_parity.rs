//! Replays `scripts/sortformer_parity.sh`'s NeMo dumps: our log-mel features against NeMo's preprocessor, then the
//! streaming speaker probabilities, and how many of the segments cut from NeMo's probabilities ours reproduce.

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use inference_models_speech::diarization::{
    DEFAULT_THRESHOLD, SortformerDiarizer, SpeakerSegment, speaker_segments,
};
use inference_tensor::{DType, Device};

const FIRST_DIFF_TOLERANCE: f32 = 1e-4;
// the cache's top-k keeps whichever of two tied frames float noise favours, and later chunks follow it, so a clip
// that parts that way is judged on the decisions the two agree on
const MIN_DECISION_AGREEMENT: f64 = 0.995;

#[derive(Parser)]
struct Args {
    /// The Sortformer `.nemo`.
    #[arg(long)]
    nemo: PathBuf,
    /// `<n>.safetensors` dumps from the reference.
    #[arg(long)]
    dumps: PathBuf,
    #[arg(long, default_value_t = 1e-3)]
    max_diff: f32,
    /// NeMo's preprocessor runs in f32 and ours in f64; the log of quiet frames' power magnifies the difference.
    #[arg(long, default_value_t = 5e-3)]
    max_mel_diff: f32,
    #[arg(long)]
    cpu: bool,
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };
    let model = SortformerDiarizer::load(&args.nemo, &device, DType::F32)?;
    let mut dumps = std::fs::read_dir(&args.dumps)?
        .map(|e| e.map(|e| e.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    dumps.retain(|p| p.extension().is_some_and(|e| e == "safetensors"));
    dumps.sort();
    let mut failed = false;
    for path in &dumps {
        let t = inference_tensor::safetensors::load(path, &Device::Cpu)?;
        let get = |name: &str| {
            t.get(name)
                .ok_or_else(|| anyhow::anyhow!("{} has no {name}", path.display()))
        };
        let pcm = get("pcm")?.to_vec1::<f32>()?;
        let want_mel = get("mel")?.flatten_all()?.to_vec1::<f32>()?;
        let want = get("probs")?.flatten_all()?.to_vec1::<f32>()?;
        let (mel, _) = model.features(&pcm)?;
        let mel_diff = max_diff(&mel, &want_mel);

        let start = std::time::Instant::now();
        let diarization = model.diarize(&pcm, model.sample_rate(), None)?;
        let elapsed = start.elapsed().as_secs_f64();
        let ours = &diarization.probabilities;
        let s = diarization.num_speakers;
        let prob_diff = max_diff(ours, &want);
        // the first frame past a tight tolerance locates a divergence: a chunk's model or the cache between
        let first_diff = ours.chunks(s).zip(want.chunks(s)).position(|(a, b)| {
            a.iter()
                .zip(b)
                .any(|(x, y)| (x - y).abs() > FIRST_DIFF_TOLERANCE)
        });
        let frames = |segments: &[SpeakerSegment]| {
            segments
                .iter()
                .map(|g| {
                    (
                        g.speaker,
                        (g.start / diarization.frame_seconds).round() as i64,
                        (g.end / diarization.frame_seconds).round() as i64,
                    )
                })
                .collect::<Vec<_>>()
        };
        let want_segments = frames(&speaker_segments(
            &want,
            s,
            diarization.frame_seconds,
            DEFAULT_THRESHOLD,
        ));
        let matched = frames(&diarization.segments)
            .iter()
            .filter(|g| want_segments.contains(g))
            .count();
        let agreed = ours
            .iter()
            .zip(&want)
            .filter(|(a, b)| (**a > DEFAULT_THRESHOLD) == (**b > DEFAULT_THRESHOLD))
            .count();
        let agreement = agreed as f64 / want.len().max(1) as f64;
        let ok = mel.len() == want_mel.len()
            && mel_diff <= args.max_mel_diff
            && ours.len() == want.len()
            && (prob_diff <= args.max_diff || agreement >= MIN_DECISION_AGREEMENT);
        failed |= !ok;
        println!(
            "{} {}: mel {} frames ({} want), max diff {mel_diff:.2e}; {} frames ({} want), max prob diff \
             {prob_diff:.2e} (first past 1e-4 at frame {}), speaking decisions {:.3}% agree, segments {matched}/{} \
             exact, {elapsed:.2}s",
            if ok { "ok  " } else { "FAIL" },
            path.file_stem().unwrap_or_default().to_string_lossy(),
            mel.len() / model.mel_bins(),
            want_mel.len() / model.mel_bins(),
            ours.len() / s,
            want.len() / s,
            first_diff.map_or("none".to_string(), |f| f.to_string()),
            agreement * 100.0,
            want_segments.len(),
        );
    }
    anyhow::ensure!(!failed, "parity failed");
    Ok(())
}
