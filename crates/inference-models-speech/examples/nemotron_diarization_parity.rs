//! Replays `scripts/nemotron_diarization_parity.sh`'s reference dumps: probabilities within `--max-diff` over the
//! valid frames, and how many reference segments ours reproduce to the frame.

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use inference_models_speech::diarization::{
    DEFAULT_THRESHOLD, Nemotron3Diarizer, Nemotron3Files, speaker_segments,
};
use inference_tensor::{DType, Device};

const FRAMES_PER_SECOND: f64 = 100.0;
const FIRST_DIFF_TOLERANCE: f32 = 1e-4;
// encoder frames are this many 10 ms frames
const SUBSAMPLING: usize = 8;
// the speaker cache's top-k keeps whichever of two near-equal frames float noise favours, and later chunks follow
// it: transformers on CPU and on CUDA part this way too, so such clips are judged on the decisions they agree on
const MIN_DECISION_AGREEMENT: f64 = 0.995;

#[derive(Parser)]
struct Args {
    /// Directory with `config.json`, `processor_config.json` and `model.safetensors`.
    #[arg(long)]
    model: PathBuf,
    /// `<n>.safetensors` dumps from the reference.
    #[arg(long)]
    dumps: PathBuf,
    #[arg(long, default_value_t = 1e-3)]
    max_diff: f32,
    #[arg(long)]
    cpu: bool,
    /// Load in BF16: probabilities drift past `--max-diff`, so decisions and segments are what to read.
    #[arg(long)]
    bf16: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };
    let files = Nemotron3Files {
        config: args.model.join("config.json"),
        processor_config: args.model.join("processor_config.json"),
        weights: vec![args.model.join("model.safetensors")],
    };
    let model = Nemotron3Diarizer::load(
        &files,
        &device,
        if args.bf16 { DType::BF16 } else { DType::F32 },
    )?;
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
        let want = get("probs")?.flatten_all()?.to_vec1::<f32>()?;
        let want_segments: Vec<(usize, i64, i64)> = get("segments")?
            .to_vec2::<i64>()?
            .into_iter()
            .map(|r| (r[0] as usize, r[1], r[2]))
            .collect();
        let start = std::time::Instant::now();
        let diarization = model.diarize(&pcm, model.sample_rate(), None)?;
        let elapsed = start.elapsed().as_secs_f64();
        let ours = &diarization.probabilities;
        let s = diarization.num_speakers;
        // transformers keeps the masked trailing feature frame, so its upsampler reads one padding group's hidden
        // state past the audio's end; ours sees zero padding there, so the final group is reported apart
        let tail = (ours.len() / s).saturating_sub(SUBSAMPLING) * s;
        let diff = |range: std::ops::Range<usize>| {
            ours[range.clone()]
                .iter()
                .zip(&want[range])
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max)
        };
        let max_diff = diff(0..tail.min(want.len()));
        let tail_diff = diff(tail.min(want.len())..ours.len().min(want.len()));
        // the first 10 ms frame past a tight tolerance locates a divergence: a chunk's model or the cache between
        let first_diff = ours.chunks(s).zip(want.chunks(s)).position(|(a, b)| {
            a.iter()
                .zip(b)
                .any(|(x, y)| (x - y).abs() > FIRST_DIFF_TOLERANCE)
        });
        let to_frames = |segments: &[inference_models_speech::diarization::SpeakerSegment]| {
            segments
                .iter()
                .map(|g| {
                    (
                        g.speaker,
                        (g.start * FRAMES_PER_SECOND).round() as i64,
                        (g.end * FRAMES_PER_SECOND).round() as i64,
                    )
                })
                .collect::<Vec<_>>()
        };
        // the reference's segments from its own probabilities isolate the cutting from the model
        let cut_reference = to_frames(&speaker_segments(
            &want,
            s,
            diarization.frame_seconds,
            DEFAULT_THRESHOLD,
        ));
        let ours_segments = to_frames(&diarization.segments);
        let matched = ours_segments
            .iter()
            .filter(|g| want_segments.contains(g))
            .count();
        let agreed = ours
            .iter()
            .zip(&want)
            .filter(|(a, b)| (**a > DEFAULT_THRESHOLD) == (**b > DEFAULT_THRESHOLD))
            .count();
        let agreement = agreed as f64 / want.len().max(1) as f64;
        let ok = ours.len() == want.len()
            && (max_diff <= args.max_diff || agreement >= MIN_DECISION_AGREEMENT)
            && cut_reference == want_segments;
        failed |= !ok;
        println!(
            "{} {}: {} frames ({} want), max prob diff {max_diff:.2e} before the final 80 ms ({tail_diff:.2e} within it; first past 1e-4 at frame {}), speaking decisions {:.3}% agree, segments {matched}/{} exact (cut of reference probs {}), {elapsed:.2}s",
            if ok { "ok  " } else { "FAIL" },
            path.file_stem().unwrap_or_default().to_string_lossy(),
            ours.len() / s,
            want.len() / s,
            first_diff.map_or("none".to_string(), |f| f.to_string()),
            agreement * 100.0,
            want_segments.len(),
            if cut_reference == want_segments {
                "matches"
            } else {
                "differs"
            },
        );
    }
    anyhow::ensure!(!failed, "parity failed");
    Ok(())
}
