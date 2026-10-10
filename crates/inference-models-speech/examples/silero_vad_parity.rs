//! Replays `scripts/silero_vad_parity.sh`'s reference dumps: probabilities within `--max-diff`, segments exact.

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use inference_models_speech::silero::{SegmentOptions, SileroVad, read_gguf, speech_segments};
use inference_tensor::Device;

#[derive(Parser)]
struct Args {
    /// A Silero VAD GGUF.
    #[arg(long)]
    gguf: PathBuf,
    /// `<n>.safetensors` dumps from the reference.
    #[arg(long)]
    dumps: PathBuf,
    #[arg(long, default_value_t = 1e-4)]
    max_diff: f32,
    /// The reference's `max_speech_duration_s`, as the script passed it.
    #[arg(long)]
    max_speech_s: Option<f64>,
    #[arg(long)]
    cpu: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };
    let (config, vb) = read_gguf(&mut std::fs::File::open(&args.gguf)?, &device)?;
    let model = SileroVad::new(config, vb)?;
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
        let want_probs = get("probs")?.to_vec1::<f32>()?;
        let want_segments: Vec<(usize, usize)> = get("segments")?
            .to_vec2::<i64>()?
            .into_iter()
            .map(|r| (r[0] as usize, r[1] as usize))
            .collect();
        let start = std::time::Instant::now();
        let probs = model.probabilities(&pcm)?;
        let elapsed = start.elapsed().as_secs_f64();
        let cfg = model.config();
        let options = SegmentOptions {
            max_speech_duration_s: args.max_speech_s,
            ..SegmentOptions::default()
        };
        // the reference's segments from its own probabilities isolate the port of the cutting from the model
        let cut_reference = speech_segments(
            &want_probs,
            pcm.len(),
            cfg.sample_rate,
            cfg.chunk_size,
            &options,
        );
        let cut_ours =
            speech_segments(&probs, pcm.len(), cfg.sample_rate, cfg.chunk_size, &options);
        let max_diff = probs
            .iter()
            .zip(&want_probs)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let ok = probs.len() == want_probs.len()
            && max_diff <= args.max_diff
            && cut_reference == want_segments
            && cut_ours == want_segments;
        failed |= !ok;
        println!(
            "{} {}: {} chunks, max prob diff {max_diff:.2e}, segments {} (cut of reference probs {}, of ours {}), {elapsed:.3}s",
            if ok { "ok  " } else { "FAIL" },
            path.file_stem().unwrap_or_default().to_string_lossy(),
            probs.len(),
            want_segments.len(),
            if cut_reference == want_segments {
                "match"
            } else {
                "differ"
            },
            if cut_ours == want_segments {
                "match"
            } else {
                "differ"
            },
        );
    }
    anyhow::ensure!(!failed, "parity failed");
    Ok(())
}
