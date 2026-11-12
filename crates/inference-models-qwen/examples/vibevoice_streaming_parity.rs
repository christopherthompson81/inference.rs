//! Replays `scripts/vibevoice_asr_parity.sh --streaming`'s reference: each window's greedy token ids from Microsoft's
//! `streaming_generate`, window by window over one 24 kHz clip, then the turns the texts cut into.
//! `--pass-positions <n>` instead transcribes the clip in passes of that many cache positions and prints the turns.
//! Usage: vibevoice_streaming_parity <checkpoint dir> <24 kHz clip.wav> <ref.safetensors> [--cuda] [--context <text>]

use inference_audio::AudioInput;
use inference_models_qwen::vibevoice_asr::streaming::{
    StreamingConfig, StreamingFrames, VibeVoiceStreaming, WindowText, transcript,
};
use inference_quant::ShardedSafeTensors;
use inference_tensor::{DType, Device};

// the reference's ids, windows separated by this
const SEPARATOR: i64 = -1;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [dir, wav, dump, ..] = args.as_slice() else {
        anyhow::bail!(
            "usage: vibevoice_streaming_parity <checkpoint dir> <clip.wav> <ref.safetensors> [--cuda]"
        );
    };
    let cuda = args.iter().any(|a| a == "--cuda");
    let context = args
        .iter()
        .position(|a| a == "--context")
        .and_then(|i| args.get(i + 1));
    let (device, dtype) = if cuda {
        (
            Device::cuda_if_available(0)?,
            if args.iter().any(|a| a == "--f16") {
                DType::F16
            } else {
                DType::BF16
            },
        )
    } else {
        (Device::Cpu, DType::F32)
    };
    let dir = std::path::Path::new(dir);
    let read = |name: &str| std::fs::read_to_string(dir.join(name));
    let config: StreamingConfig = serde_json::from_str(&read("config.json")?)?;
    let frames: StreamingFrames = serde_json::from_str(&read("preprocessor_config.json")?)?;
    let tokenizer =
        tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
    let shards: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
        .collect();
    // SAFETY: the weights are read-only for the run
    let vb = unsafe {
        ShardedSafeTensors::sharded(&shards, dtype, &device, None, std::sync::Arc::new(|_| true))?
    };
    let model = VibeVoiceStreaming::new(&config, &frames, tokenizer, vb)?;
    let reference: Vec<i64> = inference_tensor::safetensors::load(dump, &Device::Cpu)?
        .get("ids")
        .ok_or_else(|| anyhow::anyhow!("dump has no `ids`"))?
        .to_vec1()?;
    let reference: Vec<Vec<u32>> = reference
        .split(|&t| t == SEPARATOR)
        .map(|w| w.iter().map(|&t| t as u32).collect())
        .collect();
    let pcm = AudioInput::read_wav(wav)?.to_mono();
    if let Some(parts) = args
        .iter()
        .position(|a| a == "--parts")
        .and_then(|i| args.get(i + 1))
    {
        let parts = inference_tensor::safetensors::load(parts, &Device::Cpu)?;
        let prompt: Vec<i64> = parts["prompt"].to_vec1()?;
        let ours: Vec<i64> = model.prompt_ids().iter().map(|&t| i64::from(t)).collect();
        println!(
            "prompt: {} ids ours, {} reference, equal {}",
            ours.len(),
            prompt.len(),
            ours == prompt
        );
        let (window, _) = model.window_shape();
        let embedded: Vec<f32> = model
            .embed_window(&pcm[..window])?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        let reference: Vec<f32> = parts["window0"].flatten_all()?.to_vec1()?;
        let dot: f64 = embedded
            .iter()
            .zip(&reference)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum();
        let norm = |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
        println!(
            "window 0 embeddings: {} vs {} values, cosine {:.6}; ours {:?}",
            embedded.len(),
            reference.len(),
            dot / (norm(&embedded) * norm(&reference)),
            &embedded[..5]
        );
        return Ok(());
    }
    if let Some(positions) = args
        .iter()
        .position(|a| a == "--pass-positions")
        .and_then(|i| args.get(i + 1))
    {
        let model = model.with_pass_positions(positions.parse()?);
        let start = std::time::Instant::now();
        let out = model.transcribe(&pcm, context.map(String::as_str))?;
        println!(
            "{} turns in {:.2} s",
            out.segments.len(),
            start.elapsed().as_secs_f64()
        );
        for segment in out.segments {
            println!(
                "  {:6.2}-{:6.2} {:?}: {}",
                segment.start, segment.end, segment.speaker, segment.text
            );
        }
        return Ok(());
    }
    let (window, chunk) = model.window_shape();
    let start = std::time::Instant::now();
    let mut session = model.start(context.map(String::as_str))?;
    let mut failed = false;
    let mut windows = Vec::new();
    let mut at = 0;
    let mut index = 0;
    while at < pcm.len() {
        let end = (at + window).min(pcm.len());
        let (ids, text) = model.window(&mut session, &pcm[at..end])?;
        let expected = reference.get(index).cloned().unwrap_or_default();
        let equal = ids
            .iter()
            .zip(&expected)
            .take_while(|(a, b)| a == b)
            .count();
        println!(
            "window {index}: {} ids ours, {} reference, first {equal} equal: {text:?}",
            ids.len(),
            expected.len()
        );
        failed |= ids != expected;
        windows.push(WindowText {
            start: at as f64 / 24_000.0,
            end: (at + chunk).min(pcm.len()) as f64 / 24_000.0,
            text,
            pass: 0,
        });
        at += chunk;
        index += 1;
    }
    println!("{index} windows in {:.2} s", start.elapsed().as_secs_f64());
    for segment in transcript(&windows, pcm.len() as f64 / 24_000.0).segments {
        println!(
            "  {:6.2}-{:6.2} {:?}: {}",
            segment.start, segment.end, segment.speaker, segment.text
        );
    }
    if failed {
        anyhow::bail!("VibeVoice-ASR Streaming parity failed");
    }
    Ok(())
}
