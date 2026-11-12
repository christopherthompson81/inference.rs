//! Compares VibeVoice-ASR's audio half against a transformers dump (`scripts/vibevoice_asr_parity.sh`): the
//! normalized, padded clip and the projected embeddings the decoder's placeholders take.
//! Usage: vibevoice_audio_parity <checkpoint dir> <24 kHz clip.wav> <ref.safetensors> [--cuda]

use inference_audio::AudioInput;
use inference_models_qwen::vibevoice_asr::inputs_processor::prepare_clip;
use inference_models_qwen::vibevoice_asr::{AudioEmbedder, VibeVoiceAsrConfig};
use inference_quant::ShardedSafeTensors;
use inference_tensor::{DType, Device, Tensor};

const MIN_COSINE: f64 = 0.9999;

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    let norm = |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    dot / (norm(a) * norm(b))
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [dir, wav, dump, ..] = args.as_slice() else {
        anyhow::bail!(
            "usage: vibevoice_audio_parity <checkpoint dir> <clip.wav> <ref.safetensors> [--cuda]"
        );
    };
    let device = if args.iter().any(|a| a == "--cuda") {
        Device::cuda_if_available(0)?
    } else {
        Device::Cpu
    };
    let dir = std::path::Path::new(dir);
    let cfg = VibeVoiceAsrConfig::parse(&std::fs::read_to_string(dir.join("config.json"))?)?;
    let shards: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
        .collect();
    // SAFETY: the weights are read-only for the run
    let vb = unsafe {
        ShardedSafeTensors::sharded(
            &shards,
            DType::F32,
            &device,
            None,
            std::sync::Arc::new(|_| true),
        )?
    };
    let embedder = AudioEmbedder::new(&cfg, &vb)?;
    let dump = inference_tensor::safetensors::load(dump, &Device::Cpu)?;
    let get = |name: &str| -> anyhow::Result<Vec<f32>> {
        Ok(dump
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("dump has no `{name}`"))?
            .flatten_all()?
            .to_vec1()?)
    };
    let mut failed = false;
    let audio = AudioInput::read_wav(wav)?;
    let ours = prepare_clip(&audio.to_mono(), cfg.hop_length());
    let reference = get("input_values")?;
    let diff = ours
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!(
        "clip: {} vs {} samples, max diff {diff:.2e}",
        ours.len(),
        reference.len()
    );
    failed |= ours.len() != reference.len();

    let start = std::time::Instant::now();
    let len = ours.len();
    let embedded: Vec<f32> = embedder
        .embed(&Tensor::from_vec(ours, (1, 1, len), &device)?)?
        .flatten_all()?
        .to_vec1()?;
    let reference = get("audio")?;
    let cos = cosine(&embedded, &reference);
    println!(
        "embeddings: {} vs {} values, cosine {cos:.6} ({:.2} s)",
        embedded.len(),
        reference.len(),
        start.elapsed().as_secs_f64()
    );
    failed |= embedded.len() != reference.len() || cos < MIN_COSINE;
    if failed {
        anyhow::bail!("VibeVoice-ASR audio parity failed");
    }
    Ok(())
}
