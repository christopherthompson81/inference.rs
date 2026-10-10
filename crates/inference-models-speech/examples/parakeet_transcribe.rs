//! Transcribes a 16-bit PCM wav with a Parakeet checkpoint, optionally windowed at a Silero VAD's silences.

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use inference_models_speech::parakeet::{Parakeet, ParakeetFiles};
use inference_models_speech::silero::{SileroVad, read_gguf};
use inference_tensor::{DType, Device};

#[derive(Parser)]
struct Args {
    /// Directory with `config.json`, `processor_config.json`, `tokenizer.json` and `model.safetensors`.
    #[arg(long)]
    model: PathBuf,
    /// A Silero VAD GGUF, to window long audio at its silences.
    #[arg(long)]
    vad: Option<PathBuf>,
    #[arg(long)]
    cpu: bool,
    wav: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };
    let files = ParakeetFiles {
        config: args.model.join("config.json"),
        processor_config: args.model.join("processor_config.json"),
        tokenizer: args.model.join("tokenizer.json"),
        weights: vec![args.model.join("model.safetensors")],
    };
    let model = Parakeet::load(&files, &device, DType::F32)?;
    let vad = args
        .vad
        .map(|path| -> Result<SileroVad> {
            let (config, vb) = read_gguf(&mut std::fs::File::open(path)?, &device)?;
            Ok(SileroVad::new(config, vb)?)
        })
        .transpose()?;
    let audio = inference_audio::AudioInput::read_wav(&args.wav.to_string_lossy())?;
    let pcm = audio.to_mono();
    let start = std::time::Instant::now();
    let transcript = match &vad {
        Some(vad) => model.transcribe_with_vad(&pcm, audio.sample_rate, vad)?,
        None => model.transcribe(&pcm, audio.sample_rate)?,
    };
    eprintln!(
        "{:.1}s of audio in {:.2}s, {} words",
        transcript.duration,
        start.elapsed().as_secs_f64(),
        transcript.words.len()
    );
    println!("{}", transcript.text);
    Ok(())
}
