//! Builds a random-weight Silero VAD GGUF at test time, through the same converter real weights go through.
#![allow(dead_code)]

use inference_models_speech::silero::{SileroVad, VadConfig, write_gguf};
use inference_tensor::{Device, Tensor};

#[path = "recording.rs"]
mod recording;

const SAMPLE_RATE: u32 = 16_000;
const CHUNK_SIZE: usize = 512;
const CONTEXT_SIZE: usize = 64;
const HEAD_BIAS: &str = "final_conv.bias";
// a head bias this large saturates the sigmoid, so every chunk reads as speech
const ALWAYS_SPEECH_BIAS: f32 = 20.0;
const GGUF: &str = "silero_vad.gguf";

/// A random-weight Silero VAD GGUF in a new directory; `always_speech` pins every probability near 1.
pub fn tiny_silero_gguf(always_speech: bool) -> anyhow::Result<tempfile::TempDir> {
    let config = VadConfig {
        sample_rate: SAMPLE_RATE,
        chunk_size: CHUNK_SIZE,
        context_size: CONTEXT_SIZE,
    };
    let dir = recording::record_plain_checkpoint(&[], |vb| SileroVad::new(config, vb).map(|_| ()))?;
    let weights = dir.path().join("model.safetensors");
    if always_speech {
        let mut tensors = inference_tensor::safetensors::load(&weights, &Device::Cpu)?;
        tensors.insert(
            HEAD_BIAS.to_string(),
            Tensor::new(&[ALWAYS_SPEECH_BIAS], &Device::Cpu)?,
        );
        inference_tensor::safetensors::save(&tensors, &weights)?;
    }
    let mut out = std::io::BufWriter::new(std::fs::File::create(dir.path().join(GGUF))?);
    write_gguf(&weights, &mut out)?;
    drop(out);
    std::fs::remove_file(&weights)?;
    Ok(dir)
}
