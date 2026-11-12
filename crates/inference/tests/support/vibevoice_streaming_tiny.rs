//! Builds a tiny random-weight VibeVoice-ASR Streaming checkpoint at test time, in Microsoft's layout: windows of 25
//! frames and 5 of lookahead at an 80-sample hop, so a fraction of a second crosses several.
#![allow(dead_code)]

use inference_models_qwen::vibevoice_asr::streaming::{
    StreamingConfig, StreamingFrames, VibeVoiceStreaming,
};

#[path = "recording.rs"]
mod recording;

pub const TINY: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../inference/tests/fixtures/vibevoice_asr/streaming_tiny"
);
/// The samples one window advances by.
pub const CHUNK_SAMPLES: usize = 25 * 80;

fn parts(dir: &str) -> anyhow::Result<(StreamingConfig, StreamingFrames, tokenizers::Tokenizer)> {
    let read = |name: &str| std::fs::read_to_string(format!("{dir}/{name}"));
    Ok((
        serde_json::from_str(&read("config.json")?)?,
        serde_json::from_str(&read("preprocessor_config.json")?)?,
        tokenizers::Tokenizer::from_file(format!("{dir}/tokenizer.json"))
            .map_err(anyhow::Error::msg)?,
    ))
}

/// The committed tiny configs and tokenizer, and random weights under the upstream names.
pub fn tiny_checkpoint() -> anyhow::Result<tempfile::TempDir> {
    let (config, frames, tokenizer) = parts(TINY)?;
    let files = recording::fixture_files(TINY)?;
    let files = files.iter().map(|path| path.as_path()).collect::<Vec<_>>();
    recording::record_checkpoint_seeded_by_name(&files, 2, &[], |vb, _| {
        VibeVoiceStreaming::new(&config, &frames, tokenizer, vb).map(|_| ())
    })
}

/// The model of a recorded checkpoint, f32 on the CPU.
pub fn load(dir: &std::path::Path) -> anyhow::Result<VibeVoiceStreaming> {
    let (config, frames, tokenizer) = parts(&dir.to_string_lossy())?;
    let shards: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
        .collect();
    // SAFETY: the test's own files, read-only
    let vb = unsafe {
        inference_quant::ShardedSafeTensors::sharded(
            &shards,
            inference_tensor::DType::F32,
            &inference_tensor::Device::Cpu,
            None,
            std::sync::Arc::new(|_| true),
        )?
    };
    Ok(VibeVoiceStreaming::new(&config, &frames, tokenizer, vb)?)
}
