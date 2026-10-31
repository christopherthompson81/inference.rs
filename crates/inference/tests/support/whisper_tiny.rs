//! Builds a tiny random-weight Whisper checkpoint (safetensors, configs, tokenizer) at test time.
#![allow(dead_code)]

use inference_models_speech::whisper::{Whisper, WhisperConfig, WhisperGenerationConfig};

#[path = "recording.rs"]
mod recording;

// through the crates directory, so the server crate's tests resolve it too
pub const TINY: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../inference/tests/fixtures/whisper_tiny"
);
const CONFIG: &str = "config.json";
const GENERATION_CONFIG: &str = "generation_config.json";
const TOKENIZER: &str = "tokenizer.json";

/// The committed tiny config, generation config and tokenizer, and random weights.
pub fn tiny_whisper_checkpoint() -> anyhow::Result<tempfile::TempDir> {
    let read = |file: &str| std::fs::read_to_string(format!("{TINY}/{file}"));
    let config: WhisperConfig = serde_json::from_str(&read(CONFIG)?)?;
    let generation: WhisperGenerationConfig = serde_json::from_str(&read(GENERATION_CONFIG)?)?;
    let tokenizer = tokenizers::Tokenizer::from_file(format!("{TINY}/{TOKENIZER}"))
        .map_err(anyhow::Error::msg)?;
    let files = recording::fixture_files(TINY)?;
    let files = files.iter().map(|path| path.as_path()).collect::<Vec<_>>();
    recording::record_checkpoint_seeded_by_name(&files, 0, &[], |vb, _metadata| {
        Whisper::new(&config, generation, tokenizer, vb).map(|_| ())
    })
}
