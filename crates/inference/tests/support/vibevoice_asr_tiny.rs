//! Builds a tiny random-weight VibeVoice-ASR checkpoint at test time: two small speech encoders with an 80-sample hop.
#![allow(dead_code)]

use inference_models_qwen::vibevoice_asr::{VibeVoiceAsrConfig, VibeVoiceAsrModel};
use inference_nn::paged_attention::AttentionImplementation;

#[path = "recording.rs"]
mod recording;

pub const TINY: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../inference/tests/fixtures/vibevoice_asr/tiny"
);
/// The tiny encoders' samples per embedding.
pub const HOP: usize = 80;

/// The committed tiny config, tokenizer and the repo's template, and random weights.
pub fn tiny_checkpoint() -> anyhow::Result<tempfile::TempDir> {
    let config = std::fs::read_to_string(format!("{TINY}/config.json"))?;
    let cfg = VibeVoiceAsrConfig::from_json(&config)?;
    let files = recording::fixture_files(TINY)?;
    let files = files.iter().map(|path| path.as_path()).collect::<Vec<_>>();
    recording::record_checkpoint_seeded_by_name(
        &files,
        cfg.text().num_hidden_layers,
        &[],
        |vb, metadata| {
            VibeVoiceAsrModel::new(&cfg, vb, true, metadata, AttentionImplementation::Eager)
                .map(|_| ())
        },
    )
}
