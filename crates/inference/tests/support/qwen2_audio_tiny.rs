//! Builds a tiny random-weight Qwen2-Audio checkpoint at test time.
#![allow(dead_code)]

use inference_models_qwen::qwen2_audio::{Qwen2AudioConfig, Qwen2AudioModel};
use inference_nn::paged_attention::AttentionImplementation;

#[path = "recording.rs"]
mod recording;

pub const TINY: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../inference/tests/fixtures/qwen2_audio/tiny"
);

/// The committed tiny config, tokenizer and the released text-only template, and random weights.
pub fn tiny_checkpoint() -> anyhow::Result<tempfile::TempDir> {
    let config = std::fs::read_to_string(format!("{TINY}/config.json"))?;
    let cfg = Qwen2AudioConfig::from_json(&config)?;
    let files = recording::fixture_files(TINY)?;
    let files = files.iter().map(|path| path.as_path()).collect::<Vec<_>>();
    recording::record_checkpoint_seeded_by_name(
        &files,
        cfg.spec().text.num_hidden_layers,
        &[],
        |vb, metadata| {
            Qwen2AudioModel::new(
                &cfg.spec(),
                vb,
                true,
                metadata,
                AttentionImplementation::Eager,
            )
            .map(|_| ())
        },
    )
}
