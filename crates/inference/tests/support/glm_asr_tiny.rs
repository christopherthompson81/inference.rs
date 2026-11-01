//! Builds a tiny random-weight GLM-ASR checkpoint at test time.
#![allow(dead_code)]

use inference_models_llama::glm_asr::{GlmAsrConfig, GlmAsrModel};
use inference_nn::paged_attention::AttentionImplementation;

#[path = "recording.rs"]
mod recording;

pub const TINY: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../inference/tests/fixtures/glm_asr/tiny"
);

/// The committed tiny config, tokenizer and the released template, and random weights.
pub fn tiny_checkpoint() -> anyhow::Result<tempfile::TempDir> {
    let config = std::fs::read_to_string(format!("{TINY}/config.json"))?;
    let cfg = GlmAsrConfig::from_json(&config)?;
    let files = recording::fixture_files(TINY)?;
    let files = files.iter().map(|path| path.as_path()).collect::<Vec<_>>();
    recording::record_checkpoint_seeded_by_name(
        &files,
        cfg.text_config.num_hidden_layers,
        &[],
        |vb, metadata| {
            GlmAsrModel::new(&cfg, vb, true, metadata, AttentionImplementation::Eager).map(|_| ())
        },
    )
}
