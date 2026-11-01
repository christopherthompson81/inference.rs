//! Builds a tiny random-weight Ultravox checkpoint at test time.
#![allow(dead_code)]

use inference_models_llama::ultravox::{UltravoxConfig, UltravoxModel};
use inference_nn::paged_attention::AttentionImplementation;

#[path = "recording.rs"]
mod recording;

// the tiny config has no Llama 3 RoPE scaling, so no frequency factors
const ABSENT: &[&str] = &["language_model.model.rope_freqs.weight"];

pub const TINY: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../inference/tests/fixtures/ultravox/tiny"
);

/// The committed tiny config, tokenizer and a Llama 3 template, and random weights.
pub fn tiny_checkpoint() -> anyhow::Result<tempfile::TempDir> {
    let config = std::fs::read_to_string(format!("{TINY}/config.json"))?;
    let cfg = UltravoxConfig::from_json(&config)?;
    let files = recording::fixture_files(TINY)?;
    let files = files.iter().map(|path| path.as_path()).collect::<Vec<_>>();
    recording::record_checkpoint_seeded_by_name(
        &files,
        cfg.text_config.num_hidden_layers,
        ABSENT,
        |vb, metadata| {
            UltravoxModel::new(&cfg, vb, true, metadata, AttentionImplementation::Eager).map(|_| ())
        },
    )
}
