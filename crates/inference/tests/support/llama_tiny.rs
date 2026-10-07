//! Builds a tiny random-weight Llama checkpoint at test time, for the text load paths (plain, ISQ, UQFF).
#![allow(dead_code)]

use std::path::Path;

use inference_models_llama::llama::{Config, Llama};
use inference_nn::paged_attention::AttentionImplementation;

#[path = "recording.rs"]
mod recording;

// Llama 3 checkpoints may carry per-frequency rope factors; the tiny one uses plain rope.
const ROPE_FREQS: &str = "model.rope_freqs.weight";
const LLAMA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/llama_tiny");
// A plain text tokenizer with a small vocabulary; the Llama config's vocab_size matches it.
const TOKENIZER: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/paddleocr_vl/tiny/tokenizer.json"
);

/// The committed tiny Llama config and templates, a shared tokenizer, and random weights for every loaded tensor.
pub fn tiny_llama_checkpoint() -> anyhow::Result<tempfile::TempDir> {
    tiny_llama_checkpoint_with(serde_json::json!({}))
}

/// As [`tiny_llama_checkpoint`], with `patch`'s keys overriding the committed config's.
pub fn tiny_llama_checkpoint_with(patch: serde_json::Value) -> anyhow::Result<tempfile::TempDir> {
    let mut config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{LLAMA}/config.json"))?)?;
    for (key, value) in patch.as_object().expect("a config patch is a JSON object") {
        config[key] = value.clone();
    }
    let cfg: Config = serde_json::from_value(config.clone())?;
    let staging = tempfile::tempdir()?;
    let config_path = staging.path().join("config.json");
    std::fs::write(&config_path, serde_json::to_string(&config)?)?;
    let mut files = recording::fixture_files(LLAMA)?;
    files.retain(|path| path.file_name().is_some_and(|name| name != "config.json"));
    files.push(config_path);
    files.push(TOKENIZER.into());
    let files = files
        .iter()
        .map(|path| path.as_path())
        .collect::<Vec<&Path>>();
    recording::record_checkpoint(
        &files,
        cfg.num_hidden_layers,
        &[ROPE_FREQS],
        |vb, metadata| {
            Llama::new(
                &cfg.decoder_spec(),
                vb,
                true,
                metadata,
                AttentionImplementation::Eager,
            )
            .map(|_| ())
        },
    )
}
