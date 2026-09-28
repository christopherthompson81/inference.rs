//! Builds a tiny random-weight PaddleOCR-VL checkpoint at test time; shared by the SDK, server and ABI tests.
// Each test crate includes this file and uses a different subset of it.
#![allow(dead_code)]

use inference_models_other::paddleocr_vl::{config::Config, PaddleOcrVlModel};
use inference_nn::paged_attention::AttentionImplementation;

#[path = "recording.rs"]
mod recording;

// Resolves from any crate under crates/, so every test crate reads the same committed config and tokenizer.
pub const TINY: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../inference/tests/fixtures/paddleocr_vl/tiny"
);
pub const FIXTURES: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../inference/tests/fixtures/paddleocr_vl"
);

/// The committed tiny config, tokenizer and templates plus random weights for exactly the tensors the model loads.
pub fn tiny_checkpoint() -> anyhow::Result<tempfile::TempDir> {
    let files = recording::fixture_files(TINY)?;
    let cfg: Config =
        serde_json::from_str(&std::fs::read_to_string(format!("{TINY}/config.json"))?)?;
    let files = files.iter().map(|path| path.as_path()).collect::<Vec<_>>();
    recording::record_checkpoint(
        &files,
        cfg.text_config().num_hidden_layers,
        &[],
        |vb, metadata| {
            PaddleOcrVlModel::new(&cfg, vb, metadata, AttentionImplementation::Eager).map(|_| ())
        },
    )
}
