//! Builds a tiny random-weight Qwen3 embedding checkpoint (sentence-transformers layout) at test time.
#![allow(dead_code)]

use std::path::Path;

use inference_models_qwen::qwen3_embedding::{Config, Model};
use inference_nn::paged_attention::AttentionImplementation;

#[path = "recording.rs"]
mod recording;

const EMBEDDING: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/qwen3_embedding_tiny"
);
const TOKENIZER: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/paddleocr_vl/tiny/tokenizer.json"
);
// sentence-transformers keeps each module's config in its own directory.
const POOLING_DIR: &str = "1_Pooling";
const POOLING_CONFIG: &str = "pooling_config.json";

/// The committed tiny config and module layout, a shared tokenizer, and random weights for every loaded tensor.
pub fn tiny_embedding_checkpoint() -> anyhow::Result<tempfile::TempDir> {
    let mut files = recording::fixture_files(EMBEDDING)?;
    files.retain(|path| !path.ends_with(POOLING_CONFIG));
    files.push(TOKENIZER.into());
    let cfg: Config = serde_json::from_str(&std::fs::read_to_string(format!(
        "{EMBEDDING}/config.json"
    ))?)?;
    let files = files
        .iter()
        .map(|path| path.as_path())
        .collect::<Vec<&Path>>();
    let dir = recording::record_checkpoint(&files, cfg.num_hidden_layers, &[], |vb, metadata| {
        Model::new(&cfg, vb, true, metadata, AttentionImplementation::Eager).map(|_| ())
    })?;
    std::fs::create_dir(dir.path().join(POOLING_DIR))?;
    std::fs::copy(
        format!("{EMBEDDING}/{POOLING_CONFIG}"),
        dir.path().join(POOLING_DIR).join("config.json"),
    )?;
    Ok(dir)
}
