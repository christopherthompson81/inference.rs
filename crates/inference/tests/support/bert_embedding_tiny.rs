//! Builds a tiny random-weight BERT embedder (sentence-transformers layout, CLS pooling) at test time.
#![allow(dead_code)]

use std::path::Path;

use inference_models_other::bert::{Config, Model};
use inference_nn::paged_attention::AttentionImplementation;

#[path = "recording.rs"]
mod recording;

const EMBEDDING: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/bert_embedding_tiny"
);
const TOKENIZER: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/paddleocr_vl/tiny/tokenizer.json"
);
const CONFIG: &str = "config.json";
// sentence-transformers keeps each module's config in its own directory.
const POOLING_DIR: &str = "1_Pooling";
const POOLING_CONFIG: &str = "pooling_config.json";
// the classification head sits beside the base model prefix, not under it
const CLASSIFIER_HEAD: &str = "classifier";

/// The committed tiny config, edited by `edit_config`, with a shared tokenizer and random weights.
pub fn tiny_bert_checkpoint(
    edit_config: impl FnOnce(&mut serde_json::Value),
) -> anyhow::Result<tempfile::TempDir> {
    let mut files = recording::fixture_files(EMBEDDING)?;
    files.retain(|path| !path.ends_with(POOLING_CONFIG) && !path.ends_with(CONFIG));
    files.push(TOKENIZER.into());
    let mut config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{EMBEDDING}/{CONFIG}"))?)?;
    edit_config(&mut config);
    let cfg: Config = serde_json::from_value(config.clone())?;
    let files = files
        .iter()
        .map(|path| path.as_path())
        .collect::<Vec<&Path>>();
    let dir = recording::record_checkpoint(&files, cfg.num_hidden_layers, &[], |vb, metadata| {
        Model::new(&cfg, vb, metadata, AttentionImplementation::Eager).map(|_| ())
    })?;
    std::fs::write(dir.path().join(CONFIG), serde_json::to_string(&config)?)?;
    std::fs::create_dir(dir.path().join(POOLING_DIR))?;
    std::fs::copy(
        format!("{EMBEDDING}/{POOLING_CONFIG}"),
        dir.path().join(POOLING_DIR).join(CONFIG),
    )?;
    Ok(dir)
}

/// Moves the encoder's weights under `prefix`, as `*ForSequenceClassification` checkpoints store them.
pub fn prefix_encoder_weights(dir: &Path, prefix: &str) -> anyhow::Result<()> {
    let path = dir.join("model.safetensors");
    let tensors = inference_tensor::safetensors::load(&path, &inference_tensor::Device::Cpu)?;
    let renamed = tensors
        .into_iter()
        .map(|(name, tensor)| match name.starts_with(CLASSIFIER_HEAD) {
            true => (name, tensor),
            false => (format!("{prefix}.{name}"), tensor),
        })
        .collect::<std::collections::HashMap<_, _>>();
    inference_tensor::safetensors::save(&renamed, &path)?;
    Ok(())
}
