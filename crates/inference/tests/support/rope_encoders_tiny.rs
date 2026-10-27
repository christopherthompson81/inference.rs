//! Builds tiny random-weight Nomic BERT and ModernBERT embedders (sentence-transformers layout, mean pooling).
#![allow(dead_code)]

use std::path::Path;

use inference_models_other::{modern_bert, nomic_bert};
use inference_nn::paged_attention::AttentionImplementation;

#[path = "recording.rs"]
mod recording;

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
const NOMIC: &str = "nomic_bert_embedding_tiny";
const MODERN: &str = "modern_bert_embedding_tiny";
const TOKENIZER: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/paddleocr_vl/tiny/tokenizer.json"
);
const CONFIG: &str = "config.json";
// sentence-transformers keeps each module's config in its own directory.
const POOLING_DIR: &str = "1_Pooling";
const POOLING_CONFIG: &str = "pooling_config.json";

fn checkpoint(
    fixture: &str,
    edit_config: impl FnOnce(&mut serde_json::Value),
    record: impl FnOnce(&serde_json::Value, &[&Path]) -> anyhow::Result<tempfile::TempDir>,
) -> anyhow::Result<tempfile::TempDir> {
    let dir = format!("{FIXTURES}/{fixture}");
    let mut files = recording::fixture_files(&dir)?;
    files.retain(|path| !path.ends_with(POOLING_CONFIG) && !path.ends_with(CONFIG));
    files.push(TOKENIZER.into());
    let mut config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{dir}/{CONFIG}"))?)?;
    edit_config(&mut config);
    let files = files.iter().map(|p| p.as_path()).collect::<Vec<&Path>>();
    let out = record(&config, &files)?;
    std::fs::write(out.path().join(CONFIG), serde_json::to_string(&config)?)?;
    std::fs::create_dir(out.path().join(POOLING_DIR))?;
    std::fs::copy(
        format!("{dir}/{POOLING_CONFIG}"),
        out.path().join(POOLING_DIR).join(CONFIG),
    )?;
    Ok(out)
}

pub fn tiny_nomic_bert_checkpoint() -> anyhow::Result<tempfile::TempDir> {
    checkpoint(
        NOMIC,
        |_| {},
        |config, files| {
            let cfg: nomic_bert::Config = serde_json::from_value(config.clone())?;
            recording::record_checkpoint(files, cfg.num_hidden_layers(), &[], |vb, metadata| {
                nomic_bert::Model::new(&cfg, vb, metadata, AttentionImplementation::Eager)
                    .map(|_| ())
            })
        },
    )
}

/// The committed config, edited by `edit_config`, recorded with ModernBERT's weights.
pub fn tiny_modern_bert_checkpoint(
    edit_config: impl FnOnce(&mut serde_json::Value),
) -> anyhow::Result<tempfile::TempDir> {
    checkpoint(MODERN, edit_config, |config, files| {
        let cfg: modern_bert::Config = serde_json::from_value(config.clone())?;
        recording::record_checkpoint(files, cfg.num_hidden_layers, &[], |vb, metadata| {
            modern_bert::Model::new(&cfg, vb, metadata, AttentionImplementation::Eager).map(|_| ())
        })
    })
}

/// Moves every weight under `prefix`, as `*ForMaskedLM` checkpoints store the encoder.
pub fn prefix_weights(dir: &Path, prefix: &str) -> anyhow::Result<()> {
    let path = dir.join("model.safetensors");
    let tensors = inference_tensor::safetensors::load(&path, &inference_tensor::Device::Cpu)?;
    let renamed = tensors
        .into_iter()
        .map(|(name, tensor)| (format!("{prefix}.{name}"), tensor))
        .collect::<std::collections::HashMap<_, _>>();
    inference_tensor::safetensors::save(&renamed, &path)?;
    Ok(())
}
