//! Builds a tiny random-weight DeepSeek-OCR checkpoint at test time.
#![allow(dead_code)]

use std::collections::HashMap;

use inference_models_other::deepseek_ocr::{Config, DeepseekOcrModel};
use inference_nn::paged_attention::AttentionImplementation;

#[path = "recording.rs"]
mod recording;

pub const TINY: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../inference/tests/fixtures/deepseek_ocr/tiny"
);

// the MoE layers' experts, whose shapes the loader reads before their weights
fn expert_shapes(cfg: &Config) -> HashMap<String, Vec<usize>> {
    let text = &cfg.language_config;
    let (hidden, inner) = (text.hidden_size, text.moe_intermediate_size);
    (text.first_k_dense_replace..text.num_hidden_layers)
        .flat_map(|layer| {
            (0..text.n_routed_experts).flat_map(move |expert| {
                let p = format!("model.layers.{layer}.mlp.experts.{expert}");
                [
                    (format!("{p}.gate_proj.weight"), vec![inner, hidden]),
                    (format!("{p}.up_proj.weight"), vec![inner, hidden]),
                    (format!("{p}.down_proj.weight"), vec![hidden, inner]),
                ]
            })
        })
        .collect()
}

/// The committed tiny config and tokenizer (the loader supplies the template), and random weights.
pub fn tiny_checkpoint() -> anyhow::Result<tempfile::TempDir> {
    tiny_checkpoint_with(|_| {})
}

/// As [`tiny_checkpoint`], the config edited first: the weights follow the edited config, which the checkpoint keeps.
pub fn tiny_checkpoint_with(
    edit: impl FnOnce(&mut serde_json::Value),
) -> anyhow::Result<tempfile::TempDir> {
    let mut config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{TINY}/config.json"))?)?;
    edit(&mut config);
    let cfg: Config = serde_json::from_value(config.clone())?;
    let layers = cfg.language_config.num_hidden_layers;
    let files = recording::fixture_files(TINY)?;
    let files = files.iter().map(|path| path.as_path()).collect::<Vec<_>>();
    let dir = recording::record_checkpoint_seeded_with_shapes(
        &files,
        layers,
        &[],
        expert_shapes(&cfg),
        |vb, metadata| {
            DeepseekOcrModel::new(&cfg, vb, true, metadata, AttentionImplementation::Eager)
                .map(|_| ())
        },
    )?;
    std::fs::write(
        dir.path().join("config.json"),
        serde_json::to_string(&config)?,
    )?;
    Ok(dir)
}
