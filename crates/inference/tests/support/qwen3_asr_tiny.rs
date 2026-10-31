//! Builds a tiny random-weight Qwen3-ASR checkpoint at test time.
#![allow(dead_code)]

use inference_models_qwen::qwen3_asr::{Config, Qwen3AsrModel};
use inference_nn::paged_attention::AttentionImplementation;

#[path = "recording.rs"]
mod recording;

pub const TINY: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../inference/tests/fixtures/qwen3_asr/tiny"
);

/// The committed tiny config, tokenizer and template, and random weights.
pub fn tiny_checkpoint() -> anyhow::Result<tempfile::TempDir> {
    let cfg: Config =
        serde_json::from_str(&std::fs::read_to_string(format!("{TINY}/config.json"))?)?;
    let layers = cfg.thinker_config.text_config.num_hidden_layers;
    let files = recording::fixture_files(TINY)?;
    let files = files.iter().map(|path| path.as_path()).collect::<Vec<_>>();
    recording::record_checkpoint_seeded_by_name(&files, layers, &[], |vb, metadata| {
        Qwen3AsrModel::new(&cfg, vb, true, metadata, AttentionImplementation::Eager).map(|_| ())
    })
}
