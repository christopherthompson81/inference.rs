//! Builds a tiny random-weight MERaLiON-2 checkpoint at test time.
#![allow(dead_code)]

use inference_models_gemma::meralion2::{MERaLiON2Config, MERaLiON2Model};
use inference_nn::paged_attention::AttentionImplementation;

#[path = "recording.rs"]
mod recording;

pub const TINY: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../inference/tests/fixtures/meralion2/tiny"
);

/// The committed tiny config, tokenizer and the repo's Gemma template, and random weights.
pub fn tiny_checkpoint() -> anyhow::Result<tempfile::TempDir> {
    let config = std::fs::read_to_string(format!("{TINY}/config.json"))?;
    let cfg = MERaLiON2Config::from_json(&config)?;
    let files = recording::fixture_files(TINY)?;
    let files = files.iter().map(|path| path.as_path()).collect::<Vec<_>>();
    recording::record_checkpoint_seeded_by_name(
        &files,
        cfg.text_config.num_hidden_layers,
        &[],
        |vb, metadata| {
            MERaLiON2Model::new(&cfg, vb, true, metadata, AttentionImplementation::Eager)
                .map(|_| ())
        },
    )
}
