//! Builds a tiny random-weight Voxtral checkpoint at test time.
#![allow(dead_code)]

use inference_models_llama::voxtral_chat::{VoxtralChatConfig, VoxtralChatModel};
use inference_nn::paged_attention::AttentionImplementation;

#[path = "recording.rs"]
mod recording;

pub const TINY: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../inference/tests/fixtures/voxtral_chat/tiny"
);

/// The committed tiny config and tokenizer (no template: the loader brings its own), and random weights.
pub fn tiny_checkpoint() -> anyhow::Result<tempfile::TempDir> {
    let config = std::fs::read_to_string(format!("{TINY}/config.json"))?;
    let cfg = VoxtralChatConfig::from_json(&config)?;
    let files = recording::fixture_files(TINY)?;
    let files = files.iter().map(|path| path.as_path()).collect::<Vec<_>>();
    recording::record_checkpoint_seeded_by_name(
        &files,
        cfg.text_config.num_hidden_layers,
        &[],
        |vb, metadata| {
            VoxtralChatModel::new(&cfg, vb, true, metadata, AttentionImplementation::Eager)
                .map(|_| ())
        },
    )
}
