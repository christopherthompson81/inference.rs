//! Builds a tiny random-weight Granite 4 Vision checkpoint at test time.
#![allow(dead_code)]

use inference_models_other::granite4_vision::{Config, Granite4VisionModel};
use inference_nn::paged_attention::AttentionImplementation;

#[path = "recording.rs"]
mod recording;

pub const TINY: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../inference/tests/fixtures/granite4_vision/tiny"
);

/// The committed tiny config, tokenizer and templates, and random weights; the text layers keep a Llama-style `mlp`.
pub fn tiny_checkpoint() -> anyhow::Result<tempfile::TempDir> {
    let cfg: Config =
        serde_json::from_str(&std::fs::read_to_string(format!("{TINY}/config.json"))?)?;
    let layers = cfg.text_config.num_hidden_layers;
    let absent = (0..layers)
        .flat_map(|layer| {
            let prefix = format!("model.language_model.layers.{layer}");
            [
                format!("{prefix}.shared_mlp.input_linear.weight"),
                format!("{prefix}.self_attn.sinks"),
            ]
        })
        .collect::<Vec<_>>();
    let absent = absent.iter().map(String::as_str).collect::<Vec<_>>();
    let files = recording::fixture_files(TINY)?;
    let files = files.iter().map(|path| path.as_path()).collect::<Vec<_>>();
    recording::record_checkpoint_seeded_by_name(&files, layers, &absent, |vb, metadata| {
        Granite4VisionModel::new(&cfg, vb, true, metadata, AttentionImplementation::Eager)
            .map(|_| ())
    })
}
