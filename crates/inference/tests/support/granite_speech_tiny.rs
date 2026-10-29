//! Builds a tiny random-weight Granite Speech checkpoint at test time.
#![allow(dead_code)]

use inference_models_other::granite_speech::{Config, GraniteSpeechModel};
use inference_nn::paged_attention::AttentionImplementation;

#[path = "recording.rs"]
mod recording;

pub const TINY: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../inference/tests/fixtures/granite_speech/tiny"
);
const WEIGHTS: &str = "model.safetensors";
const RUNNING_VAR: &str = ".batch_norm.running_var";

/// The committed tiny config, tokenizer and template, and random weights, BatchNorm variances made positive.
pub fn tiny_checkpoint() -> anyhow::Result<tempfile::TempDir> {
    let cfg: Config =
        serde_json::from_str(&std::fs::read_to_string(format!("{TINY}/config.json"))?)?;
    let layers = cfg.text_config.num_hidden_layers;
    let absent = (0..layers)
        .flat_map(|layer| {
            let prefix = format!("language_model.model.layers.{layer}");
            [
                format!("{prefix}.shared_mlp.input_linear.weight"),
                format!("{prefix}.self_attn.sinks"),
            ]
        })
        .collect::<Vec<_>>();
    let absent = absent.iter().map(String::as_str).collect::<Vec<_>>();
    let files = recording::fixture_files(TINY)?;
    let files = files.iter().map(|path| path.as_path()).collect::<Vec<_>>();
    let dir =
        recording::record_checkpoint_seeded_by_name(&files, layers, &absent, |vb, metadata| {
            GraniteSpeechModel::new(&cfg, vb, true, metadata, AttentionImplementation::Eager)
                .map(|_| ())
        })?;
    let path = dir.path().join(WEIGHTS);
    let mut weights = inference_tensor::safetensors::load(&path, &inference_tensor::Device::Cpu)?;
    for (name, tensor) in weights.iter_mut() {
        if name.ends_with(RUNNING_VAR) {
            *tensor = tensor.abs()?;
        }
    }
    inference_tensor::safetensors::save(&weights, &path)?;
    Ok(dir)
}
