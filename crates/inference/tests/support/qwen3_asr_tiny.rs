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

const CONFIG: &str = "config.json";
const WEIGHTS: &str = "model.safetensors";

/// The committed tiny config, tokenizer and template, and random weights, in the original repos' layout.
pub fn tiny_checkpoint() -> anyhow::Result<tempfile::TempDir> {
    record(&std::fs::read_to_string(format!("{TINY}/{CONFIG}"))?)
}

/// [`tiny_checkpoint`]'s weights in transformers' `-hf` layout: the config flat, the tensors renamed.
pub fn tiny_transformers_checkpoint() -> anyhow::Result<tempfile::TempDir> {
    let original = tiny_checkpoint()?;
    let mut config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(original.path().join(CONFIG))?)?;
    let thinker = config
        .as_object_mut()
        .and_then(|config| config.remove("thinker_config"))
        .ok_or_else(|| anyhow::anyhow!("the tiny config has no thinker_config"))?;
    for (key, value) in thinker.as_object().into_iter().flatten() {
        config[key] = value.clone();
    }
    std::fs::write(original.path().join(CONFIG), config.to_string())?;
    let weights = original.path().join(WEIGHTS);
    let renamed = inference_tensor::safetensors::load(&weights, &inference_tensor::Device::Cpu)?
        .into_iter()
        .map(|(name, tensor)| (transformers_name(&name), tensor))
        .collect::<std::collections::HashMap<_, _>>();
    inference_tensor::safetensors::save(&renamed, &weights)?;
    Ok(original)
}

fn transformers_name(name: &str) -> String {
    for (from, to) in [
        (
            "thinker.audio_tower.proj1",
            "model.multi_modal_projector.linear_1",
        ),
        (
            "thinker.audio_tower.proj2",
            "model.multi_modal_projector.linear_2",
        ),
        ("thinker.audio_tower", "model.audio_tower"),
        ("thinker.model", "model.language_model"),
        ("thinker.lm_head", "lm_head"),
    ] {
        if let Some(rest) = name.strip_prefix(from) {
            return format!("{to}{rest}");
        }
    }
    name.to_string()
}

fn record(config: &str) -> anyhow::Result<tempfile::TempDir> {
    let cfg = Config::from_json(config)?;
    let layers = cfg.thinker_config.text_config.num_hidden_layers;
    let scratch = tempfile::tempdir()?;
    let config_path = scratch.path().join(CONFIG);
    std::fs::write(&config_path, config)?;
    let mut files = recording::fixture_files(TINY)?;
    files.retain(|path| !path.ends_with(CONFIG));
    files.push(config_path);
    let files = files.iter().map(|path| path.as_path()).collect::<Vec<_>>();
    recording::record_checkpoint_seeded_by_name(&files, layers, &[], |vb, metadata| {
        Qwen3AsrModel::new(&cfg, vb, true, metadata, AttentionImplementation::Eager).map(|_| ())
    })
}
