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
        Qwen3AsrModel::new(
            &cfg,
            vb,
            true,
            metadata,
            AttentionImplementation::Eager,
            None,
        )
        .map(|_| ())
    })
}

// the tiny aligner's `<timestamp>` token, the next id after the fixture's added tokens, and its head's time steps
const TIMESTAMP_TOKEN: u32 = 265;
const ALIGNER_CLASSES: usize = 64;
const SEGMENT_MS: f64 = 80.0;
const TOKENIZER: &str = "tokenizer.json";

/// A tiny Qwen3-ForcedAligner in the original layout: the fixture's model with a `<timestamp>` token and a head of
/// 64 time steps of 80 ms (5.12 s) in place of the vocabulary's.
pub fn tiny_aligner_checkpoint() -> anyhow::Result<tempfile::TempDir> {
    let mut config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{TINY}/{CONFIG}"))?)?;
    config["timestamp_token_id"] = TIMESTAMP_TOKEN.into();
    config["timestamp_segment_time"] = SEGMENT_MS.into();
    let thinker = &mut config["thinker_config"];
    thinker["model_type"] = "qwen3_forced_aligner".into();
    thinker["classify_num"] = ALIGNER_CLASSES.into();
    thinker["text_config"]["vocab_size"] = (TIMESTAMP_TOKEN + 1).into();
    let mut tokenizer: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{TINY}/{TOKENIZER}"))?)?;
    let added = tokenizer["added_tokens"]
        .as_array_mut()
        .ok_or_else(|| anyhow::anyhow!("the tiny tokenizer has no added tokens"))?;
    added.push(serde_json::json!({
        "id": TIMESTAMP_TOKEN, "content": "<timestamp>", "single_word": false, "lstrip": false,
        "rstrip": false, "normalized": false, "special": true
    }));
    let config = config.to_string();
    let cfg = Config::from_json(&config)?;
    let layers = cfg.thinker_config.text_config.num_hidden_layers;
    let scratch = tempfile::tempdir()?;
    let config_path = scratch.path().join(CONFIG);
    let tokenizer_path = scratch.path().join(TOKENIZER);
    std::fs::write(&config_path, &config)?;
    std::fs::write(&tokenizer_path, tokenizer.to_string())?;
    let mut files = recording::fixture_files(TINY)?;
    files.retain(|path| !path.ends_with(CONFIG) && !path.ends_with(TOKENIZER));
    files.extend([config_path, tokenizer_path.clone()]);
    let files = files.iter().map(|path| path.as_path()).collect::<Vec<_>>();
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("tiny aligner tokenizer: {e}"))?;
    recording::record_checkpoint_seeded_by_name(&files, layers, &[], |vb, metadata| {
        Qwen3AsrModel::new(
            &cfg,
            vb,
            true,
            metadata,
            AttentionImplementation::Eager,
            Some(tokenizer),
        )
        .map(|_| ())
    })
}
