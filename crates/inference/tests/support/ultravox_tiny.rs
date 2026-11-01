//! Builds a tiny random-weight Ultravox checkpoint at test time.
#![allow(dead_code)]

use inference_models_llama::ultravox::{UltravoxConfig, UltravoxModel};
use inference_nn::paged_attention::AttentionImplementation;
use inference_tensor::Device;

#[path = "recording.rs"]
mod recording;

// the tiny config has no Llama 3 RoPE scaling, so no frequency factors
const ABSENT: &[&str] = &["model.rope_freqs.weight"];
const WEIGHTS: &str = "model.safetensors";
const CONFIG: &str = "config.json";
// the text model's tensors, which a safetensors repo leaves to the repository its `text_model_id` names
const TEXT_PREFIXES: [&str; 2] = ["model.", "lm_head."];
// what a repo's own placeholder is, one past the text model's vocabulary
const REPO_AUDIO_TOKEN_INDEX: u64 = 266;

pub const TINY: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../inference/tests/fixtures/ultravox/tiny"
);

/// The committed tiny config, tokenizer and a Llama 3 template, and random weights.
pub fn tiny_checkpoint() -> anyhow::Result<tempfile::TempDir> {
    let config = std::fs::read_to_string(format!("{TINY}/config.json"))?;
    let cfg = UltravoxConfig::from_json(&config)?;
    let files = recording::fixture_files(TINY)?;
    let files = files.iter().map(|path| path.as_path()).collect::<Vec<_>>();
    recording::record_checkpoint_seeded_by_name(
        &files,
        cfg.text_config.num_hidden_layers,
        ABSENT,
        |vb, metadata| {
            UltravoxModel::new(&cfg, vb, true, metadata, AttentionImplementation::Eager).map(|_| ())
        },
    )
}

/// The tiny checkpoint as Ultravox publishes it: a repo holding the tower and projector, its config naming the
/// text model's directory (the second dir) in place of describing it.
pub fn tiny_repo_pair() -> anyhow::Result<(tempfile::TempDir, tempfile::TempDir)> {
    let whole = tiny_checkpoint()?;
    let tensors = inference_tensor::safetensors::load(whole.path().join(WEIGHTS), &Device::Cpu)?;
    let (text, audio): (
        std::collections::HashMap<_, _>,
        std::collections::HashMap<_, _>,
    ) = tensors
        .into_iter()
        .partition(|(name, _)| TEXT_PREFIXES.iter().any(|prefix| name.starts_with(prefix)));
    let mut config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(whole.path().join(CONFIG))?)?;
    let text_config = config
        .as_object_mut()
        .and_then(|config| config.remove("text_config"))
        .ok_or_else(|| anyhow::anyhow!("the tiny config has no text_config"))?;
    let text_dir = tempfile::tempdir()?;
    std::fs::write(text_dir.path().join(CONFIG), text_config.to_string())?;
    inference_tensor::safetensors::save(&text, text_dir.path().join(WEIGHTS))?;
    config.as_object_mut().unwrap().remove("audio_token_id");
    config["text_model_id"] = text_dir.path().to_string_lossy().into_owned().into();
    config["audio_token_index"] = REPO_AUDIO_TOKEN_INDEX.into();
    std::fs::write(whole.path().join(CONFIG), config.to_string())?;
    inference_tensor::safetensors::save(&audio, whole.path().join(WEIGHTS))?;
    Ok((whole, text_dir))
}
