//! Builds a tiny random-weight BERT embedder (sentence-transformers layout, CLS pooling) at test time.
#![allow(dead_code)]

use std::path::Path;

use inference_models_other::bert::{Config, Model};
use inference_models_other::jina_v3;
use inference_nn::paged_attention::AttentionImplementation;

#[path = "recording.rs"]
mod recording;

const EMBEDDING: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/bert_embedding_tiny"
);
const TOKENIZER: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/paddleocr_vl/tiny/tokenizer.json"
);
const CONFIG: &str = "config.json";
// sentence-transformers keeps each module's config in its own directory.
const POOLING_DIR: &str = "1_Pooling";
const POOLING_CONFIG: &str = "pooling_config.json";
// the classification head sits beside the base model prefix, not under it
const CLASSIFIER_HEAD: &str = "classifier";

/// The committed tiny config, edited by `edit_config`, with a shared tokenizer and random weights.
pub fn tiny_bert_checkpoint(
    edit_config: impl FnOnce(&mut serde_json::Value),
) -> anyhow::Result<tempfile::TempDir> {
    let mut files = recording::fixture_files(EMBEDDING)?;
    files.retain(|path| !path.ends_with(POOLING_CONFIG) && !path.ends_with(CONFIG));
    files.push(TOKENIZER.into());
    let mut config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{EMBEDDING}/{CONFIG}"))?)?;
    edit_config(&mut config);
    let cfg: Config = serde_json::from_value(config.clone())?;
    let files = files
        .iter()
        .map(|path| path.as_path())
        .collect::<Vec<&Path>>();
    let dir = recording::record_checkpoint(&files, cfg.num_hidden_layers, &[], |vb, metadata| {
        Model::new(&cfg, vb, metadata, AttentionImplementation::Eager).map(|_| ())
    })?;
    std::fs::write(dir.path().join(CONFIG), serde_json::to_string(&config)?)?;
    std::fs::create_dir(dir.path().join(POOLING_DIR))?;
    std::fs::copy(
        format!("{EMBEDDING}/{POOLING_CONFIG}"),
        dir.path().join(POOLING_DIR).join(CONFIG),
    )?;
    Ok(dir)
}

/// Moves the encoder's weights under `prefix`, as `*ForSequenceClassification` checkpoints store them.
pub fn prefix_encoder_weights(dir: &Path, prefix: &str) -> anyhow::Result<()> {
    let path = dir.join("model.safetensors");
    let tensors = inference_tensor::safetensors::load(&path, &inference_tensor::Device::Cpu)?;
    let renamed = tensors
        .into_iter()
        .map(|(name, tensor)| match name.starts_with(CLASSIFIER_HEAD) {
            true => (name, tensor),
            false => (format!("{prefix}.{name}"), tensor),
        })
        .collect::<std::collections::HashMap<_, _>>();
    inference_tensor::safetensors::save(&renamed, &path)?;
    Ok(())
}

/// A Jina v3 config: XLM-R with rotary positions and two task adapters, the first with an instruction.
pub fn jina_v3_config(config: &mut serde_json::Value) {
    *config = serde_json::json!({
        "architectures": ["XLMRobertaModel"],
        "vocab_size": 266,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "hidden_act": "gelu",
        "max_position_embeddings": 66,
        "type_vocab_size": 1,
        "layer_norm_eps": 1e-5,
        "pad_token_id": 1,
        "position_embedding_type": "rotary",
        "rotary_emb_base": 20000.0,
        "lora_adaptations": [JINA_V3_TASKS[0], JINA_V3_TASKS[1]],
        "task_instructions": {JINA_V3_TASKS[0]: JINA_V3_INSTRUCTION, JINA_V3_TASKS[1]: ""},
        "lora_rank": 2,
        "lora_alpha": 1.0
    });
}

pub const JINA_V3_TASKS: [&str; 2] = ["retrieval.query", "text-matching"];
pub const JINA_V3_INSTRUCTION: &str = "query: ";

/// The tiny Jina v3, recorded with its own weights (base, LoRA adapters and all) in sentence-transformers layout.
pub fn tiny_jina_v3_checkpoint() -> anyhow::Result<tempfile::TempDir> {
    let mut files = recording::fixture_files(EMBEDDING)?;
    files.retain(|path| !path.ends_with(POOLING_CONFIG) && !path.ends_with(CONFIG));
    files.push(TOKENIZER.into());
    let mut config = serde_json::Value::Null;
    jina_v3_config(&mut config);
    let cfg: jina_v3::Config = serde_json::from_value(config.clone())?;
    let files = files
        .iter()
        .map(|path| path.as_path())
        .collect::<Vec<&Path>>();
    let dir = recording::record_checkpoint(&files, cfg.num_hidden_layers, &[], |vb, metadata| {
        jina_v3::Model::new(&cfg, vb, metadata, AttentionImplementation::Eager).map(|_| ())
    })?;
    std::fs::write(dir.path().join(CONFIG), serde_json::to_string(&config)?)?;
    std::fs::create_dir(dir.path().join(POOLING_DIR))?;
    // Jina v3 mean pools
    let mut pooling: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!(
        "{EMBEDDING}/{POOLING_CONFIG}"
    ))?)?;
    pooling["pooling_mode_cls_token"] = false.into();
    pooling["pooling_mode_mean_tokens"] = true.into();
    std::fs::write(
        dir.path().join(POOLING_DIR).join(CONFIG),
        serde_json::to_string(&pooling)?,
    )?;
    Ok(dir)
}
