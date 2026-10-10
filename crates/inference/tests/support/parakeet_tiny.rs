//! Builds a tiny random-weight Parakeet checkpoint (safetensors, configs, tokenizer) for any of its heads at test time.
#![allow(dead_code)]

use std::path::Path;

use inference_models_speech::parakeet::{Parakeet, ParakeetConfig, ProcessorConfig};
use inference_tensor::Device;

#[path = "recording.rs"]
mod recording;

// through the crates directory, so the server crate's tests resolve it too
const FIXTURES: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../inference/tests/fixtures/parakeet_tiny"
);
const CONFIG: &str = "config.json";
const PROCESSOR_CONFIG: &str = "processor_config.json";
const TOKENIZER: &str = "tokenizer.json";
const WEIGHTS: &str = "model.safetensors";
const RUNNING_VAR: &str = "running_var";
// the tiny tokenizer's `<blank>`, which a CTC checkpoint names as its pad token
const BLANK_ID: u64 = 32;

/// The heads a Parakeet checkpoint can carry, as `model_type`s, then the streaming Nemotron ASR ones (RNN-T over a
/// chunked-limited causal encoder; 3.5 also takes a language prompt).
pub const HEADS: [&str; 5] = [
    "parakeet_tdt",
    "parakeet_rnnt",
    "parakeet_ctc",
    "nemotron_asr_streaming",
    "nemotron3_5_asr",
];
pub const PROMPTED: &str = "nemotron3_5_asr";
// the tiny streaming encoder: 4 frames back and a 2-frame chunk, so a few seconds cross many chunk boundaries
const STREAMING_WINDOW: u64 = 5;
const STREAMING_LOOKAHEAD: u64 = 1;
const PROMPTS: u64 = 4;
const PROMPT_INNER: u64 = 8;
/// The tiny 3.5 checkpoint's languages and its auto prompt.
pub const LANGUAGES: [(&str, u64); 3] = [("en-US", 0), ("de-DE", 1), ("auto", 3)];

/// The committed fixtures with `model_type` set to `head`, and random weights for every tensor the model loads.
pub fn tiny_parakeet_checkpoint(head: &str) -> anyhow::Result<tempfile::TempDir> {
    let fixtures = Path::new(FIXTURES);
    let mut config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fixtures.join(CONFIG))?)?;
    config["model_type"] = head.into();
    if head == "parakeet_ctc" {
        config["pad_token_id"] = BLANK_ID.into();
    }
    if head != "parakeet_tdt" {
        config["durations"] = serde_json::json!([]);
    }
    let streaming = head.starts_with("nemotron");
    if streaming {
        let encoder = &mut config["encoder_config"];
        encoder["model_type"] = "nemotron_asr_streaming_encoder".into();
        encoder["sliding_window"] = STREAMING_WINDOW.into();
        encoder["default_num_lookahead_tokens"] = STREAMING_LOOKAHEAD.into();
    }
    let mut processor_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fixtures.join(PROCESSOR_CONFIG))?)?;
    if head == PROMPTED {
        config["num_prompts"] = PROMPTS.into();
        config["prompt_intermediate_size"] = PROMPT_INNER.into();
        config["default_prompt_id"] = LANGUAGES[2].1.into();
        processor_json["prompt_dictionary"] = LANGUAGES
            .iter()
            .map(|(l, id)| (l.to_string(), serde_json::Value::from(*id)))
            .collect::<serde_json::Map<_, _>>()
            .into();
    }
    let staged = tempfile::tempdir()?;
    let staged_config = staged.path().join(CONFIG);
    std::fs::write(&staged_config, serde_json::to_string_pretty(&config)?)?;
    let staged_processor = staged.path().join(PROCESSOR_CONFIG);
    std::fs::write(
        &staged_processor,
        serde_json::to_string_pretty(&processor_json)?,
    )?;
    let processor: ProcessorConfig = serde_json::from_value(processor_json)?;
    let parsed: ParakeetConfig = serde_json::from_value(config)?;
    let tokenizer =
        tokenizers::Tokenizer::from_file(fixtures.join(TOKENIZER)).map_err(anyhow::Error::msg)?;
    let dir = recording::record_plain_checkpoint(
        &[
            staged_config.as_path(),
            &staged_processor,
            &fixtures.join(TOKENIZER),
        ],
        |vb| Parakeet::new(parsed, processor, tokenizer, vb).map(|_| ()),
    )?;
    // drawn from a normal, a batch norm's variance would be negative half the time
    let weights = dir.path().join(WEIGHTS);
    let mut tensors = inference_tensor::safetensors::load(&weights, &Device::Cpu)?;
    for (name, t) in tensors.iter_mut() {
        if name.ends_with(RUNNING_VAR) {
            *t = (t.abs()? + 1.0)?;
        }
    }
    inference_tensor::safetensors::save(&tensors, &weights)?;
    Ok(dir)
}
