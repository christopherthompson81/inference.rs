//! Builds a tiny random-weight Cohere Transcribe checkpoint at test time: the committed config and tokenizer, and
//! weights under the release's NeMo names.
#![allow(dead_code)]

use inference_models_speech::cohere_asr::{CohereAsr, CohereAsrConfig};
use inference_models_speech::nemo::nemo_encoder_name;

#[path = "recording.rs"]
mod recording;

// through the crates directory, so the server crate's tests resolve it too
pub const TINY: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../inference/tests/fixtures/cohere_asr_tiny"
);
const CONFIG: &str = "config.json";
const TOKENIZER: &str = "tokenizer.json";
const WEIGHTS: &str = "model.safetensors";
const LAYERS: usize = 2;
// the conv modules' BatchNorm statistics stay out, so random variances cannot go negative
const STATISTICS: [&str; 2] = ["running_mean", "running_var"];

/// The committed tiny config and tokenizer, and random weights.
pub fn tiny_cohere_asr_checkpoint() -> anyhow::Result<tempfile::TempDir> {
    let config: CohereAsrConfig =
        serde_json::from_str(&std::fs::read_to_string(format!("{TINY}/{CONFIG}"))?)?;
    let tokenizer = tokenizers::Tokenizer::from_file(format!("{TINY}/{TOKENIZER}"))
        .map_err(anyhow::Error::msg)?;
    let absent: Vec<String> = (0..LAYERS)
        .flat_map(|i| STATISTICS.map(|s| format!("encoder.layers.{i}.conv.norm.{s}")))
        .collect();
    let absent: Vec<&str> = absent.iter().map(String::as_str).collect();
    let tensors = recording::record_plain_and_sharded(&absent, |encoder, vb| {
        CohereAsr::new(&config, tokenizer, encoder, vb).map(|_| ())
    })?;
    let named = tensors
        .into_iter()
        .map(|(name, t)| (nemo_encoder_name(&name), t))
        .collect();
    let dir = tempfile::tempdir()?;
    for file in recording::fixture_files(TINY)? {
        std::fs::copy(&file, dir.path().join(file.file_name().unwrap()))?;
    }
    inference_tensor::safetensors::save(&named, dir.path().join(WEIGHTS))?;
    Ok(dir)
}
