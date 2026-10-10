//! Builds a tiny random-weight Nemotron-3 Diarization checkpoint (safetensors and configs) at test time.
#![allow(dead_code)]

use std::path::Path;

use inference_models_speech::diarization::Nemotron3Diarizer;

#[path = "recording.rs"]
mod recording;

// through the crates directory, so other crates' tests resolve it too
const FIXTURES: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../inference/tests/fixtures/nemotron3_diarization_tiny"
);
const CONFIG: &str = "config.json";
const PROCESSOR_CONFIG: &str = "processor_config.json";

/// The committed tiny configs and random weights for every tensor the model loads.
pub fn tiny_nemotron3_diarization() -> anyhow::Result<tempfile::TempDir> {
    let fixtures = Path::new(FIXTURES);
    let (config, processor) = (fixtures.join(CONFIG), fixtures.join(PROCESSOR_CONFIG));
    recording::record_plain_checkpoint(&[&config, &processor], |vb| {
        Nemotron3Diarizer::from_configs(&config, &processor, vb).map(|_| ())
    })
}
