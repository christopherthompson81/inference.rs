//! Builds a tiny random-weight Streaming Sortformer `.nemo` at test time.
#![allow(dead_code)]

use std::path::Path;

use inference_models_speech::diarization::{SortformerConfig, SortformerDiarizer};
use inference_models_speech::nemo::nemo_encoder_name;
use inference_tensor::Tensor;

#[path = "recording.rs"]
mod recording;

// through the crates directory, so other crates' tests resolve it too
const CONFIG: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../inference/tests/fixtures/sortformer_tiny/model_config.yaml"
);
pub const CHECKPOINT: &str = "tiny_sortformer.nemo";
const RUNNING_VAR: &str = "running_var";

/// A directory holding one `.nemo` of the committed config and random weights, under NeMo's own names.
pub fn tiny_sortformer() -> anyhow::Result<tempfile::TempDir> {
    let yaml = std::fs::read_to_string(CONFIG)?;
    let config = SortformerConfig::from_yaml(&yaml)?;
    let mut tensors =
        recording::record_tensors(|vb| SortformerDiarizer::new(config, vb).map(|_| ()))?;
    // a batch norm's variance is positive
    for (name, t) in tensors.iter_mut() {
        if name.ends_with(RUNNING_VAR) {
            *t = (t.abs()? + 1.0)?;
        }
    }
    let mut renamed: Vec<(String, Tensor)> = tensors
        .into_iter()
        .map(|(n, t)| (nemo_encoder_name(&n), t))
        .collect();
    renamed.sort_by(|a, b| a.0.cmp(&b.0));
    let named: Vec<(&str, &Tensor)> = renamed.iter().map(|(n, t)| (n.as_str(), t)).collect();
    let dir = tempfile::tempdir()?;
    inference_models_speech::nemo::write_nemo(&dir.path().join(CHECKPOINT), &yaml, &named)?;
    Ok(dir)
}

pub fn checkpoint(dir: &Path) -> std::path::PathBuf {
    dir.join(CHECKPOINT)
}
