//! Builds a tiny random-weight Gemma 3 checkpoint (SigLIP tower, sliding and global text layers) at test time.
#![allow(dead_code)]

use inference_models_gemma::gemma3::{Gemma3Model, config::Gemma3Config};
use inference_nn::paged_attention::AttentionImplementation;

#[path = "recording.rs"]
mod recording;

const GEMMA3: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/gemma3/gemma3");
// From gemma3/config.json, written by make_tiny.py.
const TEXT_LAYERS: usize = 2;

pub fn tiny_gemma3() -> anyhow::Result<tempfile::TempDir> {
    let cfg: Gemma3Config =
        serde_json::from_str(&std::fs::read_to_string(format!("{GEMMA3}/config.json"))?)?;
    let files = recording::fixture_files(GEMMA3)?;
    let files = files.iter().map(|path| path.as_path()).collect::<Vec<_>>();
    recording::record_checkpoint_seeded_by_name(&files, TEXT_LAYERS, &[], |vb, metadata| {
        Gemma3Model::new(&cfg, vb, true, metadata, AttentionImplementation::Eager).map(|_| ())
    })
}
