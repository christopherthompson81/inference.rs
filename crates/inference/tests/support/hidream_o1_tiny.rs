//! Builds a tiny random-weight HiDream-O1 checkpoint at test time, under a name that selects the full or Dev variant.
#![allow(dead_code)]

use std::path::PathBuf;

use inference_models_diffusion::hidream::model::{Config, HiDreamO1};

#[path = "recording.rs"]
mod recording;

const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/hidream_o1");
const TEXT_LAYERS: usize = 2;
const WEIGHTS: &str = "model.safetensors";
const INDEX: &str = "model.safetensors.index.json";

/// The checkpoint in a temporary directory, and its path, named `name` (a `-dev` suffix selects Dev).
pub fn tiny_hidream_o1(name: &str) -> anyhow::Result<(tempfile::TempDir, PathBuf)> {
    let cfg = Config::from_json(&std::fs::read_to_string(format!("{FIXTURE}/config.json"))?)?;
    let files = recording::fixture_files(FIXTURE)?
        .into_iter()
        .filter(|path| path.extension().is_some_and(|e| e == "json"))
        .collect::<Vec<_>>();
    let files = files.iter().map(|path| path.as_path()).collect::<Vec<_>>();
    let recorded =
        recording::record_checkpoint_seeded_by_name(&files, TEXT_LAYERS, &[], |vb, metadata| {
            HiDreamO1::new(&cfg, vb, metadata).map(|_| ())
        })?;
    let parent = tempfile::tempdir()?;
    let dir = parent.path().join(name);
    std::fs::rename(recorded.path(), &dir)?;
    // the index is what tells HiDream-O1 from a Qwen3-VL chat model
    let names =
        inference_tensor::safetensors::load(dir.join(WEIGHTS), &inference_tensor::Device::Cpu)?
            .into_keys()
            .map(|name| (name, serde_json::Value::from(WEIGHTS)))
            .collect::<serde_json::Map<_, _>>();
    std::fs::write(
        dir.join(INDEX),
        serde_json::json!({ "weight_map": names }).to_string(),
    )?;
    Ok((parent, dir))
}
