//! Builds tiny random-weight Qwen2-VL and Qwen3-VL checkpoints at test time, for the image and video input paths.
#![allow(dead_code)]

use inference_models_qwen::qwen2vl::{Config as Qwen2VLConfig, Qwen2VLModel};
use inference_models_qwen::qwen3_vl::{Config as Qwen3VLConfig, Qwen3VLModel};
use inference_nn::paged_attention::AttentionImplementation;

#[path = "recording.rs"]
mod recording;

const QWEN2_VL: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/qwen_vl/qwen2_vl"
);
const QWEN3_VL: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/qwen_vl/qwen3_vl"
);
const TEXT_LAYERS: usize = 2;
// The MLX layout the constructors probe for; reporting it absent selects the HF tensor names.
const MLX_PROBES: &[&str] = &[
    "vision_tower.patch_embed.proj.weight",
    "language_model.model.embed_tokens.weight",
];

fn files(dir: &str) -> anyhow::Result<Vec<std::path::PathBuf>> {
    recording::fixture_files(dir)
}

pub fn tiny_qwen2_vl() -> anyhow::Result<tempfile::TempDir> {
    let cfg: Qwen2VLConfig =
        serde_json::from_str(&std::fs::read_to_string(format!("{QWEN2_VL}/config.json"))?)?;
    let files = files(QWEN2_VL)?;
    let files = files.iter().map(|path| path.as_path()).collect::<Vec<_>>();
    recording::record_checkpoint(&files, TEXT_LAYERS, MLX_PROBES, |vb, metadata| {
        Qwen2VLModel::new(&cfg, vb, true, metadata, AttentionImplementation::Eager).map(|_| ())
    })
}

pub fn tiny_qwen3_vl() -> anyhow::Result<tempfile::TempDir> {
    let cfg: Qwen3VLConfig =
        serde_json::from_str(&std::fs::read_to_string(format!("{QWEN3_VL}/config.json"))?)?;
    let files = files(QWEN3_VL)?;
    let files = files.iter().map(|path| path.as_path()).collect::<Vec<_>>();
    recording::record_checkpoint(&files, TEXT_LAYERS, MLX_PROBES, |vb, metadata| {
        Qwen3VLModel::new(&cfg, vb, true, metadata, AttentionImplementation::Eager).map(|_| ())
    })
}
