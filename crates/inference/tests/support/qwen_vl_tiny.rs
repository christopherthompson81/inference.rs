//! Builds tiny random-weight Qwen2-VL and Qwen3-VL checkpoints at test time, for the image and video input paths.
#![allow(dead_code)]

use inference_models_qwen::qwen2vl::{Config as Qwen2VLConfig, Qwen2VLModel};
use inference_models_qwen::qwen3_5::{Config as Qwen3_5Config, Qwen3_5Model};
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
const QWEN3_5_MOE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/qwen_vl/qwen3_5_moe"
);
const TEXT_LAYERS: usize = 2;
// From qwen3_5_moe/config.json, written by make_tiny.py.
const MOE_LAYERS: usize = 4;
const MOE_EXPERTS: usize = 4;
const MOE_HIDDEN: usize = 128;
// moe_gemm takes expert widths in multiples of 64
const MOE_INTERMEDIATE: usize = 64;
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

// HF stores the Qwen3.5 MoE experts one by one; the layout detection reads these shapes before any tensor.
fn qwen3_5_moe_expert_shapes() -> std::collections::HashMap<String, Vec<usize>> {
    (0..MOE_LAYERS)
        .flat_map(|layer| (0..MOE_EXPERTS).map(move |expert| (layer, expert)))
        .flat_map(|(layer, expert)| {
            let p = format!("model.language_model.layers.{layer}.mlp.experts.{expert}");
            [
                (
                    format!("{p}.gate_proj.weight"),
                    vec![MOE_INTERMEDIATE, MOE_HIDDEN],
                ),
                (
                    format!("{p}.up_proj.weight"),
                    vec![MOE_INTERMEDIATE, MOE_HIDDEN],
                ),
                (
                    format!("{p}.down_proj.weight"),
                    vec![MOE_HIDDEN, MOE_INTERMEDIATE],
                ),
            ]
        })
        .collect()
}

pub fn tiny_qwen3_5_moe() -> anyhow::Result<tempfile::TempDir> {
    let cfg: Qwen3_5Config = serde_json::from_str(&std::fs::read_to_string(format!(
        "{QWEN3_5_MOE}/config.json"
    ))?)?;
    let files = files(QWEN3_5_MOE)?;
    let files = files.iter().map(|path| path.as_path()).collect::<Vec<_>>();
    recording::record_checkpoint_with_shapes(
        &files,
        MOE_LAYERS,
        MLX_PROBES,
        qwen3_5_moe_expert_shapes(),
        |vb, metadata| {
            Qwen3_5Model::new(&cfg, vb, true, metadata, AttentionImplementation::Eager).map(|_| ())
        },
    )
}
