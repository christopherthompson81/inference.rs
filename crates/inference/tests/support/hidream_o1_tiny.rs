//! Builds a tiny random-weight HiDream-O1 checkpoint at test time, under a name that selects the full or Dev variant.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

use inference_models_diffusion::hidream::model::{Config, HiDreamO1};
use inference_nn::vision::qwen3_vl::Qwen3VLVisionModel;
use inference_tensor::quantized::gguf_file::{self, Value};
use inference_tensor::quantized::{GgmlDType, QTensor};

#[path = "recording.rs"]
mod recording;

// through the crates directory, so the server crate's tests resolve it too
const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../inference/tests/fixtures/hidream_o1"
);
const TEXT_LAYERS: usize = 2;
const WEIGHTS: &str = "model.safetensors";
const INDEX: &str = "model.safetensors.index.json";
const COPIED: [&str; 2] = ["config.json", "tokenizer.json"];

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
            // the model builds its ViT on the first edit, so the checkpoint asks for it here
            Qwen3VLVisionModel::new(&cfg.vision_config, vb.pp("model").pp("visual"))?;
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

/// `checkpoint` as a ComfyUI-GGUF HiDream file at `gguf`, every tensor in F32, config and tokenizer beside it.
pub fn tiny_hidream_o1_f32_gguf(checkpoint: &Path, gguf: &Path) -> anyhow::Result<()> {
    write_f32_gguf(checkpoint, gguf, |_| true)
}

/// As [`tiny_hidream_o1_f32_gguf`], less the tensors under `prefix`.
pub fn tiny_hidream_o1_f32_gguf_without(
    checkpoint: &Path,
    gguf: &Path,
    prefix: &str,
) -> anyhow::Result<()> {
    write_f32_gguf(checkpoint, gguf, |name| !name.starts_with(prefix))
}

fn write_f32_gguf(
    checkpoint: &Path,
    gguf: &Path,
    keep: impl Fn(&str) -> bool,
) -> anyhow::Result<()> {
    copy_configs(checkpoint, gguf)?;
    let weights = inference_tensor::safetensors::load(
        checkpoint.join(WEIGHTS),
        &inference_tensor::Device::Cpu,
    )?;
    let mut tensors = Vec::with_capacity(weights.len());
    for (name, t) in weights.into_iter().filter(|(name, _)| keep(name)) {
        let t = t.to_dtype(inference_tensor::DType::F32)?;
        tensors.push((name, QTensor::quantize(&t, GgmlDType::F32)?));
    }
    let metadata = [
        ("general.architecture", Value::String("qwen".to_string())),
        ("general.type", Value::String("diffusion".to_string())),
    ];
    let metadata: Vec<(&str, &Value)> = metadata.iter().map(|(k, v)| (*k, v)).collect();
    let refs: Vec<(&str, &QTensor)> = tensors.iter().map(|(n, q)| (n.as_str(), q)).collect();
    gguf_file::write(
        &mut std::io::BufWriter::new(std::fs::File::create(gguf)?),
        &metadata,
        &refs,
    )?;
    Ok(())
}

/// The config and tokenizer the GGUF `gguf` loads with, copied beside it from `checkpoint`.
pub fn copy_configs(checkpoint: &Path, gguf: &Path) -> anyhow::Result<()> {
    let dir = gguf.parent().expect("the GGUF has a directory");
    for name in COPIED {
        std::fs::copy(checkpoint.join(name), dir.join(name))?;
    }
    Ok(())
}
