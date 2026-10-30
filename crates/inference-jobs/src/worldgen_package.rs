//! The world-generation checkpoints laid out for distribution: WorldStereo-2 and umT5 as GGUFs, the other Wan, MoGe-2
//! and LPIPS files as they are, under the models-root layout `JobsConfig::with_models_root` reads. WorldMirror is left
//! out: jobs fetch it from its own repository.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use inference_recon::worldstereo::gguf_writer::{write_umt5_gguf, write_worldstereo_gguf};
use inference_tensor::quantized::GgmlDType;

use crate::{LPIPS_FILE, MOGE_DIR, WAN_DIR, WORLDSTEREO_DIR};
use inference_api::GgufWriteSummary;

const TEXT_ENCODER_DIR: &str = "text_encoder";
// MoGe-2's original pickle, which the converted safetensors replace
const MOGE_KEEP: [&str; 3] = ["config.json", "model.safetensors", "README.md"];

/// One file the package holds, with its GGUF summary when it was written rather than copied.
#[derive(Debug)]
pub struct PackagedFile {
    pub path: PathBuf,
    pub bytes: u64,
    pub gguf: Option<GgufWriteSummary>,
}

/// Lays the checkpoints under `models` (the models-root layout) out under `out`, quantizing WorldStereo-2's main
/// blocks and umT5's matrices to the type named `matrices`; unchanged files are hard-linked where the filesystem allows.
pub fn write_worldgen_package(
    models: &Path,
    out: &Path,
    matrices: &str,
) -> Result<Vec<PackagedFile>> {
    let dtype = GgmlDType::from_name(matrices)?;
    let suffix = dtype
        .name()
        .map_or_else(|| format!("{dtype:?}"), str::to_string)
        .to_uppercase();
    let mut files = Vec::new();

    let worldstereo = models.join(WORLDSTEREO_DIR);
    let ws_out = out.join(WORLDSTEREO_DIR);
    std::fs::create_dir_all(&ws_out)?;
    files.push(place(
        &worldstereo.join("config.json"),
        &ws_out.join("config.json"),
    )?);
    let name = Path::new(WORLDSTEREO_DIR)
        .file_name()
        .context("a checkpoint folder has a name")?;
    let gguf = ws_out.join(format!("{}-{suffix}.gguf", name.to_string_lossy()));
    let written = write_worldstereo_gguf(&worldstereo, &gguf, dtype)?;
    files.push(written_file(gguf, written)?);

    let wan = models.join(WAN_DIR);
    for source in walk(&wan)? {
        let relative = source.strip_prefix(&wan)?;
        let in_text_encoder = relative.starts_with(TEXT_ENCODER_DIR);
        let weights = relative.extension().is_some_and(|e| e == "safetensors")
            || relative
                .to_string_lossy()
                .ends_with(".safetensors.index.json");
        if in_text_encoder && weights {
            continue;
        }
        files.push(place(&source, &out.join(WAN_DIR).join(relative))?);
    }
    let gguf = out
        .join(WAN_DIR)
        .join(TEXT_ENCODER_DIR)
        .join(format!("umt5-xxl-encoder-{suffix}.gguf"));
    let written = write_umt5_gguf(&wan.join(TEXT_ENCODER_DIR), &gguf, dtype)?;
    files.push(written_file(gguf, written)?);

    for name in MOGE_KEEP {
        let source = models.join(MOGE_DIR).join(name);
        if source.exists() {
            files.push(place(&source, &out.join(MOGE_DIR).join(name))?);
        }
    }
    files.push(place(&models.join(LPIPS_FILE), &out.join(LPIPS_FILE))?);
    Ok(files)
}

fn written_file(path: PathBuf, tensors: Vec<(String, GgmlDType)>) -> Result<PackagedFile> {
    let gguf = GgufWriteSummary::of(&path, tensors.into_iter().map(|(_, d)| d))?;
    Ok(PackagedFile {
        bytes: gguf.bytes,
        path,
        gguf: Some(gguf),
    })
}

fn place(source: &Path, dest: &Path) -> Result<PackagedFile> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if dest.exists() {
        std::fs::remove_file(dest)?;
    }
    if std::fs::hard_link(source, dest).is_err() {
        std::fs::copy(source, dest).with_context(|| source.display().to_string())?;
    }
    Ok(PackagedFile {
        path: dest.to_path_buf(),
        bytes: std::fs::metadata(dest)?.len(),
        gguf: None,
    })
}

fn walk(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir).with_context(|| dir.display().to_string())? {
        let path = entry?.path();
        if path.is_dir() {
            files.extend(walk(&path)?);
        } else {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}
