//! Repacks a tiny checkpoint directory as audio.cpp's converter lays one out: each source's safetensors under its own
//! names (prefixed `<source>/` when the family has several sources) and the checkpoint's other files embedded.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

use inference_tensor::quantized::GgmlDType;

/// The package of `dir`'s `sources` ((prefix, safetensors path) pairs; an empty prefix for a single source) and
/// `files` (paths relative to `dir`), written as `out` with every tensor kept in F32.
pub fn repack(
    dir: &Path,
    family: &str,
    sources: &[(&str, &str)],
    files: &[&str],
    out: &Path,
) -> anyhow::Result<PathBuf> {
    let mut tensors = Vec::new();
    for (prefix, weights) in sources {
        for (name, t) in
            inference_tensor::safetensors::load(dir.join(weights), &inference_tensor::Device::Cpu)?
        {
            let name = match prefix {
                &"" => name,
                prefix => format!("{prefix}/{name}"),
            };
            tensors.push((name, t));
        }
    }
    tensors.sort_by(|a, b| a.0.cmp(&b.0));
    let stored = tensors
        .iter()
        .map(|(name, t)| (name.as_str(), t, GgmlDType::F32))
        .collect::<Vec<_>>();
    let bytes = files
        .iter()
        .map(|name| std::fs::read(dir.join(name)).map(|bytes| (*name, bytes)))
        .collect::<std::io::Result<Vec<_>>>()?;
    let embedded = bytes
        .iter()
        .map(|(name, bytes)| (*name, bytes.as_slice()))
        .collect::<Vec<_>>();
    inference_gguf::audiocpp::write_package(out, family, &stored, &embedded)?;
    Ok(out.to_path_buf())
}

/// Every file under `dir` but its safetensors, relative to `dir`: what a package embeds.
pub fn other_files(dir: &Path) -> anyhow::Result<Vec<String>> {
    let mut files = Vec::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(next) = pending.pop() {
        for entry in std::fs::read_dir(&next)? {
            let path = entry?.path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_none_or(|e| e != "safetensors") {
                files.push(path.strip_prefix(dir)?.to_string_lossy().replace('\\', "/"));
            }
        }
    }
    files.sort();
    Ok(files)
}

/// [`repack`] with every other file of `dir` embedded.
pub fn repack_all(
    dir: &Path,
    family: &str,
    sources: &[(&str, &str)],
    out: &Path,
) -> anyhow::Result<PathBuf> {
    let files = other_files(dir)?;
    let files = files.iter().map(String::as_str).collect::<Vec<_>>();
    repack(dir, family, sources, &files, out)
}
