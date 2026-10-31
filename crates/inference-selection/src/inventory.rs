//! The models on this machine, in local folders and the Hugging Face cache, inspected without loading them.

use std::{
    collections::{BTreeMap, HashMap},
    fs,
    io::BufReader,
    path::{Path, PathBuf},
    time::UNIX_EPOCH,
};

use anyhow::{Context, Result, bail};
use inference_core::{DetectedModelKind, default_hf_cache_dir, detect_local_model};
use inference_gguf::base_model::source_hf_base_model_id;
use inference_nn::paged_attention::{KvFingerprint, kv_profile_name};
use inference_quant::GgufDType;
use inference_tensor::quantized::gguf_file::{self, Value};
use serde::Serialize;

use crate::quant::{
    is_auxiliary_gguf, is_gguf, is_projector, parse_split, projector_belongs, quant_label,
};

const MODEL_DIR_MARKERS: [&str; 3] = ["config.json", "model_index.json", "params.json"];
const NEMO_EXTENSION: &str = "nemo";
// a checkpoint directory's weights; GGUFs beside a config are listed as files, each its own model
const WEIGHT_EXTENSIONS: [&str; 5] = ["safetensors", NEMO_EXTENSION, "bin", "pth", "pt"];
// org/model/variant layouts nest a few levels; a deeper tree is not a model library
const SCAN_MAX_DEPTH: usize = 6;
const HF_REPO_DIR_PREFIX: &str = "models--";
const HF_REPO_SEPARATOR: &str = "--";
const HF_DEFAULT_REF: &str = "main";
const PROJECTOR_GGUF_TYPE: &str = "mmproj";
const IMATRIX_GGUF_TYPE: &str = "imatrix";
const PROJECTOR_GGUF_ARCHITECTURE: &str = "clip";
// inference_layout's `model_type` and GGUF `general.architecture`; that crate's kernels are too heavy to depend on here
const LAYOUT_ARCHITECTURE: &str = "pp_doclayout_v3";
// `architectures` of the DFlash drafter checkpoints (`DFlashDraftModel`, `DFlash2DraftModel`)
const DRAFTER_ARCHITECTURE_PREFIX: &str = "DFlash";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelFormat {
    Gguf,
    Safetensors,
    Nemo,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InventoryKind {
    Text,
    Multimodal,
    Embedding,
    ImageGeneration,
    Speech,
    Transcription,
    VoiceActivity,
    Diarization,
    /// A multimodal projector (mmproj GGUF), loaded beside a text GGUF rather than on its own.
    Projector,
    /// A document layout detector (PP-DocLayoutV3), loaded apart from the engine.
    Layout,
    /// A speculative decoding drafter (DFlash), attached to the model it drafts for (`mtp.model`).
    Drafter,
}

impl From<DetectedModelKind> for InventoryKind {
    fn from(kind: DetectedModelKind) -> Self {
        match kind {
            DetectedModelKind::Text => Self::Text,
            DetectedModelKind::Multimodal => Self::Multimodal,
            DetectedModelKind::Embedding => Self::Embedding,
            DetectedModelKind::ImageGeneration => Self::ImageGeneration,
            DetectedModelKind::Speech => Self::Speech,
            DetectedModelKind::Transcription => Self::Transcription,
            DetectedModelKind::VoiceActivity => Self::VoiceActivity,
            DetectedModelKind::Diarization => Self::Diarization,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InventorySource {
    Local,
    HfCache,
}

/// One loadable model: a model directory, a GGUF (all its split shards) or a `.nemo` file.
#[derive(Debug, Clone, Serialize)]
pub struct InventoryModel {
    /// The file or directory a spec names to load it.
    pub path: String,
    pub source: InventorySource,
    /// The Hugging Face repository, for a model in the cache.
    pub repo_id: Option<String>,
    pub format: ModelFormat,
    /// How it loads; unset when it could not be inspected.
    pub kind: Option<InventoryKind>,
    /// The loader a kind with several picks (`qwen3_5`, `flux`, `dia`).
    pub loader: Option<String>,
    pub architecture: Option<String>,
    pub name: Option<String>,
    /// The parameter count its metadata states (`27B`).
    pub size_label: Option<String>,
    pub quantization: Option<String>,
    pub context_length: Option<u64>,
    /// The Hugging Face model it derives from.
    pub base_model: Option<String>,
    /// The measured KV cache profile a quantized cache on it follows (its shape matched one); unset, the general rule.
    pub kv_profile: Option<String>,
    pub size_bytes: u64,
    /// Last modification, in Unix seconds.
    pub modified: Option<u64>,
    /// A GGUF's files, its split shards in order.
    pub files: Vec<String>,
    /// For a text GGUF, the projector GGUFs beside it made for the same base model.
    pub projectors: Vec<String>,
    /// Why it could not be fully inspected.
    pub error: Option<String>,
}

#[derive(Debug, Clone, Default, inference_json::FromJson)]
#[serde(deny_unknown_fields)]
pub struct ScanRequest {
    /// Folders searched for models, a few levels deep.
    #[serde(default)]
    pub folders: Vec<PathBuf>,
    /// Also list the Hugging Face cache.
    #[serde(default)]
    pub hf_cache: bool,
    /// The cache to list instead of the environment's (`HF_HUB_CACHE`, `HF_HOME/hub`).
    #[serde(default)]
    pub hf_cache_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ScanError {
    pub path: String,
    pub error: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct HfCacheRepo {
    pub repo_id: String,
    /// Every revision's files.
    pub size_bytes: u64,
    pub modified: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct HfCacheListing {
    pub path: String,
    pub repos: Vec<HfCacheRepo>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelScan {
    pub models: Vec<InventoryModel>,
    /// Present when the request asked for the cache.
    pub hf_cache: Option<HfCacheListing>,
    /// Folders that could not be read.
    pub errors: Vec<ScanError>,
}

#[derive(Debug, Clone, Serialize)]
pub struct HfCacheDeletion {
    pub repo_id: String,
    pub freed_bytes: u64,
}

#[derive(Clone, Copy)]
struct Origin<'a> {
    source: InventorySource,
    repo_id: Option<&'a str>,
}

/// Inspects one model directory, GGUF (a split GGUF by any shard) or `.nemo` file.
pub fn inspect_model(path: &Path) -> Result<InventoryModel> {
    let origin = Origin {
        source: InventorySource::Local,
        repo_id: None,
    };
    if path.is_dir() {
        return Ok(dir_model(path, origin));
    }
    if !path.is_file() {
        bail!("`{}` is not a file or directory", path.display());
    }
    // a bare file name's parent is "", which lists nothing
    let dir = path
        .parent()
        .filter(|dir| !dir.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    if has_extension(path, NEMO_EXTENSION) {
        return Ok(nemo_model(path, origin));
    }
    if !is_gguf(&path.to_string_lossy()) {
        bail!(
            "`{}` is not a model directory, GGUF or `.nemo`",
            path.display()
        );
    }
    let siblings = files_in(dir)?;
    let groups = gguf_groups(&siblings);
    let group = groups
        .into_iter()
        .find(|group| {
            group
                .iter()
                .any(|shard| shard.file_name() == path.file_name())
        })
        .context("the GGUF is an auxiliary file (LoRA, MTP or DFlash), not a model")?;
    gguf_models_in(&[group], &siblings, origin)
        .pop()
        .context("the GGUF is an importance matrix, data for quantizing rather than a model")
}

/// Lists the models in `request`'s folders and, when asked, the Hugging Face cache.
pub fn scan_models(request: &ScanRequest) -> ModelScan {
    let mut models = BTreeMap::new();
    let mut errors = Vec::new();
    let local = Origin {
        source: InventorySource::Local,
        repo_id: None,
    };
    for folder in &request.folders {
        if !folder.is_dir() {
            errors.push(ScanError {
                path: folder.to_string_lossy().into_owned(),
                error: "not a directory".to_string(),
            });
            continue;
        }
        scan_dir(folder, 0, local, &mut models, &mut errors);
    }
    let hf_cache = request
        .hf_cache
        .then(|| scan_hf_cache(request.hf_cache_path.as_deref(), &mut models, &mut errors));
    ModelScan {
        models: models.into_values().collect(),
        hf_cache: hf_cache.flatten(),
        errors,
    }
}

/// The repositories in the Hugging Face cache (`hub`, or the environment's), each with all its revisions' size.
pub fn hf_cache_listing(hub: Option<&Path>) -> Option<HfCacheListing> {
    let hub = hub.map(Path::to_path_buf).or_else(default_hf_cache_dir)?;
    let mut repos = repo_dirs(&hub)
        .into_iter()
        .map(|(repo_id, dir)| HfCacheRepo {
            repo_id,
            size_bytes: tree_size(&dir),
            modified: modified(&dir),
        })
        .collect::<Vec<_>>();
    repos.sort_by(|a, b| a.repo_id.cmp(&b.repo_id));
    Some(HfCacheListing {
        path: hub.to_string_lossy().into_owned(),
        repos,
    })
}

/// Deletes `repo_id` (`org/name`) from the Hugging Face cache (`hub`, or the environment's), all revisions.
pub fn delete_hf_cache_repo(repo_id: &str, hub: Option<&Path>) -> Result<HfCacheDeletion> {
    let mut parts = repo_id.split('/');
    let valid_part = |part: Option<&str>| {
        part.is_some_and(|part| {
            !part.is_empty()
                && !part.starts_with('.')
                && !part.contains(HF_REPO_SEPARATOR)
                && part
                    .chars()
                    .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.'))
        })
    };
    // `org/name`, or a legacy repository's bare `name` (`gpt2`)
    let first = parts.next();
    let second = parts.next();
    if !valid_part(first)
        || second.is_some_and(|name| !valid_part(Some(name)))
        || parts.next().is_some()
    {
        bail!("`{repo_id}` is not a Hugging Face repository id (`org/name`)");
    }
    let hub = hub
        .map(Path::to_path_buf)
        .or_else(default_hf_cache_dir)
        .context("cannot determine the Hugging Face cache directory")?;
    let dir = hub.join(repo_dir_name(repo_id));
    if !dir.is_dir() {
        bail!("`{repo_id}` is not in the Hugging Face cache");
    }
    let freed_bytes = tree_size(&dir);
    fs::remove_dir_all(&dir).with_context(|| format!("cannot delete `{}`", dir.display()))?;
    Ok(HfCacheDeletion {
        repo_id: repo_id.to_string(),
        freed_bytes,
    })
}

fn repo_dir_name(repo_id: &str) -> String {
    format!(
        "{HF_REPO_DIR_PREFIX}{}",
        repo_id.replace('/', HF_REPO_SEPARATOR)
    )
}

fn repo_dirs(hub: &Path) -> Vec<(String, PathBuf)> {
    let Ok(entries) = fs::read_dir(hub) else {
        return Vec::new();
    };
    entries
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            let name = entry.file_name().into_string().ok()?;
            let repo = name.strip_prefix(HF_REPO_DIR_PREFIX)?;
            let repo_id = repo.replacen(HF_REPO_SEPARATOR, "/", 1);
            entry.path().is_dir().then(|| (repo_id, entry.path()))
        })
        .collect()
}

fn scan_hf_cache(
    hub: Option<&Path>,
    models: &mut BTreeMap<String, InventoryModel>,
    errors: &mut Vec<ScanError>,
) -> Option<HfCacheListing> {
    let listing = hf_cache_listing(hub)?;
    for (repo_id, dir) in repo_dirs(Path::new(&listing.path)) {
        let Some(snapshot) = current_snapshot(&dir) else {
            continue;
        };
        let origin = Origin {
            source: InventorySource::HfCache,
            repo_id: Some(&repo_id),
        };
        scan_dir(&snapshot, 0, origin, models, errors);
    }
    Some(listing)
}

// the snapshot `refs/main` names, else the newest one
fn current_snapshot(repo_dir: &Path) -> Option<PathBuf> {
    let snapshots = repo_dir.join("snapshots");
    if let Ok(commit) = fs::read_to_string(repo_dir.join("refs").join(HF_DEFAULT_REF)) {
        let snapshot = snapshots.join(commit.trim());
        if snapshot.is_dir() {
            return Some(snapshot);
        }
    }
    fs::read_dir(&snapshots)
        .ok()?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_dir())
        .max_by_key(|path| modified(path))
}

fn scan_dir(
    dir: &Path,
    depth: usize,
    origin: Origin<'_>,
    models: &mut BTreeMap<String, InventoryModel>,
    errors: &mut Vec<ScanError>,
) {
    let files = match files_in(dir) {
        Ok(files) => files,
        Err(error) => {
            errors.push(ScanError {
                path: dir.to_string_lossy().into_owned(),
                error: error.to_string(),
            });
            return;
        }
    };
    let mut found = Vec::new();
    let model_dir = MODEL_DIR_MARKERS.iter().any(|marker| {
        files
            .iter()
            .any(|file| file.file_name() == Some(marker.as_ref()))
    });
    // a config without checkpoint weights (a cache entry only probed, a GGUF repo's config) is no checkpoint
    let checkpoint = model_dir && walk_files(dir).iter().any(|file| is_weight_file(file));
    if checkpoint {
        found.push(dir_model(dir, origin));
    }
    found.extend(gguf_models_in(&gguf_groups(&files), &files, origin));
    found.extend(
        files
            .iter()
            .filter(|file| has_extension(file, NEMO_EXTENSION))
            .map(|file| nemo_model(file, origin)),
    );
    for model in found {
        models.entry(model.path.clone()).or_insert(model);
    }
    // a model directory's subfolders are its parts (a diffusers text_encoder), not more models
    if checkpoint || depth >= SCAN_MAX_DEPTH {
        return;
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut subdirs = entries
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry.file_type().is_ok_and(|kind| kind.is_dir())
                && !entry.file_name().to_string_lossy().starts_with('.')
        })
        .map(|entry| entry.path())
        .collect::<Vec<_>>();
    subdirs.sort();
    for subdir in subdirs {
        scan_dir(&subdir, depth + 1, origin, models, errors);
    }
}

// files directly in `dir`, following links (a cache snapshot links its files to blobs)
fn files_in(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = fs::read_dir(dir)
        .with_context(|| format!("cannot list `{}`", dir.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_file())
        .collect::<Vec<_>>();
    files.sort();
    Ok(files)
}

// a directory's GGUF models, split shards grouped in order; auxiliary GGUFs (imatrix, LoRA) are not models
fn gguf_groups(files: &[PathBuf]) -> Vec<Vec<PathBuf>> {
    let mut groups: BTreeMap<String, Vec<(usize, PathBuf)>> = BTreeMap::new();
    for file in files {
        let name = file_name(file);
        if !is_gguf(&name) || is_auxiliary_gguf(&name) {
            continue;
        }
        let (key, index) = match parse_split(&name) {
            Some(split) => (split.prefix, split.index),
            None => (name, 0),
        };
        groups.entry(key).or_default().push((index, file.clone()));
    }
    groups
        .into_values()
        .map(|mut shards| {
            shards.sort_by_key(|(index, _)| *index);
            shards.into_iter().map(|(_, file)| file).collect()
        })
        .collect()
}

struct GgufFacts {
    architecture: Option<String>,
    name: Option<String>,
    size_label: Option<String>,
    context_length: Option<u64>,
    base_model: Option<String>,
    projector: bool,
    /// An importance matrix (llama.cpp's `general.type = imatrix`), data for quantizing rather than a model.
    imatrix: bool,
    dominant_dtype: Option<String>,
    kv_profile: Option<String>,
}

// a GGUF's shape as the loaded model reports it: block_count counts the MTP layers, a hybrid may list KV heads per layer
fn gguf_kv_fingerprint(metadata: &HashMap<String, Value>, arch: &str) -> Option<KvFingerprint> {
    let count = |key: &str| {
        let value = metadata.get(&format!("{arch}.{key}"))?;
        match value.to_vec() {
            Ok(values) => values.iter().filter_map(|v| v.to_u64().ok()).max(),
            Err(_) => value.to_u64().ok(),
        }
        .map(|n| n as usize)
    };
    let hidden = count("embedding_length")?;
    let heads = count("attention.head_count")?;
    Some(KvFingerprint {
        layers: count("block_count")? - count("nextn_predict_layers").unwrap_or(0),
        hidden,
        heads,
        kv_heads: count("attention.head_count_kv").unwrap_or(heads),
        head_dim: count("attention.key_length").unwrap_or(hidden / heads),
    })
}

fn config_kv_fingerprint(config: &serde_json::Value) -> Option<KvFingerprint> {
    let text = if config["text_config"].is_object() {
        &config["text_config"]
    } else {
        config
    };
    let count = |key: &str| text[key].as_u64().map(|n| n as usize);
    let hidden = count("hidden_size")?;
    let heads = count("num_attention_heads")?;
    Some(KvFingerprint {
        layers: count("num_hidden_layers")?,
        hidden,
        heads,
        kv_heads: count("num_key_value_heads").unwrap_or(heads),
        head_dim: count("head_dim").unwrap_or(hidden / heads),
    })
}

fn gguf_facts(shards: &[PathBuf]) -> Result<GgufFacts> {
    let headers = shards
        .iter()
        .map(|shard| {
            let mut reader = BufReader::new(
                fs::File::open(shard)
                    .with_context(|| format!("cannot open `{}`", shard.display()))?,
            );
            gguf_file::Header::read_summary(&mut reader)
                .with_context(|| format!("cannot read the GGUF header of `{}`", shard.display()))
        })
        .collect::<Result<Vec<_>>>()?;
    let metadata = &headers[0].metadata;
    let architecture = meta_string(metadata, "general.architecture");
    let context_length = architecture
        .as_deref()
        .and_then(|arch| metadata.get(&format!("{arch}.context_length")))
        .and_then(|value| value.to_u64().ok());
    let label = file_name(&shards[0]);
    let base_model = source_hf_base_model_id(&label, metadata).ok().flatten();
    let gguf_type = meta_string(metadata, "general.type");
    let projector = is_projector(&label)
        || gguf_type.as_deref() == Some(PROJECTOR_GGUF_TYPE)
        || architecture.as_deref() == Some(PROJECTOR_GGUF_ARCHITECTURE);
    let imatrix = gguf_type.as_deref() == Some(IMATRIX_GGUF_TYPE);
    let mut bytes_by_dtype: HashMap<&'static str, usize> = HashMap::new();
    for tensor in headers
        .iter()
        .flat_map(|header| header.tensor_infos.values())
    {
        let dims = tensor.shape.dims();
        let (Some((&ne0, rows)), true) = (dims.split_last(), dims.len() >= 2) else {
            continue;
        };
        let dtype = GgufDType::new(tensor.ggml_dtype);
        let bytes = dtype.row_size(ne0).unwrap_or(0) * rows.iter().product::<usize>();
        *bytes_by_dtype.entry(dtype.name()).or_default() += bytes;
    }
    let dominant_dtype = bytes_by_dtype
        .into_iter()
        .max_by_key(|(name, bytes)| (*bytes, *name))
        .map(|(name, _)| name.to_string());
    let kv_profile = architecture
        .as_deref()
        .and_then(|arch| gguf_kv_fingerprint(metadata, arch))
        .and_then(|fingerprint| kv_profile_name(&fingerprint))
        .map(str::to_string);
    Ok(GgufFacts {
        kv_profile,
        name: meta_string(metadata, "general.name"),
        size_label: meta_string(metadata, "general.size_label"),
        architecture,
        context_length,
        base_model,
        projector,
        imatrix,
        dominant_dtype,
    })
}

fn meta_string(metadata: &HashMap<String, Value>, key: &str) -> Option<String> {
    metadata
        .get(key)
        .and_then(|value| value.to_string().ok())
        .cloned()
}

// `groups` are models among `siblings`, the files of their directory, whose projectors they may own
fn gguf_models_in(
    groups: &[Vec<PathBuf>],
    siblings: &[PathBuf],
    origin: Origin<'_>,
) -> Vec<InventoryModel> {
    let mut facts = gguf_groups(siblings)
        .into_iter()
        .map(|shards| {
            let facts = gguf_facts(&shards).map_err(|error| error.to_string());
            (shards, facts)
        })
        .collect::<Vec<_>>();
    let sibling_projectors = facts
        .iter()
        .filter_map(|(shards, facts)| {
            let facts = facts.as_ref().ok()?;
            facts
                .projector
                .then(|| (shards[0].clone(), facts.base_model.clone()))
        })
        .collect::<Vec<_>>();
    let facts = groups
        .iter()
        .map(|shards| {
            let index = facts
                .iter()
                .position(|(sibling, _)| sibling == shards)
                .expect("each group is one of its directory's GGUF groups");
            facts.swap_remove(index).1
        })
        .collect::<Vec<_>>();
    groups
        .iter()
        .zip(facts)
        .filter(|(_, facts)| !facts.as_ref().is_ok_and(|facts| facts.imatrix))
        .map(|(shards, facts)| {
            let mut model = base_model_entry(shards[0].clone(), ModelFormat::Gguf, origin);
            model.files = shards
                .iter()
                .map(|shard| shard.to_string_lossy().into_owned())
                .collect();
            model.size_bytes = shards.iter().map(|shard| file_size(shard)).sum();
            model.modified = shards.iter().filter_map(|shard| modified(shard)).max();
            let facts = match facts {
                Ok(facts) => facts,
                Err(error) => {
                    model.error = Some(error);
                    return model;
                }
            };
            let stem = file_name(&shards[0]);
            model.quantization =
                quant_label(stem.trim_end_matches(".gguf")).or(facts.dominant_dtype);
            if facts.projector {
                model.kind = Some(InventoryKind::Projector);
            } else if facts.architecture.as_deref() == Some(LAYOUT_ARCHITECTURE) {
                model.kind = Some(InventoryKind::Layout);
            } else {
                // a HiDream GGUF names its Qwen3-VL backbone's text architecture: detection routes it to diffusion
                let hidream = inference_core::DiffusionLoaderType::auto_detect_hidream_gguf(
                    &shards[0].to_string_lossy(),
                )
                .is_some();
                let text = !hidream
                    && facts
                        .architecture
                        .as_deref()
                        .is_some_and(inference_gguf::loads_architecture);
                // other loadable text architectures are never one of the GGUFs detection routes elsewhere
                if text {
                    model.kind = Some(InventoryKind::Text);
                } else if let Ok(detected) = detect_local_model(&shards[0]) {
                    model.kind = Some(detected.kind.into());
                    model.loader = detected.loader;
                } else {
                    model.error = Some(match facts.architecture.as_deref() {
                        Some(arch) => format!("unsupported GGUF architecture `{arch}`"),
                        None => "no `general.architecture`".to_string(),
                    });
                }
                // the loader's own rule, so what this reports is what a load beside them picks from
                let model_base = facts.base_model.as_deref();
                let any_match = sibling_projectors.iter().any(|(_, base)| {
                    model_base.is_some_and(|model| {
                        base.as_deref()
                            .is_some_and(|base| base.eq_ignore_ascii_case(model))
                    })
                });
                model.projectors = sibling_projectors
                    .iter()
                    .filter(|(_, base)| projector_belongs(model_base, base.as_deref(), any_match))
                    .map(|(path, _)| path.to_string_lossy().into_owned())
                    .collect();
            }
            model.architecture = facts.architecture;
            model.name = facts.name;
            model.size_label = facts.size_label;
            model.context_length = facts.context_length;
            model.base_model = facts.base_model;
            model.kv_profile = facts.kv_profile;
            model
        })
        .collect()
}

fn nemo_model(path: &Path, origin: Origin<'_>) -> InventoryModel {
    let mut model = base_model_entry(path.to_path_buf(), ModelFormat::Nemo, origin);
    model.files = vec![path.to_string_lossy().into_owned()];
    model.size_bytes = file_size(path);
    model.modified = modified(path);
    model.name = path
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned());
    apply_detection(&mut model, path);
    model
}

fn dir_model(dir: &Path, origin: Origin<'_>) -> InventoryModel {
    let files = walk_files(dir);
    let format = if files.iter().any(|file| has_extension(file, "safetensors")) {
        ModelFormat::Safetensors
    } else if files.iter().any(|file| is_gguf(&file.to_string_lossy())) {
        ModelFormat::Gguf
    } else if files.iter().any(|file| has_extension(file, NEMO_EXTENSION)) {
        ModelFormat::Nemo
    } else {
        ModelFormat::Other
    };
    let mut model = base_model_entry(dir.to_path_buf(), format, origin);
    model.size_bytes = files.iter().map(|file| file_size(file)).sum();
    model.modified = modified(dir);
    model.name = origin.repo_id.map(str::to_string).or_else(|| {
        dir.file_name()
            .map(|name| name.to_string_lossy().into_owned())
    });
    if let Ok(config) = fs::read_to_string(dir.join("config.json"))
        && let Ok(config) = serde_json::from_str::<serde_json::Value>(&config)
    {
        model.architecture = config["architectures"][0]
            .as_str()
            .or_else(|| config["model_type"].as_str())
            .map(str::to_string);
        model.context_length = config["max_position_embeddings"]
            .as_u64()
            .or_else(|| config["text_config"]["max_position_embeddings"].as_u64());
        model.quantization = config["quantization_config"]["quant_method"]
            .as_str()
            .map(str::to_string);
        model.kv_profile = config_kv_fingerprint(&config)
            .and_then(|fingerprint| kv_profile_name(&fingerprint))
            .map(str::to_string);
        if config["model_type"].as_str() == Some(LAYOUT_ARCHITECTURE) {
            model.kind = Some(InventoryKind::Layout);
            return model;
        }
        if model
            .architecture
            .as_deref()
            .is_some_and(|arch| arch.starts_with(DRAFTER_ARCHITECTURE_PREFIX))
        {
            model.kind = Some(InventoryKind::Drafter);
            return model;
        }
    }
    apply_detection(&mut model, dir);
    if model.kind.is_none()
        && let Some(arch) = &model.architecture
    {
        model.error = Some(format!("unsupported architecture `{arch}`"));
    }
    model
}

fn apply_detection(model: &mut InventoryModel, path: &Path) {
    match detect_local_model(path) {
        Ok(detected) => {
            model.kind = Some(detected.kind.into());
            model.loader = detected.loader;
        }
        Err(error) => model.error = Some(error.to_string()),
    }
}

fn base_model_entry(path: PathBuf, format: ModelFormat, origin: Origin<'_>) -> InventoryModel {
    InventoryModel {
        path: path.to_string_lossy().into_owned(),
        source: origin.source,
        repo_id: origin.repo_id.map(str::to_string),
        format,
        kind: None,
        loader: None,
        architecture: None,
        name: None,
        size_label: None,
        quantization: None,
        context_length: None,
        base_model: None,
        kv_profile: None,
        size_bytes: 0,
        modified: None,
        files: Vec::new(),
        projectors: Vec::new(),
        error: None,
    }
}

fn walk_files(dir: &Path) -> Vec<PathBuf> {
    walkdir::WalkDir::new(dir)
        .follow_links(true)
        .into_iter()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().is_file())
        .map(|entry| entry.into_path())
        .collect()
}

// the bytes on disk, counting a cache's blobs once however many snapshots link them
fn tree_size(dir: &Path) -> u64 {
    walkdir::WalkDir::new(dir)
        .into_iter()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().is_file())
        .filter_map(|entry| entry.metadata().ok())
        .map(|metadata| metadata.len())
        .sum()
}

fn file_size(path: &Path) -> u64 {
    fs::metadata(path).map_or(0, |metadata| metadata.len())
}

fn modified(path: &Path) -> Option<u64> {
    fs::metadata(path)
        .ok()?
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|duration| duration.as_secs())
}

fn is_weight_file(path: &Path) -> bool {
    WEIGHT_EXTENSIONS
        .iter()
        .any(|extension| has_extension(path, extension))
}

// the discovery helpers read Hub-style relative names: a folder above the model must not count as a token
fn file_name(path: &Path) -> String {
    path.file_name()
        .map_or_else(String::new, |name| name.to_string_lossy().into_owned())
}

fn has_extension(path: &Path, extension: &str) -> bool {
    path.extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case(extension))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_gguf(path: &Path, metadata: &[(&str, Value)]) {
        let metadata = metadata
            .iter()
            .map(|(key, value)| (*key, value))
            .collect::<Vec<_>>();
        let mut file = fs::File::create(path).unwrap();
        gguf_file::write(&mut file, &metadata, &[]).unwrap();
    }

    fn text(value: &str) -> Value {
        Value::String(value.to_string())
    }

    fn base_model(id: &str) -> (&'static str, Value) {
        (
            "general.base_model.0.repo_url",
            text(&format!("https://huggingface.co/{id}")),
        )
    }

    fn by_name<'a>(scan: &'a ModelScan, name: &str) -> &'a InventoryModel {
        scan.models
            .iter()
            .find(|model| model.path.ends_with(name))
            .unwrap_or_else(|| panic!("`{name}` is missing from {:?}", scan.models))
    }

    #[test]
    fn a_hidream_gguf_is_an_image_model_despite_its_text_backbone() {
        use inference_tensor::{
            Device, Tensor,
            quantized::{GgmlDType, QTensor},
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("HiDream-O1-Dev-Q8_0.gguf");
        let patch_embed = QTensor::quantize(
            &Tensor::zeros((2, 2), inference_tensor::DType::F32, &Device::Cpu).unwrap(),
            GgmlDType::F32,
        )
        .unwrap();
        let metadata = [
            ("general.architecture", text("qwen3vl")),
            ("general.type", text("diffusion")),
        ];
        let metadata = metadata
            .iter()
            .map(|(key, value)| (*key, value))
            .collect::<Vec<_>>();
        let mut file = fs::File::create(&path).unwrap();
        gguf_file::write(
            &mut file,
            &metadata,
            &[("model.x_embedder.proj1.weight", &patch_embed)],
        )
        .unwrap();
        let scan = scan_models(&ScanRequest {
            folders: vec![dir.path().to_path_buf()],
            ..Default::default()
        });
        let model = by_name(&scan, "HiDream-O1-Dev-Q8_0.gguf");
        assert_eq!(model.kind, Some(InventoryKind::ImageGeneration));
        assert_eq!(model.loader.as_deref(), Some("hidream-o1-dev"));
    }

    #[test]
    fn a_models_shape_names_its_kv_profile() {
        let dir = tempfile::tempdir().unwrap();
        // the 27B hybrid's shape, with the MTP layer block_count counts
        write_gguf(
            &dir.path().join("hybrid-IQ4_XS.gguf"),
            &[
                ("general.architecture", text("qwen35")),
                ("qwen35.block_count", Value::U32(65)),
                ("qwen35.nextn_predict_layers", Value::U32(1)),
                ("qwen35.embedding_length", Value::U32(5120)),
                ("qwen35.attention.head_count", Value::U32(24)),
                ("qwen35.attention.head_count_kv", Value::U32(4)),
                ("qwen35.attention.key_length", Value::U32(256)),
            ],
        );
        write_gguf(
            &dir.path().join("unmeasured-Q8_0.gguf"),
            &[
                ("general.architecture", text("qwen3")),
                ("qwen3.block_count", Value::U32(30)),
                ("qwen3.embedding_length", Value::U32(1024)),
                ("qwen3.attention.head_count", Value::U32(16)),
            ],
        );
        let checkpoint = dir.path().join("small-dense");
        fs::create_dir(&checkpoint).unwrap();
        fs::write(
            checkpoint.join("config.json"),
            r#"{"architectures": ["Qwen3ForCausalLM"], "num_hidden_layers": 28, "hidden_size": 1024,
                "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128}"#,
        )
        .unwrap();
        fs::write(checkpoint.join("model.safetensors"), b"").unwrap();
        let scan = scan_models(&ScanRequest {
            folders: vec![dir.path().to_path_buf()],
            ..Default::default()
        });
        let profile = |name| by_name(&scan, name).kv_profile.as_deref();
        assert_eq!(profile("hybrid-IQ4_XS.gguf"), Some("Qwen3.8 27B"));
        assert_eq!(profile("small-dense"), Some("Qwen3 0.6B"));
        assert_eq!(profile("unmeasured-Q8_0.gguf"), None);
    }

    #[test]
    fn layout_detectors_are_their_own_kind() {
        let dir = tempfile::tempdir().unwrap();
        let checkpoint = dir.path().join("layout-detector");
        fs::create_dir(&checkpoint).unwrap();
        fs::write(
            checkpoint.join("config.json"),
            r#"{"model_type": "pp_doclayout_v3"}"#,
        )
        .unwrap();
        fs::write(checkpoint.join("model.safetensors"), b"").unwrap();
        write_gguf(
            &dir.path().join("layout-detector.gguf"),
            &[("general.architecture", text(LAYOUT_ARCHITECTURE))],
        );
        let scan = scan_models(&ScanRequest {
            folders: vec![dir.path().to_path_buf()],
            ..Default::default()
        });
        for name in ["layout-detector", "layout-detector.gguf"] {
            let model = by_name(&scan, name);
            assert_eq!(model.kind, Some(InventoryKind::Layout), "{model:?}");
            assert_eq!(model.error, None);
        }
    }

    #[test]
    fn scan_classifies_a_folders_models() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        write_gguf(
            &root.join("Alpha-7B-Q4_K_M.gguf"),
            &[
                ("general.architecture", text("qwen3")),
                ("general.name", text("Alpha 7B")),
                ("qwen3.context_length", Value::U32(32768)),
                base_model("acme/Alpha-7B"),
            ],
        );
        write_gguf(
            &root.join("mmproj-Alpha-F16.gguf"),
            &[
                ("general.architecture", text("clip")),
                base_model("Acme/alpha-7b"),
            ],
        );
        write_gguf(
            &root.join("mmproj-Beta-F16.gguf"),
            &[
                ("general.architecture", text("clip")),
                base_model("acme/Beta-30B"),
            ],
        );
        write_gguf(
            &root.join("Alpha-7B-imatrix.gguf"),
            &[
                ("general.architecture", text("qwen3")),
                ("general.type", text("imatrix")),
            ],
        );
        write_gguf(
            &root.join("Other-audio-Q8_0.gguf"),
            &[("general.architecture", text("audiocpp"))],
        );
        let shards = root.join("big");
        fs::create_dir(&shards).unwrap();
        for shard in [
            "Gamma-Q8_0-00001-of-00002.gguf",
            "Gamma-Q8_0-00002-of-00002.gguf",
        ] {
            write_gguf(
                &shards.join(shard),
                &[("general.architecture", text("llama"))],
            );
        }
        let checkpoint = root.join("tiny-llama");
        fs::create_dir(&checkpoint).unwrap();
        fs::write(
            checkpoint.join("config.json"),
            r#"{"architectures": ["LlamaForCausalLM"], "max_position_embeddings": 4096}"#,
        )
        .unwrap();
        fs::write(checkpoint.join("model.safetensors"), b"").unwrap();
        let config_only = root.join("config-only");
        fs::create_dir(&config_only).unwrap();
        fs::write(
            config_only.join("config.json"),
            r#"{"architectures": ["LlamaForCausalLM"]}"#,
        )
        .unwrap();

        let scan = scan_models(&ScanRequest {
            folders: vec![root.to_path_buf()],
            hf_cache: false,
            hf_cache_path: None,
        });

        assert!(scan.errors.is_empty());
        assert_eq!(scan.models.len(), 6, "{:?}", scan.models);
        let alpha = by_name(&scan, "Alpha-7B-Q4_K_M.gguf");
        assert_eq!(alpha.kind, Some(InventoryKind::Text));
        assert_eq!(alpha.architecture.as_deref(), Some("qwen3"));
        assert_eq!(alpha.name.as_deref(), Some("Alpha 7B"));
        assert_eq!(alpha.quantization.as_deref(), Some("Q4_K_M"));
        assert_eq!(alpha.context_length, Some(32768));
        assert_eq!(alpha.base_model.as_deref(), Some("acme/Alpha-7B"));
        assert_eq!(alpha.projectors.len(), 1);
        assert!(alpha.projectors[0].ends_with("mmproj-Alpha-F16.gguf"));
        assert_eq!(
            by_name(&scan, "mmproj-Beta-F16.gguf").kind,
            Some(InventoryKind::Projector)
        );
        let foreign = by_name(&scan, "Other-audio-Q8_0.gguf");
        assert_eq!(foreign.kind, None);
        assert!(foreign.error.as_deref().unwrap().contains("audiocpp"));
        let gamma = by_name(&scan, "Gamma-Q8_0-00001-of-00002.gguf");
        assert_eq!(gamma.files.len(), 2);
        assert_eq!(gamma.kind, Some(InventoryKind::Text));
        let llama = by_name(&scan, "tiny-llama");
        assert_eq!(llama.format, ModelFormat::Safetensors);
        assert_eq!(llama.kind, Some(InventoryKind::Text));
        assert_eq!(llama.loader.as_deref(), Some("llama"));
        assert_eq!(llama.context_length, Some(4096));

        let inspected = inspect_model(&shards.join("Gamma-Q8_0-00002-of-00002.gguf")).unwrap();
        assert_eq!(inspected.path, gamma.path);
        assert!(inspect_model(&root.join("Alpha-7B-imatrix.gguf")).is_err());
    }

    #[test]
    fn folder_names_gguf_repo_configs_and_base_model_less_models_scan_as_loads_see_them() {
        let dir = tempfile::tempdir().unwrap();
        // a folder named like an auxiliary file's marker above the model
        let models = dir.path().join("lora").join("models");
        fs::create_dir_all(&models).unwrap();
        write_gguf(
            &models.join("Delta-Q4_K_M.gguf"),
            &[("general.architecture", text("llama"))],
        );
        write_gguf(
            &models.join("mmproj-one-F16.gguf"),
            &[
                ("general.architecture", text("clip")),
                base_model("acme/One"),
            ],
        );
        write_gguf(
            &models.join("mmproj-two-F16.gguf"),
            &[("general.architecture", text("clip"))],
        );
        // a GGUF repository that ships a config: its quants, in subfolders, are the models
        let repo = dir.path().join("repo");
        fs::create_dir_all(repo.join("Q8_0")).unwrap();
        fs::write(
            repo.join("config.json"),
            r#"{"architectures": ["LlamaForCausalLM"]}"#,
        )
        .unwrap();
        write_gguf(
            &repo.join("Q8_0").join("Epsilon-Q8_0.gguf"),
            &[("general.architecture", text("llama"))],
        );

        let scan = scan_models(&ScanRequest {
            folders: vec![dir.path().to_path_buf()],
            hf_cache: false,
            hf_cache_path: None,
        });

        let delta = by_name(&scan, "Delta-Q4_K_M.gguf");
        assert_eq!(delta.kind, Some(InventoryKind::Text));
        // naming no base model, it keeps every projector, as the loader does
        assert_eq!(delta.projectors.len(), 2);
        assert_eq!(
            by_name(&scan, "Epsilon-Q8_0.gguf").kind,
            Some(InventoryKind::Text)
        );
        assert!(
            scan.models
                .iter()
                .all(|model| !model.path.ends_with("repo"))
        );

        // nextest runs each test in its own process, so the working directory is this test's
        std::env::set_current_dir(&models).unwrap();
        assert_eq!(
            inspect_model(Path::new("Delta-Q4_K_M.gguf")).unwrap().kind,
            Some(InventoryKind::Text)
        );
    }

    #[test]
    fn hf_cache_snapshots_list_and_delete() {
        let hub = tempfile::tempdir().unwrap();
        let repo = hub.path().join("models--acme--tiny");
        let snapshot = repo.join("snapshots").join("abc123");
        fs::create_dir_all(&snapshot).unwrap();
        fs::create_dir_all(repo.join("refs")).unwrap();
        fs::write(repo.join("refs").join("main"), "abc123").unwrap();
        fs::write(
            snapshot.join("config.json"),
            r#"{"architectures": ["Qwen3ForCausalLM"]}"#,
        )
        .unwrap();
        fs::write(snapshot.join("model.safetensors"), [0u8; 64]).unwrap();

        let scan = scan_models(&ScanRequest {
            folders: Vec::new(),
            hf_cache: true,
            hf_cache_path: Some(hub.path().to_path_buf()),
        });
        let listing = scan.hf_cache.as_ref().unwrap();
        assert_eq!(listing.repos.len(), 1);
        assert_eq!(listing.repos[0].repo_id, "acme/tiny");
        assert_eq!(scan.models.len(), 1);
        let model = &scan.models[0];
        assert_eq!(model.source, InventorySource::HfCache);
        assert_eq!(model.repo_id.as_deref(), Some("acme/tiny"));
        assert_eq!(model.loader.as_deref(), Some("qwen3"));

        for invalid in [
            "../escape",
            "acme/tiny/extra",
            "acme",
            "acme/..",
            "acme--x/tiny",
        ] {
            assert!(
                delete_hf_cache_repo(invalid, Some(hub.path())).is_err(),
                "{invalid}"
            );
        }
        assert!(delete_hf_cache_repo("acme/missing", Some(hub.path())).is_err());
        let legacy = hub.path().join("models--gpt2");
        fs::create_dir_all(legacy.join("snapshots")).unwrap();
        assert_eq!(
            delete_hf_cache_repo("gpt2", Some(hub.path()))
                .unwrap()
                .repo_id,
            "gpt2"
        );
        assert!(!legacy.exists());
        let deleted = delete_hf_cache_repo("acme/tiny", Some(hub.path())).unwrap();
        assert!(deleted.freed_bytes >= 64);
        assert!(!repo.exists());
    }
}
