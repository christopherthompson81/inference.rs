//! audio.cpp's GGUF container (`general.architecture = audiocpp`): a checkpoint's tensors under their own names (long
//! or repeated ones stored as `_audiocpp.<i>`, with the name in a table) and its other files embedded byte for byte.
//! A package unpacks those files once into a cache directory, so a loader reads it as it reads a checkpoint directory.

use std::collections::HashMap;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use inference_quant::{
    GgufArchive, GgufTensorBackend, GgufTensorBinding, GgufWeightSource, ShardedVarBuilder,
};
use inference_tensor::nn::VarBuilder;
use inference_tensor::quantized::gguf_file::{self, Header, ReadSeek, Value, peek_architecture};
use inference_tensor::quantized::{GgmlDType, QTensor};
use inference_tensor::{DType, Device, Tensor};

pub const ARCHITECTURE: &str = "audiocpp";
const GGUF_EXTENSION: &str = "gguf";
const CONFIG_FILE: &str = "config.json";
const FAMILY_KEY: &str = "audiocpp.model_spec.family";
const TENSOR_NAMES_KEY: &str = "audiocpp.tensor_names";
const TENSOR_RANKS_KEY: &str = "audiocpp.tensor_ranks";
const TENSOR_SHAPES_KEY: &str = "audiocpp.tensor_shapes";
const SOURCE_NAMES_KEY: &str = "audiocpp.tensor_sources.names";
const FILE_NAMES_KEY: &str = "audiocpp.embedded_files.names";
const FILE_OFFSETS_KEY: &str = "audiocpp.embedded_files.offsets";
const FILE_DATA_KEY: &str = "audiocpp.embedded_files.data";
// a family with several tensor sources names each tensor `<source>/<name>`
const SOURCE_SEPARATOR: char = '/';
const CACHE_DIR: &str = "inference-rs";
const UNPACK_DIR: &str = "audiocpp";
// written last, so a directory without it is an interrupted unpack
const COMPLETE_MARKER: &str = ".unpacked";
const ARCHITECTURE_KEY: &str = "general.architecture";
// GGML's integer types (I8, I16, I32, I64): BatchNorm's step counters and the like, which no model reads as weights
const INTEGER_TYPES: std::ops::RangeInclusive<u32> = 24..=27;
// GGML's `GGML_MAX_NAME`: a name this long or longer is stored under an alias
const MAX_STORED_NAME: usize = 64;
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0100_0000_01b3;

/// Whether `path` is an audio.cpp GGUF.
pub fn is_audiocpp_gguf(path: &Path) -> bool {
    let read = || -> inference_tensor::Result<bool> {
        let mut file = BufReader::new(std::fs::File::open(path)?);
        Ok(peek_architecture(&mut file)?.is_some_and(|a| a == ARCHITECTURE))
    };
    path.extension().is_some_and(|e| e == GGUF_EXTENSION)
        && path.is_file()
        && read().unwrap_or(false)
}

/// The audio.cpp GGUF `model_id` names: the file, or a directory holding exactly one.
pub fn find(model_id: &Path) -> Option<PathBuf> {
    if model_id.is_file() {
        return is_audiocpp_gguf(model_id).then(|| model_id.to_path_buf());
    }
    let mut found = std::fs::read_dir(model_id)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| is_audiocpp_gguf(p));
    let gguf = found.next()?;
    found.next().is_none().then_some(gguf)
}

/// An audio.cpp GGUF and the directory its embedded files are unpacked into.
pub struct Package {
    pub gguf: PathBuf,
    pub dir: PathBuf,
}

/// The package `model_id` names, unpacked: an audio.cpp GGUF, or a directory holding one and no `config.json` of its
/// own.
pub fn unpack(model_id: &Path) -> Result<Option<Package>> {
    if model_id.is_dir() && model_id.join(CONFIG_FILE).exists() {
        return Ok(None);
    }
    let Some(gguf) = find(model_id) else {
        return Ok(None);
    };
    let dir = unpack_dir(&gguf)?;
    if !dir.join(COMPLETE_MARKER).exists() {
        AudioCppGguf::open(&gguf)?.unpack_into(&dir)?;
    }
    Ok(Some(Package { gguf, dir }))
}

// one directory per file and version of it: its path, size and modification time
fn unpack_dir(gguf: &Path) -> Result<PathBuf> {
    let meta = std::fs::metadata(gguf)?;
    let modified = meta
        .modified()?
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    let canonical = std::fs::canonicalize(gguf)?;
    let key = format!("{}|{}|{modified}", canonical.display(), meta.len());
    let hash = key.bytes().fold(FNV_OFFSET, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(FNV_PRIME)
    });
    let stem = gguf
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(UNPACK_DIR);
    let root = inference_tensor::kernel_library::user_cache()
        .unwrap_or_else(std::env::temp_dir)
        .join(CACHE_DIR)
        .join(UNPACK_DIR);
    Ok(root.join(format!("{stem}-{hash:016x}")))
}

// unpacks whose GGUF is gone or has changed since
fn prune(root: &Path) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for dir in entries.filter_map(|e| e.ok().map(|e| e.path())) {
        let Ok(source) = std::fs::read_to_string(dir.join(COMPLETE_MARKER)) else {
            continue;
        };
        if unpack_dir(Path::new(&source)).ok().as_deref() != Some(dir.as_path()) {
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}

struct LogicalTensor {
    name: String,
    stored: String,
    shape: Vec<usize>,
}

/// An opened audio.cpp GGUF: its family, its tensors by their checkpoint names, and its embedded files.
pub struct AudioCppGguf {
    path: PathBuf,
    family: String,
    sources: Vec<String>,
    tensors: Vec<LogicalTensor>,
}

fn values<T>(
    header: &Header,
    key: &str,
    get: fn(&Value) -> inference_tensor::Result<T>,
) -> Result<Vec<T>> {
    let Some(value) = header.metadata.get(key) else {
        bail!("the audio.cpp GGUF has no `{key}`")
    };
    Ok(value
        .to_vec()?
        .iter()
        .map(get)
        .collect::<inference_tensor::Result<_>>()?)
}

fn strings(header: &Header, key: &str) -> Result<Vec<String>> {
    values(header, key, |v| v.to_string().cloned())
}

// the name a tensor is stored under when its own is too long or repeats
fn alias(index: usize) -> String {
    format!("_audiocpp.{index}")
}

impl AudioCppGguf {
    pub fn open(path: &Path) -> Result<Self> {
        let mut file = BufReader::new(std::fs::File::open(path)?);
        // the header alone: a package may hold tensors of types no model reads (BatchNorm's I64 counters)
        let header =
            Header::read(&mut file).with_context(|| format!("reading {}", path.display()))?;
        match header.metadata.get(ARCHITECTURE_KEY) {
            Some(Value::String(a)) if a == ARCHITECTURE => {}
            other => bail!(
                "{} is not an audio.cpp GGUF (architecture {other:?})",
                path.display()
            ),
        }
        let family = header
            .metadata
            .get(FAMILY_KEY)
            .map(|v| v.to_string().cloned())
            .transpose()?
            .unwrap_or_default();
        let sources = strings(&header, SOURCE_NAMES_KEY).unwrap_or_default();
        let names = strings(&header, TENSOR_NAMES_KEY)?;
        if names.len() != header.tensor_infos.len() {
            bail!(
                "{} tensor names for {} tensors",
                names.len(),
                header.tensor_infos.len()
            );
        }
        let ranks = values(&header, TENSOR_RANKS_KEY, Value::to_i32)?;
        let dims = values(&header, TENSOR_SHAPES_KEY, Value::to_i64)?;
        if ranks.len() != names.len() {
            bail!("{} tensor ranks for {} tensors", ranks.len(), names.len());
        }
        let mut cursor = 0;
        let mut tensors = Vec::with_capacity(names.len());
        for (index, (name, rank)) in names.into_iter().zip(ranks).enumerate() {
            let stored = match alias(index) {
                aliased if header.tensor_infos.contains_key(&aliased) => aliased,
                _ if header.tensor_infos.contains_key(&name) => name.clone(),
                _ => bail!("the audio.cpp GGUF stores no tensor for `{name}`"),
            };
            let rank = usize::try_from(rank)?;
            let shape = dims
                .get(cursor..cursor + rank)
                .context("the tensor shapes end early")?
                .iter()
                .map(|&d| usize::try_from(d))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            cursor += rank;
            tensors.push(LogicalTensor {
                name,
                stored,
                shape,
            });
        }
        Ok(Self {
            path: path.to_path_buf(),
            family,
            sources,
            tensors,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// audio.cpp's family name (`vibevoice_asr_streaming`, `parakeet_tdt`, ...).
    pub fn family(&self) -> &str {
        &self.family
    }

    /// The directory holding the embedded files, unpacked on first use and kept for later loads of the same file.
    pub fn unpacked(&self) -> Result<PathBuf> {
        let dir = unpack_dir(&self.path)?;
        if !dir.join(COMPLETE_MARKER).exists() {
            self.unpack_into(&dir)?;
        }
        Ok(dir)
    }

    fn unpack_into(&self, dir: &Path) -> Result<()> {
        let name = dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(UNPACK_DIR);
        let staging = dir.with_file_name(format!(".{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&staging);
        std::fs::create_dir_all(&staging)?;
        let mut file = BufReader::new(std::fs::File::open(&self.path)?);
        let header = Header::read(&mut file)?;
        for (name, bytes) in embedded_files(&header, &mut file)? {
            let target = staging.join(&name);
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(target, bytes)?;
        }
        let source = std::fs::canonicalize(&self.path)?;
        std::fs::write(staging.join(COMPLETE_MARKER), source.display().to_string())?;
        // another process may have finished the same unpack first; either copy serves
        if std::fs::rename(&staging, dir).is_err() {
            let _ = std::fs::remove_dir_all(&staging);
        }
        if !dir.join(COMPLETE_MARKER).exists() {
            bail!(
                "could not unpack {} into {}",
                self.path.display(),
                dir.display()
            );
        }
        if let Some(root) = dir.parent() {
            prune(root);
        }
        Ok(())
    }

    /// The tensors of `source` (all of them for a single-source file) under their checkpoint names, in `dtype`.
    pub fn var_builder(
        &self,
        source: Option<&str>,
        dtype: DType,
        device: &Device,
    ) -> Result<ShardedVarBuilder> {
        Ok(self
            .weight_source(source, dtype, None)?
            .sharded_var_builder(device.clone()))
    }

    /// The same tensors for a model built on plain `VarBuilder`s, each dequantized into `dtype` as it loads, under
    /// `rename`'s names where the model reads its own.
    pub fn dense_var_builder(
        &self,
        source: Option<&str>,
        dtype: DType,
        device: &Device,
        rename: Option<fn(&str) -> String>,
    ) -> Result<VarBuilder<'static>> {
        let backend = GgufTensorBackend::new(self.weight_source(source, dtype, rename)?);
        Ok(VarBuilder::from_backend(
            Box::new(backend),
            dtype,
            device.clone(),
        ))
    }

    fn weight_source(
        &self,
        source: Option<&str>,
        dtype: DType,
        rename: Option<fn(&str) -> String>,
    ) -> Result<Arc<GgufWeightSource>> {
        let archive = Arc::new(GgufArchive::open([&self.path])?);
        let prefix = source.map(|s| format!("{s}{SOURCE_SEPARATOR}"));
        let mut bindings = HashMap::new();
        for tensor in &self.tensors {
            let name = match &prefix {
                Some(prefix) => match tensor.name.strip_prefix(prefix.as_str()) {
                    Some(name) => name,
                    None => continue,
                },
                None => tensor.name.as_str(),
            };
            let info = archive.tensor_info(&tensor.stored)?;
            if INTEGER_TYPES.contains(&info.stored_dtype().raw()) {
                continue;
            }
            let stored = info.shape().to_vec();
            let binding = GgufTensorBinding::tensor(tensor.stored.clone());
            let binding = if stored == tensor.shape {
                binding
            } else {
                GgufTensorBinding::Reshape {
                    input: Box::new(binding),
                    dims: tensor.shape.clone(),
                }
            };
            bindings.insert(
                rename.map_or_else(|| name.to_string(), |r| r(name)),
                binding,
            );
        }
        if bindings.is_empty() {
            bail!(
                "{} has no tensors of source {source:?} (its sources: {:?})",
                self.path.display(),
                self.sources
            );
        }
        Ok(Arc::new(GgufWeightSource::new(archive, &bindings, dtype)?))
    }
}

/// The embedded files: names, and n + 1 offsets into one byte blob.
fn embedded_files(header: &Header, reader: &mut dyn ReadSeek) -> Result<Vec<(PathBuf, Vec<u8>)>> {
    let names = strings(header, FILE_NAMES_KEY).unwrap_or_default();
    if names.is_empty() {
        return Ok(Vec::new());
    }
    let offsets = values(header, FILE_OFFSETS_KEY, Value::to_u64)?;
    if offsets.len() != names.len() + 1 {
        bail!(
            "{} embedded files need {} offsets, got {}",
            names.len(),
            names.len() + 1,
            offsets.len()
        );
    }
    let inline = match header.metadata.get(FILE_DATA_KEY) {
        Some(value) => Some(
            value
                .to_vec()?
                .iter()
                .map(Value::to_u8)
                .collect::<inference_tensor::Result<Vec<u8>>>()?,
        ),
        None => None,
    };
    names
        .into_iter()
        .zip(offsets.windows(2))
        .map(|(name, range)| {
            let relative = Path::new(&name);
            if relative.is_absolute()
                || relative
                    .components()
                    .any(|c| !matches!(c, std::path::Component::Normal(_)))
            {
                bail!("the audio.cpp GGUF embeds a file outside its package: `{name}`");
            }
            let bytes = match &inline {
                Some(data) => data
                    .get(usize::try_from(range[0])?..usize::try_from(range[1])?)
                    .with_context(|| format!("`{name}` is past the embedded data"))?
                    .to_vec(),
                None => header.read_large_array(reader, FILE_DATA_KEY, range[0]..range[1])?,
            };
            Ok((relative.to_path_buf(), bytes))
        })
        .collect()
}

/// Writes a single-source package as audio.cpp's converter lays one out: each tensor under its name (an alias past
/// GGML's name limit), its exact shape recorded (a scalar stored as one element), and `files` embedded.
pub fn write_package(
    path: &Path,
    family: &str,
    tensors: &[(&str, &Tensor, GgmlDType)],
    files: &[(&str, &[u8])],
) -> Result<()> {
    let mut stored = Vec::with_capacity(tensors.len());
    let (mut ranks, mut shapes) = (Vec::new(), Vec::new());
    for (index, (name, tensor, dtype)) in tensors.iter().enumerate() {
        let physical = if name.len() >= MAX_STORED_NAME {
            alias(index)
        } else {
            name.to_string()
        };
        let one = if tensor.rank() == 0 {
            tensor.reshape(1)?
        } else {
            (*tensor).clone()
        };
        ranks.push(Value::I32(i32::try_from(tensor.rank())?));
        for &d in tensor.dims() {
            shapes.push(Value::I64(i64::try_from(d)?));
        }
        stored.push((physical, QTensor::quantize(&one, *dtype)?));
    }
    let mut offsets = vec![Value::U64(0)];
    let mut data = Vec::new();
    for (_, bytes) in files {
        data.extend(bytes.iter().map(|&b| Value::U8(b)));
        offsets.push(Value::U64(u64::try_from(data.len())?));
    }
    let strings =
        |values: Vec<String>| Value::Array(values.into_iter().map(Value::String).collect());
    let values = [
        (ARCHITECTURE_KEY, Value::String(ARCHITECTURE.into())),
        (FAMILY_KEY, Value::String(family.into())),
        (SOURCE_NAMES_KEY, strings(vec![String::new()])),
        (
            TENSOR_NAMES_KEY,
            strings(tensors.iter().map(|(n, _, _)| n.to_string()).collect()),
        ),
        (TENSOR_RANKS_KEY, Value::Array(ranks)),
        (TENSOR_SHAPES_KEY, Value::Array(shapes)),
        (
            FILE_NAMES_KEY,
            strings(files.iter().map(|(n, _)| n.to_string()).collect()),
        ),
        (FILE_OFFSETS_KEY, Value::Array(offsets)),
        (FILE_DATA_KEY, Value::Array(data)),
    ];
    let metadata = values.iter().map(|(k, v)| (*k, v)).collect::<Vec<_>>();
    let tensors = stored
        .iter()
        .map(|(n, q)| (n.as_str(), q))
        .collect::<Vec<_>>();
    let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
    gguf_file::write(&mut file, &metadata, &tensors)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const LONG_NAME: &str =
        "model.encoder.stages.0.0.mixer.conv.conv.conv.a_name_long_enough_to_need_an_alias.weight";

    #[test]
    fn a_package_reads_back_its_tensors_and_files() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("tiny.gguf");
        let device = Device::Cpu;
        let matrix = Tensor::arange(0f32, 64.0, &device)?.reshape((2, 32))?;
        let conv = Tensor::arange(0f32, 24.0, &device)?.reshape((2, 3, 4))?;
        let scalar = Tensor::new(0.5f32, &device)?;
        write_package(
            &path,
            "tiny_family",
            &[
                ("lm_head.weight", &matrix, GgmlDType::Q8_0),
                (LONG_NAME, &conv, GgmlDType::F32),
                ("model.scale", &scalar, GgmlDType::F32),
            ],
            &[("config.json", b"{}"), ("sub/tokenizer.json", b"tokens")],
        )?;
        assert!(is_audiocpp_gguf(&path));
        let package = AudioCppGguf::open(&path)?;
        assert_eq!(package.family(), "tiny_family");
        let vb = package.var_builder(None, DType::F32, &device)?;
        let back = vb.get((2, 3, 4), LONG_NAME)?;
        assert_eq!(back.to_vec3::<f32>()?, conv.to_vec3::<f32>()?);
        assert_eq!(vb.get((), "model.scale")?.to_scalar::<f32>()?, 0.5);
        let dense = package.dense_var_builder(None, DType::F32, &device, None)?;
        assert_eq!(
            dense.get((2, 3, 4), LONG_NAME)?.to_vec3::<f32>()?,
            conv.to_vec3::<f32>()?
        );
        // Q8_0 rounds each value to one of 127 steps of its block's largest
        let step = 63.0 / 127.0;
        let error = (vb.get((2, 32), "lm_head.weight")? - &matrix)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(error <= step, "{error}");
        let unpacked = package.unpacked()?;
        assert_eq!(
            std::fs::read(unpacked.join("sub/tokenizer.json"))?,
            b"tokens"
        );
        assert_eq!(unpack(&path)?.map(|p| p.dir), Some(unpacked));
        Ok(())
    }

    // GGML's I64 id, and the bytes of an F32 pair, which reread as one I64 counter
    const I64_TYPE: u32 = 27;
    const COUNTER: &str = "model.norm.num_batches_tracked";

    // Parakeet's package: BatchNorm counters stored as I64, and the file ending at its last tensor's bytes, unpadded
    #[test]
    fn a_package_skips_integer_tensors_and_reads_without_trailing_padding() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("tiny.gguf");
        let device = Device::Cpu;
        let pair = Tensor::new(&[0f32, 0.0], &device)?;
        let scalar = Tensor::new(0.5f32, &device)?;
        write_package(
            &path,
            "tiny_family",
            &[
                (COUNTER, &pair, GgmlDType::F32),
                ("model.scale", &scalar, GgmlDType::F32),
            ],
            &[],
        )?;
        let mut bytes = std::fs::read(&path)?;
        // the counter's table entry (after the name table, which names it too): its name, one dim, the dim, its type
        let name = COUNTER.as_bytes();
        let at = bytes
            .windows(name.len())
            .rposition(|w| w == name)
            .expect("the counter's entry")
            + name.len();
        let (dim, dtype) = (at + 4, at + 4 + 8);
        bytes[dim..dim + 8].copy_from_slice(&1u64.to_le_bytes());
        bytes[dtype..dtype + 4].copy_from_slice(&I64_TYPE.to_le_bytes());
        let header = Header::read(&mut std::io::Cursor::new(&bytes))?;
        let last = &header.tensor_infos["model.scale"];
        let end = header.tensor_data_offset + last.offset + std::mem::size_of::<f32>() as u64;
        let end = usize::try_from(end)?;
        assert!(end < bytes.len(), "the writer pads its last tensor");
        bytes.truncate(end);
        std::fs::write(&path, &bytes)?;

        let package = AudioCppGguf::open(&path)?;
        let vb = package.var_builder(None, DType::F32, &device)?;
        assert_eq!(vb.get((), "model.scale")?.to_scalar::<f32>()?, 0.5);
        assert!(!vb.contains_tensor(COUNTER));
        Ok(())
    }
}
