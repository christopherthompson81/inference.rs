//! Projector GGUFs in llama.cpp's mmproj layout, written from a checkpoint's own tensors: each family gives a spec (the
//! file's keys and, per tensor, the checkpoint tensor it is, reshaped or cut), beside the binding that reads it back.

use std::collections::BTreeSet;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use inference_quant::GgufType;
use inference_tensor::quantized::gguf_file::{self, Value};
use inference_tensor::safetensors::MmapedSafetensors;
use inference_tensor::{DType, Device, Tensor};

const CONFIG: &str = "config.json";
const PREPROCESSOR_CONFIG: &str = "preprocessor_config.json";
const INDEX: &str = "model.safetensors.index.json";
const TEMPLATE: &str = "chat_template.jinja";
const TOKENIZER_CONFIG: &str = "tokenizer_config.json";
/// The checkpoint's chat template, which a published text GGUF may carry without its media parts.
pub const CHAT_TEMPLATE_KEY: &str = "clip.chat_template";
const SAFETENSORS_EXTENSION: &str = "safetensors";
pub const MMPROJ_ARCHITECTURE: &str = "clip";
pub const MMPROJ_TYPE: &str = "mmproj";

/// A checkpoint tensor as the file stores it.
#[derive(Clone, Debug)]
pub enum Source {
    Tensor(String),
    /// The same elements under another (row-major) shape.
    Reshape(Box<Source>, Vec<usize>),
    /// `len` rows from `start` of the outermost dim (one of q, k, v from a fused qkv).
    Rows {
        input: Box<Source>,
        start: usize,
        len: usize,
    },
    /// A square grid of `width`-wide rows (a position table) resized bilinearly to `side x side`, without aligned
    /// corners, as `(1, side * side, width)`.
    Grid {
        input: Box<Source>,
        side: usize,
    },
}

impl Source {
    pub fn tensor(name: impl Into<String>) -> Self {
        Self::Tensor(name.into())
    }

    pub fn reshape(self, dims: Vec<usize>) -> Self {
        Self::Reshape(Box::new(self), dims)
    }

    pub fn grid(self, side: usize) -> Self {
        Self::Grid {
            input: Box::new(self),
            side,
        }
    }

    pub fn rows(self, start: usize, len: usize) -> Self {
        Self::Rows {
            input: Box::new(self),
            start,
            len,
        }
    }

    fn materialize(&self, weights: &MmapedSafetensors) -> Result<Tensor> {
        Ok(match self {
            Self::Tensor(name) => weights
                .load(name, &Device::Cpu)
                .with_context(|| format!("checkpoint tensor `{name}`"))?,
            Self::Reshape(input, dims) => input.materialize(weights)?.reshape(dims.as_slice())?,
            Self::Grid { input, side } => {
                let table = input.materialize(weights)?.to_dtype(DType::F32)?;
                let width = *table.dims().last().unwrap_or(&1);
                let cells = table.elem_count() / width;
                let from = (cells as f64).sqrt() as usize;
                if from * from != cells {
                    bail!("a {cells}-cell grid is not square");
                }
                table
                    .reshape((from, from, width))?
                    .permute((2, 0, 1))?
                    .unsqueeze(0)?
                    .upsample_bilinear2d(*side, *side, false)?
                    .squeeze(0)?
                    .permute((1, 2, 0))?
                    .reshape((1, side * side, width))?
            }
            Self::Rows { input, start, len } => {
                input.materialize(weights)?.narrow(0, *start, *len)?
            }
        })
    }
}

/// How a tensor is stored: matrices at the writer's matrix type, everything else (norms, biases, convolutions,
/// embeddings, position tables) in f32, as llama.cpp's converter keeps them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Matrix,
    Float,
}

#[derive(Clone, Debug)]
pub struct MmprojTensor {
    pub name: String,
    pub source: Source,
    pub role: Role,
}

impl MmprojTensor {
    pub fn new(name: impl Into<String>, source: Source, role: Role) -> Self {
        Self {
            name: name.into(),
            source,
            role,
        }
    }
}

/// What a family writes: the file's keys past `general.architecture` / `general.type`, and its tensors.
#[derive(Default)]
pub struct MmprojSpec {
    pub metadata: Vec<(String, Value)>,
    pub tensors: Vec<MmprojTensor>,
}

/// The checkpoint a spec is built from: its configs and every tensor of the safetensors files present.
pub struct Checkpoint {
    pub config: serde_json::Value,
    pub preprocessor: Option<serde_json::Value>,
    pub chat_template: Option<String>,
    pub weights: MmapedSafetensors,
}

impl Checkpoint {
    /// A checkpoint directory; shards its index lists but that are not on disk are left out, so a family whose towers
    /// sit in one shard needs only that one.
    pub fn open(dir: &Path) -> Result<Self> {
        let read = |name: &str| -> Result<Option<serde_json::Value>> {
            let path = dir.join(name);
            if !path.is_file() {
                return Ok(None);
            }
            let text = std::fs::read_to_string(&path)
                .with_context(|| format!("reading {}", path.display()))?;
            // Python's `json` writes non-finite floats bare (Nano VL's `time_step_limit`)
            let text = inference_nn::loaders::encode_non_finite_floats(&text);
            Ok(Some(
                serde_json::from_str(&text)
                    .with_context(|| format!("parsing {}", path.display()))?,
            ))
        };
        let config = read(CONFIG)?.with_context(|| format!("{} has no {CONFIG}", dir.display()))?;
        let files = shard_files(dir, read(INDEX)?)?;
        if files.is_empty() {
            bail!("{} has no safetensors files", dir.display());
        }
        // SAFETY: the files are only read, and not expected to change while they are mapped
        let weights = unsafe { MmapedSafetensors::multi(&files)? };
        let template = dir.join(TEMPLATE);
        let chat_template = if template.is_file() {
            Some(std::fs::read_to_string(&template)?)
        } else {
            read(TOKENIZER_CONFIG)?
                .and_then(|tokenizer| tokenizer["chat_template"].as_str().map(str::to_string))
        };
        Ok(Self {
            config,
            preprocessor: read(PREPROCESSOR_CONFIG)?,
            chat_template,
            weights,
        })
    }

    pub fn shape(&self, name: &str) -> Result<Vec<usize>> {
        Ok(self
            .weights
            .get(name)
            .with_context(|| format!("checkpoint tensor `{name}`"))?
            .shape()
            .to_vec())
    }

    pub fn contains(&self, name: &str) -> bool {
        self.weights.get(name).is_ok()
    }

    pub fn architecture(&self) -> Option<&str> {
        self.config["architectures"][0].as_str()
    }
}

fn shard_files(dir: &Path, index: Option<serde_json::Value>) -> Result<Vec<PathBuf>> {
    let names = match index {
        Some(index) => index["weight_map"]
            .as_object()
            .context("the safetensors index has no `weight_map`")?
            .values()
            .filter_map(|file| file.as_str().map(str::to_string))
            .collect::<BTreeSet<_>>(),
        None => std::fs::read_dir(dir)?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| {
                path.extension()
                    .is_some_and(|ext| ext == SAFETENSORS_EXTENSION)
            })
            .filter_map(|path| {
                path.file_name()
                    .map(|name| name.to_string_lossy().into_owned())
            })
            .collect(),
    };
    Ok(names
        .into_iter()
        .map(|name| dir.join(name))
        .filter(|path| path.is_file())
        .collect())
}

// families by the checkpoint's `architectures[0]`
type SpecBuilder = fn(&Checkpoint) -> Result<MmprojSpec>;
const FAMILIES: [(&str, SpecBuilder); 2] = [
    (
        "NemotronH_Nano_Omni_Reasoning_V3",
        super::nemotron_omni_bindings::nano_omni_mmproj_spec,
    ),
    (
        "NemotronH_Nano_VL_V2",
        super::nemotron_omni_bindings::nano_vl_mmproj_spec,
    ),
];

/// The architectures `write_mmproj` writes a projector for.
pub fn mmproj_architectures() -> impl Iterator<Item = &'static str> {
    FAMILIES.iter().map(|(name, _)| *name)
}

pub struct MmprojSummary {
    pub architecture: String,
    pub tensors: usize,
    pub bytes: usize,
}

/// The projector of the checkpoint in `dir`, its matrices as `matrix_type` (`bf16`, `f16`, `q8_0`, ...; the
/// checkpoint's own float type when `None`).
pub fn write_mmproj(dir: &Path, out: &Path, matrix_type: Option<&str>) -> Result<MmprojSummary> {
    let checkpoint = Checkpoint::open(dir)?;
    let architecture = checkpoint
        .architecture()
        .context("the checkpoint's config has no `architectures`")?
        .to_string();
    let Some((_, build)) = FAMILIES.iter().find(|(name, _)| *name == architecture) else {
        bail!(
            "no projector writer for `{architecture}`; written: {:?}",
            mmproj_architectures().collect::<Vec<_>>()
        );
    };
    let mut spec = build(&checkpoint)?;
    if let Some(template) = &checkpoint.chat_template {
        spec.metadata.push((
            CHAT_TEMPLATE_KEY.to_string(),
            Value::String(template.clone()),
        ));
    }
    let matrix_type = matrix_type
        .map(inference_quant::requantize::writable_type)
        .transpose()?;
    let (tensors, bytes) = write_spec(&spec, &checkpoint.weights, out, matrix_type)?;
    Ok(MmprojSummary {
        architecture,
        tensors,
        bytes,
    })
}

struct Encoded {
    dims: Vec<usize>,
    ggml_dtype: u32,
    data: Vec<u8>,
}

fn float_bytes(tensor: &Tensor, dtype: DType) -> Result<(u32, Vec<u8>)> {
    let flat = tensor.flatten_all()?;
    Ok(match dtype {
        DType::BF16 => (
            GgufType::BF16.id(),
            flat.to_dtype(DType::BF16)?
                .to_vec1::<half::bf16>()?
                .iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect(),
        ),
        DType::F16 => (
            GgufType::F16.id(),
            flat.to_dtype(DType::F16)?
                .to_vec1::<half::f16>()?
                .iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect(),
        ),
        _ => (
            GgufType::F32.id(),
            flat.to_dtype(DType::F32)?
                .to_vec1::<f32>()?
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        ),
    })
}

fn encode(tensor: &Tensor, role: Role, matrix_type: Option<GgufType>) -> Result<(u32, Vec<u8>)> {
    if role == Role::Float || tensor.rank() < 2 {
        return float_bytes(tensor, DType::F32);
    }
    let Some(ty) = matrix_type else {
        return float_bytes(tensor, tensor.dtype());
    };
    match ty.candle() {
        Some(inference_tensor::quantized::GgmlDType::F32) => {
            return float_bytes(tensor, DType::F32);
        }
        Some(inference_tensor::quantized::GgmlDType::F16) => {
            return float_bytes(tensor, DType::F16);
        }
        Some(inference_tensor::quantized::GgmlDType::BF16) => {
            return float_bytes(tensor, DType::BF16);
        }
        _ => {}
    }
    let cols = *tensor.dims().last().unwrap_or(&1);
    // a row that is not whole blocks of the type stays at the checkpoint's float type
    if !cols.is_multiple_of(ty.block_size()) {
        return float_bytes(tensor, tensor.dtype());
    }
    let values = tensor
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    let rows = values.len() / cols;
    Ok((
        ty.id(),
        inference_quant::requantize::quantize_matrix(ty, cols, &values, None, rows)?,
    ))
}

fn write_spec(
    spec: &MmprojSpec,
    weights: &MmapedSafetensors,
    out: &Path,
    matrix_type: Option<GgufType>,
) -> Result<(usize, usize)> {
    let encoded = spec
        .tensors
        .iter()
        .map(|tensor| {
            let value = tensor.source.materialize(weights)?;
            let (ggml_dtype, data) = encode(&value, tensor.role, matrix_type)
                .with_context(|| format!("encoding `{}`", tensor.name))?;
            Ok(Encoded {
                dims: value.dims().to_vec(),
                ggml_dtype,
                data,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut metadata = vec![
        (
            "general.architecture".to_string(),
            Value::String(MMPROJ_ARCHITECTURE.to_string()),
        ),
        (
            "general.type".to_string(),
            Value::String(MMPROJ_TYPE.to_string()),
        ),
    ];
    metadata.extend(spec.metadata.iter().cloned());
    let metadata = metadata
        .iter()
        .map(|(key, value)| (key.as_str(), value))
        .collect::<Vec<_>>();
    let entries = spec
        .tensors
        .iter()
        .zip(&encoded)
        .map(|(tensor, encoded)| gguf_file::TensorEntry {
            name: &tensor.name,
            dims: &encoded.dims,
            ggml_dtype: encoded.ggml_dtype,
            size_in_bytes: encoded.data.len(),
        })
        .collect::<Vec<_>>();
    let mut writer = BufWriter::new(
        std::fs::File::create(out).with_context(|| format!("creating {}", out.display()))?,
    );
    gguf_file::write_header(
        &mut writer,
        metadata.len(),
        &gguf_file::metadata_bytes(&metadata)?,
        &entries,
    )?;
    let mut bytes = 0;
    for tensor in &encoded {
        gguf_file::write_tensor_data(&mut writer, &tensor.data)?;
        bytes += tensor.data.len();
    }
    std::io::Write::flush(&mut writer)?;
    Ok((encoded.len(), bytes))
}
