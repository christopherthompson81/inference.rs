//! A float GGUF to a quantized one: each tensor's type from the first rule whose pattern matches its name, the
//! imatrix (llama.cpp's, keyed by GGUF names) weighing the encoders that take one. Streamed a slice at a time (an
//! expert, or a bounded run of rows), so memory stays at one slice whatever the model's size.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use half::{bf16, f16};
pub use inference_tensor::quantized::Quantizer;
use inference_tensor::quantized::gguf_file::{
    self, Header, TensorEntry, write_header, write_tensor_padding,
};
use inference_tensor::quantized::{GgmlDType, QTensor, quantize_blocks};
use inference_tensor::{Device, Result, Tensor};
use rayon::prelude::*;
use regex::Regex;

use super::archive::{GgufDType, OVERLAY_GGUF_TYPE};
use super::iq_quantize;
use super::kernel::GgufType;

// rows a parallel task quantizes through `QTensor`
const ROWS_PER_TASK: usize = 64;
// the source bytes read at once: one expert of an expert stack, else rows up to this
const SLICE_BYTES: usize = 64 << 20;
/// The share of a Hessian's mean diagonal GPTQ adds to it by default (its `percdamp`): 0.01 fit the calibration's
/// domain and lost on other text, 0.1 held up across held-out text (quant_allocation_investigation Run 9).
pub const DEFAULT_GPTQ_DAMP: f64 = 0.1;
// f32 bytes of rows the GPU quantizes in one call (its scratch takes about three times this)
const GPU_CHUNK_BYTES: usize = 1 << 30;
const GGUF_ALIGNMENT: u64 = 32;
const GENERAL_ARCHITECTURE: &str = "general.architecture";
const GENERAL_TYPE: &str = "general.type";

// the formats `QTensor` encodes, each matching llama.cpp's encoder byte for byte
const CANDLE_WRITABLE: [GgmlDType; 13] = [
    GgmlDType::F32,
    GgmlDType::F16,
    GgmlDType::BF16,
    GgmlDType::Q4_0,
    GgmlDType::Q4_1,
    GgmlDType::Q5_0,
    GgmlDType::Q5_1,
    GgmlDType::Q8_0,
    GgmlDType::Q2K,
    GgmlDType::Q3K,
    GgmlDType::Q4K,
    GgmlDType::Q5K,
    GgmlDType::Q6K,
];

fn writable_types() -> impl Iterator<Item = GgufType> {
    let raw = GgufType::RAW_BLOCKS
        .into_iter()
        .filter(|ty| iq_quantize::can_quantize(*ty));
    CANDLE_WRITABLE.into_iter().map(GgufType::from).chain(raw)
}

/// A tensor type by its llama.cpp name (`q4_k`, `iq4_ks`), if this module writes it.
pub fn writable_type(name: &str) -> Result<GgufType> {
    writable_types()
        .find(|ty| GgufDType::new(ty.id()).name().eq_ignore_ascii_case(name))
        .ok_or_else(|| {
            let known = writable_types()
                .map(|ty| GgufDType::new(ty.id()).name().to_ascii_lowercase())
                .collect::<Vec<_>>();
            inference_tensor::Error::Msg(format!("`{name}` is not one of {known:?}"))
        })
}

/// Rows of `cols` floats as `ty`, each `rows_per_slice` rows weighed by the next `cols` of `imatrix` (an expert
/// stack has one slice per expert).
pub fn quantize_matrix(
    ty: GgufType,
    cols: usize,
    values: &[f32],
    imatrix: Option<&[f32]>,
    rows_per_slice: usize,
    quantizer: Quantizer,
) -> Result<Vec<u8>> {
    let slice_values = rows_per_slice * cols;
    if let Some(w) = imatrix
        && w.len() * rows_per_slice != values.len()
    {
        inference_tensor::bail!(
            "an imatrix of {} for {} rows of {cols} in slices of {rows_per_slice}",
            w.len(),
            values.len() / cols
        );
    }
    let mut out = Vec::new();
    for (i, slice) in values.chunks(slice_values).enumerate() {
        let w = imatrix.map(|w| &w[i * cols..(i + 1) * cols]);
        out.extend(quantize_slice(ty, cols, slice, w, quantizer)?);
    }
    Ok(out)
}

fn quantize_slice(
    ty: GgufType,
    cols: usize,
    values: &[f32],
    imatrix: Option<&[f32]>,
    quantizer: Quantizer,
) -> Result<Vec<u8>> {
    if iq_quantize::can_quantize(ty) {
        return iq_quantize::quantize_rows(ty, cols, values, imatrix, quantizer);
    }
    let Some(dtype) = ty.candle() else {
        inference_tensor::bail!("{ty:?} has no quantizer");
    };
    if !matches!(dtype, GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16) {
        return quantize_blocks(dtype, values, imatrix, cols, quantizer);
    }
    let chunks = values
        .par_chunks(ROWS_PER_TASK * cols)
        .map(|chunk| {
            let t = Tensor::from_slice(chunk, (chunk.len() / cols, cols), &Device::Cpu)?;
            Ok(QTensor::quantize(&t, dtype)?.data()?.into_owned())
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(chunks.concat())
}

/// Where `quantize gguf --hessians` finds a tensor's input Hessian (`X^T X / n`, `[cols, cols]` f32 row-major).
pub trait HessianSource: Send + Sync {
    /// The Hessian of tensor `name` (by GGUF name), or of expert `expert` of a stack (None where the stack has one
    /// for all its experts or none).
    fn hessian(&self, name: &str, expert: Option<usize>) -> Result<Option<Vec<f32>>>;
}

/// A safetensors file of Hessians keyed by GGUF tensor name.
pub struct HessianFile(inference_tensor::safetensors::MmapedSafetensors);

impl HessianFile {
    pub fn open(path: &Path) -> Result<Self> {
        // SAFETY: the file is only read while it is open here
        Ok(Self(unsafe {
            inference_tensor::safetensors::MmapedSafetensors::new(path)?
        }))
    }
}

impl HessianSource for HessianFile {
    fn hessian(&self, name: &str, expert: Option<usize>) -> Result<Option<Vec<f32>>> {
        if expert.is_some() {
            return Ok(None);
        }
        let Ok(h) = self.0.load(name, &Device::Cpu) else {
            return Ok(None);
        };
        Ok(Some(
            h.to_dtype(inference_tensor::DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?,
        ))
    }
}

/// The rules for which tensors become which type, in order: the first pattern matching a name wins, and a tensor
/// no pattern matches is copied as it is.
pub struct RequantizePlan {
    pub rules: Vec<(Regex, GgufType)>,
    /// llama.cpp's per-tensor importances (mean squared activations, an expert stack's experts one after another).
    pub imatrix: Option<HashMap<String, Vec<f32>>>,
    pub quantizer: Quantizer,
    /// Earlier outputs from the same source, imatrix and quantizer: a tensor one of them holds at its planned type is
    /// copied from it instead of quantized again.
    pub reuse: Vec<PathBuf>,
    /// Writes only the tensors this matches, as an overlay the source's quantized model loads beside it.
    pub only: Option<Regex>,
    /// The device whose quantizers write the types they cover (the same bytes as the CPU's, sooner).
    pub gpu: Option<Device>,
    /// Input Hessians: a tensor with one (each expert of a stack with its own or the stack's), of a type GPTQ covers,
    /// is quantized with GPTQ-style error feedback on `gpu` (CUDA).
    pub hessians: Option<Box<dyn HessianSource>>,
    /// With `hessians`, only their diagonals weigh the quantizer (the baseline the feedback is measured against).
    pub hessian_weights_only: bool,
    /// The share of a Hessian's mean diagonal added to it before the feedback.
    pub gptq_damp: f64,
}

#[derive(Debug, Default)]
pub struct RequantizeSummary {
    pub tensors: usize,
    pub quantized: usize,
    /// Of the quantized, the ones copied from a `reuse` file.
    pub reused: usize,
    /// Of the quantized, the ones written with GPTQ-style error feedback (or, weights only, their Hessian diagonals).
    pub gptq: usize,
    /// Quantized tensors the imatrix had no entry for.
    pub without_imatrix: Vec<String>,
    pub bytes_in: u64,
    pub bytes_out: u64,
    pub by_type: Vec<(String, usize, u64)>,
}

fn to_f32(ggml_dtype: u32, bytes: &[u8]) -> Result<Vec<f32>> {
    let halves = || bytes.as_chunks::<2>().0.iter();
    Ok(match ggml_dtype {
        0 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        1 => halves().map(|c| f16::from_le_bytes(*c).to_f32()).collect(),
        30 => halves().map(|c| bf16::from_le_bytes(*c).to_f32()).collect(),
        other => inference_tensor::bail!(
            "only float tensors are requantized, not {}",
            GgufDType::new(other).name()
        ),
    })
}

struct Planned<'a> {
    name: &'a str,
    dims: Vec<usize>,
    source_type: u32,
    source_bytes: usize,
    target: Option<GgufType>,
    size: usize,
}

fn plan_tensor<'a>(name: &'a str, header: &Header, plan: &RequantizePlan) -> Result<Planned<'a>> {
    let info = &header.tensor_infos[name];
    let dims = info.shape.dims().to_vec();
    let cols = *dims.last().unwrap_or(&1);
    let rows = info.shape.elem_count() / cols.max(1);
    let source = GgufDType::new(info.ggml_dtype);
    let source_bytes = rows
        * source.row_size(cols).ok_or_else(|| {
            inference_tensor::Error::Msg(format!(
                "{name}: rows of {cols} are not whole {} blocks",
                source.name()
            ))
        })?;
    let target = if dims.len() < 2 {
        None
    } else {
        plan.rules
            .iter()
            .find(|(re, _)| re.is_match(name))
            .map(|(_, ty)| *ty)
            .filter(|ty| ty.id() != info.ggml_dtype)
    };
    let size = match target {
        Some(ty) => {
            rows * ty.row_bytes(cols).ok_or_else(|| {
                inference_tensor::Error::Msg(format!("{name}: rows of {cols} cannot be {ty:?}"))
            })?
        }
        None => source_bytes,
    };
    Ok(Planned {
        name,
        dims,
        source_type: info.ggml_dtype,
        source_bytes,
        target,
        size,
    })
}

// `t` quantized with GPTQ feedback when `hessians` has its Hessian (a stack's experts each with its own, or the stack's
// for all); None leaves it to the ordinary path, the reader where it was
#[cfg(feature = "cuda")]
fn gptq_tensor(
    t: &Planned<'_>,
    ty: GgufType,
    hessians: &dyn HessianSource,
    plan: &RequantizePlan,
    device: &Device,
    reader: &mut BufReader<File>,
    offset: u64,
) -> Result<Option<Vec<u8>>> {
    let cols = *t.dims.last().unwrap_or(&1);
    let experts = if t.dims.len() == 3 { t.dims[0] } else { 1 };
    let shared = hessians.hessian(t.name, None)?;
    let per_expert = if t.dims.len() == 3 {
        (0..experts)
            .map(|e| hessians.hessian(t.name, Some(e)))
            .collect::<Result<Vec<_>>>()?
    } else {
        Vec::new()
    };
    if shared.is_none() && per_expert.iter().all(Option::is_none) {
        return Ok(None);
    }
    let mut bytes = vec![0u8; t.source_bytes];
    reader.seek(SeekFrom::Start(offset))?;
    reader.read_exact(&mut bytes)?;
    let values = to_f32(t.source_type, &bytes)?;
    let per = values.len() / experts;
    let mut out = Vec::with_capacity(t.size);
    for e in 0..experts {
        let h = per_expert
            .get(e)
            .and_then(Option::as_ref)
            .or(shared.as_ref())
            .ok_or_else(|| {
                inference_tensor::Error::Msg(format!("{}: no Hessian for expert {e}", t.name))
            })?;
        out.extend(
            super::gptq::quantize(
                ty,
                cols,
                &values[e * per..(e + 1) * per],
                h,
                plan.quantizer,
                !plan.hessian_weights_only,
                plan.gptq_damp,
                device,
                quantize_slice,
                dequantize_any,
            )
            .map_err(|err| inference_tensor::Error::Msg(format!("{}: {err}", t.name)))?,
        );
    }
    Ok(Some(out))
}

// rows of `cols` of `ty` back to f32: the raw types' dequantizers, or Candle's for the types it holds
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn dequantize_any(ty: GgufType, cols: usize, bytes: &[u8]) -> Result<Vec<f32>> {
    if let Some(dtype) = ty.candle()
        && !matches!(dtype, GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16)
        && !matches!(ty, GgufType::Iq4Nl | GgufType::Iq4Xs)
    {
        let rows = bytes.len() / ty.row_bytes(cols).unwrap_or(1).max(1);
        let storage = inference_tensor::quantized::QStorage::from_data(
            std::borrow::Cow::Borrowed(bytes),
            &Device::Cpu,
            dtype,
        )?;
        return QTensor::new(storage, (rows, cols))?
            .dequantize(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>();
    }
    super::raw::dequantize_rows(ty, cols, bytes)
}

#[cfg(feature = "cuda")]
type GpuDevice = inference_tensor::CudaDevice;
#[cfg(not(feature = "cuda"))]
type GpuDevice = std::convert::Infallible;

// the plan's GPU, when its quantizer writes `ty` with the plan's search
fn gpu_for(plan: &RequantizePlan, ty: GgufType) -> Option<&GpuDevice> {
    #[cfg(feature = "cuda")]
    if let Some(Device::Cuda(dev)) = &plan.gpu
        && plan.quantizer == Quantizer::IkLlama
        && super::quantize_cuda::can_quantize(ty)
    {
        return Some(dev);
    }
    let _ = (plan, ty);
    None
}

fn gpu_quantize(
    dev: &GpuDevice,
    ty: GgufType,
    cols: usize,
    dtype: u32,
    source: &[u8],
    imatrix: Option<&[f32]>,
    rows_per_imatrix: usize,
) -> Result<Vec<u8>> {
    #[cfg(feature = "cuda")]
    return super::quantize_cuda::quantize_rows(
        dev,
        ty,
        cols,
        dtype,
        source,
        imatrix,
        rows_per_imatrix,
    );
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (ty, cols, dtype, source, imatrix, rows_per_imatrix);
        match *dev {}
    }
}

/// `src` rewritten to `dst` under `plan`; its metadata copied as it is. `progress` hears each tensor's name as it
/// starts.
pub fn requantize_gguf(
    src: &Path,
    dst: &Path,
    plan: &RequantizePlan,
    mut progress: impl FnMut(usize, usize, &str),
) -> Result<RequantizeSummary> {
    let mut reader = BufReader::new(File::open(src)?);
    let header = Header::read_summary(&mut reader)?;
    if header
        .metadata
        .get("general.alignment")
        .is_some_and(|v| v.to_u32().ok() != Some(GGUF_ALIGNMENT as u32))
    {
        inference_tensor::bail!(
            "{}: only 32-byte aligned GGUFs are rewritten",
            src.display()
        );
    }
    let tensors = header
        .tensor_order
        .iter()
        .filter(|name| plan.only.as_ref().is_none_or(|only| only.is_match(name)))
        .map(|name| plan_tensor(name, &header, plan))
        .collect::<Result<Vec<_>>>()?;

    let mut kv_bytes = vec![0u8; (header.metadata_span.end - header.metadata_span.start) as usize];
    reader.seek(SeekFrom::Start(header.metadata_span.start))?;
    reader.read_exact(&mut kv_bytes)?;
    let mut kv_count = {
        reader.seek(SeekFrom::Start(16))?;
        let mut count = [0u8; 8];
        reader.read_exact(&mut count)?;
        u64::from_le_bytes(count) as usize
    };
    if plan.only.is_some() {
        let architecture = header
            .metadata
            .get(GENERAL_ARCHITECTURE)
            .ok_or_else(|| inference_tensor::Error::msg("the source has no architecture"))?;
        let overlay = gguf_file::Value::String(OVERLAY_GGUF_TYPE.to_string());
        let metadata = [
            (GENERAL_ARCHITECTURE, architecture),
            (GENERAL_TYPE, &overlay),
        ];
        kv_bytes = gguf_file::metadata_bytes(&metadata)?;
        kv_count = metadata.len();
    }
    let mut reuse = plan
        .reuse
        .iter()
        .map(|path| -> Result<_> {
            let mut reader = BufReader::new(File::open(path)?);
            let header = Header::read_summary(&mut reader)?;
            Ok((reader, header))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut writer = BufWriter::new(File::create(dst)?);
    let entries = tensors
        .iter()
        .map(|t| TensorEntry {
            name: t.name,
            dims: &t.dims,
            ggml_dtype: t.target.map_or(t.source_type, |ty| ty.id()),
            size_in_bytes: t.size,
        })
        .collect::<Vec<_>>();
    write_header(&mut writer, kv_count, &kv_bytes, &entries)?;

    let mut summary = RequantizeSummary {
        tensors: tensors.len(),
        ..Default::default()
    };
    let mut by_type = HashMap::<u32, (usize, u64)>::new();
    for (i, t) in tensors.iter().enumerate() {
        progress(i, tensors.len(), t.name);
        let info = &header.tensor_infos[t.name];
        reader.seek(SeekFrom::Start(header.tensor_data_offset + info.offset))?;
        let cols = *t.dims.last().unwrap_or(&1);
        let rows = t.dims.iter().product::<usize>() / cols.max(1);
        let source_row = t.source_bytes / rows.max(1);
        let imatrix = t
            .target
            .and(plan.imatrix.as_ref())
            .and_then(|m| m.get(t.name));
        if t.target.is_some() {
            if plan.imatrix.is_some() && imatrix.is_none() {
                summary.without_imatrix.push(t.name.to_string());
            }
            summary.quantized += 1;
        }
        // an expert stack's imatrix holds `cols` per expert, so its slices are whole experts
        let expert_rows = (t.dims.len() > 2).then(|| t.dims[t.dims.len() - 2]);
        if let Some(w) = imatrix
            && w.len() != cols * (rows / expert_rows.unwrap_or(rows))
        {
            inference_tensor::bail!("{}: an imatrix of {} for rows of {cols}", t.name, w.len());
        }
        let reused = t.target.and_then(|ty| {
            reuse.iter_mut().find(|(_, h)| {
                h.tensor_infos.get(t.name).is_some_and(|info| {
                    info.ggml_dtype == ty.id() && info.shape.dims() == t.dims.as_slice()
                })
            })
        });
        if let Some((reader, h)) = reused {
            reader.seek(SeekFrom::Start(
                h.tensor_data_offset + h.tensor_infos[t.name].offset,
            ))?;
            let copied = std::io::copy(&mut reader.by_ref().take(t.size as u64), &mut writer)?;
            if copied != t.size as u64 {
                inference_tensor::bail!("{}: reused {copied} bytes, planned {}", t.name, t.size);
            }
            write_tensor_padding(&mut writer, t.size)?;
            summary.reused += 1;
            summary.bytes_in += t.source_bytes as u64;
            summary.bytes_out += t.size as u64;
            let entry = by_type
                .entry(t.target.map_or(t.source_type, |ty| ty.id()))
                .or_default();
            entry.0 += 1;
            entry.1 += t.size as u64;
            continue;
        }
        // GPTQ takes the whole matrix (each expert's) at once, its Hessian beside it
        #[cfg(feature = "cuda")]
        if let (Some(ty), Some(h), Some(device @ Device::Cuda(_))) =
            (t.target, &plan.hessians, &plan.gpu)
            && super::gptq::supports(ty)
            && let Some(out) = gptq_tensor(
                t,
                ty,
                h.as_ref(),
                plan,
                device,
                &mut reader,
                header.tensor_data_offset + info.offset,
            )?
        {
            if out.len() != t.size {
                inference_tensor::bail!(
                    "{}: wrote {} bytes, planned {}",
                    t.name,
                    out.len(),
                    t.size
                );
            }
            writer.write_all(&out)?;
            write_tensor_padding(&mut writer, t.size)?;
            summary.gptq += 1;
            summary.bytes_in += t.source_bytes as u64;
            summary.bytes_out += t.size as u64;
            let entry = by_type.entry(ty.id()).or_default();
            entry.0 += 1;
            entry.1 += t.size as u64;
            continue;
        }
        let gpu = t.target.and_then(|ty| gpu_for(plan, ty));
        let slice_rows = match (gpu.is_some(), expert_rows) {
            // the GPU takes whole experts by the chunk, each weighed by its own imatrix slice
            (true, Some(e)) => e * (GPU_CHUNK_BYTES / (e * cols * 4).max(1)).max(1),
            (true, None) => (GPU_CHUNK_BYTES / (cols * 4).max(1)).clamp(1, rows.max(1)),
            (false, Some(e)) => e,
            (false, None) => (SLICE_BYTES / source_row.max(1)).clamp(1, rows.max(1)),
        };
        let mut written = 0usize;
        let mut bytes = Vec::new();
        for first in (0..rows).step_by(slice_rows) {
            let n = slice_rows.min(rows - first);
            bytes.resize(n * source_row, 0);
            reader.read_exact(&mut bytes)?;
            let Some(ty) = t.target else {
                writer.write_all(&bytes)?;
                written += bytes.len();
                continue;
            };
            let out = match gpu {
                Some(dev) => {
                    let (w, per) = match expert_rows {
                        Some(e) => (
                            imatrix.map(|w| &w[first / e * cols..(first + n) / e * cols]),
                            e,
                        ),
                        None => (imatrix.map(Vec::as_slice), n),
                    };
                    gpu_quantize(dev, ty, cols, t.source_type, &bytes, w, per)
                }
                None => {
                    let w = imatrix.map(|w| {
                        let expert = expert_rows.map_or(0, |e| first / e);
                        &w[expert * cols..(expert + 1) * cols]
                    });
                    quantize_slice(ty, cols, &to_f32(t.source_type, &bytes)?, w, plan.quantizer)
                }
            }
            .map_err(|e| inference_tensor::Error::Msg(format!("{}: {e}", t.name)))?;
            writer.write_all(&out)?;
            written += out.len();
        }
        if written != t.size {
            inference_tensor::bail!("{}: wrote {written} bytes, planned {}", t.name, t.size);
        }
        write_tensor_padding(&mut writer, t.size)?;
        summary.bytes_in += t.source_bytes as u64;
        summary.bytes_out += t.size as u64;
        let entry = by_type
            .entry(t.target.map_or(t.source_type, |ty| ty.id()))
            .or_default();
        entry.0 += 1;
        entry.1 += t.size as u64;
    }
    writer
        .into_inner()
        .map_err(|e| e.into_error())?
        .sync_all()?;
    let mut by_type = by_type
        .into_iter()
        .map(|(id, (n, bytes))| (GgufDType::new(id).name().to_string(), n, bytes))
        .collect::<Vec<_>>();
    by_type.sort_by_key(|(_, _, bytes)| std::cmp::Reverse(*bytes));
    summary.by_type = by_type;
    Ok(summary)
}

/// The float tensors of a safetensors file that `keep` selects, copied as they are (F32, F16, BF16) into a GGUF with
/// `metadata`, each recorded with the shape `dims` gives it (its own, or the same elements reshaped); returns how many
/// were written.
pub fn write_safetensors_gguf(
    src: &Path,
    dst: &Path,
    keep: impl Fn(&str) -> bool,
    dims: impl Fn(&str, &[usize]) -> Vec<usize>,
    metadata: &[(&str, &gguf_file::Value)],
) -> Result<usize> {
    use safetensors::Dtype;
    // SAFETY: the file is only read, and not expected to change while it is mapped
    let st = unsafe { inference_tensor::safetensors::MmapedSafetensors::new(src)? };
    let mut tensors = st
        .tensors()
        .into_iter()
        .filter(|(name, _)| keep(name))
        .collect::<Vec<_>>();
    tensors.sort_by(|a, b| a.0.cmp(&b.0));
    let ids = tensors
        .iter()
        .map(|(name, view)| match view.dtype() {
            Dtype::F32 => Ok(GgufType::F32.id()),
            Dtype::F16 => Ok(GgufType::F16.id()),
            Dtype::BF16 => Ok(GgufType::BF16.id()),
            other => inference_tensor::bail!("{name}: {other:?} tensors have no GGUF float type"),
        })
        .collect::<Result<Vec<_>>>()?;
    let shapes = tensors
        .iter()
        .map(|(name, view)| {
            let shape = dims(name, view.shape());
            if shape.iter().product::<usize>() != view.shape().iter().product::<usize>() {
                inference_tensor::bail!(
                    "{name}: {:?} does not hold its {:?} elements",
                    shape,
                    view.shape()
                );
            }
            Ok(shape)
        })
        .collect::<Result<Vec<_>>>()?;
    let entries = tensors
        .iter()
        .zip(&ids)
        .zip(&shapes)
        .map(|(((name, view), &ggml_dtype), dims)| TensorEntry {
            name,
            dims,
            ggml_dtype,
            size_in_bytes: view.data().len(),
        })
        .collect::<Vec<_>>();
    let mut writer = BufWriter::new(File::create(dst)?);
    write_header(
        &mut writer,
        metadata.len(),
        &gguf_file::metadata_bytes(metadata)?,
        &entries,
    )?;
    for (_, view) in &tensors {
        writer.write_all(view.data())?;
        write_tensor_padding(&mut writer, view.data().len())?;
    }
    writer
        .into_inner()
        .map_err(|e| e.into_error())?
        .sync_all()?;
    Ok(tensors.len())
}

/// llama.cpp's imatrix, legacy `.dat` or GGUF (`<tensor>.in_sum2` over `<tensor>.counts` per expert), as per-tensor
/// mean squared activations.
pub fn load_imatrix(path: &Path) -> Result<HashMap<String, Vec<f32>>> {
    if path.extension().is_none_or(|e| e != "gguf") {
        return inference_tensor::quantized::imatrix_file::load_imatrix(path);
    }
    let mut reader = BufReader::new(File::open(path)?);
    let content = gguf_file::Content::read(&mut reader)?;
    let mut out = HashMap::new();
    for name in content.tensor_infos.keys() {
        let Some(tensor) = name.strip_suffix(".in_sum2") else {
            continue;
        };
        let read = |name: &str, reader: &mut BufReader<File>| -> Result<Vec<f32>> {
            content
                .tensor(reader, name, &Device::Cpu)?
                .dequantize(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()
        };
        let sums = read(name, &mut reader)?;
        let counts = read(&format!("{tensor}.counts"), &mut reader)?;
        let per = sums.len() / counts.len().max(1);
        let values = sums
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let c = counts[i / per];
                // llama.cpp stores 1 where an expert saw no tokens
                if c == 0.0 { 1.0 } else { s / c }
            })
            .collect();
        out.insert(tensor.to_string(), values);
    }
    Ok(out)
}

/// `entries` as llama.cpp's legacy `.dat` imatrix (one call per entry, so the values are read back as they are).
pub fn save_legacy_imatrix(
    entries: &HashMap<String, Vec<f32>>,
    dataset: &str,
    path: &Path,
) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    let as_i32 = |n: usize| i32::try_from(n).map_err(inference_tensor::Error::msg);
    let mut names = entries.keys().collect::<Vec<_>>();
    names.sort();
    w.write_all(&as_i32(names.len())?.to_le_bytes())?;
    for name in names {
        let values = &entries[name];
        w.write_all(&as_i32(name.len())?.to_le_bytes())?;
        w.write_all(name.as_bytes())?;
        w.write_all(&1i32.to_le_bytes())?;
        w.write_all(&as_i32(values.len())?.to_le_bytes())?;
        for v in values {
            w.write_all(&v.to_le_bytes())?;
        }
    }
    // the chunk count, then the dataset's name
    w.write_all(&1i32.to_le_bytes())?;
    w.write_all(&as_i32(dataset.len())?.to_le_bytes())?;
    w.write_all(dataset.as_bytes())?;
    w.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use inference_tensor::quantized::gguf_file::Value;

    const PIN_ROWS: usize = 8;
    const PIN_COLS: usize = 512;

    // a reproducible layer: small values with sparse outliers, and a skewed imatrix (the same LCG regenerates them
    // outside Rust to check the pins against ik_llama.cpp's own quantizer)
    fn pinned_layer() -> (Vec<f32>, Vec<f32>) {
        let mut state = 12_345u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / (1u32 << 24) as f32, state)
        };
        let values = (0..PIN_ROWS * PIN_COLS)
            .map(|_| {
                let (u, s) = next();
                let scale = if (s >> 4) & 31 == 0 { 0.4 } else { 0.04 };
                (u - 0.5) * scale
            })
            .collect();
        let imatrix = (0..PIN_COLS)
            .map(|_| 0.05 + 4.0 * next().0.powi(3))
            .collect();
        (values, imatrix)
    }

    fn fnv1a(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
        })
    }

    // ik_llama.cpp's quantizer, as its AVX2 build (gcc, -march=native) writes these bytes: the pins are its own
    // `ggml_quantize_chunk`'s on the same layer (scripts/iq_quantize_parity.py compares more against a build of it)
    #[test]
    fn the_ik_quantizer_writes_ik_llamas_bytes() -> Result<()> {
        let (values, imatrix) = pinned_layer();
        let pins: Vec<(&str, u64, u64)> = PINNED_IK.into_iter().chain(PINNED_IK_GRID).collect();
        let mut got = Vec::new();
        for &(name, _, _) in &pins {
            let ty = writable_type(name)?;
            let hash = |w| {
                quantize_matrix(ty, PIN_COLS, &values, w, PIN_ROWS, Quantizer::IkLlama)
                    .map(|b| fnv1a(&b))
            };
            // the types that need an imatrix have no plain bytes to pin
            let plain = if iq_quantize::requires_imatrix(ty) {
                0
            } else {
                hash(None)?
            };
            got.push((name, plain, hash(Some(&imatrix))?));
        }
        assert_eq!(got, pins);
        Ok(())
    }

    const PINNED_IK_GRID: [(&str, u64, u64); 7] = [
        ("iq2_xxs", 0, 3960069783596285320),
        ("iq2_xs", 0, 3351784405228797559),
        ("iq2_s", 1979530409175927814, 173424413650317151),
        ("iq3_xxs", 6333467982077396785, 267139717862269199),
        ("iq3_s", 5272524117869529413, 3851133074760413839),
        ("iq1_s", 0, 15644970516814970719),
        ("iq1_m", 4877935477014574425, 546727153653379684),
    ];

    const PINNED_IK: [(&str, u64, u64); 15] = [
        ("q2_k", 13302739417286923110, 11601392929549359756),
        ("q3_k", 16392672201002081248, 17943740573164000344),
        ("q4_k", 14679928973253263191, 14544912929349767743),
        ("q5_k", 15584132874035254720, 5277963835315783023),
        ("q6_k", 15677125580282521591, 18396251129581993316),
        ("q4_0", 10936624288346114449, 11794455730010133618),
        ("q4_1", 10411377467337899343, 2434393147449071549),
        ("q5_0", 337873762312862314, 16479908197838349031),
        ("q5_1", 1211858028360391952, 13589153157721327241),
        ("q8_0", 11372417339874904451, 11372417339874904451),
        ("iq4_nl", 9596728955645266052, 12830532639945862795),
        ("iq4_xs", 1175151790720974940, 8216451137976519682),
        ("iq4_ks", 15587783422239331424, 2233139334062329897),
        ("iq4_kss", 6028622500172932859, 11259105053868730158),
        ("iq3_ks", 9489437392515969846, 7909008254536332893),
    ];

    #[test]
    fn a_rewritten_gguf_keeps_metadata_and_quantizes_by_rule() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let (src, dst) = (dir.path().join("src.gguf"), dir.path().join("dst.gguf"));
        let cols = 512;
        let values = (0..4 * cols)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 64.0)
            .collect::<Vec<_>>();
        let w = Tensor::from_slice(&values, (4, cols), &Device::Cpu)?;
        let experts = Tensor::from_slice(&values, (2, 2, cols), &Device::Cpu)?;
        let norm = Tensor::ones(cols, inference_tensor::DType::F32, &Device::Cpu)?;
        let tensors = [
            (
                "blk.0.attn_q.weight",
                QTensor::quantize(&w, GgmlDType::BF16)?,
            ),
            (
                "blk.0.ffn_up_exps.weight",
                QTensor::quantize(&experts, GgmlDType::BF16)?,
            ),
            (
                "blk.0.attn_norm.weight",
                QTensor::quantize(&norm, GgmlDType::F32)?,
            ),
        ];
        let arch = Value::String("tiny".into());
        let tokens = Value::Array((0..100).map(|i| Value::String(format!("t{i}"))).collect());
        let mut file = File::create(&src)?;
        gguf_file::write(
            &mut file,
            &[
                ("general.architecture", &arch),
                ("tokenizer.ggml.tokens", &tokens),
            ],
            &tensors.iter().map(|(n, t)| (*n, t)).collect::<Vec<_>>(),
        )?;
        drop(file);

        let imatrix = HashMap::from([(
            "blk.0.ffn_up_exps.weight".to_string(),
            (0..2 * cols)
                .map(|i| 1.0 + (i % 7) as f32)
                .collect::<Vec<_>>(),
        )]);
        let plan = RequantizePlan {
            rules: vec![
                (
                    Regex::new(r"_exps\.weight$").unwrap(),
                    writable_type("iq4_ks")?,
                ),
                (Regex::new(r".*").unwrap(), writable_type("q8_0")?),
            ],
            imatrix: Some(imatrix.clone()),
            quantizer: Quantizer::LlamaCpp,
            reuse: Vec::new(),
            only: None,
            gpu: None,
            hessians: None,
            hessian_weights_only: false,
            gptq_damp: 0.0,
        };
        let summary = requantize_gguf(&src, &dst, &plan, |_, _, _| {})?;
        assert_eq!(summary.quantized, 2);
        assert_eq!(summary.without_imatrix, ["blk.0.attn_q.weight"]);
        // a second run reusing the first copies both quantized tensors and writes the same file
        let again = dst.with_extension("again.gguf");
        let plan = RequantizePlan {
            reuse: vec![dst.clone()],
            ..plan
        };
        let summary_again = requantize_gguf(&src, &again, &plan, |_, _, _| {})?;
        assert_eq!(summary_again.reused, 2);
        assert_eq!(std::fs::read(&again)?, std::fs::read(&dst)?);
        // an overlay of one tensor at another type replaces that tensor when listed after the model
        let overlay = dst.with_extension("overlay.gguf");
        let overlaying = RequantizePlan {
            rules: vec![(Regex::new(r".*").unwrap(), writable_type("f16")?)],
            imatrix: None,
            quantizer: Quantizer::LlamaCpp,
            reuse: Vec::new(),
            only: Some(Regex::new(r"attn_q\.weight$").unwrap()),
            gpu: None,
            hessians: None,
            hessian_weights_only: false,
            gptq_damp: 0.0,
        };
        let overlaid = requantize_gguf(&src, &overlay, &overlaying, |_, _, _| {})?;
        assert_eq!(overlaid.tensors, 1);
        let archive = super::super::archive::GgufArchive::open([&dst, &overlay])?;
        let q = &archive.tensors()["blk.0.attn_q.weight"];
        assert_eq!((q.dtype().raw(), q.shard_index()), (1, 1));
        assert_eq!(
            archive.tensors()["blk.0.ffn_up_exps.weight"].shard_index(),
            0
        );

        let mut reader = BufReader::new(File::open(&dst)?);
        let header = Header::read(&mut reader)?;
        assert_eq!(header.tensor_order, tensors.map(|(n, _)| n.to_string()));
        assert_eq!(header.metadata["general.architecture"].to_string()?, "tiny");
        assert_eq!(
            header.metadata["tokenizer.ggml.tokens"].to_vec()?.len(),
            100
        );
        let ids = |n: &str| header.tensor_infos[n].ggml_dtype;
        assert_eq!(ids("blk.0.attn_q.weight"), GgufType::Q8_0.id());
        assert_eq!(ids("blk.0.ffn_up_exps.weight"), GgufType::Iq4Ks.id());
        assert_eq!(ids("blk.0.attn_norm.weight"), 0);

        let info = &header.tensor_infos["blk.0.ffn_up_exps.weight"];
        let mut bytes = vec![0u8; summary.bytes_out as usize];
        reader.seek(SeekFrom::Start(header.tensor_data_offset + info.offset))?;
        let row = GgufType::Iq4Ks.row_bytes(cols).unwrap();
        reader.read_exact(&mut bytes[..4 * row])?;
        let w = &imatrix["blk.0.ffn_up_exps.weight"];
        let expected = quantize_matrix(
            GgufType::Iq4Ks,
            cols,
            &values,
            Some(w),
            2,
            Quantizer::LlamaCpp,
        )?;
        assert_eq!(&bytes[..4 * row], expected.as_slice());
        // the second expert's rows weighed by the second slice, not the first
        let first_only = iq_quantize::quantize_rows(
            GgufType::Iq4Ks,
            cols,
            &values,
            Some(&w[..cols]),
            Quantizer::LlamaCpp,
        )?;
        assert_ne!(&bytes[2 * row..4 * row], &first_only[2 * row..]);

        let dat = dir.path().join("imatrix.dat");
        save_legacy_imatrix(&imatrix, "synthetic", &dat)?;
        assert_eq!(load_imatrix(&dat)?, imatrix);
        Ok(())
    }
}
