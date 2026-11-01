//! A float GGUF to a quantized one: each tensor's type from the first rule whose pattern matches its name, the
//! imatrix (llama.cpp's, keyed by GGUF names) weighing the encoders that take one. Streamed a slice at a time (an
//! expert, or a bounded run of rows), so memory stays at one slice whatever the model's size.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use half::{bf16, f16};
use inference_tensor::quantized::gguf_file::{
    self, Header, TensorEntry, write_header, write_tensor_padding,
};
use inference_tensor::quantized::{GgmlDType, QTensor};
use inference_tensor::{Device, Result, Tensor};
use rayon::prelude::*;
use regex::Regex;

use super::archive::GgufDType;
use super::iq_quantize;
use super::kernel::GgufType;

// rows a parallel task quantizes through `QTensor`
const ROWS_PER_TASK: usize = 64;
// the source bytes read at once: one expert of an expert stack, else rows up to this
const SLICE_BYTES: usize = 64 << 20;
const GGUF_ALIGNMENT: u64 = 32;

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
        out.extend(quantize_slice(ty, cols, slice, w)?);
    }
    Ok(out)
}

fn quantize_slice(
    ty: GgufType,
    cols: usize,
    values: &[f32],
    imatrix: Option<&[f32]>,
) -> Result<Vec<u8>> {
    if iq_quantize::can_quantize(ty) {
        return iq_quantize::quantize_rows(ty, cols, values, imatrix);
    }
    let Some(dtype) = ty.candle() else {
        inference_tensor::bail!("{ty:?} has no quantizer");
    };
    // ggml's Q8_0 and float types ignore an imatrix
    let weighted = !matches!(
        dtype,
        GgmlDType::Q8_0
            | GgmlDType::Q8_1
            | GgmlDType::Q8K
            | GgmlDType::F32
            | GgmlDType::F16
            | GgmlDType::BF16
    );
    let chunks = values
        .par_chunks(ROWS_PER_TASK * cols)
        .map(|chunk| {
            let t = Tensor::from_slice(chunk, (chunk.len() / cols, cols), &Device::Cpu)?;
            let q = match imatrix.filter(|_| weighted) {
                Some(w) => QTensor::quantize_imatrix(&t, w, dtype)?,
                None => QTensor::quantize(&t, dtype)?,
            };
            Ok(q.data()?.into_owned())
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(chunks.concat())
}

/// The rules for which tensors become which type, in order: the first pattern matching a name wins, and a tensor
/// no pattern matches is copied as it is.
pub struct RequantizePlan {
    pub rules: Vec<(Regex, GgufType)>,
    /// llama.cpp's per-tensor importances (mean squared activations, an expert stack's experts one after another).
    pub imatrix: Option<HashMap<String, Vec<f32>>>,
}

#[derive(Debug, Default)]
pub struct RequantizeSummary {
    pub tensors: usize,
    pub quantized: usize,
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
        .map(|name| plan_tensor(name, &header, plan))
        .collect::<Result<Vec<_>>>()?;

    let mut kv_bytes = vec![0u8; (header.metadata_span.end - header.metadata_span.start) as usize];
    reader.seek(SeekFrom::Start(header.metadata_span.start))?;
    reader.read_exact(&mut kv_bytes)?;
    let kv_count = {
        reader.seek(SeekFrom::Start(16))?;
        let mut count = [0u8; 8];
        reader.read_exact(&mut count)?;
        u64::from_le_bytes(count) as usize
    };
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
        let slice_rows =
            expert_rows.unwrap_or_else(|| (SLICE_BYTES / source_row.max(1)).clamp(1, rows.max(1)));
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
            let w = imatrix.map(|w| {
                let expert = expert_rows.map_or(0, |e| first / e);
                &w[expert * cols..(expert + 1) * cols]
            });
            let out = quantize_slice(ty, cols, &to_f32(t.source_type, &bytes)?, w)
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
/// `metadata`; returns how many were written.
pub fn write_safetensors_gguf(
    src: &Path,
    dst: &Path,
    keep: impl Fn(&str) -> bool,
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
    let entries = tensors
        .iter()
        .zip(&ids)
        .map(|((name, view), &ggml_dtype)| TensorEntry {
            name,
            dims: view.shape(),
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
        };
        let summary = requantize_gguf(&src, &dst, &plan, |_, _, _| {})?;
        assert_eq!(summary.quantized, 2);
        assert_eq!(summary.without_imatrix, ["blk.0.attn_q.weight"]);

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
        let expected = quantize_matrix(GgufType::Iq4Ks, cols, &values, Some(w), 2)?;
        assert_eq!(&bytes[..4 * row], expected.as_slice());
        // the second expert's rows weighed by the second slice, not the first
        let first_only =
            iq_quantize::quantize_rows(GgufType::Iq4Ks, cols, &values, Some(&w[..cols]))?;
        assert_ne!(&bytes[2 * row..4 * row], &first_only[2 * row..]);

        let dat = dir.path().join("imatrix.dat");
        save_legacy_imatrix(&imatrix, "synthetic", &dat)?;
        assert_eq!(load_imatrix(&dat)?, imatrix);
        Ok(())
    }
}
