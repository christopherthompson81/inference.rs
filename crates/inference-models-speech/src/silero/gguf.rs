use std::collections::HashMap;
use std::io::{Read, Seek, Write};
use std::path::Path;

use inference_tensor::nn::VarBuilder;
use inference_tensor::quantized::{GgmlDType, QTensor, gguf_file};
use inference_tensor::{DType, Device, Result, Tensor, bail};

pub const ARCHITECTURE: &str = "silero_vad";
const ARCHITECTURE_KEY: &str = "general.architecture";
const NAME_KEY: &str = "general.name";
const SOURCE_KEY: &str = "general.source.url";
const SAMPLE_RATE_KEY: &str = "silero_vad.sample_rate";
const CHUNK_KEY: &str = "silero_vad.chunk_size";
const CONTEXT_KEY: &str = "silero_vad.context_size";
const MODEL_NAME: &str = "Silero VAD";
const SOURCE_URL: &str = "https://github.com/snakers4/silero-vad";
const SAMPLE_RATE: u32 = 16_000;
const CHUNK_SIZE: usize = 512;
const CONTEXT_SIZE: usize = 64;
// mlx-community's conversion prefixes every tensor and stores its kernels (out, k, in)
const MLX_PREFIX: &str = "vad_16k.";
const GGUF_EXTENSION: &str = "gguf";

/// The 16 kHz model's framing.
#[derive(Debug, Clone, PartialEq)]
pub struct VadConfig {
    pub sample_rate: u32,
    pub chunk_size: usize,
    pub context_size: usize,
}

/// Whether `path` is a Silero VAD GGUF.
pub fn is_silero_gguf(path: &Path) -> bool {
    let read = || -> Result<bool> {
        let mut file = std::io::BufReader::new(std::fs::File::open(path)?);
        Ok(gguf_file::peek_architecture(&mut file)?.is_some_and(|a| a == ARCHITECTURE))
    };
    path.extension().is_some_and(|e| e == GGUF_EXTENSION)
        && path.is_file()
        && read().unwrap_or(false)
}

// one of the two published layouts to PyTorch's (out, in, k) kernels and separate LSTM biases
fn pytorch_layout(tensors: HashMap<String, Tensor>) -> Result<Vec<(String, Tensor)>> {
    if !tensors.keys().any(|k| k.starts_with(MLX_PREFIX)) {
        return Ok(tensors.into_iter().collect());
    }
    let mut out = Vec::new();
    for (name, t) in tensors {
        let name = name.trim_start_matches(MLX_PREFIX).to_string();
        match name.as_str() {
            "lstm.Wx" => out.push(("lstm_cell.weight_ih".to_string(), t)),
            "lstm.Wh" => out.push(("lstm_cell.weight_hh".to_string(), t)),
            "lstm.bias" => {
                // the conversion summed the two biases; the cell only ever adds both
                out.push(("lstm_cell.bias_hh".to_string(), t.zeros_like()?));
                out.push(("lstm_cell.bias_ih".to_string(), t));
            }
            _ if t.rank() == 3 => out.push((name, t.transpose(1, 2)?.contiguous()?)),
            _ => out.push((name, t)),
        }
    }
    Ok(out)
}

/// Writes Silero VAD's 16 kHz weights (mlx-community's `model.safetensors`, or the PyPI package's
/// `silero_vad_16k.safetensors`) as an F32 GGUF.
pub fn write_gguf<W: Write + Seek>(safetensors: &Path, w: &mut W) -> Result<()> {
    let tensors = pytorch_layout(inference_tensor::safetensors::load(
        safetensors,
        &Device::Cpu,
    )?)?;
    let mut tensors = tensors
        .into_iter()
        .map(|(name, t)| {
            Ok((
                name,
                QTensor::quantize(&t.to_dtype(DType::F32)?, GgmlDType::F32)?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    tensors.sort_by(|a, b| a.0.cmp(&b.0));
    let values = [
        (
            ARCHITECTURE_KEY,
            gguf_file::Value::String(ARCHITECTURE.into()),
        ),
        (NAME_KEY, gguf_file::Value::String(MODEL_NAME.into())),
        (SOURCE_KEY, gguf_file::Value::String(SOURCE_URL.into())),
        (SAMPLE_RATE_KEY, gguf_file::Value::U32(SAMPLE_RATE)),
        (CHUNK_KEY, gguf_file::Value::U32(CHUNK_SIZE as u32)),
        (CONTEXT_KEY, gguf_file::Value::U32(CONTEXT_SIZE as u32)),
    ];
    let metadata = values.iter().map(|(k, v)| (*k, v)).collect::<Vec<_>>();
    let tensors = tensors
        .iter()
        .map(|(n, q)| (n.as_str(), q))
        .collect::<Vec<_>>();
    gguf_file::write(w, &metadata, &tensors)
}

/// The config and F32 weights on `device` of a GGUF `write_gguf` made.
pub fn read_gguf<R: Read + Seek>(
    r: &mut R,
    device: &Device,
) -> Result<(VadConfig, VarBuilder<'static>)> {
    let content = gguf_file::Content::read(r)?;
    let get = |key: &str| {
        content.metadata.get(key).ok_or_else(|| {
            inference_tensor::Error::Msg(format!("not a {MODEL_NAME} GGUF: no {key}"))
        })
    };
    let arch = get(ARCHITECTURE_KEY)?.to_string()?;
    if arch != ARCHITECTURE {
        bail!("GGUF architecture is {arch}, expected {ARCHITECTURE}")
    }
    let config = VadConfig {
        sample_rate: get(SAMPLE_RATE_KEY)?.to_u32()?,
        chunk_size: get(CHUNK_KEY)?.to_u32()? as usize,
        context_size: get(CONTEXT_KEY)?.to_u32()? as usize,
    };
    let mut tensors = HashMap::with_capacity(content.tensor_infos.len());
    for name in content.tensor_infos.keys() {
        tensors.insert(
            name.clone(),
            content.tensor(r, name, &Device::Cpu)?.dequantize(device)?,
        );
    }
    Ok((
        config,
        VarBuilder::from_tensors(tensors, DType::F32, device),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    // mlx-community's layout (prefixed names, (out, k, in) kernels, one summed bias) reads back in PyTorch's
    #[test]
    fn the_mlx_layout_converts_and_reads_back() -> Result<()> {
        let dev = Device::Cpu;
        let kernel = Tensor::arange(0f32, 24., &dev)?.reshape((2, 3, 4))?;
        let bias = Tensor::new(&[1f32, 2.], &dev)?;
        let mlx: HashMap<String, Tensor> = [
            (format!("{MLX_PREFIX}conv1.weight"), kernel.clone()),
            (format!("{MLX_PREFIX}lstm.bias"), bias.clone()),
        ]
        .into_iter()
        .collect();
        let dir = tempfile::tempdir().map_err(inference_tensor::Error::wrap)?;
        let st = dir.path().join("model.safetensors");
        inference_tensor::safetensors::save(&mlx, &st)?;
        let mut buf = std::io::Cursor::new(Vec::new());
        write_gguf(&st, &mut buf)?;
        buf.set_position(0);
        let (config, vb) = read_gguf(&mut buf, &dev)?;
        assert_eq!(
            (config.sample_rate, config.chunk_size, config.context_size),
            (SAMPLE_RATE, CHUNK_SIZE, CONTEXT_SIZE)
        );
        let conv = vb.get((2, 4, 3), "conv1.weight")?;
        assert_eq!(
            conv.to_vec3::<f32>()?,
            kernel.transpose(1, 2)?.to_vec3::<f32>()?
        );
        let summed = (vb.get(2, "lstm_cell.bias_ih")? + vb.get(2, "lstm_cell.bias_hh")?)?;
        assert_eq!(summed.to_vec1::<f32>()?, bias.to_vec1::<f32>()?);
        Ok(())
    }
}
