use std::collections::HashMap;
use std::io::{Read, Seek, Write};
use std::path::Path;

use inference_tensor::nn::var_builder::SimpleBackend;
use inference_tensor::nn::{Init, VarBuilder};
use inference_tensor::quantized::{GgmlDType, QTensor, gguf_file};
use inference_tensor::safetensors::MmapedSafetensors;
use inference_tensor::{DType, Device, Result, Shape, Tensor, bail};

pub const ARCHITECTURE: &str = "pp_doclayout_v3";
const ARCHITECTURE_KEY: &str = "general.architecture";
const NAME_KEY: &str = "general.name";
const MODEL_NAME: &str = "PP-DocLayoutV3";
const CONFIG_KEY: &str = "pp_doclayout_v3.config";
const PREPROCESSOR_CONFIG_KEY: &str = "pp_doclayout_v3.preprocessor_config";
// llama.cpp's GGML_MAX_NAME is 64 with the terminating NUL; the HF names reach 88, so segments are shortened.
const MAX_TENSOR_NAME: usize = 63;
// whole dotted segments only, and no short form is a segment of the checkpoint, so the mapping inverts exactly
const SHORT_SEGMENTS: &[(&str, &str)] = &[
    ("backbone", "bb"),
    ("encoder", "enc"),
    ("stages", "stg"),
    ("blocks", "blk"),
    ("layers", "lyr"),
    ("normalization", "bn"),
    ("running_mean", "rmean"),
    ("running_var", "rvar"),
    ("convolution", "cv"),
    ("bottlenecks", "btl"),
];

/// The model's configs as JSON and its F32 weights, read from a GGUF `write_gguf` made.
pub struct GgufCheckpoint {
    pub config: String,
    pub preprocessor_config: String,
    pub weights: VarBuilder<'static>,
}

/// Weights by HF name; a convolution kernel is restored from the (out, in * kh * kw) matrix it is stored as.
struct GgufWeights(HashMap<String, Tensor>);

impl SimpleBackend for GgufWeights {
    fn get(&self, s: Shape, name: &str, h: Init, dtype: DType, dev: &Device) -> Result<Tensor> {
        if let [out, kernel @ ..] = s.dims()
            && kernel.len() > 1
            && let Some(t) = self.0.get(name)
            && t.dims() == [*out, kernel.iter().product()]
        {
            return t.reshape(s)?.to_device(dev)?.to_dtype(dtype);
        }
        SimpleBackend::get(&self.0, s, name, h, dtype, dev)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> Result<Tensor> {
        self.0.get_unchecked(name, dtype, dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.0.contains_key(name)
    }
}

fn rename(name: &str, map: impl Fn(&str) -> Option<&'static str>) -> String {
    name.split('.')
        .map(|segment| map(segment).unwrap_or(segment))
        .collect::<Vec<_>>()
        .join(".")
}

fn gguf_name(hf_name: &str) -> String {
    rename(hf_name, |s| {
        SHORT_SEGMENTS.iter().find(|p| p.0 == s).map(|p| p.1)
    })
}

fn hf_name(gguf_name: &str) -> String {
    rename(gguf_name, |s| {
        SHORT_SEGMENTS.iter().find(|p| p.1 == s).map(|p| p.0)
    })
}

// Vectors (norms, biases, BN statistics) stay F32, and a block type the row length doesn't divide falls back to F16.
fn storage_dtype(dtype: GgmlDType, dims: &[usize]) -> GgmlDType {
    match dims {
        [] | [_] => GgmlDType::F32,
        [.., row] if row % dtype.block_size() != 0 => GgmlDType::F16,
        _ => dtype,
    }
}

/// Writes the HF `PP-DocLayoutV3_safetensors` directory `dir` as one GGUF, its matrices and kernels in `dtype`.
pub fn write_gguf<W: Write + Seek>(dir: &Path, w: &mut W, dtype: GgmlDType) -> Result<()> {
    let read =
        |name: &str| std::fs::read_to_string(dir.join(name)).map_err(inference_tensor::Error::wrap);
    // SAFETY: the weight file is mmapped read-only for the duration of the conversion.
    let st = unsafe { MmapedSafetensors::new(dir.join("model.safetensors"))? };
    let tensors = st
        .tensors()
        .into_iter()
        .map(|(name, _)| Ok((st.load(&name, &Device::Cpu)?, name)))
        .collect::<Result<Vec<_>>>()?;
    write_checkpoint(
        w,
        read("config.json")?,
        read("preprocessor_config.json")?,
        tensors,
        dtype,
    )
}

fn write_checkpoint<W: Write + Seek>(
    w: &mut W,
    config: String,
    preprocessor_config: String,
    weights: Vec<(Tensor, String)>,
    dtype: GgmlDType,
) -> Result<()> {
    let mut tensors = Vec::with_capacity(weights.len());
    for (t, name) in weights {
        let short = gguf_name(&name);
        if short.len() > MAX_TENSOR_NAME {
            bail!("tensor name {short} is longer than {MAX_TENSOR_NAME} bytes")
        }
        if hf_name(&short) != name {
            bail!(
                "tensor name {name} has a segment that is a short form, so it would not read back"
            )
        }
        // ggml blocks run along the innermost dim, a kernel's kw, so kernels are stored flat for the block types
        let t = if t.rank() > 2 { t.flatten_from(1)? } else { t };
        tensors.push((
            short,
            QTensor::quantize(&t, storage_dtype(dtype, t.dims()))?,
        ));
    }
    tensors.sort_by(|a, b| a.0.cmp(&b.0));
    let values = [
        (
            ARCHITECTURE_KEY,
            gguf_file::Value::String(ARCHITECTURE.into()),
        ),
        (NAME_KEY, gguf_file::Value::String(MODEL_NAME.into())),
        (CONFIG_KEY, gguf_file::Value::String(config)),
        (
            PREPROCESSOR_CONFIG_KEY,
            gguf_file::Value::String(preprocessor_config),
        ),
    ];
    let metadata = values.iter().map(|(k, v)| (*k, v)).collect::<Vec<_>>();
    let tensors = tensors
        .iter()
        .map(|(n, q)| (n.as_str(), q))
        .collect::<Vec<_>>();
    gguf_file::write(w, &metadata, &tensors)
}

/// Reads a GGUF `write_gguf` made, dequantizing every weight to F32 on `device`.
pub fn read_gguf<R: Read + Seek>(r: &mut R, device: &Device) -> Result<GgufCheckpoint> {
    let content = gguf_file::Content::read(r)?;
    let text = |key: &str| match content.metadata.get(key) {
        Some(v) => v.to_string().cloned(),
        None => bail!("not a {MODEL_NAME} GGUF: no {key}"),
    };
    let arch = text(ARCHITECTURE_KEY)?;
    if arch != ARCHITECTURE {
        bail!("GGUF architecture is {arch}, expected {ARCHITECTURE}")
    }
    let mut tensors = HashMap::with_capacity(content.tensor_infos.len());
    for name in content.tensor_infos.keys() {
        let t = content.tensor(r, name, &Device::Cpu)?.dequantize(device)?;
        tensors.insert(hf_name(name), t);
    }
    Ok(GgufCheckpoint {
        config: text(CONFIG_KEY)?,
        preprocessor_config: text(PREPROCESSOR_CONFIG_KEY)?,
        weights: VarBuilder::from_backend(
            Box::new(GgufWeights(tensors)),
            DType::F32,
            device.clone(),
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_names_invert() {
        let longs = SHORT_SEGMENTS.iter().map(|p| p.0).collect::<Vec<_>>();
        for (_, short) in SHORT_SEGMENTS {
            assert!(!longs.contains(short), "{short} is also a long segment");
        }
        let name = "model.backbone.model.encoder.stages.3.blocks.0.layers.5.conv2.normalization.running_mean";
        assert_eq!(
            gguf_name(name),
            "model.bb.model.enc.stg.3.blk.0.lyr.5.conv2.bn.rmean"
        );
        assert_eq!(hf_name(&gguf_name(name)), name);
    }

    #[test]
    fn vectors_and_ragged_rows_keep_precision() {
        assert_eq!(storage_dtype(GgmlDType::Q8_0, &[256]), GgmlDType::F32);
        assert_eq!(
            storage_dtype(GgmlDType::Q8_0, &[64, 3 * 3 * 3]),
            GgmlDType::F16
        );
        assert_eq!(
            storage_dtype(GgmlDType::Q8_0, &[64, 32 * 3 * 3]),
            GgmlDType::Q8_0
        );
        assert_eq!(storage_dtype(GgmlDType::F16, &[64, 27]), GgmlDType::F16);
    }

    fn round_trip(weights: Vec<(Tensor, String)>, dtype: GgmlDType) -> Result<GgufCheckpoint> {
        let mut buf = std::io::Cursor::new(Vec::new());
        write_checkpoint(&mut buf, "{}".into(), "[]".into(), weights, dtype)?;
        buf.set_position(0);
        read_gguf(&mut buf, &Device::Cpu)
    }

    #[test]
    fn checkpoints_round_trip() -> Result<()> {
        let dev = Device::Cpu;
        // values in [0, 1), so Q8_0's rounding stays under half of 1/127
        let n = (64 * 32 * 3 * 3) as f64;
        let kernel = (Tensor::arange(0f32, n as f32, &dev)?.reshape((64, 32, 3, 3))? / n)?;
        let vector = Tensor::new(&[0.1f32, -2.5, 1e-7], &dev)?;
        let name =
            "model.backbone.model.encoder.stages.3.blocks.0.layers.5.conv2.convolution.weight";
        let ckpt = round_trip(
            vec![
                (kernel.clone(), name.into()),
                (vector.clone(), "model.norm.bias".into()),
            ],
            GgmlDType::Q8_0,
        )?;
        assert_eq!(
            (ckpt.config.as_str(), ckpt.preprocessor_config.as_str()),
            ("{}", "[]")
        );
        let restored = ckpt.weights.get((64, 32, 3, 3), name)?;
        let err = (restored - &kernel)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(err < 0.01, "Q8_0 kernel error {err}");
        let bias = ckpt.weights.get(3, "model.norm.bias")?.to_vec1::<f32>()?;
        assert_eq!(bias, vector.to_vec1::<f32>()?);
        Ok(())
    }

    #[test]
    fn unreadable_names_and_foreign_files_are_refused() -> Result<()> {
        let v = Tensor::zeros(2, DType::F32, &Device::Cpu)?;
        assert!(round_trip(vec![(v.clone(), "model.bn.weight".into())], GgmlDType::F32).is_err());
        let long = format!("model.{}.weight", "x".repeat(MAX_TENSOR_NAME));
        assert!(round_trip(vec![(v, long)], GgmlDType::F32).is_err());

        let mut buf = std::io::Cursor::new(Vec::new());
        let arch = gguf_file::Value::String("llama".into());
        gguf_file::write(&mut buf, &[(ARCHITECTURE_KEY, &arch)], &[])?;
        buf.set_position(0);
        let err = read_gguf(&mut buf, &Device::Cpu).err().unwrap().to_string();
        assert!(err.contains("architecture is llama"), "{err}");
        assert!(
            read_gguf(
                &mut std::io::Cursor::new(b"not a gguf".to_vec()),
                &Device::Cpu
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn kernels_come_back_in_their_shape() -> Result<()> {
        let kernel = Tensor::arange(0f32, 24., &Device::Cpu)?.reshape((2, 3, 2, 2))?;
        let weights = GgufWeights(HashMap::from([
            ("conv.weight".to_string(), kernel.flatten_from(1)?),
            (
                "conv.bias".to_string(),
                Tensor::zeros(2, DType::F32, &Device::Cpu)?,
            ),
        ]));
        let vb = VarBuilder::from_backend(Box::new(weights), DType::F32, Device::Cpu);
        let restored = vb.get((2, 3, 2, 2), "conv.weight")?;
        assert_eq!(restored.dims(), [2, 3, 2, 2]);
        assert_eq!(
            restored.flatten_all()?.to_vec1::<f32>()?,
            kernel.flatten_all()?.to_vec1::<f32>()?
        );
        assert!(vb.get((3, 2, 2, 2), "conv.weight").is_err());
        assert_eq!(vb.get(2, "conv.bias")?.dims(), [2]);
        Ok(())
    }
}
