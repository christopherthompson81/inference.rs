//! WorldStereo-2 and its umT5 text encoder written as GGUFs: the large matrices quantized, the rest kept fine.

use std::path::{Path, PathBuf};

use inference_tensor::quantized::gguf_file::{self, Value};
use inference_tensor::quantized::{GgmlDType, QTensor};
use inference_tensor::safetensors::MmapedSafetensors;
use inference_tensor::{DType, Device, Result, Tensor};
use rayon::prelude::*;

// the main blocks' matrices, the ones the loader streams and quantizes; the rest stays as the checkpoint stores it
const WORLDSTEREO_BLOCKS: &str = "blocks.";
const WORLDSTEREO_ARCHITECTURE: &str = "worldstereo";
// llama.cpp's and ComfyUI-GGUF's name for a T5 encoder file
const T5_ARCHITECTURE: &str = "t5encoder";
// a [buckets, heads] table looked up per position pair, which quantizing would gain nothing on
const T5_RELATIVE_BIAS: &str = "attn_rel_b";

/// WorldStereo-2's `model.safetensors` in `dir` written to `out`, its main blocks' matrices as `matrices`; returns each
/// tensor's type.
pub fn write_worldstereo_gguf(
    dir: &Path,
    out: &Path,
    matrices: GgmlDType,
) -> Result<Vec<(String, GgmlDType)>> {
    let shards = [dir.join("model.safetensors")];
    write(
        &shards,
        out,
        WORLDSTEREO_ARCHITECTURE,
        |name| Some(name.to_string()),
        |name, t| {
            if name.starts_with(WORLDSTEREO_BLOCKS) && fills_blocks(t, matrices) {
                matrices
            } else {
                stored(t)
            }
        },
    )
}

/// The umT5 encoder's safetensors shards in `dir` written to `out` under llama.cpp's `t5encoder` names, its matrices
/// and token embedding as `matrices` and its norms and position biases in F32; returns each tensor's type.
pub fn write_umt5_gguf(
    dir: &Path,
    out: &Path,
    matrices: GgmlDType,
) -> Result<Vec<(String, GgmlDType)>> {
    let mut shards: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
        .collect();
    shards.sort();
    write(
        &shards,
        out,
        T5_ARCHITECTURE,
        crate::gguf::t5_gguf_name,
        |name, t| {
            if t.rank() == 2 && !name.contains(T5_RELATIVE_BIAS) && fills_blocks(t, matrices) {
                matrices
            } else {
                GgmlDType::F32
            }
        },
    )
}

fn fills_blocks(t: &Tensor, dtype: GgmlDType) -> bool {
    t.rank() == 2 && t.dims()[1].is_multiple_of(dtype.block_size())
}

fn stored(t: &Tensor) -> GgmlDType {
    match t.dtype() {
        DType::BF16 => GgmlDType::BF16,
        DType::F16 => GgmlDType::F16,
        _ => GgmlDType::F32,
    }
}

// Each tensor `gguf_name` keeps, under that name, typed by `dtype_for` from its checkpoint precision
fn write(
    shards: &[PathBuf],
    out: &Path,
    architecture: &str,
    gguf_name: impl Fn(&str) -> Option<String> + Sync,
    dtype_for: impl Fn(&str, &Tensor) -> GgmlDType + Sync,
) -> Result<Vec<(String, GgmlDType)>> {
    if shards.is_empty() {
        inference_tensor::bail!("no safetensors weights to write to `{}`", out.display());
    }
    // SAFETY: the shards are only read, and not changed while the writer runs
    let weights = unsafe { MmapedSafetensors::multi(shards)? };
    let mut names: Vec<(String, String)> = weights
        .tensors()
        .into_iter()
        .filter_map(|(name, _)| Some((gguf_name(&name)?, name)))
        .collect();
    names.sort();
    let tensors = names
        .par_iter()
        .map(|(gguf, name)| {
            let t = weights.load(name, &Device::Cpu)?;
            let dtype = dtype_for(name, &t);
            let q = QTensor::quantize(&t.to_dtype(DType::F32)?, dtype)?;
            Ok((gguf.clone(), q, dtype))
        })
        .collect::<Result<Vec<_>>>()?;
    let metadata = [(
        "general.architecture",
        Value::String(architecture.to_string()),
    )];
    let metadata: Vec<(&str, &Value)> = metadata.iter().map(|(k, v)| (*k, v)).collect();
    let refs: Vec<(&str, &QTensor)> = tensors.iter().map(|(n, q, _)| (n.as_str(), q)).collect();
    let mut file = std::io::BufWriter::new(std::fs::File::create(out)?);
    gguf_file::write(&mut file, &metadata, &refs)?;
    // dropping a BufWriter would lose a failed final write, as on a full disk
    std::io::Write::flush(&mut file)?;
    Ok(tensors.into_iter().map(|(n, _, d)| (n, d)).collect())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use inference_quant::ShardedSafeTensors;
    use inference_tensor::Tensor;

    use super::*;
    use crate::wan::ops::Linear;

    const PROMPTS: [&str; 4] = [
        "A furnished room with carved wooden panels, a round window and a table.",
        "A sunlit courtyard with stone paths, potted trees and wooden benches.",
        "A narrow alley at night, wet cobblestones reflecting neon signs.",
        "An empty concert hall, rows of red velvet seats facing a dark stage.",
    ];
    // how much further from f32 than bf16 a Q8_0 encoder may land
    const SLACK: f32 = 1.1;

    fn safetensors_builder(
        path: &Path,
        dtype: DType,
    ) -> Result<inference_quant::ShardedVarBuilder> {
        // SAFETY: the test's own file, unchanged while it is read
        unsafe {
            ShardedSafeTensors::sharded(
                &[path],
                dtype,
                &Device::Cpu,
                None,
                std::sync::Arc::new(|_| true),
            )
        }
    }

    #[test]
    fn a_worldstereo_gguf_block_linear_matches_quantizing_at_load() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let (out_dim, in_dim) = (64, 96);
        let weight =
            Tensor::randn(0f32, 1.0, (out_dim, in_dim), &Device::Cpu)?.to_dtype(DType::BF16)?;
        let bias = Tensor::randn(0f32, 1.0, out_dim, &Device::Cpu)?.to_dtype(DType::BF16)?;
        let tensors = HashMap::from([
            ("blocks.0.attn1.to_q.weight".to_string(), weight.clone()),
            ("blocks.0.attn1.to_q.bias".to_string(), bias.clone()),
            ("proj_out.weight".to_string(), weight),
            ("proj_out.bias".to_string(), bias),
        ]);
        inference_tensor::safetensors::save(&tensors, dir.path().join("model.safetensors"))?;
        let gguf = dir.path().join("model.gguf");
        let types: HashMap<_, _> = write_worldstereo_gguf(dir.path(), &gguf, GgmlDType::Q8_0)?
            .into_iter()
            .collect();
        assert_eq!(types["blocks.0.attn1.to_q.weight"], GgmlDType::Q8_0);
        assert_eq!(types["proj_out.weight"], GgmlDType::BF16);

        let st = safetensors_builder(&dir.path().join("model.safetensors"), DType::BF16)?;
        let (vb, _) =
            crate::gguf::var_builder(&gguf, |n| Some(n.to_string()), DType::BF16, &Device::Cpu)?;
        let at_load = Linear::load(in_dim, out_dim, st.pp("blocks.0.attn1.to_q"))?
            .quantized(GgmlDType::Q8_0)?;
        let read = Linear::load(in_dim, out_dim, vb.pp("blocks.0.attn1.to_q"))?;
        assert_eq!(read.quant_dtype(), Some(GgmlDType::Q8_0));
        let xs = Tensor::randn(0f32, 1.0, (3, in_dim), &Device::Cpu)?;
        let diff = (at_load.forward(&xs)? - read.forward(&xs)?)?
            .to_dtype(DType::F32)?
            .abs()?
            .max_all()?;
        assert_eq!(diff.to_scalar::<f32>()?, 0.0);
        let dense = Linear::load(in_dim, out_dim, vb.pp("proj_out"))?;
        assert_eq!(dense.quant_dtype(), None);
        Ok(())
    }

    #[test]
    fn umt5_tensors_take_llama_cpp_names_and_their_types() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let zeros = |dims: &[usize]| Tensor::zeros(dims, DType::F32, &Device::Cpu);
        let tensors = HashMap::from([
            ("shared.weight".to_string(), zeros(&[8, 64])?),
            (
                "encoder.block.0.layer.0.SelfAttention.q.weight".to_string(),
                zeros(&[64, 64])?,
            ),
            (
                "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight".to_string(),
                zeros(&[32, 4])?,
            ),
            (
                "encoder.block.0.layer.0.layer_norm.weight".to_string(),
                zeros(&[64])?,
            ),
            ("encoder.final_layer_norm.weight".to_string(), zeros(&[64])?),
        ]);
        inference_tensor::safetensors::save(&tensors, dir.path().join("model.safetensors"))?;
        let gguf = dir.path().join("encoder.gguf");
        let mut types = write_umt5_gguf(dir.path(), &gguf, GgmlDType::Q8_0)?;
        types.sort_by(|a, b| a.0.cmp(&b.0));
        let want = [
            ("enc.blk.0.attn_norm.weight", GgmlDType::F32),
            ("enc.blk.0.attn_q.weight", GgmlDType::Q8_0),
            ("enc.blk.0.attn_rel_b.weight", GgmlDType::F32),
            ("enc.output_norm.weight", GgmlDType::F32),
            ("token_embd.weight", GgmlDType::Q8_0),
        ];
        let got: Vec<_> = types.iter().map(|(n, d)| (n.as_str(), *d)).collect();
        assert_eq!(got, want);
        Ok(())
    }

    // WAN_TEXT_ENCODER_DIR: the diffusers `text_encoder` folder, its `tokenizer` folder beside it
    #[test]
    #[ignore = "needs the umT5-XXL weights and a GPU"]
    fn a_q8_0_umt5_is_no_further_from_f32_than_bf16() -> Result<()> {
        let Some(dir) = std::env::var_os("WAN_TEXT_ENCODER_DIR").map(PathBuf::from) else {
            return Ok(());
        };
        let device = Device::new_cuda(0)?;
        let cfg: crate::t5::Config =
            serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)
                .map_err(inference_tensor::Error::wrap)?;
        let tokenizer_path = dir
            .parent()
            .expect("a parent folder")
            .join("tokenizer/tokenizer.json");
        let tokenizer = || {
            tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(inference_tensor::Error::msg)
        };
        let out = tempfile::tempdir()?;
        let gguf = out.path().join("umt5-q8_0.gguf");
        write_umt5_gguf(&dir, &gguf, GgmlDType::Q8_0)?;

        let encode = |vb| -> Result<Vec<Tensor>> {
            let mut encoder = crate::wan::text::WanTextEncoder::new(vb, &cfg, tokenizer()?)?;
            PROMPTS
                .iter()
                .map(|p| {
                    let n = encoder.tokens(p)?.len();
                    encoder
                        .encode(p)?
                        .squeeze(0)?
                        .narrow(0, 0, n)?
                        .to_device(&Device::Cpu)
                })
                .collect()
        };
        let mut shards: Vec<PathBuf> = std::fs::read_dir(&dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
            .collect();
        shards.sort();
        let safetensors = |files: &[PathBuf], dtype, device: &Device| {
            // SAFETY: the checkpoint is not changed while the test reads it
            unsafe {
                ShardedSafeTensors::sharded(
                    files,
                    dtype,
                    device,
                    None,
                    std::sync::Arc::new(|_| true),
                )
            }
        };
        let reference = encode(safetensors(&shards, DType::F32, &Device::Cpu)?)?;
        let bf16 = encode(safetensors(&shards, DType::BF16, &device)?)?;
        let q8 = encode(
            crate::gguf::var_builder(&gguf, crate::gguf::t5_native_name, DType::BF16, &device)?.0,
        )?;
        // the smallest per-token cosine and the mean relative RMS error against f32
        let distance = |runs: &[Tensor]| -> Result<(f32, f32)> {
            let (mut worst, mut relative) = (f32::INFINITY, 0f32);
            for (a, b) in runs.iter().zip(&reference) {
                let dot = (a * b)?.sum(1)?;
                let norms = (a.sqr()?.sum(1)?.sqrt()? * b.sqr()?.sum(1)?.sqrt()?)?;
                let cosine = (dot / norms)?.to_vec1::<f32>()?;
                worst = cosine.into_iter().fold(worst, f32::min);
                let rms =
                    |t: &Tensor| -> Result<f32> { t.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>() };
                relative += rms(&(a - b)?)? / rms(b)? / runs.len() as f32;
            }
            Ok((worst, relative))
        };
        let (bf16_cosine, bf16_rms) = distance(&bf16)?;
        let (q8_cosine, q8_rms) = distance(&q8)?;
        println!(
            "vs f32: bf16 worst cosine {bf16_cosine:.5}, RMS {bf16_rms:.4}; q8_0 {q8_cosine:.5}, {q8_rms:.4}"
        );
        assert!(1.0 - q8_cosine <= (1.0 - bf16_cosine) * SLACK);
        assert!(q8_rms <= bf16_rms * SLACK);
        Ok(())
    }
}
