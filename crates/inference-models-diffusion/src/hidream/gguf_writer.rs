//! A HiDream-O1 checkpoint written as a ComfyUI-GGUF file: the text stack's matrices quantized, the rest kept fine.

use std::path::Path;

use inference_tensor::quantized::gguf_file::{self, Value};
use inference_tensor::quantized::{GgmlDType, QTensor};
use inference_tensor::safetensors::MmapedSafetensors;
use inference_tensor::{Device, Result, Tensor};
use rayon::prelude::*;

const TEXT_LAYERS: &str = "model.language_model.layers.";
const EMBEDDING: &str = "model.language_model.embed_tokens.weight";
// the prediction head writes pixels directly, so it stays at full precision as the published GGUFs keep it
const HEAD: &str = "model.final_layer2.";
// a row that does not fill whole blocks of the wanted type takes this, then F16
const FALLBACK: GgmlDType = GgmlDType::Q8_0;
// the writable types whose CUDA matmuls keep each 32-value activation block's sum in F16 (MMQ's DS4 layout),
// which HiDream's outlier channels overflow
const F16_SUM_TYPES: [GgmlDType; 3] = [GgmlDType::Q4_0, GgmlDType::Q4K, GgmlDType::Q5K];
// ComfyUI-GGUF's keys, which these files are read by besides us
const ARCHITECTURE: &str = "qwen";
const DIFFUSION: &str = "diffusion";

/// The text stack's matrix type, and the ones whose names contain a fragment, the first match winning.
#[derive(Debug, Clone)]
pub struct Policy {
    pub matrices: GgmlDType,
    pub overrides: Vec<(String, GgmlDType)>,
}

impl Policy {
    fn check(&self) -> Result<()> {
        let types = std::iter::once(self.matrices).chain(self.overrides.iter().map(|(_, d)| *d));
        if let Some(bad) = types.into_iter().find(|d| F16_SUM_TYPES.contains(d)) {
            inference_tensor::bail!(
                "{bad:?} overflows on HiDream-O1: its CUDA kernels sum activations in F16, and the model's \
                 outlier channels exceed that range; use q8_0, q6_k, q5_0 or q3_k"
            );
        }
        Ok(())
    }

    fn wanted(&self, name: &str) -> GgmlDType {
        self.overrides
            .iter()
            .find(|(fragment, _)| name.contains(fragment.as_str()))
            .map_or(self.matrices, |(_, d)| *d)
    }
}

fn dtype_for(name: &str, t: &Tensor, policy: &Policy) -> GgmlDType {
    if t.rank() < 2 || name.starts_with(HEAD) {
        return GgmlDType::F32;
    }
    if name == EMBEDDING {
        return GgmlDType::F16;
    }
    if !name.starts_with(TEXT_LAYERS) {
        return GgmlDType::BF16;
    }
    let row = t.dims()[t.rank() - 1];
    [policy.wanted(name), FALLBACK]
        .into_iter()
        .find(|d| row.is_multiple_of(d.block_size()))
        .unwrap_or(GgmlDType::F16)
}

/// Writes the checkpoint in `dir` to `out` under `policy`; returns each tensor's type.
pub fn write_gguf(dir: &Path, out: &Path, policy: &Policy) -> Result<Vec<(String, GgmlDType)>> {
    policy.check()?;
    let names = std::fs::read_dir(dir)?.filter_map(|e| e.ok()?.file_name().into_string().ok());
    let shards: Vec<_> = super::weight_shards(names)
        .iter()
        .map(|n| dir.join(n))
        .collect();
    if shards.is_empty() {
        inference_tensor::bail!("`{}` has no `model*.safetensors` weights", dir.display());
    }
    // SAFETY: the shards are only read, and not changed while the writer runs
    let weights = unsafe { MmapedSafetensors::multi(&shards)? };
    let mut names: Vec<String> = weights
        .tensors()
        .into_iter()
        .map(|(name, _)| name)
        .filter(|name| crate::gguf::hidream_native_name(name).is_some())
        .collect();
    names.sort();
    let tensors = names
        .par_iter()
        .map(|name| {
            let t = weights
                .load(name, &Device::Cpu)?
                .to_dtype(inference_tensor::DType::F32)?;
            let dtype = dtype_for(name, &t, policy);
            Ok((name.clone(), QTensor::quantize(&t, dtype)?, dtype))
        })
        .collect::<Result<Vec<_>>>()?;
    let metadata = [
        (
            "general.architecture",
            Value::String(ARCHITECTURE.to_string()),
        ),
        (super::GGUF_TYPE_KEY, Value::String(DIFFUSION.to_string())),
    ];
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
    use super::*;
    use inference_tensor::DType;

    fn zeros(dims: &[usize]) -> Tensor {
        Tensor::zeros(dims, DType::F32, &Device::Cpu).unwrap()
    }

    #[test]
    fn each_tensor_takes_its_kind_of_precision() {
        let policy = Policy {
            matrices: GgmlDType::Q6K,
            overrides: vec![("down_proj".to_string(), GgmlDType::Q8_0)],
        };
        // only the row width matters, so the shapes are small stand-ins
        for (name, dims, want) in [
            (
                "model.language_model.layers.3.mlp.down_proj.weight",
                vec![2, 256],
                GgmlDType::Q8_0,
            ),
            (
                "model.language_model.layers.0.mlp.up_proj.weight",
                vec![2, 256],
                GgmlDType::Q6K,
            ),
            (
                "model.language_model.layers.0.input_layernorm.weight",
                vec![8],
                GgmlDType::F32,
            ),
            (EMBEDDING, vec![2, 256], GgmlDType::F16),
            (
                "model.x_embedder.proj1.weight",
                vec![2, 256],
                GgmlDType::BF16,
            ),
            (
                "model.final_layer2.linear.weight",
                vec![2, 256],
                GgmlDType::F32,
            ),
            // a 96-wide row fills no Q6_K block, so it takes Q8_0
            (
                "model.language_model.layers.0.self_attn.q_proj.weight",
                vec![2, 96],
                GgmlDType::Q8_0,
            ),
        ] {
            assert_eq!(dtype_for(name, &zeros(&dims), &policy), want, "{name}");
        }
        let refused = Policy {
            matrices: GgmlDType::Q4K,
            overrides: Vec::new(),
        };
        assert!(refused.check().is_err());
    }
}
