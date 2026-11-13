//! One EXL3 linear's dense weight from a safetensors file, as little-endian f32 `out x in`, for
//! `scripts/exl3_parity.py`. Usage: exl3_reconstruct <model.safetensors> <tensor prefix> <out.f32>

use inference_quant::exl3;
use inference_tensor::{DType, Device};

fn main() -> inference_tensor::Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    let [_, path, key, out] = args.as_slice() else {
        inference_tensor::bail!(
            "usage: exl3_reconstruct <model.safetensors> <tensor prefix> <out.f32>"
        );
    };
    let tensors = inference_tensor::safetensors::load(path, &Device::Cpu)?;
    let get = |name: &str| {
        tensors
            .get(&format!("{key}.{name}"))
            .ok_or_else(|| inference_tensor::Error::msg(format!("no {key}.{name}")))
    };
    let codebook = exl3::Exl3Codebook::from_markers(
        tensors.contains_key(&format!("{key}.mcg")),
        tensors.contains_key(&format!("{key}.mul1")),
    );
    let weight = exl3::dense_weight(
        get("trellis")?,
        get("suh")?,
        get("svh")?,
        codebook,
        DType::F32,
        &Device::Cpu,
    )?;
    let bytes = weight
        .flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    std::fs::write(out, bytes)?;
    Ok(())
}
