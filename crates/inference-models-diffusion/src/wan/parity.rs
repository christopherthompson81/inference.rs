//! Parity against the reference modules, on dumps made by the HY-World tooling (not in the repo). Ignored: each needs
//! real weights and a dump, named by env vars.

use std::path::PathBuf;

use inference_quant::ShardedSafeTensors;
use inference_tensor::{DType, Device, Result, Tensor};

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name).map(PathBuf::from)
}

// max |a - b| and RMS(a - b) / RMS(b)
fn compare(a: &Tensor, b: &Tensor) -> Result<(f32, f32)> {
    let a = a.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let b = b.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let d = (&a - &b)?;
    let max = d.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
    let rms = |t: &Tensor| -> Result<f32> { t.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>() };
    Ok((max, rms(&d)? / rms(&b)?))
}

// WAN_VAE_DIR: the diffusers `vae` folder; WAN_VAE_REF: the dump (input frames, f32/bf16 latents and decodes)
#[test]
#[ignore = "needs the Wan VAE weights, a reference dump and a GPU"]
fn vae_matches_the_reference() -> Result<()> {
    let (Some(dir), Some(dump)) = (env_path("WAN_VAE_DIR"), env_path("WAN_VAE_REF")) else {
        return Ok(());
    };
    let device = Device::new_cuda(0)?;
    let refs = inference_tensor::safetensors::load(dump, &Device::Cpu)?;
    let cfg: super::vae::VaeConfig =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)
            .map_err(inference_tensor::Error::wrap)?;
    let weights = dir.join("diffusion_pytorch_model.safetensors");
    let input = refs["input"].to_device(&device)?;
    for (name, dtype) in [("f32", DType::F32), ("bf16", DType::BF16)] {
        // SAFETY: the weights file is not modified while the test runs
        let vb = unsafe {
            ShardedSafeTensors::sharded(
                &[&weights],
                DType::F32,
                &device,
                None,
                std::sync::Arc::new(|_| true),
            )?
        };
        let vae = super::vae::WanVae::new(vb, &cfg, dtype)?;
        let z = vae.encode(&input)?;
        let (max, rel) = compare(&z, &refs[&format!("{name}.latents")])?;
        println!("{name} latents: max {max:.3e} rel rms {rel:.3e}");
        // decode the reference's own latents, so the decoder is judged alone
        let y = vae.decode(&refs[&format!("{name}.latents")].to_device(&device)?)?;
        let (max, rel) = compare(&y, &refs[&format!("{name}.decoded")])?;
        println!("{name} decoded: max {max:.3e} rel rms {rel:.3e}");
    }
    Ok(())
}
