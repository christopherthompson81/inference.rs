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

// WAN_DIR: the Wan diffusers folder; WAN_CLIP_REF: the dump (processor pixel values, f32/bf16 penultimate states);
// WAN_CLIP_IMAGE: the image the dump started from
#[test]
#[ignore = "needs the Wan image encoder, a reference dump and a GPU"]
fn image_encoder_matches_the_reference() -> Result<()> {
    let (Some(dir), Some(dump), Some(image)) = (
        env_path("WAN_DIR"),
        env_path("WAN_CLIP_REF"),
        env_path("WAN_CLIP_IMAGE"),
    ) else {
        return Ok(());
    };
    let device = Device::new_cuda(0)?;
    let refs = inference_tensor::safetensors::load(dump, &Device::Cpu)?;
    let enc_dir = dir.join("image_encoder");
    let cfg: inference_nn::vision::clip::ClipConfig =
        serde_json::from_str(&std::fs::read_to_string(enc_dir.join("config.json"))?)
            .map_err(inference_tensor::Error::wrap)?;
    let weights = enc_dir.join("model.safetensors");
    // the pipeline's start frame: PIL bicubic to the 832x480 bucket
    let img = image::open(&image)
        .map_err(inference_tensor::Error::wrap)?
        .to_rgb8();
    let frame = inference_vision::pil_resize_bicubic(
        img.as_raw(),
        (img.width() as usize, img.height() as usize),
        (832, 480),
    );
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
        let encoder = super::clip::WanImageEncoder::new(vb, &cfg, dtype)?;
        let pixels = encoder.pixel_values(&frame, (832, 480), &device)?;
        if name == "f32" {
            let (max, rel) = compare(&pixels, &refs["pixel_values"])?;
            println!("pixel values: max {max:.3e} rel rms {rel:.3e}");
        }
        let states = encoder.forward(&refs["pixel_values"].to_device(&device)?)?;
        let (max, rel) = compare(&states.squeeze(0)?, &refs[&format!("{name}.penultimate")])?;
        println!("{name} penultimate: max {max:.3e} rel rms {rel:.3e}");
    }
    Ok(())
}

// WAN_DIR (the Wan diffusers folder, for transformer/config.json), WORLDSTEREO_DIR (the memory-dmd checkpoint) and
// WORLDSTEREO_SMALL_REF: a dump of the first WORLDSTEREO_LAYERS blocks with the full ControlNet on small inputs
#[test]
#[ignore = "needs the WorldStereo checkpoint, a reference dump and a GPU"]
fn small_transformer_matches_the_reference() -> Result<()> {
    let (Some(wan), Some(ckpt), Some(dump)) = (
        env_path("WAN_DIR"),
        env_path("WORLDSTEREO_DIR"),
        env_path("WORLDSTEREO_SMALL_REF"),
    ) else {
        return Ok(());
    };
    use super::transformer::{StepInputs, WanConfig, WorldStereoConfig, WorldStereoTransformer};
    let layers = std::env::var("WORLDSTEREO_LAYERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    let device = Device::new_cuda(0)?;
    let refs = inference_tensor::safetensors::load(dump, &device)?;
    let read = |p: PathBuf| -> Result<String> { Ok(std::fs::read_to_string(p)?) };
    let mut cfg: WanConfig = serde_json::from_str(&read(wan.join("transformer/config.json"))?)
        .map_err(inference_tensor::Error::wrap)?;
    cfg.num_layers = layers;
    let ws: WorldStereoConfig = serde_json::from_str(&read(ckpt.join("config.json"))?)
        .map_err(inference_tensor::Error::wrap)?;
    let weights = ckpt.join("model.safetensors");
    // SAFETY: the weights file is not modified while the test runs
    let vb = unsafe {
        ShardedSafeTensors::sharded(
            &[&weights],
            DType::BF16,
            &device,
            None,
            std::sync::Arc::new(|_| true),
        )?
    };
    let model = WorldStereoTransformer::new(&cfg, &ws, vb)?;
    let bf = |k: &str| refs[k].to_dtype(DType::BF16);
    let t = refs["timestep"].to_vec1::<f32>()?[0] as f64;
    let cond = model.conditions(t, &bf("text")?.unsqueeze(0)?, &bf("image")?.unsqueeze(0)?)?;
    let render_conditions = Tensor::cat(&[&bf("render_mask")?, &bf("camera_embedding")?], 0)?;
    let ref_index = refs["ref_index"]
        .to_dtype(DType::U32)?
        .to_vec1::<u32>()?
        .iter()
        .map(|&i| i as usize)
        .collect::<Vec<_>>();
    let (latents, render_latent, reference) = (
        bf("latents")?,
        bf("render_latent")?,
        bf("reference_latent")?,
    );
    let (qt, qt_ref) = (bf("camera_qt")?, bf("camera_qt_ref")?);
    let inputs = StepInputs {
        latents: &latents,
        render_latent: &render_latent,
        render_conditions: &render_conditions,
        camera_qt: &qt,
        camera_qt_ref: Some(&qt_ref),
        reference_latent: Some(&reference),
        ref_index: &ref_index,
    };
    let pred = model.forward(&inputs, &cond)?;
    let (max, rel) = compare(&pred, &refs["pred"])?;
    println!("{layers} blocks, prediction: max {max:.3e} rel rms {rel:.3e}");
    Ok(())
}

// A real-size step (21 latent frames at 60 x 104, WORLDSTEREO_REFS memory frames) with the bf16 blocks streamed from
// host memory: time and device memory, on random inputs
#[test]
#[ignore = "needs the WorldStereo checkpoint and a GPU; a benchmark"]
fn full_size_streamed_step() -> Result<()> {
    let (Some(wan), Some(ckpt)) = (env_path("WAN_DIR"), env_path("WORLDSTEREO_DIR")) else {
        return Ok(());
    };
    use super::transformer::{StepInputs, WanConfig, WorldStereoConfig, WorldStereoTransformer};
    let refs_n: usize = std::env::var("WORLDSTEREO_REFS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let device = Device::new_cuda(0)?;
    let read = |p: PathBuf| -> Result<String> { Ok(std::fs::read_to_string(p)?) };
    let cfg: WanConfig = serde_json::from_str(&read(wan.join("transformer/config.json"))?)
        .map_err(inference_tensor::Error::wrap)?;
    let ws: WorldStereoConfig = serde_json::from_str(&read(ckpt.join("config.json"))?)
        .map_err(inference_tensor::Error::wrap)?;
    let weights = ckpt.join("model.safetensors");
    let start = std::time::Instant::now();
    // SAFETY: the weights file is not modified while the test runs
    let vb = unsafe {
        ShardedSafeTensors::sharded(
            &[&weights],
            DType::BF16,
            &device,
            None,
            std::sync::Arc::new(|_| true),
        )?
    };
    let model = WorldStereoTransformer::new_streamed(&cfg, &ws, vb)?;
    println!("loaded in {:.1} s", start.elapsed().as_secs_f64());
    let f: usize = std::env::var("WORLDSTEREO_FRAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(21);
    let (h, w) = (60, 104);
    let rand = |shape: &[usize]| -> Result<Tensor> {
        Tensor::randn(0f32, 1.0, shape, &device)?.to_dtype(DType::BF16)
    };
    let latents = rand(&[36, f, h, w])?;
    let render_latent = rand(&[16, f, h, w])?;
    let render_conditions = rand(&[7, f, 8 * h, 8 * w])?;
    let qt = rand(&[f, 7])?;
    let qt_ref = rand(&[refs_n, 7])?;
    let reference = rand(&[36, refs_n, h, w])?;
    let ref_index = (0..refs_n).collect::<Vec<_>>();
    let text = (rand(&[1, 512, 4096])? * 0.2)?;
    let image = rand(&[1, 257, 1280])?;
    let inputs = StepInputs {
        latents: &latents,
        render_latent: &render_latent,
        render_conditions: &render_conditions,
        camera_qt: &qt,
        camera_qt_ref: Some(&qt_ref),
        reference_latent: if refs_n > 0 { Some(&reference) } else { None },
        ref_index: &ref_index,
    };
    for round in 0..2 {
        device.synchronize()?;
        let start = std::time::Instant::now();
        let cond = model.conditions(937.5, &text, &image)?;
        let pred = model.forward(&inputs, &cond)?;
        device.synchronize()?;
        let used = inference_nn::utils::memory_usage::MemoryUsage
            .query(&device)
            .map(|m| (m.total() - m.available()) >> 20)
            .unwrap_or_default();
        println!(
            "round {round}: step {:.1} s, pred rms {:.3}, {used} MiB in use",
            start.elapsed().as_secs_f64(),
            pred.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?
        );
    }
    Ok(())
}
