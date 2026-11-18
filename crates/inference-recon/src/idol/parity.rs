// Parity against the reference's stages (`ref_cond.py` and the like, scratch): IDOL_DINOV3 the DINOv3 ViT-H+/16
// checkpoint file, IDOL_VAE the FLUX.2 VAE's, IDOL_REF the dump. Ignored by default: they need the checkpoints and a GPU.

use std::path::PathBuf;

use inference_tensor::{DType, Device, Result};
use rand::SeedableRng;
use rand::rngs::StdRng;

use super::conditioning::{ImageEncoders, VaeLatent};
use crate::worldmirror::parity::compare;

fn path(var: &str) -> Option<PathBuf> {
    std::env::var_os(var).map(PathBuf::from)
}

#[test]
#[ignore = "needs DINOv3 ViT-H+, the FLUX.2 VAE, a reference dump and a GPU"]
fn the_image_tokens_match_the_reference() -> Result<()> {
    let (Some(dinov3), Some(vae), Some(reference)) =
        (path("IDOL_DINOV3"), path("IDOL_VAE"), path("IDOL_REF"))
    else {
        return Ok(());
    };
    let device = Device::new_cuda(0)?;
    let refs = inference_tensor::safetensors::load(&reference, &Device::Cpu)?;
    let pixels = refs["pixels"].to_device(&device)?;
    for (name, dtype) in [("f32", DType::F32), ("bf16", DType::BF16)] {
        let encoders = ImageEncoders::load(&dinov3, &vae, dtype, dtype, &device)?;
        let (feature1, _) =
            encoders.encode(&pixels, VaeLatent::Mean, &mut StdRng::seed_from_u64(0))?;
        let (max, rms) = compare(&feature1, &refs[&format!("dino_{name}")])?;
        println!("{name} dinov3 tokens: max {max:.3e} rms {rms:.3e}");
        let (mean, std) = encoders.vae_moments(&pixels)?;
        let (mmax, mrms) = compare(&mean, &refs[&format!("vae_mean_{name}")])?;
        let (smax, srms) = compare(&std, &refs[&format!("vae_std_{name}")])?;
        println!(
            "{name} vae mean: max {mmax:.3e} rms {mrms:.3e}; std: max {smax:.3e} rms {srms:.3e}"
        );
        let tolerance = if dtype == DType::F32 { 1e-3 } else { 2e-2 };
        assert!(
            rms < tolerance && mrms < tolerance && srms < tolerance,
            "{name}"
        );
    }
    Ok(())
}

// IDOL_DIR the speridlabs/idol folder; IDOL_REF_F16 / IDOL_REF_F32 the `ref_model.py` dumps (shipped fp16, f32)
fn model_dumps() -> Option<(PathBuf, Vec<(DType, PathBuf)>)> {
    let dir = path("IDOL_DIR")?;
    let dumps = [("IDOL_REF_F16", DType::F16), ("IDOL_REF_F32", DType::F32)]
        .into_iter()
        .filter_map(|(var, dtype)| Some((dtype, path(var)?)))
        .collect();
    Some((dir, dumps))
}

fn weights(
    dir: &std::path::Path,
    name: &str,
    device: &Device,
) -> Result<inference_tensor::nn::VarBuilder<'static>> {
    // SAFETY: the checkpoint files are not modified while the model is loaded
    unsafe {
        inference_tensor::nn::VarBuilder::from_mmaped_safetensors(
            &[dir.join(format!("{name}.safetensors"))],
            DType::F32,
            device,
        )
    }
}

#[test]
#[ignore = "needs the Idol checkpoint, reference dumps and a GPU"]
fn the_denoiser_velocity_matches_the_reference() -> Result<()> {
    let Some((dir, dumps)) = model_dumps() else {
        return Ok(());
    };
    let device = Device::new_cuda(0)?;
    for (dtype, dump) in dumps {
        let refs = inference_tensor::safetensors::load(&dump, &device)?;
        let dit = super::dit::Dit::load(
            super::dit::IDOL_DIT,
            dtype,
            weights(&dir, "denoiser", &device)?,
        )?;
        let condition = dit.condition(&refs["feature1"], &refs["feature2"])?;
        let t = inference_tensor::Tensor::new(&[700f32], &device)?;
        let v = dit.forward(&refs["x_latent"], &refs["x_camera"], &t, &condition)?;
        let (lmax, lrms) = compare(&v.latent, &refs["v_latent"])?;
        let (cmax, crms) = compare(&v.camera, &refs["v_camera"])?;
        println!(
            "{dtype:?} velocity latent: max {lmax:.3e} rms {lrms:.3e}; camera: max {cmax:.3e} rms {crms:.3e}"
        );
        let tolerance = if dtype == DType::F32 { 1e-3 } else { 2e-2 };
        assert!(lrms < tolerance && crms < tolerance, "{dtype:?}");
    }
    Ok(())
}

#[test]
#[ignore = "needs the Idol checkpoint, reference dumps and a GPU"]
fn the_decoders_match_the_reference() -> Result<()> {
    use super::decoder::{
        GaussianDecoder, IDOL_GAUSSIAN_SETTINGS, IDOL_GAUSSIANS, IDOL_OCTREE, OctreeDecoder,
    };
    let Some((dir, dumps)) = model_dumps() else {
        return Ok(());
    };
    let device = Device::new_cuda(0)?;
    for (dtype, dump) in dumps {
        let refs = inference_tensor::safetensors::load(&dump, &device)?;
        let tolerance = if dtype == DType::F32 { 1e-3 } else { 3e-2 };
        let latent = &refs["sample_latent"];
        let octree = OctreeDecoder::load(IDOL_OCTREE, dtype, weights(&dir, "decoder", &device)?)?;
        for level in [1usize, 4, 8] {
            let logits = octree.forward(&refs["octree_points"], 1 << level, latent)?;
            let (max, rms) = compare(&logits, &refs[&format!("octree_logits_{level}")])?;
            println!("{dtype:?} octree logits level {level}: max {max:.3e} rms {rms:.3e}");
            assert!(rms < tolerance, "{dtype:?} level {level}");
        }
        let decoder = GaussianDecoder::load(
            IDOL_GAUSSIANS,
            IDOL_GAUSSIAN_SETTINGS,
            dtype,
            weights(&dir, "decoder_gs", &device)?,
        )?;
        let features = decoder.forward(&refs["gs_anchors"], latent)?;
        let (max, rms) = compare(&features, &refs["gs_features"])?;
        println!("{dtype:?} gaussian features: max {max:.3e} rms {rms:.3e}");
        assert!(rms < tolerance, "{dtype:?} gaussian features");
        // the activations on the reference's own outputs, so they compare exactly
        let g = decoder.gaussians(&refs["gs_anchors"], &refs["gs_features"])?;
        let flat = |rows: Vec<f32>, n: usize| {
            inference_tensor::Tensor::from_vec(rows, (g.xyz.len(), n), &Device::Cpu)
        };
        for (name, ours, n) in [
            ("xyz", flat(g.xyz.concat(), 3)?, 3),
            ("features_dc", flat(g.dc.concat(), 3)?, 3),
            ("scaling", flat(g.scaling.concat(), 3)?, 3),
            ("rotation", flat(g.rotation.concat(), 4)?, 4),
            ("opacity", flat(g.opacity.clone(), 1)?, 1),
        ] {
            let theirs = refs[&format!("gaussian_{name}")].reshape(((), n))?;
            let (max, rms) = compare(&ours, &theirs)?;
            println!("{dtype:?} gaussian {name}: max {max:.3e} rms {rms:.3e}");
            assert!(rms < 1e-5, "{dtype:?} {name}");
        }
    }
    Ok(())
}

fn rgba(path: &std::path::Path) -> Result<crate::jovox::Rgba> {
    let decoded = crate::read_image(path)
        .map_err(inference_tensor::Error::wrap)?
        .to_rgba8();
    Ok(crate::jovox::Rgba {
        width: decoded.width() as usize,
        height: decoded.height() as usize,
        pixels: decoded.into_raw(),
    })
}

// IDOL_IMAGE the image `ref_cond.py` read
#[test]
#[ignore = "needs a reference dump (and the Idol checkpoint for an image without transparency)"]
fn the_prepared_image_matches_the_reference() -> Result<()> {
    let (Some(image), Some(reference)) = (path("IDOL_IMAGE"), path("IDOL_REF")) else {
        return Ok(());
    };
    let refs = inference_tensor::safetensors::load(&reference, &Device::Cpu)?;
    let mut birefnet = || {
        let dir = path("IDOL_DIR").expect("IDOL_DIR for background removal");
        crate::birefnet::BiRefNet::load_file(
            &dir.join("birefnet.safetensors"),
            &Device::new_cuda(0)?,
        )
    };
    let ours = super::preprocess::prepare(
        &rgba(&image)?,
        super::preprocess::MARGIN_SYNTHETIC,
        &mut birefnet,
    )?;
    let theirs: Vec<u8> = (refs["pixels"].squeeze(0)?.permute((1, 2, 0))? * 255.0)?
        .round()?
        .flatten_all()?
        .to_dtype(DType::U8)?
        .to_vec1()?;
    let differing = ours.iter().zip(&theirs).filter(|(a, b)| a != b).count();
    let worst = ours
        .iter()
        .zip(&theirs)
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap_or(0);
    println!(
        "prepared image: {differing} of {} bytes differ, by at most {worst}",
        ours.len()
    );
    assert!(
        differing * 100 < ours.len() && worst <= 2,
        "the prepared image differs"
    );
    Ok(())
}

// IDOL_OUT where the end-to-end run writes its PLY and SPZ
#[test]
#[ignore = "needs the Idol checkpoints, an image and a GPU"]
fn generates_gaussians_end_to_end() -> Result<()> {
    let (Some(dir), Some(dinov3), Some(vae), Some(image), Some(out)) = (
        path("IDOL_DIR"),
        path("IDOL_DINOV3"),
        path("IDOL_VAE"),
        path("IDOL_IMAGE"),
        path("IDOL_OUT"),
    ) else {
        return Ok(());
    };
    let device = Device::new_cuda(0)?;
    let paths = super::IdolPaths {
        idol: dir,
        dinov3,
        vae,
    };
    let start = std::time::Instant::now();
    let result = super::generate(
        &paths,
        &rgba(&image)?,
        &super::IdolSettings::default(),
        &crate::progress::NoProgress,
        &device,
    )?;
    println!(
        "generated {} Gaussians in {:.1} s, camera {:?}",
        result.gaussians.xyz.len(),
        start.elapsed().as_secs_f64(),
        result.camera
    );
    let data = super::SplatData::from_gaussians(&result.gaussians);
    std::fs::create_dir_all(&out)?;
    crate::splatting::spz::write_ply(&out.join("idol.ply"), &data.rows())?;
    crate::splatting::spz::write_spz(&out.join("idol.spz"), &data.rows())?;
    let start = std::time::Instant::now();
    let solid =
        crate::printable::Solid::from_gaussians(&result.gaussians, crate::printable::SPLAT_GRID);
    let print = solid
        .printable(crate::printable::DEFAULT_PRINT_SIZE_MM)
        .expect("a solid to print");
    println!(
        "printable: {} faces, {:.1} cm3, {:.2} s",
        print.faces,
        print.volume_mm3 / 1000.0,
        start.elapsed().as_secs_f64()
    );
    std::fs::write(out.join("idol.stl"), print.stl)?;
    Ok(())
}
