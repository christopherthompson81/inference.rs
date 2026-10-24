// Layer parity against tensors dumped from the reference (`dump_ref.py`): WORLDMIRROR_DIR is the checkpoint directory,
// WORLDMIRROR_REF the dump. Ignored by default: they need the real checkpoint and a GPU.

use std::path::PathBuf;

use inference_tensor::{DType, Device, Result, Tensor};

use super::WorldMirror;

fn paths() -> Option<(PathBuf, PathBuf)> {
    Some((
        std::env::var_os("WORLDMIRROR_DIR")?.into(),
        std::env::var_os("WORLDMIRROR_REF")?.into(),
    ))
}

// max |a - b| and the relative RMS error of `a` against the reference `b`
pub(super) fn compare(a: &Tensor, b: &Tensor) -> Result<(f32, f32)> {
    let a = a
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .flatten_all()?;
    let b = b
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .flatten_all()?;
    let diff = (&a - &b)?;
    let max = diff.abs()?.max(0)?.to_scalar::<f32>()?;
    let rms = (diff.sqr()?.mean_all()?.to_scalar::<f32>()?
        / b.sqr()?.mean_all()?.to_scalar::<f32>()?)
    .sqrt();
    Ok((max, rms))
}

#[test]
#[ignore = "needs the WorldMirror checkpoint, a reference dump and a GPU"]
fn trunk_features_match_the_reference() -> Result<()> {
    let Some((dir, reference)) = paths() else {
        return Ok(());
    };
    let device = Device::new_cuda(0)?;
    let refs = inference_tensor::safetensors::load(&reference, &Device::Cpu)?;
    let model = WorldMirror::load(&dir, DType::BF16, &device)?;
    let mut first = None;
    let out = model.trunk(&refs["input.img"], &mut |i, local, global| {
        if i == 0 {
            first = Some((local.clone(), global.clone()));
        }
    })?;
    let (local, global) = first.expect("layer 0 ran");
    let (lmax, lrms) = compare(&local, &refs["vgt.frame.0"])?;
    let (gmax, grms) = compare(
        &global.reshape(refs["vgt.global.0"].shape())?,
        &refs["vgt.global.0"],
    )?;
    println!("layer  0: frame max {lmax:.3e} rms {lrms:.3e}, global max {gmax:.3e} rms {grms:.3e}");
    for (layer, features) in [4, 11, 17, 23].iter().zip(&out.features) {
        let dim = features.dim(2)? / 2;
        let (frames, tokens) = (features.dim(0)?, features.dim(1)?);
        let global = refs[&format!("vgt.global.{layer}")].reshape((frames, tokens, dim))?;
        let (lmax, lrms) = compare(
            &features.narrow(2, 0, dim)?,
            &refs[&format!("vgt.frame.{layer}")],
        )?;
        let (gmax, grms) = compare(&features.narrow(2, dim, dim)?, &global)?;
        println!(
            "layer {layer:2}: frame max {lmax:.3e} rms {lrms:.3e}, global max {gmax:.3e} rms {grms:.3e}"
        );
        assert!(grms < 5e-2, "layer {layer} relative rms {grms}");
    }
    Ok(())
}

#[test]
#[ignore = "needs the WorldMirror checkpoint, a reference dump and a GPU"]
fn first_frame_block_matches_on_the_reference_input() -> Result<()> {
    let Some((dir, reference)) = paths() else {
        return Ok(());
    };
    let device = Device::new_cuda(0)?;
    let refs = inference_tensor::safetensors::load(&reference, &Device::Cpu)?;
    let model = WorldMirror::load(&dir, DType::BF16, &device)?;
    let img = &refs["input.img"];
    let (hp, wp) = (img.dim(2)? / 14, img.dim(3)? / 14);
    let mut seen = std::collections::HashMap::new();
    let out = model.trunk.first_frame_block(
        &refs["blk0.in"].to_device(&device)?,
        hp,
        wp,
        &mut |name, t| {
            seen.insert(name.to_string(), t.clone());
        },
    )?;
    for name in ["norm1", "q_norm", "q_rope", "attn_out", "norm2", "mlp_out"] {
        let (max, rms) = compare(&seen[name], &refs[&format!("blk0.{name}")])?;
        println!("block 0 {name:9}: max {max:.3e} rms {rms:.3e}");
    }
    let (max, rms) = compare(&out, &refs["vgt.frame.0"])?;
    println!("block 0 output   : max {max:.3e} rms {rms:.3e}");
    Ok(())
}

#[test]
#[ignore = "needs the WorldMirror checkpoint, a reference dump and a GPU"]
fn cameras_match_the_reference() -> Result<()> {
    let Some((dir, reference)) = paths() else {
        return Ok(());
    };
    let device = Device::new_cuda(0)?;
    let refs = inference_tensor::safetensors::load(&reference, &Device::Cpu)?;
    let model = WorldMirror::load(&dir, DType::BF16, &device)?;
    let pred = model.predict(&refs["input.img"])?;
    let reference_params = refs["pred.camera_params"].squeeze(0)?;
    let (max, rms) = compare(&pred.camera_params, &reference_params)?;
    println!("camera params: max {max:.3e} rms {rms:.3e}");
    println!(
        "ours {:?}",
        pred.camera_params
            .to_device(&Device::Cpu)?
            .to_vec2::<f32>()?
    );
    println!("reference {:?}", reference_params.to_vec2::<f32>()?);
    let cameras = super::decode_cameras(&pred.camera_params, pred.height, pred.width)?;
    let poses = refs["pred.camera_poses"].squeeze(0)?.to_vec3::<f32>()?;
    let intrs = refs["pred.camera_intrs"].squeeze(0)?.to_vec3::<f32>()?;
    for (i, camera) in cameras.iter().enumerate() {
        let pose_err = (0..4)
            .flat_map(|r| (0..4).map(move |c| (r, c)))
            .map(|(r, c)| (camera.camera_to_world[r][c] - poses[i][r][c]).abs())
            .fold(0f32, f32::max);
        let focal_err = (camera.intrinsics[0][0] - intrs[i][0][0]).abs() / intrs[i][0][0];
        println!("frame {i}: c2w max err {pose_err:.3e}, fx rel err {focal_err:.3e}");
        assert!(pose_err < 2e-2 && focal_err < 1e-2);
    }
    Ok(())
}

#[test]
#[ignore = "needs the WorldMirror checkpoint, a reference dump and a GPU"]
fn dense_outputs_match_the_reference() -> Result<()> {
    let Some((dir, reference)) = paths() else {
        return Ok(());
    };
    let device = Device::new_cuda(0)?;
    let refs = inference_tensor::safetensors::load(&reference, &Device::Cpu)?;
    let model = WorldMirror::load(&dir, DType::BF16, &device)?;
    // the heads alone on the reference's trunk features, unless the whole model is asked for
    let pred = if std::env::var_os("WORLDMIRROR_WHOLE").is_some() {
        model.predict(&refs["input.img"])?
    } else {
        let features = (0..4)
            .map(|i| refs[&format!("vgt.features.{i}")].to_device(&device))
            .collect::<Result<Vec<_>>>()?;
        model.heads(&features, 7, &refs["input.img"].to_device(&device)?)?
    };
    let pairs = [
        ("depth", &pred.depth, "pred.depth"),
        ("depth_conf", &pred.depth_conf, "pred.depth_conf"),
        (
            "depth_mask_logits",
            &pred.depth_mask_logits,
            "pred.depth_mask_logits",
        ),
        ("pts3d", &pred.points, "pred.pts3d"),
        ("normals", &pred.normals, "pred.normals"),
        ("gs_depth", &pred.gs_depth, "pred.gs_depth"),
        (
            "gs_mask_logits",
            &pred.gs_depth_mask_logits,
            "pred.gs_depth_mask_logits",
        ),
        ("splat means", &pred.splats.means, "pred.splats.means"),
        ("splat quats", &pred.splats.quats, "pred.splats.quats"),
        ("splat scales", &pred.splats.scales, "pred.splats.scales"),
        (
            "splat opacities",
            &pred.splats.opacities,
            "pred.splats.opacities",
        ),
        ("splat sh", &pred.splats.sh_dc, "pred.splats.sh"),
        ("splat weights", &pred.splats.weights, "pred.splats.weights"),
    ];
    for (name, ours, key) in pairs {
        let reference = refs[key].reshape(ours.shape())?;
        let (max, rms) = compare(ours, &reference)?;
        println!("{name:18}: max {max:.3e} rms {rms:.3e}");
    }
    Ok(())
}

// One forward over WORLDMIRROR_FRAMES (default 32) frames of 504 x 504, the trunk and heads timed apart
#[test]
#[ignore = "benchmark: needs the WorldMirror checkpoint and a GPU"]
fn thirty_two_frames_forward() -> Result<()> {
    let Some((dir, _)) = paths() else {
        return Ok(());
    };
    let frames = std::env::var("WORLDMIRROR_FRAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    let device = Device::new_cuda(0)?;
    let model = WorldMirror::load(&dir, DType::BF16, &device)?;
    let used = || -> String {
        inference_nn::utils::memory_usage::MemoryUsage
            .query(&device)
            .map(|m| format!("{} MiB used", (m.total() - m.available()) >> 20))
            .unwrap_or_default()
    };
    println!("loaded: {}", used());
    let images = Tensor::rand(0f32, 1f32, (frames, 3, 504, 504), &device)?;
    for round in 0..2 {
        device.synchronize()?;
        let start = std::time::Instant::now();
        let out = model.trunk(&images, &mut |_, _, _| {})?;
        device.synchronize()?;
        let trunk = start.elapsed().as_secs_f64();
        println!("round {round}: trunk {trunk:.2} s, {}", used());
        let pred = model.heads(&out.features, out.patch_start, &images)?;
        device.synchronize()?;
        println!(
            "round {round}: heads {:.2} s, {} splats, {}",
            start.elapsed().as_secs_f64() - trunk,
            pred.splats.means.dim(0)?,
            used()
        );
    }
    Ok(())
}

// The reference's input tensor from the same files: PIL's bicubic ported bit for bit, the decode by the image crate
#[test]
#[ignore = "needs the reference dump and the sample images (WORLDMIRROR_IMAGES)"]
fn preprocessing_matches_the_reference_input() -> Result<()> {
    let (Some((_, reference)), Some(dir)) = (paths(), std::env::var_os("WORLDMIRROR_IMAGES"))
    else {
        return Ok(());
    };
    let refs = inference_tensor::safetensors::load(&reference, &Device::Cpu)?;
    let mut files = std::fs::read_dir(dir)?
        .map(|e| e.map(|e| e.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    files.sort();
    let images = files
        .iter()
        .map(|f| {
            let rgb = image::open(f)
                .map_err(inference_tensor::Error::wrap)?
                .to_rgb8();
            Ok(super::preprocess::RgbImage {
                width: rgb.width() as usize,
                height: rgb.height() as usize,
                pixels: rgb.into_raw(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let ours = super::preprocess::images_to_tensor(&images, 518, 14, &Device::Cpu)?;
    let diff = ((&ours - &refs["input.img"])?.abs()? * 255.0)?;
    let max = diff.flatten_all()?.max(0)?.to_scalar::<f32>()?;
    let off = diff
        .gt(0.5)?
        .to_dtype(DType::F32)?
        .mean_all()?
        .to_scalar::<f32>()?;
    println!(
        "input: max {max:.1} levels, {:.3}% of values differ",
        off * 100.0
    );
    Ok(())
}

// A whole run on WORLDMIRROR_IMAGES into WORLDMIRROR_OUT, printing the report
#[test]
#[ignore = "needs the WorldMirror checkpoint, images and a GPU; writes files"]
fn reconstruct_writes_every_output() -> Result<()> {
    let (Some((dir, _)), Some(images), Some(out)) = (
        paths(),
        std::env::var_os("WORLDMIRROR_IMAGES"),
        std::env::var_os("WORLDMIRROR_OUT"),
    ) else {
        return Ok(());
    };
    let device = Device::new_cuda(0)?;
    let model = WorldMirror::load(&dir, DType::BF16, &device)?;
    let report = super::run::reconstruct(
        &model,
        &dir,
        std::path::Path::new(&images),
        std::path::Path::new(&out),
        &super::run::ReconstructOptions::default(),
    )?;
    println!("{report:?}");
    Ok(())
}
