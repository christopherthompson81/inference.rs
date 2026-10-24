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
    let a = a.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.flatten_all()?;
    let b = b.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.flatten_all()?;
    let diff = (&a - &b)?;
    let max = diff.abs()?.max(0)?.to_scalar::<f32>()?;
    let rms = (diff.sqr()?.mean_all()?.to_scalar::<f32>()? / b.sqr()?.mean_all()?.to_scalar::<f32>()?).sqrt();
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
    let (gmax, grms) = compare(&global.reshape(refs["vgt.global.0"].shape())?, &refs["vgt.global.0"])?;
    println!("layer  0: frame max {lmax:.3e} rms {lrms:.3e}, global max {gmax:.3e} rms {grms:.3e}");
    for (layer, features) in [4, 11, 17, 23].iter().zip(&out.features) {
        let dim = features.dim(2)? / 2;
        let (frames, tokens) = (features.dim(0)?, features.dim(1)?);
        let global = refs[&format!("vgt.global.{layer}")].reshape((frames, tokens, dim))?;
        let (lmax, lrms) = compare(&features.narrow(2, 0, dim)?, &refs[&format!("vgt.frame.{layer}")])?;
        let (gmax, grms) = compare(&features.narrow(2, dim, dim)?, &global)?;
        println!("layer {layer:2}: frame max {lmax:.3e} rms {lrms:.3e}, global max {gmax:.3e} rms {grms:.3e}");
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
    let out = model.trunk.first_frame_block(&refs["blk0.in"].to_device(&device)?, hp, wp, &mut |name, t| {
        seen.insert(name.to_string(), t.clone());
    })?;
    for name in ["norm1", "q_norm", "q_rope", "attn_out", "norm2", "mlp_out"] {
        let (max, rms) = compare(&seen[name], &refs[&format!("blk0.{name}")])?;
        println!("block 0 {name:9}: max {max:.3e} rms {rms:.3e}");
    }
    let (max, rms) = compare(&out, &refs["vgt.frame.0"])?;
    println!("block 0 output   : max {max:.3e} rms {rms:.3e}");
    Ok(())
}
