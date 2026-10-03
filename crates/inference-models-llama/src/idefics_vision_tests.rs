//! Locks the Idefics2 and Idefics3 SigLIP vision towers on made-up weights, a fixed batch and a ragged patch mask.

use std::collections::HashMap;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use inference_nn::attention::AttentionMask;
use inference_nn::testing::{fill, load_synthesized, names_digest};
use inference_nn::vision::siglip::{SiglipVisionConfig, SiglipVisionTransformer};

use crate::layers::Activation;

const HIDDEN: usize = 32;
const IMAGE: usize = 28;
const PATCH: usize = 7;
const SIDE: usize = IMAGE / PATCH;
// Relative, as the logit snapshots use
const TOL: f32 = 1e-4;
// Both towers read the same SigLIP tensor names
const NAMES_DIGEST: u64 = 0x5450_397b_b663_712a;

// Two images; the second covers only its top-left 3x3 patches, so the ragged position ids and the mask both matter.
fn inputs() -> Result<(Tensor, Tensor)> {
    let pixels = fill("pixel_values", &[2, 3, IMAGE, IMAGE])?;
    let mask = (0..2 * SIDE * SIDE)
        .map(|i| {
            let (image, row, col) = (i / (SIDE * SIDE), (i / SIDE) % SIDE, i % SIDE);
            u8::from(image == 0 || (row < 3 && col < 3))
        })
        .collect::<Vec<_>>();
    Ok((
        pixels,
        Tensor::from_vec(mask, (2, SIDE, SIDE), &Device::Cpu)?,
    ))
}

// First values of each image's first patch, then the sum and L2 norm over the whole output.
fn snapshot(out: &Tensor) -> Result<([f32; 4], f32, f32)> {
    assert_eq!(out.dims(), [2, SIDE * SIDE, HIDDEN]);
    let first = out.get(0)?.get(0)?.to_vec1::<f32>()?;
    let second = out.get(1)?.get(0)?.to_vec1::<f32>()?;
    let probes = [first[0], first[1], second[0], second[1]];
    let sum = out.sum_all()?.to_scalar::<f32>()?;
    let l2 = out.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
    Ok((probes, sum, l2))
}

fn check(actual: ([f32; 4], f32, f32), expected: ([f32; 4], f32, f32)) {
    let close = |a: f32, e: f32| (a - e).abs() <= TOL * e.abs().max(1.0);
    let ok = actual.0.iter().zip(&expected.0).all(|(a, e)| close(*a, *e))
        && close(actual.1, expected.1)
        && close(actual.2, expected.2);
    assert!(ok, "vision output moved: {actual:?}");
}

const SNAPSHOT: ([f32; 4], f32, f32) = (
    [-0.17192292, -0.38786536, -0.87687206, -0.10393213],
    11.595032,
    30.884478,
);

fn idefics2_config() -> SiglipVisionConfig {
    crate::idefics2::VisionConfig {
        hidden_size: HIDDEN,
        intermediate_size: 48,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_channels: 3,
        image_size: IMAGE,
        patch_size: PATCH,
        hidden_act: Activation::GeluPytorchTanh,
        layer_norm_eps: 1e-6,
        attn_dropout: 0.0,
        initializer_range: 0.02,
    }
    .siglip()
}

fn idefics3_config() -> SiglipVisionConfig {
    crate::idefics3::config::Idefics3VisionConfig {
        hidden_size: HIDDEN,
        intermediate_size: 48,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_channels: 3,
        image_size: IMAGE,
        patch_size: PATCH,
        hidden_act: Activation::GeluPytorchTanh,
        layer_norm_eps: 1e-6,
    }
    .siglip()
}

fn tower(cfg: &SiglipVisionConfig) -> Result<SiglipVisionTransformer> {
    let (tower, names) = load_synthesized(&[], HashMap::new(), DType::F32, |vb| {
        Ok(SiglipVisionTransformer::new(cfg, vb)?)
    })?;
    assert_eq!(names_digest(names.keys()), NAMES_DIGEST);
    Ok(tower)
}

#[test]
fn idefics_vision_towers() -> Result<()> {
    let (pixels, mask) = inputs()?;
    for cfg in [idefics2_config(), idefics3_config()] {
        let out = tower(&cfg)?.forward(&pixels, &AttentionMask::Custom(mask.clone()), None)?;
        check(snapshot(&out)?, SNAPSHOT);
    }
    Ok(())
}

// Padded patches are masked out of attention (HF's _prepare_4d_attention_mask), so their pixels cannot move the rest.
#[test]
fn padded_patches_do_not_reach_valid_ones() -> Result<()> {
    let tower = tower(&idefics3_config())?;
    let (pixels, mask) = inputs()?;
    // Rows 21.. of the second image lie in its masked last patch row
    let padded_rows = IMAGE - PATCH;
    let noise = fill("noise", &[1, 3, PATCH, IMAGE])?;
    let second = pixels.get(1)?.unsqueeze(0)?;
    let changed = second.slice_assign(&[0..1, 0..3, padded_rows..IMAGE, 0..IMAGE], &noise)?;
    let changed = Tensor::cat(&[&pixels.get(0)?.unsqueeze(0)?, &changed], 0)?;
    let before = tower.forward(&pixels, &AttentionMask::Custom(mask.clone()), None)?;
    let after = tower.forward(&changed, &AttentionMask::Custom(mask), None)?;
    // the second image's valid patches are its first 3 patches of each of the first 3 rows
    for patch in [0, 1, 2, 4, 5, 6, 8, 9, 10] {
        let diff = (before.get(1)?.get(patch)? - after.get(1)?.get(patch)?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(diff == 0.0, "valid patch {patch} moved by {diff}");
    }
    Ok(())
}
