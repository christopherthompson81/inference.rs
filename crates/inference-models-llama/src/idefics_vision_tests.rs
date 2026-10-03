//! Locks the Idefics2 and Idefics3 SigLIP vision towers on made-up weights, a fixed batch and a ragged patch mask.

use std::collections::HashMap;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use inference_nn::attention::AttentionMask;
use inference_nn::testing::{fill, load_synthesized, names_digest};

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

#[test]
fn idefics2_vision_tower() -> Result<()> {
    let cfg = crate::idefics2::VisionConfig {
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
    };
    let (tower, names) = load_synthesized(&[], HashMap::new(), DType::F32, |vb| {
        Ok(crate::idefics2::VisionTransformer::new(&cfg, vb)?)
    })?;
    assert_eq!(names_digest(names.keys()), NAMES_DIGEST);
    let (pixels, mask) = inputs()?;
    let out = tower.forward(&pixels, &AttentionMask::Custom(mask))?;
    check(
        snapshot(&out)?,
        (
            [-0.17192861, -0.38771605, -0.79541326, -0.19759844],
            11.76387,
            30.913307,
        ),
    );
    Ok(())
}

#[test]
fn idefics3_vision_tower() -> Result<()> {
    let cfg = crate::idefics3::config::Idefics3VisionConfig {
        hidden_size: HIDDEN,
        intermediate_size: 48,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_channels: 3,
        image_size: IMAGE,
        patch_size: PATCH,
        hidden_act: Activation::GeluPytorchTanh,
        layer_norm_eps: 1e-6,
    };
    let (tower, names) = load_synthesized(&[], HashMap::new(), DType::F32, |vb| {
        Ok(crate::idefics3::vision::Idefics3VisionTransformer::new(
            &cfg, vb,
        )?)
    })?;
    assert_eq!(names_digest(names.keys()), NAMES_DIGEST);
    let (pixels, mask) = inputs()?;
    let out = tower.forward(&pixels, &AttentionMask::Custom(mask))?;
    check(
        snapshot(&out)?,
        (
            [-0.17192292, -0.38786536, -0.79548895, -0.19761042],
            11.76403,
            30.913313,
        ),
    );
    Ok(())
}
