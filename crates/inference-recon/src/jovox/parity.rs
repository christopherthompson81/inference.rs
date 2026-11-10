// Parity against the reference's recorded forwards on a synthetic condition: JOVOX_DIR is the checkpoint directory,
// JOVOX_REF the dump (`plan`, and `f{i}_{coords,state,r,level,cond,logits}` for a few forwards).

use std::path::PathBuf;

use inference_tensor::nn::VarBuilder;
use inference_tensor::{DType, Device, Result, Tensor};

use super::flow::Flow;
use super::sampler::{SLOTS, Tokens};
use super::{JOVOX_64, ROOT_END, SCHEDULE};
use crate::worldmirror::parity::compare;

// recorded forwards: the first, one with the cell level filling in, one late
const FORWARDS: [usize; 3] = [0, 30, 60];

fn paths() -> Option<(PathBuf, PathBuf)> {
    Some((
        std::env::var_os("JOVOX_DIR")?.into(),
        std::env::var_os("JOVOX_REF")?.into(),
    ))
}

#[test]
fn the_plan_ends_on_its_cleanup_steps() {
    let plan = SCHEDULE.plan(ROOT_END, 72, 1.0);
    assert_eq!(plan.len(), 72 + SCHEDULE.cleanup_steps);
    let last = plan[plan.len() - 1];
    assert_eq!(last, [1.0, 1.0, 0.0, 1.0]);
    // the root window closes by 0.8: every root slot is revealed by then
    let closed = (0.8f64 * 72.0) as usize;
    assert_eq!(plan[closed][2], 1.0);
}

#[test]
#[ignore = "needs a reference dump"]
fn the_plan_matches_the_reference() -> Result<()> {
    let Some((_, reference)) = paths() else {
        return Ok(());
    };
    let refs = inference_tensor::safetensors::load(&reference, &Device::Cpu)?;
    for (name, (end, steps, phase)) in [("plan", (0.8, 72, 1.0)), ("plan_b", (0.7, 40, 0.3))] {
        let want = refs[name].to_dtype(DType::F64)?.to_vec2::<f64>()?;
        let got = SCHEDULE.plan(end, steps, phase);
        assert_eq!(got.len(), want.len(), "{name} rows");
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            for (a, b) in g.iter().zip(w) {
                assert!((a - b).abs() < 1e-12, "{name} row {i}: {g:?} vs {w:?}");
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "needs the JoVox checkpoint, a reference dump and a GPU"]
fn recorded_forwards_match_the_reference() -> Result<()> {
    let Some((dir, reference)) = paths() else {
        return Ok(());
    };
    let device = Device::new_cuda(0)?;
    let refs = inference_tensor::safetensors::load(&reference, &Device::Cpu)?;
    // SAFETY: the checkpoint file is not modified while the model is loaded
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::F32, &device)?
    };
    let flow = Flow::load(JOVOX_64, DType::BF16, vb.pp("sparse_structure_flow_model"))?;
    for i in FORWARDS {
        let get = |k: &str| refs[&format!("f{i}_{k}")].clone();
        let coords = get("coords").to_dtype(DType::I64)?.to_vec2::<i64>()?;
        let state = get("state").to_dtype(DType::I64)?.to_vec2::<i64>()?;
        let levels = get("level").to_dtype(DType::I64)?.to_vec1::<i64>()?;
        let r = get("r").to_vec2::<f32>()?;
        // the guided batch repeats the tokens under batch 1; keep batch 0's
        let rows: Vec<usize> = (0..coords.len()).filter(|&t| coords[t][0] == 0).collect();
        let tokens = Tokens {
            levels: rows.iter().map(|&t| levels[t] as u8).collect(),
            coords: rows
                .iter()
                .map(|&t| std::array::from_fn(|a| coords[t][1 + a] as usize))
                .collect(),
            slots: rows
                .iter()
                .map(|&t| std::array::from_fn::<u16, SLOTS, _>(|s| state[t][s] as u16))
                .collect(),
            ratios: [r[0][0], r[0][1]],
        };
        let kv = flow.condition(&get("cond").to_device(&device)?)?;
        let logits = flow.forward(&tokens, &kv)?;
        let n = rows.len();
        let want = get("logits").reshape((2, n, SLOTS, ()))?;
        let (max, rms) = compare(&logits, &want)?;
        let agree = argmax_agreement(&logits, &want)?;
        println!(
            "forward {i}: {n} tokens, max {max:.3e} rms {rms:.3e}, argmax agreement {agree:.5}"
        );
        assert!(rms < 2e-2, "forward {i}: relative rms {rms}");
    }
    Ok(())
}

fn argmax_agreement(a: &Tensor, b: &Tensor) -> Result<f64> {
    let a = a
        .to_device(&Device::Cpu)?
        .argmax(3)?
        .flatten_all()?
        .to_vec1::<u32>()?;
    let b = b.argmax(3)?.flatten_all()?.to_vec1::<u32>()?;
    let same = a.iter().zip(&b).filter(|(x, y)| x == y).count();
    Ok(same as f64 / a.len() as f64)
}

#[test]
#[ignore = "needs the JoVox checkpoint, a reference dump and a GPU"]
fn a_structure_samples_under_the_recorded_condition() -> Result<()> {
    let Some((dir, reference)) = paths() else {
        return Ok(());
    };
    let device = Device::new_cuda(0)?;
    let refs = inference_tensor::safetensors::load(&reference, &Device::Cpu)?;
    // SAFETY: the checkpoint file is not modified while the model is loaded
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::F32, &device)?
    };
    let flow = Flow::load(JOVOX_64, DType::BF16, vb.pp("sparse_structure_flow_model"))?;
    let both = refs["f0_cond"].to_device(&device)?;
    let kv = flow.condition(&both)?;
    for seed in 0..3 {
        let start = std::time::Instant::now();
        let options = super::SampleOptions {
            seed,
            ..Default::default()
        };
        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(seed);
        let voxels =
            super::sample_structure(&flow, &kv, &options, &mut rng, &crate::progress::NoProgress)?;
        println!(
            "seed {seed}: {} voxels, {:.2} s",
            voxels.len(),
            start.elapsed().as_secs_f64()
        );
    }
    Ok(())
}

// A counter hash (splitmix64) both sides draw from, so the samplers can be compared draw for draw
struct Splitmix(u64);

impl super::sampler::Uniform for Splitmix {
    fn fill(&mut self, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| {
                self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = self.0;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                (z >> 40) as f32 / (1u64 << 24) as f32
            })
            .collect()
    }
}

fn iou(a: &[[usize; 3]], b: &[[usize; 3]]) -> f64 {
    let a: std::collections::HashSet<_> = a.iter().collect();
    let b: std::collections::HashSet<_> = b.iter().collect();
    a.intersection(&b).count() as f64 / a.union(&b).count().max(1) as f64
}

// JOVOX_DIR, JOVOX_DINOV3 (the encoder's directory), JOVOX_IMAGE and JOVOX_E2E (`e2e_ref.py`'s dump for that image,
// with `det.{seed}` structures sampled under the shared draws). JOVOX_F32 runs the flow in f32, against a dump made
// with F32=1: the samples must then be the reference's; in bf16 rounding flips near-tied picks and they drift apart.
#[test]
#[ignore = "needs the JoVox and DINOv3 checkpoints, a reference dump and a GPU"]
fn an_image_matches_the_reference_end_to_end() -> Result<()> {
    let (Some(dir), Some(dino), Some(image), Some(reference)) = (
        std::env::var_os("JOVOX_DIR"),
        std::env::var_os("JOVOX_DINOV3"),
        std::env::var_os("JOVOX_IMAGE"),
        std::env::var_os("JOVOX_E2E"),
    ) else {
        return Ok(());
    };
    let f32_flow = std::env::var_os("JOVOX_F32").is_some();
    let device = Device::new_cuda(0)?;
    let refs = inference_tensor::safetensors::load(PathBuf::from(reference), &Device::Cpu)?;
    let rgba = crate::read_image(std::path::Path::new(&image))
        .map_err(inference_tensor::Error::wrap)?
        .to_rgba8();
    let input = super::Rgba {
        width: rgba.width() as usize,
        height: rgba.height() as usize,
        pixels: rgba.into_raw(),
    };
    let prepared = super::prepare(&input, super::DEFAULT_FILL)?;
    let want = refs["prep"].flatten_all()?.to_vec1::<u8>()?;
    assert_eq!(prepared.pixels, want, "prepared image");

    let dtype = if f32_flow { DType::F32 } else { DType::BF16 };
    let model = super::JoVox::load(dir.as_ref(), dino.as_ref(), dtype, &device)?;
    let cond = model.condition(&prepared)?;
    let (max, rms) = compare(&cond, &refs["cond"])?;
    println!("condition: max {max:.3e} rms {rms:.3e}");
    assert!(rms < 1e-4, "condition relative rms {rms}");

    let kv = model
        .flow
        .condition(&Tensor::cat(&[&cond, &cond.zeros_like()?], 0)?)?;
    for seed in 0.. {
        let Some(want) = refs.get(&format!("det.{seed}")) else {
            break;
        };
        let want: Vec<[usize; 3]> = want
            .to_vec2::<i64>()?
            .iter()
            .map(|r| [r[0] as usize, r[1] as usize, r[2] as usize])
            .collect();
        let ours = super::sample_structure(
            &model.flow,
            &kv,
            &Default::default(),
            &mut Splitmix(seed),
            &crate::progress::NoProgress,
        )?;
        let iou = iou(&ours, &want);
        println!(
            "shared draws, seed {seed}: ours {} voxels, reference {}, IoU {iou:.4}",
            ours.len(),
            want.len()
        );
        if f32_flow {
            assert_eq!(ours, want, "seed {seed}");
        }
    }
    Ok(())
}
