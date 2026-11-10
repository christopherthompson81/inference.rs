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
        let voxels = super::sample_structure(&flow, &kv, &options, &crate::progress::NoProgress)?;
        println!(
            "seed {seed}: {} voxels, {:.2} s",
            voxels.len(),
            start.elapsed().as_secs_f64()
        );
    }
    Ok(())
}
