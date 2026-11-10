// Parity against the reference's second stage on a JoVox structure (`mesh_ref.py`, scratch): TRELLIS2_DIR is the
// TRELLIS.2-4B folder, TRELLIS2_REF the dump. Ignored by default: they need the checkpoint and a GPU.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use inference_tensor::nn::VarBuilder;
use inference_tensor::{DType, Device, Result, Tensor};

use super::decoder::{SHAPE_DECODER, ShapeDecoder, dual_grid};
use super::mesh::dual_grid_mesh;
use super::shape_flow::{SHAPE_FLOW_1024, SHAPE_SAMPLER, ShapeFlow};
use crate::worldmirror::parity::compare;

const FLOW_FILE: &str = "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors";
const DECODER_FILE: &str = "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors";

fn paths() -> Option<(PathBuf, PathBuf)> {
    Some((
        std::env::var_os("TRELLIS2_DIR")?.into(),
        std::env::var_os("TRELLIS2_REF")?.into(),
    ))
}

fn coords(t: &Tensor) -> Result<Vec<[i32; 3]>> {
    Ok(t.to_dtype(DType::I64)?
        .to_vec2::<i64>()?
        .iter()
        .map(|r| [r[0] as i32, r[1] as i32, r[2] as i32])
        .collect())
}

fn vb(path: PathBuf, device: &Device) -> Result<VarBuilder<'static>> {
    // SAFETY: the checkpoint file is not modified while the model is loaded
    unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device) }
}

#[test]
#[ignore = "needs the TRELLIS.2 checkpoint, a reference dump and a GPU"]
fn the_shape_flow_samples_the_reference_latents() -> Result<()> {
    let Some((dir, reference)) = paths() else {
        return Ok(());
    };
    let device = Device::new_cuda(0)?;
    let refs = inference_tensor::safetensors::load(&reference, &Device::Cpu)?;
    // TRELLIS2_F32 runs the blocks in f32, against a dump made with F32=1
    let dtype = if std::env::var_os("TRELLIS2_F32").is_some() {
        DType::F32
    } else {
        DType::BF16
    };
    let flow = ShapeFlow::load(SHAPE_FLOW_1024, dtype, vb(dir.join(FLOW_FILE), &device)?)?;
    let cells = coords(&refs["coords"])?;
    let cell_coords: Vec<[usize; 3]> = cells.iter().map(|c| c.map(|v| v as usize)).collect();
    let rope = flow.rope(&cell_coords)?;
    let cond = refs["cond"].to_device(&device)?;
    let kv = flow.condition(&Tensor::cat(&[&cond, &cond.zeros_like()?], 0)?)?;
    for call in 0..2 {
        let x = refs[&format!("call{call}.x")]
            .to_device(&device)?
            .unsqueeze(0)?;
        let t = refs[&format!("call{call}.t")].to_vec1::<f32>()?[0];
        let kv1: Vec<_> = kv.iter().map(|k| k.row(call)).collect::<Result<_>>()?;
        let v = flow
            .forward_inspect(&x, t, &rope, &kv1, &mut |name, h| {
                let key = format!("call0.{name}");
                if call == 0
                    && let Some(want) = refs.get(&key)
                {
                    let (max, rms) =
                        compare(&h.squeeze(0).unwrap_or(h.clone()), want).expect("comparable");
                    println!("  {name}: max {max:.3e} rms {rms:.3e}");
                }
            })?
            .squeeze(0)?;
        let (max, rms) = compare(&v, &refs[&format!("call{call}.out")])?;
        println!(
            "first step, {}: max {max:.3e} rms {rms:.3e}",
            ["condition", "zero condition"][call]
        );
    }
    let noise = refs["noise"].to_device(&device)?.unsqueeze(0)?;
    let start = std::time::Instant::now();
    let x = flow.sample(&noise, &rope, &kv, &SHAPE_SAMPLER, &mut |_| Ok(()))?;
    println!("sampled in {:.2} s", start.elapsed().as_secs_f64());
    let norm: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("pipeline.json"))?)
            .map_err(inference_tensor::Error::wrap)?;
    let stat = |k: &str| -> Vec<f32> {
        norm["args"]["shape_slat_normalization"][k]
            .as_array()
            .expect("32 values")
            .iter()
            .map(|v| v.as_f64().expect("a number") as f32)
            .collect()
    };
    let slat = super::shape_flow::denormalize(&x, &stat("mean"), &stat("std"))?.squeeze(0)?;
    let (max, rms) = compare(&slat, &refs["slat"])?;
    println!("latents: max {max:.3e} rms {rms:.3e}");
    assert!(rms < 5e-2, "latent relative rms {rms}");
    Ok(())
}

#[test]
#[ignore = "needs the TRELLIS.2 checkpoint, a reference dump and a GPU"]
fn the_decoder_and_mesh_match_the_reference() -> Result<()> {
    let Some((dir, reference)) = paths() else {
        return Ok(());
    };
    let device = Device::new_cuda(0)?;
    let refs = inference_tensor::safetensors::load(&reference, &Device::Cpu)?;
    let decoder = ShapeDecoder::load(
        SHAPE_DECODER,
        DType::F16,
        vb(dir.join(DECODER_FILE), &device)?,
    )?;
    let cells = coords(&refs["slat.coords"])?;
    let latents = refs["slat"].to_device(&device)?;
    let start = std::time::Instant::now();
    let mut stages = Vec::new();
    let out = decoder.forward(&cells, &latents, &mut |i, x| stages.push((i, x.clone())))?;
    println!("decoded in {:.2} s", start.elapsed().as_secs_f64());
    for (i, x) in &stages {
        let Some(want) = refs.get(&format!("dec.{i}.last.coords")) else {
            continue;
        };
        let want = coords(want)?;
        let ours: HashSet<_> = x.coords.iter().collect();
        let theirs: HashSet<_> = want.iter().collect();
        let shared = ours.intersection(&theirs).count();
        let row: HashMap<_, usize> = want.iter().enumerate().map(|(r, c)| (*c, r)).collect();
        let (mut a, mut b) = (Vec::new(), Vec::new());
        for (r, c) in x.coords.iter().enumerate() {
            if let Some(&w) = row.get(c) {
                a.push(r as u32);
                b.push(w as u32);
            }
        }
        let pick = |t: &Tensor, rows: Vec<u32>| -> Result<Tensor> {
            let n = rows.len();
            t.to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .index_select(&Tensor::from_vec(rows, n, &Device::Cpu)?, 0)
        };
        let (max, rms) = compare(
            &pick(&x.feats, a)?,
            &pick(&refs[&format!("dec.{i}.last.feats")], b)?,
        )?;
        println!(
            "stage {i}: {} cells vs {} ({} shared), features on shared: max {max:.3e} rms {rms:.3e}",
            x.coords.len(),
            want.len(),
            shared
        );
    }

    // the mesh from the reference's own decoder output, so it is checked apart from the decoder's drift
    let ref_cells = coords(&refs["dec.out.coords"])?;
    let (dual, crossed, split) = dual_grid(&refs["dec.out"])?;
    let mesh = dual_grid_mesh(&ref_cells, &dual, &crossed, &split, 1024);
    let want_v = refs["mesh.v"].to_vec2::<f32>()?;
    let want_f = refs["mesh.f"].to_dtype(DType::I64)?.to_vec2::<i64>()?;
    let worst = mesh
        .vertices
        .iter()
        .zip(&want_v)
        .flat_map(|(a, b)| (0..3).map(move |k| (a[k] - b[k]).abs()))
        .fold(0f32, f32::max);
    let ours: HashSet<[u32; 3]> = mesh.faces.iter().copied().collect();
    let theirs: HashSet<[u32; 3]> = want_f
        .iter()
        .map(|f| [f[0] as u32, f[1] as u32, f[2] as u32])
        .collect();
    println!(
        "mesh from the reference's outputs: {} faces vs {}, {} equal; vertices {} vs {}, worst {worst:.2e}",
        mesh.faces.len(),
        want_f.len(),
        ours.intersection(&theirs).count(),
        mesh.vertices.len(),
        want_v.len()
    );
    assert_eq!(ours, theirs, "faces");

    let (dual, crossed, split) = dual_grid(&out.feats)?;
    let ours = dual_grid_mesh(&out.coords, &dual, &crossed, &split, 1024);
    println!(
        "mesh from our decoder: {} vertices, {} faces",
        ours.vertices.len(),
        ours.faces.len()
    );
    Ok(())
}
