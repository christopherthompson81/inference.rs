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
    let x = flow.sample(&noise, &rope, &kv, None, &SHAPE_SAMPLER, &mut |_| Ok(()))?;
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
    let (out, _) = decoder.forward(&cells, &latents, None, &mut |i, x| {
        stages.push((i, x.clone()))
    })?;
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
    let grid = dual_grid(&refs["dec.out"])?;
    let mesh = dual_grid_mesh(&ref_cells, &grid.vertices, &grid.crossed, &grid.split, 1024);
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

    let grid = dual_grid(&out.feats)?;
    let ours = dual_grid_mesh(
        &out.coords,
        &grid.vertices,
        &grid.crossed,
        &grid.split,
        1024,
    );
    println!(
        "mesh from our decoder: {} vertices, {} faces",
        ours.vertices.len(),
        ours.faces.len()
    );
    Ok(())
}

// Mean and 99th-percentile distance from each of `a`'s vertices to the nearest of `b`'s (a grid of `cell` side)
fn nearest(a: &[[f32; 3]], b: &[[f32; 3]], cell: f32) -> (f32, f32) {
    let key = |p: &[f32; 3]| p.map(|v| (v / cell).floor() as i32);
    let mut grid: HashMap<[i32; 3], Vec<usize>> = HashMap::new();
    for (i, p) in b.iter().enumerate() {
        grid.entry(key(p)).or_default().push(i);
    }
    let mut d: Vec<f32> = a
        .iter()
        .map(|p| {
            let k = key(p);
            let mut best = f32::INFINITY;
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if let Some(ids) = grid.get(&[k[0] + dx, k[1] + dy, k[2] + dz]) {
                            for &i in ids {
                                let q = b[i];
                                let dist = ((p[0] - q[0]).powi(2)
                                    + (p[1] - q[1]).powi(2)
                                    + (p[2] - q[2]).powi(2))
                                .sqrt();
                                best = best.min(dist);
                            }
                        }
                    }
                }
            }
            best.min(cell)
        })
        .collect();
    d.sort_by(f32::total_cmp);
    let mean = d.iter().sum::<f32>() / d.len() as f32;
    (mean, d[d.len() * 99 / 100])
}

fn mesh_of(refs: &HashMap<String, Tensor>, v: &str, f: &str) -> Result<super::mesh::Mesh> {
    Ok(super::mesh::Mesh {
        vertices: refs[v]
            .to_vec2::<f32>()?
            .iter()
            .map(|r| [r[0], r[1], r[2]])
            .collect(),
        faces: refs[f]
            .to_dtype(DType::I64)?
            .to_vec2::<i64>()?
            .iter()
            .map(|r| [r[0] as u32, r[1] as u32, r[2] as u32])
            .collect(),
    })
}

#[test]
#[ignore = "needs a reference dump"]
fn holes_and_simplification_match_the_reference() -> Result<()> {
    let Some((_, reference)) = paths() else {
        return Ok(());
    };
    let refs = inference_tensor::safetensors::load(&reference, &Device::Cpu)?;
    let mut mesh = mesh_of(&refs, "mesh.v", "mesh.f")?;
    let start = std::time::Instant::now();
    super::cleanup::fill_holes(&mut mesh, super::cleanup::MAX_HOLE_PERIMETER);
    let filled = mesh_of(&refs, "filled.v", "filled.f")?;
    println!(
        "filled in {:.2} s: {} vertices / {} faces vs {} / {}",
        start.elapsed().as_secs_f64(),
        mesh.vertices.len(),
        mesh.faces.len(),
        filled.vertices.len(),
        filled.faces.len()
    );
    assert_eq!(mesh.faces.len(), filled.faces.len(), "filled faces");
    let start = std::time::Instant::now();
    let mut ours = filled.clone();
    super::cleanup::simplify(&mut ours, super::cleanup::SIMPLIFY_TARGET);
    let theirs = mesh_of(&refs, "simple.v", "simple.f")?;
    let cell = 2.0 / 1024.0;
    let (ab_mean, ab_99) = nearest(&ours.vertices, &theirs.vertices, cell);
    let (ba_mean, ba_99) = nearest(&theirs.vertices, &ours.vertices, cell);
    let (fb_mean, fb_99) = nearest(&ours.vertices, &filled.vertices, cell);
    println!(
        "simplified in {:.2} s: {} vertices / {} faces vs {} / {}; nearest vertex ours->ref mean {ab_mean:.2e} p99 \
         {ab_99:.2e}, ref->ours mean {ba_mean:.2e} p99 {ba_99:.2e}, ours->unsimplified mean {fb_mean:.2e} p99 {fb_99:.2e}",
        start.elapsed().as_secs_f64(),
        ours.vertices.len(),
        ours.faces.len(),
        theirs.vertices.len(),
        theirs.faces.len()
    );
    Ok(())
}

// JOVOX_DIR, JOVOX_DINOV3 and JOVOX_E2E (the JoVox dump whose `prep` image TRELLIS2_REF was made from)
#[test]
#[ignore = "needs the JoVox and DINOv3 checkpoints, reference dumps and a GPU"]
fn the_condition_matches_the_reference() -> Result<()> {
    let (Some((_, reference)), Some(jovox), Some(dino), Some(e2e)) = (
        paths(),
        std::env::var_os("JOVOX_DIR"),
        std::env::var_os("JOVOX_DINOV3"),
        std::env::var_os("JOVOX_E2E"),
    ) else {
        return Ok(());
    };
    let device = Device::new_cuda(0)?;
    let refs = inference_tensor::safetensors::load(&reference, &Device::Cpu)?;
    let e2e = inference_tensor::safetensors::load(PathBuf::from(e2e), &Device::Cpu)?;
    let prep = &e2e["prep"];
    let (h, w, _) = prep.dims3()?;
    let image = crate::jovox::Rgba {
        width: w,
        height: h,
        pixels: prep.flatten_all()?.to_vec1::<u8>()?,
    };
    let (rgb, size) = super::crop_object(&image)?;
    let want = refs["pre"].flatten_all()?.to_vec1::<u8>()?;
    let (ph, pw, _) = refs["pre"].dims3()?;
    assert_eq!(size, (pw, ph), "crop size");
    assert_eq!(rgb, want, "cropped object");
    let model = crate::jovox::JoVox::load(jovox.as_ref(), dino.as_ref(), DType::BF16, &device)?;
    let cond = model.encode_rgb(
        &super::condition_image(&image)?,
        (super::RESOLUTION, super::RESOLUTION),
    )?;
    let (max, rms) = compare(&cond, &refs["cond"])?;
    println!("crop {pw}x{ph} equal; condition at 1024: max {max:.3e} rms {rms:.3e}");
    assert!(rms < 1e-4, "condition relative rms {rms}");
    Ok(())
}

fn set_iou(a: &[[i32; 3]], b: &[[i32; 3]]) -> (usize, usize, usize) {
    let a: HashSet<_> = a.iter().collect();
    let b: HashSet<_> = b.iter().collect();
    (a.len(), b.len(), a.intersection(&b).count())
}

// TRELLIS2_DIR, TRELLIS_LARGE_DIR (TRELLIS-image-large) and TRELLIS2_FULL_REF (`full_ref.py ... 1024_cascade`): each
// stage fed the reference's own inputs (its noise, cells and previous stage)
#[test]
#[ignore = "needs the TRELLIS.2 checkpoints, a reference dump and a GPU"]
fn every_pipeline_stage_matches_the_reference() -> Result<()> {
    use super::decoder::TEXTURE_DECODER;
    use super::shape_flow::{
        STRUCTURE_FLOW, STRUCTURE_SAMPLER, TEXTURE_FLOW, TEXTURE_SAMPLER, denormalize,
    };
    use super::structure::{STRUCTURE_DECODER, StructureDecoder, grid_cells, occupied};
    let (Some(dir), Some(large), Some(reference)) = (
        std::env::var_os("TRELLIS2_DIR").map(PathBuf::from),
        std::env::var_os("TRELLIS_LARGE_DIR").map(PathBuf::from),
        std::env::var_os("TRELLIS2_FULL_REF"),
    ) else {
        return Ok(());
    };
    let device = Device::new_cuda(0)?;
    let refs = inference_tensor::safetensors::load(PathBuf::from(reference), &Device::Cpu)?;
    let get = |k: &str| refs[k].to_device(&device);
    let pipeline: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("pipeline.json"))?)
            .map_err(inference_tensor::Error::wrap)?;
    let stat = |group: &str, k: &str| -> Vec<f32> {
        pipeline["args"][group][k]
            .as_array()
            .expect("32 values")
            .iter()
            .map(|v| v.as_f64().expect("a number") as f32)
            .collect()
    };
    let shape_stats = (
        stat("shape_slat_normalization", "mean"),
        stat("shape_slat_normalization", "std"),
    );
    let tex_stats = (
        stat("tex_slat_normalization", "mean"),
        stat("tex_slat_normalization", "std"),
    );
    let report = |name: &str, a: &Tensor, b: &Tensor| -> Result<f32> {
        let (max, rms) = compare(a, b)?;
        println!("{name:>24}: max {max:.3e} rms {rms:.3e}");
        Ok(rms)
    };
    let flow = |file: &str, spec| -> Result<ShapeFlow> {
        ShapeFlow::load(
            spec,
            DType::BF16,
            vb(dir.join("ckpts").join(file), &device)?,
        )
    };
    let sample = |flow: &ShapeFlow,
                  cells: &[[usize; 3]],
                  cond: &Tensor,
                  noise: &Tensor,
                  concat: Option<&Tensor>,
                  params| {
        let rope = flow.rope(cells)?;
        let kv = flow.condition(&Tensor::cat(&[cond, &cond.zeros_like()?], 0)?)?;
        flow.sample(noise, &rope, &kv, concat, params, &mut |_| Ok(()))
    };
    let cells_of = |k: &str| -> Result<Vec<[usize; 3]>> {
        Ok(coords(&refs[k])?
            .iter()
            .map(|c| c.map(|v| v as usize))
            .collect())
    };

    // structure flow: [1, 8, 16, 16, 16] channel-first in the reference, x-major tokens here
    let grid: Vec<[usize; 3]> = grid_cells(16)
        .iter()
        .map(|c| c.map(|v| v as usize))
        .collect();
    let to_tokens =
        |t: Tensor| -> Result<Tensor> { t.reshape((1, 8, 4096))?.transpose(1, 2)?.contiguous() };
    let z = sample(
        &flow("ss_flow_img_dit_1_3B_64_bf16.safetensors", STRUCTURE_FLOW)?,
        &grid,
        &get("cond512")?,
        &to_tokens(get("s0.noise")?)?,
        None,
        &STRUCTURE_SAMPLER,
    )?;
    report("structure latent", &z, &to_tokens(refs["s0.out"].clone())?)?;
    let decoder = StructureDecoder::load(
        STRUCTURE_DECODER,
        16,
        DType::F16,
        vb(
            large.join("ckpts/ss_dec_conv3d_16l8_fp16.safetensors"),
            &device,
        )?,
    )?;
    let logits = decoder.forward(&to_tokens(get("s0.out")?)?.squeeze(0)?)?;
    report(
        "structure logits",
        &logits,
        &refs["ss.logits"].flatten_all()?,
    )?;
    let ours: Vec<[i32; 3]> = occupied(&logits.to_vec1::<f32>()?, 64, 32)
        .iter()
        .map(|c| c.map(|v| v as i32))
        .collect();
    println!(
        "structure cells (ours, ref, shared): {:?}",
        set_iou(&ours, &coords(&refs["ss.coords"])?)
    );

    // shape at 512, upsampled cells, shape at 1024, texture at 1024
    let lr = sample(
        &flow(
            "slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
            SHAPE_FLOW_1024,
        )?,
        &cells_of("s1.coords")?,
        &get("cond512")?,
        &get("s1.noise")?.unsqueeze(0)?,
        None,
        &SHAPE_SAMPLER,
    )?;
    report("shape latents 512", &lr, &refs["s1.out"])?;
    let shape_decoder = ShapeDecoder::load(
        SHAPE_DECODER,
        DType::F16,
        vb(dir.join(DECODER_FILE), &device)?,
    )?;
    let lr_ref = denormalize(
        &get("s1.out")?.unsqueeze(0)?,
        &shape_stats.0,
        &shape_stats.1,
    )?
    .squeeze(0)?;
    let up = shape_decoder.upsample(&coords(&refs["s1.coords"])?, &lr_ref, 4)?;
    println!(
        "upsampled cells (ours, ref, shared): {:?}",
        set_iou(&up, &coords(&refs["up.coords"])?)
    );
    let mut q: Vec<[i32; 3]> = up
        .iter()
        .map(|c| c.map(|v| ((v as f32 + 0.5) / 512.0 * 64.0) as i32))
        .collect();
    q.sort_unstable();
    q.dedup();
    println!(
        "1024 cells (ours, ref, shared): {:?}",
        set_iou(&q, &coords(&refs["s2.coords"])?)
    );
    let hr = sample(
        &flow(
            "slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors",
            SHAPE_FLOW_1024,
        )?,
        &cells_of("s2.coords")?,
        &get("cond1024")?,
        &get("s2.noise")?.unsqueeze(0)?,
        None,
        &SHAPE_SAMPLER,
    )?;
    report("shape latents 1024", &hr, &refs["s2.out"])?;
    let tex = sample(
        &flow(
            "slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors",
            TEXTURE_FLOW,
        )?,
        &cells_of("s3.coords")?,
        &get("cond1024")?,
        &get("s3.noise")?.unsqueeze(0)?,
        Some(&get("s3.concat")?.unsqueeze(0)?),
        &TEXTURE_SAMPLER,
    )?;
    report("texture latents", &tex, &refs["s3.out"])?;

    // decoding the reference's latents: shape subdivisions guide the texture decoder
    let hr_ref = denormalize(
        &get("s2.out")?.unsqueeze(0)?,
        &shape_stats.0,
        &shape_stats.1,
    )?
    .squeeze(0)?;
    let tex_ref =
        denormalize(&get("s3.out")?.unsqueeze(0)?, &tex_stats.0, &tex_stats.1)?.squeeze(0)?;
    let cells = coords(&refs["s2.coords"])?;
    let (shape_out, subs) = shape_decoder.forward(&cells, &hr_ref, None, &mut |_, _| {})?;
    drop(shape_decoder);
    let tex_decoder = ShapeDecoder::load(
        TEXTURE_DECODER,
        DType::F16,
        vb(
            dir.join("ckpts/tex_dec_next_dc_f16c32_fp16.safetensors"),
            &device,
        )?,
    )?;
    let (tex_out, _) = tex_decoder.forward(&cells, &tex_ref, Some(&subs), &mut |_, _| {})?;
    assert_eq!(
        tex_out.coords, shape_out.coords,
        "the texture follows the shape's cells"
    );
    let attrs = ((tex_out.feats * 0.5)? + 0.5)?;
    let want = coords(&refs["tex.coords"])?;
    let row: HashMap<_, usize> = want.iter().enumerate().map(|(r, c)| (*c, r)).collect();
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for (r, c) in tex_out.coords.iter().enumerate() {
        if let Some(&w) = row.get(c) {
            a.push(r as u32);
            b.push(w as u32);
        }
    }
    println!(
        "texture cells: {} vs {}, {} shared",
        tex_out.coords.len(),
        want.len(),
        a.len()
    );
    let n = a.len();
    let pick = |t: &Tensor, rows: Vec<u32>| -> Result<Tensor> {
        t.to_device(&Device::Cpu)?
            .index_select(&Tensor::from_vec(rows, n, &Device::Cpu)?, 0)
    };
    report(
        "texture attributes",
        &pick(&attrs, a)?,
        &pick(&refs["tex.attrs"], b)?,
    )?;

    // vertex attributes sampled from the reference's own volume at its own vertices
    let values: Vec<[f32; 6]> = refs["tex.attrs"]
        .to_vec2::<f32>()?
        .iter()
        .map(|r| std::array::from_fn(|k| r[k]))
        .collect();
    let points: Vec<[f32; 3]> = refs["mesh.v"]
        .to_vec2::<f32>()?
        .iter()
        .map(|v| [0, 1, 2].map(|k| (v[k] + 0.5) * 1024.0))
        .collect();
    let sampled = super::glb::sample_attributes(&want, &values, &points);
    let flat: Vec<f32> = sampled
        .iter()
        .flat_map(|p| {
            [
                p.base_color[0],
                p.base_color[1],
                p.base_color[2],
                p.metallic,
                p.roughness,
                p.alpha,
            ]
        })
        .collect();
    let ours = Tensor::from_vec(flat, (sampled.len(), 6), &Device::Cpu)?;
    let rms = report("vertex attributes", &ours, &refs["vertex.attrs"])?;
    assert!(rms < 1e-5, "vertex attributes rms {rms}");
    Ok(())
}
