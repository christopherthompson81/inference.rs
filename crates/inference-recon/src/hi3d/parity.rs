// Parity against the official Hi3D pipeline, stage by stage from the reference's own dump (docs/investigations/
// hi3d_investigation.md): HI3D_DIR is Hi3Dv1.0's folder, HI3D_DINO and HI3D_CLIP the transformers folders of
// facebook/dinov2-large and openai/clip-vit-large-patch14, HI3D_REF the dump's `stages.safetensors` (its folder also
// holds `prepared.png`). Ignored by default: they need the checkpoints and a GPU.

use std::collections::HashMap;
use std::path::PathBuf;

use inference_tensor::nn::VarBuilder;
use inference_tensor::{DType, Device, Result, Tensor};

use super::conditioner::Conditioner;
use super::extract::point_to_mesh;
use super::schedule::{self, Schedule};
use super::transformer::{Grid, Transformer};
use super::vae::{Decoded, Decoder};
use super::{LATENT_MEAN, LATENT_STD};
use crate::worldmirror::parity::compare;

struct Setup {
    dir: PathBuf,
    dino: PathBuf,
    clip: PathBuf,
    reference: PathBuf,
    refs: HashMap<String, Tensor>,
    device: Device,
}

fn setup() -> Result<Option<Setup>> {
    let var = |k: &str| std::env::var_os(k).map(PathBuf::from);
    let (Some(dir), Some(dino), Some(clip), Some(reference)) = (
        var("HI3D_DIR"),
        var("HI3D_DINO"),
        var("HI3D_CLIP"),
        var("HI3D_REF"),
    ) else {
        return Ok(None);
    };
    Ok(Some(Setup {
        dir,
        dino,
        clip,
        refs: inference_tensor::safetensors::load(&reference, &Device::Cpu)?,
        reference,
        device: Device::new_cuda(0)?,
    }))
}

impl Setup {
    fn get(&self, k: &str) -> Result<Tensor> {
        self.refs[k].to_device(&self.device)
    }

    fn cells(&self, k: &str) -> Result<Vec<[i32; 3]>> {
        Ok(self.refs[k]
            .to_dtype(DType::I64)?
            .to_vec2::<i64>()?
            .iter()
            .map(|r| {
                let n = r.len();
                [r[n - 3] as i32, r[n - 2] as i32, r[n - 1] as i32]
            })
            .collect())
    }

    fn transformer(&self, name: &str) -> Result<Transformer> {
        let path = self.dir.join(name).join("model.safetensors");
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &self.device)? };
        Transformer::load(DType::F16, vb)
    }
}

fn report(name: &str, a: &Tensor, b: &Tensor, limit: f32) -> Result<()> {
    let (max, rel) = compare(a, b)?;
    println!("{name}: rel rms {rel:.3e}, max abs {max:.3e}");
    assert!(rel < limit, "{name}: rel rms {rel:.3e} past {limit:.0e}");
    Ok(())
}

#[test]
#[ignore]
fn the_condition_matches() -> Result<()> {
    let Some(s) = setup()? else { return Ok(()) };
    let prepared = image::open(s.reference.with_file_name("prepared.png"))
        .map_err(|e| inference_tensor::Error::Msg(e.to_string()))?
        .to_rgb8();
    let conditioner = Conditioner::load(&s.dino, &s.clip, &s.device)?;
    let cond = conditioner.forward(&prepared)?;
    let want = s.get("cond")?;
    // PIL's own Lanczos resize, to tell the resampler from the tower
    let pil = image::open(s.reference.with_file_name("clip224.png"))
        .map_err(|e| inference_tensor::Error::Msg(e.to_string()))?
        .to_rgb8();
    let ours = super::conditioner::clip_image(&prepared);
    let diff = ours
        .as_raw()
        .iter()
        .zip(pil.as_raw())
        .map(|(&a, &b)| (a as i32 - b as i32).abs())
        .max()
        .unwrap_or(0);
    println!("224 resize: max pixel difference {diff}");
    report(
        "clip pooled of PIL's resize",
        &conditioner.clip_embedding(&pil)?,
        &want.narrow(2, 1024, 1024)?.narrow(1, 0, 1)?.squeeze(1)?,
        1e-3,
    )?;
    report(
        "dino tokens",
        &cond.narrow(2, 0, 1024)?,
        &want.narrow(2, 0, 1024)?,
        1e-3,
    )?;
    report(
        "clip pooled",
        &cond.narrow(2, 1024, 1024)?,
        &want.narrow(2, 1024, 1024)?,
        1e-2,
    )
}

#[test]
#[ignore]
fn the_first_velocities_match() -> Result<()> {
    let Some(s) = setup()? else { return Ok(()) };
    let model = s.transformer("transformer_512")?;
    let cond = s.get("cond")?;
    let kv = model.condition(&cond)?;
    let uncond = model.condition(&cond.zeros_like()?)?;
    let grid = Grid::new(&s.cells("coords64")?, &s.device)?;
    for i in 0..3 {
        let x = s.get(&format!("call{i}_x"))?;
        let t = s.refs[&format!("call{i}_t")].to_vec1::<f32>()?[0];
        let zero = s.refs[&format!("call{i}_cond_zero")].to_vec1::<f32>()?[0] > 0.5;
        let v = model.forward(&x, t, &grid, if zero { &uncond } else { &kv })?;
        report(
            &format!("call {i} (t {t}, zero cond {zero})"),
            &v,
            &s.get(&format!("call{i}_v"))?,
            1e-2,
        )?;
    }
    Ok(())
}

#[test]
#[ignore]
fn sampling_reaches_the_reference_latents() -> Result<()> {
    let Some(s) = setup()? else { return Ok(()) };
    let model = s.transformer("transformer_512")?;
    let cond = s.get("cond")?;
    let (kv, uncond) = (
        model.condition(&cond)?,
        model.condition(&cond.zeros_like()?)?,
    );
    let coords = s.cells("coords64")?;
    let grid = Grid::new(&coords, &s.device)?;
    let schedule = Schedule::new(schedule::STEPS, coords.len());
    let calls: usize = schedule
        .times
        .iter()
        .zip(&schedule.sigmas[1..])
        .map(|(&t, &next)| {
            [t, next]
                .iter()
                .map(|&u| if schedule::guidance(u) != 1.0 { 2 } else { 1 })
                .sum::<usize>()
        })
        .sum();
    println!("{calls} transformer calls");
    let x = schedule::sample(
        &model,
        &s.get("noise")?,
        &grid,
        &kv,
        &uncond,
        schedule::STEPS,
        &mut |_| Ok(()),
    )?;
    let c = LATENT_MEAN.len();
    let mean = Tensor::from_slice(&LATENT_MEAN, (1, c), &s.device)?;
    let std = Tensor::from_slice(&LATENT_STD, (1, c), &s.device)?;
    let latents = x.broadcast_mul(&std)?.broadcast_add(&mean)?;
    report("latents", &latents, &s.get("latents")?, 5e-2)
}

#[test]
#[ignore]
fn the_transformer_layers_match() -> Result<()> {
    let Some(s) = setup()? else { return Ok(()) };
    let layers = inference_tensor::safetensors::load(
        s.reference.with_file_name("layers.safetensors"),
        &Device::Cpu,
    )?;
    let model = s.transformer("transformer_512")?;
    let kv = model.condition(&s.get("cond")?)?;
    let grid = Grid::new(&s.cells("coords64")?, &s.device)?;
    let t = s.refs["call0_t"].to_vec1::<f32>()?[0];
    let mut got = Vec::new();
    model.forward_inspect(&s.get("call0_x")?, t, &grid, &kv, &mut |name, x| {
        got.push((name.to_string(), x.clone()))
    })?;
    for (name, x) in got {
        let (max, rel) = compare(&x, &layers[&name])?;
        println!("{name}: rel rms {rel:.3e}, max abs {max:.3e}");
    }
    Ok(())
}

#[test]
#[ignore]
fn the_first_block_matches_from_the_reference_input() -> Result<()> {
    let Some(s) = setup()? else { return Ok(()) };
    let layers = inference_tensor::safetensors::load(
        s.reference.with_file_name("layers.safetensors"),
        &Device::Cpu,
    )?;
    let model = s.transformer("transformer_512")?;
    let kv = model.condition(&s.get("cond")?)?;
    let grid = Grid::new(&s.cells("coords64")?, &s.device)?;
    let t = s.refs["call0_t"].to_vec1::<f32>()?[0];
    let conv = model.in0_conv1(&layers["input_layer"].to_device(&s.device)?, &grid)?;
    let (max, rel) = compare(&conv, &layers["in0_conv1"])?;
    println!("in0 conv1: rel rms {rel:.3e}, max abs {max:.3e}");
    let x = layers["in1"].to_device(&s.device)?;
    let mut got = Vec::new();
    let out = model.block0_inspect(&x, t, &grid, &kv[0], &mut |name, x| {
        got.push((name.to_string(), x.clone()))
    })?;
    got.push(("block0".into(), out.squeeze(0)?));
    for (name, x) in got {
        let key = match name.as_str() {
            "fused" => "block0".to_string(),
            "block0" => "block0".to_string(),
            n => format!("b0.{n}"),
        };
        if let Some(want) = layers.get(&key) {
            let (max, rel) = compare(&x.squeeze(0)?, want)?;
            println!("{name}: rel rms {rel:.3e}, max abs {max:.3e}");
        }
    }
    Ok(())
}

fn decoder(s: &Setup) -> Result<Decoder> {
    let path = s.dir.join("vae").join("model.safetensors");
    Decoder::load(DType::F16, unsafe {
        VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &s.device)?
    })
}

#[test]
#[ignore]
fn decoding_the_reference_latents_matches() -> Result<()> {
    let Some(s) = setup()? else { return Ok(()) };
    let decoded = decoder(&s)?.forward(&s.get("latents")?, &s.cells("latent_coords")?, 64)?;
    let want = s.cells("dec_grid")?;
    println!(
        "decoded {} cells at {}, the reference {}",
        decoded.coords.len(),
        decoded.resolution,
        want.len()
    );
    let ours: HashMap<[i32; 3], usize> = decoded
        .coords
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, i))
        .collect();
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for (j, c) in want.iter().enumerate() {
        if let Some(&i) = ours.get(c) {
            a.push(i as u32);
            b.push(j as u32);
        }
    }
    println!("{} cells in common", a.len());
    let pick = |t: &Tensor, rows: &[u32]| {
        t.contiguous()
            .unwrap()
            .index_select(&Tensor::from_slice(rows, rows.len(), &s.device).unwrap(), 0)
    };
    report(
        "sign logits",
        &pick(&decoded.signs, &a)?,
        &pick(&s.get("dec_sign")?, &b)?,
        5e-2,
    )?;
    report(
        "attributes",
        &pick(&decoded.attrs, &a)?,
        &pick(&s.get("dec_feat")?, &b)?,
        5e-2,
    )?;
    let shared = a.len() as f64 / want.len().max(decoded.coords.len()) as f64;
    assert!(shared > 0.99, "only {shared:.4} of the cells in common");
    Ok(())
}

#[test]
#[ignore]
fn extraction_from_the_reference_cells_matches() -> Result<()> {
    let Some(s) = setup()? else { return Ok(()) };
    let decoded = Decoded {
        coords: s.cells("dec_grid")?,
        resolution: 512,
        signs: s.refs["dec_sign"].clone(),
        attrs: s.refs["dec_feat"].clone(),
    };
    let t0 = std::time::Instant::now();
    let mut mesh = point_to_mesh(&decoded, &s.cells("latent_coords")?)?;
    let extracted = (mesh.vertices.len(), mesh.faces.len());
    super::extract::fill_holes(&mut mesh);
    let want_v = s.refs["mesh_v"].dims2()?.0;
    let want_f = s.refs["mesh_f"].dims2()?.0;
    println!(
        "extracted {extracted:?}, filled {} faces in {:.1?}; the reference {want_v} vertices, {want_f} faces",
        mesh.faces.len(),
        t0.elapsed()
    );
    // each axis's sorted values against the reference's
    let theirs = s.refs["mesh_v"].to_vec2::<f32>()?;
    let mut max = 0f32;
    for d in 0..3 {
        let mut a: Vec<f32> = mesh.vertices.iter().map(|v| v[d]).collect();
        let mut b: Vec<f32> = theirs.iter().map(|v| v[d]).collect();
        a.sort_by(f32::total_cmp);
        b.sort_by(f32::total_cmp);
        max = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(max, f32::max);
    }
    println!("per-axis sorted vertices: largest difference {max:.3e}");
    assert_eq!(mesh.vertices.len(), want_v);
    assert_eq!(mesh.faces.len(), want_f);
    let faces = s.refs["mesh_f"].to_dtype(DType::I64)?.to_vec2::<i64>()?;
    let same = faces
        .iter()
        .zip(&mesh.faces)
        .filter(|(a, b)| a.iter().zip(b.iter()).all(|(&x, &y)| x == y as i64))
        .count();
    let extracted_same = faces[..extracted.1]
        .iter()
        .zip(&mesh.faces)
        .all(|(a, b)| a.iter().zip(b.iter()).all(|(&x, &y)| x == y as i64));
    println!("{same} of {want_f} faces identical; the extracted ones all: {extracted_same}");
    assert!(extracted_same, "the extracted faces differ");
    // the ear clipper's f32 angles may break a near tie differently from numpy's
    assert!(
        want_f - same < (want_f - extracted.1) / 100,
        "{} filled faces differ",
        want_f - same
    );
    Ok(())
}

// HI3D_COARSE is the coarse mesh the dump refined
#[test]
#[ignore]
fn voxelizing_the_coarse_mesh_matches() -> Result<()> {
    let Some(s) = setup()? else { return Ok(()) };
    let Some(path) = std::env::var_os("HI3D_COARSE") else {
        return Ok(());
    };
    let mesh = crate::trellis2::mesh_io::read_mesh(std::path::Path::new(&path))
        .map_err(|e| inference_tensor::Error::Msg(e.to_string()))?
        .mesh;
    let t0 = std::time::Instant::now();
    let (cells, res) =
        super::voxelize::adaptive(&mesh, 64, 80_000).expect("a grid fits the budget");
    let want = s.cells("coords64")?;
    println!(
        "{} cells at {res} in {:.1?}, the reference {}",
        cells.len(),
        t0.elapsed(),
        want.len()
    );
    assert_eq!(cells, want);
    Ok(())
}

// the whole refinement of HI3D_COARSE under the dump's image at HI3D_RESOLUTION (512 when unset), written to HI3D_OUT
#[test]
#[ignore]
fn a_refinement_runs_end_to_end() -> Result<()> {
    let Some(s) = setup()? else { return Ok(()) };
    let (Some(coarse), Some(out)) = (
        std::env::var_os("HI3D_COARSE"),
        std::env::var_os("HI3D_OUT"),
    ) else {
        return Ok(());
    };
    let resolution = std::env::var("HI3D_RESOLUTION")
        .ok()
        .and_then(|r| r.parse().ok())
        .unwrap_or(512);
    let image = image::open(s.reference.with_file_name("prepared.png"))
        .map_err(|e| inference_tensor::Error::Msg(e.to_string()))?;
    let coarse = crate::trellis2::mesh_io::read_mesh(std::path::Path::new(&coarse))
        .map_err(|e| inference_tensor::Error::Msg(e.to_string()))?
        .mesh;
    let paths = super::pipeline::Hi3dPaths {
        hi3d: s.dir.clone(),
        dinov2: s.dino.clone(),
        clip: s.clip.clone(),
    };
    let t0 = std::time::Instant::now();
    let refined = super::pipeline::refine(
        &paths,
        &image,
        &coarse,
        &super::pipeline::Settings::new(resolution, 0),
        &Timer(std::sync::Mutex::new(t0)),
        &s.device,
    )?;
    println!(
        "{resolution}: {} vertices, {} faces in {:.1?}; passes {:?}",
        refined.mesh.vertices.len(),
        refined.mesh.faces.len(),
        t0.elapsed(),
        refined.passes
    );
    std::fs::write(
        PathBuf::from(out).join(format!("refined_{resolution}.ply")),
        refined.mesh.to_ply(crate::trellis2::mesh::UpAxis::Y),
    )?;
    Ok(())
}

// prints each stage's start, seconds since the run began
struct Timer(std::sync::Mutex<std::time::Instant>);

impl crate::progress::Progress for Timer {
    fn stage(&self, name: &str, _total: usize) {
        println!(
            "{:7.1} s  {name}",
            self.0.lock().unwrap().elapsed().as_secs_f32()
        );
    }
}
