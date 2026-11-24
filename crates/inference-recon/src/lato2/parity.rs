// Parity against the official LATO.2 image pipeline, stage by stage from the reference's own inputs (`ref.py --f32`,
// scratch; docs/investigations/lato2_investigation.md): LATO2_DIR is 0x4c48/LATO.2's folder, TRELLIS_LARGE_DIR
// TRELLIS-image-large's, DINOV2 dinov2-with-registers-large's model.safetensors, LATO2_REF the dump. Ignored by default: they need the
// checkpoints and a GPU.

use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;

use inference_tensor::{DType, Device, Result, Tensor};

use super::structure::{
    LATENT_CHANNELS, LATENT_SIDE, STRUCTURE_SAMPLER, StructureFlow, y_up_voxels,
};
use super::topology::{LATENT_SCALE, TOPO_STEPS, TopologyDecoder, TopologyFlow, VoxelConditioner};
use super::vertex_flow::{VERTEX_SAMPLER, VertexFlow};
use super::vertex_vae::{THRESHOLD, VertexDecoder};
use super::*;
use crate::trellis2::structure::{STRUCTURE_DECODER, StructureDecoder};
use crate::worldmirror::parity::compare;

// the reference's own vertex count
const VERTICES: usize = 2000;

struct Setup {
    dir: PathBuf,
    large: PathBuf,
    dino: PathBuf,
    refs: HashMap<String, Tensor>,
    device: Device,
}

fn setup() -> Result<Option<Setup>> {
    let var = |k: &str| std::env::var_os(k).map(PathBuf::from);
    let (Some(dir), Some(large), Some(dino), Some(reference)) = (
        var("LATO2_DIR"),
        var("TRELLIS_LARGE_DIR"),
        var("DINOV2"),
        var("LATO2_REF"),
    ) else {
        return Ok(None);
    };
    Ok(Some(Setup {
        dir,
        large,
        dino,
        refs: inference_tensor::safetensors::load(&reference, &Device::Cpu)?,
        device: Device::new_cuda(0)?,
    }))
}

impl Setup {
    fn get(&self, k: &str) -> Result<Tensor> {
        self.refs[k].to_device(&self.device)
    }

    fn lato(&self, f: &str) -> Result<inference_tensor::nn::VarBuilder<'static>> {
        pt_weights(&self.dir.join(f), Some("state_dict"), &self.device)
    }

    fn cells(&self, k: &str) -> Result<Vec<[i32; 3]>> {
        Ok(self.refs[k]
            .to_dtype(DType::I64)?
            .to_vec2::<i64>()?
            .iter()
            .map(|r| [r[0] as i32, r[1] as i32, r[2] as i32])
            .collect())
    }

    fn both(&self) -> Result<Tensor> {
        let cond = self.get("cond")?;
        Tensor::cat(&[&cond, &cond.zeros_like()?], 0)
    }
}

fn report(name: &str, a: &Tensor, b: &Tensor, limit: f32) -> Result<()> {
    let (max, rel) = compare(a, b)?;
    println!("{name}: rel rms {rel:.3e}, max abs {max:.3e}");
    assert!(rel < limit, "{name}: rel rms {rel:.3e} past {limit:.0e}");
    Ok(())
}

#[test]
#[ignore = "needs the LATO.2, TRELLIS and DINOv2 checkpoints, a reference dump and a GPU"]
fn the_condition_matches_the_reference() -> Result<()> {
    let Some(s) = setup()? else { return Ok(()) };
    let rgba = s.refs["rgba"].to_vec3::<u8>()?;
    let (h, w) = (rgba.len(), rgba[0].len());
    let image = Rgba {
        width: w,
        height: h,
        pixels: rgba.into_iter().flatten().flatten().collect(),
    };
    let ours = condition_image(&image)?;
    let theirs: Vec<u8> = s.refs["cond_image"].flatten_all()?.to_vec1()?;
    let off = ours.iter().zip(&theirs).filter(|(a, b)| a != b).count();
    println!("conditioning image: {off} of {} bytes differ", ours.len());
    assert_eq!(off, 0);
    let dino = load_dinov2(&s.dino, &s.device)?;
    let cond = condition_tokens(&dino, &theirs, &s.device)?;
    report("dinov2 tokens", &cond, &s.get("cond")?, 1e-4)
}

#[test]
#[ignore = "needs the LATO.2, TRELLIS and DINOv2 checkpoints, a reference dump and a GPU"]
fn the_scaffold_matches_the_reference() -> Result<()> {
    let Some(s) = setup()? else { return Ok(()) };
    let flow = StructureFlow::load(
        DType::F32,
        safetensors(&s.large.join(STRUCTURE_FLOW_FILE), &s.device)?,
    )?;
    let kv = flow.condition(&s.both()?)?;
    let cells = LATENT_SIDE * LATENT_SIDE * LATENT_SIDE;
    let noise = s
        .get("ss_noise")?
        .reshape((LATENT_CHANNELS, cells))?
        .t()?
        .unsqueeze(0)?
        .contiguous()?;
    let latent = flow.sample(&noise, &kv, &STRUCTURE_SAMPLER, &mut |_| Ok(()))?;
    let theirs = s.get("ss_latent")?.reshape((LATENT_CHANNELS, cells))?.t()?;
    report("structure latent", &latent.squeeze(0)?, &theirs, 1e-3)?;
    drop((flow, kv));
    let decoder = StructureDecoder::load(
        STRUCTURE_DECODER,
        LATENT_SIDE,
        DType::F32,
        safetensors(&s.large.join(STRUCTURE_DECODER_FILE), &s.device)?,
    )?;
    let logits = decoder.forward(&theirs.contiguous()?)?;
    report(
        "occupancy logits",
        &logits,
        &s.get("ss_logits")?.flatten_all()?,
        1e-4,
    )?;
    let ours = y_up_voxels(&logits.to_vec1()?, 4 * LATENT_SIDE);
    assert_eq!(ours, s.cells("voxels")?);
    Ok(())
}

#[test]
#[ignore = "needs the LATO.2, TRELLIS and DINOv2 checkpoints, a reference dump and a GPU"]
fn the_vertex_latents_match_the_reference() -> Result<()> {
    let Some(s) = setup()? else { return Ok(()) };
    let flow = VertexFlow::load(DType::F32, s.lato("vflow.pt")?)?;
    let kv = flow.condition(&s.both()?)?;
    let cells = s.cells("voxels")?;
    let rope = flow.rope(&cells)?;
    let noise = s.get("vflow_noise")?.unsqueeze(0)?;
    // LATO2_VSTEP (`vstep.py`): the first step's velocities alone
    if let Some(step) = std::env::var_os("LATO2_VSTEP") {
        let step = inference_tensor::safetensors::load(step, &Device::Cpu)?;
        let density = (VERTICES as f64 / super::vertex_flow::MAX_VERTICES * 1000.0) as f32;
        let v = flow.forward(&noise.repeat((2, 1, 1))?, 1000.0, density, &rope, &kv)?;
        report("first velocity, cond", &v.get(0)?, &step["v_cond"], 1e-4)?;
        report(
            "first velocity, uncond",
            &v.get(1)?,
            &step["v_uncond"],
            1e-4,
        )?;
    }
    let latent = flow.sample(&noise, VERTICES, &rope, &kv, &VERTEX_SAMPLER, &mut |_| {
        Ok(())
    })?;
    // guidance scales each step's rounding by 1 + 2s, so 24 steps end ~1e-3 apart from a first step ~1e-5 apart
    report(
        "vertex latents",
        &latent.squeeze(0)?,
        &s.get("vflow_latent")?,
        5e-3,
    )
}

#[test]
#[ignore = "needs the LATO.2, TRELLIS and DINOv2 checkpoints, a reference dump and a GPU"]
fn the_vertices_match_the_reference() -> Result<()> {
    let Some(s) = setup()? else { return Ok(()) };
    let decoder = VertexDecoder::load(DType::F32, s.lato("vvae.pt")?, s.lato("offset_head.pt")?)?;
    let cells = s.cells("voxels")?;
    let mut levels = Vec::new();
    let vertices = decoder.decode(
        &s.get("vflow_latent")?,
        &cells,
        THRESHOLD,
        &mut |i, c, f| {
            levels.push((i, c.to_vec(), f.clone()));
        },
    )?;
    for (i, coords, feats) in &levels[1..] {
        let theirs = s.cells(&format!("vvae_coords_{i}"))?;
        println!(
            "level {i}: {} cells, reference {}",
            coords.len(),
            theirs.len()
        );
        assert_eq!(coords, &theirs, "level {i} keeps other cells");
        report(
            &format!("level {i} features"),
            feats,
            &s.get(&format!("vvae_feats_{i}"))?,
            1e-3,
        )?;
    }
    let offsets: Vec<f32> = vertices.offsets.iter().flatten().copied().collect();
    let n = vertices.offsets.len();
    report(
        "offsets",
        &Tensor::from_vec(offsets, (n, 3), &Device::Cpu)?,
        &s.refs["vert_offsets"],
        1e-3,
    )
}

#[test]
#[ignore = "needs the LATO.2, TRELLIS and DINOv2 checkpoints, a reference dump and a GPU"]
fn the_topology_matches_the_reference() -> Result<()> {
    let Some(s) = setup()? else { return Ok(()) };
    let scaffold = s.cells("voxels")?;
    let cond = VoxelConditioner::load(s.lato("voxel_encoder.pt")?)?.forward(&scaffold)?;
    report("voxel condition", &cond, &s.get("voxel_cond")?, 1e-4)?;
    let verts = s.cells("vert_int")?;
    let flow = TopologyFlow::load(DType::F32, s.lato("tflow.pt")?)?;
    let z = flow.sample(
        &s.get("tflow_noise")?,
        &verts,
        &s.get("voxel_cond")?,
        TOPO_STEPS,
        &mut |_| Ok(()),
    )?;
    report(
        "topology latents",
        &(z * LATENT_SCALE)?,
        &s.get("tflow_latent")?,
        1e-3,
    )?;
    drop(flow);
    let decoder = TopologyDecoder::load(DType::F32, s.lato("tvae.pt")?)?;
    let feats = decoder.features(&(s.get("tflow_latent")? / LATENT_SCALE)?, &verts)?;
    report("vertex features", &feats, &s.get("tvae_feats")?, 1e-3)?;
    let ours: BTreeSet<[u32; 2]> = decoder.edges(&feats, 0.0)?.into_iter().collect();
    let theirs: BTreeSet<[u32; 2]> = s.refs["edges"]
        .to_vec2::<i32>()?
        .iter()
        .map(|e| [e[0] as u32, e[1] as u32])
        .collect();
    let missing = theirs.difference(&ours).count();
    let extra = ours.difference(&theirs).count();
    println!(
        "edges: {} ours, {} reference, {missing} missing, {extra} extra",
        ours.len(),
        theirs.len()
    );
    assert!(missing + extra <= theirs.len() / 1000, "edges differ");
    Ok(())
}

#[test]
#[ignore = "needs a reference dump"]
fn the_faces_match_the_reference() -> Result<()> {
    let Some(reference) = std::env::var_os("LATO2_REF") else {
        return Ok(());
    };
    let refs = inference_tensor::safetensors::load(reference, &Device::Cpu)?;
    let edges: Vec<[u32; 2]> = refs["edges"]
        .to_vec2::<i32>()?
        .iter()
        .map(|e| [e[0] as u32, e[1] as u32])
        .collect();
    let cells: Vec<[f32; 3]> = refs["vert_int"]
        .to_vec2::<i32>()?
        .iter()
        .map(|c| [c[0] as f32, c[1] as f32, c[2] as f32])
        .collect();
    let set = |t: &Tensor| -> Result<BTreeSet<[u32; 3]>> {
        Ok(t.to_vec2::<i32>()?
            .iter()
            .map(|f| {
                let mut f = [f[0] as u32, f[1] as u32, f[2] as u32];
                f.sort_unstable();
                f
            })
            .collect())
    };
    let sorted = |faces: Vec<[u32; 3]>| -> BTreeSet<[u32; 3]> {
        faces
            .into_iter()
            .map(|mut f| {
                f.sort_unstable();
                f
            })
            .collect()
    };
    let raw = sorted(faces::faces(cells.len(), &edges, &cells, 0));
    assert_eq!(raw, set(&refs["faces_raw"])?);
    let filled = sorted(faces::faces(cells.len(), &edges, &cells, faces::MAX_RING));
    let theirs = set(&refs["faces"])?;
    println!(
        "filled faces: {} ours, {} reference, {} shared",
        filled.len(),
        theirs.len(),
        filled.intersection(&theirs).count()
    );
    Ok(())
}
