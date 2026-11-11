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

// distance difference allowed against cuBVH's
const DISTANCE_TOLERANCE: f32 = 1e-6;
// every this many texels sampled for the surface color comparison
const SAMPLE_STRIDE: usize = 7;
// mean color difference (of 255) allowed between our baked surface and the reference's
const MAX_MEAN_COLOR_DIFF: f32 = 3.0;
// relative difference allowed in chart count and seam length
const CHART_TOLERANCE: f64 = 1e-3;

// total length of the edges between charts or on the boundary
fn seam_length(mesh: &super::mesh::Mesh, charts: &[u32]) -> f64 {
    let mut edges: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
    for (f, t) in mesh.faces.iter().enumerate() {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            edges
                .entry((a.min(b), a.max(b)))
                .or_default()
                .push(f as u32);
        }
    }
    edges
        .iter()
        .filter(|(_, fs)| fs.len() != 2 || charts[fs[0] as usize] != charts[fs[1] as usize])
        .map(|(&(a, b), _)| {
            let (p, q) = (mesh.vertices[a as usize], mesh.vertices[b as usize]);
            (0..3)
                .map(|k| f64::from(p[k] - q[k]).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .sum()
}

// each face's chart named by the chart's lowest face, so one diverging merge does not renumber the rest
fn chart_roots(charts: &[u32]) -> Vec<u32> {
    let mut lowest = HashMap::new();
    for (f, &c) in charts.iter().enumerate() {
        lowest.entry(c).or_insert(f as u32);
    }
    charts.iter().map(|c| lowest[c]).collect()
}

// TRELLIS2_CHARTS_REF: CuMesh's `compute_charts` with `to_glb`'s options (`charts_ref.py`, scratch)
#[test]
#[ignore = "needs a reference dump"]
fn charts_match_the_reference() -> Result<()> {
    let Some(reference) = std::env::var_os("TRELLIS2_CHARTS_REF") else {
        return Ok(());
    };
    let refs = inference_tensor::safetensors::load(reference, &Device::Cpu)?;
    let mut mesh = mesh_of(&refs, "vertices", "faces")?;
    let faces = mesh.faces.len();
    super::charts::remove_degenerate_faces(&mut mesh);
    assert_eq!(
        mesh.faces.len(),
        faces,
        "the reference's faces are already non-degenerate"
    );
    let theirs: Vec<u32> = refs["charts"]
        .to_dtype(DType::I64)?
        .to_vec1::<i64>()?
        .iter()
        .map(|&c| c as u32)
        .collect();
    let start = std::time::Instant::now();
    let ours = super::charts::compute_charts(
        &mesh,
        super::charts::CONE_HALF_ANGLE,
        super::charts::AREA_PENALTY,
        super::charts::PERIMETER_AREA_PENALTY,
        &|| false,
    );
    let count = |c: &[u32]| c.iter().max().map_or(0, |&m| m as usize + 1);
    let (a, b) = (chart_roots(&ours), chart_roots(&theirs));
    let seconds = start.elapsed().as_secs_f64();
    let same = a.iter().zip(&b).filter(|(x, y)| x == y).count();
    // CuMesh normalizes with rsqrtf and sums in tree order, so near ties flip: compare the clustering's quality
    let seams = |c: &[u32]| seam_length(&mesh, c);
    let sizes = |c: &[u32]| {
        let mut n = vec![0usize; count(c)];
        for &x in c {
            n[x as usize] += 1;
        }
        n.sort_unstable();
        [0.5, 0.9, 0.99, 1.0].map(|q| n[((n.len() - 1) as f64 * q) as usize])
    };
    let (ours_seam, theirs_seam) = (seams(&ours), seams(&theirs));
    println!(
        "charts in {seconds:.2} s: {} vs {}, {same}/{faces} faces in the same chart; seams {ours_seam:.3} vs \
         {theirs_seam:.3}; faces per chart p50/p90/p99/max {:?} vs {:?}",
        count(&ours),
        count(&theirs),
        sizes(&ours),
        sizes(&theirs)
    );
    let ratio = |a: f64, b: f64| (a - b).abs() / b;
    assert!(
        ratio(count(&ours) as f64, count(&theirs) as f64) < CHART_TOLERANCE,
        "chart count"
    );
    assert!(
        ratio(ours_seam, theirs_seam) < CHART_TOLERANCE,
        "seam length"
    );
    Ok(())
}

// the uv area the faces cover, and how many have none
fn uv_coverage(faces: &[[u32; 3]], uvs: &[[f32; 2]]) -> (f64, usize) {
    let areas: Vec<f64> = faces
        .iter()
        .map(|t| {
            let [a, b, c] = t.map(|v| uvs[v as usize].map(f64::from));
            ((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])).abs() / 2.0
        })
        .collect();
    (
        areas.iter().sum(),
        areas.iter().filter(|&&a| a == 0.0).count(),
    )
}

// TRELLIS2_CHARTS_REF's mesh unwrapped, against CuMesh's `uv_unwrap` (TRELLIS2_UNWRAP_REF, `unwrap_ref.py`, scratch)
#[test]
#[ignore = "needs reference dumps"]
fn the_unwrap_matches_the_reference() -> Result<()> {
    let (Some(input), Some(reference)) = (
        std::env::var_os("TRELLIS2_CHARTS_REF"),
        std::env::var_os("TRELLIS2_UNWRAP_REF"),
    ) else {
        return Ok(());
    };
    let mesh = mesh_of(
        &inference_tensor::safetensors::load(input, &Device::Cpu)?,
        "vertices",
        "faces",
    )?;
    let refs = inference_tensor::safetensors::load(reference, &Device::Cpu)?;
    let theirs = mesh_of(&refs, "vertices", "faces")?;
    let their_uvs: Vec<[f32; 2]> = refs["uvs"]
        .to_vec2::<f32>()?
        .iter()
        .map(|r| [r[0], r[1]])
        .collect();
    let start = std::time::Instant::now();
    let ours = super::charts::uv_unwrap_reference(&mesh)
        .map_err(|e| inference_tensor::Error::Msg(e.to_string()))?;
    let seconds = start.elapsed().as_secs_f64();
    let (ours_area, ours_empty) = uv_coverage(&ours.mesh.faces, &ours.uvs);
    let (theirs_area, theirs_empty) = uv_coverage(&theirs.faces, &their_uvs);
    println!(
        "unwrapped in {seconds:.2} s into {}x{}: {} vertices / {} faces vs {} / {}; uv coverage {ours_area:.4} vs \
         {theirs_area:.4}; faces without uv area {ours_empty} vs {theirs_empty}",
        ours.width,
        ours.height,
        ours.mesh.vertices.len(),
        ours.mesh.faces.len(),
        theirs.vertices.len(),
        theirs.faces.len()
    );
    assert_eq!(
        ours.mesh.faces.len(),
        theirs.faces.len(),
        "every face mapped"
    );
    let ratio = |a: f64, b: f64| (a - b).abs() / b;
    assert!(
        ratio(
            ours.mesh.vertices.len() as f64,
            theirs.vertices.len() as f64
        ) < CHART_TOLERANCE * 10.0
    );
    assert!(
        ratio(ours_area, theirs_area) < CHART_TOLERANCE * 10.0,
        "uv coverage"
    );
    Ok(())
}

// manifold edges both of whose faces run it the same way
fn inconsistent_edges(mesh: &super::mesh::Mesh) -> usize {
    let at = |t: [u32; 3], v: u32| {
        t.iter()
            .position(|&x| x == v)
            .expect("a vertex of the face")
    };
    super::charts::manifold_adjacency(mesh)
        .iter()
        .filter(|&&(f0, f1, v0, v1)| {
            let dir = |t: [u32; 3]| (at(t, v1) + 3 - at(t, v0)) % 3;
            dir(mesh.faces[f0 as usize]) == dir(mesh.faces[f1 as usize])
        })
        .count()
}

type MeshPass = fn(&mut super::mesh::Mesh);

// TRELLIS2_TOPOLOGY_REF: CuMesh's duplicate, non-manifold, component and orientation passes, each from the
// reference's previous stage (`topology_ref.py`, scratch)
#[test]
#[ignore = "needs a reference dump"]
fn topology_passes_match_the_reference() -> Result<()> {
    let Some(reference) = std::env::var_os("TRELLIS2_TOPOLOGY_REF") else {
        return Ok(());
    };
    let refs = inference_tensor::safetensors::load(reference, &Device::Cpu)?;
    let stage = |name: &str| mesh_of(&refs, &format!("{name}.v"), &format!("{name}.f"));
    let passes: [(&str, &str, MeshPass); 4] = [
        ("input", "dedup", super::cleanup::remove_duplicate_faces),
        ("dedup", "repair", super::cleanup::repair_non_manifold_edges),
        ("repair", "small", |m| {
            super::cleanup::remove_small_connected_components(m, super::cleanup::MIN_COMPONENT_AREA)
        }),
        ("small", "unify", super::cleanup::unify_face_orientations),
    ];
    for (from, to, pass) in passes {
        let mut ours = stage(from)?;
        let start = std::time::Instant::now();
        pass(&mut ours);
        let theirs = stage(to)?;
        let same = ours
            .faces
            .iter()
            .zip(&theirs.faces)
            .filter(|(a, b)| a == b)
            .count();
        println!(
            "{to} in {:.2} s: {} vertices / {} faces vs {} / {}, {same} faces identical, vertices identical: {}",
            start.elapsed().as_secs_f64(),
            ours.vertices.len(),
            ours.faces.len(),
            theirs.vertices.len(),
            theirs.faces.len(),
            ours.vertices == theirs.vertices
        );
        if to == "unify" {
            // pieces with conflicting loops wind by union order, and CuMesh's concurrent hooks race there
            let flips = ours
                .faces
                .iter()
                .zip(&theirs.faces)
                .filter(|(a, b)| a != b && [a[0], a[2], a[1]] == **b)
                .count();
            let (a, b) = (inconsistent_edges(&ours), inconsistent_edges(&theirs));
            println!(
                "unify: {flips} differing faces are flips; inconsistently wound manifold edges {a} vs {b}"
            );
            assert_eq!(flips + same, ours.faces.len(), "unify only flips");
            assert!(a <= b, "no more inconsistent edges than the reference");
            continue;
        }
        assert_eq!(ours.faces, theirs.faces, "{to} faces");
        assert_eq!(ours.vertices, theirs.vertices, "{to} vertices");
    }
    Ok(())
}

// barycentric weights of a point on the triangle
fn weights(q: [f32; 3], [a, b, c]: [[f32; 3]; 3]) -> [f32; 3] {
    let d = |x: [f32; 3], y: [f32; 3]| x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
    let s = |x: [f32; 3], y: [f32; 3]| [x[0] - y[0], x[1] - y[1], x[2] - y[2]];
    let (v0, v1, v2) = (s(b, a), s(c, a), s(q, a));
    let (d00, d01, d11, d20, d21) = (d(v0, v0), d(v0, v1), d(v1, v1), d(v2, v0), d(v2, v1));
    let den = d00 * d11 - d01 * d01;
    let v = (d11 * d20 - d01 * d21) / den;
    let w = (d00 * d21 - d01 * d20) / den;
    [1.0 - v - w, v, w]
}

// TRELLIS2_FULL_REF's mesh and texture volume through o-voxel's `to_glb` (TRELLIS2_BAKE_REF, `bake_ref.py`, scratch)
#[test]
#[ignore = "needs reference dumps"]
fn the_bake_matches_the_reference() -> Result<()> {
    let (Some(input), Some(reference)) = (
        std::env::var_os("TRELLIS2_FULL_REF"),
        std::env::var_os("TRELLIS2_BAKE_REF"),
    ) else {
        return Ok(());
    };
    let refs = inference_tensor::safetensors::load(reference, &Device::Cpu)?;
    let bytes = |k: &str| -> Result<Vec<u8>> { refs[k].flatten_all()?.to_vec1::<u8>() };
    let unknown: Vec<bool> = bytes("unknown")?.iter().map(|&m| m != 0).collect();
    let size = refs["unknown"].dim(0)?;
    // OpenCV's Telea on the reference's own unfilled textures
    for (name, channels, radius) in [
        ("color", 3, 3.0),
        ("metallic", 1, 1.0),
        ("roughness", 1, 1.0),
        ("alpha", 1, 1.0),
    ] {
        let mut ours = bytes(&format!("{name}.in"))?;
        let start = std::time::Instant::now();
        inference_vision::inpaint_telea(&mut ours, size, size, channels, &unknown, radius);
        let theirs = bytes(&format!("{name}.out"))?;
        let differ = ours.iter().zip(&theirs).filter(|(a, b)| a != b).count();
        println!(
            "{name} inpainted in {:.2} s: {differ} of {} values differ",
            start.elapsed().as_secs_f64(),
            ours.len()
        );
        assert_eq!(differ, 0, "{name} inpainting");
    }
    let full = inference_tensor::safetensors::load(input, &Device::Cpu)?;
    let mesh = mesh_of(&full, "mesh.v", "mesh.f")?;
    let coords = coords(&full["tex.coords"])?;
    let values: Vec<[f32; 6]> = full["tex.attrs"]
        .to_vec2::<f32>()?
        .iter()
        .map(|r| std::array::from_fn(|k| r[k]))
        .collect();
    let resolution = full["resolution"].to_dtype(DType::I64)?.to_vec1::<i64>()?[0] as usize;
    let start = std::time::Instant::now();

    let baked = super::bake::bake(
        &mesh,
        &coords,
        &values,
        resolution,
        &Default::default(),
        &Device::Cpu,
    )
    .map_err(|e| inference_tensor::Error::Msg(e.to_string()))?;
    let theirs = mesh_of(&refs, "final.v", "final.f")?;
    let their_uvs: Vec<[f32; 2]> = refs["final.uv"]
        .to_vec2::<f32>()?
        .iter()
        .map(|r| [r[0], r[1]])
        .collect();
    println!(
        "baked in {:.1} s: {} vertices / {} faces vs {} / {}, {} texels \
         covered vs {}",
        start.elapsed().as_secs_f64(),
        baked.unwrapped.mesh.vertices.len(),
        baked.unwrapped.mesh.faces.len(),
        theirs.vertices.len(),
        theirs.faces.len(),
        baked.covered,
        unknown.iter().filter(|&&u| !u).count()
    );
    // the reference's covered texels placed on its surface, looked up through our atlas
    let their_texels = super::bake::rasterize(&theirs.faces, &their_uvs, size);
    let bvh = super::bake::Bvh::new(&baked.unwrapped.mesh);
    let their_color = bytes("color.in")?;
    let s = size as f64;
    let diffs: Vec<f32> = (0..size * size)
        .step_by(SAMPLE_STRIDE)
        .filter(|&t| their_texels[t] > 0 && !unknown[t])
        .map(|t| {
            let f = theirs.faces[their_texels[t] as usize - 1];
            let uv = f.map(|v| their_uvs[v as usize].map(|c| f64::from(c) * s));
            let p = [(t % size) as f64 + 0.5, (t / size) as f64 + 0.5];
            let area = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| {
                (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            };
            let total = area(uv[0], uv[1], uv[2]);
            let w = [
                area(uv[1], uv[2], p) / total,
                area(uv[2], uv[0], p) / total,
                area(uv[0], uv[1], p) / total,
            ];
            let pos = (0..3).fold([0f32; 3], |acc, k| {
                let v = theirs.vertices[f[k] as usize];
                [0, 1, 2].map(|a| acc[a] + v[a] * w[k] as f32)
            });
            let (face, q) = bvh.closest_face(pos);
            let ours_f = baked.unwrapped.mesh.faces[face as usize];
            let bw = weights(q, ours_f.map(|v| baked.unwrapped.mesh.vertices[v as usize]));
            let uv = (0..3).fold([0f32; 2], |acc, k| {
                let u = baked.unwrapped.uvs[ours_f[k] as usize];
                [acc[0] + u[0] * bw[k], acc[1] + u[1] * bw[k]]
            });
            let texel = |c: f32| ((c * size as f32) as usize).min(size - 1);
            let ours_t = texel(uv[1]) * size + texel(uv[0]);
            (0..3)
                .map(|c| {
                    (f32::from(baked.base_color[4 * ours_t + c])
                        - f32::from(their_color[3 * t + c]))
                    .abs()
                })
                .sum::<f32>()
                / 3.0
        })
        .collect();
    let mut sorted = diffs.clone();
    sorted.sort_by(f32::total_cmp);
    let mean = diffs.iter().sum::<f32>() / diffs.len() as f32;
    let at = |q: f64| sorted[((sorted.len() - 1) as f64 * q) as usize];
    println!(
        "surface color over {} texels: mean abs diff {mean:.2} levels, p50 {:.1} p90 {:.1} p99 {:.1}",
        diffs.len(),
        at(0.5),
        at(0.9),
        at(0.99)
    );
    assert!(mean < MAX_MEAN_COLOR_DIFF, "surface color");
    Ok(())
}

// TRELLIS2_REMESH_REF: cuBVH distances at random points and CuMesh's remesh as `to_glb` runs it on the hole-filled
// 1024 mesh (`remesh_ref.py`, scratch)
#[test]
#[ignore = "needs a reference dump"]
fn distances_and_the_remesh_match_the_reference() -> Result<()> {
    let Some(reference) = std::env::var_os("TRELLIS2_REMESH_REF") else {
        return Ok(());
    };
    let refs = inference_tensor::safetensors::load(reference, &Device::Cpu)?;
    let filled = mesh_of(&refs, "filled.v", "filled.f")?;
    let start = std::time::Instant::now();
    let bvh = super::bake::Bvh::new(&filled);
    let built = start.elapsed().as_secs_f64();
    let points: Vec<[f32; 3]> = refs["pts"]
        .to_vec2::<f32>()?
        .iter()
        .map(|r| [r[0], r[1], r[2]])
        .collect();
    let theirs = refs["dist"].to_vec1::<f32>()?;
    let start = std::time::Instant::now();
    let ours = bvh.unsigned_distance(&points);
    let queried = start.elapsed().as_secs_f64();
    let worst = ours
        .iter()
        .zip(&theirs)
        .map(|(o, t)| (o.0 - t).abs())
        .fold(0f32, f32::max);
    println!(
        "bvh over {} faces in {built:.2} s; {} distances in {queried:.2} s, worst difference {worst:.2e}",
        filled.faces.len(),
        points.len()
    );
    assert!(worst < DISTANCE_TOLERANCE, "distances");
    let domain = super::remesh::Domain::unit_cube(1024);
    let start = std::time::Instant::now();
    let host = super::bake::Distances::new(&bvh, &Device::Cpu)?;
    let shell = super::remesh::remesh(
        &host,
        &domain,
        super::remesh::BAND,
        super::remesh::PROJECT_BACK,
    )?;
    let seconds = start.elapsed().as_secs_f64();
    let reference = mesh_of(&refs, "remesh.v", "remesh.f")?;
    let cell = 1.0 / 1024.0;
    let (ab_mean, ab_99) = nearest(&shell.vertices, &reference.vertices, cell);
    let same = shell.vertices.len() == reference.vertices.len()
        && shell
            .vertices
            .iter()
            .zip(&reference.vertices)
            .all(|(a, b)| (0..3).all(|k| (a[k] - b[k]).abs() < 1e-5));
    println!(
        "remeshed in {seconds:.2} s: {} vertices / {} faces vs {} / {}; nearest ours->ref mean {ab_mean:.2e} p99 \
         {ab_99:.2e}; vertex for vertex: {same}, faces identical: {}",
        shell.vertices.len(),
        shell.faces.len(),
        reference.vertices.len(),
        reference.faces.len(),
        shell.faces == reference.faces
    );
    let ratio = (shell.vertices.len() as f64 - reference.vertices.len() as f64).abs()
        / reference.vertices.len() as f64;
    assert!(ratio < 1e-5, "vertex count");
    assert!(ab_99 < DISTANCE_TOLERANCE, "vertex positions");
    // the same on a device, when there is one
    let device = Device::cuda_if_available(0)?;
    if !device.is_cpu() {
        let on_device = super::bake::Distances::new(&bvh, &device)?;
        let start = std::time::Instant::now();
        let theirs = on_device.unsigned_distance(&points)?;
        let queried = start.elapsed().as_secs_f64();
        let start = std::time::Instant::now();
        let device_shell = super::remesh::remesh(
            &on_device,
            &domain,
            super::remesh::BAND,
            super::remesh::PROJECT_BACK,
        )?;
        println!(
            "on {device:?}: distances in {queried:.2} s, remeshed in {:.2} s; same distances: {}, same shell: {}",
            start.elapsed().as_secs_f64(),
            theirs == ours,
            mesh_hash(&device_shell) == mesh_hash(&shell)
        );
        assert!(theirs == ours, "device distances");
        assert_eq!(mesh_hash(&device_shell), mesh_hash(&shell), "device shell");
    }
    Ok(())
}

// a mesh's bits, to tell a change that keeps simplification's output from one that moves it
fn mesh_hash(mesh: &super::mesh::Mesh) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for v in &mesh.vertices {
        v.map(f32::to_bits).hash(&mut h);
    }
    mesh.faces.hash(&mut h);
    h.finish()
}

// TRELLIS2_REMESH_REF's remeshed shell (8.3M faces) simplified to the export's budget, timed and hashed
#[test]
#[ignore = "needs a reference dump"]
fn simplifying_the_remesh_is_timed() -> Result<()> {
    let Some(reference) = std::env::var_os("TRELLIS2_REMESH_REF") else {
        return Ok(());
    };
    let refs = inference_tensor::safetensors::load(reference, &Device::Cpu)?;
    let input = mesh_of(&refs, "remesh.v", "remesh.f")?;
    let mut hashes = Vec::new();
    for device in [Device::Cpu, Device::cuda_if_available(0)?] {
        let mut mesh = input.clone();
        let start = std::time::Instant::now();
        super::cleanup::simplify_on(&mut mesh, super::bake::DECIMATION_TARGET, &device)?;
        println!(
            "simplified on {device:?} in {:.2} s to {} vertices / {} faces, hash {:x}",
            start.elapsed().as_secs_f64(),
            mesh.vertices.len(),
            mesh.faces.len(),
            mesh_hash(&mesh)
        );
        hashes.push(mesh_hash(&mesh));
    }
    assert_eq!(hashes[0], hashes[1], "the device's result is the host's");
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

fn tensor_hash(t: &Tensor) -> Result<u64> {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    t.to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .for_each(|v| v.to_bits().hash(&mut h));
    Ok(h.finish())
}

// TRELLIS2_DIR and TRELLIS2_FULL_REF: the 1024 shape and texture latents decoded (after a warm-up), timed and hashed
#[test]
#[ignore = "needs the TRELLIS.2 checkpoint, a reference dump and a GPU"]
fn decoding_is_timed() -> Result<()> {
    use super::decoder::TEXTURE_DECODER;
    let (Some(dir), Some(reference)) = (
        std::env::var_os("TRELLIS2_DIR").map(PathBuf::from),
        std::env::var_os("TRELLIS2_FULL_REF"),
    ) else {
        return Ok(());
    };
    let device = Device::new_cuda(0)?;
    let refs = inference_tensor::safetensors::load(PathBuf::from(reference), &Device::Cpu)?;
    let cells = coords(&refs["s2.coords"])?;
    let shape = ShapeDecoder::load(
        SHAPE_DECODER,
        DType::F16,
        vb(dir.join(DECODER_FILE), &device)?,
    )?;
    let texture = ShapeDecoder::load(
        TEXTURE_DECODER,
        DType::F16,
        vb(
            dir.join("ckpts/tex_dec_next_dc_f16c32_fp16.safetensors"),
            &device,
        )?,
    )?;
    let shape_latents = refs["s2.out"].to_device(&device)?;
    let tex_latents = refs["s3.out"].to_device(&device)?;
    for run in 0..2 {
        device.synchronize()?;
        let start = std::time::Instant::now();
        let (shape_out, subs) = shape.forward(&cells, &shape_latents, None, &mut |_, _| {})?;
        device.synchronize()?;
        let shaped = start.elapsed().as_secs_f64();
        let (tex_out, _) = texture.forward(&cells, &tex_latents, Some(&subs), &mut |_, _| {})?;
        device.synchronize()?;
        println!(
            "run {run}: shape {shaped:.2} s, texture {:.2} s, {} cells, hashes {:x} {:x}",
            start.elapsed().as_secs_f64() - shaped,
            shape_out.coords.len(),
            tensor_hash(&shape_out.feats)?,
            tensor_hash(&tex_out.feats)?
        );
    }
    Ok(())
}

// an unwrap's measures: uv islands, 3D length of the cut mesh's boundary (seams plus the mesh's own boundary),
// area-weighted L2 stretch (Sander et al.) normalized so a uniform isometric atlas scores 1, and its worst 1% of area
fn unwrap_measures(u: &super::charts::Unwrapped) -> (usize, f64, f64, f64) {
    let (v, f) = (&u.mesh.vertices, &u.mesh.faces);
    let mut parent: Vec<u32> = (0..v.len() as u32).collect();
    fn find(p: &mut [u32], mut x: u32) -> u32 {
        while p[x as usize] != x {
            p[x as usize] = p[p[x as usize] as usize];
            x = p[x as usize];
        }
        x
    }
    let mut edges: HashMap<(u32, u32), u32> = HashMap::new();
    for t in f {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
            parent[ra as usize] = rb;
            *edges.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    let islands: HashSet<u32> = f.iter().map(|t| find(&mut parent, t[0])).collect();
    let len = |a: u32, b: u32| {
        let (p, q) = (v[a as usize], v[b as usize]);
        f64::from(((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt())
    };
    let boundary: f64 = edges
        .iter()
        .filter(|&(_, &n)| n == 1)
        .map(|(&(a, b), _)| len(a, b))
        .sum();
    let mut faces: Vec<(f64, f64)> = Vec::new();
    let (mut area3, mut area2) = (0f64, 0f64);
    for t in f {
        let q = t.map(|i| v[i as usize].map(f64::from));
        let p = t.map(|i| u.uvs[i as usize].map(f64::from));
        let a2 = ((p[1][0] - p[0][0]) * (p[2][1] - p[0][1])
            - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1]))
            / 2.0;
        let e1 = [0, 1, 2].map(|k| q[1][k] - q[0][k]);
        let e2 = [0, 1, 2].map(|k| q[2][k] - q[0][k]);
        let cr = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        let a3 = (cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]).sqrt() / 2.0;
        if a3 <= 0.0 {
            continue;
        }
        area3 += a3;
        area2 += a2.abs();
        if a2.abs() < 1e-30 {
            faces.push((a3, f64::INFINITY));
            continue;
        }
        let (s, t2) = (p.map(|x| x[0]), p.map(|x| x[1]));
        let ss: [f64; 3] = std::array::from_fn(|k| {
            (q[0][k] * (t2[1] - t2[2]) + q[1][k] * (t2[2] - t2[0]) + q[2][k] * (t2[0] - t2[1]))
                / (2.0 * a2)
        });
        let st: [f64; 3] = std::array::from_fn(|k| {
            (q[0][k] * (s[2] - s[1]) + q[1][k] * (s[0] - s[2]) + q[2][k] * (s[1] - s[0]))
                / (2.0 * a2)
        });
        let l2 = ((ss.iter().map(|x| x * x).sum::<f64>() + st.iter().map(|x| x * x).sum::<f64>())
            / 2.0)
            .sqrt();
        faces.push((a3, l2));
    }
    let scale = (area2 / area3).sqrt();
    let finite: Vec<(f64, f64)> = faces
        .iter()
        .map(|&(a, l)| (a, l * scale))
        .filter(|&(_, l)| l.is_finite())
        .collect();
    let mean = (finite.iter().map(|&(a, l)| a * l * l).sum::<f64>()
        / finite.iter().map(|&(a, _)| a).sum::<f64>())
    .sqrt();
    let mut sorted = finite.clone();
    sorted.sort_by(|a, b| b.1.total_cmp(&a.1));
    let (mut acc, mut worst) = (0.0, 0.0);
    for &(a, l) in &sorted {
        acc += a;
        worst = l;
        if acc >= 0.01 * area3 {
            break;
        }
    }
    (islands.len(), boundary, mean, worst)
}

// TRELLIS2_CHARTS_REF's mesh unwrapped as CuMesh does and with each cluster taken as one chart (the default at 30
// degrees), compared
#[test]
#[ignore = "needs a reference dump"]
fn single_chart_unwrap_is_compared() -> Result<()> {
    let Some(input) = std::env::var_os("TRELLIS2_CHARTS_REF") else {
        return Ok(());
    };
    let mesh = mesh_of(
        &inference_tensor::safetensors::load(input, &Device::Cpu)?,
        "vertices",
        "faces",
    )?;
    let full = super::charts::CONE_HALF_ANGLE;
    let modes: Vec<(f32, bool)> = match std::env::var("TRELLIS2_UNWRAP_MODES").as_deref() {
        Ok("default") => vec![(90.0, false)],
        _ => vec![
            (90.0, false),
            (90.0, true),
            (60.0, true),
            (45.0, true),
            (30.0, true),
            (20.0, true),
        ],
    };
    for (degrees, single_chart) in modes {
        let cone = full * degrees / 90.0;
        let start = std::time::Instant::now();
        let u = super::charts::uv_unwrap_with(
            &mesh,
            cone,
            inference_xatlas::Options { single_chart },
            &|| false,
        )
        .map_err(|e| inference_tensor::Error::Msg(e.to_string()))?;
        let seconds = start.elapsed().as_secs_f64();
        let (coverage, empty) = uv_coverage(&u.mesh.faces, &u.uvs);
        let (islands, boundary, stretch, worst) = unwrap_measures(&u);
        println!(
            "cone {degrees} single chart {single_chart}: {seconds:.2} s, {}x{}, {} vertices, {islands} islands, boundary {boundary:.2}, \
             coverage {coverage:.4}, {empty} faces without area, L2 stretch {stretch:.4} (worst 1% {worst:.3}), hash {:x}",
            u.width,
            u.height,
            u.mesh.vertices.len(),
            {
                use std::hash::{Hash, Hasher};
                let mut h = std::collections::hash_map::DefaultHasher::new();
                u.uvs.iter().for_each(|p| p.map(f32::to_bits).hash(&mut h));
                u.source.hash(&mut h);
                u.mesh.faces.hash(&mut h);
                h.finish()
            }
        );
    }
    Ok(())
}

// TRELLIS2_TEX_REF: `tex_ref.py mesh image out.safetensors` (scratch), the texturing pipeline's stages on a mesh
fn texturing_reference() -> Option<std::collections::HashMap<String, Tensor>> {
    let path = std::env::var_os("TRELLIS2_TEX_REF")?;
    inference_tensor::safetensors::load(path, &Device::Cpu).ok()
}

fn reference_mesh(refs: &std::collections::HashMap<String, Tensor>) -> Result<super::mesh::Mesh> {
    let vertices = refs["mesh.v"]
        .to_vec2::<f32>()?
        .iter()
        .map(|r| [r[0], r[1], r[2]])
        .collect();
    let faces = refs["mesh.f"]
        .to_dtype(DType::I64)?
        .to_vec2::<i64>()?
        .iter()
        .map(|r| [r[0] as u32, r[1] as u32, r[2] as u32])
        .collect();
    Ok(super::mesh::Mesh { vertices, faces })
}

#[test]
#[ignore = "needs a texturing reference dump"]
fn the_dual_grid_matches_o_voxel() -> Result<()> {
    let Some(refs) = texturing_reference() else {
        return Ok(());
    };
    let mesh = reference_mesh(&refs)?;
    let start = std::time::Instant::now();
    let ours = super::voxelize::flexible_dual_grid(&mesh, 1024);
    println!(
        "{} cells in {:.2} s",
        ours.cells.len(),
        start.elapsed().as_secs_f64()
    );
    let cells = coords(&refs["fdg.voxels"])?;
    let dual = refs["fdg.dual"].to_vec2::<f32>()?;
    let crossed = refs["fdg.intersected"]
        .to_dtype(DType::U32)?
        .to_vec2::<u32>()?;
    let theirs: HashMap<[i32; 3], usize> = cells.iter().enumerate().map(|(i, &c)| (c, i)).collect();
    assert_eq!(ours.cells.len(), cells.len(), "cell counts");
    let (mut worst, mut sum, mut flags) = (0f32, 0f64, 0usize);
    for (k, c) in ours.cells.iter().enumerate() {
        let i = *theirs
            .get(c)
            .unwrap_or_else(|| panic!("cell {c:?} not in the reference"));
        for a in 0..3 {
            let d = (ours.offsets[k][a] - (dual[i][a] * 1024.0 - c[a] as f32)).abs();
            worst = worst.max(d);
            sum += f64::from(d);
            flags += usize::from(ours.crossed[k][a] != (crossed[i][a] != 0));
        }
    }
    let mut over = [0usize; 4];
    for (k, c) in ours.cells.iter().enumerate() {
        let i = theirs[c];
        let d = (0..3)
            .map(|a| (ours.offsets[k][a] - (dual[i][a] * 1024.0 - c[a] as f32)).abs())
            .fold(0f32, f32::max);
        for (t, n) in [1e-4, 1e-3, 1e-2, 1e-1].iter().zip(&mut over) {
            *n += usize::from(d > *t);
        }
    }
    println!("cells off by more than 1e-4 / 1e-3 / 1e-2 / 1e-1: {over:?}");
    println!(
        "offsets: worst {worst:.2e}, mean {:.2e} (cell units); crossing flags differing {flags}",
        sum / (3 * cells.len()) as f64
    );
    assert_eq!(flags, 0);
    Ok(())
}

// the reference's dual grid, as the encoder reads it
fn reference_grid(
    refs: &std::collections::HashMap<String, Tensor>,
) -> Result<super::voxelize::DualGrid> {
    let cells = coords(&refs["fdg.voxels"])?;
    let dual = refs["fdg.dual"].to_vec2::<f32>()?;
    let crossed = refs["fdg.intersected"]
        .to_dtype(DType::U32)?
        .to_vec2::<u32>()?;
    Ok(super::voxelize::DualGrid {
        offsets: cells
            .iter()
            .zip(&dual)
            .map(|(c, d)| std::array::from_fn(|a| d[a] * 1024.0 - c[a] as f32))
            .collect(),
        crossed: crossed
            .iter()
            .map(|r| [r[0] != 0, r[1] != 0, r[2] != 0])
            .collect(),
        cells,
    })
}

// rows of `ours` on `our_cells` against `theirs` on `their_cells`, matched by cell
fn compare_by_cell(
    name: &str,
    our_cells: &[[i32; 3]],
    ours: &Tensor,
    their_cells: &[[i32; 3]],
    theirs: &Tensor,
) -> Result<()> {
    let at: HashMap<[i32; 3], usize> = their_cells
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, i))
        .collect();
    assert_eq!(our_cells.len(), their_cells.len(), "{name}: cell counts");
    let index: Vec<u32> = our_cells.iter().map(|c| at[c] as u32).collect();
    let n = index.len();
    let theirs = theirs.index_select(&Tensor::from_vec(index, n, &Device::Cpu)?, 0)?;
    let (max, rms) = compare(ours, &theirs)?;
    println!("{name}: max |diff| {max:.3e}, relative RMS {rms:.3e}");
    Ok(())
}

#[test]
#[ignore = "needs the TRELLIS.2 checkpoints, a texturing reference dump and a GPU"]
fn the_shape_encoder_and_texture_decoder_match_the_reference() -> Result<()> {
    let (Some(refs), Some((dir, _))) = (texturing_reference(), paths_dir()) else {
        return Ok(());
    };
    let device = Device::new_cuda(0)?;
    let grid = reference_grid(&refs)?;
    let hierarchy = super::encoder::Hierarchy::new(&grid.cells, &device)?;
    let encoder = super::encoder::ShapeEncoder::load(
        super::encoder::SHAPE_ENCODER,
        DType::F16,
        vb(
            dir.join("ckpts/shape_enc_next_dc_f16c32_fp16.safetensors"),
            &device,
        )?,
    )?;
    let latents = encoder.forward(&grid, &hierarchy, &device)?;
    compare_by_cell(
        "encoder latents",
        hierarchy.coarsest(),
        &latents,
        &coords(&refs["enc.coords"])?,
        &refs["enc.feats"],
    )?;
    // the texture decoder on the reference's texture latents, unfolding onto the encoder's levels
    let tex_cells = coords(&refs["enc.coords"])?;
    let at: HashMap<[i32; 3], usize> = tex_cells.iter().enumerate().map(|(i, &c)| (c, i)).collect();
    let index: Vec<u32> = hierarchy.coarsest().iter().map(|c| at[c] as u32).collect();
    let n = index.len();
    // the sampler's output is normalized; the decoder reads it denormalized
    let pipeline: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("texturing_pipeline.json"))?)
            .map_err(inference_tensor::Error::wrap)?;
    let stat = |k: &str| -> Vec<f32> {
        pipeline["args"]["tex_slat_normalization"][k]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_f64())
                    .map(|v| v as f32)
                    .collect()
            })
            .unwrap_or_default()
    };
    let tex = super::shape_flow::denormalize(
        &refs["tex.out"]
            .index_select(&Tensor::from_vec(index, n, &Device::Cpu)?, 0)?
            .unsqueeze(0)?,
        &stat("mean"),
        &stat("std"),
    )?
    .squeeze(0)?
    .to_device(&device)?;
    let decoder = ShapeDecoder::load(
        super::decoder::TEXTURE_DECODER,
        DType::F16,
        vb(
            dir.join("ckpts/tex_dec_next_dc_f16c32_fp16.safetensors"),
            &device,
        )?,
    )?;
    let (out, _) = decoder.forward(
        hierarchy.coarsest(),
        &tex,
        Some(&hierarchy.subdivisions),
        &mut |_, _| {},
    )?;
    let attrs = ((out.feats * 0.5)? + 0.5)?;
    compare_by_cell(
        "texture decoder",
        &out.coords,
        &attrs,
        &coords(&refs["dec.coords"])?,
        &refs["dec.attrs"],
    )
}

fn paths_dir() -> Option<(PathBuf, ())> {
    Some((std::env::var_os("TRELLIS2_DIR")?.into(), ()))
}
