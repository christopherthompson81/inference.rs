//! `Sparc3DPipeline`: the condition, then one pass per rung of the ladder up to the asked resolution, each sampling
//! latents on the previous surface's cells and decoding them to a finer surface; the last surface cleaned of debris,
//! simplified and normalized.

use std::path::{Path, PathBuf};

use image::DynamicImage;
use inference_tensor::nn::VarBuilder;
use inference_tensor::{DType, Device, Result, Tensor};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};

use super::conditioner::{Conditioner, prepare};
use super::extract::{fill_holes, point_to_mesh};
use super::schedule;
use super::transformer::{Grid, LATENT_CHANNELS, Transformer};
use super::vae::Decoder;
use super::voxelize;
use super::{LATENT_MEAN, LATENT_STD};
use crate::progress::{Progress, check};
use crate::trellis2::mesh::Mesh;

/// Each rung: its structure grid, the surface resolution it decodes to, the cell budget voxelization targets and
/// the face count the final mesh is simplified to.
const RUNGS: [Rung; 3] = [
    Rung {
        structure: 64,
        resolution: 512,
        budget: 80_000,
        faces: 1_000_000,
    },
    Rung {
        structure: 128,
        resolution: 1024,
        budget: 90_000,
        faces: 1_000_000,
    },
    Rung {
        structure: 192,
        resolution: 1536,
        budget: 110_000,
        faces: 2_000_000,
    },
];
/// The resolutions a refinement can stop at.
pub const RESOLUTIONS: [usize; 3] = [512, 1024, 1536];
// decoded cells past which another pass would not fit
const DECODE_CEILING: usize = 12_000_000;
// components with a smaller share of the faces are dropped as debris (the largest always stays)
const COMPONENT_SHARE: f64 = 0.005;

struct Rung {
    structure: usize,
    resolution: usize,
    budget: usize,
    faces: usize,
}

/// Hi3D's folder (`transformer_512`, `transformer_1024`, `vae`) and the transformers folders of DINOv2-large and CLIP
/// ViT-L/14.
pub struct Hi3dPaths {
    pub hi3d: PathBuf,
    pub dinov2: PathBuf,
    pub clip: PathBuf,
}

/// What a refinement asks for.
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    /// 512, 1024 or 1536: one, two or three passes
    pub resolution: usize,
    pub steps: usize,
    pub seed: u64,
}

impl Settings {
    pub fn new(resolution: usize, seed: u64) -> Self {
        Self {
            resolution,
            steps: schedule::STEPS,
            seed,
        }
    }
}

/// One pass: the grid its cells settled on (smaller than the rung's when the budget shrank it), its cells and the
/// decoded cells.
#[derive(Clone, Debug)]
pub struct PassReport {
    pub structure: usize,
    pub cells: usize,
    pub decoded: usize,
}

pub struct Refined {
    /// Centred on its vertices' mean, its farthest vertex at radius 1, in the input mesh's axes
    pub mesh: Mesh,
    pub passes: Vec<PassReport>,
}

fn weights(path: &Path, device: &Device) -> Result<VarBuilder<'static>> {
    // SAFETY: the checkpoint files are not modified while the model is loaded
    unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device) }
}

/// `coarse` refined under `image` (its alpha frames the subject; an opaque image is used whole). The transformers run
/// in f16 on a GPU (f32 on the CPU), the encoders and the decoder in f32.
pub fn refine(
    paths: &Hi3dPaths,
    image: &DynamicImage,
    coarse: &Mesh,
    settings: &Settings,
    progress: &dyn Progress,
    device: &Device,
) -> Result<Refined> {
    let Some(last) = RUNGS
        .iter()
        .position(|r| r.resolution == settings.resolution)
    else {
        inference_tensor::bail!(
            "resolution must be one of {RESOLUTIONS:?}, got {}",
            settings.resolution
        );
    };
    if settings.steps == 0 {
        inference_tensor::bail!("steps must be positive");
    }
    if coarse.faces.is_empty() {
        inference_tensor::bail!("the coarse mesh has no faces");
    }
    let dtype = if device.is_cpu() {
        DType::F32
    } else {
        DType::F16
    };

    progress.stage("condition", 0);
    let cond = Conditioner::load(&paths.dinov2, &paths.clip, device)?.forward(&prepare(image))?;
    check(progress)?;
    let decoder = Decoder::load(
        dtype,
        weights(&paths.hi3d.join("vae").join("model.safetensors"), device)?,
    )?;
    let mut rng = StdRng::seed_from_u64(settings.seed);
    let mut surface = coarse.clone();
    let mut passes = Vec::new();
    let mut model: Option<(usize, Transformer)> = None;
    for (pass, rung) in RUNGS[..=last].iter().enumerate() {
        let name = if pass == 0 {
            "transformer_512"
        } else {
            "transformer_1024"
        };
        let label = |what: &str| format!("pass {} of {}: {what}", pass + 1, last + 1);
        progress.stage(&label("voxelizing"), 0);
        let Some((cells, settled)) = voxelize::adaptive(&surface, rung.structure, rung.budget)
        else {
            inference_tensor::bail!("no grid holds the surface in {} cells", rung.budget);
        };
        if model.as_ref().is_none_or(|(p, _)| (*p == 0) != (pass == 0)) {
            // one transformer resident at a time
            model = None;
            let vb = weights(&paths.hi3d.join(name).join("model.safetensors"), device)?;
            model = Some((pass, Transformer::load(dtype, vb)?));
        }
        let transformer = &model.as_ref().expect("loaded above").1;
        let kv = transformer.condition(&cond)?;
        let uncond = transformer.condition(&cond.zeros_like()?)?;
        let grid = Grid::new(&cells, device)?;
        let noise: Vec<f32> = (0..cells.len() * LATENT_CHANNELS)
            .map(|_| StandardNormal.sample(&mut rng))
            .collect();
        let noise = Tensor::from_vec(noise, (cells.len(), LATENT_CHANNELS), device)?;

        progress.stage(&label("sampling"), settings.steps);
        let x = schedule::sample(
            transformer,
            &noise,
            &grid,
            &kv,
            &uncond,
            settings.steps,
            &mut |done| {
                progress.advance(done);
                check(progress)
            },
        )?;
        drop((kv, uncond, grid, noise));
        if pass == last {
            // the finest decode wants the transformer's memory
            drop(model.take());
        }
        let c = LATENT_MEAN.len();
        let latents = x
            .to_dtype(DType::F32)?
            .broadcast_mul(&Tensor::from_slice(&LATENT_STD, (1, c), device)?)?
            .broadcast_add(&Tensor::from_slice(&LATENT_MEAN, (1, c), device)?)?;

        progress.stage(&label("decoding"), 0);
        let decoded = decoder.forward(&latents, &cells, rung.structure)?;
        if pass < last && decoded.coords.len() > DECODE_CEILING {
            inference_tensor::bail!(
                "{} decoded cells before another pass, past the {DECODE_CEILING} that fit",
                decoded.coords.len()
            );
        }
        check(progress)?;
        progress.stage(&label("meshing"), 0);
        let mut mesh = point_to_mesh(&decoded, &cells)?;
        if mesh.faces.is_empty() {
            inference_tensor::bail!("mesh extraction produced an empty surface");
        }
        fill_holes(&mut mesh);
        passes.push(PassReport {
            structure: settled,
            cells: cells.len(),
            decoded: decoded.coords.len(),
        });
        surface = mesh;
    }
    progress.stage("simplifying", 0);
    let mut mesh = drop_debris(&surface);
    if mesh.faces.len() > RUNGS[last].faces {
        crate::trellis2::cleanup::simplify_on(&mut mesh, RUNGS[last].faces, device)?;
    }
    normalize(&mut mesh);
    Ok(Refined { mesh, passes })
}

/// The components (faces joined across edges two faces share) holding at least 0.5% of the faces, and the largest,
/// with only the vertices they use.
pub fn drop_debris(mesh: &Mesh) -> Mesh {
    let n = mesh.faces.len();
    let mut by_edge: std::collections::HashMap<[u32; 2], Vec<u32>> =
        std::collections::HashMap::new();
    for (f, face) in mesh.faces.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (face[k], face[(k + 1) % 3]);
            by_edge
                .entry([a.min(b), a.max(b)])
                .or_default()
                .push(f as u32);
        }
    }
    let mut parent: Vec<u32> = (0..n as u32).collect();
    fn find(parent: &mut [u32], mut x: u32) -> u32 {
        while parent[x as usize] != x {
            parent[x as usize] = parent[parent[x as usize] as usize];
            x = parent[x as usize];
        }
        x
    }
    for faces in by_edge.values().filter(|f| f.len() == 2) {
        let (a, b) = (find(&mut parent, faces[0]), find(&mut parent, faces[1]));
        if a != b {
            parent[a.max(b) as usize] = a.min(b);
        }
    }
    let roots: Vec<u32> = (0..n as u32).map(|f| find(&mut parent, f)).collect();
    let mut sizes = vec![0usize; n];
    for &r in &roots {
        sizes[r as usize] += 1;
    }
    let largest = (0..n).max_by_key(|&r| sizes[r]).unwrap_or(0);
    let keep =
        |r: u32| r as usize == largest || sizes[r as usize] as f64 / n as f64 >= COMPONENT_SHARE;
    let mut remap = vec![u32::MAX; mesh.vertices.len()];
    let mut out = Mesh {
        vertices: Vec::new(),
        faces: Vec::new(),
    };
    for (face, &r) in mesh.faces.iter().zip(&roots) {
        if !keep(r) {
            continue;
        }
        out.faces.push(face.map(|v| {
            if remap[v as usize] == u32::MAX {
                remap[v as usize] = out.vertices.len() as u32;
                out.vertices.push(mesh.vertices[v as usize]);
            }
            remap[v as usize]
        }));
    }
    out
}

/// Centred on the vertices' mean, scaled so the farthest vertex sits at radius 1.
pub fn normalize(mesh: &mut Mesh) {
    let n = mesh.vertices.len().max(1) as f64;
    let mut center = [0f64; 3];
    for v in &mesh.vertices {
        for d in 0..3 {
            center[d] += v[d] as f64;
        }
    }
    let center = center.map(|c| c / n);
    let radius = mesh
        .vertices
        .iter()
        .map(|v| {
            (0..3)
                .map(|d| (v[d] as f64 - center[d]).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .fold(0f64, f64::max);
    if radius > 0.0 {
        for v in &mut mesh.vertices {
            *v = [0, 1, 2].map(|d| ((v[d] as f64 - center[d]) / radius) as f32);
        }
    }
}
