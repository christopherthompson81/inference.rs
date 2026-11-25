//! A mesh refined under an image with Hi3D v1.0: the image's object cut out (BiRefNet unless it already has
//! transparency), the coarse mesh given or made by TRELLIS.2, up to three Hi3D passes at 512, 1024 and 1536, the result
//! exported untextured or painted by TRELLIS.2's texturing. Writes `model.glb`, `mesh.ply`, the conditioning image and
//! a `meta.json`.

use std::path::{Path, PathBuf};
use std::time::Instant;

use inference_recon::hi3d::pipeline::{
    self as hi3d, ENCODER_FILES, FIRST_PASS_FILES, Hi3dPaths, LATER_PASS_FILES,
};
use inference_recon::trellis2::glb::{Pbr, to_glb, y_up};
use inference_recon::trellis2::mesh::Mesh;
use inference_recon::trellis2::mesh_io::read_mesh;
use inference_recon::trellis2::pipeline::{self as trellis2, Variant};
use inference_tensor::Device;
use serde::Serialize;

use crate::image_to_3d::{GLB_FILE, MESH_FILE, META_FILE, fetch_files};
pub use inference_recon::hi3d::pipeline::{RESOLUTIONS, Settings};
pub use inference_recon::hi3d::schedule::STEPS;
pub use inference_recon::progress::{NoProgress, Progress};

/// What Hi3D conditions on: the cut-out object framed on white
pub const CONDITION_FILE: &str = "condition.png";
pub use crate::image_to_3d::PREPARED_FILE;
/// The coarse mesh TRELLIS.2 made, when none was given
pub const COARSE_FILE: &str = "coarse.ply";
// texturing writes its own files into a folder of its own, then its model takes the untextured one's place
const TEXTURED_DIR: &str = "textured";
/// The most sampling steps a pass may take
pub const MAX_STEPS: usize = 100;
// the untextured GLB's surface: light grey, fully rough
const SURFACE: Pbr = Pbr {
    base_color: [0.8, 0.8, 0.8],
    metallic: 0.0,
    roughness: 1.0,
    alpha: 1.0,
};
// TRELLIS.2's coarse shape: one 512 pass is plenty for the 64^3 grid Hi3D's first pass voxelizes it on
const COARSE_VARIANT: Variant = Variant::Res512;

/// The checkpoints a refinement reads.
#[derive(Clone, Debug)]
pub struct RefineMeshModels {
    pub hi3d: Hi3dPaths,
    /// BiRefNet's folder, for images without a transparent background
    pub birefnet: Option<PathBuf>,
    /// TRELLIS.2's, for a coarse mesh when none is given
    pub coarse: Option<crate::image_to_3d::ImageTo3dModels>,
    /// TRELLIS.2's texturing, when the result is to be painted
    pub texture: Option<crate::texture_mesh::TextureMeshModels>,
}

#[derive(Clone, Copy, Debug)]
pub struct RefineMeshOptions {
    pub settings: Settings,
    /// Remove the background even when the image has transparency of its own
    pub remove_background: bool,
    /// Paint the refined mesh with TRELLIS.2's texturing at its own resolution
    pub texture: Option<crate::texture_mesh::TextureMeshOptions>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RefineMeshReport {
    pub resolution: usize,
    pub seed: u64,
    pub steps: usize,
    pub background_removed: bool,
    /// Whether TRELLIS.2 made the coarse mesh
    pub coarse_generated: bool,
    pub coarse_faces: usize,
    pub passes: Vec<PassReport>,
    pub vertices: usize,
    pub faces: usize,
    pub texture: Option<crate::texture_mesh::TextureMeshReport>,
    pub total_seconds: f64,
}

/// One Hi3D pass: the grid its cells settled on, the cells sampled and the cells decoded.
#[derive(Clone, Debug, Serialize)]
pub struct PassReport {
    pub structure: usize,
    pub cells: usize,
    pub decoded: usize,
}

/// The checkpoints given, the others fetched into the hub cache: Hi3D v1.0 (`transformer_1024` only for a resolution
/// above 512), DINOv2-large, CLIP ViT-L/14, and as wanted BiRefNet, TRELLIS.2's generation and its texturing.
#[allow(clippy::too_many_arguments)]
pub fn fetch_models(
    jobs: &crate::JobsConfig,
    resolution: usize,
    birefnet: bool,
    coarse: bool,
    texture: bool,
    progress: &dyn Progress,
) -> anyhow::Result<RefineMeshModels> {
    let mut files: Vec<&str> = FIRST_PASS_FILES.to_vec();
    if resolution > RESOLUTIONS[0] {
        files.extend(LATER_PASS_FILES);
    }
    let hi3d = match &jobs.hi3d {
        Some(p) => p.clone(),
        None => fetch_files(crate::HI3D_REPO, &files, progress)?,
    };
    let dinov2 = match &jobs.dinov2_large {
        Some(p) => p.clone(),
        None => fetch_files(crate::DINOV2_LARGE_REPO, &ENCODER_FILES, progress)?,
    };
    let clip = match &jobs.clip_large {
        Some(p) => p.clone(),
        None => fetch_files(crate::CLIP_LARGE_REPO, &ENCODER_FILES, progress)?,
    };
    let birefnet_dir = || -> anyhow::Result<PathBuf> {
        match &jobs.birefnet {
            Some(p) => Ok(p.clone()),
            None => fetch_files(crate::BIREFNET_REPO, &["model.safetensors"], progress),
        }
    };
    let birefnet = if birefnet {
        Some(birefnet_dir()?)
    } else {
        None
    };
    let coarse = if coarse {
        Some(crate::image_to_3d::fetch_models(
            jobs.trellis2.clone(),
            jobs.trellis_large.clone(),
            jobs.dinov3.clone(),
            None,
            progress,
        )?)
    } else {
        None
    };
    let texture = if texture {
        Some(crate::texture_mesh::fetch_models(
            jobs.trellis2.clone(),
            jobs.dinov3.clone(),
            None,
            progress,
        )?)
    } else {
        None
    };
    Ok(RefineMeshModels {
        hi3d: Hi3dPaths { hi3d, dinov2, clip },
        birefnet,
        coarse,
        texture,
    })
}

// the y-up mesh as `to_glb` wants it (z up), so its y-up turn gives it back
fn z_up(mesh: &Mesh) -> Mesh {
    Mesh {
        vertices: mesh.vertices.iter().map(|v| [v[0], -v[2], v[1]]).collect(),
        faces: mesh.faces.clone(),
    }
}

/// `mesh` (or, when `None`, TRELLIS.2's shape of the object in `image`) refined under `image` into `out_dir`:
/// `model.glb` (untextured, or painted when `options.texture` asks), `mesh.ply` (the refined surface, y up, centred,
/// radius 1), `prepared.png` (the cut-out object), `condition.png`, `coarse.ply` when TRELLIS.2 made it, and
/// `meta.json`.
pub fn refine_mesh_on(
    models: &RefineMeshModels,
    mesh: Option<&Path>,
    image: &Path,
    out_dir: &Path,
    options: &RefineMeshOptions,
    progress: &dyn Progress,
    device: &Device,
) -> anyhow::Result<RefineMeshReport> {
    let start = Instant::now();
    std::fs::create_dir_all(out_dir)?;
    let (input, removed) = crate::voxels::object_image(
        image,
        models.birefnet.as_deref(),
        options.remove_background,
        progress,
        device,
    )?;
    let rgba = image::RgbaImage::from_raw(
        input.width as u32,
        input.height as u32,
        input.pixels.clone(),
    )
    .expect("a whole image");
    let prepared_path = out_dir.join(PREPARED_FILE);
    rgba.save(&prepared_path)?;
    let condition_path = out_dir.join(CONDITION_FILE);
    inference_recon::hi3d::conditioner::prepare(&image::DynamicImage::ImageRgba8(rgba.clone()))
        .save(&condition_path)?;
    progress.preview("what Hi3D sees", &condition_path);

    let (coarse, generated) = match mesh {
        Some(path) => (read_mesh(path).map_err(|e| anyhow::anyhow!(e))?.mesh, false),
        None => {
            let Some(trellis) = &models.coarse else {
                anyhow::bail!("no coarse mesh given and TRELLIS.2 not loaded to make one");
            };
            let settings = trellis2::Settings::new(COARSE_VARIANT, options.settings.seed);
            let mut shape = None;
            trellis2::generate_with(
                &trellis.paths,
                &input,
                &settings,
                progress,
                device,
                &mut |_, mesh, _| {
                    shape = Some(mesh.clone());
                },
            )?;
            let shape = shape.ok_or_else(|| anyhow::anyhow!("TRELLIS.2 made no shape"))?;
            // TRELLIS.2 works z up; Hi3D sees meshes as glTF lays them out, y up
            let mesh = Mesh {
                vertices: shape.vertices.iter().map(|&v| y_up(v)).collect(),
                faces: shape.faces,
            };
            std::fs::write(out_dir.join(COARSE_FILE), mesh.to_ply())?;
            (mesh, true)
        }
    };

    let refined = hi3d::refine(
        &models.hi3d,
        &image::DynamicImage::ImageRgba8(rgba),
        &coarse,
        &options.settings,
        progress,
        device,
    )?;
    progress.stage("export", 0);
    let mesh_path = out_dir.join(MESH_FILE);
    std::fs::write(&mesh_path, refined.mesh.to_ply())?;
    let surface = vec![SURFACE; refined.mesh.vertices.len()];
    std::fs::write(
        out_dir.join(GLB_FILE),
        to_glb(&z_up(&refined.mesh), &surface, true),
    )?;

    let texture = match (&models.texture, &options.texture) {
        (Some(models), Some(texture_options)) => {
            let dir = out_dir.join(TEXTURED_DIR);
            // the object already cut out: texturing finds its transparency and keeps it
            let report = crate::texture_mesh::texture_mesh_on(
                models,
                &mesh_path,
                &prepared_path,
                &dir,
                &crate::texture_mesh::TextureMeshOptions {
                    remove_background: false,
                    ..*texture_options
                },
                progress,
                device,
            )?;
            std::fs::rename(dir.join(GLB_FILE), out_dir.join(GLB_FILE))?;
            std::fs::remove_dir_all(&dir)?;
            Some(report)
        }
        _ => None,
    };
    let report = RefineMeshReport {
        resolution: options.settings.resolution,
        seed: options.settings.seed,
        steps: options.settings.steps,
        background_removed: removed,
        coarse_generated: generated,
        coarse_faces: coarse.faces.len(),
        passes: refined
            .passes
            .iter()
            .map(|p| PassReport {
                structure: p.structure,
                cells: p.cells,
                decoded: p.decoded,
            })
            .collect(),
        vertices: refined.mesh.vertices.len(),
        faces: refined.mesh.faces.len(),
        texture,
        total_seconds: start.elapsed().as_secs_f64(),
    };
    std::fs::write(out_dir.join(META_FILE), serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}
