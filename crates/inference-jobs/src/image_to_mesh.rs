//! Image to mesh with LATO.2: the object cut out (BiRefNet unless it already has transparency), a voxel scaffold from
//! TRELLIS's structure flow, vertices from V-Flow and the vertex VAE (about as many as asked), and their connectivity
//! from T-Flow and the topology VAE. Writes the mesh as a GLB and an OBJ, the conditioning image and a `meta.json`.

use std::path::{Path, PathBuf};
use std::time::Instant;

use inference_recon::lato2::{self, Lato2Options, Lato2Paths};
use inference_recon::trellis2::glb::{Pbr, to_glb};
use inference_recon::trellis2::mesh::Mesh;
use inference_tensor::Device;
use serde::Serialize;

pub use inference_recon::lato2::faces::{MAX_LOOP, MAX_RING};
pub use inference_recon::lato2::structure::{STRUCTURE_SAMPLER, StructureSampler};
pub use inference_recon::lato2::topology::TOPO_STEPS;
pub use inference_recon::lato2::vertex_flow::{VERTEX_SAMPLER, VertexSampler};
pub use inference_recon::lato2::{DEFAULT_VERTICES, Lato2Options as Settings, VERTICES_RANGE};
pub use inference_recon::progress::{NoProgress, Progress};

pub const GLB_FILE: &str = "model.glb";
pub const OBJ_FILE: &str = "mesh.obj";
pub const CONDITION_FILE: &str = "condition.png";
pub const META_FILE: &str = "meta.json";
const WEIGHTS_FILE: &str = "model.safetensors";
// the untextured GLB's surface: light grey, fully rough
const SURFACE: Pbr = Pbr {
    base_color: [0.8, 0.8, 0.8],
    metallic: 0.0,
    roughness: 1.0,
    alpha: 1.0,
};

/// The checkpoints an image-to-mesh run reads.
#[derive(Clone, Debug)]
pub struct ImageToMeshModels {
    pub paths: Lato2Paths,
    /// BiRefNet's folder, for images without a transparent background
    pub birefnet: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ImageToMeshOptions {
    pub lato2: Lato2Options,
    /// Remove the background even when the image has transparency of its own
    pub remove_background: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct ImageToMeshReport {
    pub seed: u64,
    pub requested_vertices: usize,
    pub background_removed: bool,
    pub scaffold_cells: usize,
    pub vertices: usize,
    pub edges: usize,
    pub faces: usize,
    pub total_seconds: f64,
}

/// The checkpoints given, the others fetched into the hub cache: LATO.2 (`0x4c48/LATO.2`), TRELLIS-image-large's
/// structure flow and decoder, DINOv2 ViT-L/14 with registers and BiRefNet. `birefnet` is `None` when not wanted,
/// `Some(None)` to fetch it.
pub fn fetch_models(
    lato2: Option<PathBuf>,
    trellis_large: Option<PathBuf>,
    dinov2: Option<PathBuf>,
    birefnet: Option<Option<PathBuf>>,
    progress: &dyn Progress,
) -> anyhow::Result<ImageToMeshModels> {
    let fetch = crate::image_to_3d::fetch_files;
    let lato2 = match lato2 {
        Some(p) => p,
        None => fetch(crate::LATO2_REPO, &lato2::FILES, progress)?,
    };
    let trellis_large = match trellis_large {
        Some(p) => p,
        None => fetch(
            crate::TRELLIS_LARGE_REPO,
            &[lato2::STRUCTURE_FLOW_FILE, lato2::STRUCTURE_DECODER_FILE],
            progress,
        )?,
    };
    let dinov2 = match dinov2 {
        Some(p) => p,
        None => fetch(crate::DINOV2_REG_REPO, &[WEIGHTS_FILE], progress)?,
    }
    .join(WEIGHTS_FILE);
    let birefnet = match birefnet {
        Some(Some(p)) => Some(p),
        Some(None) => Some(fetch(crate::BIREFNET_REPO, &[WEIGHTS_FILE], progress)?),
        None => None,
    };
    Ok(ImageToMeshModels {
        paths: Lato2Paths {
            lato2,
            trellis_large,
            dinov2,
        },
        birefnet,
    })
}

/// The mesh as a Wavefront OBJ, vertices in the order generated.
pub fn to_obj(mesh: &Mesh) -> String {
    let mut out = String::with_capacity(mesh.vertices.len() * 32 + mesh.faces.len() * 24);
    for v in &mesh.vertices {
        out.push_str(&format!("v {} {} {}\n", v[0], v[1], v[2]));
    }
    for f in &mesh.faces {
        out.push_str(&format!("f {} {} {}\n", f[0] + 1, f[1] + 1, f[2] + 1));
    }
    out
}

/// The mesh of the object in `image` into `out_dir`: `model.glb` (untextured, double-sided: the faces are wound outward
/// where that can be told, and a wrong one still shows), `mesh.obj` (both y up, in the unit cube
/// around the origin), `condition.png` (the crop the models see) and `meta.json`.
pub fn generate_mesh_on(
    models: &ImageToMeshModels,
    image: &Path,
    out_dir: &Path,
    options: &ImageToMeshOptions,
    progress: &dyn Progress,
    device: &Device,
) -> anyhow::Result<ImageToMeshReport> {
    let start = Instant::now();
    std::fs::create_dir_all(out_dir)?;
    let (input, removed) = crate::voxels::object_image(
        image,
        models.birefnet.as_deref(),
        options.remove_background,
        progress,
        device,
    )?;
    let generated = lato2::generate(&models.paths, &input, &options.lato2, progress, device)?;
    let side = lato2::IMAGE_SIZE as u32;
    let condition_path = out_dir.join(CONDITION_FILE);
    image::RgbImage::from_raw(side, side, generated.condition.clone())
        .expect("a whole image")
        .save(&condition_path)?;
    progress.preview("what the model sees", &condition_path);
    progress.stage("export", 0);
    let mesh = &generated.mesh;
    // `to_glb` turns a z-up mesh y up; this one already is
    let z_up = Mesh {
        vertices: mesh.vertices.iter().map(|v| [v[0], -v[2], v[1]]).collect(),
        faces: mesh.faces.clone(),
    };
    std::fs::write(
        out_dir.join(GLB_FILE),
        to_glb(&z_up, &vec![SURFACE; mesh.vertices.len()], true),
    )?;
    std::fs::write(out_dir.join(OBJ_FILE), to_obj(mesh))?;
    let report = ImageToMeshReport {
        seed: options.lato2.seed,
        requested_vertices: options.lato2.vertices,
        background_removed: removed,
        scaffold_cells: generated.scaffold.len(),
        vertices: mesh.vertices.len(),
        edges: generated.edges,
        faces: mesh.faces.len(),
        total_seconds: start.elapsed().as_secs_f64(),
    };
    std::fs::write(out_dir.join(META_FILE), serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}
