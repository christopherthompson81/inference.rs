//! A mesh painted from an image with TRELLIS.2's texturing pipeline: the image's object cut out (BiRefNet unless it
//! already has transparency), the mesh fitted to the unit cube, encoded and painted, and baked onto its own texture
//! coordinates (or an unwrap) into a GLB, beside the prepared image and a `meta.json`.

use std::path::{Path, PathBuf};
use std::time::Instant;

use inference_recon::trellis2::bake;
use inference_recon::trellis2::glb::to_textured_glb;
use inference_recon::trellis2::mesh_io::read_mesh;
use inference_recon::trellis2::pipeline::Trellis2Paths;
use inference_recon::trellis2::texturing::{self, fit_mesh, paint};
use inference_tensor::Device;
use serde::Serialize;

use crate::image_to_3d::{GLB_FILE, META_FILE, PREPARED_FILE, fetch_files};
pub use inference_recon::progress::{NoProgress, Progress};
pub use inference_recon::trellis2::texturing::{RESOLUTIONS, TexturingSettings};

const WEIGHTS_FILE: &str = "model.safetensors";
/// The mesh files texturing reads
pub const MESH_EXTENSIONS: [&str; 3] = ["glb", "obj", "ply"];

/// The checkpoints a texturing run reads.
#[derive(Clone, Debug)]
pub struct TextureMeshModels {
    pub trellis2: PathBuf,
    pub dinov3: PathBuf,
    /// BiRefNet's folder, for images without a transparent background
    pub birefnet: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug)]
pub struct TextureMeshOptions {
    pub settings: TexturingSettings,
    /// Width and height of the baked textures
    pub texture_size: usize,
    /// Unwrap the mesh even when it has texture coordinates of its own
    pub unwrap: bool,
    /// Remove the background even when the image has transparency of its own
    pub remove_background: bool,
}

impl Default for TextureMeshOptions {
    fn default() -> Self {
        Self {
            settings: TexturingSettings::default(),
            texture_size: bake::TEXTURE_SIZE,
            unwrap: false,
            remove_background: false,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct TextureMeshReport {
    pub resolution: usize,
    pub seed: u64,
    pub background_removed: bool,
    /// Whether the mesh's own texture coordinates were used (otherwise it was unwrapped)
    pub own_uvs: bool,
    pub vertices: usize,
    pub faces: usize,
    pub surface_cells: usize,
    pub latent_cells: usize,
    pub texture_size: usize,
    pub total_seconds: f64,
}

/// The checkpoints given, the others fetched (TRELLIS.2-4B's texturing files, the gated DINOv3 ViT-L/16, BiRefNet) into
/// the hub cache. `birefnet` is `None` when not wanted, `Some(None)` to fetch it.
pub fn fetch_models(
    trellis2: Option<PathBuf>,
    dinov3: Option<PathBuf>,
    birefnet: Option<Option<PathBuf>>,
    progress: &dyn Progress,
) -> anyhow::Result<TextureMeshModels> {
    let trellis2 = match trellis2 {
        Some(p) => p,
        None => fetch_files(crate::TRELLIS2_REPO, &texturing::FILES, progress)?,
    };
    let dinov3 = match dinov3 {
        Some(p) => p,
        None => fetch_files(crate::DINOV3_REPO, &[WEIGHTS_FILE], progress)?,
    };
    let birefnet = match birefnet {
        Some(Some(p)) => Some(p),
        Some(None) => Some(fetch_files(
            crate::BIREFNET_REPO,
            &[WEIGHTS_FILE],
            progress,
        )?),
        None => None,
    };
    Ok(TextureMeshModels {
        trellis2,
        dinov3,
        birefnet,
    })
}

/// Whether `path` names a mesh texturing reads, by its extension.
pub fn is_mesh(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| MESH_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
}

/// `mesh` painted from the object in `image` into `out_dir`: `model.glb` (the mesh fitted to the unit cube, y up, with
/// base color and metallic-roughness textures), `prepared.png` (the cut-out object) and `meta.json`.
pub fn texture_mesh_on(
    models: &TextureMeshModels,
    mesh: &Path,
    image: &Path,
    out_dir: &Path,
    options: &TextureMeshOptions,
    progress: &dyn Progress,
    device: &Device,
) -> anyhow::Result<TextureMeshReport> {
    let start = Instant::now();
    std::fs::create_dir_all(out_dir)?;
    let read = read_mesh(mesh).map_err(|e| anyhow::anyhow!(e))?;
    let own_uvs = read.uvs.is_some() && !options.unwrap;
    let uvs = read.uvs.filter(|_| !options.unwrap);
    let mut fitted = read.mesh;
    fit_mesh(&mut fitted);
    let (input, removed) = crate::voxels::object_image(
        image,
        models.birefnet.as_deref(),
        options.remove_background,
        progress,
        device,
    )?;
    let prepared_path = out_dir.join(PREPARED_FILE);
    image::RgbaImage::from_raw(
        input.width as u32,
        input.height as u32,
        input.pixels.clone(),
    )
    .expect("a whole image")
    .save(&prepared_path)?;
    progress.preview("the object", &prepared_path);
    let paths = Trellis2Paths {
        trellis2: models.trellis2.clone(),
        structure_decoder: PathBuf::new(),
        dinov3: models.dinov3.clone(),
    };
    let settings = options.settings;
    // the mesh's unwrap and texel layout run on their own thread while it is painted
    let (painted, prepared) = std::thread::scope(|scope| {
        let mesh = &fitted;
        let layout = scope.spawn(move || {
            bake::prepare_given(
                mesh,
                uvs,
                settings.resolution,
                options.texture_size,
                &|| progress.cancelled(),
            )
        });
        let painted = paint(&paths, &input, mesh, &settings, progress, device);
        let prepared = layout
            .join()
            .map_err(|_| anyhow::anyhow!("the mesh's layout panicked"));
        (painted, prepared)
    });
    let (painted, prepared) = (painted?, prepared??);
    progress.stage("export", 0);
    let baked = bake::finish(prepared, &painted.voxels, &painted.attrs)?;
    std::fs::write(out_dir.join(GLB_FILE), to_textured_glb(&baked))?;
    let report = TextureMeshReport {
        resolution: settings.resolution,
        seed: settings.seed,
        background_removed: removed,
        own_uvs,
        vertices: baked.unwrapped.mesh.vertices.len(),
        faces: baked.unwrapped.mesh.faces.len(),
        surface_cells: painted.surface_cells,
        latent_cells: painted.latent_cells,
        texture_size: options.texture_size,
        total_seconds: start.elapsed().as_secs_f64(),
    };
    std::fs::write(out_dir.join(META_FILE), serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}
