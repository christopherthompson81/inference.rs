//! Image to 3D: JoVox's 64^3 structure of the object in one image (its background removed by BiRefNet unless the
//! image already carries a transparent one), then TRELLIS.2's second stage meshes it. Writes the prepared image, the
//! voxels as a cube mesh, the mesh and a `meta.json`.

use std::path::{Path, PathBuf};
use std::time::Instant;

use inference_recon::birefnet::BiRefNet;
use inference_recon::jovox::{self, JoVox, Rgba, SampleOptions};
use inference_recon::trellis2::{self, ShapeStage, cleanup};
use inference_tensor::{DType, Device};
use serde::Serialize;

pub use inference_recon::jovox::{DEFAULT_CFG, DEFAULT_FILL, DEFAULT_STEPS, STEPS_RANGE};
pub use inference_recon::progress::{NoProgress, Progress};

pub const PREPARED_FILE: &str = "prepared.png";
pub const VOXELS_FILE: &str = "voxels.ply";
pub const MESH_FILE: &str = "mesh.ply";
pub const META_FILE: &str = "meta.json";

const WEIGHTS_FILE: &str = "model.safetensors";

/// The checkpoints a voxel run reads.
#[derive(Clone, Debug)]
pub struct VoxelModels {
    /// JoVox-64's folder (`model.safetensors`)
    pub jovox: PathBuf,
    /// DINOv3 ViT-L/16's folder (`model.safetensors`)
    pub dinov3: PathBuf,
    /// BiRefNet's folder, for images without a transparent background
    pub birefnet: Option<PathBuf>,
    /// TRELLIS.2-4B's folder (`pipeline.json` and the shape flow and decoder under `ckpts/`), to mesh the structure
    pub trellis2: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug)]
pub struct VoxelOptions {
    pub sample: SampleOptions,
    /// The share of the prepared image the object's longest side fills
    pub fill: f32,
    /// Remove the background even when the image has transparency of its own
    pub remove_background: bool,
    /// Mesh the structure with TRELLIS.2's second stage
    pub mesh: bool,
}

impl Default for VoxelOptions {
    fn default() -> Self {
        Self {
            sample: SampleOptions::default(),
            fill: DEFAULT_FILL,
            remove_background: false,
            mesh: true,
        }
    }
}

/// The mesh stage's counts and time.
#[derive(Clone, Debug, Serialize)]
pub struct MeshReport {
    /// The mesh as extracted, before hole filling and simplification
    pub raw_faces: usize,
    pub vertices: usize,
    pub faces: usize,
    pub seconds: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct VoxelReport {
    pub voxels: usize,
    pub resolution: usize,
    pub background_removed: bool,
    pub seed: u64,
    pub steps: usize,
    pub cfg_strength: f32,
    pub temperature: f32,
    pub fill: f32,
    pub sample_seconds: f64,
    pub mesh: Option<MeshReport>,
    pub total_seconds: f64,
}

/// The checkpoints given, the others fetched from their repositories (JoVox-64, the gated DINOv3 ViT-L/16, BiRefNet,
/// TRELLIS.2-4B's shape stage) into the hub cache. `birefnet` and `trellis2` are `None` when not wanted, `Some(None)`
/// to fetch them.
pub fn fetch_models(
    jovox: Option<PathBuf>,
    dinov3: Option<PathBuf>,
    birefnet: Option<Option<PathBuf>>,
    trellis2: Option<Option<PathBuf>>,
    progress: &dyn Progress,
) -> anyhow::Result<VoxelModels> {
    let mut slots = [
        (jovox, crate::JOVOX_REPO),
        (dinov3, crate::DINOV3_REPO),
        (birefnet.clone().flatten(), crate::BIREFNET_REPO),
    ];
    let wanted = [true, true, birefnet.is_some()];
    let missing = slots
        .iter()
        .zip(wanted)
        .filter(|((path, _), want)| *want && path.is_none())
        .count();
    if missing > 0 {
        let api = crate::hub_api()?;
        progress.stage(crate::DOWNLOAD_STAGE, missing);
        let mut done = 0;
        for ((path, repo), want) in slots.iter_mut().zip(wanted) {
            if !want || path.is_some() {
                continue;
            }
            inference_recon::progress::check(progress)?;
            let file = api
                .model(repo.to_string())
                .get(WEIGHTS_FILE)
                .map_err(|e| anyhow::anyhow!("fetching `{repo}`: {e}"))?;
            *path = file.parent().map(Path::to_path_buf);
            done += 1;
            progress.advance(done);
        }
    }
    let [(jovox, _), (dinov3, _), (birefnet, _)] = slots;
    let trellis2 = match trellis2 {
        Some(None) => {
            let repo = crate::hub_api()?.model(crate::TRELLIS2_REPO.to_string());
            progress.stage(crate::DOWNLOAD_STAGE, trellis2::FILES.len());
            let mut root = None;
            for (i, file) in trellis2::FILES.iter().enumerate() {
                inference_recon::progress::check(progress)?;
                let path = repo
                    .get(file)
                    .map_err(|e| anyhow::anyhow!("fetching `{}`: {e}", crate::TRELLIS2_REPO))?;
                root = root.or_else(|| {
                    path.ancestors()
                        .nth(Path::new(file).components().count())
                        .map(Path::to_path_buf)
                });
                progress.advance(i + 1);
            }
            root
        }
        Some(path) => path,
        None => None,
    };
    Ok(VoxelModels {
        jovox: jovox.expect("fetched"),
        dinov3: dinov3.expect("fetched"),
        birefnet,
        trellis2,
    })
}

/// Whether a run on `image` needs BiRefNet: it has no transparency, or `remove_background` asks for it anyway.
pub fn needs_background_removal(image: &Path, remove_background: bool) -> anyhow::Result<bool> {
    if remove_background {
        return Ok(true);
    }
    let decoded = inference_recon::read_image(image)
        .map_err(|e| anyhow::anyhow!("{}: {e}", image.display()))?;
    Ok(!decoded.color().has_alpha() || decoded.to_rgba8().pixels().all(|p| p.0[3] == 255))
}

/// `image` decoded, shrunk to a longest side of 1024 (PIL Lanczos), and its background removed with BiRefNet when it
/// has no transparency or `remove_background` asks; whether it was removed.
pub(crate) fn object_image(
    image: &Path,
    birefnet: Option<&Path>,
    remove_background: bool,
    progress: &dyn Progress,
    device: &Device,
) -> anyhow::Result<(Rgba, bool)> {
    let decoded = inference_recon::read_image(image)
        .map_err(|e| anyhow::anyhow!("{}: {e}", image.display()))?
        .to_rgba8();
    let input = jovox::shrink(&Rgba {
        width: decoded.width() as usize,
        height: decoded.height() as usize,
        pixels: decoded.into_raw(),
    });
    let remove = remove_background || !jovox::has_transparency(&input);
    let input = if remove {
        let Some(dir) = birefnet else {
            anyhow::bail!(
                "the image has no transparent background and no BiRefNet checkpoint is configured to remove it"
            );
        };
        progress.stage("background", 0);
        let birefnet = BiRefNet::load(dir, device)?;
        let rgb: Vec<u8> = input
            .pixels
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|px| [px[0], px[1], px[2]])
            .collect();
        Rgba {
            pixels: birefnet.cut_out(&rgb, (input.width, input.height))?,
            ..input
        }
    } else {
        input
    };
    Ok((input, remove))
}

/// [`generate_voxels_on`] on the first CUDA device, or the CPU without one.
pub fn generate_voxels(
    models: &VoxelModels,
    image: &Path,
    out_dir: &Path,
    options: &VoxelOptions,
    progress: &dyn Progress,
) -> anyhow::Result<VoxelReport> {
    let device = Device::cuda_if_available(0)?;
    generate_voxels_on(models, image, out_dir, options, progress, &device)
}

/// The voxel structure of the object in `image` into `out_dir`: `prepared.png` (what the model sees), `voxels.ply`
/// (the occupied cells as cubes in the unit cube), with `options.mesh` `mesh.ply` (TRELLIS.2's surface, holes filled,
/// at most 1M faces), and `meta.json` (this report). The flows run in bf16 on a GPU, f32 on the CPU.
pub fn generate_voxels_on(
    models: &VoxelModels,
    image: &Path,
    out_dir: &Path,
    options: &VoxelOptions,
    progress: &dyn Progress,
    device: &Device,
) -> anyhow::Result<VoxelReport> {
    let start = Instant::now();
    std::fs::create_dir_all(out_dir)?;
    let (input, remove) = object_image(
        image,
        models.birefnet.as_deref(),
        options.remove_background,
        progress,
        device,
    )?;
    inference_recon::progress::check(progress)?;
    let prepared = jovox::prepare(&input, options.fill)?;
    let prepared_path = out_dir.join(PREPARED_FILE);
    image::RgbaImage::from_raw(
        prepared.width as u32,
        prepared.height as u32,
        prepared.pixels.clone(),
    )
    .expect("a whole image")
    .save(&prepared_path)?;
    progress.preview("the prepared object", &prepared_path);

    let dtype = if device.is_cpu() {
        DType::F32
    } else {
        DType::BF16
    };
    if options.mesh && models.trellis2.is_none() {
        anyhow::bail!("meshing needs the TRELLIS.2-4B checkpoint");
    }
    let model = JoVox::load(&models.jovox, &models.dinov3, dtype, device)?;
    let cond = model.condition(&prepared)?;
    let sampled = Instant::now();
    let voxels = model.sample(&cond, &options.sample, progress)?;
    let sample_seconds = sampled.elapsed().as_secs_f64();
    std::fs::write(out_dir.join(VOXELS_FILE), jovox::voxel_ply(&voxels))?;
    let mesh = match &models.trellis2 {
        Some(dir) if options.mesh && !voxels.is_empty() => {
            let meshed = Instant::now();
            let size = (trellis2::RESOLUTION, trellis2::RESOLUTION);
            let cond = model.encode_rgb(&trellis2::condition_image(&prepared)?, size)?;
            drop(model);
            let stage = ShapeStage::load(dir, device)?;
            let mut mesh = stage.mesh(&cond, &voxels, options.sample.seed, progress)?;
            drop(stage);
            let raw_faces = mesh.faces.len();
            progress.stage("cleanup", 0);
            cleanup::fill_holes(&mut mesh, cleanup::MAX_HOLE_PERIMETER);
            cleanup::simplify(&mut mesh, cleanup::SIMPLIFY_TARGET);
            std::fs::write(out_dir.join(MESH_FILE), mesh.to_ply())?;
            Some(MeshReport {
                raw_faces,
                vertices: mesh.vertices.len(),
                faces: mesh.faces.len(),
                seconds: meshed.elapsed().as_secs_f64(),
            })
        }
        _ => None,
    };
    let s = options.sample;
    let report = VoxelReport {
        voxels: voxels.len(),
        resolution: jovox::sampler::OUTPUT_SIDE,
        background_removed: remove,
        seed: s.seed,
        steps: s.steps,
        cfg_strength: s.cfg_strength,
        temperature: s.temperature,
        fill: options.fill,
        sample_seconds,
        mesh,
        total_seconds: start.elapsed().as_secs_f64(),
    };
    std::fs::write(out_dir.join(META_FILE), serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}
