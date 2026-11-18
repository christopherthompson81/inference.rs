//! Image to Gaussian splats with Idol (speridlabs/idol): the object in one image as 3D Gaussians. Writes the splats as
//! a 3DGS PLY and as SPZ, a printable STL of the solid they enclose, the image the model saw and a `meta.json` with
//! the input's predicted camera.

use std::path::{Path, PathBuf};
use std::time::Instant;

use inference_recon::idol::{self, IdolOutput, IdolPaths, SplatData};
use inference_recon::jovox::Rgba;
use inference_recon::printable::{SPLAT_GRID, Solid};
use inference_recon::splatting::spz::{write_ply, write_spz};
use inference_tensor::Device;
use serde::Serialize;

pub use crate::print::{DEFAULT_PRINT_SIZE_MM, PRINT_FILE, PRINT_SIZE_RANGE_MM, PrintReport};
pub use inference_recon::idol::{
    IdolSettings, MARGIN_REAL, MARGIN_SYNTHETIC, MAX_GAUSSIANS, MAX_STEPS, MIN_GAUSSIANS,
};
pub use inference_recon::progress::{NoProgress, Progress};

pub const PLY_FILE: &str = "splat.ply";
pub const SPZ_FILE: &str = "splat.spz";
pub const PREPARED_FILE: &str = "prepared.png";
pub const META_FILE: &str = "meta.json";
/// Idol's own weights in its repository.
pub const IDOL_FILES: [&str; 5] = [
    "config.json",
    "denoiser.safetensors",
    "decoder.safetensors",
    "decoder_gs.safetensors",
    "birefnet.safetensors",
];
const DINOV3_FILE: &str = "model.safetensors";
// FLUX.2's VAE, from FLUX.2-klein-4B (Apache 2.0, ungated): the same weights as FLUX.2-dev's once in bf16, which
// is how Idol runs it
const VAE_FILE: &str = "vae/diffusion_pytorch_model.safetensors";

/// The checkpoints given, the others fetched into the hub cache: Idol, the gated DINOv3 ViT-H+/16, FLUX.2's VAE.
pub fn fetch_models(
    idol: Option<PathBuf>,
    dinov3: Option<PathBuf>,
    vae: Option<PathBuf>,
    progress: &dyn Progress,
) -> anyhow::Result<IdolPaths> {
    let idol = match idol {
        Some(p) => p,
        None => crate::image_to_3d::fetch_files(crate::IDOL_REPO, &IDOL_FILES, progress)?,
    };
    let dinov3 = match dinov3 {
        Some(p) => p,
        None => {
            crate::image_to_3d::fetch_files(crate::DINOV3_HPLUS_REPO, &[DINOV3_FILE], progress)?
                .join(DINOV3_FILE)
        }
    };
    let vae = match vae {
        Some(p) => p,
        None => crate::image_to_3d::fetch_files(crate::FLUX2_VAE_REPO, &[VAE_FILE], progress)?
            .join(VAE_FILE),
    };
    Ok(IdolPaths { idol, dinov3, vae })
}

#[derive(Clone, Copy, Debug)]
pub struct ImageToSplatOptions {
    pub settings: IdolSettings,
    /// The printable STL's longest side, in millimetres
    pub print_size_mm: f64,
}

impl Default for ImageToSplatOptions {
    fn default() -> Self {
        Self {
            settings: IdolSettings::default(),
            print_size_mm: DEFAULT_PRINT_SIZE_MM,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct CameraReport {
    /// radians
    pub yaw: f64,
    /// radians
    pub pitch: f64,
    pub radius: f64,
    /// vertical field of view, degrees
    pub fov: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ImageToSplatReport {
    pub seed: u64,
    pub gaussians: usize,
    pub steps: usize,
    pub margin: f64,
    /// the input's predicted camera
    pub camera: CameraReport,
    /// The printable STL, unless nothing came out solid
    pub print: Option<PrintReport>,
    pub total_seconds: f64,
}

/// The splats of the object in `image` into `out_dir`: `splat.ply`, `splat.spz`, `print.stl`, `prepared.png` and
/// `meta.json`.
pub fn generate_splat_on(
    models: &IdolPaths,
    image: &Path,
    out_dir: &Path,
    options: &ImageToSplatOptions,
    progress: &dyn Progress,
    device: &Device,
) -> anyhow::Result<ImageToSplatReport> {
    let start = Instant::now();
    let settings = &options.settings;
    std::fs::create_dir_all(out_dir)?;
    let decoded = inference_recon::read_image(image)
        .map_err(|e| anyhow::anyhow!("{}: {e}", image.display()))?
        .to_rgba8();
    let input = Rgba {
        width: decoded.width() as usize,
        height: decoded.height() as usize,
        pixels: decoded.into_raw(),
    };
    let IdolOutput {
        gaussians,
        camera,
        prepared,
    } = idol::generate(models, &input, settings, progress, device)?;
    let side = idol::IDOL_IMAGE_SIZE as u32;
    let prepared_path = out_dir.join(PREPARED_FILE);
    image::RgbImage::from_raw(side, side, prepared)
        .expect("a whole image")
        .save(&prepared_path)?;
    progress.preview("the object", &prepared_path);
    progress.stage("export", 0);
    let data = SplatData::from_gaussians(&gaussians);
    let count = write_ply(&out_dir.join(PLY_FILE), &data.rows())?;
    write_spz(&out_dir.join(SPZ_FILE), &data.rows())?;
    progress.stage("printable solid", 0);
    let print = crate::print::write_print(
        || Solid::from_gaussians(&gaussians, SPLAT_GRID),
        out_dir,
        options.print_size_mm,
    )?;
    let report = ImageToSplatReport {
        seed: settings.seed,
        gaussians: count,
        steps: settings.steps,
        margin: settings.margin,
        camera: CameraReport {
            yaw: camera.yaw,
            pitch: camera.pitch,
            radius: camera.radius,
            fov: camera.fov,
        },
        print,
        total_seconds: start.elapsed().as_secs_f64(),
    };
    std::fs::write(out_dir.join(META_FILE), serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}
