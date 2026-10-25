//! One reconstruction run: image files in, an output folder with every artifact and a `meta.json` out.

use std::path::{Path, PathBuf};
use std::time::Instant;

use inference_tensor::Result;
use serde::Serialize;

use super::WorldMirror;
use super::output::{OutputSettings, WrittenCounts, write_outputs};
use super::preprocess::{RgbImage, adaptive_size, images_to_tensor, resized_dims};
use super::priors::{PriorInputs, load_prior_files};

/// The longest edge the reference infers at by default
pub const DEFAULT_TARGET_SIZE: usize = 952;
/// The most frames a run takes: global attention over every frame's tokens grows with the square of the count
pub const MAX_FRAMES: usize = 64;
// Image files a folder input takes, as the reference globs them
const IMAGE_EXTENSIONS: [&str; 4] = ["jpg", "jpeg", "png", "webp"];
const PATCH: usize = 14;
const MODEL_NAME: &str = "Tencent HY-World 2.0 WorldMirror 2.0";

/// What to reconstruct and how.
#[derive(Clone, Debug)]
pub struct ReconstructOptions {
    /// the longest edge to infer at (the first image's, capped at this and floored to 14)
    pub target_size: usize,
    pub output: OutputSettings,
    /// known cameras: a JSON of `extrinsics` (camera-to-world) and `intrinsics` (original pixels) by `camera_id`
    pub prior_cameras: Option<PathBuf>,
    /// known z-depth maps named by image stem
    pub prior_depth: Option<PathBuf>,
}

impl Default for ReconstructOptions {
    fn default() -> Self {
        Self {
            target_size: DEFAULT_TARGET_SIZE,
            output: OutputSettings::default(),
            prior_cameras: None,
            prior_depth: None,
        }
    }
}

/// The image files an input path names: the file itself, or a folder's images sorted by name.
pub fn input_images(input: &Path) -> Result<Vec<PathBuf>> {
    if input.is_file() {
        return Ok(vec![input.to_path_buf()]);
    }
    let mut files = std::fs::read_dir(input)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| IMAGE_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
        })
        .collect::<Vec<_>>();
    files.sort();
    Ok(files)
}

// An image decoded to 8-bit RGB, alpha composited over white as the reference does
fn decode(path: &Path) -> Result<RgbImage> {
    let image = image::open(path)
        .map_err(|e| inference_tensor::Error::msg(format!("{}: {e}", path.display())))?;
    let rgba = image.to_rgba8();
    let (width, height) = (rgba.width() as usize, rgba.height() as usize);
    let pixels = rgba
        .pixels()
        .flat_map(|p| {
            let a = u32::from(p.0[3]);
            [0, 1, 2].map(|c| ((u32::from(p.0[c]) * a + 255 * (255 - a) + 127) / 255) as u8)
        })
        .collect();
    Ok(RgbImage {
        width,
        height,
        pixels,
    })
}

#[derive(Debug, Serialize)]
pub struct Axes {
    pub x: &'static str,
    pub y: &'static str,
    pub z: &'static str,
}

pub const OPENCV_AXES: Axes = Axes {
    x: "right",
    y: "down",
    z: "forward",
};

#[derive(Serialize)]
struct GaussianFormat {
    file: &'static str,
    opacity: &'static str,
    scale: &'static str,
    rotation: &'static str,
    sh_degree: u32,
    color: &'static str,
}

#[derive(Serialize)]
struct PriorsUsed {
    camera_poses: bool,
    intrinsics: bool,
    depth: bool,
}

#[derive(Serialize)]
struct Timings {
    load_and_preprocess_s: f64,
    forward_s: f64,
    write_s: f64,
}

/// `meta.json`: conventions, model and settings of a run.
#[derive(Serialize)]
struct Meta {
    model: &'static str,
    checkpoint: String,
    engine: String,
    quantization: &'static str,
    trunk_dtype: String,
    heads_dtype: &'static str,
    camera_convention: &'static str,
    camera_axes: Axes,
    extrinsics: &'static str,
    world_frame: &'static str,
    world_up: &'static str,
    handedness: &'static str,
    scale: &'static str,
    intrinsics: &'static str,
    inputs: Vec<String>,
    priors: PriorsUsed,
    video: Option<VideoSource>,
    frames: usize,
    inference_width: usize,
    inference_height: usize,
    target_size: usize,
    settings: OutputSettings,
    gaussians: GaussianFormat,
    written: WrittenCounts,
    timings: Timings,
    differences_from_reference: Vec<&'static str>,
}

/// What a run wrote, for the caller's report.
#[derive(Debug)]
pub struct RunReport {
    pub frames: usize,
    pub width: usize,
    pub height: usize,
    pub written: WrittenCounts,
    pub forward_seconds: f64,
    pub total_seconds: f64,
}

/// Reconstructs from the images at `input` into `out_dir` (created), with `model` loaded from `checkpoint`.
pub fn reconstruct(
    model: &WorldMirror,
    checkpoint: &Path,
    input: &Path,
    out_dir: &Path,
    options: &ReconstructOptions,
) -> Result<RunReport> {
    let files = input_images(input)?;
    if files.is_empty() {
        inference_tensor::bail!(
            "{} holds no images ({})",
            input.display(),
            IMAGE_EXTENSIONS.join(", ")
        );
    }
    if files.len() > MAX_FRAMES {
        inference_tensor::bail!(
            "{} images given; WorldMirror takes at most {MAX_FRAMES} (subsample them first)",
            files.len()
        );
    }
    let frames = files
        .iter()
        .map(|f| Ok((f.display().to_string(), decode(f)?)))
        .collect::<Result<Vec<_>>>()?;
    reconstruct_frames(model, checkpoint, frames, None, out_dir, options)
}

/// Each frame's z-depth from the depth head alone, inferred at `target_size` with `priors` and resized (nearest) back
/// to the frame's size, all frames sharing the first one's size.
pub fn depth_with_priors(
    model: &WorldMirror,
    images: &[RgbImage],
    priors: &PriorInputs,
    target_size: usize,
) -> Result<Vec<Vec<f32>>> {
    let Some(first) = images.first() else {
        inference_tensor::bail!("no frames to infer depth for");
    };
    let (w, h) = (first.width, first.height);
    let tensor = images_to_tensor(images, target_size, PATCH, model.device())?;
    let depth = model
        .predict_depth_with_priors(&tensor, priors, (w, h))?
        .to_dtype(inference_tensor::DType::F32)?
        .to_device(&inference_tensor::Device::Cpu)?;
    let (frames, dh, dw, _) = depth.dims4()?;
    let values = depth.flatten_all()?.to_vec1::<f32>()?;
    Ok((0..frames)
        .map(|f| {
            let plane = &values[f * dh * dw..(f + 1) * dh * dw];
            // cv2.INTER_NEAREST: source index floor(dst * src / dst_size)
            (0..w * h)
                .map(|p| {
                    let (x, y) = (p % w, p / w);
                    plane[(y * dh / h).min(dh - 1) * dw + (x * dw / w).min(dw - 1)]
                })
                .collect()
        })
        .collect())
}

/// Where a run's frames came from when they were sampled from a video.
#[derive(Clone, Debug, Serialize)]
pub struct VideoSource {
    pub path: String,
    pub native_fps: f64,
    pub sampling_fps: f64,
    pub total_frames: usize,
    pub sampled_frames: Vec<usize>,
}

/// Reconstructs from decoded frames, each with the name `meta.json` lists it under.
pub fn reconstruct_frames(
    model: &WorldMirror,
    checkpoint: &Path,
    frames: Vec<(String, RgbImage)>,
    video: Option<VideoSource>,
    out_dir: &Path,
    options: &ReconstructOptions,
) -> Result<RunReport> {
    if frames.len() > MAX_FRAMES {
        inference_tensor::bail!(
            "{} frames given; WorldMirror takes at most {MAX_FRAMES}",
            frames.len()
        );
    }
    let stems = frames
        .iter()
        .map(|(n, _)| {
            Path::new(n)
                .file_stem()
                .map_or_else(|| n.clone(), |s| s.to_string_lossy().into_owned())
        })
        .collect::<Vec<_>>();
    let priors = load_prior_files(
        options.prior_cameras.as_deref(),
        options.prior_depth.as_deref(),
        &stems,
    )?;
    reconstruct_with_priors(model, checkpoint, frames, video, priors, out_dir, options)
}

/// As [`reconstruct_frames`] with the priors given directly and no frame cap (the caller bounds the count).
pub fn reconstruct_with_priors(
    model: &WorldMirror,
    checkpoint: &Path,
    frames: Vec<(String, RgbImage)>,
    video: Option<VideoSource>,
    priors: PriorInputs,
    out_dir: &Path,
    options: &ReconstructOptions,
) -> Result<RunReport> {
    let start = Instant::now();
    if frames.is_empty() {
        inference_tensor::bail!("no frames to reconstruct from");
    }
    let (names, images): (Vec<String>, Vec<RgbImage>) = frames.into_iter().unzip();
    let size = adaptive_size(
        (images[0].width, images[0].height),
        options.target_size,
        PATCH,
    );
    for (name, image) in names.iter().zip(&images) {
        let (w, h) = resized_dims(image.width, image.height, size, PATCH);
        if w < 2 * PATCH || h < 2 * PATCH {
            inference_tensor::bail!(
                "{name} ({}x{}) resizes to {w}x{h}, below the {}-pixel minimum side",
                image.width,
                image.height,
                2 * PATCH
            );
        }
    }
    let tensor = images_to_tensor(&images, options.target_size, PATCH, model.device())?;
    let loaded = start.elapsed().as_secs_f64();
    let forward_start = Instant::now();
    let pred = model.predict_with_priors(&tensor, &priors, (images[0].width, images[0].height))?;
    model.device().synchronize()?;
    let forward = forward_start.elapsed().as_secs_f64();
    let write_start = Instant::now();
    let written = write_outputs(out_dir, &pred, &tensor, &options.output)?;
    let meta = Meta {
        model: MODEL_NAME,
        checkpoint: checkpoint.display().to_string(),
        engine: format!("inference.rs {}", env!("CARGO_PKG_VERSION")),
        quantization: "none (f32 weights; trunk matmuls in the trunk dtype)",
        trunk_dtype: format!("{:?}", model.trunk_dtype()),
        heads_dtype: "F32",
        camera_convention: "opencv",
        camera_axes: OPENCV_AXES,
        extrinsics: "camera_to_world",
        world_frame: "the first camera's frame (its pose is the identity), also when camera priors are given",
        world_up: "not gravity aligned; approximately -y of the first camera",
        handedness: "right",
        scale: "arbitrary (up to scale; depth, poses and splats share one unit)",
        intrinsics: "pixels of the inference image (inference_width x inference_height), principal point at its centre",
        inputs: names,
        priors: PriorsUsed {
            camera_poses: priors.camera_to_world.is_some(),
            intrinsics: priors.intrinsics.is_some(),
            depth: priors.depths.is_some(),
        },
        video,
        frames: images.len(),
        inference_width: pred.width,
        inference_height: pred.height,
        target_size: options.target_size,
        settings: options.output.clone(),
        gaussians: GaussianFormat {
            file: "gaussians.ply, standard 3DGS binary PLY",
            opacity: "logit (apply sigmoid)",
            scale: "log (apply exp)",
            rotation: "unit quaternion (w, x, y, z) in rot_0..rot_3",
            sh_degree: 0,
            color: "SH DC coefficients f_dc_0..2 (rgb = 0.5 + 0.28209479 * f_dc)",
        },
        written: written.clone(),
        timings: Timings {
            load_and_preprocess_s: loaded,
            forward_s: forward,
            write_s: write_start.elapsed().as_secs_f64(),
        },
        differences_from_reference: vec![
            "gaussians.ply opacity is a logit; the reference writes the activated value in [0, 1]",
            "the sky mask is the model's own (gs depth mask > threshold); the reference also ANDs a skyseg ONNX mask",
            "over-cap subsampling of points and splats uses its own seeded sampler, not NumPy's",
            "JPEG decoding by the image crate differs from libjpeg-turbo by a few levels in some pixels",
        ],
    };
    let text = serde_json::to_string_pretty(&meta).map_err(inference_tensor::Error::wrap)?;
    std::fs::write(out_dir.join("meta.json"), text)?;
    Ok(RunReport {
        frames: images.len(),
        width: pred.width,
        height: pred.height,
        written,
        forward_seconds: forward,
        total_seconds: start.elapsed().as_secs_f64(),
    })
}
