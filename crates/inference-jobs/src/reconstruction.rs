//! 3D reconstruction and world generation: WorldMirror 2.0 (images or a video in; cameras, depth, normals, a point
//! cloud and Gaussian splats out) and WorldStereo-2 (a panorama in, generated keyframes and a reconstruction out), each
//! run writing a `meta.json`.

use std::path::Path;

use inference_api::VideoFrameSampling;
use inference_recon::worldmirror::preprocess::RgbImage;
use inference_recon::worldmirror::run::{VideoSource, reconstruct_frames};
use inference_tensor::{DType, Device};

pub use inference_recon::progress::{CANCELLED, NoProgress, Progress};
pub use inference_recon::worldmirror::output::OutputSettings;
pub use inference_recon::worldmirror::run::{
    DEFAULT_TARGET_SIZE, MAX_FRAMES, ReconstructOptions, RunReport,
};

/// Frames a second sampled from a video by default, as the reference samples
pub const DEFAULT_VIDEO_FPS: f64 = 1.0;
/// The most frames sampled from a video by default, as the reference
pub const DEFAULT_VIDEO_MAX_FRAMES: usize = 32;
// Files read as video rather than as an image
const VIDEO_EXTENSIONS: [&str; 7] = ["mp4", "m4v", "mov", "avi", "mkv", "webm", "gif"];

/// How frames are sampled from a video input.
#[derive(Clone, Copy, Debug)]
pub struct VideoOptions {
    pub fps: f64,
    pub max_frames: usize,
}

impl Default for VideoOptions {
    fn default() -> Self {
        Self {
            fps: DEFAULT_VIDEO_FPS,
            max_frames: DEFAULT_VIDEO_MAX_FRAMES,
        }
    }
}

fn is_video(path: &Path) -> bool {
    path.is_file()
        && path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| VIDEO_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
}

/// Loads the WorldMirror checkpoint in `checkpoint` (a folder with `config.json` and `model.safetensors`) and
/// reconstructs `input` into `out_dir`: a folder of images, one image, or a video sampled at `video.fps` up to
/// `video.max_frames` frames (FFmpeg for anything but GIF). Runs on the first CUDA device unless `cpu` is set; the
/// trunk runs in bf16 on CUDA, as the reference does, and in f32 on the CPU.
pub async fn reconstruct(
    checkpoint: &Path,
    input: &Path,
    out_dir: &Path,
    options: &ReconstructOptions,
    video: VideoOptions,
    cpu: bool,
) -> anyhow::Result<RunReport> {
    let device = if cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };
    reconstruct_on(checkpoint, input, out_dir, options, video, &device).await
}

/// [`reconstruct`] on `device`.
pub async fn reconstruct_on(
    checkpoint: &Path,
    input: &Path,
    out_dir: &Path,
    options: &ReconstructOptions,
    video: VideoOptions,
    device: &Device,
) -> anyhow::Result<RunReport> {
    let frames = if is_video(input) {
        if video.max_frames == 0 || video.max_frames > MAX_FRAMES {
            anyhow::bail!(
                "--max-frames must be 1..={MAX_FRAMES}, got {}",
                video.max_frames
            );
        }
        let sampling = VideoFrameSampling::Fps {
            fps: video.fps,
            min_frames: 1,
            max_frames: video.max_frames,
        };
        let path = input.to_string_lossy();
        let bytes = tokio::fs::read(input).await?;
        let attachments = inference_api::media_source::MediaAttachments::new(vec![
            inference_api::media_source::MediaAttachment {
                bytes,
                mime_type: None,
            },
        ]);
        let first = format!("{}0", inference_api::media_source::ATTACHMENT_PREFIX);
        let decoded =
            inference_api::video::parse_video_url(&first, Some(sampling), &attachments).await?;
        let source = VideoSource {
            path: path.to_string(),
            native_fps: decoded.fps,
            sampling_fps: video.fps,
            total_frames: decoded.total_num_frames,
            sampled_frames: decoded.sampled_indices.clone(),
        };
        let frames = decoded
            .frames
            .iter()
            .zip(&decoded.sampled_indices)
            .map(|(frame, index)| {
                let rgb = frame.to_rgb8();
                let image = RgbImage {
                    width: rgb.width() as usize,
                    height: rgb.height() as usize,
                    pixels: rgb.into_raw(),
                };
                (format!("{path}#frame{index}"), image)
            })
            .collect::<Vec<_>>();
        Some((frames, source))
    } else {
        None
    };
    let dtype = if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };
    let model = inference_recon::worldmirror::WorldMirror::load(checkpoint, dtype, device)?;
    let report = match frames {
        Some((frames, source)) => {
            reconstruct_frames(&model, checkpoint, frames, Some(source), out_dir, options)?
        }
        None => inference_recon::worldmirror::run::reconstruct(
            &model, checkpoint, input, out_dir, options,
        )?,
    };
    Ok(report)
}

pub use inference_recon::worldstereo::run::{
    AttentionKind, GgmlDType, WorldGenModels, WorldGenOptions, WorldGenReport,
};

/// Generates a world from an equirectangular panorama (an image file) into `out_dir` with WorldStereo-2: depth and a
/// point cloud from MoGe-2, the regular trajectories, keyframes along each, then WorldMirror 2.0 on the frames. Runs on
/// the first CUDA device.
pub fn generate_world(
    models: &WorldGenModels,
    panorama: &Path,
    out_dir: &Path,
    options: &WorldGenOptions,
    progress: &dyn Progress,
) -> anyhow::Result<WorldGenReport> {
    generate_world_on(
        models,
        panorama,
        out_dir,
        options,
        progress,
        &Device::cuda_if_available(0)?,
    )
}

/// [`generate_world`] on `device`.
pub fn generate_world_on(
    models: &WorldGenModels,
    panorama: &Path,
    out_dir: &Path,
    options: &WorldGenOptions,
    progress: &dyn Progress,
    device: &Device,
) -> anyhow::Result<WorldGenReport> {
    if !device.is_cuda() {
        anyhow::bail!("world generation needs a CUDA device");
    }
    let img = inference_recon::read_image(panorama)
        .map_err(|e| anyhow::anyhow!("{}: {e}", panorama.display()))?
        .to_rgb8();
    let pano = inference_recon::worldstereo::panorama::Rgb8 {
        width: img.width() as usize,
        height: img.height() as usize,
        pixels: img.into_raw(),
    };
    Ok(inference_recon::worldstereo::run::generate_world(
        models, &pano, out_dir, options, device, progress,
    )?)
}

#[cfg(feature = "cuda")]
pub use inference_recon::splatting::train::{
    DEFAULT_STEPS as SPLAT_DEFAULT_STEPS, SplatReport, TrainOptions,
};

/// Trains 3D Gaussian splats on a HY-World `gs_data` directory (images, depths, normals, cameras, the point clouds)
/// and writes `splats.ply` and `splats.json` into `out_dir`. Runs on the first CUDA device.
#[cfg(feature = "cuda")]
pub fn train_splats(
    data_dir: &Path,
    out_dir: &Path,
    options: &TrainOptions,
    progress: &dyn Progress,
) -> anyhow::Result<SplatReport> {
    train_splats_on(
        data_dir,
        out_dir,
        options,
        progress,
        &Device::cuda_if_available(0)?,
    )
}

/// [`train_splats`] on `device`.
#[cfg(feature = "cuda")]
pub fn train_splats_on(
    data_dir: &Path,
    out_dir: &Path,
    options: &TrainOptions,
    progress: &dyn Progress,
    device: &Device,
) -> anyhow::Result<SplatReport> {
    if !device.is_cuda() {
        anyhow::bail!("splat training needs a CUDA device");
    }
    Ok(inference_recon::splatting::train::train_splats(
        data_dir, out_dir, options, device, progress,
    )?)
}
