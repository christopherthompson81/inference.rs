//! Feed-forward 3D reconstruction (WorldMirror 2.0): images or a video in, cameras, depth, normals, a point cloud and
//! Gaussian splats out, with a `meta.json` describing conventions and settings.

use std::path::Path;

use inference_core::VideoFrameSampling;
use inference_recon::worldmirror::preprocess::RgbImage;
use inference_recon::worldmirror::run::{VideoSource, reconstruct_frames};
use inference_tensor::{DType, Device};

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
        let decoded = crate::video::parse_video_url(&path, Some(sampling)).await?;
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
    let device = if cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };
    let dtype = if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };
    let model = inference_recon::worldmirror::WorldMirror::load(checkpoint, dtype, &device)?;
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
