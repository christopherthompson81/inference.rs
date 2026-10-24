//! Feed-forward 3D reconstruction (WorldMirror 2.0): image files in, cameras, depth, normals, a point cloud and
//! Gaussian splats out, with a `meta.json` describing conventions and settings.

use std::path::Path;

use inference_tensor::{DType, Device};

pub use inference_recon::worldmirror::output::OutputSettings;
pub use inference_recon::worldmirror::run::{
    DEFAULT_TARGET_SIZE, MAX_FRAMES, ReconstructOptions, RunReport,
};

/// Loads the WorldMirror checkpoint in `checkpoint` (a folder with `config.json` and `model.safetensors`) and
/// reconstructs the images at `input` (a folder, or one image) into `out_dir`. Runs on the first CUDA device unless
/// `cpu` is set; the trunk runs in bf16 on CUDA, as the reference does, and in f32 on the CPU.
pub fn reconstruct(
    checkpoint: &Path,
    input: &Path,
    out_dir: &Path,
    options: &ReconstructOptions,
    cpu: bool,
) -> anyhow::Result<RunReport> {
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
    Ok(inference_recon::worldmirror::run::reconstruct(
        &model, checkpoint, input, out_dir, options,
    )?)
}
