//! Tencent's WorldMirror 2.0: feed-forward reconstruction of cameras, depth, normals, points and Gaussian splats from a
//! set of images or video frames.

mod camera;
mod dpt;
mod gaussians;
pub(crate) mod layers;
pub mod output;
pub mod preprocess;
pub mod priors;
mod rope;
pub mod run;
mod trunk;

use std::path::Path;

use inference_tensor::nn::VarBuilder;
use inference_tensor::{DType, Device, Result, Tensor};

pub use camera::{Camera, decode_cameras};
pub use gaussians::{Splats, unproject};

// The camera head's transformer depth, heads and refinement steps
const CAMERA_DEPTH: usize = 4;
const CAMERA_HEADS: usize = 16;
const CAMERA_STEPS: usize = 4;
// ImageNet statistics the trunk normalizes its [0, 1] input with
const IMAGE_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGE_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// The fields of the checkpoint's `config.json` the inference path reads.
#[derive(Clone, Debug, inference_json::FromJson)]
pub struct WorldMirrorConfig {
    pub embed_dim: usize,
    pub depth: usize,
    pub num_heads: usize,
    pub patch_size: usize,
    pub num_register_tokens: usize,
    pub rope_base: f64,
    pub gs_dim: usize,
}

impl WorldMirrorConfig {
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(dir.join("config.json"))?;
        serde_json::from_str(&text).map_err(inference_tensor::Error::wrap)
    }
}

pub struct WorldMirror {
    trunk: trunk::Trunk,
    camera: camera::CameraHead,
    depth: dpt::DptHead,
    points: dpt::DptHead,
    normals: dpt::DptHead,
    gs_depth: dpt::DptHead,
    gaussians: gaussians::GaussianParams,
    trunk_dtype: DType,
    device: Device,
}

/// What one forward predicts for `frames` images of `height x width`; per-pixel maps are `[frames, H, W, ...]`.
pub struct Prediction {
    /// `[frames, 9]`: translation, quaternion (scalar last), vertical and horizontal field of view (world-to-camera).
    pub camera_params: Tensor,
    pub cameras: Vec<Camera>,
    /// z-depth `[frames, H, W, 1]`, its confidence and the validity mask's logits `[frames, H, W]`
    pub depth: Tensor,
    pub depth_conf: Tensor,
    pub depth_mask_logits: Tensor,
    /// world points `[frames, H, W, 3]` from the point head
    pub points: Tensor,
    /// unit normals in camera coordinates `[frames, H, W, 3]`
    pub normals: Tensor,
    /// the Gaussian head's own depth, which places the splats, and its mask logits
    pub gs_depth: Tensor,
    pub gs_depth_mask_logits: Tensor,
    pub splats: Splats,
    pub height: usize,
    pub width: usize,
}

impl WorldMirror {
    /// Loads `dir`'s `model.safetensors` onto `device`; the trunk's matmuls run in `trunk_dtype` (bf16 matches the
    /// reference, whose trunk runs under bf16 autocast).
    pub fn load(dir: &Path, trunk_dtype: DType, device: &Device) -> Result<Self> {
        let cfg = WorldMirrorConfig::from_dir(dir)?;
        // SAFETY: the checkpoint file is not modified while the model is loaded
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[dir.join("model.safetensors")],
                DType::F32,
                device,
            )?
        };
        let (dim, patch) = (2 * cfg.embed_dim, cfg.patch_size);
        Ok(Self {
            trunk: trunk::Trunk::load(&cfg, trunk_dtype, vb.pp("visual_geometry_transformer"))?,
            camera: camera::CameraHead::load(
                2 * cfg.embed_dim,
                CAMERA_HEADS,
                CAMERA_DEPTH,
                CAMERA_STEPS,
                vb.pp("cam_head"),
            )?,
            depth: dpt::DptHead::load(
                dim,
                patch,
                dpt::Activation::DepthWithMask,
                false,
                vb.pp("depth_head"),
            )?,
            points: dpt::DptHead::load(
                dim,
                patch,
                dpt::Activation::Points,
                false,
                vb.pp("pts_head"),
            )?,
            normals: dpt::DptHead::load(
                dim,
                patch,
                dpt::Activation::Normals,
                false,
                vb.pp("norm_head"),
            )?,
            gs_depth: dpt::DptHead::load(
                dim,
                patch,
                dpt::Activation::DepthWithMask,
                true,
                vb.pp("gs_head"),
            )?,
            gaussians: gaussians::GaussianParams::load(dpt::HEAD_FEATURES, vb.pp("gs_renderer"))?,
            trunk_dtype,
            device: device.clone(),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn trunk_dtype(&self) -> DType {
        self.trunk_dtype
    }

    /// Predicts everything for images `[frames, 3, H, W]` in [0, 1].
    pub fn predict(&self, images: &Tensor) -> Result<Prediction> {
        self.predict_with_priors(
            images,
            &priors::PriorInputs::default(),
            (images.dim(3)?, images.dim(2)?),
        )
    }

    /// `predict` with known poses, intrinsics or depth for every frame; `original` is the first image's size before
    /// resizing (its intrinsics are in its pixels).
    pub fn predict_with_priors(
        &self,
        images: &Tensor,
        inputs: &priors::PriorInputs,
        original: (usize, usize),
    ) -> Result<Prediction> {
        let images = images.to_device(&self.device)?.to_dtype(DType::F32)?;
        let (_, _, height, width) = images.dims4()?;
        let priors = priors::normalize(inputs, original, height, width, &self.device)?;
        let out = self.trunk(&images, &priors, &mut |_, _, _| {})?;
        self.heads(&out.features, out.patch_start, &images)
    }

    /// Z-depth `[frames, H, W, 1]` alone, with known priors: the trunk, then only the depth head (the reference's
    /// worldgen runs WorldMirror so, its normal, point and Gaussian heads off, to align each frame's depth).
    pub fn predict_depth_with_priors(
        &self,
        images: &Tensor,
        inputs: &priors::PriorInputs,
        original: (usize, usize),
    ) -> Result<Tensor> {
        let images = images.to_device(&self.device)?.to_dtype(DType::F32)?;
        let (_, _, height, width) = images.dims4()?;
        let priors = priors::normalize(inputs, original, height, width, &self.device)?;
        let out = self.trunk(&images, &priors, &mut |_, _, _| {})?;
        // the trunk's and each chunk's transient shapes differ; cached beside each other they near fill a 24 GB card
        trim_pool(&self.device)?;
        let trim = &mut |_: usize, _: &dpt::DptOutput| trim_pool(&self.device);
        Ok(self
            .depth
            .forward(&out.features, out.patch_start, &images, trim)?
            .attr)
    }

    // every head over the trunk's four feature maps `[frames, tokens, 2 * dim]` and the [0, 1] images on the device
    fn heads(&self, features: &[Tensor], start: usize, images: &Tensor) -> Result<Prediction> {
        let (_, _, height, width) = images.dims4()?;
        let last = features
            .last()
            .expect("the trunk returns its head features");
        let camera_params = self.camera.forward(&last.to_device(images.device())?)?;
        let cameras = decode_cameras(&camera_params, height, width)?;
        let images = images.clone();
        let none = &mut |_: usize, _: &dpt::DptOutput| Ok(());
        let depth = self.depth.forward(features, start, &images, none)?;
        let points = self.points.forward(features, start, &images, none)?;
        let normals = self.normals.forward(features, start, &images, none)?;
        // the splats a chunk of frames at a time, from the Gaussian head's fused features of that chunk
        let mut splats = Vec::new();
        let gs = self
            .gs_depth
            .forward(features, start, &images, &mut |first, chunk| {
                let fused = chunk
                    .fused
                    .as_ref()
                    .expect("a chunk carries its fused features");
                let frames = fused.dim(0)?;
                splats.push(self.gaussians.forward(
                    fused,
                    &images.narrow(0, first, frames)?,
                    &chunk.attr,
                    &cameras[first..first + frames],
                )?);
                Ok(())
            })?;
        let splats = Splats::cat(&splats)?;
        Ok(Prediction {
            camera_params,
            cameras,
            depth: depth.attr,
            depth_conf: depth.conf,
            depth_mask_logits: depth.mask_logits.expect("the depth head predicts a mask"),
            points: points.attr,
            normals: normals.attr,
            gs_depth: gs.attr,
            gs_depth_mask_logits: gs.mask_logits.expect("the Gaussian head predicts a mask"),
            splats,
            height,
            width,
        })
    }

    // the trunk's head features for images `[frames, 3, H, W]` in [0, 1], each layer's outputs to `inspect`
    fn trunk(
        &self,
        images: &Tensor,
        priors: &priors::Priors,
        inspect: &mut dyn FnMut(usize, &Tensor, &Tensor),
    ) -> Result<trunk::TrunkOutput> {
        self.trunk.forward_inspect(
            &normalize(&images.to_device(&self.device)?)?,
            priors,
            inspect,
        )
    }
}

// ImageNet normalization of [0, 1] images `[frames, 3, H, W]`
fn normalize(images: &Tensor) -> Result<Tensor> {
    let device = images.device();
    let mean = Tensor::new(&IMAGE_MEAN, device)?.reshape((1, 3, 1, 1))?;
    let std = Tensor::new(&IMAGE_STD, device)?.reshape((1, 3, 1, 1))?;
    images
        .to_dtype(DType::F32)?
        .broadcast_sub(&mean)?
        .broadcast_div(&std)
}

// returns the CUDA pool's free blocks to the device once the queued work is done
fn trim_pool(device: &Device) -> Result<()> {
    device.synchronize()?;
    #[cfg(feature = "cuda")]
    inference_nn::utils::memory_usage::MemoryUsage.trim_cuda_memory_pool(device, 0)?;
    Ok(())
}

#[cfg(test)]
mod parity;
