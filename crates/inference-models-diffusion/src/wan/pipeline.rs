//! One WorldStereo-2 clip: 21 keyframes along a camera path, from the start frame, point-cloud renders, cameras, memory
//! frames and a caption, as the reference's DMD keyframe pipeline runs it.

use inference_tensor::{DType, Device, Result, Tensor};

use super::clip::WanImageEncoder;
use super::sampler::DmdSchedule;
use super::transformer::{StepInputs, WorldStereoTransformer};
use super::vae::WanVae;

// Channels of the first-frame mask in the I2V condition (the VAE's temporal compression)
const MASK_CHANNELS: usize = 4;
const LATENT_CHANNELS: usize = 16;
// The VAE's spatial downsampling
const SPATIAL_SCALE: usize = 8;

/// Memory frames retrieved from earlier clips.
pub struct MemoryFrames<'a> {
    /// `[R, 3, H, W]` in [-1, 1]
    pub frames: &'a Tensor,
    /// `[R, 7]` quaternion and translation of each, normalized with the clip's cameras
    pub camera_qt: &'a Tensor,
    /// the clip frame (0..20) each memory frame sits beside
    pub index: &'a [usize],
}

pub struct ClipRequest<'a> {
    /// `[3, H, W]` in [-1, 1] at the bucket size, and its 8-bit RGB pixels for the image encoder
    pub start: &'a Tensor,
    pub start_rgb: &'a [u8],
    /// `[F, 3, H, W]` point-cloud renders in [-1, 1] (frame 0 the start image) and `[F, 1, H, W]` their 0/1 masks
    pub renders: &'a Tensor,
    pub render_mask: &'a Tensor,
    /// `[6, F, H, W]` per-pixel ray embedding and `[F, 7]` per-frame quaternion and translation
    pub ray_embedding: &'a Tensor,
    pub camera_qt: &'a Tensor,
    pub memory: Option<MemoryFrames<'a>>,
    /// `[1, 512, 4096]` umT5 states of the caption
    pub text_states: &'a Tensor,
    /// `[16, F, H/8, W/8]` starting noise; drawn from `seed` when absent
    pub noise: Option<&'a Tensor>,
    pub seed: u64,
}

// Returns the memory pool's free blocks to the driver between phases whose allocations differ in shape (the
// transformer's few large tensors, the VAE's many full-resolution ones): otherwise each phase's cached blocks sit
// beside the next phase's new ones, ~8 GB over a clip
fn trim_pool(dev: &Device) -> Result<()> {
    #[cfg(feature = "cuda")]
    if dev.is_cuda() {
        dev.synchronize()?;
        inference_nn::utils::memory_usage::MemoryUsage.trim_cuda_memory_pool(dev, 0)?;
    }
    let _ = dev;
    Ok(())
}

pub struct WorldStereoPipeline {
    pub vae: WanVae,
    pub image_encoder: WanImageEncoder,
    pub transformer: WorldStereoTransformer,
    pub schedule: DmdSchedule,
    pub device: Device,
}

impl WorldStereoPipeline {
    /// The clip's keyframes `[F, 3, H, W]` in [-1, 1].
    pub fn generate(&self, req: &ClipRequest) -> Result<Tensor> {
        let latents = self.denoise(req)?;
        trim_pool(&self.device)?;
        let frames = self.vae.decode(&latents.permute((1, 0, 2, 3))?)?;
        trim_pool(&self.device)?;
        Ok(frames)
    }

    /// The clip's final latents `[16, F, H/8, W/8]` (normalized).
    pub fn denoise(&self, req: &ClipRequest) -> Result<Tensor> {
        self.denoise_observed(req, &mut |_, _| {})
    }

    /// As [`Self::denoise`], handing `observe` the latents after each step.
    pub fn denoise_observed(
        &self,
        req: &ClipRequest,
        observe: &mut dyn FnMut(usize, &Tensor),
    ) -> Result<Tensor> {
        let dev = &self.device;
        let (f, _, height, width) = req.renders.dims4()?;
        let (h, w) = (height / SPATIAL_SCALE, width / SPATIAL_SCALE);
        let image_states = self
            .image_encoder
            .forward(
                &self
                    .image_encoder
                    .pixel_values(req.start_rgb, (width, height), dev)?,
            )?;
        // I2V condition: the start frame's latent for frame 0, the zero image's for the rest, behind a frame-0 mask
        let first = self.vae.encode(&req.start.unsqueeze(0)?)?;
        let zero = self.vae.encode(&req.start.zeros_like()?.unsqueeze(0)?)?;
        let latent_condition =
            Tensor::cat(&[first, zero.repeat((f - 1, 1, 1, 1))?], 0)?.permute((1, 0, 2, 3))?;
        let mask = Tensor::cat(
            &[
                Tensor::ones((MASK_CHANNELS, 1, h, w), DType::F32, dev)?,
                Tensor::zeros((MASK_CHANNELS, f - 1, h, w), DType::F32, dev)?,
            ],
            1,
        )?;
        let condition = Tensor::cat(&[&mask, &latent_condition], 0)?.to_dtype(DType::BF16)?;
        let render_latent = self
            .vae
            .encode(req.renders)?
            .permute((1, 0, 2, 3))?
            .contiguous()?;
        let render_conditions = Tensor::cat(
            &[&req.render_mask.permute((1, 0, 2, 3))?, req.ray_embedding],
            0,
        )?
        .to_dtype(DType::BF16)?;
        let memory = match &req.memory {
            Some(m) => {
                let z = self.vae.encode(m.frames)?.permute((1, 0, 2, 3))?;
                let ones = Tensor::ones((MASK_CHANNELS, z.dim(1)?, h, w), DType::F32, dev)?;
                Some(Tensor::cat(&[&z, &ones, &z], 0)?.to_dtype(DType::BF16)?)
            }
            None => None,
        };
        trim_pool(dev)?;
        let qt = req.camera_qt.to_dtype(DType::BF16)?;
        let qt_ref = req
            .memory
            .as_ref()
            .map(|m| m.camera_qt.to_dtype(DType::BF16))
            .transpose()?;
        let mut latents = match req.noise {
            Some(n) => n.to_dtype(DType::F32)?,
            None => {
                crate::noise::SeededNoise::new(req.seed).normal((LATENT_CHANNELS, f, h, w), dev)?
            }
        };
        let text = req.text_states.to_dtype(DType::BF16)?;
        let image = image_states.to_dtype(DType::BF16)?;
        let timesteps = self.schedule.timesteps();
        for (i, &t) in timesteps.iter().enumerate() {
            let cond = self.transformer.conditions(t, &text, &image)?;
            let input = Tensor::cat(&[&latents.to_dtype(DType::BF16)?, &condition], 0)?;
            let inputs = StepInputs {
                latents: &input,
                render_latent: &render_latent,
                render_conditions: &render_conditions,
                camera_qt: &qt,
                camera_qt_ref: qt_ref.as_ref(),
                reference_latent: memory.as_ref(),
                ref_index: req.memory.as_ref().map_or(&[], |m| m.index),
            };
            let v = self.transformer.forward(&inputs, &cond)?;
            latents = self.schedule.step(&timesteps, i, &latents, &v)?;
            observe(i, &latents);
        }
        Ok(latents)
    }
}
