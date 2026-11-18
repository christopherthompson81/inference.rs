//! The image tokens Idol's denoiser attends to: DINOv3 ViT-H+/16's `last_hidden_state` (layer-normed again) and the
//! FLUX.2 VAE's latent of the same image, patchified 2x2 and padded at the front to DINOv3's class and register tokens.

use std::path::Path;

use inference_models_diffusion::flux::autoencoder::Flux2Encoder;
use inference_tensor::nn::VarBuilder;
use inference_tensor::{DType, Device, Result, Tensor};
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};

use crate::dinov3::{Dinov3, Dinov3Spec};

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];
// DINOv3's class token and registers, which the VAE's tokens are padded to line up with
const PREFIX_TOKENS: usize = 5;

/// How the VAE's latent is taken from its distribution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VaeLatent {
    /// A sample, as the reference draws it.
    Sample,
    /// The mean (deterministic: the parity tests).
    #[cfg(test)]
    Mean,
}

pub(crate) struct ImageEncoders {
    dinov3: Dinov3,
    vae: Flux2Encoder,
    dinov3_dtype: DType,
    vae_dtype: DType,
}

impl ImageEncoders {
    /// `dinov3` the ViT-H+/16 checkpoint file, `vae` the FLUX.2 VAE's; the reference runs both in bf16.
    pub(crate) fn load(
        dinov3: &Path,
        vae: &Path,
        dinov3_dtype: DType,
        vae_dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        // SAFETY: the checkpoint files are not modified while the model is loaded
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[dinov3], DType::F32, device)? };
        Ok(Self {
            dinov3: Dinov3::load(Dinov3Spec::vith16plus(dinov3_dtype), vb)?,
            vae: Flux2Encoder::load(vae, vae_dtype, device)?,
            dinov3_dtype,
            vae_dtype,
        })
    }

    /// The VAE's latent mean and standard deviation for `pixels` in [0, 1].
    pub(crate) fn vae_moments(&self, pixels: &Tensor) -> Result<(Tensor, Tensor)> {
        self.vae
            .moments(&((pixels * 2.0)? - 1.0)?.to_dtype(self.vae_dtype)?)
    }

    /// `(feature1, feature2)`: `[1, 4101, 1280]` and `[1, 4101, 128]` f32 for `pixels` `[1, 3, 1024, 1024]` in [0, 1].
    pub(crate) fn encode(
        &self,
        pixels: &Tensor,
        latent: VaeLatent,
        rng: &mut StdRng,
    ) -> Result<(Tensor, Tensor)> {
        let device = pixels.device();
        let mean = Tensor::new(&IMAGENET_MEAN, device)?.reshape((1, 3, 1, 1))?;
        let std = Tensor::new(&IMAGENET_STD, device)?.reshape((1, 3, 1, 1))?;
        let normalized = pixels.broadcast_sub(&mean)?.broadcast_div(&std)?;
        let feature1 = self
            .dinov3
            .forward(&normalized.to_dtype(self.dinov3_dtype)?)?;

        let (mean, std) = self.vae_moments(pixels)?;
        let latents = match latent {
            #[cfg(test)]
            VaeLatent::Mean => mean,
            VaeLatent::Sample => {
                let draws: Vec<f32> = (0..mean.elem_count())
                    .map(|_| StandardNormal.sample(rng))
                    .collect();
                let noise = Tensor::from_vec(draws, mean.shape(), device)?;
                (mean + std.mul(&noise)?)?
            }
        };
        // the reference samples in the VAE's dtype
        let latents = latents.to_dtype(self.vae_dtype)?.to_dtype(DType::F32)?;
        let tokens = self
            .vae
            .normalize(&latents)?
            .flatten_from(2)?
            .transpose(1, 2)?
            .contiguous()?;
        let (b, _, c) = tokens.dims3()?;
        let padding = Tensor::zeros((b, PREFIX_TOKENS, c), DType::F32, device)?;
        Ok((feature1, Tensor::cat(&[&padding, &tokens], 1)?))
    }
}
