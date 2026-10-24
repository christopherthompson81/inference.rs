//! Tencent's WorldMirror 2.0: feed-forward reconstruction of cameras, depth, normals, points and Gaussian splats from a
//! set of images or video frames.

mod dinov2;
mod layers;
mod rope;
mod trunk;

use std::path::Path;

use inference_tensor::nn::VarBuilder;
use inference_tensor::{DType, Device, Result, Tensor};
use serde::Deserialize;

// ImageNet statistics the trunk normalizes its [0, 1] input with
const IMAGE_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGE_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// The fields of the checkpoint's `config.json` the inference path reads.
#[derive(Clone, Debug, Deserialize)]
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
    device: Device,
}

impl WorldMirror {
    /// Loads `dir`'s `model.safetensors` onto `device`; the trunk's matmuls run in `trunk_dtype` (bf16 matches the
    /// reference, whose trunk runs under bf16 autocast).
    pub fn load(dir: &Path, trunk_dtype: DType, device: &Device) -> Result<Self> {
        let cfg = WorldMirrorConfig::from_dir(dir)?;
        // SAFETY: the checkpoint file is not modified while the model is loaded
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::F32, device)?
        };
        Ok(Self {
            trunk: trunk::Trunk::load(&cfg, trunk_dtype, vb.pp("visual_geometry_transformer"))?,
            device: device.clone(),
        })
    }

    // the trunk's head features for images `[frames, 3, H, W]` in [0, 1], each layer's outputs to `inspect`
    fn trunk(
        &self,
        images: &Tensor,
        inspect: &mut dyn FnMut(usize, &Tensor, &Tensor),
    ) -> Result<trunk::TrunkOutput> {
        self.trunk
            .forward_inspect(&normalize(&images.to_device(&self.device)?)?, inspect)
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

#[cfg(test)]
mod parity;
