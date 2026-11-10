//! BiRefNet (ZhengPeng7/BiRefNet, MIT): dichotomous image segmentation, used here to cut an object out of its
//! background. A Swin-L encoder runs on the image and on it at half size, its four scales concatenated, then a
//! deformable-ASPP decoder predicts the foreground mask at 1024^2.

mod decoder;
#[cfg(test)]
mod parity;
mod swin;

use std::path::Path;

use inference_tensor::nn::VarBuilder;
use inference_tensor::{DType, Device, Result, Tensor};
use inference_vision::PilFilter;

use decoder::{DecBlock, Decoder};
use swin::{SWIN_L, Swin};

/// The size the model sees, as the reference's `transforms.Resize((1024, 1024))`.
pub const INPUT_SIZE: usize = 1024;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];
// the encoder's widths, deepest first, each doubled by the half-size pass
const CHANNELS: [usize; 4] = [3072, 1536, 768, 384];
// the shallower scales concatenated onto the deepest before the squeeze block
const CONTEXT: usize = 3;

pub struct BiRefNet {
    encoder: Swin,
    squeeze: DecBlock,
    decoder: Decoder,
    device: Device,
}

impl BiRefNet {
    /// Loads `dir`'s `model.safetensors` onto `device`; it runs in f32.
    pub fn load(dir: &Path, device: &Device) -> Result<Self> {
        // SAFETY: the checkpoint file is not modified while the model is loaded
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[dir.join("model.safetensors")],
                DType::F32,
                device,
            )?
        };
        let context: usize = CHANNELS[1..].iter().sum();
        Ok(Self {
            encoder: Swin::load(SWIN_L, vb.pp("bb"))?,
            squeeze: DecBlock::load(
                CHANNELS[0] + context,
                CHANNELS[0],
                vb.pp("squeeze_module").pp(0),
            )?,
            decoder: Decoder::load(CHANNELS, vb.pp("decoder"))?,
            device: device.clone(),
        })
    }

    // the encoder at full and half size, each scale's two concatenated, the shallower three added to the deepest
    fn encode(&self, x: &Tensor) -> Result<[Tensor; 4]> {
        let (_, _, h, w) = x.dims4()?;
        let full = self.encoder.forward(x)?;
        let half = self
            .encoder
            .forward(&x.upsample_bilinear2d(h / 2, w / 2, true)?)?;
        let feats = full
            .iter()
            .zip(&half)
            .map(|(f, h)| {
                let (_, _, fh, fw) = f.dims4()?;
                Tensor::cat(&[f, &h.upsample_bilinear2d(fh, fw, true)?], 1)
            })
            .collect::<Result<Vec<_>>>()?;
        let (_, _, h4, w4) = feats[3].dims4()?;
        let mut deepest = feats[3 - CONTEXT..3]
            .iter()
            .map(|f| f.upsample_bilinear2d(h4, w4, true))
            .collect::<Result<Vec<_>>>()?;
        deepest.push(feats[3].clone());
        let x4 = Tensor::cat(&deepest, 1)?;
        let [x1, x2, x3, _]: [Tensor; 4] = feats.try_into().expect("four scales");
        Ok([x1, x2, x3, x4])
    }

    /// The mask logits `[B, 1, H, W]` for normalized images `[B, 3, H, W]` (sides a multiple of 32).
    pub fn logits(&self, images: &Tensor) -> Result<Tensor> {
        let [x1, x2, x3, x4] = self.encode(images)?;
        let x4 = self.squeeze.forward(&x4)?;
        self.decoder.forward(images, [&x1, &x2, &x3, &x4])
    }

    /// The foreground mask of an RGB image (three bytes a pixel, `width x height`) at its own size, as the reference
    /// makes it: PIL bilinear to 1024^2, sigmoid, truncated to 8 bits, PIL bicubic back.
    pub fn mask(&self, rgb: &[u8], (width, height): (usize, usize)) -> Result<Vec<u8>> {
        let resized = inference_vision::pil_resize(
            rgb,
            3,
            (width, height),
            (INPUT_SIZE, INPUT_SIZE),
            PilFilter::Bilinear,
        );
        let plane = INPUT_SIZE * INPUT_SIZE;
        let mut data = vec![0f32; 3 * plane];
        for (i, px) in resized.as_chunks::<3>().0.iter().enumerate() {
            for c in 0..3 {
                data[c * plane + i] = (f32::from(px[c]) / 255.0 - MEAN[c]) / STD[c];
            }
        }
        let images = Tensor::from_vec(data, (1, 3, INPUT_SIZE, INPUT_SIZE), &self.device)?;
        let probs = inference_tensor::nn::ops::sigmoid(&self.logits(&images)?)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mask: Vec<u8> = probs.iter().map(|p| (p * 255.0) as u8).collect();
        Ok(inference_vision::pil_resize(
            &mask,
            1,
            (INPUT_SIZE, INPUT_SIZE),
            (width, height),
            PilFilter::Bicubic,
        ))
    }

    /// The image with the mask as its alpha (four bytes a pixel).
    pub fn cut_out(&self, rgb: &[u8], size: (usize, usize)) -> Result<Vec<u8>> {
        let mask = self.mask(rgb, size)?;
        Ok(rgb
            .as_chunks::<3>()
            .0
            .iter()
            .zip(&mask)
            .flat_map(|(px, &a)| [px[0], px[1], px[2], a])
            .collect())
    }
}
