//! The Wan I2V image condition: an OpenCLIP ViT-H/14 vision tower's penultimate hidden states for the start frame.

use inference_nn::vision::clip::{ClipConfig, ClipVisionTransformer};
use inference_quant::ShardedVarBuilder;
use inference_tensor::{DType, Device, Result, Tensor};

const IMAGE_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
const IMAGE_STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];
// The pipeline reads hidden_states[-2]: the output of the next-to-last layer
const LAYERS_FROM_END: usize = 2;

pub struct WanImageEncoder {
    tower: ClipVisionTransformer,
    image_size: usize,
    layer: usize,
    dtype: DType,
}

impl WanImageEncoder {
    /// From the diffusers `image_encoder` folder's config and weights (`vision_model.*`), the tower run in `dtype`.
    pub fn new(vb: ShardedVarBuilder, cfg: &ClipConfig, dtype: DType) -> Result<Self> {
        Ok(Self {
            tower: ClipVisionTransformer::new(vb.pp("vision_model").set_dtype(dtype), cfg)?,
            image_size: cfg.image_size,
            layer: cfg.num_hidden_layers - LAYERS_FROM_END,
            dtype,
        })
    }

    /// The tower's input for an 8-bit RGB image: PIL bicubic to the square input size, then normalized as the WorldStereo
    /// pipeline's processor does, which skips the 1/255 rescale (it builds it with `do_rescale=False`), so the
    /// tower sees raw 0..255 values normalized as if they were 0..1. The model was tuned on exactly this input.
    pub fn pixel_values(
        &self,
        pixels: &[u8],
        size: (usize, usize),
        device: &Device,
    ) -> Result<Tensor> {
        let s = self.image_size;
        let resized = inference_vision::pil_resize_bicubic(pixels, size, (s, s));
        let mut chw = vec![0f32; 3 * s * s];
        for (p, rgb) in resized.as_chunks::<3>().0.iter().enumerate() {
            for c in 0..3 {
                chw[c * s * s + p] = (f32::from(rgb[c]) - IMAGE_MEAN[c]) / IMAGE_STD[c];
            }
        }
        Tensor::from_vec(chw, (1, 3, s, s), device)
    }

    /// Penultimate hidden states `[b, tokens, hidden]` (f32) of pixel values `[b, 3, s, s]`.
    pub fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let states = self
            .tower
            .forward_get_hidden_states(&pixel_values.to_dtype(self.dtype)?)?;
        states[self.layer].to_dtype(DType::F32)
    }
}
