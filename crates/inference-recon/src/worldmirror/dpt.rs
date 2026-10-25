//! The DPT heads (depth, points, normals and the Gaussian head's depth): four trunk layers' patch tokens projected to
//! four scales, fused coarse to fine, upsampled to the image and decoded per pixel. They run in f32, as the reference's
//! heads do.

use inference_tensor::nn::VarBuilder;
use inference_tensor::{D, DType, Device, Result, Tensor};

use super::layers::F32LayerNorm;
use crate::conv::{Conv, Upsample};

// Channels of each projected scale, the fused width, and the output stack's widths
const SCALE_CHANNELS: [usize; 4] = [256, 512, 1024, 1024];
const FEATURES: usize = 256;
pub(crate) const HEAD_FEATURES: usize = FEATURES / 2;
const OUTPUT_HIDDEN: usize = 32;
const DPT_EPS: f64 = 1e-5;
// The sinusoidal UV position embedding's weight and base frequency
const POS_EMBED_RATIO: f64 = 0.1;
const POS_EMBED_OMEGA: f64 = 100.0;
// The Gaussian head's image branch: a 7x7 conv over the raw pixels
const IMAGE_KERNEL: usize = 7;
// Frames run through a head together, bounding the full-resolution activations
pub(crate) const FRAMES_PER_CHUNK: usize = 1;

/// How a head turns its last channels into its outputs.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Activation {
    /// `exp` attribute, `1 + exp` confidence, then a linear mask logit (depth and the Gaussian depth)
    DepthWithMask,
    /// `sign(x) (exp|x| - 1)` points, `1 + exp` confidence
    Points,
    /// unit normals, `1 + exp` confidence
    Normals,
}

impl Activation {
    fn channels(self) -> usize {
        match self {
            Self::DepthWithMask => 3,
            Self::Points | Self::Normals => 4,
        }
    }
}

enum Resize {
    Up(Upsample),
    Same,
    Down(Conv),
}

impl Resize {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Up(up) => up.forward(xs),
            Self::Same => Ok(xs.clone()),
            Self::Down(conv) => conv.forward(xs),
        }
    }
}

// relu(x) + conv2(relu(conv1(relu(x)))): the reference's shared in-place ReLU turns the skip into relu(x)
struct ResidualUnit {
    conv1: Conv,
    conv2: Conv,
}

impl ResidualUnit {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            conv1: Conv::load(FEATURES, FEATURES, 3, 1, 1, true, vb.pp("conv1"))?,
            conv2: Conv::load(FEATURES, FEATURES, 3, 1, 1, true, vb.pp("conv2"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x = xs.relu()?;
        let out = self.conv2.forward(&self.conv1.forward(&x)?.relu()?)?;
        out + x
    }
}

struct FusionBlock {
    residual: Option<ResidualUnit>,
    refine: ResidualUnit,
    out_conv: Conv,
}

impl FusionBlock {
    fn load(has_residual: bool, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            residual: if has_residual {
                Some(ResidualUnit::load(vb.pp("resConfUnit1"))?)
            } else {
                None
            },
            refine: ResidualUnit::load(vb.pp("resConfUnit2"))?,
            out_conv: Conv::load(FEATURES, FEATURES, 1, 1, 0, true, vb.pp("out_conv"))?,
        })
    }

    // fuses `xs` (and the skip, when this block has one), then resizes to `size` (twice the input when `None`)
    fn forward(
        &self,
        xs: &Tensor,
        skip: Option<&Tensor>,
        size: Option<(usize, usize)>,
    ) -> Result<Tensor> {
        let mut out = xs.clone();
        if let (Some(residual), Some(skip)) = (&self.residual, skip) {
            out = (out + residual.forward(skip)?)?;
        }
        out = self.refine.forward(&out)?;
        let (h, w) = size.unwrap_or((2 * out.dim(2)?, 2 * out.dim(3)?));
        self.out_conv.forward(&out.upsample_bilinear2d(h, w, true)?)
    }
}

/// A head's fused full-resolution features `[frames, 128, H, W]` and its decoded outputs, each `[frames, H, W, ...]`.
pub(crate) struct DptOutput {
    /// only in the chunks `forward` hands its callback; the whole-input output leaves it out
    pub fused: Option<Tensor>,
    pub attr: Tensor,
    pub conf: Tensor,
    pub mask_logits: Option<Tensor>,
}

pub(crate) struct DptHead {
    norm: F32LayerNorm,
    projects: Vec<Conv>,
    resizes: Vec<Resize>,
    layer_rn: Vec<Conv>,
    refine: Vec<FusionBlock>,
    output_conv1: Conv,
    output_hidden: Conv,
    output_last: Conv,
    image_branch: Option<Conv>,
    activation: Activation,
    patch: usize,
}

impl DptHead {
    pub(crate) fn load(
        dim: usize,
        patch: usize,
        activation: Activation,
        image_branch: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let projects = SCALE_CHANNELS
            .iter()
            .enumerate()
            .map(|(i, &oc)| Conv::load(dim, oc, 1, 1, 0, true, vb.pp("projects").pp(i)))
            .collect::<Result<Vec<_>>>()?;
        let resize = vb.pp("resize_layers");
        let resizes = vec![
            Resize::Up(Upsample::load(SCALE_CHANNELS[0], 4, resize.pp(0))?),
            Resize::Up(Upsample::load(SCALE_CHANNELS[1], 2, resize.pp(1))?),
            Resize::Same,
            Resize::Down(Conv::load(
                SCALE_CHANNELS[3],
                SCALE_CHANNELS[3],
                3,
                2,
                1,
                true,
                resize.pp(3),
            )?),
        ];
        let scratch = vb.pp("scratch");
        let layer_rn = SCALE_CHANNELS
            .iter()
            .enumerate()
            .map(|(i, &oc)| {
                Conv::load(
                    oc,
                    FEATURES,
                    3,
                    1,
                    1,
                    false,
                    scratch.pp(format!("layer{}_rn", i + 1)),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let refine = (1..=4)
            .map(|i| FusionBlock::load(i != 4, scratch.pp(format!("refinenet{i}"))))
            .collect::<Result<Vec<_>>>()?;
        let output2 = scratch.pp("output_conv2");
        Ok(Self {
            norm: F32LayerNorm::load(dim, DPT_EPS, vb.pp("norm"))?,
            projects,
            resizes,
            layer_rn,
            refine,
            output_conv1: Conv::load(
                FEATURES,
                HEAD_FEATURES,
                3,
                1,
                1,
                true,
                scratch.pp("output_conv1"),
            )?,
            output_hidden: Conv::load(HEAD_FEATURES, OUTPUT_HIDDEN, 3, 1, 1, true, output2.pp(0))?,
            output_last: Conv::load(
                OUTPUT_HIDDEN,
                activation.channels(),
                1,
                1,
                0,
                true,
                output2.pp(2),
            )?,
            image_branch: if image_branch {
                Some(Conv::load(
                    3,
                    HEAD_FEATURES,
                    IMAGE_KERNEL,
                    1,
                    IMAGE_KERNEL / 2,
                    true,
                    vb.pp("input_merger").pp(0),
                )?)
            } else {
                None
            },
            activation,
            patch,
        })
    }

    /// The head over the trunk's four feature maps `[frames, tokens, dim]` (patches from `patch_start`) and the raw
    /// [0, 1] images `[frames, 3, H, W]`, a chunk of frames at a time.
    pub(crate) fn forward(
        &self,
        features: &[Tensor],
        patch_start: usize,
        images: &Tensor,
        per_chunk: &mut dyn FnMut(usize, &DptOutput) -> Result<()>,
    ) -> Result<DptOutput> {
        let frames = images.dim(0)?;
        let (mut attr, mut conf, mut masks) = (Vec::new(), Vec::new(), Vec::new());
        for start in (0..frames).step_by(FRAMES_PER_CHUNK) {
            let n = FRAMES_PER_CHUNK.min(frames - start);
            let chunk = features
                .iter()
                .map(|f| f.narrow(0, start, n))
                .collect::<Result<Vec<_>>>()?;
            let out = self.forward_frames(&chunk, patch_start, &images.narrow(0, start, n)?)?;
            per_chunk(start, &out)?;
            attr.push(out.attr);
            conf.push(out.conf);
            masks.extend(out.mask_logits);
        }
        Ok(DptOutput {
            fused: None,
            attr: Tensor::cat(&attr, 0)?,
            conf: Tensor::cat(&conf, 0)?,
            mask_logits: if masks.is_empty() {
                None
            } else {
                Some(Tensor::cat(&masks, 0)?)
            },
        })
    }

    fn forward_frames(
        &self,
        features: &[Tensor],
        patch_start: usize,
        images: &Tensor,
    ) -> Result<DptOutput> {
        let (frames, _, h, w) = images.dims4()?;
        let (hp, wp) = (h / self.patch, w / self.patch);
        let aspect = w as f64 / h as f64;
        let mut scales = Vec::with_capacity(features.len());
        for ((tokens, project), resize) in features.iter().zip(&self.projects).zip(&self.resizes) {
            let patches = self
                .norm
                .forward(&tokens.narrow(1, patch_start, hp * wp)?)?;
            let dim = patches.dim(2)?;
            let map = patches.transpose(1, 2)?.reshape((frames, dim, hp, wp))?;
            let projected = project.forward(&map)?;
            let with_pos = add_pos_embed(&projected, aspect)?;
            scales.push(resize.forward(&with_pos)?);
        }
        let rn = scales
            .iter()
            .zip(&self.layer_rn)
            .map(|(s, conv)| conv.forward(s))
            .collect::<Result<Vec<_>>>()?;
        let size = |t: &Tensor| -> Result<(usize, usize)> { Ok((t.dim(2)?, t.dim(3)?)) };
        let mut out = self.refine[3].forward(&rn[3], None, Some(size(&rn[2])?))?;
        out = self.refine[2].forward(&out, Some(&rn[2]), Some(size(&rn[1])?))?;
        out = self.refine[1].forward(&out, Some(&rn[1]), Some(size(&rn[0])?))?;
        out = self.refine[0].forward(&out, Some(&rn[0]), None)?;
        let out = self.output_conv1.forward(&out)?;
        let fused = add_pos_embed(
            &out.upsample_bilinear2d(hp * self.patch, wp * self.patch, true)?,
            aspect,
        )?;
        let head = self
            .output_last
            .forward(&self.output_hidden.forward(&fused)?.relu()?)?
            .permute((0, 2, 3, 1))?;
        let (attr, conf, mask_logits) = self.activate(&head)?;
        let fused = match &self.image_branch {
            Some(conv) => (fused + conv.forward(images)?.relu()?)?,
            None => fused,
        };
        Ok(DptOutput {
            fused: Some(fused),
            attr,
            conf,
            mask_logits,
        })
    }

    // the head's last channels (NHWC) into its attribute, confidence and mask logits
    fn activate(&self, head: &Tensor) -> Result<(Tensor, Tensor, Option<Tensor>)> {
        let channels = head.dim(D::Minus1)?;
        let conf_at = match self.activation {
            Activation::DepthWithMask => channels - 2,
            Activation::Points | Activation::Normals => channels - 1,
        };
        let raw = head.narrow(D::Minus1, 0, conf_at)?;
        let conf = (head
            .narrow(D::Minus1, conf_at, 1)?
            .squeeze(D::Minus1)?
            .exp()?
            + 1.0)?;
        let attr = match self.activation {
            Activation::DepthWithMask => raw.exp()?,
            Activation::Points => {
                let magnitude = (raw.abs()?.exp()? - 1.0)?;
                (raw.sign()? * magnitude)?
            }
            Activation::Normals => {
                raw.broadcast_div(&raw.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?)?
            }
        };
        let mask = match self.activation {
            Activation::DepthWithMask => Some(
                head.narrow(D::Minus1, channels - 1, 1)?
                    .squeeze(D::Minus1)?,
            ),
            _ => None,
        };
        Ok((attr, conf, mask))
    }
}

// `xs + 0.1 * emb` for features `[frames, C, h, w]`: a sinusoidal embedding of a UV grid spanning the image's aspect,
// normalized by its diagonal (MoGe's grid), in f64 then f32 as the reference computes it
fn add_pos_embed(xs: &Tensor, aspect: f64) -> Result<Tensor> {
    let (_, channels, h, w) = xs.dims4()?;
    let diag = (aspect * aspect + 1.0).sqrt();
    let (span_x, span_y) = (aspect / diag, 1.0 / diag);
    let linspace = |span: f64, n: usize| -> Vec<f64> {
        let (lo, hi) = (
            -span * (n - 1) as f64 / n as f64,
            span * (n - 1) as f64 / n as f64,
        );
        (0..n)
            .map(|i| {
                let v = if n == 1 {
                    lo
                } else {
                    lo + (hi - lo) * i as f64 / (n - 1) as f64
                };
                f64::from(v as f32)
            })
            .collect()
    };
    let (us, vs) = (linspace(span_x, w), linspace(span_y, h));
    let quarter = channels / 4;
    let omega = (0..quarter)
        .map(|k| 1.0 / POS_EMBED_OMEGA.powf(k as f64 / quarter as f64))
        .collect::<Vec<_>>();
    // sin then cos of each coordinate at each frequency: [2 * quarter, n], separable over the grid's two axes
    let table = |coords: &[f64]| -> Vec<f32> {
        [f64::sin, f64::cos]
            .iter()
            .flat_map(|f| {
                omega.iter().flat_map(move |&o| {
                    coords
                        .iter()
                        .map(move |&c| (f(c * o) * POS_EMBED_RATIO) as f32)
                })
            })
            .collect()
    };
    let device = xs.device();
    let u = Tensor::from_vec(table(&us), (2 * quarter, 1, w), &Device::Cpu)?
        .to_device(device)?
        .broadcast_as((2 * quarter, h, w))?;
    let v = Tensor::from_vec(table(&vs), (2 * quarter, h, 1), &Device::Cpu)?
        .to_device(device)?
        .broadcast_as((2 * quarter, h, w))?;
    let emb = Tensor::cat(&[&u, &v], 0)?.unsqueeze(0)?;
    xs.to_dtype(DType::F32)?.broadcast_add(&emb)
}
