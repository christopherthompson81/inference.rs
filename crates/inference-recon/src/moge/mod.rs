//! MoGe-2 (ViT-L, with normals): a metric point map, a validity mask and normals for one image at a known field of
//! view, as `MoGeModel.infer(image, fov_x=..., apply_mask=False)` computes them.

use std::path::Path;

use inference_tensor::nn::VarBuilder;
use inference_tensor::{D, DType, Device, Result, Tensor};
use serde::Deserialize;

use crate::conv::{Conv, Upsample};
use crate::dinov2::{Dinov2, Dinov2Spec, PosInterp};

const PATCH: usize = 14;
// DINOv2's default position-embedding interpolation offset
const POS_OFFSET: f64 = 0.1;
// infer's resolution_level 9 picks the top of num_tokens_range
const RESOLUTION_LEVEL: f64 = 9.0;
const LEVELS: usize = 5;
// the shift is fit on a nearest-downsampled point map of this size
const SHIFT_FIT_SIZE: usize = 64;
const SHIFT_ITERATIONS: usize = 50;
const MASK_THRESHOLD: f32 = 0.5;
// the memory selector's descriptor is taken at this input size
const FEATURE_SIZE: usize = 224;

#[derive(Debug, Clone, Deserialize)]
struct EncoderConfig {
    intermediate_layers: Vec<usize>,
    dim_out: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct StackConfig {
    dim_in: Vec<usize>,
    // null for the neck, which passes every level's features on
    dim_out: Option<Vec<Option<usize>>>,
    dim_res_blocks: Vec<usize>,
    num_res_blocks: Vec<usize>,
    resamplers: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct ScaleConfig {
    dims: Vec<usize>,
}

/// The checkpoint's `model_config`.
#[derive(Debug, Clone, Deserialize)]
pub struct MoGeConfig {
    encoder: EncoderConfig,
    neck: StackConfig,
    points_head: StackConfig,
    normal_head: StackConfig,
    mask_head: StackConfig,
    scale_head: ScaleConfig,
    num_tokens_range: [usize; 2],
}

// Edge-replicate one pixel on every side of `[b, c, h, w]`
fn replicate_pad(xs: &Tensor) -> Result<Tensor> {
    let (_, _, h, w) = xs.dims4()?;
    let xs = Tensor::cat(&[&xs.narrow(2, 0, 1)?, xs, &xs.narrow(2, h - 1, 1)?], 2)?;
    Tensor::cat(&[&xs.narrow(3, 0, 1)?, &xs, &xs.narrow(3, w - 1, 1)?], 3)
}

// A 3x3 conv with replicate padding
struct Conv3(Conv);

impl Conv3 {
    fn load(cin: usize, cout: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self(Conv::load(cin, cout, 3, 1, 0, true, vb)?))
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.0.forward(&replicate_pad(xs)?)
    }
}

// relu, conv, relu, conv, plus the input (both norms are identities in this config)
struct ResBlock {
    conv1: Conv3,
    conv2: Conv3,
}

impl ResBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let ys = self
            .conv2
            .forward(&self.conv1.forward(&xs.relu()?)?.relu()?)?;
        ys + xs
    }
}

enum Resampler {
    // a stride-2 transposed conv, then a 3x3 conv
    Transpose(Upsample, Conv3),
    // bilinear 2x, then a 3x3 conv
    Bilinear(Conv3),
}

impl Resampler {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Transpose(up, conv) => conv.forward(&up.forward(xs)?),
            Self::Bilinear(conv) => {
                let (_, _, h, w) = xs.dims4()?;
                conv.forward(&xs.upsample_bilinear2d(2 * h, 2 * w, false)?)
            }
        }
    }
}

// MoGe's ConvStack: per level an optional 1x1 input projection added in, residual blocks, an optional 1x1 output,
// then a 2x resampler into the next level
struct ConvStack {
    inputs: Vec<Option<Conv>>,
    blocks: Vec<Vec<ResBlock>>,
    outputs: Vec<Option<Conv>>,
    resamplers: Vec<Resampler>,
}

impl ConvStack {
    fn load(cfg: &StackConfig, vb: VarBuilder) -> Result<Self> {
        let dims = &cfg.dim_res_blocks;
        let inputs = cfg
            .dim_in
            .iter()
            .zip(dims)
            .enumerate()
            .map(|(i, (&cin, &d))| {
                Ok(Some(Conv::load(
                    cin,
                    d,
                    1,
                    1,
                    0,
                    true,
                    vb.pp("input_blocks").pp(i),
                )?))
            })
            .collect::<Result<Vec<_>>>()?;
        let blocks = dims
            .iter()
            .enumerate()
            .map(|(i, &d)| {
                (0..cfg.num_res_blocks[i])
                    .map(|j| {
                        let layers = vb.pp("res_blocks").pp(i).pp(j).pp("layers");
                        Ok(ResBlock {
                            conv1: Conv3::load(d, d, layers.pp(2))?,
                            conv2: Conv3::load(d, d, layers.pp(5))?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        let no_outputs = vec![None; dims.len()];
        let outputs = cfg
            .dim_out
            .as_ref()
            .unwrap_or(&no_outputs)
            .iter()
            .zip(dims)
            .enumerate()
            .map(|(i, (out, &d))| match out {
                Some(o) => Ok(Some(Conv::load(
                    d,
                    *o,
                    1,
                    1,
                    0,
                    true,
                    vb.pp("output_blocks").pp(i),
                )?)),
                None => Ok(None),
            })
            .collect::<Result<Vec<_>>>()?;
        let resamplers = cfg
            .resamplers
            .iter()
            .enumerate()
            .map(|(i, kind)| {
                let r = vb.pp("resamplers").pp(i);
                let (cin, cout) = (dims[i], dims[i + 1]);
                match kind.as_str() {
                    "conv_transpose" => Ok(Resampler::Transpose(
                        Upsample::load_with(cin, cout, 2, r.pp(0))?,
                        Conv3::load(cout, cout, r.pp(1))?,
                    )),
                    "bilinear" => Ok(Resampler::Bilinear(Conv3::load(cin, cout, r.pp(1))?)),
                    other => inference_tensor::bail!("MoGe resampler {other} is not supported"),
                }
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            inputs,
            blocks,
            outputs,
            resamplers,
        })
    }

    fn forward(&self, features: &[Tensor]) -> Result<Vec<Tensor>> {
        let mut out = Vec::with_capacity(self.blocks.len());
        let mut x: Option<Tensor> = None;
        let levels = self
            .inputs
            .iter()
            .zip(&self.blocks)
            .zip(&self.outputs)
            .zip(features);
        for (i, (((input, blocks), output), feature)) in levels.enumerate() {
            let feature = match input {
                Some(conv) => conv.forward(feature)?,
                None => feature.clone(),
            };
            let mut h = match x {
                Some(prev) => (prev + feature)?,
                None => feature,
            };
            for block in blocks {
                h = block.forward(&h)?;
            }
            out.push(match output {
                Some(conv) => conv.forward(&h)?,
                None => h.clone(),
            });
            x = match self.resamplers.get(i) {
                Some(r) => Some(r.forward(&h)?),
                None => None,
            };
        }
        Ok(out)
    }
}

/// MoGe-2's outputs for one image, at its input size.
pub struct MoGeOutput {
    /// `[H, W, 3]` metric camera-space points (OpenCV axes), from the depth through the intrinsics
    pub points: Tensor,
    /// `[H, W]` metric z-depth
    pub depth: Tensor,
    /// `[H, W]` 0/1 validity (mask probability above 0.5 and positive depth)
    pub mask: Tensor,
    /// `[H, W, 3]` unit normals
    pub normal: Tensor,
}

pub struct MoGe {
    encoder: Dinov2,
    projections: Vec<Conv>,
    layers: Vec<usize>,
    neck: ConvStack,
    points_head: ConvStack,
    normal_head: ConvStack,
    mask_head: ConvStack,
    scale_head: Vec<Conv>,
    image_mean: Tensor,
    image_std: Tensor,
    num_tokens: usize,
    device: Device,
}

impl MoGe {
    /// From a folder with `config.json` (the checkpoint's model_config) and `model.safetensors`.
    pub fn load(dir: &Path, dtype: DType, device: &Device) -> Result<Self> {
        let cfg: MoGeConfig =
            serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)
                .map_err(inference_tensor::Error::wrap)?;
        // SAFETY: the weights file is not modified while the model is loaded
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[dir.join("model.safetensors")],
                DType::F32,
                device,
            )?
        };
        let enc = vb.pp("encoder");
        let dim = cfg.encoder.dim_out;
        let encoder = Dinov2::load(
            Dinov2Spec {
                dim,
                depth: cfg
                    .encoder
                    .intermediate_layers
                    .iter()
                    .copied()
                    .max()
                    .unwrap_or(0)
                    + 1,
                heads: dim / 64,
                patch: PATCH,
                registers: 0,
                pos_interp: PosInterp::Bicubic { offset: POS_OFFSET },
                dtype,
            },
            enc.pp("backbone"),
        )?;
        let projections = (0..cfg.encoder.intermediate_layers.len())
            .map(|i| Conv::load(dim, dim, 1, 1, 0, true, enc.pp("output_projections").pp(i)))
            .collect::<Result<Vec<_>>>()?;
        let dims = &cfg.scale_head.dims;
        // the scale MLP as 1x1 convs over a 1 x 1 "image" of the class token: Linear, ReLU, ..., Linear
        let scale_head = (0..dims.len() - 1)
            .map(|i| {
                let l = vb.pp("scale_head").pp(2 * i);
                let weight = l.get((dims[i + 1], dims[i]), "weight")?.reshape((
                    dims[i + 1],
                    dims[i],
                    1,
                    1,
                ))?;
                Conv::from_parts(weight, Some(l.get(dims[i + 1], "bias")?))
            })
            .collect::<Result<Vec<_>>>()?;
        let [lo, hi] = cfg.num_tokens_range;
        Ok(Self {
            encoder,
            projections,
            layers: cfg.encoder.intermediate_layers.clone(),
            neck: ConvStack::load(&cfg.neck, vb.pp("neck"))?,
            points_head: ConvStack::load(&cfg.points_head, vb.pp("points_head"))?,
            normal_head: ConvStack::load(&cfg.normal_head, vb.pp("normal_head"))?,
            mask_head: ConvStack::load(&cfg.mask_head, vb.pp("mask_head"))?,
            scale_head,
            image_mean: enc.get((1, 3, 1, 1), "image_mean")?,
            image_std: enc.get((1, 3, 1, 1), "image_std")?,
            num_tokens: (lo as f64 + RESOLUTION_LEVEL / 9.0 * (hi - lo) as f64) as usize,
            device: device.clone(),
        })
    }

    /// One image `[3, H, W]` in [0, 1] seen with horizontal field of view `fov_x_deg`.
    pub fn infer(&self, image: &Tensor, fov_x_deg: f64) -> Result<MoGeOutput> {
        let (_, img_h, img_w) = image.dims3()?;
        let aspect = img_w as f64 / img_h as f64;
        let rows = ((self.num_tokens as f64 / aspect).sqrt()) as usize;
        let cols = ((self.num_tokens as f64 * aspect).sqrt()) as usize;
        let image = image
            .to_device(&self.device)?
            .to_dtype(DType::F32)?
            .unsqueeze(0)?;
        // upsampled to the token grid (bilinear; antialiasing only changes a downsample)
        let resized = image.upsample_bilinear2d(rows * PATCH, cols * PATCH, false)?;
        let normed = resized
            .broadcast_sub(&self.image_mean)?
            .broadcast_div(&self.image_std)?;
        let layers = self.encoder.forward_layers(&normed, &self.layers)?;
        let mut feature: Option<Tensor> = None;
        for (proj, (tokens, _)) in self.projections.iter().zip(&layers) {
            let grid = tokens
                .transpose(1, 2)?
                .reshape((1, (), rows, cols))?
                .contiguous()?;
            let f = proj.forward(&grid)?;
            feature = Some(match feature {
                Some(acc) => (acc + f)?,
                None => f,
            });
        }
        let cls = &layers.last().expect("intermediate layers").1;
        let mut features = Vec::with_capacity(LEVELS);
        for level in 0..LEVELS {
            let uv = view_plane_uv(cols << level, rows << level, aspect, &self.device)?;
            features.push(match (level, &feature) {
                (0, Some(f)) => Tensor::cat(&[f, &uv], 1)?,
                _ => uv,
            });
        }
        let neck = self.neck.forward(&features)?;
        let last = |head: &ConvStack| -> Result<Tensor> {
            let out = head.forward(&neck)?.pop().expect("levels");
            out.upsample_bilinear2d(img_h, img_w, false)?
                .squeeze(0)?
                .permute((1, 2, 0))
        };
        let raw = last(&self.points_head)?;
        // remap "exp": z = exp(z), xy scaled by it
        let z = raw.narrow(2, 2, 1)?.exp()?;
        let raw = Tensor::cat(&[&raw.narrow(2, 0, 2)?.broadcast_mul(&z)?, &z], 2)?;
        let normal = last(&self.normal_head)?;
        let normal = normal.broadcast_div(
            &normal
                .sqr()?
                .sum_keepdim(2)?
                .sqrt()?
                .clamp(1e-12, f64::MAX)?,
        )?;
        let mask_prob = inference_tensor::nn::ops::sigmoid(&last(&self.mask_head)?.squeeze(2)?)?;
        let mut s = cls.reshape((1, (), 1, 1))?;
        for (i, layer) in self.scale_head.iter().enumerate() {
            s = layer.forward(&s)?;
            if i + 1 < self.scale_head.len() {
                s = s.relu()?;
            }
        }
        let metric_scale = f64::from(s.flatten_all()?.exp()?.to_vec1::<f32>()?[0]);

        let focal = aspect / (1.0 + aspect * aspect).sqrt() / (fov_x_deg.to_radians() / 2.0).tan();
        let mask_binary = mask_prob.gt(MASK_THRESHOLD)?;
        let shift = fit_shift(&raw, &mask_binary, focal, aspect)?;
        let depth = (raw.narrow(2, 2, 1)?.squeeze(2)? + shift)?;
        let mask = (mask_binary.to_dtype(DType::F32)? * depth.gt(0.0)?.to_dtype(DType::F32)?)?;
        let diag = (1.0 + aspect * aspect).sqrt();
        let (fx, fy) = (focal / 2.0 * diag / aspect, focal / 2.0 * diag);
        let points = depth_to_points(&depth, fx, fy)?;
        Ok(MoGeOutput {
            points: (points * metric_scale)?,
            depth: (depth * metric_scale)?,
            mask,
            normal,
        })
    }
}

impl MoGe {
    /// An image descriptor for the memory selector: the encoder's last class token (normed) for the image PIL-resized
    /// to 224 x 224 and ImageNet-normalized (the reference uses DINOv2-base's pooled output for the same job).
    pub fn image_features(
        &self,
        pixels: &[u8],
        (width, height): (usize, usize),
    ) -> Result<Vec<f32>> {
        let s = FEATURE_SIZE;
        let resized = inference_vision::pil_resize_bicubic(pixels, (width, height), (s, s));
        let chw = Tensor::from_vec(
            resized
                .iter()
                .map(|&p| f32::from(p) / 255.0)
                .collect::<Vec<_>>(),
            (s, s, 3),
            &Device::Cpu,
        )?
        .permute((2, 0, 1))?
        .unsqueeze(0)?
        .to_device(&self.device)?;
        let normed = chw
            .broadcast_sub(&self.image_mean)?
            .broadcast_div(&self.image_std)?;
        let last = *self.layers.last().expect("intermediate layers");
        let layers = self.encoder.forward_layers(&normed, &[last])?;
        layers[0]
            .1
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()
    }
}

/// UV `[1, 2, h, w]` with corners at `(+-w, +-h) / diagonal` scaled to pixel centres, as normalized_view_plane_uv.
fn view_plane_uv(width: usize, height: usize, aspect: f64, device: &Device) -> Result<Tensor> {
    let span_x = aspect / (1.0 + aspect * aspect).sqrt();
    let span_y = 1.0 / (1.0 + aspect * aspect).sqrt();
    let line = |n: usize, span: f64| -> Vec<f32> {
        let lo = -span * (n - 1) as f64 / n as f64;
        (0..n)
            .map(|i| {
                (lo + 2.0 * span * (n - 1) as f64 / n as f64 * i as f64 / (n.max(2) - 1) as f64)
                    as f32
            })
            .collect()
    };
    let (us, vs) = (line(width, span_x), line(height, span_y));
    let mut out = Vec::with_capacity(2 * width * height);
    out.extend((0..height).flat_map(|_| us.iter().copied()));
    out.extend(vs.iter().flat_map(|&v| std::iter::repeat_n(v, width)));
    Tensor::from_vec(out, (1, 2, height, width), device)
}

// The z shift minimizing |focal * xy / (z + shift) - uv| over masked cells of the 64 x 64 nearest-downsampled map
// (scipy's least_squares in the reference; here Levenberg-Marquardt on the one unknown)
fn fit_shift(points: &Tensor, mask: &Tensor, focal: f64, aspect: f64) -> Result<f64> {
    let (h, w, _) = points.dims3()?;
    let pts = points
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .to_vec3::<f32>()?;
    let m = mask
        .to_device(&Device::Cpu)?
        .to_dtype(DType::U8)?
        .to_vec2::<u8>()?;
    let uv = view_plane_uv(w, h, aspect, &Device::Cpu)?
        .squeeze(0)?
        .to_vec3::<f32>()?;
    let n = SHIFT_FIT_SIZE;
    let mut samples = Vec::with_capacity(n * n);
    for i in 0..n {
        let y = i * h / n;
        for j in 0..n {
            let x = j * w / n;
            if m[y][x] != 0 {
                let p = &pts[y][x];
                samples.push((
                    f64::from(p[0]),
                    f64::from(p[1]),
                    f64::from(p[2]),
                    f64::from(uv[0][y][x]),
                    f64::from(uv[1][y][x]),
                ));
            }
        }
    }
    if samples.len() < 2 {
        return Ok(0.0);
    }
    let cost = |s: f64| -> (f64, f64, f64) {
        // residual sum of squares, gradient and Gauss-Newton curvature in s
        samples
            .iter()
            .fold((0.0, 0.0, 0.0), |(c, g, hh), &(x, y, z, u, v)| {
                let d = z + s;
                let (rx, ry) = (focal * x / d - u, focal * y / d - v);
                let (jx, jy) = (-focal * x / (d * d), -focal * y / (d * d));
                (
                    c + rx * rx + ry * ry,
                    g + rx * jx + ry * jy,
                    hh + jx * jx + jy * jy,
                )
            })
    };
    let (mut s, mut lambda) = (0.0f64, 1e-3f64);
    let (mut c, mut g, mut hh) = cost(s);
    for _ in 0..SHIFT_ITERATIONS {
        let step = -g / (hh * (1.0 + lambda));
        let (c_new, g_new, h_new) = cost(s + step);
        if c_new < c {
            s += step;
            let done = (c - c_new) <= 1e-12 * c.max(1e-30);
            (c, g, hh, lambda) = (c_new, g_new, h_new, lambda * 0.1);
            if done {
                break;
            }
        } else {
            lambda *= 10.0;
        }
    }
    Ok(s)
}

// Camera-space points `[H, W, 3]` of z-depth `[H, W]` with normalized intrinsics (fx, fy relative to the image, centre
// 0.5), pixel centres at (x + 0.5) / W
fn depth_to_points(depth: &Tensor, fx: f64, fy: f64) -> Result<Tensor> {
    let (h, w) = depth.dims2()?;
    let device = depth.device();
    let us = Tensor::arange(0u32, w as u32, device)?.to_dtype(DType::F32)?;
    let vs = Tensor::arange(0u32, h as u32, device)?.to_dtype(DType::F32)?;
    let x = ((((us + 0.5)? / w as f64)? - 0.5)? / fx)?
        .reshape((1, w))?
        .broadcast_mul(depth)?;
    let y = ((((vs + 0.5)? / h as f64)? - 0.5)? / fy)?
        .reshape((h, 1))?
        .broadcast_mul(depth)?;
    Tensor::stack(&[&x, &y, depth], D::Minus1)
}

#[cfg(test)]
mod parity {
    use inference_tensor::{DType, Device, Result, Tensor};

    fn compare(a: &Tensor, b: &Tensor) -> Result<(f32, f32)> {
        let a = a.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let b = b.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let d = (&a - &b)?;
        let rms = |t: &Tensor| -> Result<f32> { t.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>() };
        Ok((
            d.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?,
            rms(&d)? / rms(&b)?,
        ))
    }

    // MOGE_DIR (config.json + model.safetensors) and MOGE_REF: infer outputs on a 512 x 512 crop at fov_x 90
    #[test]
    #[ignore = "needs the MoGe-2 weights, a reference dump and a GPU"]
    fn moge_matches_the_reference() -> Result<()> {
        let (Some(dir), Some(dump)) = (std::env::var_os("MOGE_DIR"), std::env::var_os("MOGE_REF"))
        else {
            return Ok(());
        };
        let device = Device::new_cuda(0)?;
        let refs = inference_tensor::safetensors::load(dump, &Device::Cpu)?;
        for dtype in [DType::F32, DType::BF16] {
            let model = super::MoGe::load(std::path::Path::new(&dir), dtype, &device)?;
            let out = model.infer(&refs["image"], 90.0)?;
            for name in ["f32", "amp"] {
                let (dmax, drel) = compare(&out.depth, &refs[&format!("{name}.depth")])?;
                let (_, nrel) = compare(&out.normal, &refs[&format!("{name}.normal")])?;
                let ours = out.mask.to_device(&Device::Cpu)?;
                let theirs = refs[&format!("{name}.mask")].to_dtype(DType::F32)?;
                let flips = (ours - theirs)?.abs()?.mean_all()?.to_scalar::<f32>()?;
                println!(
                    "{dtype:?} trunk vs reference {name}: depth max {dmax:.3e} rel rms {drel:.3e}, normal rel rms {nrel:.3e}, mask flips {:.4}%",
                    flips * 100.0
                );
            }
        }
        Ok(())
    }
}
