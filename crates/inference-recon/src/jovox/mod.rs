//! JoVox: an image-conditioned 64^3 voxel structure from discrete flow matching over a two-level occupancy pyramid
//! (4^3 -> 16^3 -> 64^3), one model for every level. DINOv3 ViT-L/16 embeds the object's image; the flow, TRELLIS.2's
//! sparse DiT, draws each level's child-occupancy bytes as the sampler unmasks them.

pub mod flow;
#[cfg(test)]
mod parity;
pub mod sampler;

use std::path::Path;

use inference_tensor::nn::VarBuilder;
use inference_tensor::{D, DType, Device, Result, Tensor};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, weighted::WeightedIndex};

use crate::dinov3::{Dinov3, Dinov3Spec};
use crate::progress::{Progress, check};
use flow::{CLASSES, Flow, FlowSpec};
use sampler::{SLOTS, Schedule, Tokens};

/// The released 64^3 model's flow (`configs/voxelgen64.yaml`).
pub const JOVOX_64: FlowSpec = FlowSpec {
    channels: 1536,
    cond_channels: 1024,
    blocks: 30,
    heads: 12,
};
const SCHEDULE: Schedule = Schedule {
    power: 2.0,
    cleanup_steps: 8,
    remask_steps: 1,
};
// where the root level's window ends at inference, and the training range of grid steps
const ROOT_END: f64 = 0.8;
pub const STEPS_RANGE: std::ops::RangeInclusive<usize> = 36..=108;
pub const DEFAULT_STEPS: usize = 72;
pub const DEFAULT_CFG: f32 = 1.5;
/// The share of the image the object's longest side fills, as the reference's prep stage
pub const DEFAULT_FILL: f32 = 0.74;
// prep: inputs are first shrunk to this longest side; alpha above this is foreground; the model sees this size
const MAX_INPUT: usize = 1024;
const FOREGROUND_ALPHA: u8 = 204;
pub const IMAGE_SIZE: usize = 512;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];
const RGBA: usize = 4;

/// How a structure is sampled.
#[derive(Clone, Copy, Debug)]
pub struct SampleOptions {
    pub seed: u64,
    pub steps: usize,
    /// Classifier-free guidance; 1 runs the conditional branch alone
    pub cfg_strength: f32,
    /// 0 takes each slot's most likely value
    pub temperature: f32,
}

impl Default for SampleOptions {
    fn default() -> Self {
        Self {
            seed: 0,
            steps: DEFAULT_STEPS,
            cfg_strength: DEFAULT_CFG,
            temperature: 0.0,
        }
    }
}

/// An RGBA image, four bytes a pixel, row-major.
#[derive(Clone)]
pub struct Rgba {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u8>,
}

/// The reference's prep stage: the foreground (alpha above 204) cropped, centered on a transparent square its longest
/// side fills `fill` of, resized to 512^2. The image must carry its own transparent background.
pub fn prepare(image: &Rgba, fill: f32) -> Result<Rgba> {
    if !(0.0..=1.0).contains(&fill) || fill == 0.0 {
        inference_tensor::bail!("fill must be in (0, 1], got {fill}");
    }
    if image.pixels.chunks_exact(RGBA).all(|px| px[3] == 255) {
        inference_tensor::bail!(
            "the image has no transparent background: give the object on a transparent background (RGBA)"
        );
    }
    let mut image = image.clone();
    let longest = image.width.max(image.height);
    if longest > MAX_INPUT {
        let s = MAX_INPUT as f64 / longest as f64;
        let (w, h) = (
            (image.width as f64 * s) as usize,
            (image.height as f64 * s) as usize,
        );
        image.pixels = inference_vision::pil_resize_rgba_lanczos(
            &image.pixels,
            (image.width, image.height),
            (w, h),
        );
        (image.width, image.height) = (w, h);
    }
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
    for (i, px) in image.pixels.chunks_exact(RGBA).enumerate() {
        if px[3] > FOREGROUND_ALPHA {
            let (x, y) = (i % image.width, i / image.width);
            (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1));
        }
    }
    if x0 == usize::MAX {
        inference_tensor::bail!(
            "the image has no foreground (no pixel with alpha above {FOREGROUND_ALPHA})"
        );
    }
    let (w, h) = (x1 - x0, y1 - y0);
    let side = (w.max(h) as f64 / f64::from(fill)).ceil() as usize;
    let (ox, oy) = ((side - w) / 2, (side - h) / 2);
    let mut canvas = vec![0u8; side * side * RGBA];
    for y in 0..h {
        let src = &image.pixels[((y0 + y) * image.width + x0) * RGBA..][..w * RGBA];
        canvas[((oy + y) * side + ox) * RGBA..][..w * RGBA].copy_from_slice(src);
    }
    Ok(Rgba {
        width: IMAGE_SIZE,
        height: IMAGE_SIZE,
        pixels: inference_vision::pil_resize_rgba_lanczos(
            &canvas,
            (side, side),
            (IMAGE_SIZE, IMAGE_SIZE),
        ),
    })
}

// the color times alpha, normalized as DINOv3 expects, `[1, 3, H, W]`
fn image_tensor(image: &Rgba, device: &Device) -> Result<Tensor> {
    let plane = image.width * image.height;
    let mut data = vec![0f32; 3 * plane];
    for (i, px) in image.pixels.chunks_exact(RGBA).enumerate() {
        let alpha = f32::from(px[3]) / 255.0;
        for c in 0..3 {
            data[c * plane + i] = (f32::from(px[c]) / 255.0 * alpha - MEAN[c]) / STD[c];
        }
    }
    Tensor::from_vec(data, (1, 3, image.height, image.width), device)
}

pub struct JoVox {
    encoder: Dinov3,
    flow: Flow,
    device: Device,
}

impl JoVox {
    /// Loads JoVox's `model.safetensors` from `dir` and DINOv3 ViT-L/16's from `dinov3_dir`. The flow's blocks run in
    /// `dtype` (bf16 as the reference); DINOv3 runs in f32, as the reference loads it.
    pub fn load(dir: &Path, dinov3_dir: &Path, dtype: DType, device: &Device) -> Result<Self> {
        // SAFETY: the checkpoint files are not modified while the model is loaded
        let (vb, dino) = unsafe {
            (
                VarBuilder::from_mmaped_safetensors(
                    &[dir.join("model.safetensors")],
                    DType::F32,
                    device,
                )?,
                VarBuilder::from_mmaped_safetensors(
                    &[dinov3_dir.join("model.safetensors")],
                    DType::F32,
                    device,
                )?,
            )
        };
        Ok(Self {
            encoder: Dinov3::load(Dinov3Spec::vitl16(DType::F32), dino)?,
            flow: Flow::load(JOVOX_64, dtype, vb.pp("sparse_structure_flow_model"))?,
            device: device.clone(),
        })
    }

    /// The condition `[1, tokens, 1024]` for a prepared image (see [`prepare`]).
    pub fn condition(&self, image: &Rgba) -> Result<Tensor> {
        self.encoder.forward(&image_tensor(image, &self.device)?)
    }

    /// Samples a structure under `cond`: the occupied cells `xyz` of the 64^3 grid.
    pub fn sample(
        &self,
        cond: &Tensor,
        options: &SampleOptions,
        progress: &dyn Progress,
    ) -> Result<Vec<[usize; 3]>> {
        if !STEPS_RANGE.contains(&options.steps) {
            inference_tensor::bail!(
                "steps must be in {}..={}, the range the model was trained on, got {}",
                STEPS_RANGE.start(),
                STEPS_RANGE.end(),
                options.steps
            );
        }
        if options.cfg_strength < 1.0 || options.temperature < 0.0 {
            inference_tensor::bail!("cfg_strength must be >= 1 and temperature >= 0");
        }
        let both = if options.cfg_strength != 1.0 {
            Tensor::cat(&[cond, &cond.zeros_like()?], 0)?
        } else {
            cond.clone()
        };
        sample_structure(&self.flow, &self.flow.condition(&both)?, options, progress)
    }
}

// The sampler driving `flow` under `kv`: the condition alone, or it then the zero condition when guided
fn sample_structure(
    flow: &Flow,
    kv: &[flow::CrossKv],
    options: &SampleOptions,
    progress: &dyn Progress,
) -> Result<Vec<[usize; 3]>> {
    let guided = options.cfg_strength != 1.0;
    let plan = SCHEDULE.plan(ROOT_END, options.steps, 1.0);
    let mut rng = StdRng::seed_from_u64(options.seed);
    let mut draw_rng = StdRng::seed_from_u64(options.seed.wrapping_add(1));
    progress.stage("structure", plan.len());
    let mut step = 0;
    let mut draw = |tokens: &Tokens| -> Result<Vec<[u16; SLOTS]>> {
        check(progress)?;
        let logits = flow.forward(tokens, kv)?;
        let logits = if guided {
            let (pos, neg) = (logits.get(0)?, logits.get(1)?);
            (&neg + ((pos - &neg)? * f64::from(options.cfg_strength))?)?
        } else {
            logits.get(0)?
        };
        step += 1;
        progress.advance(step);
        pick(&logits, options.temperature, &mut draw_rng)
    };
    sampler::run(&plan, &mut rng, &mut draw)
}

// Each slot's value from its logits `[tokens, 8, 256]`: the most likely, or drawn at `temperature`
fn pick(logits: &Tensor, temperature: f32, rng: &mut StdRng) -> Result<Vec<[u16; SLOTS]>> {
    let values: Vec<u32> = if temperature == 0.0 {
        logits.argmax(D::Minus1)?.flatten_all()?.to_vec1()?
    } else {
        let probs =
            inference_tensor::nn::ops::softmax_last_dim(&(logits / f64::from(temperature))?)?
                .flatten_to(1)?
                .to_vec2::<f32>()?;
        probs
            .iter()
            .map(|p| {
                WeightedIndex::new(p)
                    .map(|w| w.sample(rng) as u32)
                    .map_err(inference_tensor::Error::wrap)
            })
            .collect::<Result<_>>()?
    };
    Ok(values
        .chunks_exact(SLOTS)
        .map(|c| std::array::from_fn(|i| c[i].min(CLASSES as u32 - 1) as u16))
        .collect())
}

/// A binary PLY of the voxels as cubes in the unit cube centered on the origin (`(xyz + 0.5) / 64 - 0.5`), with only
/// the faces between an occupied and an empty cell.
pub fn voxel_ply(voxels: &[[usize; 3]]) -> Vec<u8> {
    let n = sampler::OUTPUT_SIDE;
    let mut grid = vec![false; n * n * n];
    let at = |[x, y, z]: [usize; 3]| (x * n + y) * n + z;
    for &v in voxels {
        grid[at(v)] = true;
    }
    let filled = |x: isize, y: isize, z: isize| {
        let r = 0..n as isize;
        r.contains(&x)
            && r.contains(&y)
            && r.contains(&z)
            && grid[at([x as usize, y as usize, z as usize])]
    };
    let mut vertices: Vec<[f32; 3]> = Vec::new();
    let mut faces: Vec<[u32; 4]> = Vec::new();
    // each axis' two faces: its normal and the face's corners, wound outward
    for &[x, y, z] in voxels {
        let (xi, yi, zi) = (x as isize, y as isize, z as isize);
        for axis in 0..3 {
            for dir in [-1isize, 1] {
                let mut d = [0isize; 3];
                d[axis] = dir;
                if filled(xi + d[0], yi + d[1], zi + d[2]) {
                    continue;
                }
                let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
                let plane = if dir > 0 { 1.0 } else { 0.0 };
                let corners = if dir > 0 {
                    [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
                } else {
                    [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]
                };
                let base = vertices.len() as u32;
                for (a, b) in corners {
                    let mut p = [x as f32, y as f32, z as f32];
                    p[axis] += plane;
                    p[u] += a;
                    p[v] += b;
                    vertices.push(p.map(|c| c / n as f32 - 0.5));
                }
                faces.push([base, base + 1, base + 2, base + 3]);
            }
        }
    }
    let mut out = format!(
        "ply\nformat binary_little_endian 1.0\nelement vertex {}\nproperty float x\nproperty float y\n\
         property float z\nelement face {}\nproperty list uchar uint vertex_indices\nend_header\n",
        vertices.len(),
        faces.len()
    )
    .into_bytes();
    for v in &vertices {
        for c in v {
            out.extend_from_slice(&c.to_le_bytes());
        }
    }
    for f in &faces {
        out.push(4);
        for i in f {
            out.extend_from_slice(&i.to_le_bytes());
        }
    }
    out
}
