//! world_gs_trainer's single-GPU run (HY-World 2.0's 3DGS stage): Gaussians from the downsampled cloud, Adam over
//! raw parameters, L1 + SSIM + LPIPS on colour, disparity L1 on depth with the sky filled from the sky cloud, normals
//! from rendered depth against the frames' normals, a scale-ratio penalty, gsplat's default densification, and a
//! MaskGaussian gate with its sparsity term and one stochastic prune.

use std::path::Path;
use std::time::Instant;

use inference_nn::cuda::gsplat::{self, Camera, Frame, Gaussians, RasterSettings};
use inference_tensor::{D, DType, Device, Result, Tensor, Var};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

use super::data::{Mat4, SplatData, View};
use super::lpips::Lpips;
use super::mesh::{DepthFrame, Mesh, Tsdf, TsdfSettings};
use super::ssim;
use crate::geometry::invert4;
use crate::worldstereo::trajectory::PointGrid;

// the trainer's flags at 8000 single-GPU steps (README's 1500 x steps_scaler 5.3334); refine steps scale with steps
pub const DEFAULT_STEPS: usize = 8000;
const REFINE_START: usize = 800;
const REFINE_STOP: usize = 4000;
const REFINE_EVERY: usize = 533;
const SCALE2D_STOP: usize = 4000;
const MASK_FROM: usize = 300;
const MASK_UNTIL: usize = 1500;
const MASK_PRUNE_SAMPLES: i32 = 4;
const MASK_TEMPERATURE: f32 = 0.7;
const MASK_LAMBDA: f64 = 0.003;
const MASK_INIT: [f32; 2] = [4.0, 1.0];
const INIT_OPACITY: f64 = 0.1;
const SCENE_SCALE_MARGIN: f64 = 1.1;
const LR_MEANS: f64 = 1.6e-4;
const LR_SCALES: f64 = 5e-3;
const LR_QUATS: f64 = 1e-3;
const LR_OPACITIES: f64 = 5e-2;
const LR_SH0: f64 = 2.5e-3;
const LR_MASK: f64 = 1e-2;
const MEANS_LR_FINAL: f64 = 0.01;
const ADAM_BETAS: (f64, f64) = (0.9, 0.999);
const ADAM_EPS: f64 = 1e-15;
const SSIM_LAMBDA: f64 = 0.2;
const LPIPS_LAMBDAS: (f64, f64) = (0.2, 0.1);
const DEPTH_LAMBDAS: (f64, f64) = (0.2, 0.01);
const NORMAL_LAMBDAS: (f64, f64) = (0.1, 0.5);
const SCALE_REG_EVERY: usize = 10;
const SCALE_REG_WEIGHT: f64 = 0.1;
const MAX_GAUSS_RATIO: f64 = 5.0;
const GROW_GRAD2D: f32 = 1e-4;
const GROW_SCALE3D: f64 = 0.01;
const GROW_SCALE2D: f32 = 0.05;
const PRUNE_OPACITY: f32 = 0.005;
const SPLIT_SHRINK: f32 = 1.6;
const MIN_GT_DEPTH: f32 = 1e-4;
const NORMAL_MASK_MIN: f32 = 0.1;
const SKY_NORMAL_MAX: f32 = 0.1;
const ALPHA_FLOOR: f64 = 1e-10;
const SH_C0: f32 = 0.282_094_8;
const SKY_OPACITY_LOGIT: f32 = 30.0;
const SEED: u64 = 42;
// world_gs_trainer's export_mesh, in the normalized frame's units
// the share of the mesh's triangles the simplified copy keeps (its downsample_perc)
const MESH_KEEP: f64 = 0.1;
const MESH_TSDF: TsdfSettings = TsdfSettings {
    voxel: 0.05,
    sdf_trunc: 0.3,
    depth_trunc: 5.0,
};
const LOG_EVERY: usize = 100;

pub struct TrainOptions {
    pub steps: usize,
    /// VGG16 features and the LPIPS v0.1 heads in one safetensors file (see [`Lpips::load`])
    pub lpips: std::path::PathBuf,
}

/// Parameters as optimized, `[n, ...]` f32.
pub struct Splats {
    pub means: Tensor,
    pub quats: Tensor,
    pub log_scales: Tensor,
    pub logits: Tensor,
    pub sh0: Tensor,
    pub mask_score: Tensor,
    /// sky points kept from the init and their duplicates' originals: never pruned
    pub is_anchor: Vec<bool>,
}

const PARAMS: usize = 6;

impl Splats {
    fn params(&self) -> [&Tensor; PARAMS] {
        [
            &self.means,
            &self.quats,
            &self.log_scales,
            &self.logits,
            &self.sh0,
            &self.mask_score,
        ]
    }

    fn set_params(&mut self, p: [Tensor; PARAMS]) {
        let [means, quats, log_scales, logits, sh0, mask_score] = p;
        *self = Self {
            means,
            quats,
            log_scales,
            logits,
            sh0,
            mask_score,
            is_anchor: std::mem::take(&mut self.is_anchor),
        };
    }

    pub fn len(&self) -> usize {
        self.is_anchor.len()
    }

    pub fn is_empty(&self) -> bool {
        self.is_anchor.is_empty()
    }

    fn gaussians(&self) -> Gaussians<'_> {
        Gaussians {
            means: &self.means,
            quats: &self.quats,
            log_scales: &self.log_scales,
            logit_opacities: &self.logits,
            sh0: &self.sh0,
        }
    }

    /// The keep probability, softmax(mask_score)[:, 0].
    pub fn keep_prob(&self) -> Result<Tensor> {
        let s0 = self.mask_score.narrow(1, 0, 1)?;
        let s1 = self.mask_score.narrow(1, 1, 1)?;
        inference_tensor::nn::ops::sigmoid(&(s0 - s1)?)?.squeeze(1)
    }
}

// every row's Adam moments
struct Moments {
    m: [Tensor; PARAMS],
    v: [Tensor; PARAMS],
}

struct ViewData {
    camera: Camera,
    image: Tensor,
    /// the frame's depth (rescaled; 0 invalid) with the sky filled from the sky cloud, and where the frame's own
    /// depth was valid before that
    depth: Option<(Tensor, Tensor)>,
    /// unit normals `[3, h, w]` and where the PNG's raw vector was long enough
    normals: Option<(Tensor, Tensor)>,
    /// `inv(K) [x, y, 1]` at integer pixels, `[3, h, w]`
    rays: Tensor,
}

fn camera_of(view: &View) -> Camera {
    let w2c = invert4(&view.c2w);
    let mut viewmat = [0f32; 16];
    for r in 0..4 {
        for c in 0..4 {
            viewmat[r * 4 + c] = w2c[r][c] as f32;
        }
    }
    let mut k = [0f32; 9];
    for r in 0..3 {
        for c in 0..3 {
            k[r * 3 + c] = view.k[r][c] as f32;
        }
    }
    Camera {
        viewmat,
        k,
        width: view.width,
        height: view.height,
    }
}

fn decode_rgb(path: &Path) -> Result<(Vec<f32>, usize, usize)> {
    let img = image::open(path)
        .map_err(inference_tensor::Error::wrap)?
        .to_rgb8();
    let (w, h) = img.dimensions();
    Ok((
        img.into_raw()
            .into_iter()
            .map(|v| f32::from(v) / 255.0)
            .collect(),
        w as usize,
        h as usize,
    ))
}

// float16 bits stored as 16-bit grayscale
fn decode_depth(path: &Path, rescale: f32) -> Result<Vec<f32>> {
    let img = image::open(path)
        .map_err(inference_tensor::Error::wrap)?
        .to_luma16();
    Ok(img
        .into_raw()
        .into_iter()
        .map(|b| half::f16::from_bits(b).to_f32() * rescale)
        .collect())
}

fn rays_of(view: &View, device: &Device) -> Result<Tensor> {
    let kinv = {
        let mut m = [[0.0; 4]; 4];
        for (row, k) in m.iter_mut().zip(&view.k) {
            row[..3].copy_from_slice(k);
        }
        m[3][3] = 1.0;
        invert4(&m)
    };
    let (w, h) = (view.width, view.height);
    let mut out = vec![0f32; 3 * w * h];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                out[c * w * h + y * w + x] =
                    (kinv[c][0] * x as f64 + kinv[c][1] * y as f64 + kinv[c][2]) as f32;
            }
        }
    }
    Tensor::from_vec(out, (3, h, w), device)
}

fn mat4_f32(m: &Mat4) -> [[f32; 4]; 4] {
    m.map(|r| r.map(|v| v as f32))
}

/// A step's loss and its unweighted terms (1 - SSIM for ssim), and the gate's open count.
#[derive(Clone, Copy, Debug, Default)]
pub struct Terms {
    pub loss: f64,
    pub l1: f64,
    pub ssim: f64,
    pub lpips: f64,
    pub depth: f64,
    pub normal: f64,
    pub active: usize,
}

pub struct Trainer {
    data: SplatData,
    views: Vec<ViewData>,
    pub splats: Splats,
    moments: Moments,
    lpips: Lpips,
    window: Tensor,
    settings: RasterSettings,
    scene_scale: f64,
    steps: usize,
    device: Device,
    rng: StdRng,
    adam_t: i32,
    grad2d: Tensor,
    count: Tensor,
    max_radii: Tensor,
}

fn zeros_like_all(p: [&Tensor; PARAMS]) -> Result<[Tensor; PARAMS]> {
    Ok([
        p[0].zeros_like()?,
        p[1].zeros_like()?,
        p[2].zeros_like()?,
        p[3].zeros_like()?,
        p[4].zeros_like()?,
        p[5].zeros_like()?,
    ])
}

// rows of `t` at `idx` (host indices)
fn rows(t: &Tensor, idx: &[u32]) -> Result<Tensor> {
    if idx.is_empty() {
        let mut dims = t.dims().to_vec();
        dims[0] = 0;
        return Tensor::zeros(dims, t.dtype(), t.device());
    }
    let i = Tensor::new(idx, t.device())?;
    t.index_select(&i, 0)
}

fn cat0(parts: &[&Tensor]) -> Result<Tensor> {
    Tensor::cat(parts, 0)?.contiguous()
}

impl Trainer {
    pub fn new(data: SplatData, options: &TrainOptions, device: &Device) -> Result<Self> {
        let t = Instant::now();
        let scene_scale = data.scene_scale * SCENE_SCALE_MARGIN;
        let n = data.points.len();
        // scales: the rms distance to the three nearest other points
        let grid = PointGrid::new(&data.points);
        let log_scales = data
            .points
            .par_iter()
            .map(|p| {
                let d = grid.knn_sq(p.map(f64::from), 4);
                let ms = d.iter().skip(1).sum::<f64>() / d.len().saturating_sub(1).max(1) as f64;
                (ms.sqrt().ln()) as f32
            })
            .collect::<Vec<_>>();
        let mut rng = StdRng::seed_from_u64(SEED);
        let quats = (0..n * 4).map(|_| rng.random::<f32>()).collect::<Vec<_>>();
        let logit = (INIT_OPACITY / (1.0 - INIT_OPACITY)).ln() as f32;
        let sh0 = data
            .colors
            .iter()
            .flat_map(|c| c.map(|v| (f32::from(v) / 255.0 - 0.5) / SH_C0))
            .collect::<Vec<_>>();
        let splats = Splats {
            means: Tensor::from_vec(data.points.concat(), (n, 3), device)?,
            quats: Tensor::from_vec(quats, (n, 4), device)?,
            log_scales: Tensor::from_vec(
                log_scales.iter().flat_map(|&s| [s; 3]).collect(),
                (n, 3),
                device,
            )?,
            logits: Tensor::full(logit, n, device)?,
            sh0: Tensor::from_vec(sh0, (n, 3), device)?,
            mask_score: Tensor::from_vec(MASK_INIT.repeat(n), (n, 2), device)?,
            is_anchor: data.is_sky.clone(),
        };
        let zeros = zeros_like_all(splats.params())?;
        let moments = Moments {
            m: zeros.clone(),
            v: zeros,
        };
        let views = Self::load_views(&data, device)?;
        let lpips = Lpips::load(&options.lpips, device)?;
        let state = Tensor::zeros(n, DType::F32, device)?;
        let mut trainer = Self {
            window: ssim::window(3, device)?,
            views,
            splats,
            moments,
            lpips,
            settings: RasterSettings::default(),
            scene_scale,
            steps: options.steps,
            device: device.clone(),
            rng,
            adam_t: 0,
            grad2d: state.clone(),
            count: state.clone(),
            max_radii: state,
            data,
        };
        trainer.fill_sky_depth()?;
        eprintln!(
            "splat init: {n} gaussians ({} sky), {} train / {} val views, scene scale {:.4}, {:.1} s",
            trainer.data.is_sky.iter().filter(|s| **s).count(),
            trainer.data.train.len(),
            trainer.data.val.len(),
            scene_scale,
            t.elapsed().as_secs_f64()
        );
        Ok(trainer)
    }

    fn load_views(data: &SplatData, device: &Device) -> Result<Vec<ViewData>> {
        let rescale = data.rescale as f32;
        let decoded = data
            .views
            .par_iter()
            .map(|v| -> Result<_> {
                let (rgb, w, h) = decode_rgb(&v.image)?;
                let depth = v
                    .depth
                    .as_deref()
                    .map(|p| decode_depth(p, rescale))
                    .transpose()?;
                let normal = v.normal.as_deref().map(decode_rgb).transpose()?;
                Ok((rgb, w, h, depth, normal))
            })
            .collect::<Result<Vec<_>>>()?;
        data.views
            .iter()
            .zip(decoded)
            .map(|(v, (rgb, w, h, depth, normal))| {
                let image = Tensor::from_vec(rgb, (h, w, 3), device)?;
                let depth = depth
                    .map(|d| -> Result<_> {
                        let t = Tensor::from_vec(d, (h, w), device)?;
                        let valid = t.gt(f64::from(MIN_GT_DEPTH))?.to_dtype(DType::F32)?;
                        Ok((t, valid))
                    })
                    .transpose()?;
                let normals = normal
                    .map(|(n, _, _)| -> Result<_> {
                        let n = Tensor::from_vec(n, (h, w, 3), device)?
                            .permute((2, 0, 1))?
                            .affine(2.0, -1.0)?
                            .contiguous()?;
                        let len2 = n.sqr()?.sum_keepdim(0)?;
                        let mask = len2
                            .gt(f64::from(NORMAL_MASK_MIN))?
                            .to_dtype(DType::F32)?
                            .squeeze(0)?;
                        // F.normalize's 1e-12 floor
                        let unit = n.broadcast_div(&len2.sqrt()?.clamp(1e-12f32, f32::MAX)?)?;
                        Ok((unit, mask))
                    })
                    .transpose()?;
                Ok(ViewData {
                    camera: camera_of(v),
                    image,
                    depth,
                    normals,
                    rays: rays_of(v, device)?,
                })
            })
            .collect()
    }

    // sky_depth_from_pcd: the sky points as opaque isotropic Gaussians rendered (classic mode) into each training
    // frame with a depth, their expected depth filling the frame's invalid sky pixels (no depth, a short raw normal)
    fn fill_sky_depth(&mut self) -> Result<()> {
        let sky = self
            .data
            .points
            .iter()
            .zip(&self.data.is_sky)
            .filter(|(_, s)| **s)
            .map(|(p, _)| *p)
            .collect::<Vec<_>>();
        if sky.is_empty() {
            return Ok(());
        }
        let n = sky.len();
        let grid = PointGrid::new(&sky);
        let log_scales = sky
            .par_iter()
            .map(|p| {
                let d = grid.knn_sq(p.map(f64::from), 4);
                (d.iter().skip(1).sum::<f64>() / 3.0).sqrt().ln() as f32
            })
            .flat_map_iter(|s| [s; 3])
            .collect::<Vec<_>>();
        let dev = &self.device;
        let means = Tensor::from_vec(sky.concat(), (n, 3), dev)?;
        let quats = Tensor::from_vec([1f32, 0., 0., 0.].repeat(n), (n, 4), dev)?;
        let log_scales = Tensor::from_vec(log_scales, (n, 3), dev)?;
        let logits = Tensor::full(SKY_OPACITY_LOGIT, n, dev)?;
        let sh0 = Tensor::full(-0.5 / SH_C0, (n, 3), dev)?;
        let settings = RasterSettings {
            antialiased: false,
            ..RasterSettings::default()
        };
        let train = self.data.train.clone();
        let mut filled = 0usize;
        for i in train {
            let view = &self.views[i];
            let (Some((depth, valid)), Some(_)) = (&view.depth, &view.normals) else {
                continue;
            };
            let r = gsplat::render(&Frame {
                gaussians: Gaussians {
                    means: &means,
                    quats: &quats,
                    log_scales: &log_scales,
                    logit_opacities: &logits,
                    sh0: &sh0,
                },
                gate: None,
                camera: view.camera,
                settings,
            })?;
            let ed = r
                .image
                .narrow(2, 3, 1)?
                .squeeze(2)?
                .div(&r.alpha.clamp(ALPHA_FLOOR as f32, f32::MAX)?)?;
            // the raw normal's length: the unit normal times the mask is not it, so decode it again
            let raw = self.data.views[i]
                .normal
                .as_deref()
                .map(decode_rgb)
                .transpose()?;
            let Some((raw, w, h)) = raw else { continue };
            let raw = Tensor::from_vec(raw, (h, w, 3), dev)?.affine(2.0, -1.0)?;
            let short = raw
                .sqr()?
                .sum(D::Minus1)?
                .sqrt()?
                .lt(f64::from(SKY_NORMAL_MAX))?
                .to_dtype(DType::F32)?;
            let fill = ((1.0 - valid)? * short)?.mul(&ed.gt(0f64)?.to_dtype(DType::F32)?)?;
            filled += fill.sum_all()?.to_scalar::<f32>()? as usize;
            let merged = (depth.mul(&(1.0 - &fill)?)? + ed.mul(&fill)?)?;
            self.views[i].depth = Some((merged, valid.clone()));
        }
        eprintln!("sky depth: {n} sky gaussians, {filled} pixels filled");
        Ok(())
    }

    // gumbel-max sample of the 2-class gate (0/1) and the straight-through derivative of its soft version
    fn sample_gate(&self) -> Result<(Tensor, Tensor)> {
        let n = self.splats.len();
        let u = Tensor::rand(1e-20f32, 1.0, (n, 2), &self.device)?;
        let g = u.log()?.neg()?.log()?.neg()?;
        let s = &self.splats.mask_score;
        let d = ((s.narrow(1, 0, 1)? - s.narrow(1, 1, 1)?)?
            + (g.narrow(1, 0, 1)? - g.narrow(1, 1, 1)?)?)?
        .squeeze(1)?;
        let gate = d.gt(0f64)?.to_dtype(DType::F32)?;
        let soft = inference_tensor::nn::ops::sigmoid(&(d / f64::from(MASK_TEMPERATURE))?)?;
        let dsoft = (soft.mul(&(1.0 - &soft)?)? / f64::from(MASK_TEMPERATURE))?;
        Ok((gate, dsoft))
    }

    fn frame<'a>(&'a self, view: &ViewData, gate: Option<&'a Tensor>) -> Frame<'a> {
        Frame {
            gaussians: self.splats.gaussians(),
            gate,
            camera: view.camera,
            settings: self.settings,
        }
    }

    fn weights(&self, step: usize) -> (f64, f64, f64) {
        let decay = (self.steps - step) as f64 / self.steps as f64;
        let lerp = |(start, end): (f64, f64)| end + decay * (start - end);
        (
            lerp(LPIPS_LAMBDAS),
            lerp(DEPTH_LAMBDAS),
            lerp(NORMAL_LAMBDAS),
        )
    }

    /// One optimization step on `view_index`; the loss.
    /// One optimization step on `view_index`; the loss and its terms (see [`Terms`]).
    pub fn step(&mut self, step: usize, view_index: usize) -> Result<Terms> {
        let mut terms = Terms::default();
        let (gate, dsoft) = self.sample_gate()?;
        let view = &self.views[view_index];
        let frame = self.frame(view, Some(&gate));
        let render = gsplat::render(&frame)?;
        let image_var = Var::from_tensor(&render.image)?;
        let alpha_var = Var::from_tensor(&render.alpha)?;
        let (w_lpips, w_depth, w_normal) = self.weights(step);

        let colors = image_var.narrow(2, 0, 3)?;
        let pixels = &view.image;
        let l1 = (&colors - pixels)?.abs()?.mean_all()?;
        let chw = |t: &Tensor| t.permute((2, 0, 1)).and_then(|t| t.unsqueeze(0));
        let ssim = ssim::ssim(&chw(&colors)?, &chw(pixels)?, &self.window)?;
        let lpips = self
            .lpips
            .forward(&chw(&colors.clamp(0f32, 1f32)?)?, &chw(pixels)?)?;
        terms.l1 = f64::from(l1.to_scalar::<f32>()?);
        terms.ssim = 1.0 - f64::from(ssim.to_scalar::<f32>()?);
        terms.lpips = f64::from(lpips.to_scalar::<f32>()?);
        terms.active = gate.sum_all()?.to_scalar::<f32>()? as usize;
        let mut loss = ((l1 * (1.0 - SSIM_LAMBDA))? + ((1.0 - ssim)? * SSIM_LAMBDA)?)?;
        loss = (loss + (lpips * w_lpips)?)?;
        let depth = image_var
            .narrow(2, 3, 1)?
            .squeeze(2)?
            .div(&alpha_var.clamp(ALPHA_FLOOR as f32, f32::MAX)?)?;
        if let Some((gt, valid_before_sky)) = &view.depth {
            let m = depth
                .detach()
                .gt(0f64)?
                .to_dtype(DType::F32)?
                .mul(&gt.gt(f64::from(MIN_GT_DEPTH))?.to_dtype(DType::F32)?)?;
            let count = m.sum_all()?.to_scalar::<f32>()?;
            if count > 0.0 {
                let safe = |d: &Tensor| (d.mul(&m)? + (1.0 - &m)?)?.recip();
                let disp = (safe(&depth)? - safe(gt)?)?.abs()?.mul(&m)?.sum_all()?;
                let depth_loss = ((disp / f64::from(count))? * self.scene_scale)?;
                terms.depth = f64::from(depth_loss.to_scalar::<f32>()?);
                loss = (loss + (depth_loss * w_depth)?)?;
            }
            if let Some((normals_gt, normal_mask_gt)) = &view.normals {
                let valid = depth
                    .detach()
                    .gt(0f64)?
                    .to_dtype(DType::F32)?
                    .mul(valid_before_sky)?;
                if let Some(l) =
                    normal_loss(&depth, &valid, &view.rays, normals_gt, normal_mask_gt)?
                {
                    terms.normal = f64::from(l.to_scalar::<f32>()?);
                    loss = (loss + (l * w_normal)?)?;
                }
            }
        }
        terms.loss = f64::from(loss.to_scalar::<f32>()?);
        let grads = loss.backward()?;
        let v_image = grads
            .get(&image_var)
            .cloned()
            .unwrap_or(render.image.zeros_like()?)
            .contiguous()?;
        let v_alpha = grads
            .get(&alpha_var)
            .cloned()
            .unwrap_or(render.alpha.zeros_like()?)
            .contiguous()?;
        let g = gsplat::render_backward(&frame, &render, &v_image, &v_alpha, false)?;

        // the gate: rasterizer gradient plus the sparsity term (mean gate)^2, through the straight-through soft gate
        let mut v_gate = g.gate.clone().expect("gated render");
        let n = self.splats.len() as f64;
        if (MASK_FROM..=MASK_UNTIL).contains(&step) {
            let mean = gate.mean_all()?.to_scalar::<f32>()? as f64;
            v_gate = (v_gate + 2.0 * mean / n * MASK_LAMBDA)?;
        }
        let v_s0 = v_gate.mul(&dsoft)?.unsqueeze(1)?;
        let v_mask = Tensor::cat(&[&v_s0, &v_s0.neg()?], 1)?;

        let mut v_log_scales = g.log_scales.clone();
        if step.is_multiple_of(SCALE_REG_EVERY) {
            v_log_scales = (v_log_scales + scale_reg_grad(&self.splats.log_scales)?)?;
        }
        let grads = [
            g.means.clone(),
            g.quats.clone(),
            v_log_scales,
            g.logit_opacities.clone(),
            g.sh0.clone(),
            v_mask,
        ];
        if step < REFINE_STOP * self.steps / DEFAULT_STEPS {
            gsplat::densify_stats(
                &render,
                &g.means2d,
                &view.camera,
                &self.grad2d,
                &self.count,
                &self.max_radii,
            )?;
        }
        self.adam(step, &grads)?;
        Ok(terms)
    }

    fn adam(&mut self, step: usize, grads: &[Tensor; PARAMS]) -> Result<()> {
        self.adam_t += 1;
        let (b1, b2) = ADAM_BETAS;
        let bc1 = 1.0 - b1.powi(self.adam_t);
        let bc2 = (1.0 - b2.powi(self.adam_t)).sqrt();
        let gamma = MEANS_LR_FINAL.powf(1.0 / self.steps as f64);
        let lrs = [
            LR_MEANS * self.scene_scale * gamma.powi(step as i32),
            LR_QUATS,
            LR_SCALES,
            LR_OPACITIES,
            LR_SH0,
            LR_MASK,
        ];
        let params = self.splats.params();
        let mut new = Vec::with_capacity(PARAMS);
        for k in 0..PARAMS {
            let g = &grads[k];
            let m = ((&self.moments.m[k] * b1)? + (g * (1.0 - b1))?)?;
            let v = ((&self.moments.v[k] * b2)? + (g.sqr()? * (1.0 - b2))?)?;
            let update = m.div(&((v.sqrt()? / bc2)? + ADAM_EPS)?)?;
            new.push((params[k] - (update * (lrs[k] / bc1))?)?);
            self.moments.m[k] = m;
            self.moments.v[k] = v;
        }
        let new: [Tensor; PARAMS] = new
            .try_into()
            .map_err(|_| inference_tensor::Error::msg("params"))?;
        self.splats.set_params(new);
        Ok(())
    }

    fn host_f32(t: &Tensor) -> Result<Vec<f32>> {
        t.flatten_all()?.to_vec1::<f32>()
    }

    // gsplat DefaultStrategy's grow then prune, then the statistics reset
    fn refine(&mut self, step: usize) -> Result<(usize, usize, usize)> {
        let n = self.splats.len();
        // the 2D-radius split rule runs until a step that does not scale with the run's length
        let scale2d_live = step < SCALE2D_STOP;
        let grad2d = Self::host_f32(&self.grad2d)?;
        let count = Self::host_f32(&self.count)?;
        let max_radii = Self::host_f32(&self.max_radii)?;
        let log_scales = Self::host_f32(&self.splats.log_scales)?;
        let big = GROW_SCALE3D * self.scene_scale;
        let mut dupli = Vec::new();
        let mut split = Vec::new();
        for i in 0..n {
            let high = grad2d[i] / count[i].max(1.0) > GROW_GRAD2D;
            let smax = (0..3)
                .map(|d| log_scales[3 * i + d])
                .fold(f32::MIN, f32::max)
                .exp();
            let small = f64::from(smax) <= big;
            if high && small {
                dupli.push(i as u32);
            }
            if (high && !small) || (scale2d_live && max_radii[i] > GROW_SCALE2D) {
                split.push(i as u32);
            }
        }
        // duplicate: copies appended, moments zero, the copies not anchors
        if !dupli.is_empty() {
            let p = self.splats.params();
            let grown = std::array::from_fn::<_, PARAMS, _>(|k| {
                rows(p[k], &dupli).and_then(|r| cat0(&[p[k], &r]))
            });
            let grown = grown.into_iter().collect::<Result<Vec<_>>>()?;
            for k in 0..PARAMS {
                let z = rows(&self.moments.m[k], &dupli)?.zeros_like()?;
                self.moments.m[k] = cat0(&[&self.moments.m[k], &z])?;
                self.moments.v[k] = cat0(&[&self.moments.v[k], &z])?;
            }
            self.splats.set_params(
                grown
                    .try_into()
                    .map_err(|_| inference_tensor::Error::msg("params"))?,
            );
            self.splats
                .is_anchor
                .extend(std::iter::repeat_n(false, dupli.len()));
        }
        // split: two children drawn inside each parent, 1.6x smaller; the parents removed
        let n_split = split.len();
        if n_split > 0 {
            let n_now = self.splats.len();
            let mut is_split = vec![false; n_now];
            for &i in &split {
                is_split[i as usize] = true;
            }
            let rest = (0..n_now as u32)
                .filter(|&i| !is_split[i as usize])
                .collect::<Vec<_>>();
            let means = Self::host_f32(&rows(&self.splats.means, &split)?)?;
            let quats = Self::host_f32(&rows(&self.splats.quats, &split)?)?;
            let scales = Self::host_f32(&rows(&self.splats.log_scales, &split)?)?;
            let mut child_means = vec![0f32; 2 * n_split * 3];
            for c in 0..2 {
                for j in 0..n_split {
                    let q = &quats[4 * j..4 * j + 4];
                    let r = quat_to_rotmat(q);
                    let s = [0, 1, 2].map(|d| scales[3 * j + d].exp());
                    let z: [f32; 3] = std::array::from_fn(|_| StandardNormal.sample(&mut self.rng));
                    for row in 0..3 {
                        let off = (0..3).map(|col| r[row][col] * s[col] * z[col]).sum::<f32>();
                        child_means[(c * n_split + j) * 3 + row] = means[3 * j + row] + off;
                    }
                }
            }
            let child_scales = scales
                .iter()
                .map(|s| (s.exp() / SPLIT_SHRINK).ln())
                .collect::<Vec<_>>();
            let dev = &self.device;
            let child_means = Tensor::from_vec(child_means, (2 * n_split, 3), dev)?;
            let child_scales = Tensor::from_vec(child_scales.repeat(2), (2 * n_split, 3), dev)?;
            let p = self.splats.params();
            let mut out = Vec::with_capacity(PARAMS);
            for (k, t) in p.iter().enumerate() {
                let kept = rows(t, &rest)?;
                let children = match k {
                    0 => child_means.clone(),
                    2 => child_scales.clone(),
                    _ => {
                        let r = rows(t, &split)?;
                        cat0(&[&r, &r])?
                    }
                };
                out.push(cat0(&[&kept, &children])?);
            }
            for (k, param) in out.iter().enumerate() {
                let z = Tensor::zeros(param.dims(), DType::F32, dev)?.narrow(0, 0, 2 * n_split)?;
                self.moments.m[k] = cat0(&[&rows(&self.moments.m[k], &rest)?, &z])?;
                self.moments.v[k] = cat0(&[&rows(&self.moments.v[k], &rest)?, &z])?;
            }
            let anchors = rest
                .iter()
                .map(|&i| self.splats.is_anchor[i as usize])
                .chain(std::iter::repeat_n(false, 2 * n_split))
                .collect();
            self.splats.set_params(
                out.try_into()
                    .map_err(|_| inference_tensor::Error::msg("params"))?,
            );
            self.splats.is_anchor = anchors;
        }
        // prune the transparent, anchors exempt
        let opac = Self::host_f32(&inference_tensor::nn::ops::sigmoid(&self.splats.logits)?)?;
        let prune = opac
            .iter()
            .zip(&self.splats.is_anchor)
            .map(|(o, a)| *o < PRUNE_OPACITY && !a)
            .collect::<Vec<_>>();
        // the statistics are cleared after a refine anyway; cleared first, they match the grown rows
        self.reset_stats()?;
        let pruned = self.remove(&prune)?;
        Ok((dupli.len(), n_split, pruned))
    }

    fn remove(&mut self, drop: &[bool]) -> Result<usize> {
        let keep = (0..drop.len() as u32)
            .filter(|&i| !drop[i as usize])
            .collect::<Vec<_>>();
        let removed = drop.len() - keep.len();
        if removed == 0 {
            return Ok(0);
        }
        let p = self.splats.params();
        let kept = p
            .iter()
            .map(|t| rows(t, &keep).and_then(|t| t.contiguous()))
            .collect::<Result<Vec<_>>>()?;
        for k in 0..PARAMS {
            self.moments.m[k] = rows(&self.moments.m[k], &keep)?.contiguous()?;
            self.moments.v[k] = rows(&self.moments.v[k], &keep)?.contiguous()?;
        }
        let anchors = keep
            .iter()
            .map(|&i| self.splats.is_anchor[i as usize])
            .collect();
        self.splats.set_params(
            kept.try_into()
                .map_err(|_| inference_tensor::Error::msg("params"))?,
        );
        self.splats.is_anchor = anchors;
        self.grad2d = rows(&self.grad2d, &keep)?.contiguous()?;
        self.count = rows(&self.count, &keep)?.contiguous()?;
        self.max_radii = rows(&self.max_radii, &keep)?.contiguous()?;
        Ok(removed)
    }

    fn reset_stats(&mut self) -> Result<()> {
        let z = Tensor::zeros(self.splats.len(), DType::F32, &self.device)?;
        self.grad2d = z.clone();
        self.count = z.clone();
        self.max_radii = z;
        Ok(())
    }

    // MaskGaussian's prune: drop a Gaussian when all of four gate samples close it, anchors exempt
    fn mask_prune(&mut self) -> Result<usize> {
        let keep = Self::host_f32(&self.splats.keep_prob()?)?;
        let drop = keep
            .iter()
            .zip(&self.splats.is_anchor)
            .map(|(&p, &a)| !a && (0..MASK_PRUNE_SAMPLES).all(|_| self.rng.random::<f32>() >= p))
            .collect::<Vec<_>>();
        self.remove(&drop)
    }

    /// Mean PSNR over the validation views with the deterministic gate.
    pub fn eval(&self) -> Result<f64> {
        let s = &self.splats.mask_score;
        let gate = s
            .narrow(1, 0, 1)?
            .gt(&s.narrow(1, 1, 1)?)?
            .to_dtype(DType::F32)?
            .squeeze(1)?;
        let mut total = 0.0;
        for &i in &self.data.val {
            let v = &self.views[i];
            let r = gsplat::render(&self.frame(v, Some(&gate)))?;
            let rgb = r.image.narrow(2, 0, 3)?.clamp(0f32, 1f32)?;
            let mse = (rgb - &v.image)?.sqr()?.mean_all()?.to_scalar::<f32>()?;
            total += -10.0 * f64::from(mse).log10();
        }
        Ok(total / self.data.val.len().max(1) as f64)
    }

    /// Runs the schedule; the final mean validation PSNR.
    pub fn train(&mut self) -> Result<f64> {
        let scale = |s: usize| s * self.steps / DEFAULT_STEPS;
        let (start, stop, every) = (
            scale(REFINE_START),
            scale(REFINE_STOP),
            scale(REFINE_EVERY).max(1),
        );
        let mut order: Vec<usize> = Vec::new();
        let started = Instant::now();
        let mut sum = Terms::default();
        for step in 0..self.steps {
            if order.is_empty() {
                order = self.data.train.clone();
                order.shuffle(&mut self.rng);
            }
            let view = order.pop().expect("a training view");
            let t = self.step(step, view)?;
            sum.loss += t.loss;
            sum.l1 += t.l1;
            sum.ssim += t.ssim;
            sum.lpips += t.lpips;
            sum.depth += t.depth;
            sum.normal += t.normal;
            sum.active = t.active;
            if step < stop && step > start && step % every == 0 {
                let (d, s, p) = self.refine(step)?;
                eprintln!(
                    "step {step}: {d} duplicated, {s} split, {p} pruned, {} gaussians",
                    self.splats.len()
                );
                if (MASK_FROM..=MASK_UNTIL).contains(&step) {
                    let p = self.mask_prune()?;
                    eprintln!(
                        "step {step}: mask prune {p}, {} gaussians",
                        self.splats.len()
                    );
                }
            }
            if (step + 1) % LOG_EVERY == 0 {
                let k = LOG_EVERY as f64;
                eprintln!(
                    "step {}: loss {:.4} (l1 {:.4} ssim {:.4} lpips {:.4} depth {:.4} normal {:.4}), {} of {} open, {:.1} s",
                    step + 1,
                    sum.loss / k,
                    sum.l1 / k,
                    sum.ssim / k,
                    sum.lpips / k,
                    sum.depth / k,
                    sum.normal / k,
                    sum.active,
                    self.splats.len(),
                    started.elapsed().as_secs_f64()
                );
                sum = Terms::default();
            }
        }
        self.eval()
    }

    pub fn transform(&self) -> [[f32; 4]; 4] {
        mat4_f32(&self.data.transform)
    }
}

fn quat_to_rotmat(q: &[f32]) -> [[f32; 3]; 3] {
    let n = (q.iter().map(|v| v * v).sum::<f32>()).sqrt();
    let (w, x, y, z) = (q[0] / n, q[1] / n, q[2] / n, q[3] / n);
    [
        [
            1. - 2. * (y * y + z * z),
            2. * (x * y - w * z),
            2. * (x * z + w * y),
        ],
        [
            2. * (x * y + w * z),
            1. - 2. * (x * x + z * z),
            2. * (y * z - w * x),
        ],
        [
            2. * (x * z - w * y),
            2. * (y * z + w * x),
            1. - 2. * (x * x + y * y),
        ],
    ]
}

// d/d(log s) of 0.1 * mean(max(smax / smin, 5) - 5)
fn scale_reg_grad(log_scales: &Tensor) -> Result<Tensor> {
    let v = Var::from_tensor(log_scales)?;
    let s = v.exp()?;
    let ratio = s.max(D::Minus1)?.div(&s.min(D::Minus1)?)?;
    let reg =
        ((ratio.maximum(MAX_GAUSS_RATIO)? - MAX_GAUSS_RATIO)?.mean_all()? * SCALE_REG_WEIGHT)?;
    // a gradient built from tracked tensors carries their graph; the moments would chain it across steps
    Ok(reg
        .backward()?
        .get(&v)
        .map(Tensor::detach)
        .unwrap_or(log_scales.zeros_like()?))
}

fn cross(a: &[Tensor; 3], b: &[Tensor; 3]) -> Result<[Tensor; 3]> {
    Ok([
        ((&a[1] * &b[2])? - (&a[2] * &b[1])?)?,
        ((&a[2] * &b[0])? - (&a[0] * &b[2])?)?,
        ((&a[0] * &b[1])? - (&a[1] * &b[0])?)?,
    ])
}

fn normalize3(v: [Tensor; 3], eps: f32) -> Result<[Tensor; 3]> {
    // max(|v|, eps) as sqrt(max(|v|^2, eps^2)): sqrt's infinite slope at a zero vector would meet the clamp's zero
    let len = ((v[0].sqr()? + v[1].sqr()?)? + v[2].sqr()?)?
        .clamp(eps * eps, f32::MAX)?
        .sqrt()?;
    Ok([v[0].div(&len)?, v[1].div(&len)?, v[2].div(&len)?])
}

// gs/utils.py's depth_to_normal: camera points from the rendered depth, the four neighbour cross products averaged
// over the valid ones, turned to face the camera; then 1 - cos against the frames' normals where both are valid
fn normal_loss(
    depth: &Tensor,
    valid: &Tensor,
    rays: &Tensor,
    normals_gt: &Tensor,
    normal_mask_gt: &Tensor,
) -> Result<Option<Tensor>> {
    let (h, w) = depth.dims2()?;
    let xyz = rays.broadcast_mul(&depth.unsqueeze(0)?)?;
    let pad = |t: &Tensor| {
        t.pad_with_zeros(D::Minus1, 1, 1)?
            .pad_with_zeros(D::Minus2, 1, 1)
    };
    let (xp, mp) = (pad(&xyz)?, pad(&valid.unsqueeze(0)?)?);
    let at = |t: &Tensor, dy: usize, dx: usize| t.narrow(1, dy, h)?.narrow(2, dx, w);
    let split = |t: Tensor| -> Result<[Tensor; 3]> {
        Ok([t.narrow(0, 0, 1)?, t.narrow(0, 1, 1)?, t.narrow(0, 2, 1)?])
    };
    let c = split(at(&xp, 1, 1)?)?;
    let sub = |a: &[Tensor; 3]| -> Result<[Tensor; 3]> {
        Ok([(&a[0] - &c[0])?, (&a[1] - &c[1])?, (&a[2] - &c[2])?])
    };
    let (up, left, down, right) = (
        sub(&split(at(&xp, 0, 1)?)?)?,
        sub(&split(at(&xp, 1, 0)?)?)?,
        sub(&split(at(&xp, 2, 1)?)?)?,
        sub(&split(at(&xp, 1, 2)?)?)?,
    );
    let (mu, ml, md, mr, mc) = (
        at(&mp, 0, 1)?,
        at(&mp, 1, 0)?,
        at(&mp, 2, 1)?,
        at(&mp, 1, 2)?,
        at(&mp, 1, 1)?,
    );
    let pairs = [
        (&up, &left, &mu, &ml),
        (&left, &down, &ml, &md),
        (&down, &right, &md, &mr),
        (&right, &up, &mr, &mu),
    ];
    let mut sum: Option<[Tensor; 3]> = None;
    let mut weight: Option<Tensor> = None;
    for (a, b, ma, mb) in pairs {
        let n = normalize3(cross(a, b)?, 1e-6)?;
        let v = ma.mul(&mc)?.mul(mb)?;
        let nv = [n[0].mul(&v)?, n[1].mul(&v)?, n[2].mul(&v)?];
        sum = Some(match sum {
            Some(s) => [(&s[0] + &nv[0])?, (&s[1] + &nv[1])?, (&s[2] + &nv[2])?],
            None => nv,
        });
        weight = Some(match weight {
            Some(w) => (w + v)?,
            None => v,
        });
    }
    let (sum, weight) = (sum.expect("four pairs"), weight.expect("four pairs"));
    let avg = normalize3(
        [
            sum[0].div(&(&weight + 1e-8)?)?,
            sum[1].div(&(&weight + 1e-8)?)?,
            sum[2].div(&(&weight + 1e-8)?)?,
        ],
        1e-12,
    )?;
    // facing the camera: dot with -xyz non-negative
    let facing = ((&avg[0] * &c[0])? + (&avg[1] * &c[1])? + (&avg[2] * &c[2])?)?.neg()?;
    let sign = facing
        .detach()
        .ge(0f64)?
        .to_dtype(DType::F32)?
        .affine(2.0, -1.0)?;
    let pred_mask = weight.gt(0f64)?.to_dtype(DType::F32)?;
    let valid = pred_mask.mul(&normal_mask_gt.unsqueeze(0)?)?;
    let count = valid.sum_all()?.to_scalar::<f32>()?;
    if count <= 0.0 {
        return Ok(None);
    }
    let pred = Tensor::cat(&avg, 0)?
        .broadcast_mul(&sign)?
        .broadcast_mul(&pred_mask)?;
    let dot = pred
        .mul(&normals_gt.broadcast_mul(&pred_mask)?)?
        .sum_keepdim(0)?;
    Ok(Some(
        ((1.0 - dot)?.mul(&valid)?.sum_all()? / f64::from(count))?,
    ))
}

/// What a training run reports.
#[derive(Debug, serde::Serialize)]
pub struct SplatReport {
    pub steps: usize,
    pub gaussians: usize,
    pub exported: usize,
    pub val_psnr: f64,
    pub mesh_triangles: usize,
    pub simplified_triangles: usize,
    pub seconds: f64,
    /// raw world to the normalized frame the splats are in, row-major
    pub transform: [[f32; 4]; 4],
}

impl Trainer {
    /// Writes the splats as a 3DGS PLY (x y z, f_dc, opacity logit, log scales, unit wxyz quaternion), each kept
    /// with its mask's keep probability; the count written.
    pub fn write_ply(&mut self, path: &Path) -> Result<usize> {
        use std::io::Write;
        let keep = Self::host_f32(&self.splats.keep_prob()?)?;
        let kept = keep
            .iter()
            .map(|&p| self.rng.random::<f32>() < p)
            .collect::<Vec<_>>();
        let means = Self::host_f32(&self.splats.means)?;
        let sh0 = Self::host_f32(&self.splats.sh0)?;
        let logits = Self::host_f32(&self.splats.logits)?;
        let scales = Self::host_f32(&self.splats.log_scales)?;
        let quats = Self::host_f32(&self.splats.quats)?;
        let rows = (0..kept.len())
            .filter(|&i| kept[i])
            .filter(|&i| {
                means[3 * i..3 * i + 3]
                    .iter()
                    .chain(&scales[3 * i..3 * i + 3])
                    .chain(&sh0[3 * i..3 * i + 3])
                    .chain(std::iter::once(&logits[i]))
                    .chain(&quats[4 * i..4 * i + 4])
                    .all(|v| v.is_finite())
            })
            .collect::<Vec<_>>();
        let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
        write!(
            out,
            "ply\nformat binary_little_endian 1.0\nelement vertex {}\n",
            rows.len()
        )?;
        for name in [
            "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1",
            "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
        ] {
            writeln!(out, "property float {name}")?;
        }
        writeln!(out, "end_header")?;
        for &i in &rows {
            let q = &quats[4 * i..4 * i + 4];
            let norm = q.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
            let values = means[3 * i..3 * i + 3]
                .iter()
                .chain(&sh0[3 * i..3 * i + 3])
                .chain(std::iter::once(&logits[i]))
                .chain(&scales[3 * i..3 * i + 3])
                .copied()
                .chain(q.iter().map(|v| v / norm));
            for v in values {
                out.write_all(&v.to_le_bytes())?;
            }
        }
        out.flush()?;
        Ok(rows.len())
    }
}

impl Trainer {
    /// Every view rendered (rgb clamped, expected depth) with a fresh gate sample each, as the reference's export
    /// does, fused into a TSDF; the largest connected piece of its surface.
    pub fn mesh(&self) -> Result<Mesh> {
        let mut tsdf = Tsdf::new(MESH_TSDF);
        for (view, data) in self.views.iter().zip(&self.data.views) {
            let (gate, _) = self.sample_gate()?;
            let r = gsplat::render(&self.frame(view, Some(&gate)))?;
            let rgb = r
                .image
                .narrow(2, 0, 3)?
                .clamp(0f32, 1f32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let depth = r
                .image
                .narrow(2, 3, 1)?
                .squeeze(2)?
                .div(&r.alpha.clamp(ALPHA_FLOOR as f32, f32::MAX)?)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let rgb = rgb.as_chunks::<3>().0.to_vec();
            tsdf.integrate(&DepthFrame {
                depth: &depth,
                rgb: &rgb,
                width: data.width,
                height: data.height,
                w2c: invert4(&data.c2w),
                k: data.k,
            });
        }
        Ok(tsdf.extract().largest_component())
    }
}

/// Trains splats on a `gs_data` directory and writes `splats.ply` and `splats.json` (the report) into `out_dir`.
pub fn train_splats(
    data_dir: &Path,
    out_dir: &Path,
    options: &TrainOptions,
    device: &Device,
) -> Result<SplatReport> {
    let t = Instant::now();
    std::fs::create_dir_all(out_dir)?;
    let data = SplatData::load(data_dir)?;
    let mut trainer = Trainer::new(data, options, device)?;
    let val_psnr = trainer.train()?;
    let exported = trainer.write_ply(&out_dir.join("splats.ply"))?;
    let mesh = trainer.mesh()?;
    mesh.write_ply(&out_dir.join("mesh.ply"))?;
    let mesh_triangles = mesh.triangles.len();
    let simplified = mesh.decimate((mesh_triangles as f64 * MESH_KEEP) as usize);
    simplified.write_ply(&out_dir.join("mesh_simplified.ply"))?;
    let report = SplatReport {
        steps: options.steps,
        gaussians: trainer.splats.len(),
        exported,
        val_psnr,
        mesh_triangles,
        simplified_triangles: simplified.triangles.len(),
        seconds: t.elapsed().as_secs_f64(),
        transform: trainer.transform(),
    };
    std::fs::write(
        out_dir.join("splats.json"),
        serde_json::to_string_pretty(&report).map_err(inference_tensor::Error::wrap)?,
    )?;
    Ok(report)
}
