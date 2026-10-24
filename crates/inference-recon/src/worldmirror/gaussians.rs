//! Per-pixel Gaussian splats from the Gaussian head's fused features: rotation, scale, opacity, colour (SH degree 0)
//! and a merge weight from a small conv stack; centres from the Gaussian head's depth through the predicted cameras.

use inference_tensor::nn::VarBuilder;
use inference_tensor::{D, DType, Result, Tensor};

use super::camera::Camera;

// The splat parameter channels: quaternion (w, x, y, z), log scale, opacity logit, colour residual, merge-weight logit
const QUAT: usize = 4;
const SCALE: usize = 3;
const COLOR: usize = 3;
const PARAMS: usize = QUAT + SCALE + 1 + COLOR + 1;
const HIDDEN: usize = 256;
// The largest splat scale, in the reconstruction's units
const MAX_SCALE: f64 = 0.3;
const QUAT_EPS: f64 = 1e-8;
// The degree-0 spherical harmonic's constant, mapping a [0, 1] colour to its DC coefficient
const SH_C0: f64 = 0.282_094_791_773_878_14;

/// Every pixel's splat, `N = frames * H * W` of them in frame-major, row-major order.
pub struct Splats {
    /// `[N, 3]` world positions
    pub means: Tensor,
    /// `[N, 4]` unit quaternions, scalar first (w, x, y, z)
    pub quats: Tensor,
    /// `[N, 3]` linear scales
    pub scales: Tensor,
    /// `[N]` in [0, 1]
    pub opacities: Tensor,
    /// `[N, 3]` SH DC colour coefficients
    pub sh_dc: Tensor,
    /// `[N]` in [0, 1], the weight a voxel merge averages with
    pub weights: Tensor,
}

pub(crate) struct GaussianParams {
    conv1: Tensor,
    conv2: Tensor,
    bias2: Tensor,
}

impl GaussianParams {
    pub(crate) fn load(features: usize, vb: VarBuilder) -> Result<Self> {
        let head = vb.pp("gs_head");
        Ok(Self {
            conv1: head.pp(0).get((HIDDEN, features, 3, 3), "weight")?,
            conv2: head.pp(2).get((PARAMS, HIDDEN, 1, 1), "weight")?,
            bias2: head.pp(2).get(PARAMS, "bias")?,
        })
    }

    /// The splats of fused Gaussian features `[frames, C, H, W]`, the raw [0, 1] images `[frames, 3, H, W]`, the
    /// Gaussian head's depth `[frames, H, W, 1]` and each frame's camera.
    pub(crate) fn forward(
        &self,
        fused: &Tensor,
        images: &Tensor,
        depth: &Tensor,
        cameras: &[Camera],
    ) -> Result<Splats> {
        let (frames, _, h, w) = images.dims4()?;
        let n = frames * h * w;
        let params = fused
            .conv2d(&self.conv1, 1, 1, 1, 1)?
            .relu()?
            .conv2d(&self.conv2, 0, 1, 1, 1)?
            .broadcast_add(&self.bias2.reshape((1, (), 1, 1))?)?
            .permute((0, 2, 3, 1))?
            .reshape((n, PARAMS))?;
        let mut at = 0;
        let mut take = |c: usize| -> Result<Tensor> {
            let t = params.narrow(1, at, c)?;
            at += c;
            Ok(t)
        };
        let quats = take(QUAT)?;
        let scales = take(SCALE)?;
        let opacity = take(1)?;
        let residual = take(COLOR)?;
        let weight = take(1)?;
        let quats =
            quats.broadcast_div(&(quats.sqr()?.sum_keepdim(D::Minus1)?.sqrt()? + QUAT_EPS)?)?;
        let rgb = images.permute((0, 2, 3, 1))?.reshape((n, COLOR))?;
        let sh_dc = ((rgb - 0.5)? / SH_C0)?.add(&residual)?;
        Ok(Splats {
            means: unproject(depth, cameras)?,
            quats,
            scales: scales.exp()?.clamp(f64::MIN, MAX_SCALE)?,
            opacities: inference_tensor::nn::ops::sigmoid(&opacity.squeeze(1)?)?,
            sh_dc,
            weights: inference_tensor::nn::ops::sigmoid(&weight.squeeze(1)?)?,
        })
    }
}

/// World points `[frames * H * W, 3]` of z-depths `[frames, H, W, 1]`: pixel (u, v) (column, row, no half-pixel
/// offset) to `((u - cx) z / fx, (v - cy) z / fy, z)`, then through camera-to-world.
pub fn unproject(depth: &Tensor, cameras: &[Camera]) -> Result<Tensor> {
    let (frames, h, w, _) = depth.dims4()?;
    let device = depth.device();
    let us = Tensor::arange(0u32, w as u32, device)?.to_dtype(DType::F32)?;
    let vs = Tensor::arange(0u32, h as u32, device)?.to_dtype(DType::F32)?;
    let mut points = Vec::with_capacity(frames);
    for (f, camera) in cameras.iter().enumerate().take(frames) {
        let k = camera.intrinsics;
        let z = depth.get(f)?.squeeze(D::Minus1)?;
        let x = ((us.reshape((1, w))? - f64::from(k[0][2]))? / f64::from(k[0][0]))?
            .broadcast_mul(&z)?;
        let y = ((vs.reshape((h, 1))? - f64::from(k[1][2]))? / f64::from(k[1][1]))?
            .broadcast_mul(&z)?;
        let cam = Tensor::stack(&[&x, &y, &z], D::Minus1)?.reshape((h * w, 3))?;
        let m = camera.camera_to_world;
        let rotation = Tensor::new(
            &[
                [m[0][0], m[0][1], m[0][2]],
                [m[1][0], m[1][1], m[1][2]],
                [m[2][0], m[2][1], m[2][2]],
            ],
            device,
        )?;
        let translation = Tensor::new(&[m[0][3], m[1][3], m[2][3]], device)?.reshape((1, 3))?;
        points.push(cam.matmul(&rotation.t()?)?.broadcast_add(&translation)?);
    }
    Tensor::cat(&points, 0)
}
