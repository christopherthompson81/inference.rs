//! Idol (speridlabs/idol): one image to 3D Gaussians. A rectified-flow transformer samples an 8192-token object latent
//! and the input's camera from DINOv3 and FLUX.2 VAE tokens; an octree decoder places anchor points from the latent and
//! a Gaussian decoder turns each anchor into 32 Gaussians.

mod conditioning;
mod decoder;
mod dit;
mod layers;
#[cfg(test)]
mod parity;
mod pipeline;
mod preprocess;

pub use decoder::Gaussians;
pub use pipeline::{
    Camera, IDOL_IMAGE_SIZE, IdolOutput, IdolPaths, IdolSettings, MAX_GAUSSIANS, MAX_STEPS,
    MIN_GAUSSIANS, generate,
};
pub use preprocess::{MARGIN_REAL, MARGIN_SYNTHETIC};

/// Gaussians as the splat exports take them ([`crate::splatting::spz`]), turned from Idol's z-up object to the y-up
/// convention viewers use: positions `(x, -z, y)`, each rotation premultiplied by +90 degrees about x.
pub struct SplatData {
    pub means: Vec<[f32; 3]>,
    pub sh0: Vec<[f32; 3]>,
    pub logits: Vec<f32>,
    pub log_scales: Vec<[f32; 3]>,
    pub quats: Vec<[f32; 4]>,
}

// how far inside (0, 1) an opacity is clamped before its logit, and a scale's floor before its log
const EXPORT_EPS: f32 = 1e-6;

impl SplatData {
    pub fn from_gaussians(g: &Gaussians) -> Self {
        let half = std::f32::consts::FRAC_1_SQRT_2;
        Self {
            means: g.xyz.iter().map(|p| [p[0], -p[2], p[1]]).collect(),
            sh0: g.dc.clone(),
            logits: g
                .opacity
                .iter()
                .map(|&o| {
                    let o = o.clamp(EXPORT_EPS, 1.0 - EXPORT_EPS);
                    (o / (1.0 - o)).ln()
                })
                .collect(),
            log_scales: g
                .scaling
                .iter()
                .map(|s| s.map(|v| v.max(EXPORT_EPS).ln()))
                .collect(),
            // (cos 45, sin 45, 0, 0) * q, the Hamilton product
            quats: g
                .rotation
                .iter()
                .map(|&[w, x, y, z]| {
                    [
                        half * (w - x),
                        half * (x + w),
                        half * (y - z),
                        half * (z + y),
                    ]
                })
                .collect(),
        }
    }

    pub fn rows(&self) -> crate::splatting::spz::SplatRows<'_> {
        crate::splatting::spz::SplatRows {
            means: &self.means,
            sh0: &self.sh0,
            logits: &self.logits,
            log_scales: &self.log_scales,
            quats_wxyz: &self.quats,
        }
    }
}
