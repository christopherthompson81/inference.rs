//! SSIM over 11x11 Gaussian windows (sigma 1.5) without padding, as fused-ssim's `padding="valid"` reports it.

use inference_tensor::{Device, Result, Tensor};

const WINDOW: usize = 11;
const SIGMA: f64 = 1.5;
const C1: f64 = 0.01 * 0.01;
const C2: f64 = 0.03 * 0.03;

/// A depthwise window for `channels` channels, `[channels, 1, 11, 11]`.
pub fn window(channels: usize, device: &Device) -> Result<Tensor> {
    let half = (WINDOW / 2) as f64;
    let g = (0..WINDOW)
        .map(|i| (-((i as f64 - half).powi(2)) / (2.0 * SIGMA * SIGMA)).exp())
        .collect::<Vec<_>>();
    let sum = g.iter().sum::<f64>();
    let w = (0..WINDOW * WINDOW)
        .map(|i| (g[i / WINDOW] * g[i % WINDOW] / (sum * sum)) as f32)
        .collect::<Vec<_>>();
    Tensor::from_vec(w, (1, 1, WINDOW, WINDOW), device)?.repeat((channels, 1, 1, 1))
}

/// Mean SSIM of `a` against `b` (`[b, c, h, w]`) over the windows that fit inside the image.
pub fn ssim(a: &Tensor, b: &Tensor, window: &Tensor) -> Result<Tensor> {
    let c = a.dim(1)?;
    let blur = |x: &Tensor| x.conv2d(window, 0, 1, 1, c);
    let (mu_a, mu_b) = (blur(a)?, blur(b)?);
    let (mu_aa, mu_bb, mu_ab) = (mu_a.sqr()?, mu_b.sqr()?, (&mu_a * &mu_b)?);
    let var_a = (blur(&a.sqr()?)? - &mu_aa)?;
    let var_b = (blur(&b.sqr()?)? - &mu_bb)?;
    let cov = (blur(&(a * b)?)? - &mu_ab)?;
    let num = ((mu_ab * 2.0)? + C1)?.mul(&((cov * 2.0)? + C2)?)?;
    let den = ((mu_aa + mu_bb)? + C1)?.mul(&((var_a + var_b)? + C2)?)?;
    (num / den)?.mean_all()
}
