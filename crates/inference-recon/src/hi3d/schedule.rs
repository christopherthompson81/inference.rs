//! `FlowMatchHeunScheduler`: a token-count-warped flow schedule integrated with Heun's method, guided from t = 0.7.

use inference_tensor::{Result, Tensor};

use super::transformer::{Grid, Transformer};
use crate::trellis2::dit::CrossKv;

pub const STEPS: usize = 25;
const SIGMA_MIN: f64 = 0.002;
const TIME_RESCALE: f64 = 2.0;
// the token warp `(19 n - 20000) / (3 n + 60000)`, clamped
const WARP_MIN: f64 = 1.0;
const WARP_MAX: f64 = 5.0;
// the churn above warped time 0.985 and below it
const CHURN_EDGE: f64 = 0.985;
const CHURN_HIGH: f64 = 0.04;
const CHURN_LOW: f64 = 0.002;
// guidance: none below 0.7, 10 t^2 up to 1, 4 from 1
const GUIDANCE_FROM: f64 = 0.7;
const GUIDANCE_SLOPE: f64 = 10.0;
const GUIDANCE_TOP: f64 = 4.0;
const TIME_SCALE: f64 = 1000.0;

fn rational(t: f64, factor: f64) -> f64 {
    factor * t / (1.0 + (factor - 1.0) * t)
}

/// The times each update evaluates at (`sigma_hats`) and the sigmas it steps between (`steps + 1`).
pub struct Schedule {
    pub times: Vec<f64>,
    pub sigmas: Vec<f64>,
}

impl Schedule {
    pub fn new(steps: usize, tokens: usize) -> Self {
        let n = tokens as f64;
        let warp = ((19.0 * n - 20000.0) / (3.0 * n + 60000.0)).clamp(WARP_MIN, WARP_MAX);
        // in f32 as the reference's linspace and arithmetic run
        let warped: Vec<f64> = (0..=steps)
            .map(|i| {
                let base = (1.0 - i as f32 / steps as f32) as f64;
                rational(rational(base, TIME_RESCALE), warp) as f32 as f64
            })
            .collect();
        let sigmas: Vec<f64> = warped
            .iter()
            .map(|&w| (SIGMA_MIN + (1.0 - SIGMA_MIN) * w) as f32 as f64)
            .collect();
        let times = warped[..steps]
            .iter()
            .zip(&sigmas)
            .map(|(&w, &s)| {
                let churn = if w > CHURN_EDGE {
                    CHURN_HIGH
                } else {
                    CHURN_LOW
                };
                (s * (1.0 + churn)) as f32 as f64
            })
            .collect();
        Self { times, sigmas }
    }
}

/// The guidance scale at time `t`, 1 where unguided.
pub fn guidance(t: f64) -> f64 {
    if t >= 1.0 {
        GUIDANCE_TOP
    } else if t >= GUIDANCE_FROM {
        GUIDANCE_SLOPE * t * t
    } else {
        1.0
    }
}

// one velocity at time `t`: the condition alone below 0.7, else guided against the zero condition
fn velocity(
    model: &Transformer,
    x: &Tensor,
    t: f64,
    grid: &Grid,
    kv: &[CrossKv],
    uncond: &[CrossKv],
) -> Result<Tensor> {
    let model_t = (TIME_SCALE * t) as f32;
    let pos = model.forward(x, model_t, grid, kv)?;
    if t < GUIDANCE_FROM {
        return Ok(pos);
    }
    let neg = model.forward(x, model_t, grid, uncond)?;
    &neg + ((pos - &neg)? * guidance(t))?
}

/// Heun integration from `noise` `[cells, 32]` (f32) to the normalized latents; `progress` hears each update.
pub fn sample(
    model: &Transformer,
    noise: &Tensor,
    grid: &Grid,
    kv: &[CrossKv],
    uncond: &[CrossKv],
    steps: usize,
    progress: &mut dyn FnMut(usize) -> Result<()>,
) -> Result<Tensor> {
    let schedule = Schedule::new(steps, noise.dim(0)?);
    let mut x = noise.clone();
    for (i, &t) in schedule.times.iter().enumerate() {
        let next = schedule.sigmas[i + 1];
        let step = t - next;
        let v1 = velocity(model, &x, t, grid, kv, uncond)?;
        let predicted = (&x - (&v1 * step)?)?;
        let v2 = velocity(model, &predicted, next, grid, kv, uncond)?;
        x = (&x - ((v1 + v2)? * (0.5 * step))?)?;
        progress(i + 1)?;
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    // FlowMatchHeunScheduler.set_timesteps(4, num_tokens=100) as the reference computes it
    const TIMES: [f64; 4] = [
        1.04,
        0.859_143_376_350_402_8,
        0.668_668_031_692_504_9,
        0.402_002_424_001_693_7,
    ];
    const SIGMAS: [f64; 5] = [
        1.0,
        0.857_428_550_720_214_8,
        0.667_333_364_486_694_3,
        0.401_200_026_273_727_4,
        0.002,
    ];
    const CLOSE: f64 = 1e-6;

    #[test]
    fn the_schedule_matches_the_reference() {
        let s = Schedule::new(4, 100);
        for (a, b) in s.times.iter().zip(TIMES).chain(s.sigmas.iter().zip(SIGMAS)) {
            assert!((a - b).abs() < CLOSE, "{a} vs {b}");
        }
    }

    #[test]
    fn guidance_rises_from_seven_tenths_and_caps_at_one() {
        assert_eq!(guidance(0.69), 1.0);
        assert!((guidance(0.8) - 6.4).abs() < 1e-12);
        assert_eq!(guidance(1.04), 4.0);
    }
}
