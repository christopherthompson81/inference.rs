//! HiDream-O1's samplers: Wan's flow UniPC (the full model) and Dev's re-noising Euler step.

use inference_tensor::{Result, Tensor};

pub const NUM_TRAIN_TIMESTEPS: usize = 1000;
const UNIPC_ORDER: usize = 2;

/// Dev's fixed schedule, in training timesteps.
pub const DEV_TIMESTEPS: [u32; 28] = [
    999, 987, 974, 960, 945, 929, 913, 895, 877, 857, 836, 814, 790, 764, 737, 707, 675, 640, 602,
    560, 515, 464, 409, 347, 278, 199, 110, 8,
];

fn shifted(shift: f64, sigma: f64) -> f64 {
    shift * sigma / (1.0 + (shift - 1.0) * sigma)
}

/// The timesteps a sampler visits and the sigmas it steps between, one more sigma than timesteps.
#[derive(Clone, Debug)]
pub struct Schedule {
    pub timesteps: Vec<f32>,
    pub sigmas: Vec<f32>,
}

impl Schedule {
    /// `FlowUniPCMultistepScheduler.set_timesteps`, which shifts a range that its constructor already shifted.
    pub fn unipc(steps: usize, shift: f64) -> Self {
        // the constructor shifts its f32 sigmas in f32; the first is 1 - 1/1000
        let first = (1.0 - 1.0 / NUM_TRAIN_TIMESTEPS as f64) as f32;
        let shift32 = shift as f32;
        let sigma_max = f64::from(shift32 * first / (1.0 + (shift32 - 1.0) * first));
        let mut timesteps = Vec::with_capacity(steps);
        let mut sigmas = Vec::with_capacity(steps + 1);
        // numpy's linspace from sigma_max to 0 over steps + 1 points: i * step + start
        let step = -sigma_max / steps as f64;
        for i in 0..steps {
            let sigma = shifted(shift, i as f64 * step + sigma_max);
            // numpy to int64 truncates
            timesteps.push((sigma * NUM_TRAIN_TIMESTEPS as f64).trunc() as f32);
            sigmas.push(sigma as f32);
        }
        sigmas.push(0.0);
        Self { timesteps, sigmas }
    }

    /// A schedule given in training timesteps, each sigma `t / 1000`, as the pipeline sets Dev's.
    pub fn fixed(timesteps: &[u32]) -> Self {
        let mut sigmas: Vec<f32> = timesteps
            .iter()
            .map(|&t| (f64::from(t) / NUM_TRAIN_TIMESTEPS as f64) as f32)
            .collect();
        sigmas.push(0.0);
        Self {
            timesteps: timesteps.iter().map(|&t| t as f32).collect(),
            sigmas,
        }
    }
}

/// The multistep state of Wan's flow UniPC: order 2, `bh2`, predicting x0, lower order on the final steps.
#[derive(Default)]
pub struct UniPc {
    model_outputs: Vec<Tensor>,
    last_sample: Option<Tensor>,
    lower_order_nums: usize,
    this_order: usize,
}

struct Coefficients {
    alpha_t: f32,
    ratio: f32,
    // h * phi_1(h), which is also B(h) under bh2
    h_phi_1: f32,
    hh: f32,
}

fn lambda(sigma: f32) -> f32 {
    (1.0 - sigma).ln() - sigma.ln()
}

fn coefficients(sigma_t: f32, sigma_s0: f32) -> Coefficients {
    let h = lambda(sigma_t) - lambda(sigma_s0);
    let hh = -h;
    Coefficients {
        alpha_t: 1.0 - sigma_t,
        ratio: sigma_t / sigma_s0,
        h_phi_1: hh.exp_m1(),
        hh,
    }
}

impl Coefficients {
    // b of the UniPC linear system, one entry per order
    fn b(&self, order: usize) -> Vec<f32> {
        let mut h_phi_k = self.h_phi_1 / self.hh - 1.0;
        let mut factorial = 1.0f32;
        let mut b = Vec::with_capacity(order);
        for i in 1..=order {
            b.push(h_phi_k * factorial / self.h_phi_1);
            factorial *= (i + 1) as f32;
            h_phi_k = h_phi_k / self.hh - 1.0 / factorial;
        }
        b
    }

    fn base(&self, x: &Tensor, m0: &Tensor) -> Result<Tensor> {
        x.affine(f64::from(self.ratio), 0.0)?
            .sub(&m0.affine(f64::from(self.alpha_t * self.h_phi_1), 0.0)?)
    }
}

impl UniPc {
    /// One step from `sample` at `sigmas[index]` given the model's velocity `model_output`, both F32.
    pub fn step(
        &mut self,
        model_output: &Tensor,
        sample: &Tensor,
        index: usize,
        schedule: &Schedule,
    ) -> Result<Tensor> {
        let sigmas = &schedule.sigmas;
        let x0 = sample.sub(&model_output.affine(f64::from(sigmas[index]), 0.0)?)?;
        let sample = match &self.last_sample {
            Some(last) if index > 0 => self.correct(&x0, last, index, sigmas)?,
            _ => sample.clone(),
        };
        if self.model_outputs.len() == UNIPC_ORDER {
            self.model_outputs.remove(0);
        }
        self.model_outputs.push(x0);
        let order = UNIPC_ORDER.min(schedule.timesteps.len() - index);
        self.this_order = order.min(self.lower_order_nums + 1);
        self.last_sample = Some(sample.clone());
        let prev = self.predict(&sample, index, sigmas)?;
        if self.lower_order_nums < UNIPC_ORDER {
            self.lower_order_nums += 1;
        }
        Ok(prev)
    }

    fn predict(&self, x: &Tensor, index: usize, sigmas: &[f32]) -> Result<Tensor> {
        let c = coefficients(sigmas[index + 1], sigmas[index]);
        let m0 = &self.model_outputs[self.model_outputs.len() - 1];
        let x_t = c.base(x, m0)?;
        if self.this_order < 2 {
            return Ok(x_t);
        }
        let m1 = &self.model_outputs[self.model_outputs.len() - 2];
        let rk = (lambda(sigmas[index - 1]) - lambda(sigmas[index]))
            / (lambda(sigmas[index + 1]) - lambda(sigmas[index]));
        // order 2 fixes rho at 0.5
        let d1 = m1.sub(m0)?.affine(f64::from(1.0 / rk), 0.0)?;
        x_t.sub(&d1.affine(f64::from(c.alpha_t * c.h_phi_1 * 0.5), 0.0)?)
    }

    fn correct(
        &self,
        model_t: &Tensor,
        last: &Tensor,
        index: usize,
        sigmas: &[f32],
    ) -> Result<Tensor> {
        let c = coefficients(sigmas[index], sigmas[index - 1]);
        let m0 = &self.model_outputs[self.model_outputs.len() - 1];
        let x_t = c.base(last, m0)?;
        let d1_t = model_t.sub(m0)?;
        let correction = if self.this_order < 2 {
            d1_t.affine(0.5, 0.0)?
        } else {
            let m1 = &self.model_outputs[self.model_outputs.len() - 2];
            let rk = (lambda(sigmas[index - 2]) - lambda(sigmas[index - 1]))
                / (lambda(sigmas[index]) - lambda(sigmas[index - 1]));
            let b = c.b(2);
            // [[1, 1], [rk, 1]] rho = b
            let rho0 = (b[0] - b[1]) / (1.0 - rk);
            let rho1 = b[0] - rho0;
            let d1 = m1.sub(m0)?.affine(f64::from(1.0 / rk), 0.0)?;
            d1.affine(f64::from(rho0), 0.0)?
                .add(&d1_t.affine(f64::from(rho1), 0.0)?)?
        };
        x_t.sub(&correction.affine(f64::from(c.alpha_t * c.h_phi_1), 0.0)?)
    }
}

/// Dev's Euler step: `sigma_next * s_noise * n + (1 - sigma_next) * denoised`, `n` clipped to `clip_std` of its std.
pub fn flash_step(
    model_output: &Tensor,
    sample: &Tensor,
    noise: &Tensor,
    sigma: f32,
    sigma_next: f32,
    s_noise: f32,
    clip_std: f32,
) -> Result<Tensor> {
    let denoised = sample.sub(&model_output.affine(f64::from(sigma), 0.0)?)?;
    let noise = if clip_std > 0.0 {
        let n = noise.elem_count() as f64;
        let mean = noise.mean_all()?.to_scalar::<f32>()? as f64;
        let sum_sq = noise.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
        // torch's std is unbiased
        let std = ((sum_sq - n * mean * mean) / (n - 1.0)).sqrt();
        let clip = f64::from(clip_std) * std;
        noise.clamp(-clip, clip)?
    } else {
        noise.clone()
    };
    noise
        .affine(f64::from(sigma_next * s_noise), 0.0)?
        .add(&denoised.affine(f64::from(1.0 - sigma_next), 0.0)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    // FlowUniPCMultistepScheduler(shift=3.0).set_timesteps(50)
    const REFERENCE_TIMESTEPS: [u32; 50] = [
        999, 993, 986, 979, 971, 964, 956, 948, 940, 931, 922, 913, 904, 895, 885, 874, 864, 853,
        841, 830, 818, 805, 792, 778, 764, 749, 734, 718, 702, 684, 666, 647, 627, 607, 585, 562,
        538, 513, 486, 458, 428, 396, 363, 328, 290, 249, 206, 160, 111, 57,
    ];
    const REFERENCE_SIGMA_BITS: [(usize, u32); 5] = [
        (0, 0x3f7ff8b6),
        (1, 0x3f7e3dda),
        (25, 0x3f3ff7cd),
        (48, 0x3de37c3a),
        (49, 0x3d6c3b5d),
    ];

    #[test]
    fn unipc_schedule_matches_the_reference() {
        let s = Schedule::unipc(50, 3.0);
        let timesteps: Vec<u32> = s.timesteps.iter().map(|&t| t as u32).collect();
        assert_eq!(timesteps, REFERENCE_TIMESTEPS);
        for (i, bits) in REFERENCE_SIGMA_BITS {
            assert_eq!(s.sigmas[i].to_bits(), bits, "sigma {i}");
        }
        assert_eq!(s.sigmas[50], 0.0);
    }

    #[test]
    fn fixed_schedule_divides_by_the_training_steps() {
        let s = Schedule::fixed(&DEV_TIMESTEPS);
        assert_eq!(s.sigmas.len(), 29);
        assert_eq!(s.sigmas[0], 0.999);
        assert_eq!(s.sigmas[27], 0.008);
        assert_eq!(s.sigmas[28], 0.0);
    }
}
