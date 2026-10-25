//! WorldStereo-2's distilled (DMD) sampler: four flow steps from pure noise, no classifier-free guidance.

use inference_tensor::{Result, Tensor};

/// The checkpoint config's DMD schedule: timesteps spread from `start` to `end` (of `end`), shifted by `shift`
#[derive(Debug, Clone, Copy)]
pub struct DmdSchedule {
    pub start: f64,
    pub end: f64,
    pub shift: f64,
    pub steps: usize,
}

impl Default for DmdSchedule {
    fn default() -> Self {
        Self {
            start: 250.0,
            end: 1000.0,
            shift: 5.0,
            steps: 4,
        }
    }
}

impl DmdSchedule {
    /// Timesteps in 0..end, highest first: linspace(start, end, steps) reversed, as fractions shifted
    /// `s t / (1 + (s - 1) t)`, times `end`.
    pub fn timesteps(&self) -> Vec<f64> {
        (0..self.steps)
            .rev()
            .map(|i| {
                let t = if self.steps == 1 {
                    self.start
                } else {
                    self.start + (self.end - self.start) * i as f64 / (self.steps - 1) as f64
                } / self.end;
                self.shift * t / (1.0 + (self.shift - 1.0) * t) * self.end
            })
            .collect()
    }

    /// One Euler step of the flow `v` at step `i`: `x + (sigma_next - sigma) v`, sigma = t / end, the last to 0.
    pub fn step(&self, timesteps: &[f64], i: usize, sample: &Tensor, v: &Tensor) -> Result<Tensor> {
        let sigma = timesteps[i] / self.end;
        let next = timesteps.get(i + 1).map_or(0.0, |t| t / self.end);
        sample + (v * (next - sigma))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // the reference prints these for its default schedule: 1000, 937.5, 833.33, 625
    #[test]
    fn default_schedule_matches_the_reference() {
        let t = DmdSchedule::default().timesteps();
        let want = [1000.0, 937.5, 833.333_333, 625.0];
        for (a, b) in t.iter().zip(want) {
            assert!((a - b).abs() < 1e-3, "{t:?}");
        }
    }
}
