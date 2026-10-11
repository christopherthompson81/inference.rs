//! Seeded standard normal noise, drawn on the host so a seed gives the same noise on every device.

use inference_tensor::{Device, Result, Shape, Tensor};
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use rand_isaac::Isaac64Rng;

pub struct SeededNoise(Isaac64Rng);

impl SeededNoise {
    pub fn new(seed: u64) -> Self {
        Self(Isaac64Rng::seed_from_u64(seed))
    }

    /// The next standard normal F32 tensor of `shape`.
    pub fn normal<S: Into<Shape>>(&mut self, shape: S, device: &Device) -> Result<Tensor> {
        let shape = shape.into();
        let values: Vec<f32> = (0..shape.elem_count())
            .map(|_| self.0.sample(StandardNormal))
            .collect();
        Tensor::from_vec(values, shape, device)
    }
}
