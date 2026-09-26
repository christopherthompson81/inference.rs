//! Document layout detection models built on candle.

mod cpu_conv;
mod cpu_direct;
#[cfg(feature = "cuda")]
mod cuda_kernels;
mod depthwise;
mod im2col;
mod layers;
mod mask_box;
mod msda;
pub mod pp_doclayout_v3;

pub use pp_doclayout_v3::{LayoutDetection, PPDocLayoutV3Detector};

/// Whether this crate's custom ops can run on `dev`; everything else takes the generic tensor-op fallbacks.
/// CUDA needs this crate's own `cuda` feature (candle's alone can be enabled by workspace feature unification).
fn has_kernels(dev: &candle_core::Device) -> bool {
    dev.is_cpu() || (dev.is_cuda() && cfg!(feature = "cuda"))
}

#[cfg(test)]
mod test_util {
    use candle_core::{Device, Result, Tensor, D};

    pub fn max_abs(a: &Tensor, b: &Tensor) -> Result<f32> {
        (a.to_device(&Device::Cpu)? - b.to_device(&Device::Cpu)?)?
            .abs()?
            .flatten_all()?
            .max(D::Minus1)?
            .to_scalar::<f32>()
    }

    /// `max_abs` relative to the reference's largest magnitude, for many-term fp32 sums.
    pub fn rel_err(a: &Tensor, reference: &Tensor) -> Result<f32> {
        let scale = reference
            .abs()?
            .flatten_all()?
            .max(D::Minus1)?
            .to_scalar::<f32>()?
            .max(1e-6);
        Ok(max_abs(a, reference)? / scale)
    }

    /// CPU, plus CUDA when this crate is built with it.
    pub fn devices() -> Result<Vec<Device>> {
        std::iter::once(Ok(Device::Cpu))
            .chain(cfg!(feature = "cuda").then(|| Device::new_cuda(0)))
            .collect()
    }
}
