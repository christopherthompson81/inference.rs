use std::sync::Arc;

pub use rustfft::Fft;
use rustfft::FftPlanner;
pub use rustfft::num_complex::{Complex32, Complex64};

// Planning here keeps rustfft's generic kernels instantiated in this crate instead of in every caller.
pub fn plan_forward_f32(len: usize) -> Arc<dyn Fft<f32>> {
    FftPlanner::<f32>::new().plan_fft_forward(len)
}

pub fn plan_forward_f64(len: usize) -> Arc<dyn Fft<f64>> {
    FftPlanner::<f64>::new().plan_fft_forward(len)
}
