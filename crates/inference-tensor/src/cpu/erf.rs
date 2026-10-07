//! The [error function](https://en.wikipedia.org/wiki/Error_function).

/// `erf` calculates the error function at `x`.
pub fn erf_f64(x: f64) -> f64 {
    libm::erf(x)
}

pub fn erf_f32(x: f32) -> f32 {
    libm::erff(x)
}
