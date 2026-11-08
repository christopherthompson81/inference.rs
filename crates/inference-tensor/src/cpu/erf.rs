//! The [error function](https://en.wikipedia.org/wiki/Error_function).

/// `erf` calculates the error function at `x`.
pub fn erf_f64(x: f64) -> f64 {
    libm::erf(x)
}

pub fn erf_f32(x: f32) -> f32 {
    libm::erff(x)
}

// Abramowitz and Stegun 7.1.26's erfc(z) for z >= 0: erf within 1.5e-7
const AS_P: f32 = 0.327_591_1;
const AS_A: [f32; 5] = [
    0.254_829_6,
    -0.284_496_74,
    1.421_413_8,
    -1.453_152,
    1.061_405_4,
];

/// Cephes-style exp, ~1e-7 relative, for inputs a softmax or erfc gives (at most 0): branch-free, so a loop of it
/// vectorizes, where libm's is a call an element.
#[inline(always)]
pub fn fast_exp(x: f32) -> f32 {
    const LOG2E: f32 = std::f32::consts::LOG2_E;
    const C0: f32 = 0.693_359_4;
    const C1: f32 = -2.121_944_4e-4;
    let x = x.clamp(-87.0, 87.0);
    let z = (x * LOG2E).round();
    let r = x - z * C0 - z * C1;
    let r2 = r * r;
    let p = r + r2
        * (0.5
            + r * (0.166_665_46 + r * (0.041_665_795 + r * (0.008_333_45 + r * 0.001_392_034_5))));
    let e = f32::from_bits((((z as i32) + 127) << 23) as u32);
    e * (1.0 + p)
}

/// GELU, `x (1 + erf(x / sqrt 2)) / 2`, through Abramowitz and Stegun's erfc (taken directly below zero, where
/// 1 + erf would cancel), as Vulkan's map kernel computes it: branch-free, so a loop of it vectorizes (libm's erff
/// had held a CPU encoder's GELU to a call an element).
#[inline(always)]
pub fn gelu_erf_f32(x: f32) -> f32 {
    let z = x.abs() * std::f32::consts::FRAC_1_SQRT_2;
    let t = 1.0 / (1.0 + AS_P * z);
    let poly = ((((AS_A[4] * t + AS_A[3]) * t + AS_A[2]) * t + AS_A[1]) * t + AS_A[0]) * t;
    let erfc = poly * fast_exp(-z * z);
    if x >= 0.0 {
        x * (1.0 - 0.5 * erfc)
    } else {
        0.5 * x * erfc
    }
}
