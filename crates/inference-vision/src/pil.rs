//! PIL's `Image.resize(..., BICUBIC)` on 8-bit RGB, bit for bit: the reference pipelines this repo ports resize with
//! Pillow, and the SIMD resizer's rounding differs by a few levels.

// PIL's fixed-point coefficient precision for 8-bit resampling: 32 bits less 8 for the pixel and 2 for headroom
const PRECISION_BITS: u32 = 32 - 8 - 2;
// PIL's bicubic: Keys cubic with a = -0.5, support 2
const BICUBIC_A: f64 = -0.5;
const BICUBIC_SUPPORT: f64 = 2.0;

/// RGB pixels `width x height` (three bytes a pixel, row-major) resized to `new_width x new_height` as PIL's 8-bit
/// bicubic does: the horizontal pass first, each skipped when its size is unchanged.
pub fn pil_resize_bicubic(
    pixels: &[u8],
    (width, height): (usize, usize),
    (new_width, new_height): (usize, usize),
) -> Vec<u8> {
    let horizontal = if new_width == width {
        pixels.to_vec()
    } else {
        resample_rows(pixels, (width, height), new_width)
    };
    if new_height == height {
        return horizontal;
    }
    resample_columns(&horizontal, (new_width, height), new_height)
}

fn bicubic(x: f64) -> f64 {
    let (a, x) = (BICUBIC_A, x.abs());
    if x < 1.0 {
        ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * a
    } else {
        0.0
    }
}

// PIL's `precompute_coeffs` and `normalize_coeffs_8bpc`: each output index's first input index and fixed-point taps
fn coefficients(input: usize, output: usize) -> Vec<(usize, Vec<i32>)> {
    let scale = input as f64 / output as f64;
    let filterscale = scale.max(1.0);
    let support = BICUBIC_SUPPORT * filterscale;
    let unit = f64::from(1u32 << PRECISION_BITS);
    (0..output)
        .map(|i| {
            let center = (i as f64 + 0.5) * scale;
            let lo = ((center - support + 0.5) as i64).max(0) as usize;
            let hi = ((center + support + 0.5) as i64).min(input as i64) as usize;
            let weights = (lo..hi)
                .map(|x| bicubic((x as f64 - center + 0.5) / filterscale))
                .collect::<Vec<_>>();
            let total: f64 = weights.iter().sum();
            let taps = weights
                .iter()
                .map(|&w| {
                    let w = if total != 0.0 { w / total } else { w };
                    if w < 0.0 {
                        (-0.5 + w * unit) as i32
                    } else {
                        (0.5 + w * unit) as i32
                    }
                })
                .collect();
            (lo, taps)
        })
        .collect()
}

// PIL's 8-bit accumulation: start at half a unit, add the taps, shift down and clip
fn clip8(acc: i64) -> u8 {
    (acc >> PRECISION_BITS).clamp(0, 255) as u8
}

fn resample_rows(pixels: &[u8], (in_width, height): (usize, usize), width: usize) -> Vec<u8> {
    let coeffs = coefficients(in_width, width);
    let mut out = vec![0u8; width * height * 3];
    for y in 0..height {
        let row = &pixels[y * in_width * 3..][..in_width * 3];
        for (x, (lo, taps)) in coeffs.iter().enumerate() {
            for c in 0..3 {
                let mut acc = 1i64 << (PRECISION_BITS - 1);
                for (k, &t) in taps.iter().enumerate() {
                    acc += i64::from(row[(lo + k) * 3 + c]) * i64::from(t);
                }
                out[(y * width + x) * 3 + c] = clip8(acc);
            }
        }
    }
    out
}

fn resample_columns(pixels: &[u8], (width, in_height): (usize, usize), height: usize) -> Vec<u8> {
    let coeffs = coefficients(in_height, height);
    let stride = width * 3;
    let mut out = vec![0u8; width * height * 3];
    for (y, (lo, taps)) in coeffs.iter().enumerate() {
        for i in 0..stride {
            let mut acc = 1i64 << (PRECISION_BITS - 1);
            for (k, &t) in taps.iter().enumerate() {
                acc += i64::from(pixels[(lo + k) * stride + i]) * i64::from(t);
            }
            out[y * stride + i] = clip8(acc);
        }
    }
    out
}
