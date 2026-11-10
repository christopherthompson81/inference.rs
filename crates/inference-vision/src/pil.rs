//! PIL's `Image.resize` on 8-bit RGB (bicubic) and RGBA (Lanczos, through premultiplied alpha), bit for bit: the
//! reference pipelines this repo ports resize with Pillow, and the SIMD resizer's rounding differs by a few levels.

// PIL's fixed-point coefficient precision for 8-bit resampling: 32 bits less 8 for the pixel and 2 for headroom
const PRECISION_BITS: u32 = 32 - 8 - 2;
// PIL's bicubic: Keys cubic with a = -0.5, support 2
const BICUBIC_A: f64 = -0.5;
const BICUBIC_SUPPORT: f64 = 2.0;
const LANCZOS_SUPPORT: f64 = 3.0;
const RGB: usize = 3;
const RGBA: usize = 4;

const BILINEAR_SUPPORT: f64 = 1.0;

/// PIL's resampling filters.
#[derive(Clone, Copy, Debug)]
pub enum PilFilter {
    Bilinear,
    Bicubic,
    Lanczos,
}

impl PilFilter {
    fn support(self) -> f64 {
        match self {
            Self::Bilinear => BILINEAR_SUPPORT,
            Self::Bicubic => BICUBIC_SUPPORT,
            Self::Lanczos => LANCZOS_SUPPORT,
        }
    }

    fn weight(self, x: f64) -> f64 {
        match self {
            Self::Bilinear => (1.0 - x.abs()).max(0.0),
            Self::Bicubic => bicubic(x),
            Self::Lanczos => lanczos(x),
        }
    }
}

/// RGB pixels `width x height` (three bytes a pixel, row-major) resized to `new_width x new_height` as PIL's 8-bit
/// bicubic does: the horizontal pass first, each skipped when its size is unchanged.
pub fn pil_resize_bicubic(
    pixels: &[u8],
    size: (usize, usize),
    new_size: (usize, usize),
) -> Vec<u8> {
    pil_resize(pixels, RGB, size, new_size, PilFilter::Bicubic)
}

/// RGBA pixels (four bytes a pixel) resized as PIL's `Image.resize(..., LANCZOS)` does an RGBA image: premultiplied
/// by alpha, resampled, then divided back.
pub fn pil_resize_rgba_lanczos(
    pixels: &[u8],
    size: (usize, usize),
    new_size: (usize, usize),
) -> Vec<u8> {
    let mut premultiplied = pixels.to_vec();
    for px in premultiplied.chunks_exact_mut(RGBA) {
        let alpha = u32::from(px[3]);
        for c in &mut px[..3] {
            let t = u32::from(*c) * alpha + 128;
            *c = (((t >> 8) + t) >> 8) as u8;
        }
    }
    let mut out = pil_resize(&premultiplied, RGBA, size, new_size, PilFilter::Lanczos);
    for px in out.chunks_exact_mut(RGBA) {
        let alpha = u32::from(px[3]);
        if alpha != 0 && alpha != 255 {
            for c in &mut px[..3] {
                *c = (255 * u32::from(*c) / alpha).min(255) as u8;
            }
        }
    }
    out
}

/// Pixels of `channels` bytes each (no alpha handling) resized as PIL's 8-bit `Image.resize` with `filter` does.
pub fn pil_resize(
    pixels: &[u8],
    channels: usize,
    (width, height): (usize, usize),
    (new_width, new_height): (usize, usize),
    filter: PilFilter,
) -> Vec<u8> {
    let horizontal = if new_width == width {
        pixels.to_vec()
    } else {
        resample_rows(pixels, channels, (width, height), new_width, filter)
    };
    if new_height == height {
        return horizontal;
    }
    resample_columns(
        &horizontal,
        channels,
        (new_width, height),
        new_height,
        filter,
    )
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

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let x = x * std::f64::consts::PI;
        x.sin() / x
    }
}

fn lanczos(x: f64) -> f64 {
    if (-LANCZOS_SUPPORT..LANCZOS_SUPPORT).contains(&x) {
        sinc(x) * sinc(x / LANCZOS_SUPPORT)
    } else {
        0.0
    }
}

// PIL's `precompute_coeffs` and `normalize_coeffs_8bpc`: each output index's first input index and fixed-point taps
fn coefficients(input: usize, output: usize, filter: PilFilter) -> Vec<(usize, Vec<i32>)> {
    let scale = input as f64 / output as f64;
    let filterscale = scale.max(1.0);
    let support = filter.support() * filterscale;
    let unit = f64::from(1u32 << PRECISION_BITS);
    (0..output)
        .map(|i| {
            let center = (i as f64 + 0.5) * scale;
            let lo = ((center - support + 0.5) as i64).max(0) as usize;
            let hi = ((center + support + 0.5) as i64).min(input as i64) as usize;
            let weights = (lo..hi)
                .map(|x| filter.weight((x as f64 - center + 0.5) / filterscale))
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

fn resample_rows(
    pixels: &[u8],
    ch: usize,
    (in_width, height): (usize, usize),
    width: usize,
    filter: PilFilter,
) -> Vec<u8> {
    let coeffs = coefficients(in_width, width, filter);
    let mut out = vec![0u8; width * height * ch];
    for y in 0..height {
        let row = &pixels[y * in_width * ch..][..in_width * ch];
        for (x, (lo, taps)) in coeffs.iter().enumerate() {
            for c in 0..ch {
                let mut acc = 1i64 << (PRECISION_BITS - 1);
                for (k, &t) in taps.iter().enumerate() {
                    acc += i64::from(row[(lo + k) * ch + c]) * i64::from(t);
                }
                out[(y * width + x) * ch + c] = clip8(acc);
            }
        }
    }
    out
}

fn resample_columns(
    pixels: &[u8],
    ch: usize,
    (width, in_height): (usize, usize),
    height: usize,
    filter: PilFilter,
) -> Vec<u8> {
    let coeffs = coefficients(in_height, height, filter);
    let stride = width * ch;
    let mut out = vec![0u8; width * height * ch];
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
