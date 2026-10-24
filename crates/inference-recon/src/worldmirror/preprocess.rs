//! Images to WorldMirror's input as the reference prepares them: one size for every frame (the first image's longest
//! edge, capped and floored to the patch grid), each image resized with PIL's 8-bit bicubic, then scaled to [0, 1].

use inference_tensor::{Device, Result, Tensor};

// PIL's fixed-point coefficient precision for 8-bit resampling: 32 bits less 8 for the pixel and 2 for headroom
const PRECISION_BITS: u32 = 32 - 8 - 2;
// PIL's bicubic: Keys cubic with a = -0.5, support 2
const BICUBIC_A: f64 = -0.5;
const BICUBIC_SUPPORT: f64 = 2.0;
// The smallest side the adaptive size allows: two patches
const MIN_SIZE: usize = 28;

/// An RGB image in 8 bits, row-major, three bytes a pixel.
pub struct RgbImage {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u8>,
}

/// The inference size for a set of images: the first image's longest edge, at most `target`, floored to a multiple of
/// `patch` and at least two patches.
pub fn adaptive_size(first: (usize, usize), target: usize, patch: usize) -> usize {
    let longest = first.0.max(first.1).min(target);
    (longest / patch * patch).max(MIN_SIZE)
}

/// The size an image of `width x height` is resized to for inference size `size`: its long side to `size`, its short
/// side to the nearest multiple of `patch` (ties to even, as Python's `round`).
pub fn resized_dims(width: usize, height: usize, size: usize, patch: usize) -> (usize, usize) {
    let scaled = |short: usize, long: usize| {
        let units = short as f64 * (size as f64 / long as f64) / patch as f64;
        round_half_even(units) as usize * patch
    };
    if width >= height {
        (size, scaled(height, width))
    } else {
        (scaled(width, height), size)
    }
}

fn round_half_even(v: f64) -> f64 {
    let r = v.round();
    if (v - v.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - v.signum()
    } else {
        r
    }
}

/// `image` resized to `width x height` with PIL's `Image.resize(..., BICUBIC)` on 8-bit RGB, bit for bit.
pub fn resize_bicubic(image: &RgbImage, width: usize, height: usize) -> RgbImage {
    let horizontal = if width == image.width {
        image.pixels.clone()
    } else {
        resample_rows(image, width)
    };
    let mid = RgbImage {
        width,
        height: image.height,
        pixels: horizontal,
    };
    if height == image.height {
        return mid;
    }
    RgbImage {
        width,
        height,
        pixels: resample_columns(&mid, height),
    }
}

/// The frames `[n, 3, H, W]` in [0, 1] for decoded images: resized to the size their first image sets for `target`.
pub fn images_to_tensor(
    images: &[RgbImage],
    target: usize,
    patch: usize,
    device: &Device,
) -> Result<Tensor> {
    let Some(first) = images.first() else {
        inference_tensor::bail!("no images to reconstruct from");
    };
    let size = adaptive_size((first.width, first.height), target, patch);
    let mut frames = Vec::with_capacity(images.len());
    let mut dims: Option<(usize, usize)> = None;
    for (i, image) in images.iter().enumerate() {
        let (w, h) = resized_dims(image.width, image.height, size, patch);
        if let Some(first) = dims
            && first != (w, h)
        {
            inference_tensor::bail!(
                "image {i} resizes to {w}x{h}, but the first image to {}x{}: every frame needs the same aspect",
                first.0,
                first.1
            );
        }
        dims = Some((w, h));
        let resized = resize_bicubic(image, w, h);
        let mut chw = vec![0f32; 3 * w * h];
        for (p, rgb) in resized.pixels.chunks_exact(3).enumerate() {
            for (c, &v) in rgb.iter().enumerate() {
                chw[c * w * h + p] = f32::from(v) / 255.0;
            }
        }
        frames.push(Tensor::from_vec(chw, (3, h, w), &Device::Cpu)?);
    }
    Tensor::stack(&frames, 0)?.to_device(device)
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

fn resample_rows(image: &RgbImage, width: usize) -> Vec<u8> {
    let coeffs = coefficients(image.width, width);
    let mut out = vec![0u8; width * image.height * 3];
    for y in 0..image.height {
        let row = &image.pixels[y * image.width * 3..][..image.width * 3];
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

fn resample_columns(image: &RgbImage, height: usize) -> Vec<u8> {
    let coeffs = coefficients(image.height, height);
    let stride = image.width * 3;
    let mut out = vec![0u8; image.width * height * 3];
    for (y, (lo, taps)) in coeffs.iter().enumerate() {
        for i in 0..stride {
            let mut acc = 1i64 << (PRECISION_BITS - 1);
            for (k, &t) in taps.iter().enumerate() {
                acc += i64::from(image.pixels[(lo + k) * stride + i]) * i64::from(t);
            }
            out[y * stride + i] = clip8(acc);
        }
    }
    out
}
