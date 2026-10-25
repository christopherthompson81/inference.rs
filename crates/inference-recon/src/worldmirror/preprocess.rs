//! Images to WorldMirror's input as the reference prepares them: one size for every frame (the first image's longest
//! edge, capped and floored to the patch grid), each image resized with PIL's 8-bit bicubic, then scaled to [0, 1].

use inference_tensor::{Device, Result, Tensor};

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
    RgbImage {
        width,
        height,
        pixels: inference_vision::pil_resize_bicubic(
            &image.pixels,
            (image.width, image.height),
            (width, height),
        ),
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
        for (p, rgb) in resized.pixels.as_chunks::<3>().0.iter().enumerate() {
            for (c, &v) in rgb.iter().enumerate() {
                chw[c * w * h + p] = f32::from(v) / 255.0;
            }
        }
        frames.push(Tensor::from_vec(chw, (3, h, w), &Device::Cpu)?);
    }
    Tensor::stack(&frames, 0)?.to_device(device)
}
