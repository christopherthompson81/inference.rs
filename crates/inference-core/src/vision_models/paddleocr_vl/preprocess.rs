//! Image preprocessing mirroring transformers' (torchvision-backed) `PaddleOCRVLImageProcessor`.
//! torchvision BICUBIC+antialias is not byte-reproducible with CatmullRom, so resized pixels differ slightly.

use candle_core::{Device, Result, Tensor};
use image::{imageops::FilterType, DynamicImage, GenericImageView};

pub const PATCH: usize = 14;
pub const MERGE: usize = 2;
pub const FACTOR: usize = PATCH * MERGE; // H and W snap to multiples of this
pub const MIN_PIXELS: usize = 144 * 28 * 28; // preprocessor_config.json
pub const MAX_PIXELS: usize = 1280 * 28 * 28;
const MAX_ASPECT_RATIO: f64 = 200.0; // image_processing_paddleocr_vl.py raises past this
const RESCALE: f64 = 0.00392156862745098; // exact 1/255 from config
const MEAN: f64 = 0.5;
const STD: f64 = 0.5;

pub fn smart_resize(height: usize, width: usize) -> Result<(usize, usize)> {
    smart_resize_bounded(height, width, MIN_PIXELS, MAX_PIXELS)
}

// Python `round()` is banker's rounding, hence `round_ties_even`.
pub fn smart_resize_bounded(
    height: usize,
    width: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize)> {
    if height == 0 || width == 0 {
        candle_core::bail!("image has a zero dimension ({width}x{height})");
    }
    let f = FACTOR as f64;
    let (mut h, mut w) = (height as f64, width as f64);
    if h < f {
        w = (w * f / h).round_ties_even();
        h = f;
    }
    if w < f {
        h = (h * f / w).round_ties_even();
        w = f;
    }
    if h.max(w) / h.min(w) > MAX_ASPECT_RATIO {
        candle_core::bail!("image aspect ratio {width}x{height} exceeds {MAX_ASPECT_RATIO}");
    }
    let mut h_bar = (h / f).round_ties_even() * f;
    let mut w_bar = (w / f).round_ties_even() * f;
    let (min_px, max_px) = (min_pixels as f64, max_pixels as f64);
    if h_bar * w_bar > max_px {
        let beta = (h * w / max_px).sqrt();
        h_bar = (h / beta / f).floor() * f;
        w_bar = (w / beta / f).floor() * f;
    } else if h_bar * w_bar < min_px {
        let beta = (min_px / (h * w)).sqrt();
        h_bar = (h * beta / f).ceil() * f;
        w_bar = (w * beta / f).ceil() * f;
    }
    Ok((h_bar as usize, w_bar as usize))
}

// f32 [3, H, W] in 0..255; two affines keep torch's rescale-then-normalize rounding; permute is HF's minus unit dims.
pub fn normalize_patchify(resized: &Tensor) -> Result<Tensor> {
    let (c, h, w) = resized.dims3()?;
    let (gh, gw) = (h / PATCH, w / PATCH);
    let x = resized
        .affine(RESCALE, 0.0)?
        .affine(1.0 / STD, -MEAN / STD)?;
    x.reshape((c, gh, PATCH, gw, PATCH))?
        .permute((1, 3, 0, 2, 4))?
        .contiguous()?
        .reshape((gh * gw, c, PATCH, PATCH))
}

// CatmullRom is our `resample=3` path; see the module note on resize divergence.
pub fn preprocess_decoded(
    img: &DynamicImage,
    dev: &Device,
) -> Result<(Tensor, (usize, usize, usize))> {
    let (w0, h0) = img.dimensions();
    let (h_bar, w_bar) = smart_resize(h0 as usize, w0 as usize)?;
    let resized = img
        .resize_exact(w_bar as u32, h_bar as u32, FilterType::CatmullRom)
        .to_rgb8();
    let buf: Vec<f32> = resized.as_raw().iter().map(|&v| v as f32).collect();
    let chw = Tensor::from_vec(buf, (h_bar, w_bar, 3), dev)?
        .permute((2, 0, 1))?
        .contiguous()?;
    let px = normalize_patchify(&chw)?;
    Ok((px, (1, h_bar / PATCH, w_bar / PATCH)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_resize_rejects_degenerate_images() -> Result<()> {
        // below min_pixels, so scaled up like the reference: ceil(80*beta/28)*28 by ceil(520*beta/28)*28
        assert_eq!(smart_resize(80, 520)?, (140, 868));
        assert!(smart_resize(0, 640).is_err());
        // 1x10000 used to produce a zero-height grid and an empty vision tower input
        assert!(smart_resize(1, 10000).is_err());
        Ok(())
    }
}
