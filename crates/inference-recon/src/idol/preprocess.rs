//! The image Idol's encoders see, as its `preprocess_image` makes it: short side to 1024 (PIL Lanczos), the alpha from
//! BiRefNet when the image has none, eroded by a pixel, a square around the object `margin` times its bounding box,
//! resized to 1024^2 and composited on black.

use inference_tensor::Result;

use crate::birefnet::BiRefNet;
use crate::jovox::Rgba;

/// The encoders' square input side.
pub(crate) const IMAGE_SIZE: usize = 1024;
/// How much room the crop leaves around the object (side = bounding box x margin): renders and generated images of
/// one object fit tight, photos looser.
pub const MARGIN_SYNTHETIC: f64 = 1.2;
pub const MARGIN_REAL: f64 = 1.7;
const RGBA: usize = 4;

/// Whether some pixel is not fully opaque (PIL's RGBA image with a real alpha).
fn has_real_alpha(image: &Rgba) -> bool {
    image
        .pixels
        .as_chunks::<RGBA>()
        .0
        .iter()
        .any(|px| px[3] < 255)
}

// PIL's `MinFilter(3)`: each pixel's 3x3 minimum, edges replicated
fn erode(alpha: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut out = vec![0u8; alpha.len()];
    for y in 0..height {
        for x in 0..width {
            let mut m = u8::MAX;
            for dy in [-1i64, 0, 1] {
                for dx in [-1i64, 0, 1] {
                    let sy = (y as i64 + dy).clamp(0, height as i64 - 1) as usize;
                    let sx = (x as i64 + dx).clamp(0, width as i64 - 1) as usize;
                    m = m.min(alpha[sy * width + sx]);
                }
            }
            out[y * width + x] = m;
        }
    }
    out
}

/// RGB bytes `IMAGE_SIZE`^2 as the encoders see the object in `image`; `birefnet` loads only when the image needs it.
pub(crate) fn prepare(
    image: &Rgba,
    margin: f64,
    birefnet: &mut dyn FnMut() -> Result<BiRefNet>,
) -> Result<Vec<u8>> {
    let (w, h) = (image.width, image.height);
    let scale = IMAGE_SIZE as f64 / w.min(h) as f64;
    let size = (
        ((w as f64 * scale).round_ties_even() as usize).max(1),
        ((h as f64 * scale).round_ties_even() as usize).max(1),
    );
    let mut pixels = inference_vision::pil_resize_rgba_lanczos(&image.pixels, (w, h), size);
    let (w, h) = size;
    if !has_real_alpha(&Rgba {
        width: w,
        height: h,
        pixels: pixels.clone(),
    }) {
        let rgb: Vec<u8> = pixels
            .as_chunks::<RGBA>()
            .0
            .iter()
            .flat_map(|px| [px[0], px[1], px[2]])
            .collect();
        let mask = birefnet()?.mask_bilinear(&rgb, (w, h))?;
        for (px, a) in pixels.as_chunks_mut::<RGBA>().0.iter_mut().zip(mask) {
            px[3] = a;
        }
    }
    let alpha: Vec<u8> = pixels
        .as_chunks::<RGBA>()
        .0
        .iter()
        .map(|px| px[3])
        .collect();
    let alpha = erode(&alpha, w, h);
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
    for (i, (px, &a)) in pixels
        .as_chunks_mut::<RGBA>()
        .0
        .iter_mut()
        .zip(&alpha)
        .enumerate()
    {
        px[3] = a;
        if a > 0 {
            let (x, y) = (i % w, i / w);
            (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
        }
    }
    if x0 == usize::MAX {
        inference_tensor::bail!("no object found in the image (the alpha channel is empty)");
    }
    let (cx, cy) = ((x0 + x1) as f64 / 2.0, (y0 + y1) as f64 / 2.0);
    let half = (x1 - x0).max(y1 - y0) as f64 / 2.0 * margin;
    // Python's int() truncates toward zero
    let (left, top) = ((cx - half) as i64, (cy - half) as i64);
    let (right, bottom) = ((cx + half) as i64, (cy + half) as i64);
    let (cw, ch) = ((right - left) as usize, (bottom - top) as usize);
    let mut crop = vec![0u8; cw * ch * RGBA];
    for y in 0..ch {
        let sy = top + y as i64;
        if sy < 0 || sy >= h as i64 {
            continue;
        }
        for x in 0..cw {
            let sx = left + x as i64;
            if sx < 0 || sx >= w as i64 {
                continue;
            }
            let src = (sy as usize * w + sx as usize) * RGBA;
            crop[(y * cw + x) * RGBA..][..RGBA].copy_from_slice(&pixels[src..src + RGBA]);
        }
    }
    let resized =
        inference_vision::pil_resize_rgba_lanczos(&crop, (cw, ch), (IMAGE_SIZE, IMAGE_SIZE));
    // PIL's paste through the alpha onto black: its BLEND rounding of src * a / 255
    Ok(resized
        .as_chunks::<RGBA>()
        .0
        .iter()
        .flat_map(|px| {
            let a = u32::from(px[3]);
            [0, 1, 2].map(|c| {
                let t = u32::from(px[c]) * a + 128;
                ((t + (t >> 8)) >> 8) as u8
            })
        })
        .collect())
}
