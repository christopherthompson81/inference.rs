use image::RgbImage;
use inference_tensor::{Device, Result, Tensor};
use rayon::prelude::*;

/// The letterbox's long side when a checkpoint records no training size.
pub const INPUT_SIZE: usize = 1024;
/// Padded sides are rounded up to this, the head's coarsest stride.
const STRIDE: usize = 32;
const PAD_VALUE: u8 = 114;
const COEF_BITS: u32 = 11;
const COEF_SCALE: f32 = (1 << COEF_BITS) as f32;
/// Rounding offset of the letterbox's two pad sides, so an odd pad puts the extra pixel at the bottom and right.
const PAD_NUDGE: f64 = 0.1;

/// Where the resized page sits in the letterboxed input.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Letterbox {
    /// `(width, height)` of the network input.
    pub input: (usize, usize),
    /// `(width, height)` of the source image.
    pub source: (u32, u32),
    /// `(width, height)` the source is resized to before padding.
    pub resized: (usize, usize),
}

impl Letterbox {
    pub fn new(source: (u32, u32), size: usize) -> Self {
        let (w0, h0) = (source.0 as f64, source.1 as f64);
        let r = (size as f64 / h0).min(size as f64 / w0);
        let unpad = (
            (w0 * r).round_ties_even() as usize,
            (h0 * r).round_ties_even() as usize,
        );
        let fit = |side: usize| side + (size - side) % STRIDE;
        Self {
            input: (fit(unpad.0), fit(unpad.1)),
            source,
            resized: unpad,
        }
    }

    /// `(left, top)` padding.
    fn offsets(&self) -> (usize, usize) {
        let resized = self.resized;
        let half = |total: usize, inner: usize| {
            ((total - inner) as f64 / 2. - PAD_NUDGE).round_ties_even() as usize
        };
        (half(self.input.0, resized.0), half(self.input.1, resized.1))
    }

    /// Maps an input-space `[x1, y1, x2, y2]` box back to source pixels, clipped to the image.
    pub fn to_source(&self, b: [f32; 4]) -> [f32; 4] {
        let (w0, h0) = (self.source.0 as f64, self.source.1 as f64);
        let (w, h) = (self.input.0 as f64, self.input.1 as f64);
        let gain = (h / h0).min(w / w0);
        let pad_x = ((w - w0 * gain) / 2. - PAD_NUDGE).round_ties_even() as f32;
        let pad_y = ((h - h0 * gain) / 2. - PAD_NUDGE).round_ties_even() as f32;
        let gain = gain as f32;
        let (w0, h0) = (w0 as f32, h0 as f32);
        [
            ((b[0] - pad_x) / gain).clamp(0., w0),
            ((b[1] - pad_y) / gain).clamp(0., h0),
            ((b[2] - pad_x) / gain).clamp(0., w0),
            ((b[3] - pad_y) / gain).clamp(0., h0),
        ]
    }
}

struct Taps {
    src: Vec<usize>,
    next: Vec<usize>,
    w0: Vec<i32>,
    w1: Vec<i32>,
}

/// cv2 INTER_LINEAR source taps and 11-bit weights along one axis.
fn linear_taps(in_len: usize, out_len: usize) -> Taps {
    let scale = in_len as f64 / out_len as f64;
    let mut taps = Taps {
        src: Vec::with_capacity(out_len),
        next: Vec::with_capacity(out_len),
        w0: Vec::with_capacity(out_len),
        w1: Vec::with_capacity(out_len),
    };
    for o in 0..out_len {
        let f = ((o as f64 + 0.5) * scale - 0.5) as f32;
        let mut s = f.floor() as i64;
        let mut t = f - s as f32;
        if s < 0 {
            (s, t) = (0, 0.);
        }
        if s >= in_len as i64 - 1 {
            (s, t) = (in_len as i64 - 1, 0.);
        }
        let s = s as usize;
        let w0 = ((1. - t) * COEF_SCALE).round_ties_even() as i32;
        taps.src.push(s);
        taps.next.push((s + 1).min(in_len - 1));
        taps.w0.push(w0);
        taps.w1.push((1 << COEF_BITS) - w0);
    }
    taps
}

/// Bilinear resize of interleaved RGB as cv2's x86 SIMD path computes it (exact when shrinking; enlarging can differ
/// by 1 on a few pixels where cv2 finishes a row in scalar code).
pub fn resize_linear(img: &RgbImage, out_w: usize, out_h: usize) -> Vec<u8> {
    let (in_w, in_h) = (img.width() as usize, img.height() as usize);
    let src = img.as_raw();
    let xs = linear_taps(in_w, out_w);
    let ys = linear_taps(in_h, out_h);
    let row = |y: usize| -> Vec<i32> {
        let line = &src[y * in_w * 3..(y + 1) * in_w * 3];
        (0..out_w * 3)
            .map(|i| {
                let (x, c) = (i / 3, i % 3);
                line[xs.src[x] * 3 + c] as i32 * xs.w0[x]
                    + line[xs.next[x] * 3 + c] as i32 * xs.w1[x]
            })
            .collect()
    };
    let mut out = vec![0u8; out_w * out_h * 3];
    out.par_chunks_mut(out_w * 3)
        .enumerate()
        .for_each(|(y, dst)| {
            let (r0, r1) = (row(ys.src[y]), row(ys.next[y]));
            let (b0, b1) = (ys.w0[y], ys.w1[y]);
            for ((d, a), b) in dst.iter_mut().zip(&r0).zip(&r1) {
                let v = (((a >> 4) * b0) >> 16) + (((b >> 4) * b1) >> 16);
                *d = ((v + 2) >> 2).clamp(0, 255) as u8;
            }
        });
    out
}

/// The `(3, h, w)` input in `[0, 1]`, RGB, letterboxed to a `size` long side.
pub fn preprocess(img: &RgbImage, size: usize, device: &Device) -> Result<(Tensor, Letterbox)> {
    let lb = Letterbox::new(img.dimensions(), size);
    let resized = lb.resized;
    let pixels = if resized == (img.width() as usize, img.height() as usize) {
        img.as_raw().clone()
    } else {
        resize_linear(img, resized.0, resized.1)
    };
    let (left, top) = lb.offsets();
    let (w, h) = lb.input;
    let mut planes = vec![PAD_VALUE as f32 / 255.; 3 * w * h];
    for y in 0..resized.1 {
        for x in 0..resized.0 {
            let s = (y * resized.0 + x) * 3;
            let d = (y + top) * w + x + left;
            for c in 0..3 {
                planes[c * w * h + d] = pixels[s + c] as f32 / 255.;
            }
        }
    }
    Ok((Tensor::from_vec(planes, (3, h, w), device)?, lb))
}
