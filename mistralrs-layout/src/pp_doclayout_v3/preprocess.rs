use candle_core::{Device, Result, Tensor};
use image::RgbImage;

/// torch / cv2 bicubic coefficient (PIL and the `image` crate use -0.5).
const CUBIC_A: f64 = -0.75;
const TAPS: usize = 4;
const MAX_WEIGHT_PRECISION: u32 = 22;
const INT16_LIMIT: i32 = 1 << 15;

fn cubic1(x: f64) -> f64 {
    ((CUBIC_A + 2.) * x - (CUBIC_A + 3.)) * x * x + 1.
}

fn cubic2(x: f64) -> f64 {
    ((CUBIC_A * x - 5. * CUBIC_A) * x + 8. * CUBIC_A) * x - 4. * CUBIC_A
}

struct Taps {
    idx: Vec<[usize; TAPS]>,
    weights: Vec<[i32; TAPS]>,
    precision: u32,
}

/// torch native CPU uint8 bicubic (antialias=False): clamped 4 taps, f64 weights quantized to int16 fixed point.
fn cubic_taps(in_len: usize, out_len: usize) -> Taps {
    let scale = in_len as f64 / out_len as f64;
    let last = in_len as i64 - 1;
    let mut idx = Vec::with_capacity(out_len);
    let mut wf = Vec::with_capacity(out_len);
    for o in 0..out_len {
        let real = scale * (o as f64 + 0.5) - 0.5;
        let base = real.floor();
        let t = real - base;
        idx.push([-1i64, 0, 1, 2].map(|d| (base as i64 + d).clamp(0, last) as usize));
        wf.push([cubic2(t + 1.), cubic1(t), cubic1(1. - t), cubic2(2. - t)]);
    }
    let w_max = wf.iter().flatten().copied().fold(f64::MIN, f64::max);
    let precision = (0..MAX_WEIGHT_PRECISION)
        .find(|p| (0.5 + w_max * (1u64 << (p + 1)) as f64) as i32 >= INT16_LIMIT)
        .unwrap_or(MAX_WEIGHT_PRECISION);
    let unit = (1u64 << precision) as f64;
    let weights = wf
        .iter()
        .map(|w| {
            w.map(|v| {
                if v < 0. {
                    (-0.5 + v * unit) as i32
                } else {
                    (0.5 + v * unit) as i32
                }
            })
        })
        .collect();
    Taps {
        idx,
        weights,
        precision,
    }
}

fn accumulate(
    px: impl Fn(usize) -> u8,
    idx: &[usize; TAPS],
    w: &[i32; TAPS],
    precision: u32,
) -> u8 {
    let mut acc = 1i32 << (precision - 1);
    for k in 0..TAPS {
        acc += px(idx[k]) as i32 * w[k];
    }
    (acc >> precision).clamp(0, 255) as u8
}

/// Bicubic resize of interleaved RGB, bit-exact with torchvision's CPU uint8 path (horizontal pass first).
pub fn resize_bicubic(img: &RgbImage, out_w: usize, out_h: usize) -> Vec<u8> {
    let (in_w, in_h) = (img.width() as usize, img.height() as usize);
    let src = img.as_raw();
    let xs = cubic_taps(in_w, out_w);
    let ys = cubic_taps(in_h, out_h);

    let mut horiz = vec![0u8; in_h * out_w * 3];
    for y in 0..in_h {
        let row = &src[y * in_w * 3..(y + 1) * in_w * 3];
        for (ox, (idx, w)) in xs.idx.iter().zip(&xs.weights).enumerate() {
            for c in 0..3 {
                horiz[(y * out_w + ox) * 3 + c] =
                    accumulate(|i| row[i * 3 + c], idx, w, xs.precision);
            }
        }
    }

    let mut out = vec![0u8; out_h * out_w * 3];
    for (oy, (idx, w)) in ys.idx.iter().zip(&ys.weights).enumerate() {
        for ox in 0..out_w {
            for c in 0..3 {
                out[(oy * out_w + ox) * 3 + c] =
                    accumulate(|r| horiz[(r * out_w + ox) * 3 + c], idx, w, ys.precision);
            }
        }
    }
    out
}

pub struct Preprocessor {
    pub height: usize,
    pub width: usize,
    /// HF fuses rescale into normalize, so the effective op is `(x - mean/rf) / (std/rf)` on raw u8 values.
    mean: [f32; 3],
    std: [f32; 3],
}

impl Preprocessor {
    pub fn new(cfg: &super::config::PPDocLayoutV3PreprocessorConfig) -> Self {
        let rf = cfg.rescale_factor;
        let ch = |v: &[f64], i: usize| (v[i] / rf) as f32;
        Self {
            height: cfg.size.height,
            width: cfg.size.width,
            mean: [0, 1, 2].map(|i| ch(&cfg.image_mean, i)),
            std: [0, 1, 2].map(|i| ch(&cfg.image_std, i)),
        }
    }

    /// `(3, H, W)` f32 CHW tensor.
    pub fn preprocess(&self, img: &RgbImage, dev: &Device) -> Result<Tensor> {
        let hwc = resize_bicubic(img, self.width, self.height);
        let plane = self.height * self.width;
        let mut chw = vec![0f32; 3 * plane];
        for (i, px) in hwc.as_chunks::<3>().0.iter().enumerate() {
            for c in 0..3 {
                chw[c * plane + i] = (px[c] as f32 - self.mean[c]) / self.std[c];
            }
        }
        Tensor::from_vec(chw, (3, self.height, self.width), dev)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_size_resize_is_identity() {
        let img = RgbImage::from_fn(13, 7, |x, y| {
            image::Rgb([(x * 19) as u8, (y * 37) as u8, ((x + y) * 11) as u8])
        });
        assert_eq!(resize_bicubic(&img, 13, 7), img.into_raw());
    }

    #[test]
    fn taps_sum_to_unity_in_fixed_point() {
        let t = cubic_taps(2339, 800);
        let unit = 1i32 << t.precision;
        for w in &t.weights {
            assert!((w.iter().sum::<i32>() - unit).abs() <= 2);
        }
    }
}
