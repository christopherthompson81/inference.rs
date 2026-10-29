//! HiDream-O1's reference images: each enters the ViT at a small size and the denoiser as full-size pixel patches.

use image::{DynamicImage, RgbImage};
use inference_tensor::{Device, Result, Tensor};

use super::model::{CHANNELS, PATCH_DIM, PATCH_SIZE};

/// The ViT's input size for up to 4 references; more shrink it, as the reference pipeline does.
const CONDITION_IMAGE_SIZE: usize = 384;
// the reference pipeline's reference counts at which each image's patches and ViT input shrink
const FEW_REFERENCES: usize = 4;
const SEVERAL_REFERENCES: usize = 8;
// its shrink factors, in parts of the full edge out of EDGE_PARTS
const EDGE_PARTS: usize = 64;
const TWO_REFERENCES_EDGE: usize = 48;
const FEW_REFERENCES_EDGE: usize = 32;
const SEVERAL_REFERENCES_EDGE: usize = 24;
const MANY_REFERENCES_EDGE: usize = 16;
const SEVERAL_CONDITION_EDGE: usize = 48;
const MANY_CONDITION_EDGE: usize = 32;
const VIT_PATCH_SIZE: usize = 16;
const VIT_MERGE_SIZE: usize = 2;
const VIT_TEMPORAL_PATCH_SIZE: usize = 2;
const VIT_ROW_LEN: usize = CHANNELS * VIT_TEMPORAL_PATCH_SIZE * VIT_PATCH_SIZE * VIT_PATCH_SIZE;
// the stock Qwen3-VL processor config HiDream ships: shortest_edge and longest_edge, in pixels
const VIT_MIN_PIXELS: usize = 65_536;
const VIT_MAX_PIXELS: usize = 16_777_216;
// Pillow's fixed-point resampling: 32 bits less 8 for the value and 2 for headroom
const PRECISION_BITS: u32 = 32 - 8 - 2;

/// The reference images as both inputs, prepared once per request.
pub struct References {
    /// Qwen3-VL processor rows (patches, 3 * 2 * 16 * 16), in its merge-window order.
    pub vit_pixels: Tensor,
    /// Each image's ViT grid as (t, h, w) in 16 px patches.
    pub vit_grids: Vec<[u32; 3]>,
    /// The images' 32 px patches in [-1, 1], (1, patches, 3072), one image after another.
    pub patches: Tensor,
    /// Each image's patch grid as (rows, cols).
    pub patch_grids: Vec<(usize, usize)>,
}

impl References {
    /// The output size for references of `dims` (width, height) and the snapped requested `size`, from sizes alone;
    /// with one reference and no size, the reference's own aspect at the model's `edge`.
    pub fn output_size(
        dims: &[(usize, usize)],
        size: Option<(usize, usize)>,
        edge: usize,
    ) -> Result<(usize, usize)> {
        Ok(size.unwrap_or(checked_plans(dims, size, edge)?[0].out))
    }

    /// `images` for an output of `size`, the snapped requested one; `None` takes the only image's own size.
    pub fn prepare(
        images: &[DynamicImage],
        size: Option<(usize, usize)>,
        edge: usize,
        device: &Device,
    ) -> Result<(Self, (usize, usize))> {
        let rgb: Vec<RgbImage> = images.iter().map(DynamicImage::to_rgb8).collect();
        let dims: Vec<(usize, usize)> = rgb
            .iter()
            .map(|img| (img.width() as usize, img.height() as usize))
            .collect();
        let plans = checked_plans(&dims, size, edge)?;
        let size = size.unwrap_or(plans[0].out);
        let condition_size = condition_size(images.len());
        let mut vit_pixels = Vec::new();
        let mut vit_grids = Vec::with_capacity(rgb.len());
        let mut patches = Vec::new();
        let mut patch_grids = Vec::with_capacity(rgb.len());
        for (img, plan) in rgb.iter().zip(&plans) {
            let img = plan.apply(img);
            let (w, h) = (img.width() as usize, img.height() as usize);
            let (cw, ch) = calculate_dimensions(condition_size, w as f64 / h as f64);
            let condition = resize(&img, cw, ch, Filter::Lanczos);
            vit_grids.push(vit_rows(&condition, &mut vit_pixels));
            patch_grids.push((h / PATCH_SIZE, w / PATCH_SIZE));
            pixel_patches(&img, &mut patches);
        }
        let vit_rows = vit_pixels.len() / VIT_ROW_LEN;
        let patch_count = patches.len() / PATCH_DIM;
        Ok((
            Self {
                vit_pixels: Tensor::from_vec(vit_pixels, (vit_rows, VIT_ROW_LEN), device)?,
                vit_grids,
                patches: Tensor::from_vec(patches, (1, patch_count, PATCH_DIM), device)?,
                patch_grids,
            },
            size,
        ))
    }

    /// How many ViT tokens each image leaves after the merger.
    pub fn vit_tokens(&self) -> Vec<usize> {
        self.vit_grids
            .iter()
            .map(|[t, h, w]| (t * h * w) as usize / (VIT_MERGE_SIZE * VIT_MERGE_SIZE))
            .collect()
    }

    /// Each image's merged ViT grid (h, w), as the prompt's M-RoPE places its pads.
    pub fn merged_grids(&self) -> Vec<(usize, usize)> {
        self.vit_grids
            .iter()
            .map(|[_, h, w]| (*h as usize / VIT_MERGE_SIZE, *w as usize / VIT_MERGE_SIZE))
            .collect()
    }

    pub fn vit_grid_tensor(&self, device: &Device) -> Result<Tensor> {
        let flat: Vec<u32> = self.vit_grids.iter().flatten().copied().collect();
        Tensor::from_vec(flat, (self.vit_grids.len(), 3), device)
    }
}

// each image's fit at the edge its count leaves, refusing (from sizes alone) one too narrow for the patches or the ViT
fn checked_plans(
    dims: &[(usize, usize)],
    size: Option<(usize, usize)>,
    matched: usize,
) -> Result<Vec<Fit>> {
    let edge = match size {
        Some((w, h)) => reference_size(dims.len(), w.max(h)),
        // one reference and no requested size: the output takes the reference's own aspect at the model's edge
        None if dims.len() == 1 => matched,
        None => inference_tensor::bail!("several reference images need an output size"),
    };
    let plans: Vec<Fit> = dims
        .iter()
        .map(|&(w, h)| Fit::plan(w, h, edge))
        .collect::<Result<_>>()?;
    for plan in &plans {
        let (w, h) = plan.out;
        let (cw, ch) = calculate_dimensions(condition_size(dims.len()), w as f64 / h as f64);
        if cw == 0 || ch == 0 {
            inference_tensor::bail!(
                "a {w}x{h} reference image is too narrow for the vision encoder"
            );
        }
    }
    Ok(plans)
}

/// The edge each reference's pixel patches fit within, smaller as more share the sequence.
fn reference_size(count: usize, edge: usize) -> usize {
    let parts = match count {
        1 => return edge,
        2 => TWO_REFERENCES_EDGE,
        3..=FEW_REFERENCES => FEW_REFERENCES_EDGE,
        _ if count <= SEVERAL_REFERENCES => SEVERAL_REFERENCES_EDGE,
        _ => MANY_REFERENCES_EDGE,
    };
    edge * parts / EDGE_PARTS
}

fn condition_size(count: usize) -> usize {
    match count {
        0..=FEW_REFERENCES => CONDITION_IMAGE_SIZE,
        _ if count <= SEVERAL_REFERENCES => {
            CONDITION_IMAGE_SIZE * SEVERAL_CONDITION_EDGE / EDGE_PARTS
        }
        _ => CONDITION_IMAGE_SIZE * MANY_CONDITION_EDGE / EDGE_PARTS,
    }
}

/// `calculate_dimensions`: the 32-aligned size of `max_size` squared area at `ratio`, truncating.
fn calculate_dimensions(max_size: usize, ratio: f64) -> (usize, usize) {
    let width = (max_size as f64 * max_size as f64 * ratio).sqrt();
    let height = width / ratio;
    let align = |v: f64| (v / PATCH_SIZE as f64) as usize * PATCH_SIZE;
    (align(width), align(height))
}

/// `resize_pilimage` of one image, planned from its size alone: BOX halvings, a BICUBIC resize, a center crop.
struct Fit {
    halvings: usize,
    resized: (usize, usize),
    crop: (usize, usize),
    out: (usize, usize),
}

impl Fit {
    // the largest patch-aligned size within `max_size` squared, covered and center-cropped; Python rounds half to even
    fn plan(width: usize, height: usize, max_size: usize) -> Result<Self> {
        let (mut w, mut h, mut halvings) = (width, height, 0);
        while w.min(h) >= 2 * max_size {
            (w, h, halvings) = (w / 2, h / 2, halvings + 1);
        }
        let (wf, hf) = (w as f64, h as f64);
        let s_max = (max_size * max_size) as f64;
        let scale = (s_max / (wf * hf)).sqrt();
        let align = |v: f64| v as usize / PATCH_SIZE * PATCH_SIZE;
        let (rw, rh) = (
            (wf * scale).round_ties_even(),
            (hf * scale).round_ties_even(),
        );
        let (fw, fh) = ((wf * scale).floor(), (hf * scale).floor());
        let mut sizes = [
            (align(rw), align(rh)),
            (align(rw), align(fh)),
            (align(fw), align(rh)),
            (align(fw), align(fh)),
        ];
        // Python's sort is stable; the first size that fits wins, else the smallest
        sizes.sort_by_key(|&(w, h)| std::cmp::Reverse(w * h));
        let (nw, nh) = sizes
            .into_iter()
            .find(|&(w, h)| (w * h) as f64 <= s_max)
            .unwrap_or(sizes[3]);
        if nw == 0 || nh == 0 {
            inference_tensor::bail!(
                "a {width}x{height} reference image is too narrow for {PATCH_SIZE} px patches"
            );
        }
        let (s1, s2) = (wf / nw as f64, hf / nh as f64);
        let (resized, crop) = if s1 < s2 {
            let tall = (hf / s1).round_ties_even() as usize;
            ((nw, tall), (0, (tall - nh) / 2))
        } else {
            let wide = (wf / s2).round_ties_even() as usize;
            ((wide, nh), ((wide - nw) / 2, 0))
        };
        Ok(Self {
            halvings,
            resized,
            crop,
            out: (nw, nh),
        })
    }

    fn apply(&self, img: &RgbImage) -> RgbImage {
        let mut img = img.clone();
        for _ in 0..self.halvings {
            let (w, h) = (img.width() as usize / 2, img.height() as usize / 2);
            img = resize(&img, w, h, Filter::Box);
        }
        let resized = resize(&img, self.resized.0, self.resized.1, Filter::Bicubic);
        let (x, y) = self.crop;
        let (w, h) = self.out;
        image::imageops::crop_imm(&resized, x as u32, y as u32, w as u32, h as u32).to_image()
    }
}

/// The Qwen3-VL processor's rows for `img` appended to `out`: (x - 127.5) / 127.5, the one frame doubled in time.
fn vit_rows(img: &RgbImage, out: &mut Vec<f32>) -> [u32; 3] {
    let (w, h) = smart_resize(img.width() as usize, img.height() as usize);
    let resized;
    let img = if (w, h) == (img.width() as usize, img.height() as usize) {
        img
    } else {
        // the processor resizes only outside its pixel range, in torchvision's bicubic; Pillow's is within 2 levels
        resized = resize(img, w, h, Filter::Bicubic);
        &resized
    };
    let (gh, gw) = (h / VIT_PATCH_SIZE, w / VIT_PATCH_SIZE);
    let p = VIT_PATCH_SIZE;
    for bh in 0..gh / VIT_MERGE_SIZE {
        for bw in 0..gw / VIT_MERGE_SIZE {
            for ih in 0..VIT_MERGE_SIZE {
                for iw in 0..VIT_MERGE_SIZE {
                    let (row, col) = (
                        (bh * VIT_MERGE_SIZE + ih) * p,
                        (bw * VIT_MERGE_SIZE + iw) * p,
                    );
                    for c in 0..CHANNELS {
                        for _ in 0..VIT_TEMPORAL_PATCH_SIZE {
                            for y in 0..p {
                                for x in 0..p {
                                    let v = img.get_pixel((col + x) as u32, (row + y) as u32)[c];
                                    out.push((f32::from(v) - 127.5) / 127.5);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    [1, gh as u32, gw as u32]
}

/// Qwen2-VL's `smart_resize` on the processor's own pixel range, at the patch-times-merge factor.
fn smart_resize(width: usize, height: usize) -> (usize, usize) {
    let factor = (VIT_PATCH_SIZE * VIT_MERGE_SIZE) as f64;
    let (w, h) = (width as f64, height as f64);
    let mut h_bar = (h / factor).round_ties_even() * factor;
    let mut w_bar = (w / factor).round_ties_even() * factor;
    if h_bar * w_bar > VIT_MAX_PIXELS as f64 {
        let beta = (h * w / VIT_MAX_PIXELS as f64).sqrt();
        h_bar = factor.max((h / beta / factor).floor() * factor);
        w_bar = factor.max((w / beta / factor).floor() * factor);
    } else if h_bar * w_bar < VIT_MIN_PIXELS as f64 {
        let beta = (VIT_MIN_PIXELS as f64 / (h * w)).sqrt();
        h_bar = (h * beta / factor).ceil() * factor;
        w_bar = (w * beta / factor).ceil() * factor;
    }
    (w_bar as usize, h_bar as usize)
}

/// `img`'s 32 px patches in [-1, 1] appended to `out`, row-major, each (C, 32, 32); torchvision scales by 1 / 255.
fn pixel_patches(img: &RgbImage, out: &mut Vec<f32>) {
    let (rows, cols) = (
        img.height() as usize / PATCH_SIZE,
        img.width() as usize / PATCH_SIZE,
    );
    for r in 0..rows {
        for c in 0..cols {
            for ch in 0..CHANNELS {
                for y in 0..PATCH_SIZE {
                    for x in 0..PATCH_SIZE {
                        let v = img
                            .get_pixel((c * PATCH_SIZE + x) as u32, (r * PATCH_SIZE + y) as u32)
                            [ch];
                        out.push((f32::from(v) * (1.0 / 255.0) - 0.5) / 0.5);
                    }
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Filter {
    Box,
    Bicubic,
    Lanczos,
}

impl Filter {
    fn support(self) -> f64 {
        match self {
            Self::Box => 0.5,
            Self::Bicubic => 2.0,
            Self::Lanczos => 3.0,
        }
    }

    fn weight(self, x: f64) -> f64 {
        match self {
            Self::Box => {
                if x > -0.5 && x <= 0.5 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Bicubic => {
                // Pillow's a = -0.5
                const A: f64 = -0.5;
                let x = x.abs();
                if x < 1.0 {
                    ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
                } else if x < 2.0 {
                    (((x - 5.0) * x + 8.0) * x - 4.0) * A
                } else {
                    0.0
                }
            }
            Self::Lanczos => {
                if (-3.0..3.0).contains(&x) {
                    sinc(x) * sinc(x / 3.0)
                } else {
                    0.0
                }
            }
        }
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

/// One output position's first input index and its fixed-point weights.
struct Taps {
    start: usize,
    weights: Vec<i32>,
}

// Pillow's precompute_coeffs and normalize_coeffs_8bpc
fn taps(in_size: usize, out_size: usize, filter: Filter) -> Vec<Taps> {
    let scale = in_size as f64 / out_size as f64;
    let filter_scale = scale.max(1.0);
    let inv_scale = 1.0 / filter_scale;
    let support = filter.support() * filter_scale;
    (0..out_size)
        .map(|xx| {
            let center = (xx as f64 + 0.5) * scale;
            let xmin = ((center - support + 0.5) as isize).max(0) as usize;
            let xmax = ((center + support + 0.5) as isize).min(in_size as isize) as usize;
            let raw: Vec<f64> = (xmin..xmax)
                .map(|x| filter.weight((x as f64 - center + 0.5) * inv_scale))
                .collect();
            let total: f64 = raw.iter().sum();
            let one = f64::from(1u32 << PRECISION_BITS);
            let weights = raw
                .iter()
                .map(|&w| {
                    let w = if total == 0.0 { w } else { w / total };
                    if w < 0.0 {
                        (-0.5 + w * one) as i32
                    } else {
                        (0.5 + w * one) as i32
                    }
                })
                .collect();
            Taps {
                start: xmin,
                weights,
            }
        })
        .collect()
}

fn clip8(acc: i32) -> u8 {
    (acc >> PRECISION_BITS).clamp(0, 255) as u8
}

/// Pillow's `Image.resize` of an RGB image: a horizontal pass, then a vertical one, each rounded to 8 bits.
fn resize(img: &RgbImage, width: usize, height: usize, filter: Filter) -> RgbImage {
    let (in_w, in_h) = (img.width() as usize, img.height() as usize);
    if (width, height) == (in_w, in_h) {
        return img.clone();
    }
    let rounding = 1i32 << (PRECISION_BITS - 1);
    let mut horizontal = img.clone();
    if width != in_w {
        let taps = taps(in_w, width, filter);
        horizontal = RgbImage::new(width as u32, in_h as u32);
        for y in 0..in_h {
            for (x, tap) in taps.iter().enumerate() {
                let mut acc = [rounding; CHANNELS];
                for (i, &w) in tap.weights.iter().enumerate() {
                    let px = img.get_pixel((tap.start + i) as u32, y as u32);
                    for c in 0..CHANNELS {
                        acc[c] += i32::from(px[c]) * w;
                    }
                }
                horizontal.put_pixel(x as u32, y as u32, image::Rgb(acc.map(clip8)));
            }
        }
    }
    if height == in_h {
        return horizontal;
    }
    let taps = taps(in_h, height, filter);
    let mut out = RgbImage::new(width as u32, height as u32);
    for (y, tap) in taps.iter().enumerate() {
        for x in 0..width {
            let mut acc = [rounding; CHANNELS];
            for (i, &w) in tap.weights.iter().enumerate() {
                let px = horizontal.get_pixel(x as u32, (tap.start + i) as u32);
                for c in 0..CHANNELS {
                    acc[c] += i32::from(px[c]) * w;
                }
            }
            out.put_pixel(x as u32, y as u32, image::Rgb(acc.map(clip8)));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_keeps_the_area_patch_aligned() {
        let img = RgbImage::from_fn(1792, 2304, |x, y| image::Rgb([x as u8, y as u8, 7]));
        let fit = Fit::plan(1792, 2304, 1024).unwrap();
        assert_eq!(fit.out, (896, 1152));
        let fitted = fit.apply(&img);
        assert_eq!((fitted.width(), fitted.height()), (896, 1152));
    }

    #[test]
    fn a_sliver_is_refused_rather_than_cropped_to_nothing() {
        let img = RgbImage::new(4000, 8);
        assert!(Fit::plan(img.width() as usize, img.height() as usize, 512).is_err());
    }

    #[test]
    fn a_resize_to_the_same_size_is_a_copy() {
        let img = RgbImage::from_fn(64, 32, |x, y| image::Rgb([x as u8, y as u8, 3]));
        assert_eq!(resize(&img, 64, 32, Filter::Lanczos), img);
    }

    #[test]
    fn box_halving_averages_each_block() {
        let img = RgbImage::from_fn(4, 2, |x, _| {
            image::Rgb([if x % 2 == 0 { 10 } else { 20 }; 3])
        });
        let half = resize(&img, 2, 1, Filter::Box);
        assert!(half.pixels().all(|p| p.0 == [15; 3]));
    }

    #[test]
    fn condition_sizes_follow_the_reference_count() {
        assert_eq!(calculate_dimensions(384, 1792.0 / 2304.0), (320, 416));
        assert_eq!(condition_size(5), 288);
        assert_eq!(reference_size(2, 2048), 1536);
    }
}
