//! Metric distance for an equirectangular panorama, as WorldStereo's `pred_pano_depth` builds it with MoGe-2: 512 x 512
//! perspective views at the directions of a once-subdivided icosahedron, each view's distance predicted, then the
//! log-distance gradients and Laplacians of all views merged in one least-squares solve, coarse to fine.

use inference_tensor::{DType, Device, Result, Tensor};
use rayon::prelude::*;

use crate::moge::MoGe;

const VIEW_SIZE: usize = 512;
const VIEW_FOV_DEG: f64 = 90.0;
// the panorama is worked on at most this wide
const MAX_WORK_WIDTH: usize = 1920;
// the merge recurses to half size while the larger side is above this
const MERGE_BASE: usize = 256;
// LSMR's stopping tolerances, as the reference passes them
const LSMR_TOL: f64 = 1e-5;
const LSMR_CONLIM: f64 = 1e8;
// unseen pixels take this quantile of the merged distance
const FILL_QUANTILE: f64 = 0.999;
// the bottom rows are blended toward their row average over this share of the height
const SOUTH_POLE_RATIO: f64 = 0.05;
// OpenCV's remap quantizes sub-pixel positions to 1 / 32
const REMAP_STEPS: f64 = 32.0;

type Mat3 = [[f64; 3]; 3];

/// An 8-bit RGB image, row-major.
pub struct Rgb8 {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u8>,
}

/// The merged panorama: distance per pixel at the input size, and whether any view saw it.
pub struct PanoDepth {
    pub width: usize,
    pub height: usize,
    pub distance: Vec<f32>,
    pub mask: Vec<bool>,
}

// utils3d's icosahedron (unnormalized) and its faces
fn icosahedron() -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let a = (1.0 + 5f64.sqrt()) / 2.0;
    let v = vec![
        [0.0, 1.0, a],
        [0.0, -1.0, a],
        [0.0, 1.0, -a],
        [0.0, -1.0, -a],
        [1.0, a, 0.0],
        [-1.0, a, 0.0],
        [1.0, -a, 0.0],
        [-1.0, -a, 0.0],
        [a, 0.0, 1.0],
        [a, 0.0, -1.0],
        [-a, 0.0, 1.0],
        [-a, 0.0, -1.0],
    ];
    let f = vec![
        [0, 1, 8],
        [0, 8, 4],
        [0, 4, 5],
        [0, 5, 10],
        [0, 10, 1],
        [3, 2, 9],
        [3, 9, 6],
        [3, 6, 7],
        [3, 7, 11],
        [3, 11, 2],
        [1, 6, 8],
        [8, 9, 4],
        [4, 2, 5],
        [5, 11, 10],
        [10, 7, 1],
        [2, 4, 9],
        [9, 8, 6],
        [6, 1, 7],
        [7, 10, 11],
        [11, 5, 2],
    ];
    (v, f)
}

fn normalize(v: [f64; 3]) -> [f64; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    v.map(|x| x / n)
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// World-to-camera rotations (OpenCV, eye at the origin, up +z) of the views: the 42 directions of a once-subdivided
/// icosahedron less the two poles, whose look-at is degenerate (the reference gets NaN cameras there, which see
/// nothing in the merge).
pub fn view_rotations() -> Vec<Mat3> {
    let (mut verts, faces) = icosahedron();
    let mut cache = std::collections::HashMap::new();
    for [a, b, c] in faces {
        for (i, j) in [(a, b), (b, c), (c, a)] {
            let key = (i.min(j), i.max(j));
            if let std::collections::hash_map::Entry::Vacant(slot) = cache.entry(key) {
                slot.insert(verts.len());
                verts.push(normalize(
                    [0, 1, 2].map(|d| (verts[i][d] + verts[j][d]) / 2.0),
                ));
            }
        }
    }
    let up = [0.0, 0.0, 1.0];
    verts
        .iter()
        .filter_map(|&z| {
            let x = cross([-up[0], -up[1], -up[2]], z);
            let xn = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt();
            if xn < 1e-9 {
                return None;
            }
            let y = cross(z, x);
            Some([normalize(x), normalize(y), normalize(z)])
        })
        .collect()
}

// ERP uv (u right, v down, both 0..1) of a direction, and back
pub(crate) fn direction_to_uv(d: [f64; 3]) -> (f64, f64) {
    let d = normalize(d);
    let u = 1.0 - (d[1].atan2(d[0]) / (2.0 * std::f64::consts::PI)).rem_euclid(1.0);
    (u, d[2].clamp(-1.0, 1.0).acos() / std::f64::consts::PI)
}

pub(crate) fn uv_to_direction(u: f64, v: f64) -> [f64; 3] {
    let (theta, phi) = (
        (1.0 - u) * 2.0 * std::f64::consts::PI,
        v * std::f64::consts::PI,
    );
    [phi.sin() * theta.cos(), phi.sin() * theta.sin(), phi.cos()]
}

// OpenCV remap's quantized bilinear taps for coordinate `x`: base index and weight of the next sample
fn remap_taps(x: f64) -> (isize, f64) {
    let q = (x * REMAP_STEPS).round() / REMAP_STEPS;
    let base = q.floor();
    (base as isize, q - base)
}

// cv2.remap(INTER_LINEAR, BORDER_WRAP) of 8-bit RGB at pixel (x, y), rounded back to 8 bits
fn sample_wrap(img: &Rgb8, x: f64, y: f64) -> [u8; 3] {
    let (x0, fx) = remap_taps(x);
    let (y0, fy) = remap_taps(y);
    let wrap = |i: isize, n: usize| i.rem_euclid(n as isize) as usize;
    let at = |xi: isize, yi: isize, c: usize| {
        f64::from(img.pixels[(wrap(yi, img.height) * img.width + wrap(xi, img.width)) * 3 + c])
    };
    [0, 1, 2].map(|c| {
        let top = at(x0, y0, c) * (1.0 - fx) + at(x0 + 1, y0, c) * fx;
        let bottom = at(x0, y0 + 1, c) * (1.0 - fx) + at(x0 + 1, y0 + 1, c) * fx;
        (top * (1.0 - fy) + bottom * fy).round().clamp(0.0, 255.0) as u8
    })
}

// cv2.resize(INTER_AREA) taps when enlarging: OpenCV's area-weighted linear coefficients
fn area_up_taps(src: usize, dst: usize) -> Vec<(usize, usize, f64)> {
    let (scale, inv) = (src as f64 / dst as f64, dst as f64 / src as f64);
    (0..dst)
        .map(|d| {
            let s = (d as f64 * scale).floor() as isize;
            let f = (d + 1) as f64 - (s + 1) as f64 * inv;
            let f = if f <= 0.0 { 0.0 } else { f - f.floor() };
            let lo = s.clamp(0, src as isize - 1) as usize;
            let hi = (s + 1).clamp(0, src as isize - 1) as usize;
            (lo, hi, f)
        })
        .collect()
}

// Area-overlap weights for shrinking `src` samples to `dst` (cv2.resize INTER_AREA when shrinking)
fn area_down_taps(src: usize, dst: usize) -> Vec<Vec<(usize, f64)>> {
    let scale = src as f64 / dst as f64;
    (0..dst)
        .map(|d| {
            let (lo, hi) = (d as f64 * scale, (d + 1) as f64 * scale);
            (lo.floor() as usize..(hi.ceil() as usize).min(src))
                .map(|s| {
                    let w = (hi.min((s + 1) as f64) - lo.max(s as f64)).max(0.0) / scale;
                    (s, w)
                })
                .filter(|t| t.1 > 0.0)
                .collect()
        })
        .collect()
}

fn resize_area_rgb8(img: &Rgb8, width: usize, height: usize) -> Rgb8 {
    let (cols, rows) = (
        area_down_taps(img.width, width),
        area_down_taps(img.height, height),
    );
    let mut pixels = vec![0u8; width * height * 3];
    pixels
        .par_chunks_mut(width * 3)
        .enumerate()
        .for_each(|(y, row)| {
            for (x, col) in cols.iter().enumerate() {
                for c in 0..3 {
                    let mut acc = 0.0;
                    for &(sy, wy) in &rows[y] {
                        for &(sx, wx) in col {
                            acc += wy * wx * f64::from(img.pixels[(sy * img.width + sx) * 3 + c]);
                        }
                    }
                    row[x * 3 + c] = acc.round().clamp(0.0, 255.0) as u8;
                }
            }
        });
    Rgb8 {
        width,
        height,
        pixels,
    }
}

/// One 512 x 512 depth view of the panorama, as pred_pano_depth splits it.
pub fn split_view(pano: &Rgb8, rot: &Mat3) -> Rgb8 {
    split_perspective(
        pano,
        rot,
        (VIEW_FOV_DEG, VIEW_FOV_DEG),
        (VIEW_SIZE, VIEW_SIZE),
    )
}

/// A perspective view of the panorama (camera at the origin, world-to-camera rotation `rot`) as
/// split_panorama_image(INTER_AREA) makes it: remapped at a "safe" size (half the panorama height tall), then resized
/// to `width x height` with OpenCV's area interpolation. 8-bit RGB.
pub fn split_perspective(
    pano: &Rgb8,
    rot: &Mat3,
    (fov_x, fov_y): (f64, f64),
    (width, height): (usize, usize),
) -> Rgb8 {
    let safe_h = pano.height / 2;
    let safe_w = (safe_h as f64 / height as f64 * width as f64).round() as usize;
    let fx = 0.5 / (fov_x.to_radians() / 2.0).tan();
    let fy = 0.5 / (fov_y.to_radians() / 2.0).tan();
    let mut safe_px = vec![0u8; safe_w * safe_h * 3];
    safe_px
        .par_chunks_mut(safe_w * 3)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..safe_w {
                let (u, v) = (
                    (x as f64 + 0.5) / safe_w as f64,
                    (y as f64 + 0.5) / safe_h as f64,
                );
                let cam = [(u - 0.5) / fx, (v - 0.5) / fy, 1.0];
                // camera to world is the transpose of the world-to-camera rotation
                let world = [0, 1, 2].map(|j| (0..3).map(|i| rot[i][j] * cam[i]).sum::<f64>());
                let (pu, pv) = direction_to_uv(world);
                let rgb = sample_wrap(
                    pano,
                    pu * pano.width as f64 - 0.5,
                    pv * pano.height as f64 - 0.5,
                );
                row[x * 3..x * 3 + 3].copy_from_slice(&rgb);
            }
        });
    let safe = Rgb8 {
        width: safe_w,
        height: safe_h,
        pixels: safe_px,
    };
    resize_area(&safe, width, height)
}

// cv2.resize(INTER_AREA) of 8-bit RGB: area averaging when shrinking, OpenCV's area-linear taps when enlarging
pub(crate) fn resize_area(img: &Rgb8, width: usize, height: usize) -> Rgb8 {
    if (width, height) == (img.width, img.height) {
        return Rgb8 {
            width,
            height,
            pixels: img.pixels.clone(),
        };
    }
    if width <= img.width && height <= img.height {
        return resize_area_rgb8(img, width, height);
    }
    let (cols, rows) = (
        area_up_taps(img.width, width),
        area_up_taps(img.height, height),
    );
    let mut out = vec![0u8; width * height * 3];
    out.par_chunks_mut(width * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let (y0, y1, fy) = rows[y];
            for (x, &(x0, x1, fx)) in cols.iter().enumerate() {
                for c in 0..3 {
                    let at =
                        |yy: usize, xx: usize| f64::from(img.pixels[(yy * img.width + xx) * 3 + c]);
                    let top = at(y0, x0) * (1.0 - fx) + at(y0, x1) * fx;
                    let bottom = at(y1, x0) * (1.0 - fx) + at(y1, x1) * fx;
                    row[x * 3 + c] =
                        (top * (1.0 - fy) + bottom * fy).round().clamp(0.0, 255.0) as u8;
                }
            }
        });
    Rgb8 {
        width,
        height,
        pixels: out,
    }
}

/// The panorama's merged distance: every view through MoGe-2, then the merge, hole fill and south-pole smoothing.
pub fn pano_depth(moge: &MoGe, pano: &Rgb8, device: &Device) -> Result<PanoDepth> {
    let work_h = MAX_WORK_WIDTH.min(MAX_WORK_WIDTH * pano.height / pano.width);
    let work_w = MAX_WORK_WIDTH.min(MAX_WORK_WIDTH * pano.width / pano.height);
    let shrunk;
    let work = if work_h < pano.height {
        shrunk = resize_area_rgb8(pano, work_w, work_h);
        &shrunk
    } else {
        pano
    };
    let rots = view_rotations();
    let mut distances = Vec::with_capacity(rots.len());
    let mut masks = Vec::with_capacity(rots.len());
    for rot in &rots {
        let view = split_view(work, rot);
        let chw = Tensor::from_vec(
            view.pixels
                .iter()
                .map(|&p| f32::from(p) / 255.0)
                .collect::<Vec<_>>(),
            (VIEW_SIZE, VIEW_SIZE, 3),
            &Device::Cpu,
        )?
        .permute((2, 0, 1))?
        .to_device(device)?;
        let out = moge.infer(&chw, VIEW_FOV_DEG)?;
        let dist = out
            .points
            .sqr()?
            .sum(2)?
            .sqrt()?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mask = out
            .mask
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .map(|&m| m > 0.0)
            .collect::<Vec<_>>();
        distances.push(dist);
        masks.push(mask);
    }
    let (mut depth, mask) = merge(work.width, work.height, &distances, &masks, &rots);
    let fill = quantile(
        &depth
            .iter()
            .map(|&d| f64::from(d))
            .filter(|d| d.is_finite())
            .collect::<Vec<_>>(),
        FILL_QUANTILE,
    );
    for (d, &m) in depth.iter_mut().zip(&mask) {
        if !m {
            *d = fill as f32;
        }
    }
    let (depth, mask) = if (work.width, work.height) != (pano.width, pano.height) {
        (
            resize_linear(&depth, (work.width, work.height), (pano.width, pano.height)),
            resize_nearest(&mask, (work.width, work.height), (pano.width, pano.height)),
        )
    } else {
        (depth, mask)
    };
    let distance = smooth_south_pole(depth, pano.width, pano.height);
    Ok(PanoDepth {
        width: pano.width,
        height: pano.height,
        distance,
        mask,
    })
}

// numpy's linear-interpolation quantile
pub(crate) fn quantile(values: &[f64], q: f64) -> f64 {
    let mut v = values.to_vec();
    v.sort_unstable_by(f64::total_cmp);
    if v.is_empty() {
        return 0.0;
    }
    let pos = q * (v.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(v.len() - 1);
    v[lo] + (v[hi] - v[lo]) * (pos - lo as f64)
}

// cv2.resize(INTER_LINEAR) on floats: half-pixel centres, edges clamped
fn resize_linear(src: &[f32], (sw, sh): (usize, usize), (dw, dh): (usize, usize)) -> Vec<f32> {
    let taps = |n_src: usize, n_dst: usize| -> Vec<(usize, usize, f32)> {
        (0..n_dst)
            .map(|d| {
                let s = ((d as f64 + 0.5) * n_src as f64 / n_dst as f64 - 0.5).max(0.0);
                let i0 = (s.floor() as usize).min(n_src - 1);
                (i0, (i0 + 1).min(n_src - 1), (s - i0 as f64) as f32)
            })
            .collect()
    };
    let (cols, rows) = (taps(sw, dw), taps(sh, dh));
    let mut out = vec![0f32; dw * dh];
    out.par_chunks_mut(dw).enumerate().for_each(|(y, row)| {
        let (y0, y1, fy) = rows[y];
        for (x, &(x0, x1, fx)) in cols.iter().enumerate() {
            let top = src[y0 * sw + x0] * (1.0 - fx) + src[y0 * sw + x1] * fx;
            let bottom = src[y1 * sw + x0] * (1.0 - fx) + src[y1 * sw + x1] * fx;
            row[x] = top * (1.0 - fy) + bottom * fy;
        }
    });
    out
}

fn resize_nearest(src: &[bool], (sw, sh): (usize, usize), (dw, dh): (usize, usize)) -> Vec<bool> {
    (0..dh)
        .flat_map(|y| {
            let sy = (y * sh / dh).min(sh - 1);
            (0..dw).map(move |x| src[sy * sw + (x * sw / dw).min(sw - 1)])
        })
        .collect()
}

// The reference's smooth_south_pole_depth: the bottom row set to a robust mean of the last three, the rows above
// blended toward their own robust (IQR-filtered) mean, fading out over SOUTH_POLE_RATIO of the height
fn smooth_south_pole(mut depth: Vec<f32>, width: usize, height: usize) -> Vec<f32> {
    let band = (height as f64 * SOUTH_POLE_RATIO) as usize;
    if band == 0 {
        return depth;
    }
    let row = |d: &Vec<f32>, y: usize| d[y * width..(y + 1) * width].to_vec();
    let reference = if band > 3 {
        (height - 3..height)
            .flat_map(|y| row(&depth, y))
            .collect::<Vec<_>>()
    } else {
        row(&depth, height - 1)
    };
    let valid = reference
        .iter()
        .map(|&v| f64::from(v))
        .filter(|v| v.is_finite() && *v > 0.0)
        .collect::<Vec<_>>();
    let average = if valid.is_empty() {
        0.0
    } else {
        let (lo, hi) = (quantile(&valid, 0.1), quantile(&valid, 0.9));
        let kept = valid
            .iter()
            .copied()
            .filter(|v| *v >= lo && *v <= hi)
            .collect::<Vec<_>>();
        if kept.is_empty() {
            quantile(&valid, 0.5)
        } else {
            kept.iter().sum::<f64>() / kept.len() as f64
        }
    };
    let original = depth.clone();
    for x in 0..width {
        depth[(height - 1) * width + x] = average as f32;
    }
    for i in 1..band {
        let y = height - 1 - i;
        let weight = (band - i) as f64 / band as f64;
        let current = row(&original, y);
        let valid = current
            .iter()
            .map(|&v| f64::from(v))
            .filter(|v| v.is_finite() && *v > 0.0)
            .collect::<Vec<_>>();
        if valid.is_empty() {
            continue;
        }
        let row_avg = if valid.len() > 1 {
            let (q25, q75) = (quantile(&valid, 0.25), quantile(&valid, 0.75));
            let (lo, hi) = (q25 - 1.5 * (q75 - q25), q75 + 1.5 * (q75 - q25));
            let kept = valid
                .iter()
                .copied()
                .filter(|v| *v >= lo && *v <= hi)
                .collect::<Vec<_>>();
            if kept.is_empty() {
                quantile(&valid, 0.5)
            } else {
                kept.iter().sum::<f64>() / kept.len() as f64
            }
        } else {
            valid[0]
        };
        for x in 0..width {
            let v = f64::from(current[x]);
            depth[y * width + x] = ((1.0 - weight) * v + weight * row_avg) as f32;
        }
    }
    depth
}

/// The reference's merge_panorama_depth_gpu: each view's log distance warped into the panorama, its x/y gradients
/// and Laplacian averaged over the views that see both ends, then solved for the log distance in least squares
/// (LSMR), the half-size solution as the starting point. Returns distance and the union of the views' masks.
pub fn merge(
    width: usize,
    height: usize,
    distances: &[Vec<f32>],
    masks: &[Vec<bool>],
    rots: &[Mat3],
) -> (Vec<f32>, Vec<bool>) {
    let init = if width.max(height) > MERGE_BASE {
        let (half, _) = merge(width / 2, height / 2, distances, masks, rots);
        Some(resize_linear(
            &half,
            (width / 2, height / 2),
            (width, height),
        ))
    } else {
        None
    };
    let n = width * height;
    let f = 0.5 / (VIEW_FOV_DEG.to_radians() / 2.0).tan();
    let mut gx_sum = vec![0f64; n];
    let mut gx_cnt = vec![0u32; n];
    let mut gy_sum = vec![0f64; n - width];
    let mut gy_cnt = vec![0u32; n - width];
    let mut lap_sum = vec![0f64; n];
    let mut lap_cnt = vec![0u32; n];
    let mut seen = vec![false; n];
    for ((dist, mask), rot) in distances.iter().zip(masks).zip(rots) {
        // this view's log distance and mask warped into the panorama
        let warped = (0..n)
            .into_par_iter()
            .map(|p| {
                let (x, y) = (p % width, p / width);
                let d = uv_to_direction(
                    (x as f64 + 0.5) / width as f64,
                    (y as f64 + 0.5) / height as f64,
                );
                let c = [0, 1, 2].map(|i| (0..3).map(|j| rot[i][j] * d[j]).sum::<f64>());
                let (u, v) = (f * c[0] / c[2] + 0.5, f * c[1] / c[2] + 0.5);
                if !(c[2] > 0.0 && u > 0.0 && v > 0.0 && u < 1.0 && v < 1.0) {
                    return (0.0, false);
                }
                let (px, py) = (
                    u.clamp(0.0, 1.0) * VIEW_SIZE as f64 - 0.5,
                    v.clamp(0.0, 1.0) * VIEW_SIZE as f64 - 0.5,
                );
                let clamp = |i: isize| i.clamp(0, VIEW_SIZE as isize - 1) as usize;
                let (x0, fx) = remap_taps(px);
                let (y0, fy) = remap_taps(py);
                let at =
                    |xi: isize, yi: isize| f64::from(dist[clamp(yi) * VIEW_SIZE + clamp(xi)]).ln();
                let top = at(x0, y0) * (1.0 - fx) + at(x0 + 1, y0) * fx;
                let bottom = at(x0, y0 + 1) * (1.0 - fx) + at(x0 + 1, y0 + 1) * fx;
                let m = mask[clamp(py.round() as isize) * VIEW_SIZE + clamp(px.round() as isize)];
                (top * (1.0 - fy) + bottom * fy, m)
            })
            .collect::<Vec<_>>();
        let (val, ok): (Vec<f64>, Vec<bool>) = warped.into_iter().unzip();
        for y in 0..height {
            for x in 0..width {
                let p = y * width + x;
                seen[p] |= ok[p];
                let right = y * width + (x + 1) % width;
                if ok[p] && ok[right] {
                    gx_sum[p] += val[p] - val[right];
                    gx_cnt[p] += 1;
                }
                if y + 1 < height && ok[p] && ok[p + width] {
                    gy_sum[p] += val[p] - val[p + width];
                    gy_cnt[p] += 1;
                }
                let up = if y > 0 { p - width } else { p };
                let down = if y + 1 < height { p + width } else { p };
                let left = y * width + (x + width - 1) % width;
                if ok[p] && ok[up] && ok[down] && ok[left] && ok[right] {
                    lap_sum[p] += val[up] + val[down] + val[left] + val[right] - 4.0 * val[p];
                    lap_cnt[p] += 1;
                }
            }
        }
    }
    let avg = |s: &[f64], c: &[u32]| -> Vec<f64> {
        s.iter()
            .zip(c)
            .map(|(&s, &c)| s / f64::from(c).max(1e-3))
            .collect()
    };
    let system = Stencil {
        width,
        height,
        gx: gx_cnt.iter().map(|&c| c > 0).collect(),
        gy: gy_cnt.iter().map(|&c| c > 0).collect(),
        lap: lap_cnt.iter().map(|&c| c > 0).collect(),
    };
    let b = system.pack(
        &avg(&gx_sum, &gx_cnt),
        &avg(&gy_sum, &gy_cnt),
        &avg(&lap_sum, &lap_cnt),
    );
    let x0 = init.map(|d| d.iter().map(|&v| f64::from(v).ln()).collect::<Vec<_>>());
    let x = lsmr(&system, &b, x0.as_deref());
    (x.iter().map(|&v| v.exp() as f32).collect(), seen)
}

// The merge's linear system as stencils: masked x-gradient rows (wrapping), y-gradient rows, then Laplacian rows
// (wrapping in x, edge-replicated in y)
struct Stencil {
    width: usize,
    height: usize,
    gx: Vec<bool>,
    gy: Vec<bool>,
    lap: Vec<bool>,
}

impl Stencil {
    fn rows(&self) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        let pick = |m: &[bool]| {
            m.iter()
                .enumerate()
                .filter(|(_, k)| **k)
                .map(|(i, _)| i)
                .collect::<Vec<_>>()
        };
        (pick(&self.gx), pick(&self.gy), pick(&self.lap))
    }

    fn pack(&self, gx: &[f64], gy: &[f64], lap: &[f64]) -> Vec<f64> {
        let (rx, ry, rl) = self.rows();
        rx.iter()
            .map(|&i| gx[i])
            .chain(ry.iter().map(|&i| gy[i]))
            .chain(rl.iter().map(|&i| lap[i]))
            .collect()
    }

    fn neighbours(&self, p: usize) -> [usize; 4] {
        let (w, h) = (self.width, self.height);
        let (x, y) = (p % w, p / w);
        [
            if y > 0 { p - w } else { p },
            if y + 1 < h { p + w } else { p },
            y * w + (x + w - 1) % w,
            y * w + (x + 1) % w,
        ]
    }

    fn apply(&self, x: &[f64], rows: &(Vec<usize>, Vec<usize>, Vec<usize>)) -> Vec<f64> {
        let w = self.width;
        let mut out = Vec::with_capacity(rows.0.len() + rows.1.len() + rows.2.len());
        out.par_extend(
            rows.0
                .par_iter()
                .map(|&p| x[p] - x[(p / w) * w + (p % w + 1) % w]),
        );
        out.par_extend(rows.1.par_iter().map(|&p| x[p] - x[p + w]));
        out.par_extend(rows.2.par_iter().map(|&p| {
            let [u, d, l, r] = self.neighbours(p);
            x[u] + x[d] + x[l] + x[r] - 4.0 * x[p]
        }));
        out
    }

    fn apply_t(&self, y: &[f64], rows: &(Vec<usize>, Vec<usize>, Vec<usize>)) -> Vec<f64> {
        let w = self.width;
        let mut out = vec![0f64; self.width * self.height];
        let (n0, n1) = (rows.0.len(), rows.1.len());
        for (k, &p) in rows.0.iter().enumerate() {
            out[p] += y[k];
            out[(p / w) * w + (p % w + 1) % w] -= y[k];
        }
        for (k, &p) in rows.1.iter().enumerate() {
            out[p] += y[n0 + k];
            out[p + w] -= y[n0 + k];
        }
        for (k, &p) in rows.2.iter().enumerate() {
            let v = y[n0 + n1 + k];
            for q in self.neighbours(p) {
                out[q] += v;
            }
            out[p] -= 4.0 * v;
        }
        out
    }
}

fn norm(v: &[f64]) -> f64 {
    v.par_iter().map(|x| x * x).sum::<f64>().sqrt()
}

// SciPy's LSMR (Fong and Saunders) with atol = btol = LSMR_TOL, solving for the correction from `x0`
fn lsmr(a: &Stencil, b: &[f64], x0: Option<&[f64]>) -> Vec<f64> {
    let rows = a.rows();
    let n = a.width * a.height;
    let mut x = x0.map_or_else(|| vec![0f64; n], <[f64]>::to_vec);
    let ax0 = a.apply(&x, &rows);
    let mut u: Vec<f64> = b.iter().zip(&ax0).map(|(b, ax)| b - ax).collect();
    let normb = norm(b);
    let mut beta = norm(&u);
    if beta > 0.0 {
        u.iter_mut().for_each(|v| *v /= beta);
    }
    let mut v = a.apply_t(&u, &rows);
    let mut alpha = norm(&v);
    if alpha > 0.0 {
        v.iter_mut().for_each(|x| *x /= alpha);
    }
    let maxiter = b.len().min(n);
    let (mut zetabar, mut alphabar, mut rho, mut rhobar, mut cbar, mut sbar) =
        (alpha * beta, alpha, 1.0, 1.0, 1.0, 0.0);
    let mut h = v.clone();
    let mut hbar = vec![0f64; n];
    let (mut betadd, mut betad, mut rhodold, mut tautildeold, mut thetatilde, mut zeta, mut d) =
        (beta, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
    let mut norma2 = alpha * alpha;
    let (mut maxrbar, mut minrbar) = (0.0f64, 1e100f64);
    let ctol = 1.0 / LSMR_CONLIM;
    if alpha * beta == 0.0 {
        return x;
    }
    for _ in 0..maxiter {
        let au = a.apply(&v, &rows);
        u.par_iter_mut()
            .zip(&au)
            .for_each(|(u, av)| *u = av - alpha * *u);
        beta = norm(&u);
        if beta > 0.0 {
            u.iter_mut().for_each(|x| *x /= beta);
            let atu = a.apply_t(&u, &rows);
            v.par_iter_mut()
                .zip(&atu)
                .for_each(|(v, at)| *v = at - beta * *v);
            alpha = norm(&v);
            if alpha > 0.0 {
                v.iter_mut().for_each(|x| *x /= alpha);
            }
        }
        let (chat, shat, alphahat) = sym_ortho(alphabar, 0.0);
        let rhoold = rho;
        let (c, s, r) = sym_ortho(alphahat, beta);
        rho = r;
        let thetanew = s * alpha;
        alphabar = c * alpha;
        let rhobarold = rhobar;
        let zetaold = zeta;
        let thetabar = sbar * rho;
        let rhotemp = cbar * rho;
        let (cb, sb, rb) = sym_ortho(cbar * rho, thetanew);
        cbar = cb;
        sbar = sb;
        rhobar = rb;
        zeta = cbar * zetabar;
        zetabar *= -sbar;
        let hb = thetabar * rho / (rhoold * rhobarold);
        hbar.par_iter_mut()
            .zip(&h)
            .for_each(|(hb_, h_)| *hb_ = h_ - hb * *hb_);
        let step = zeta / (rho * rhobar);
        x.par_iter_mut()
            .zip(&hbar)
            .for_each(|(x_, hb_)| *x_ += step * hb_);
        let hn = thetanew / rho;
        h.par_iter_mut()
            .zip(&v)
            .for_each(|(h_, v_)| *h_ = v_ - hn * *h_);
        // the residual norm estimate
        let betaacute = chat * betadd;
        let betacheck = -shat * betadd;
        let betahat = c * betaacute;
        betadd = -s * betaacute;
        let thetatildeold = thetatilde;
        let (ctildeold, stildeold, rhotildeold) = sym_ortho(rhodold, thetabar);
        thetatilde = stildeold * rhobar;
        rhodold = ctildeold * rhobar;
        betad = -stildeold * betad + ctildeold * betahat;
        tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold;
        let taud = (zeta - thetatilde * tautildeold) / rhodold;
        d += betacheck * betacheck;
        let normr = (d + (betad - taud).powi(2) + betadd * betadd).sqrt();
        norma2 += beta * beta;
        let norma = norma2.sqrt();
        norma2 += alpha * alpha;
        maxrbar = maxrbar.max(rhobarold);
        minrbar = minrbar.min(rhobarold);
        let conda = maxrbar.max(rhotemp) / minrbar.min(rhotemp);
        let normar = zetabar.abs();
        let normx = norm(&x);
        let test1 = normr / normb;
        let test2 = if norma * normr != 0.0 {
            normar / (norma * normr)
        } else {
            f64::INFINITY
        };
        let test3 = 1.0 / conda;
        let rtol = LSMR_TOL + LSMR_TOL * norma * normx / normb;
        if test3 <= ctol || test2 <= LSMR_TOL || test1 <= rtol {
            break;
        }
    }
    x
}

// SciPy's _sym_ortho: a stable Givens rotation (c, s, r) with r = hypot(a, b)
fn sym_ortho(a: f64, b: f64) -> (f64, f64, f64) {
    // numpy's sign: 0 for 0, where Rust's signum gives 1
    let sign = |v: f64| if v == 0.0 { 0.0 } else { v.signum() };
    if b == 0.0 {
        (sign(a), 0.0, a.abs())
    } else if a == 0.0 {
        (0.0, sign(b), b.abs())
    } else if b.abs() > a.abs() {
        let tau = a / b;
        let s = b.signum() / (1.0 + tau * tau).sqrt();
        (s * tau, s, b / s)
    } else {
        let tau = b / a;
        let c = a.signum() / (1.0 + tau * tau).sqrt();
        (c, c * tau, a / c)
    }
}

#[cfg(test)]
mod parity {
    use inference_tensor::{DType, Device};

    fn rel(a: &[f32], b: &[f32]) -> (f32, f32) {
        let max = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        let d2: f64 = a.iter().zip(b).map(|(x, y)| f64::from(x - y).powi(2)).sum();
        let b2: f64 = b.iter().map(|y| f64::from(*y).powi(2)).sum();
        (max, (d2 / b2).sqrt() as f32)
    }

    // PANO_REF: pred_pano_depth's split views, per-view distances and masks and merged distance; MOGE_DIR for the full run
    #[test]
    #[ignore = "needs a reference dump (and MoGe-2 weights and a GPU for the full run)"]
    fn pano_depth_matches_the_reference() -> inference_tensor::Result<()> {
        let Some(dump) = std::env::var_os("PANO_REF") else {
            return Ok(());
        };
        let refs = inference_tensor::safetensors::load(dump, &Device::Cpu)?;
        let (h, w, _) = refs["pano"].dims3()?;
        let pano = super::Rgb8 {
            width: w,
            height: h,
            pixels: refs["pano"].flatten_all()?.to_vec1::<u8>()?,
        };
        let rots = super::view_rotations();
        let views = refs["views"].to_vec3::<u8>().ok();
        let ref_views = refs["views"].flatten_all()?.to_vec1::<u8>()?;
        let per = super::VIEW_SIZE * super::VIEW_SIZE * 3;
        assert_eq!(rots.len() * per, ref_views.len(), "view count");
        drop(views);
        let mut worst = 0u8;
        let mut differing = 0usize;
        for (i, rot) in rots.iter().enumerate() {
            let ours = super::split_view(&pano, rot);
            for (a, b) in ours.pixels.iter().zip(&ref_views[i * per..(i + 1) * per]) {
                worst = worst.max(a.abs_diff(*b));
                differing += usize::from(a != b);
            }
        }
        println!(
            "split views: {:.3}% of values differ, max {worst} levels",
            100.0 * differing as f64 / ref_views.len() as f64
        );
        // the merge alone, on the reference's per-view distances and masks
        let pv = super::VIEW_SIZE * super::VIEW_SIZE;
        let dist = refs["view_distance"].flatten_all()?.to_vec1::<f32>()?;
        let mask = refs["view_mask"].flatten_all()?.to_vec1::<u8>()?;
        let distances = (0..rots.len())
            .map(|i| dist[i * pv..(i + 1) * pv].to_vec())
            .collect::<Vec<_>>();
        let masks = (0..rots.len())
            .map(|i| mask[i * pv..(i + 1) * pv].iter().map(|&m| m > 0).collect())
            .collect::<Vec<_>>();
        let start = std::time::Instant::now();
        let (merged, seen) = super::merge(w, h, &distances, &masks, &rots);
        let want = refs["distance"].flatten_all()?.to_vec1::<f32>()?;
        let want_mask = refs["mask"].flatten_all()?.to_vec1::<u8>()?;
        let both = (0..merged.len())
            .filter(|&p| seen[p] && want_mask[p] > 0)
            .collect::<Vec<_>>();
        let (max, r) = rel(
            &both.iter().map(|&p| merged[p]).collect::<Vec<_>>(),
            &both.iter().map(|&p| want[p]).collect::<Vec<_>>(),
        );
        let flips = seen
            .iter()
            .zip(&want_mask)
            .filter(|(a, b)| **a != (**b > 0))
            .count();
        println!(
            "merge on the reference views ({:.1} s): distance where both see it max {max:.3e} rel rms {r:.3e} (before fill and smoothing); mask flips {flips}",
            start.elapsed().as_secs_f64()
        );
        if let Some(dir) = std::env::var_os("MOGE_DIR") {
            let device = Device::new_cuda(0)?;
            let moge = crate::moge::MoGe::load(std::path::Path::new(&dir), DType::F32, &device)?;
            let start = std::time::Instant::now();
            let out = super::pano_depth(&moge, &pano, &device)?;
            let (max, r) = rel(&out.distance, &want);
            println!(
                "full pano depth ({:.1} s): max {max:.3e} rel rms {r:.3e}",
                start.elapsed().as_secs_f64()
            );
        }
        Ok(())
    }
}
