//! The Gaussian splatting training data, written in the reference's `gs_data` layout (gen_gs_data.py with
//! `--save_normal --split_sky` and its default densities): every clip frame (the first replaced by its start view)
//! with its aligned depth where it has one, panorama and polar crops with the cleaned panorama depth, MoGe-2 normals
//! for every image, the aligned cloud plus the panorama cloud, and a sky cloud apart.

use std::io::Write;
use std::path::Path;

use inference_tensor::{DType, Device, Result};
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde_json::{Map, Value, json};

use super::align::{Alignment, adaptive_voxel_size, voxel_first};
use super::panorama::{Rgb8, uv_to_direction};
use super::scene::{Scene, View, turn_about_z};
use super::trajectory::{self, Mat4};
use crate::geometry::invert4;
use crate::moge::MoGe;

// gen_gs_data's densities: panorama crops every 20 degrees at three pitches, upper polar crops every 45 degrees at four,
// lower every 90 at two
const PANO_STARTS: [[f64; 3]; 3] = [[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.5], [-1.0, 0.0, -0.5]];
const PANO_STEP_DEG: f64 = 20.0;
const POLAR_UP_STARTS: [[f64; 3]; 4] = [
    [-1.0, 0.0, 1.0],
    [-1.0, 0.0, 2.14],
    [-1.0, 0.0, 4.70],
    [0.1, 0.0, 1.0],
];
const POLAR_UP_STEP_DEG: f64 = 45.0;
const POLAR_DOWN_STARTS: [[f64; 3]; 2] = [[-1.0, 0.0, -1.0], [0.1, 0.0, -1.0]];
const POLAR_DOWN_STEP_DEG: f64 = 90.0;
// the panorama cloud's export size and the sky cloud's caps (convert_rgbd2pcd_panorama's dropout, then gen_gs_data's)
const GLOBAL_POINTS: usize = 2_000_000;
const SKY_DROPOUT: usize = 1_000_000;
const SKY_POINTS: usize = 300_000;
// smooth_sky_depth_boundary(transition_width=100, method="mean"): 31 x 31 box means spread into the sky
const SKY_TRANSITION: f64 = 100.0;
const SKY_BOX: usize = 31;
const SEED: u64 = 0;

pub struct GsFrame<'a> {
    pub name: String,
    pub image: &'a Rgb8,
    pub c2w: Mat4,
    pub intrinsics: [[f64; 3]; 3],
    /// metric z-depth, 0 invalid
    pub depth: Option<&'a [f32]>,
}

fn save_rgb(img: &Rgb8, path: &Path) -> Result<()> {
    image::RgbImage::from_raw(img.width as u32, img.height as u32, img.pixels.clone())
        .expect("three bytes a pixel")
        .save(path)
        .map_err(inference_tensor::Error::wrap)
}

// save_16bit_png_depth: float16 bits as 16-bit grayscale
fn save_depth(depth: &[f32], width: usize, height: usize, path: &Path) -> Result<()> {
    let bits = depth
        .iter()
        .map(|&d| half::f16::from_f32(d).to_bits())
        .collect();
    image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::from_raw(width as u32, height as u32, bits)
        .expect("one value a pixel")
        .save(path)
        .map_err(inference_tensor::Error::wrap)
}

fn write_points_ply(path: &Path, points: &[[f32; 3]], colors: &[[u8; 3]]) -> Result<()> {
    crate::worldmirror::output::write_points_ply(path, points, colors)
}

// box sums of a w x h map over (2r + 1)^2 windows, zero outside (avg_pool2d with zero padding, times the area)
fn box_sum(v: &[f64], w: usize, h: usize, r: usize) -> Vec<f64> {
    let mut sat = vec![0f64; (w + 1) * (h + 1)];
    for y in 0..h {
        let mut row = 0.0;
        for x in 0..w {
            row += v[y * w + x];
            sat[(y + 1) * (w + 1) + x + 1] = sat[y * (w + 1) + x + 1] + row;
        }
    }
    let mut out = vec![0f64; w * h];
    for y in 0..h {
        let (y0, y1) = (y.saturating_sub(r), (y + r + 1).min(h));
        for x in 0..w {
            let (x0, x1) = (x.saturating_sub(r), (x + r + 1).min(w));
            out[y * w + x] =
                sat[y1 * (w + 1) + x1] - sat[y0 * (w + 1) + x1] - sat[y1 * (w + 1) + x0]
                    + sat[y0 * (w + 1) + x0];
        }
    }
    out
}

// exact Euclidean distance from each pixel to the nearest `false` one (Felzenszwalb's separable transform)
fn distance_transform(inside: &[bool], w: usize, h: usize) -> Vec<f64> {
    const FAR: f64 = 1e20;
    // the lower envelope of parabolas rooted at each sample (z[0] = -inf keeps k from leaving the envelope)
    fn pass(f: &[f64]) -> Vec<f64> {
        let n = f.len();
        let cross = |q: usize, p: usize| {
            ((f[q] + (q * q) as f64) - (f[p] + (p * p) as f64)) / (2.0 * q as f64 - 2.0 * p as f64)
        };
        let mut v = vec![0usize; n];
        let mut z = vec![0f64; n + 1];
        let mut k = 0;
        z[0] = f64::NEG_INFINITY;
        z[1] = f64::INFINITY;
        for q in 1..n {
            let mut s = cross(q, v[k]);
            while s <= z[k] {
                k -= 1;
                s = cross(q, v[k]);
            }
            k += 1;
            v[k] = q;
            z[k] = s;
            z[k + 1] = f64::INFINITY;
        }
        k = 0;
        (0..n)
            .map(|q| {
                while z[k + 1] < q as f64 {
                    k += 1;
                }
                (q as f64 - v[k] as f64).powi(2) + f[v[k]]
            })
            .collect()
    }
    let mut g = inside
        .iter()
        .map(|&i| if i { FAR } else { 0.0 })
        .collect::<Vec<_>>();
    for x in 0..w {
        let col = (0..h).map(|y| g[y * w + x]).collect::<Vec<_>>();
        for (y, v) in pass(&col).into_iter().enumerate() {
            g[y * w + x] = v;
        }
    }
    for y in 0..h {
        let row = pass(&g[y * w..(y + 1) * w]);
        g[y * w..(y + 1) * w].copy_from_slice(&row);
    }
    g.into_iter().map(f64::sqrt).collect()
}

// smooth_sky_depth_boundary: sky at the maximum depth, eased from the surrounding foreground's mean depth at its edge
fn sky_depth(scene: &Scene) -> Vec<f32> {
    let (w, h) = (scene.width, scene.height);
    let sky = &scene.sky;
    let max = scene.distance.iter().copied().fold(0f32, f32::max);
    let depth = (0..w * h)
        .map(|p| {
            if sky[p] {
                f64::from(max)
            } else {
                f64::from(scene.distance[p])
            }
        })
        .collect::<Vec<_>>();
    let mut flood = (0..w * h)
        .map(|p| if sky[p] { 0.0 } else { depth[p] })
        .collect::<Vec<_>>();
    let mut valid = sky
        .iter()
        .map(|&s| if s { 0.0 } else { 1.0 })
        .collect::<Vec<f64>>();
    let r = SKY_BOX / 2;
    for _ in 0..(SKY_TRANSITION as usize / r + 10) {
        let sums = box_sum(&flood, w, h, r);
        let counts = box_sum(&valid, w, h, r);
        let mut grew = false;
        let (mut next_flood, mut next_valid) = (flood.clone(), valid.clone());
        for p in 0..w * h {
            if sky[p] && counts[p] > 0.0 {
                grew |= valid[p] < 0.5;
                next_flood[p] = sums[p] / (counts[p] + 1e-8);
                next_valid[p] = 1.0;
            }
        }
        if !grew {
            break;
        }
        flood = next_flood;
        valid = next_valid;
    }
    let fg_mean = {
        let fg = (0..w * h)
            .filter(|&p| !sky[p])
            .map(|p| depth[p])
            .collect::<Vec<_>>();
        fg.iter().sum::<f64>() / fg.len().max(1) as f64
    };
    let dist = distance_transform(sky, w, h);
    (0..w * h)
        .map(|p| {
            if !sky[p] {
                return depth[p] as f32;
            }
            let boundary = if flood[p] == 0.0 { fg_mean } else { flood[p] };
            let t = (dist[p] / SKY_TRANSITION).clamp(0.0, 1.0);
            let t = t * t * (3.0 - 2.0 * t);
            (boundary * (1.0 - t) + f64::from(max) * t).min(f64::from(max)) as f32
        })
        .collect()
}

fn crop_directions() -> Vec<(String, [f64; 3])> {
    let mut out = Vec::new();
    let mut ring = |kind: &str, starts: &[[f64; 3]], step: f64| {
        let n = (360.0 / step) as usize;
        for (layer, s) in starts.iter().enumerate() {
            for az in 0..n {
                let d = if az == 0 {
                    *s
                } else {
                    turn_about_z(*s, step * az as f64)
                };
                out.push((format!("{kind}_L{layer:02}_A{az:04}"), d));
            }
        }
    };
    ring("panorama", &PANO_STARTS, PANO_STEP_DEG);
    ring("polar_up", &POLAR_UP_STARTS, POLAR_UP_STEP_DEG);
    ring("polar_down", &POLAR_DOWN_STARTS, POLAR_DOWN_STEP_DEG);
    out
}

fn camera_json(c2w: &Mat4, k: &[[f64; 3]; 3]) -> Value {
    json!({ "extrinsic": invert4(c2w), "intrinsic": k })
}

/// Writes `dir` (the gs_data layout) from the clip frames, the panorama scene, the alignment and MoGe-2's normals.
pub fn write_gs_data(
    dir: &Path,
    frames: &[GsFrame],
    pano: &Rgb8,
    scene: &Scene,
    alignment: &Alignment,
    moge: &MoGe,
    device: &Device,
) -> Result<usize> {
    for sub in ["images", "depths", "normals"] {
        std::fs::create_dir_all(dir.join(sub))?;
    }
    let (vw, vh) = trajectory::view_size();
    let normal = |name: &str, img: &Rgb8| -> Result<()> {
        let chw = super::run::image_unit_tensor(img, device)?;
        let out = moge.infer(&chw, trajectory::VIEW_FOV_X_DEG)?;
        // zero where MoGe-2 finds no surface, as its infer(apply_mask) leaves them: the trainer fills sky depth there
        let n = out
            .normal
            .broadcast_mul(&out.mask.to_dtype(DType::F32)?.unsqueeze(2)?)?
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        // ((n + 1) / 2 * 255).astype(uint8)
        let px = n
            .iter()
            .map(|v| ((v + 1.0) / 2.0 * 255.0).clamp(0.0, 255.0) as u8)
            .collect();
        image::RgbImage::from_raw(img.width as u32, img.height as u32, px)
            .expect("three bytes a pixel")
            .save(dir.join("normals").join(format!("{name}.png")))
            .map_err(inference_tensor::Error::wrap)
    };
    let mut cameras = Map::new();
    for f in frames {
        save_rgb(f.image, &dir.join("images").join(format!("{}.png", f.name)))?;
        if let Some(d) = f.depth {
            save_depth(
                d,
                f.image.width,
                f.image.height,
                &dir.join("depths").join(format!("{}.png", f.name)),
            )?;
        }
        normal(&f.name, f.image)?;
        cameras.insert(f.name.clone(), camera_json(&f.c2w, &f.intrinsics));
    }
    let k = trajectory::intrinsics(
        vw,
        vh,
        trajectory::VIEW_FOV_X_DEG,
        trajectory::VIEW_FOV_Y_DEG,
    );
    for (name, d) in crop_directions() {
        let view: View = scene.view(pano, trajectory::look_at_c2w(d));
        save_rgb(&view.image, &dir.join("images").join(format!("{name}.png")))?;
        save_depth(
            &scene.view_depth(&view),
            vw,
            vh,
            &dir.join("depths").join(format!("{name}.png")),
        )?;
        normal(&name, &view.image)?;
        cameras.insert(name, camera_json(&view.c2w, &k));
    }
    let count = cameras.len();
    cameras.insert("width".into(), json!(vw));
    cameras.insert("height".into(), json!(vh));
    std::fs::write(
        dir.join("cameras.json"),
        serde_json::to_string_pretty(&Value::Object(cameras))
            .map_err(inference_tensor::Error::wrap)?,
    )?;

    // the aligned cloud, then the panorama cloud thinned to about GLOBAL_POINTS voxels
    let voxel = adaptive_voxel_size(&scene.points, GLOBAL_POINTS);
    let keep = voxel_first(&scene.points, voxel);
    let mut points = alignment.points.clone();
    let mut colors = alignment.colors.clone();
    points.extend(keep.iter().map(|&i| scene.points[i]));
    colors.extend(keep.iter().map(|&i| scene.colors[i]));
    write_points_ply(&dir.join("points.ply"), &points, &colors)?;

    // the sky at its eased depth, thinned at the panorama cloud's voxel size
    let depth = sky_depth(scene);
    let (w, h) = (scene.width, scene.height);
    let mut sky = (0..w * h).filter(|&p| scene.sky[p]).collect::<Vec<_>>();
    let mut rng = StdRng::seed_from_u64(SEED);
    if sky.len() > SKY_DROPOUT {
        sky = rand::seq::index::sample(&mut rng, sky.len(), SKY_DROPOUT)
            .into_iter()
            .map(|j| sky[j])
            .collect();
    }
    let sky_points = sky
        .iter()
        .map(|&p| {
            let ray = uv_to_direction(
                ((p % w) as f64 + 0.5) / w as f64,
                ((p / w) as f64 + 0.5) / h as f64,
            );
            ray.map(|r| (r * f64::from(depth[p])) as f32)
        })
        .collect::<Vec<_>>();
    let sky_colors = sky
        .iter()
        .map(|&p| [0, 1, 2].map(|c| pano.pixels[p * 3 + c]))
        .collect::<Vec<_>>();
    let mut keep = voxel_first(&sky_points, voxel);
    if keep.len() > SKY_POINTS {
        keep = rand::seq::index::sample(&mut rng, keep.len(), SKY_POINTS)
            .into_iter()
            .map(|j| keep[j])
            .collect();
    }
    write_points_ply(
        &dir.join("sky_points.ply"),
        &keep.iter().map(|&i| sky_points[i]).collect::<Vec<_>>(),
        &keep.iter().map(|&i| sky_colors[i]).collect::<Vec<_>>(),
    )?;
    let mut meta = std::fs::File::create(dir.join("meta_info.json"))?;
    write!(
        meta,
        "{}",
        json!({ "scene_type": if scene.outdoor { "outdoor" } else { "indoor" } })
    )?;
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distance_transform_is_euclidean() {
        let (w, h) = (9, 7);
        let mut inside = vec![true; w * h];
        inside[3 * w + 4] = false;
        let d = distance_transform(&inside, w, h);
        for y in 0..h {
            for x in 0..w {
                let want = ((x as f64 - 4.0).powi(2) + (y as f64 - 3.0).powi(2)).sqrt();
                assert!(
                    (d[y * w + x] - want).abs() < 1e-9,
                    "({x}, {y}): {} vs {want}",
                    d[y * w + x]
                );
            }
        }
    }

    #[test]
    fn crop_counts_match_gen_gs_data() {
        let names = crop_directions();
        let count = |p: &str| names.iter().filter(|(n, _)| n.starts_with(p)).count();
        assert_eq!(
            (count("panorama_"), count("polar_up_"), count("polar_down_")),
            (54, 32, 8)
        );
    }
}
