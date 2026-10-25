//! A prediction to files: cameras (JSON), depth (npy and png), normals (png), a coloured point cloud and standard 3D
//! Gaussian splats (binary PLY), with the masks and voxel merges the reference pipeline applies.

use std::io::Write;
use std::path::Path;

use inference_tensor::{D, DType, Device, Result, Tensor};
use rayon::prelude::*;
use serde::Serialize;

use super::{Camera, Prediction};

/// The settings a run writes its outputs with; the defaults are the reference pipeline's.
#[derive(Clone, Debug, Serialize)]
pub struct OutputSettings {
    /// drop pixels at depth discontinuities that are also normal discontinuities
    pub edge_mask: bool,
    /// relative depth jump, and normal angle in degrees, that mark an edge
    pub edge_depth_rtol: f32,
    pub edge_normal_degrees: f32,
    /// drop pixels the model marks as sky: `sigmoid(gs_depth_mask_logits) <= sky_threshold`
    pub sky_mask: bool,
    pub sky_threshold: f32,
    /// voxel size the point cloud and the splats are merged at, and their most points after merging
    pub voxel_size: f32,
    pub max_points: usize,
    pub max_gaussians: usize,
    /// splats whose largest scale is above this quantile are dropped
    pub gaussian_scale_quantile: f32,
}

impl Default for OutputSettings {
    fn default() -> Self {
        Self {
            edge_mask: true,
            edge_depth_rtol: 0.03,
            edge_normal_degrees: 1.0,
            sky_mask: true,
            sky_threshold: 0.45,
            voxel_size: 0.002,
            max_points: 2_000_000,
            max_gaussians: 5_000_000,
            gaussian_scale_quantile: 0.98,
        }
    }
}

// The seed the over-cap subsampling draws with
const SAMPLE_SEED: u64 = 42;
// An opacity is clamped this far inside (0, 1) before it is written as a logit
const OPACITY_EPS: f32 = 1e-6;
// points.ply rows: xyz as f32, rgb as u8; gaussians.ply rows: 17 f32 fields
const POINT_ROW_BYTES: usize = 3 * 4 + 3;
const SPLAT_FIELDS: usize = 17;

/// What a run wrote: counts for the report and `meta.json`.
#[derive(Clone, Debug, Default, Serialize)]
pub struct WrittenCounts {
    pub points: usize,
    pub gaussians: usize,
}

/// Writes every output of `pred` for the [0, 1] images `[frames, 3, H, W]` into `dir`.
pub fn write_outputs(
    dir: &Path,
    pred: &Prediction,
    images: &Tensor,
    settings: &OutputSettings,
) -> Result<WrittenCounts> {
    std::fs::create_dir_all(dir.join("depth"))?;
    std::fs::create_dir_all(dir.join("normal"))?;
    let (frames, h, w) = (pred.cameras.len(), pred.height, pred.width);
    write_cameras(&dir.join("camera_params.json"), &pred.cameras)?;
    let depth = host(&pred.depth.squeeze(D::Minus1)?)?;
    let gs_depth = host(&pred.gs_depth.squeeze(D::Minus1)?)?;
    let normals = host(&pred.normals)?;
    (0..frames)
        .into_par_iter()
        .try_for_each(|f| -> Result<()> {
            let d = &depth[f * h * w..][..h * w];
            write_npy(&dir.join(format!("depth/depth_{f:04}.npy")), d, h, w)?;
            write_depth_png(&dir.join(format!("depth/depth_{f:04}.png")), d, h, w)?;
            write_normal_png(
                &dir.join(format!("normal/normal_{f:04}.png")),
                &normals[f * h * w * 3..][..h * w * 3],
                h,
                w,
            )
        })?;
    let sky = if settings.sky_mask {
        let p = host(&inference_tensor::nn::ops::sigmoid(
            &pred.gs_depth_mask_logits,
        )?)?;
        Some(
            p.iter()
                .map(|&v| v > settings.sky_threshold)
                .collect::<Vec<_>>(),
        )
    } else {
        None
    };
    let mut points_mask = vec![true; frames * h * w];
    let mut gs_mask = vec![true; frames * h * w];
    points_mask
        .par_chunks_mut(h * w)
        .zip(gs_mask.par_chunks_mut(h * w))
        .enumerate()
        .for_each(|(f, (points_mask, gs_mask))| {
            let range = f * h * w..(f + 1) * h * w;
            if settings.edge_mask {
                let n_edge = normals_edge(
                    &normals[f * h * w * 3..][..h * w * 3],
                    h,
                    w,
                    settings.edge_normal_degrees,
                );
                let d_edge = depth_edge(&depth[range.clone()], h, w, settings.edge_depth_rtol);
                let g_edge = depth_edge(&gs_depth[range.clone()], h, w, settings.edge_depth_rtol);
                for i in 0..h * w {
                    points_mask[i] &= !(d_edge[i] && n_edge[i]);
                    gs_mask[i] &= !(g_edge[i] && n_edge[i]);
                }
            }
            if let Some(sky) = &sky {
                for (i, s) in sky[range].iter().enumerate() {
                    points_mask[i] &= s;
                    gs_mask[i] &= s;
                }
            }
        });
    let points = write_points(dir, pred, images, &depth, &points_mask, settings)?;
    let gaussians = write_gaussians(&dir.join("gaussians.ply"), pred, &gs_mask, settings)?;
    Ok(WrittenCounts { points, gaussians })
}

fn host(t: &Tensor) -> Result<Vec<f32>> {
    t.to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()
}

#[derive(Serialize)]
struct CameraEntry {
    camera_id: usize,
    matrix: Vec<Vec<f32>>,
}

#[derive(Serialize)]
struct CameraFile {
    num_cameras: usize,
    extrinsics: Vec<CameraEntry>,
    intrinsics: Vec<CameraEntry>,
}

// the reference's camera_params.json: camera-to-world 4x4 and pixel intrinsics 3x3 per camera
fn write_cameras(path: &Path, cameras: &[Camera]) -> Result<()> {
    let entry = |i: usize, rows: Vec<Vec<f32>>| CameraEntry {
        camera_id: i,
        matrix: rows,
    };
    let file = CameraFile {
        num_cameras: cameras.len(),
        extrinsics: cameras
            .iter()
            .enumerate()
            .map(|(i, c)| entry(i, c.camera_to_world.iter().map(|r| r.to_vec()).collect()))
            .collect(),
        intrinsics: cameras
            .iter()
            .enumerate()
            .map(|(i, c)| entry(i, c.intrinsics.iter().map(|r| r.to_vec()).collect()))
            .collect(),
    };
    let text = serde_json::to_string_pretty(&file).map_err(inference_tensor::Error::wrap)?;
    Ok(std::fs::write(path, text)?)
}

// A float32 `[h, w]` array in NumPy's .npy format (version 1.0)
fn write_npy(path: &Path, values: &[f32], h: usize, w: usize) -> Result<()> {
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({h}, {w}), }}");
    // magic (6) + version (2) + header length (2) + header, padded with spaces to a multiple of 64, newline last
    let total = 10 + header.len() + 1;
    header.push_str(&" ".repeat(total.next_multiple_of(64) - total));
    header.push('\n');
    let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
    out.write_all(b"\x93NUMPY\x01\x00")?;
    out.write_all(&(header.len() as u16).to_le_bytes())?;
    out.write_all(header.as_bytes())?;
    for v in values {
        out.write_all(&v.to_le_bytes())?;
    }
    Ok(out.flush()?)
}

// (d - min) / (max(d - min) + 1e-9) as 8-bit grey, truncated as the reference casts
fn write_depth_png(path: &Path, depth: &[f32], h: usize, w: usize) -> Result<()> {
    let min = depth.iter().copied().fold(f32::INFINITY, f32::min);
    let max = depth.iter().map(|d| d - min).fold(0f32, f32::max);
    let pixels = depth
        .iter()
        .map(|d| (((d - min) / (max + 1e-9)).clamp(0.0, 1.0) * 255.0) as u8)
        .collect::<Vec<_>>();
    image::GrayImage::from_raw(w as u32, h as u32, pixels)
        .expect("one byte a pixel")
        .save(path)
        .map_err(inference_tensor::Error::wrap)
}

// (n + 1) / 2 as 8-bit RGB
fn write_normal_png(path: &Path, normals: &[f32], h: usize, w: usize) -> Result<()> {
    let pixels = normals
        .iter()
        .map(|n| (((n + 1.0) / 2.0).clamp(0.0, 1.0) * 255.0) as u8)
        .collect::<Vec<_>>();
    image::RgbImage::from_raw(w as u32, h as u32, pixels)
        .expect("three bytes a pixel")
        .save(path)
        .map_err(inference_tensor::Error::wrap)
}

// The max over each 3 x 3 window, ignoring the outside, of `values` `[h, w]`
fn max_pool3(values: &[f32], h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![f32::NEG_INFINITY; h * w];
    for y in 0..h {
        for x in 0..w {
            let mut m = f32::NEG_INFINITY;
            for yy in y.saturating_sub(1)..(y + 2).min(h) {
                for xx in x.saturating_sub(1)..(x + 2).min(w) {
                    m = m.max(values[yy * w + xx]);
                }
            }
            out[y * w + x] = m;
        }
    }
    out
}

// pixels whose 3 x 3 neighbourhood spans a relative depth jump above `rtol`
fn depth_edge(depth: &[f32], h: usize, w: usize, rtol: f32) -> Vec<bool> {
    let up = max_pool3(depth, h, w);
    let down = max_pool3(&depth.iter().map(|d| -d).collect::<Vec<_>>(), h, w);
    (0..h * w)
        .map(|i| (up[i] + down[i]) / depth[i] > rtol)
        .collect()
}

// pixels whose 3 x 3 neighbourhood (edge-padded) holds a normal more than `degrees` away, max-pooled again
fn normals_edge(normals: &[f32], h: usize, w: usize, degrees: f32) -> Vec<bool> {
    let units = normals
        .as_chunks::<3>()
        .0
        .iter()
        .map(|n| {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt() + 1e-12;
            [n[0] / len, n[1] / len, n[2] / len]
        })
        .collect::<Vec<_>>();
    let unit = |i: usize| units[i];
    let mut angle = vec![0f32; h * w];
    for y in 0..h {
        for x in 0..w {
            let c = unit(y * w + x);
            let mut worst = 0f32;
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let yy = (y as i64 + dy).clamp(0, h as i64 - 1) as usize;
                    let xx = (x as i64 + dx).clamp(0, w as i64 - 1) as usize;
                    let n = unit(yy * w + xx);
                    let dot = (c[0] * n[0] + c[1] * n[1] + c[2] * n[2]).clamp(-1.0, 1.0);
                    worst = worst.max(dot.acos());
                }
            }
            angle[y * w + x] = worst;
        }
    }
    let threshold = degrees.to_radians();
    max_pool3(&angle, h, w)
        .iter()
        .map(|&a| a > threshold)
        .collect()
}

/// Indices grouped by voxel (`floor(p / size)`, offset by the minimum), groups in increasing flat-cell order and each
/// group's indices increasing.
struct VoxelGroups {
    order: Vec<u32>,
    /// group `g` is `order[starts[g]..starts[g + 1]]`
    starts: Vec<usize>,
}

impl VoxelGroups {
    fn len(&self) -> usize {
        self.starts.len() - 1
    }

    fn group(&self, g: usize) -> &[u32] {
        &self.order[self.starts[g]..self.starts[g + 1]]
    }
}

fn voxel_groups(points: &[[f32; 3]], size: f32) -> VoxelGroups {
    let cells = points
        .par_iter()
        .map(|p| p.map(|v| (v / size).floor() as i64))
        .collect::<Vec<_>>();
    let (lo, hi) = cells
        .par_iter()
        .fold(
            || ([i64::MAX; 3], [i64::MIN; 3]),
            |(mut lo, mut hi), c| {
                for d in 0..3 {
                    lo[d] = lo[d].min(c[d]);
                    hi[d] = hi[d].max(c[d]);
                }
                (lo, hi)
            },
        )
        .reduce(
            || ([i64::MAX; 3], [i64::MIN; 3]),
            |a, b| {
                (
                    [0, 1, 2].map(|d| a.0[d].min(b.0[d])),
                    [0, 1, 2].map(|d| a.1[d].max(b.1[d])),
                )
            },
        );
    let extent = [hi[1] - lo[1] + 1, hi[2] - lo[2] + 1];
    let mut keyed = cells
        .par_iter()
        .enumerate()
        .map(|(i, c)| {
            let flat = ((c[0] - lo[0]) * extent[0] + (c[1] - lo[1])) * extent[1] + (c[2] - lo[2]);
            (flat, i as u32)
        })
        .collect::<Vec<_>>();
    keyed.par_sort_unstable();
    let mut starts = Vec::with_capacity(keyed.len() + 1);
    for (k, pair) in keyed.iter().enumerate() {
        if k == 0 || pair.0 != keyed[k - 1].0 {
            starts.push(k);
        }
    }
    starts.push(keyed.len());
    VoxelGroups {
        order: keyed.into_iter().map(|(_, i)| i).collect(),
        starts,
    }
}

// `count` of `0..n` without replacement, seeded: the cap's sample (a partial Fisher-Yates over a xorshift stream)
fn sample(n: usize, count: usize) -> Vec<usize> {
    let mut state = SAMPLE_SEED.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut slots = (0..n as u32).collect::<Vec<_>>();
    (0..count)
        .map(|i| {
            let j = i + (next() % (n - i) as u64) as usize;
            slots.swap(i, j);
            slots[i] as usize
        })
        .collect()
}

fn write_points(
    dir: &Path,
    pred: &Prediction,
    images: &Tensor,
    depth: &[f32],
    mask: &[bool],
    settings: &OutputSettings,
) -> Result<usize> {
    let (h, w) = (pred.height, pred.width);
    let world = host(&super::unproject(&pred.depth, &pred.cameras)?)?;
    let rgb = host(&images.permute((0, 2, 3, 1))?)?;
    let mut points = Vec::new();
    let mut colors = Vec::new();
    for i in 0..pred.cameras.len() * h * w {
        if mask[i] && depth[i] > 1e-8 {
            points.push([world[i * 3], world[i * 3 + 1], world[i * 3 + 2]]);
            colors.push([0, 1, 2].map(|c| (rgb[i * 3 + c] * 255.0) as u8));
        }
    }
    let groups = voxel_groups(&points, settings.voxel_size);
    let (mut points, mut colors): (Vec<[f32; 3]>, Vec<[u8; 3]>) = (0..groups.len())
        .into_par_iter()
        .map(|g| {
            let g = groups.group(g);
            let n = g.len() as f64;
            let mean =
                |pick: &dyn Fn(usize) -> f64| g.iter().map(|&i| pick(i as usize)).sum::<f64>() / n;
            let p = [0, 1, 2].map(|d| mean(&|i| f64::from(points[i][d])) as f32);
            let c = [0, 1, 2]
                .map(|d| mean(&|i| f64::from(colors[i][d])).round().clamp(0.0, 255.0) as u8);
            (p, c)
        })
        .unzip();
    if points.len() > settings.max_points {
        let keep = sample(points.len(), settings.max_points);
        points = keep.iter().map(|&i| points[i]).collect();
        colors = keep.iter().map(|&i| colors[i]).collect();
    }
    let mut out = std::io::BufWriter::new(std::fs::File::create(dir.join("points.ply"))?);
    write!(
        out,
        "ply\nformat binary_little_endian 1.0\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\n\
         property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n",
        points.len()
    )?;
    let mut body = vec![0u8; points.len() * POINT_ROW_BYTES];
    body.par_chunks_mut(POINT_ROW_BYTES)
        .zip(points.par_iter().zip(&colors))
        .for_each(|(row, (p, c))| {
            for (d, v) in p.iter().enumerate() {
                row[d * 4..][..4].copy_from_slice(&v.to_le_bytes());
            }
            row[12..].copy_from_slice(c);
        });
    out.write_all(&body)?;
    out.flush()?;
    Ok(points.len())
}

// The `q` quantile (linear interpolation, as `torch.quantile`) of `values`
fn quantile(values: &[f32], q: f32) -> f32 {
    let mut values = values.to_vec();
    let pos = q * (values.len() - 1) as f32;
    let (lo, frac) = (pos.floor() as usize, pos - pos.floor());
    let (_, &mut at_lo, above) = values.select_nth_unstable_by(lo, f32::total_cmp);
    let at_hi = above
        .iter()
        .copied()
        .min_by(f32::total_cmp)
        .unwrap_or(at_lo);
    at_lo + (at_hi - at_lo) * frac
}

fn write_gaussians(
    path: &Path,
    pred: &Prediction,
    mask: &[bool],
    settings: &OutputSettings,
) -> Result<usize> {
    let s = &pred.splats;
    let (means, scales, quats) = (host(&s.means)?, host(&s.scales)?, host(&s.quats)?);
    let (colors, opacities, weights) = (host(&s.sh_dc)?, host(&s.opacities)?, host(&s.weights)?);
    let keep = (0..mask.len()).filter(|&i| mask[i]).collect::<Vec<_>>();
    let vec3 = |v: &[f32], i: usize| [v[i * 3], v[i * 3 + 1], v[i * 3 + 2]];
    let kept_means = keep.iter().map(|&i| vec3(&means, i)).collect::<Vec<_>>();
    let groups = voxel_groups(&kept_means, settings.voxel_size);
    // a merged voxel: weighted means, scales and colours, the renormalized weighted quaternion, and opacity
    // sum(w^2) / sum(w) (the reference's merge, which builds it from the weights alone)
    let merged = groups.len() < keep.len();
    let mut splats = (0..groups.len())
        .into_par_iter()
        .map(|g| {
            let g = groups.group(g);
            // with nothing merged the reference keeps every splat as predicted; once anything merges, every voxel
            // (a lone splat too) takes the merge formula
            if !merged {
                let i = keep[g[0] as usize];
                return (
                    vec3(&means, i),
                    vec3(&scales, i),
                    [
                        quats[i * 4],
                        quats[i * 4 + 1],
                        quats[i * 4 + 2],
                        quats[i * 4 + 3],
                    ],
                    vec3(&colors, i),
                    opacities[i],
                );
            }
            let wsum = g
                .iter()
                .map(|&k| weights[keep[k as usize]])
                .sum::<f32>()
                .max(1e-8);
            let avg = |v: &[f32], d: usize, n: usize| {
                g.iter()
                    .map(|&k| v[keep[k as usize] * n + d] * weights[keep[k as usize]])
                    .sum::<f32>()
                    / wsum
            };
            let mut q = [0, 1, 2, 3].map(|d| {
                g.iter()
                    .map(|&k| quats[keep[k as usize] * 4 + d] * weights[keep[k as usize]])
                    .sum::<f32>()
            });
            let len = q.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
            q = q.map(|v| v / len);
            let opacity = g
                .iter()
                .map(|&k| weights[keep[k as usize]].powi(2))
                .sum::<f32>()
                / wsum;
            (
                [0, 1, 2].map(|d| avg(&means, d, 3)),
                [0, 1, 2].map(|d| avg(&scales, d, 3)),
                q,
                [0, 1, 2].map(|d| avg(&colors, d, 3)),
                opacity,
            )
        })
        .collect::<Vec<_>>();
    if splats.len() > settings.max_gaussians {
        let pick = sample(splats.len(), settings.max_gaussians);
        splats = pick.iter().map(|&i| splats[i]).collect();
    }
    let largest = splats
        .iter()
        .map(|s| s.1[0].max(s.1[1]).max(s.1[2]))
        .collect::<Vec<_>>();
    let limit = quantile(&largest, settings.gaussian_scale_quantile);
    let kept = splats
        .iter()
        .zip(&largest)
        .filter(|(_, l)| **l <= limit)
        .map(|(s, _)| s)
        .collect::<Vec<_>>();
    let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
    write!(
        out,
        "ply\nformat binary_little_endian 1.0\nelement vertex {}\n",
        kept.len()
    )?;
    for name in [
        "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0",
        "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    ] {
        writeln!(out, "property float {name}")?;
    }
    out.write_all(b"end_header\n")?;
    let mut body = vec![0u8; kept.len() * SPLAT_FIELDS * 4];
    body.par_chunks_mut(SPLAT_FIELDS * 4).zip(&kept).for_each(
        |(bytes, (mean, scale, quat, color, opacity))| {
            let o = opacity.clamp(OPACITY_EPS, 1.0 - OPACITY_EPS);
            let mut row = [0f32; SPLAT_FIELDS];
            row[..3].copy_from_slice(mean);
            row[6..9].copy_from_slice(color);
            row[9] = (o / (1.0 - o)).ln();
            row[10..13].copy_from_slice(&scale.map(f32::ln));
            row[13..].copy_from_slice(quat);
            for (d, v) in row.iter().enumerate() {
                bytes[d * 4..][..4].copy_from_slice(&v.to_le_bytes());
            }
        },
    );
    out.write_all(&body)?;
    out.flush()?;
    Ok(kept.len())
}
