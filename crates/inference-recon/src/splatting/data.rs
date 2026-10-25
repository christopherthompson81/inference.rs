//! Gaussian splatting training data in HY-World's `gs_data` layout: `images/*.png`, optional `depths/*.png` (float16
//! bits in 16-bit PNG, 0 invalid) and `normals/*.png` ((n + 1) / 2 * 255), `cameras.json` (OpenCV world-to-camera
//! and pixel intrinsics per image), `points.ply` and `sky_points.ply`. Loaded the way world_gs_trainer's Parser does:
//! the clouds downsampled together, then cameras and points normalized (camera-centred similarity, principal axes, z
//! flipped up).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use inference_tensor::{Error, Result};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use crate::geometry::{invert4, mul4, symmetric_eigen};

pub type Mat4 = [[f64; 4]; 4];

// Parser defaults the trainer runs with
const TEST_EVERY: usize = 32;
const DOWNSAMPLE_POINTS: usize = 1_000_000;
const DOWNSAMPLE_SEED: u64 = 42;
const VOXEL_SEARCH_STEPS: usize = 10;
const VOXEL_SEARCH_TOL: f64 = 0.15;
const PANO_PREFIXES: [&str; 2] = ["panorama_", "polar_"];

pub struct View {
    pub name: String,
    /// camera-to-world in the normalized frame
    pub c2w: Mat4,
    pub k: [[f64; 3]; 3],
    pub width: usize,
    pub height: usize,
    pub image: PathBuf,
    pub depth: Option<PathBuf>,
    pub normal: Option<PathBuf>,
}

impl View {
    pub fn is_pano(&self) -> bool {
        PANO_PREFIXES.iter().any(|p| self.name.starts_with(p))
    }
}

pub struct SplatData {
    /// sorted by name
    pub views: Vec<View>,
    pub train: Vec<usize>,
    pub val: Vec<usize>,
    /// the downsampled cloud in the normalized frame; `is_sky` marks the sky points among them
    pub points: Vec<[f32; 3]>,
    pub colors: Vec<[u8; 3]>,
    pub is_sky: Vec<bool>,
    /// the first normalization's scale, applied to depths
    pub rescale: f64,
    /// largest camera distance from the cameras' mean, normalized frame
    pub scene_scale: f64,
    /// raw world to normalized
    pub transform: Mat4,
}

fn invalid(msg: impl std::fmt::Display) -> Error {
    Error::msg(msg.to_string())
}

pub type PointCloud = (Vec<[f32; 3]>, Vec<[u8; 3]>);

/// Points and 8-bit colors of a binary little-endian PLY with float x, y, z and uchar red, green, blue (other
/// properties skipped).
pub fn read_points_ply(path: &Path) -> Result<PointCloud> {
    let bytes = std::fs::read(path)?;
    let end = bytes
        .windows(11)
        .position(|w| w == b"end_header\n")
        .ok_or_else(|| invalid(format!("{}: no PLY header", path.display())))?
        + 11;
    let header = std::str::from_utf8(&bytes[..end]).map_err(Error::wrap)?;
    if !header.contains("format binary_little_endian") {
        return Err(invalid(format!(
            "{}: not binary little-endian",
            path.display()
        )));
    }
    let mut count = 0;
    let mut props: Vec<(String, usize)> = Vec::new();
    for line in header.lines() {
        let words = line.split_whitespace().collect::<Vec<_>>();
        match words.as_slice() {
            ["element", "vertex", n] => count = n.parse().map_err(Error::wrap)?,
            ["property", ty, name] => {
                let size = match *ty {
                    "float" | "float32" | "int" | "uint" => 4,
                    "double" | "float64" => 8,
                    "uchar" | "uint8" | "char" => 1,
                    "short" | "ushort" => 2,
                    _ => return Err(invalid(format!("{}: property type {ty}", path.display()))),
                };
                props.push(((*name).to_string(), size));
            }
            _ => {}
        }
    }
    let offset = |name: &str| -> Result<usize> {
        let at = props
            .iter()
            .position(|(n, _)| n == name)
            .ok_or_else(|| invalid(format!("{}: no {name}", path.display())))?;
        Ok(props[..at].iter().map(|(_, s)| s).sum())
    };
    let stride = props.iter().map(|(_, s)| s).sum::<usize>();
    let xyz = [offset("x")?, offset("y")?, offset("z")?];
    let rgb = [offset("red")?, offset("green")?, offset("blue")?];
    let body = &bytes[end..];
    if body.len() < count * stride {
        return Err(invalid(format!("{}: truncated", path.display())));
    }
    let f = |row: &[u8], o: usize| f32::from_le_bytes([row[o], row[o + 1], row[o + 2], row[o + 3]]);
    let mut points = Vec::with_capacity(count);
    let mut colors = Vec::with_capacity(count);
    for row in body[..count * stride].chunks_exact(stride) {
        points.push(xyz.map(|o| f(row, o)));
        colors.push(rgb.map(|o| row[o]));
    }
    Ok((points, colors))
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_unstable_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

// opencv.py's _voxel_count hash: per-axis voxel indices re-zeroed, then row-major over their ranges
fn voxel_hashes(points: &[[f64; 3]], lo: [f64; 3], voxel: f64) -> Vec<i64> {
    let cells = points
        .iter()
        .map(|p| [0, 1, 2].map(|d| ((p[d] - lo[d]) / voxel) as i64))
        .collect::<Vec<_>>();
    let min = [0, 1, 2].map(|d| cells.iter().map(|c| c[d]).min().unwrap_or(0));
    let max = [0, 1, 2].map(|d| cells.iter().map(|c| c[d] - min[d]).max().unwrap_or(0));
    let (my, mz) = (max[1] + 1, max[2] + 1);
    cells
        .iter()
        .map(|c| (c[0] - min[0]) * (my * mz) + (c[1] - min[1]) * mz + (c[2] - min[2]))
        .collect()
}

/// geometry_aware_downsample: a voxel size bisected for about `target` occupied voxels, one random point per voxel,
/// then trimmed or padded at random to exactly `target`.
fn geometry_aware_downsample(points: &[[f64; 3]], target: usize, rng: &mut StdRng) -> Vec<usize> {
    let n = points.len();
    if n <= target {
        return (0..n).collect();
    }
    let lo = [0, 1, 2].map(|d| points.iter().map(|p| p[d]).fold(f64::INFINITY, f64::min));
    let hi = [0, 1, 2].map(|d| {
        points
            .iter()
            .map(|p| p[d])
            .fold(f64::NEG_INFINITY, f64::max)
    });
    let diag = (0..3).map(|d| (hi[d] - lo[d]).powi(2)).sum::<f64>().sqrt();
    let (mut a, mut b) = (diag * 1e-6, diag * 0.5);
    let tol = VOXEL_SEARCH_TOL * target as f64;
    let mut best = 0.5 * (a + b);
    for _ in 0..VOXEL_SEARCH_STEPS {
        let mid = 0.5 * (a + b);
        let mut hashes = voxel_hashes(points, lo, mid);
        hashes.sort_unstable();
        hashes.dedup();
        best = mid;
        let count = hashes.len() as f64;
        if (count - target as f64).abs() <= tol {
            break;
        }
        if count > target as f64 {
            a = mid;
        } else {
            b = mid;
        }
    }
    let hashes = voxel_hashes(points, lo, best);
    let mut perm = (0..n).collect::<Vec<_>>();
    perm.shuffle(rng);
    // the first point of each voxel in permuted order, by ascending hash
    let mut first: HashMap<i64, usize> = HashMap::new();
    for &i in &perm {
        first.entry(hashes[i]).or_insert(i);
    }
    let mut keyed = first.into_iter().collect::<Vec<_>>();
    keyed.sort_unstable_by_key(|&(h, _)| h);
    let mut sel = keyed.into_iter().map(|(_, i)| i).collect::<Vec<_>>();
    if sel.len() > target {
        sel = rand::seq::index::sample(rng, sel.len(), target)
            .into_iter()
            .map(|j| sel[j])
            .collect();
    } else if sel.len() < target {
        let mut taken = vec![false; n];
        for &i in &sel {
            taken[i] = true;
        }
        let rest = (0..n).filter(|&i| !taken[i]).collect::<Vec<_>>();
        let need = target - sel.len();
        sel.extend(
            rand::seq::index::sample(rng, rest.len(), need)
                .into_iter()
                .map(|j| rest[j]),
        );
    }
    sel
}

fn apply_point(t: &Mat4, p: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|r| t[r][0] * p[0] + t[r][1] * p[1] + t[r][2] * p[2] + t[r][3])
}

// transform_cameras: T @ c2w with the rotation block's scale divided out (row 0's norm)
fn transform_camera(t: &Mat4, c2w: &Mat4) -> Mat4 {
    let mut m = mul4(t, c2w);
    let scale = (0..3).map(|j| m[0][j] * m[0][j]).sum::<f64>().sqrt();
    for row in m.iter_mut().take(3) {
        for v in row.iter_mut().take(3) {
            *v /= scale;
        }
    }
    m
}

// similarity_from_cameras (focus centring): the mean camera up turned to -y, the cameras' median focus point to the
// origin, the median camera distance to 1
fn similarity_from_cameras(c2ws: &[Mat4]) -> (Mat4, f64) {
    let n = c2ws.len() as f64;
    let mut up = [0.0; 3];
    for c in c2ws {
        for d in 0..3 {
            up[d] -= c[d][1] / n;
        }
    }
    let norm = (up.iter().map(|v| v * v).sum::<f64>()).sqrt();
    let up = up.map(|v| v / norm);
    let target = [0.0, -1.0, 0.0];
    let c = -up[1];
    let cross = [
        up[1] * target[2] - up[2] * target[1],
        up[2] * target[0] - up[0] * target[2],
        up[0] * target[1] - up[1] * target[0],
    ];
    let skew = [
        [0.0, -cross[2], cross[1]],
        [cross[2], 0.0, -cross[0]],
        [-cross[1], cross[0], 0.0],
    ];
    let r_align: [[f64; 3]; 3] = if c > -1.0 {
        std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                let ss = (0..3).map(|k| skew[i][k] * skew[k][j]).sum::<f64>();
                f64::from(u8::from(i == j)) + skew[i][j] + ss / (1.0 + c)
            })
        })
    } else {
        [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    };
    let rot = |v: [f64; 3]| [0, 1, 2].map(|i| (0..3).map(|j| r_align[i][j] * v[j]).sum::<f64>());
    let mut nearest = Vec::with_capacity(c2ws.len());
    let mut ts = Vec::with_capacity(c2ws.len());
    for cam in c2ws {
        let t = rot([cam[0][3], cam[1][3], cam[2][3]]);
        let fwd = rot([cam[0][2], cam[1][2], cam[2][2]]);
        let along = -(0..3).map(|d| fwd[d] * t[d]).sum::<f64>();
        nearest.push([0, 1, 2].map(|d| t[d] + along * fwd[d]));
        ts.push(t);
    }
    let translate = [0, 1, 2].map(|d| -median(nearest.iter().map(|p| p[d]).collect()));
    let dist = median(
        ts.iter()
            .map(|t| {
                (0..3)
                    .map(|d| (t[d] + translate[d]).powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
            .collect(),
    );
    let scale = if dist < 1e-10 { 1.0 } else { 1.0 / dist };
    let mut t1 = [[0.0; 4]; 4];
    for i in 0..3 {
        for j in 0..3 {
            t1[i][j] = r_align[i][j] * scale;
        }
        t1[i][3] = translate[i] * scale;
    }
    t1[3][3] = 1.0;
    (t1, scale)
}

// align_principal_axes: about the median, the widest spread to x and the narrowest to z, right-handed
fn principal_axes(points: &[[f64; 3]]) -> Mat4 {
    let centre = [0, 1, 2].map(|d| median(points.iter().map(|p| p[d]).collect()));
    let mut cov = [[0.0; 3]; 3];
    for p in points {
        let x = [0, 1, 2].map(|d| p[d] - centre[d]);
        for i in 0..3 {
            for j in 0..3 {
                cov[i][j] += x[i] * x[j];
            }
        }
    }
    let (values, vecs) = symmetric_eigen(cov);
    let mut order = [0, 1, 2];
    order.sort_by(|&a, &b| values[b].total_cmp(&values[a]));
    // E's columns are the axes; T = [E^T, -E^T c]
    let mut e: [[f64; 3]; 3] = std::array::from_fn(|r| std::array::from_fn(|c| vecs[r][order[c]]));
    let det = e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1])
        - e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0])
        + e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0]);
    if det < 0.0 {
        for row in &mut e {
            row[0] = -row[0];
        }
    }
    let mut t = [[0.0; 4]; 4];
    for i in 0..3 {
        for j in 0..3 {
            t[i][j] = e[j][i];
        }
        t[i][3] = -(0..3).map(|j| e[j][i] * centre[j]).sum::<f64>();
    }
    t[3][3] = 1.0;
    t
}

#[derive(serde::Deserialize)]
struct CameraEntry {
    extrinsic: [[f64; 4]; 4],
    intrinsic: [[f64; 3]; 3],
}

impl SplatData {
    pub fn load(dir: &Path) -> Result<Self> {
        let cameras: HashMap<String, serde_json::Value> =
            serde_json::from_slice(&std::fs::read(dir.join("cameras.json"))?)
                .map_err(Error::wrap)?;
        let mut names = std::fs::read_dir(dir.join("images"))?
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let name = e.file_name().into_string().ok()?;
                name.strip_suffix(".png").map(str::to_string)
            })
            .collect::<Vec<_>>();
        names.sort();
        if names.is_empty() {
            return Err(invalid(format!("{}: no images", dir.display())));
        }
        let mut views = Vec::with_capacity(names.len());
        let mut c2ws = Vec::with_capacity(names.len());
        for name in names {
            let entry: CameraEntry = serde_json::from_value(
                cameras
                    .get(&name)
                    .ok_or_else(|| invalid(format!("no camera for {name}")))?
                    .clone(),
            )
            .map_err(Error::wrap)?;
            let image = dir.join("images").join(format!("{name}.png"));
            let (width, height) = image::image_dimensions(&image).map_err(Error::wrap)?;
            let existing = |sub: &str| {
                let p = dir.join(sub).join(format!("{name}.png"));
                p.exists().then_some(p)
            };
            c2ws.push(invert4(&entry.extrinsic));
            views.push(View {
                depth: existing("depths"),
                normal: existing("normals"),
                name,
                c2w: [[0.0; 4]; 4],
                k: entry.intrinsic,
                width: width as usize,
                height: height as usize,
                image,
            });
        }

        let (mut points, mut colors) = read_points_ply(&dir.join("points.ply"))?;
        let base = points.len();
        let sky_path = dir.join("sky_points.ply");
        if sky_path.exists() {
            let (p, c) = read_points_ply(&sky_path)?;
            points.extend(p);
            colors.extend(c);
        }
        let is_sky_all = (0..points.len()).map(|i| i >= base).collect::<Vec<_>>();
        let raw = points.iter().map(|p| p.map(f64::from)).collect::<Vec<_>>();
        let mut rng = StdRng::seed_from_u64(DOWNSAMPLE_SEED);
        let keep = geometry_aware_downsample(&raw, DOWNSAMPLE_POINTS, &mut rng);

        let (t1, rescale) = similarity_from_cameras(&c2ws);
        let after_t1 = keep
            .iter()
            .map(|&i| apply_point(&t1, raw[i]))
            .collect::<Vec<_>>();
        let t2 = principal_axes(&after_t1);
        let mut transform = mul4(&t2, &t1);
        let mut normalized = after_t1
            .iter()
            .map(|&p| apply_point(&t2, p))
            .collect::<Vec<_>>();
        let z_median = median(normalized.iter().map(|p| p[2]).collect());
        let z_mean = normalized.iter().map(|p| p[2]).sum::<f64>() / normalized.len().max(1) as f64;
        if z_median > z_mean {
            let flip = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];
            transform = mul4(&flip, &transform);
            for p in &mut normalized {
                p[1] = -p[1];
                p[2] = -p[2];
            }
        }
        for (v, c2w) in views.iter_mut().zip(&c2ws) {
            v.c2w = transform_camera(&transform, c2w);
        }
        let centres = views
            .iter()
            .map(|v| [v.c2w[0][3], v.c2w[1][3], v.c2w[2][3]])
            .collect::<Vec<_>>();
        let mean =
            [0, 1, 2].map(|d| centres.iter().map(|c| c[d]).sum::<f64>() / centres.len() as f64);
        let scene_scale = centres
            .iter()
            .map(|c| (0..3).map(|d| (c[d] - mean[d]).powi(2)).sum::<f64>().sqrt())
            .fold(0.0, f64::max);

        let train = (0..views.len())
            .filter(|&i| i % TEST_EVERY != 0 || views[i].is_pano())
            .collect();
        let val = (0..views.len())
            .filter(|&i| i % TEST_EVERY == 0 && !views[i].is_pano())
            .collect();
        Ok(Self {
            views,
            train,
            val,
            points: normalized.iter().map(|p| p.map(|v| v as f32)).collect(),
            colors: keep.iter().map(|&i| colors[i]).collect(),
            is_sky: keep.iter().map(|&i| is_sky_all[i]).collect(),
            rescale,
            scene_scale,
            transform,
        })
    }
}
