//! Optional geometric priors: known camera poses, intrinsics and depth, normalized as the reference normalizes them
//! before the trunk embeds each as tokens.

use inference_tensor::{Device, Result, Tensor};

use crate::geometry::{invert4, mul4, quantile, rotation_to_quaternion};

// Translations are scaled into the unit cube with this margin on each side
const POSE_PADDING: f32 = 0.1;
// Above this many views the translation range is taken between quantiles rather than the extremes
const POSE_QUANTILE_VIEWS: usize = 10;
const POSE_LOW_QUANTILE: f32 = 0.05;
const POSE_HIGH_QUANTILE: f32 = 0.95;
// Depth is scaled between these quantiles of its positive values, given more than `DEPTH_QUANTILE_VALUES` of them
const DEPTH_LOW_QUANTILE: f32 = 0.01;
const DEPTH_HIGH_QUANTILE: f32 = 0.99;
const DEPTH_QUANTILE_VALUES: usize = 100;
const DEPTH_EPS: f32 = 1e-6;

/// Priors for every frame, as a caller has them: camera-to-world poses (OpenCV), pixel intrinsics at each image's
/// original resolution, and z-depth maps of any size.
#[derive(Clone, Debug, Default)]
pub struct PriorInputs {
    pub camera_to_world: Option<Vec<[[f32; 4]; 4]>>,
    pub intrinsics: Option<Vec<[[f32; 3]; 3]>>,
    /// `(width, height, values)` per frame, row-major
    pub depths: Option<Vec<(usize, usize, Vec<f32>)>>,
}

impl PriorInputs {
    pub fn is_empty(&self) -> bool {
        self.camera_to_world.is_none() && self.intrinsics.is_none() && self.depths.is_none()
    }
}

/// The normalized priors the trunk embeds: poses `[frames, 7]`, rays `[frames, 4]` and depth `[frames, 1, H, W]`.
#[derive(Default)]
pub(crate) struct Priors {
    pub poses: Option<Tensor>,
    pub rays: Option<Tensor>,
    pub depths: Option<Tensor>,
}

/// Normalizes `inputs` for frames resized from `original` (the first image's size, as the reference scales every
/// frame's intrinsics by it) to `height x width`.
pub(crate) fn normalize(
    inputs: &PriorInputs,
    original: (usize, usize),
    height: usize,
    width: usize,
    device: &Device,
) -> Result<Priors> {
    let poses = match &inputs.camera_to_world {
        Some(c2w) => Some(Tensor::from_vec(pose_vectors(c2w), (c2w.len(), 7), device)?),
        None => None,
    };
    let rays = match &inputs.intrinsics {
        Some(ks) => {
            let (sx, sy) = (
                width as f32 / original.0 as f32,
                height as f32 / original.1 as f32,
            );
            let rays = ks
                .iter()
                .flat_map(|k| {
                    [
                        k[0][0] * sx / width as f32,
                        k[1][1] * sy / height as f32,
                        k[0][2] * sx / width as f32,
                        k[1][2] * sy / height as f32,
                    ]
                })
                .collect::<Vec<_>>();
            Some(Tensor::from_vec(rays, (ks.len(), 4), device)?)
        }
        None => None,
    };
    let depths = match &inputs.depths {
        Some(maps) => {
            let frames = maps
                .iter()
                .map(|(w, h, d)| normalize_depth(&resize_linear(d, *w, *h, width, height)))
                .collect::<Vec<_>>();
            Some(Tensor::from_vec(
                frames.concat(),
                (maps.len(), 1, height, width),
                device,
            )?)
        }
        None => None,
    };
    Ok(Priors {
        poses,
        rays,
        depths,
    })
}

// `[tx, ty, tz, qx, qy, qz, qw]` per view: poses relative to the first, translations into the unit cube
fn pose_vectors(c2w: &[[[f32; 4]; 4]]) -> Vec<f32> {
    let to64 = |m: &[[f32; 4]; 4]| m.map(|r| r.map(f64::from));
    let first_inverse = invert4(&to64(&c2w[0]));
    let relative = c2w
        .iter()
        .map(|m| mul4(&first_inverse, &to64(m)).map(|r| r.map(|v| v as f32)))
        .collect::<Vec<_>>();
    let positions = relative
        .iter()
        .map(|m| [m[0][3], m[1][3], m[2][3]])
        .collect::<Vec<_>>();
    let (lo, hi): ([f32; 3], [f32; 3]) = if positions.len() > POSE_QUANTILE_VIEWS {
        let axis =
            |d: usize, q: f32| quantile(&positions.iter().map(|p| p[d]).collect::<Vec<_>>(), q);
        (
            [0, 1, 2].map(|d| axis(d, POSE_LOW_QUANTILE)),
            [0, 1, 2].map(|d| axis(d, POSE_HIGH_QUANTILE)),
        )
    } else {
        (
            [0, 1, 2].map(|d| positions.iter().map(|p| p[d]).fold(f32::INFINITY, f32::min)),
            [0, 1, 2].map(|d| {
                positions
                    .iter()
                    .map(|p| p[d])
                    .fold(f32::NEG_INFINITY, f32::max)
            }),
        )
    };
    let range = [0, 1, 2].map(|d| (hi[d] - lo[d]).max(1e-6f32.max(hi[d].abs() * 1e-6)));
    let scale = range
        .iter()
        .copied()
        .fold(f32::MIN, f32::max)
        .clamp(1e-6, 1e6)
        / (1.0 - 2.0 * POSE_PADDING);
    let center = [0, 1, 2].map(|d| (lo[d] + hi[d]) / 2.0);
    relative
        .iter()
        .flat_map(|m| {
            let t = [0, 1, 2].map(|d| ((m[d][3] - center[d]) / scale + 0.5).clamp(0.0, 1.0));
            let q = rotation_to_quaternion([
                [m[0][0], m[0][1], m[0][2]],
                [m[1][0], m[1][1], m[1][2]],
                [m[2][0], m[2][1], m[2][2]],
            ]);
            [t[0], t[1], t[2], q[0], q[1], q[2], q[3]]
        })
        .collect()
}

// Bilinear resize with half-pixel centres and no antialiasing (OpenCV's INTER_LINEAR on float maps)
fn resize_linear(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    if (sw, sh) == (dw, dh) {
        return src.to_vec();
    }
    let taps = |dst: usize, n_src: usize, n_dst: usize| -> (usize, usize, f32) {
        let s = ((dst as f32 + 0.5) * n_src as f32 / n_dst as f32 - 0.5).max(0.0);
        let i0 = (s.floor() as usize).min(n_src - 1);
        let i1 = (i0 + 1).min(n_src - 1);
        (i0, i1, s - i0 as f32)
    };
    let mut out = vec![0f32; dw * dh];
    for y in 0..dh {
        let (y0, y1, fy) = taps(y, sh, dh);
        for x in 0..dw {
            let (x0, x1, fx) = taps(x, sw, dw);
            let top = src[y0 * sw + x0] * (1.0 - fx) + src[y0 * sw + x1] * fx;
            let bottom = src[y1 * sw + x0] * (1.0 - fx) + src[y1 * sw + x1] * fx;
            out[y * dw + x] = top * (1.0 - fy) + bottom * fy;
        }
    }
    out
}

// One depth map into [0, 1] between its positive values' 1% and 99% quantiles
fn normalize_depth(depth: &[f32]) -> Vec<f32> {
    let clean = depth
        .iter()
        .map(|&d| {
            if d.is_finite() {
                d
            } else if d > 0.0 {
                1e6
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    let positive = clean
        .iter()
        .copied()
        .filter(|&d| d > 0.0)
        .collect::<Vec<_>>();
    let values = if positive.is_empty() {
        clean.clone()
    } else {
        positive
    };
    let (lo, mut hi) = if values.len() > DEPTH_QUANTILE_VALUES {
        (
            quantile(&values, DEPTH_LOW_QUANTILE),
            quantile(&values, DEPTH_HIGH_QUANTILE),
        )
    } else {
        (
            values.iter().copied().fold(f32::INFINITY, f32::min),
            values.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        )
    };
    if hi == lo {
        hi = lo + 1.0;
    }
    let eps = DEPTH_EPS.max((hi - lo).abs() * DEPTH_EPS);
    clean
        .iter()
        .map(|d| ((d - lo) / (hi - lo + eps)).clamp(0.0, 1.0))
        .collect()
}

/// Reads priors for frames named by `stems`: a camera JSON in the reference's (and our output's) format, `camera_id`
/// an image stem or an index, and a folder of depth maps named by stem (`.npy` float32, `.png` 16-bit in millimetres
/// when above 255, or `.exr`). A list that does not cover every frame is an error.
pub fn load_prior_files(
    cameras: Option<&std::path::Path>,
    depth_dir: Option<&std::path::Path>,
    stems: &[String],
) -> Result<PriorInputs> {
    let mut inputs = PriorInputs::default();
    if let Some(path) = cameras {
        let text = std::fs::read_to_string(path)?;
        let json: serde_json::Value =
            serde_json::from_str(&text).map_err(inference_tensor::Error::wrap)?;
        let index = |id: &serde_json::Value| -> Option<usize> {
            let id = match id {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            stems
                .iter()
                .position(|s| *s == id)
                .or_else(|| id.parse::<usize>().ok().filter(|&i| i < stems.len()))
        };
        let read = |key: &str, n: usize| -> Result<Option<Vec<Vec<f32>>>> {
            let Some(list) = json
                .get(key)
                .and_then(|v| v.as_array())
                .filter(|l| !l.is_empty())
            else {
                return Ok(None);
            };
            let mut out = vec![None; stems.len()];
            for entry in list {
                let Some(i) = entry.get("camera_id").and_then(index) else {
                    inference_tensor::bail!(
                        "{}: a {key} entry names no input image",
                        path.display()
                    );
                };
                let rows = entry
                    .get("matrix")
                    .and_then(|m| serde_json::from_value::<Vec<Vec<f32>>>(m.clone()).ok())
                    .filter(|m| m.len() == n && m.iter().all(|r| r.len() == n))
                    .ok_or_else(|| {
                        inference_tensor::Error::msg(format!(
                            "{}: a {key} matrix is not {n} x {n}",
                            path.display()
                        ))
                    })?;
                out[i] = Some(rows.concat());
            }
            let found = out.iter().filter(|m| m.is_some()).count();
            if found != stems.len() {
                inference_tensor::bail!(
                    "{}: {key} cover {found} of {} images; give one per image or none",
                    path.display(),
                    stems.len()
                );
            }
            Ok(Some(out.into_iter().flatten().collect()))
        };
        inputs.camera_to_world = read("extrinsics", 4)?.map(|ms| {
            ms.iter()
                .map(|m| std::array::from_fn(|r| std::array::from_fn(|c| m[r * 4 + c])))
                .collect()
        });
        inputs.intrinsics = read("intrinsics", 3)?.map(|ms| {
            ms.iter()
                .map(|m| std::array::from_fn(|r| std::array::from_fn(|c| m[r * 3 + c])))
                .collect()
        });
    }
    if let Some(dir) = depth_dir {
        let maps = stems
            .iter()
            .map(|stem| {
                let found = ["npy", "png", "exr"]
                    .iter()
                    .map(|ext| dir.join(format!("{stem}.{ext}")))
                    .find(|p| p.is_file())
                    .ok_or_else(|| {
                        inference_tensor::Error::msg(format!(
                            "{}: no depth map for {stem}",
                            dir.display()
                        ))
                    })?;
                read_depth(&found)
            })
            .collect::<Result<Vec<_>>>()?;
        inputs.depths = Some(maps);
    }
    Ok(inputs)
}

// A depth map as `(width, height, values)`: a 2D float32 .npy, a 16-bit PNG (millimetres when above 255), or an EXR
fn read_depth(path: &std::path::Path) -> Result<(usize, usize, Vec<f32>)> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if ext == "npy" {
        let bytes = std::fs::read(path)?;
        let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        let header = std::str::from_utf8(&bytes[10..10 + header_len])
            .map_err(inference_tensor::Error::wrap)?;
        if !header.contains("'<f4'") || header.contains("'fortran_order': True") {
            inference_tensor::bail!(
                "{}: depth .npy must be little-endian float32, C order",
                path.display()
            );
        }
        let shape = header
            .split("'shape': (")
            .nth(1)
            .and_then(|s| s.split(')').next())
            .map(|s| {
                s.split(',')
                    .filter_map(|v| v.trim().parse::<usize>().ok())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let (h, w) = match shape.as_slice() {
            [h, w] | [h, w, _] => (*h, *w),
            _ => inference_tensor::bail!("{}: depth .npy must be 2D", path.display()),
        };
        let channels = shape.get(2).copied().unwrap_or(1);
        let values = bytes[10 + header_len..]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .step_by(channels)
            .take(h * w)
            .collect::<Vec<_>>();
        return Ok((w, h, values));
    }
    let image = crate::read_image(path)
        .map_err(|e| inference_tensor::Error::msg(format!("{}: {e}", path.display())))?;
    let (w, h) = (image.width() as usize, image.height() as usize);
    let mut values = if ext == "png" {
        image
            .to_luma16()
            .into_raw()
            .iter()
            .map(|&v| f32::from(v))
            .collect::<Vec<_>>()
    } else {
        image.to_luma32f().into_raw()
    };
    if ext == "png" && values.iter().any(|&v| v > 255.0) {
        values.iter_mut().for_each(|v| *v /= 1000.0);
    }
    values
        .iter_mut()
        .filter(|v| !v.is_finite())
        .for_each(|v| *v = 0.0);
    Ok((w, h, values))
}
