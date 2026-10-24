//! The transformer's camera inputs as the reference prepares them from world-to-camera matrices: a per-pixel ray
//! embedding for the ControlNet and a quaternion plus translation per frame for the camera tokens.

use crate::geometry::{invert4, mul4, quantile, rotation_to_quaternion};

type Mat4 = [[f64; 4]; 4];

// Translations are scaled so the first camera sits this far from the cameras' centre
const CAMERA_SCALE: f64 = 2.0;
// Cameras further than this many times the 97% quantile of distances to the median are left out of the centre
const OUTLIER_FACTOR: f64 = 10.0;
const OUTLIER_QUANTILE: f32 = 0.97;
const OUTLIER_CAP: f64 = 1e6;
const NEAR_ZERO: f64 = 1e-5;

fn translation(m: &Mat4) -> [f64; 3] {
    [m[0][3], m[1][3], m[2][3]]
}

// torch.median: the lower middle value for an even count
fn lower_median(mut values: Vec<f64>) -> f64 {
    values.sort_unstable_by(f64::total_cmp);
    values[(values.len() - 1) / 2]
}

// Re-bases `w2c` on the first of `anchors` (w2c @ inv(anchor_0)), centres the cameras on the inlier mean of the anchors'
// centres and scales so the first anchor sits CAMERA_SCALE from it: the reference's camera_center_normalization (for
// anchors = all frames) and unified_camera_normalization (anchors = the target frames)
fn normalize(w2c: &[Mat4], n_anchors: usize) -> Vec<Mat4> {
    let base = invert4(&w2c[0]);
    let mut c2w = w2c
        .iter()
        .map(|m| invert4(&mul4(m, &base)))
        .collect::<Vec<_>>();
    let anchors = c2w[..n_anchors].iter().map(translation).collect::<Vec<_>>();
    let median = [0, 1, 2].map(|d| lower_median(anchors.iter().map(|t| t[d]).collect()));
    let dist = anchors
        .iter()
        .map(|t| {
            (0..3)
                .map(|d| (t[d] - median[d]).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .collect::<Vec<_>>();
    let limit = (f64::from(quantile(
        &dist.iter().map(|&d| d as f32).collect::<Vec<_>>(),
        OUTLIER_QUANTILE,
    )) * OUTLIER_FACTOR)
        .min(OUTLIER_CAP);
    let inliers = anchors
        .iter()
        .zip(&dist)
        .filter(|(_, d)| **d <= limit)
        .map(|(t, _)| t)
        .collect::<Vec<_>>();
    let center =
        [0, 1, 2].map(|d| inliers.iter().map(|t| t[d]).sum::<f64>() / inliers.len() as f64);
    for m in &mut c2w {
        for d in 0..3 {
            m[d][3] -= center[d];
        }
    }
    let first = translation(&c2w[0]);
    let norm = first.iter().map(|v| v * v).sum::<f64>().sqrt();
    let scale = if norm <= NEAR_ZERO {
        CAMERA_SCALE
    } else {
        CAMERA_SCALE / norm
    };
    c2w.iter()
        .map(|m| {
            let mut w = invert4(m);
            for row in w.iter_mut().take(3) {
                row[3] *= scale;
            }
            w
        })
        .collect()
}

/// Per-pixel ray embedding `[6, frames, height, width]` (f32, channel-major): the ray direction and the moment
/// `origin x direction` of each pixel, after normalizing the cameras. As the reference, pixel x is sampled at
/// `i - 0.5` but y at `j + 0.5`.
pub fn ray_embedding(
    w2c: &[Mat4],
    intrinsics: &[[[f64; 3]; 3]],
    height: usize,
    width: usize,
) -> Vec<f32> {
    let frames = w2c.len();
    let w2c = normalize(w2c, frames);
    let plane = height * width;
    let mut out = vec![0f32; 6 * frames * plane];
    for (f, (m, k)) in w2c.iter().zip(intrinsics).enumerate() {
        let c2w = invert4(m);
        let (fx, fy, cx, cy, skew) = (k[0][0], k[1][1], k[0][2], k[1][2], k[0][1]);
        let o = translation(&c2w);
        for y in 0..height {
            for x in 0..width {
                // K^-1 [u, v, 1] for an upper-triangular K
                let (u, v) = (x as f64 - 0.5, y as f64 + 0.5);
                let cam_y = (v - cy) / fy;
                let cam = [(u - cx - skew * cam_y) / fx, cam_y, 1.0];
                let mut d = [0, 1, 2].map(|r| (0..3).map(|c| c2w[r][c] * cam[c]).sum::<f64>());
                let len = d.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
                d = d.map(|v| v / len);
                let moment = [
                    o[1] * d[2] - o[2] * d[1],
                    o[2] * d[0] - o[0] * d[2],
                    o[0] * d[1] - o[1] * d[0],
                ];
                let p = y * width + x;
                for (c, v) in d.iter().chain(&moment).enumerate() {
                    out[(c * frames + f) * plane + p] = *v as f32;
                }
            }
        }
    }
    out
}

/// `[w, x, y, z, tx, ty, tz]` per target frame and per memory frame: both sets normalized together on the targets,
/// the rotation of each world-to-camera matrix as a PyTorch3D quaternion (real part first, non-negative).
pub fn camera_qt(w2c: &[Mat4], w2c_ref: &[Mat4]) -> (Vec<[f32; 7]>, Vec<[f32; 7]>) {
    let all = w2c.iter().chain(w2c_ref).copied().collect::<Vec<_>>();
    let normalized = normalize(&all, w2c.len());
    let qt = |m: &Mat4| -> [f32; 7] {
        let r = [0, 1, 2].map(|i| [0, 1, 2].map(|j| m[i][j] as f32));
        let [x, y, z, w] = rotation_to_quaternion(r);
        [w, x, y, z, m[0][3] as f32, m[1][3] as f32, m[2][3] as f32]
    };
    let (targets, refs) = normalized.split_at(w2c.len());
    (
        targets.iter().map(qt).collect(),
        refs.iter().map(qt).collect(),
    )
}

#[cfg(test)]
mod tests {
    use inference_tensor::{DType, Device, Tensor};

    fn mats4(t: &Tensor) -> Vec<[[f64; 4]; 4]> {
        t.to_dtype(DType::F64)
            .unwrap()
            .to_vec3::<f64>()
            .unwrap()
            .iter()
            .map(|m| std::array::from_fn(|i| std::array::from_fn(|j| m[i][j])))
            .collect()
    }

    // WORLDSTEREO_CAMERAS_REF: the reference functions' outputs on a synthetic trajectory
    #[test]
    #[ignore = "needs a reference dump"]
    fn camera_inputs_match_the_reference() {
        let Some(path) = std::env::var_os("WORLDSTEREO_CAMERAS_REF") else {
            return;
        };
        let refs = inference_tensor::safetensors::load(path, &Device::Cpu).unwrap();
        let (w2c, w2c_ref) = (mats4(&refs["w2c"]), mats4(&refs["w2c_ref"]));
        let k = refs["intrinsics"]
            .to_dtype(DType::F64)
            .unwrap()
            .to_vec3::<f64>()
            .unwrap();
        let k = k
            .iter()
            .map(|m| std::array::from_fn(|i| std::array::from_fn(|j| m[i][j])))
            .collect::<Vec<_>>();
        let (_, f, h, w) = refs["embedding"].dims4().unwrap();
        let emb = super::ray_embedding(&w2c, &k, h, w);
        let want = refs["embedding"]
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let max = emb
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!("ray embedding ({f} frames): max {max:.3e}");
        assert!(max < 1e-4);
        let (qt, qt_ref) = super::camera_qt(&w2c, &w2c_ref);
        for (ours, name) in [(qt, "qt"), (qt_ref, "qt_ref")] {
            let want = refs[name].to_vec2::<f32>().unwrap();
            let max = ours
                .iter()
                .zip(&want)
                .flat_map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x - y).abs()))
                .fold(0f32, f32::max);
            println!("{name}: max {max:.3e}");
            assert!(max < 1e-4);
        }
    }
}
