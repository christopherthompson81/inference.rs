//! WorldStereo's memory bank (`PanoramaMemoryBank`): panorama crops at first, then frames of each generated clip; for
//! every frame of a new trajectory, the stored frame whose frustum overlaps it most, at most eight distinct ones per
//! clip.

use super::panorama::Rgb8;
use super::trajectory::Mat4;
use crate::geometry::{invert4, rotation_to_quaternion};

const NEAR: f64 = 0.1;
// the far plane is this many median ground distances, at least FAR_MIN
const FAR_MEDIANS: f64 = 8.0;
const FAR_MIN: f64 = 0.15;
const MAX_REFERENCE: usize = 8;
// frames of each generated clip the bank keeps (sample_align_nframe(21, 8))
const KEPT_FRAMES: [usize; 8] = [3, 6, 9, 12, 14, 16, 18, 20];
// retrieval targets whose frame the aerial clip of the same view is forced into, for the regular turns
const SHORTCUT_TARGETS: [usize; 8] = [1, 2, 3, 5, 7, 9, 13, 17];
const SHORTCUT_BONUS: f64 = 1.0;
// the selector's mix of camera and image features, and its preference for sharp, rich images
const CAMERA_WEIGHT: f64 = 0.3;
const IMAGE_WEIGHT: f64 = 0.7;
const QUALITY_BIAS: f64 = 0.1;

pub struct MemoryFrame {
    pub w2c: Mat4,
    pub intrinsics: [[f64; 3]; 3],
    pub image: Rgb8,
    /// `view/traj/index` as the reference names it (the panorama crops are `render_results/pano_bank/NNNN`)
    pub name: String,
}

/// What retrieval hands a clip: per target frame 1..20 the chosen memory frame, and the distinct ones the transformer
/// attends to with the target frame each first appears at.
pub struct Retrieval {
    /// bank index per target frame 1..F-1
    pub per_frame: Vec<usize>,
    /// bank indices of the distinct memory frames, in first-appearance order
    pub frames: Vec<usize>,
    /// the target index minus one where each first appears (the transformer's ref_index)
    pub index: Vec<usize>,
}

pub struct MemoryBank {
    pub frames: Vec<MemoryFrame>,
    width: usize,
    height: usize,
    far: f64,
}

fn c2w_of(w2c: &Mat4) -> Mat4 {
    invert4(w2c)
}

// the eight corners of a camera's frustum between NEAR and far, in world space
fn frustum(w2c: &Mat4, k: &[[f64; 3]; 3], (w, h): (usize, usize), far: f64) -> [[f64; 3]; 8] {
    let c2w = c2w_of(w2c);
    let mut out = [[0.0; 3]; 8];
    let corners = [
        (0.0, 0.0),
        (w as f64, 0.0),
        (w as f64, h as f64),
        (0.0, h as f64),
    ];
    for (i, &(u, v)) in corners.iter().enumerate() {
        // K^-1 [u, v, 1] for an upper-triangular K, normalized
        let y = (v - k[1][2]) / k[1][1];
        let x = (u - k[0][2] - k[0][1] * y) / k[0][0];
        let n = (x * x + y * y + 1.0).sqrt();
        let ray = [x / n, y / n, 1.0 / n];
        for (j, depth) in [NEAR, far].into_iter().enumerate() {
            out[j * 4 + i] = [0, 1, 2]
                .map(|r| (0..3).map(|c| c2w[r][c] * ray[c] * depth).sum::<f64>() + c2w[r][3]);
        }
    }
    out
}

// IoU of the frustums' axis-aligned boxes, and the angle (degrees) between the optical axes
fn overlap(a: &[[f64; 3]; 8], b: &[[f64; 3]; 8], wa: &Mat4, wb: &Mat4) -> (f64, f64) {
    let bounds = |c: &[[f64; 3]; 8]| {
        let lo = [0, 1, 2].map(|d| c.iter().map(|p| p[d]).fold(f64::INFINITY, f64::min));
        let hi = [0, 1, 2].map(|d| c.iter().map(|p| p[d]).fold(f64::NEG_INFINITY, f64::max));
        (lo, hi)
    };
    let ((a_min, a_max), (b_min, b_max)) = (bounds(a), bounds(b));
    let (ilo, ihi) = (
        [0, 1, 2].map(|d| a_min[d].max(b_min[d])),
        [0, 1, 2].map(|d| a_max[d].min(b_max[d])),
    );
    let valid = (0..3).all(|d| ihi[d] > ilo[d]);
    let vol =
        |lo: [f64; 3], hi: [f64; 3]| (0..3).map(|d| (hi[d] - lo[d]).max(0.0)).product::<f64>();
    let inter = vol(ilo, ihi);
    let iou = if valid {
        inter / (vol(a_min, a_max) + vol(b_min, b_max) - inter + 1e-8)
    } else {
        0.0
    };
    let (ca, cb) = (c2w_of(wa), c2w_of(wb));
    let cos = (0..3)
        .map(|r| ca[r][2] * cb[r][2])
        .sum::<f64>()
        .clamp(-1.0, 1.0);
    (iou, cos.acos().to_degrees())
}

// the reference's image quality score: Laplacian variance, histogram entropy and contrast of a grey image made with
// OpenCV's BGR weights from RGB data (red and blue swapped, as it calls COLOR_BGR2GRAY on RGB arrays)
fn quality(img: &Rgb8) -> f64 {
    let (w, h) = (img.width, img.height);
    let gray = img
        .pixels
        .as_chunks::<3>()
        .0
        .iter()
        .map(|p| {
            (0.114 * f64::from(p[0]) + 0.587 * f64::from(p[1]) + 0.299 * f64::from(p[2]))
                .round()
                .clamp(0.0, 255.0)
        })
        .collect::<Vec<_>>();
    // BORDER_REFLECT_101
    let at = |x: isize, y: isize| {
        let r = |i: isize, n: usize| if i < 0 { -i } else if i >= n as isize { 2 * n as isize - 2 - i } else { i } as usize;
        gray[r(y, h) * w + r(x, w)]
    };
    let lap = (0..h * w)
        .map(|p| {
            let (x, y) = ((p % w) as isize, (p / w) as isize);
            at(x - 1, y) + at(x + 1, y) + at(x, y - 1) + at(x, y + 1) - 4.0 * at(x, y)
        })
        .collect::<Vec<_>>();
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let var = |v: &[f64]| {
        let m = mean(v);
        v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64
    };
    let mut hist = [0f64; 256];
    for &g in &gray {
        hist[g as usize] += 1.0;
    }
    let total = hist.iter().sum::<f64>();
    let entropy = -hist
        .iter()
        .map(|&c| c / total + 1e-10)
        .map(|p| p * p.log2())
        .sum::<f64>();
    0.4 * var(&lap).ln_1p() + 0.3 * entropy + 0.3 * (var(&gray).sqrt() / 128.0)
}

// z-score each column
fn standardize(rows: &mut [Vec<f64>]) {
    let n = rows.len() as f64;
    for d in 0..rows.first().map_or(0, Vec::len) {
        let m = rows.iter().map(|r| r[d]).sum::<f64>() / n;
        let s = (rows.iter().map(|r| (r[d] - m).powi(2)).sum::<f64>() / n).sqrt() + 1e-8;
        rows.iter_mut().for_each(|r| r[d] = (r[d] - m) / s);
    }
}

/// CameraSelector.select: quality-aware farthest point sampling over camera (position and orientation) and image
/// features, `k` of the candidates.
pub fn select(
    w2cs: &[Mat4],
    images: &[&Rgb8],
    image_features: &[Vec<f32>],
    k: usize,
) -> Vec<usize> {
    let mut camera = w2cs
        .iter()
        .map(|m| {
            let pos = invert4(m);
            let r = [0, 1, 2].map(|i| [0, 1, 2].map(|j| m[i][j] as f32));
            // SciPy's as_quat is (x, y, z, w), flipped to w >= 0; rotation_to_quaternion already gives that
            let q = rotation_to_quaternion(r);
            vec![
                pos[0][3],
                pos[1][3],
                pos[2][3],
                f64::from(q[0]),
                f64::from(q[1]),
                f64::from(q[2]),
                f64::from(q[3]),
            ]
        })
        .collect::<Vec<_>>();
    let mut image = image_features
        .iter()
        .map(|f| f.iter().map(|&v| f64::from(v)).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    standardize(&mut camera);
    standardize(&mut image);
    let features = camera
        .iter()
        .zip(&image)
        .map(|(c, i)| {
            c.iter()
                .map(|v| v * CAMERA_WEIGHT)
                .chain(i.iter().map(|v| v * IMAGE_WEIGHT))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let scores = images.iter().map(|img| quality(img)).collect::<Vec<_>>();
    let (lo, hi) = (
        scores.iter().copied().fold(f64::INFINITY, f64::min),
        scores.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    );
    let q = scores
        .iter()
        .map(|s| (s - lo) / (hi - lo + 1e-8))
        .collect::<Vec<_>>();
    let dist = |a: &[f64], b: &[f64]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    };
    let dims = features[0].len();
    let centroid = (0..dims)
        .map(|d| features.iter().map(|f| f[d]).sum::<f64>() / features.len() as f64)
        .collect::<Vec<_>>();
    let argmax = |v: &[f64]| {
        v.iter()
            .enumerate()
            .fold(
                (0, f64::NEG_INFINITY),
                |b, (i, &x)| if x > b.1 { (i, x) } else { b },
            )
            .0
    };
    let start = features
        .iter()
        .zip(&q)
        .map(|(f, qn)| -dist(f, &centroid) + QUALITY_BIAS * qn)
        .collect::<Vec<_>>();
    let mut selected = vec![argmax(&start)];
    let mut nearest = vec![f64::INFINITY; features.len()];
    for _ in 1..k {
        let current = *selected.last().expect("one selected");
        for (i, f) in features.iter().enumerate() {
            nearest[i] = nearest[i].min(dist(f, &features[current]));
        }
        let adjusted = (0..features.len())
            .map(|i| {
                if selected.contains(&i) {
                    f64::NEG_INFINITY
                } else {
                    nearest[i] * (1.0 + QUALITY_BIAS * (q[i] - 0.5))
                }
            })
            .collect::<Vec<_>>();
        selected.push(argmax(&adjusted));
    }
    selected
}

impl MemoryBank {
    /// The bank from the panorama crops, frames `width x height`, the far plane from the scene's median ground distance.
    pub fn new(
        frames: Vec<MemoryFrame>,
        (width, height): (usize, usize),
        median_ground_distance: f64,
    ) -> Self {
        Self {
            frames,
            width,
            height,
            far: (median_ground_distance * FAR_MEDIANS).max(FAR_MIN),
        }
    }

    // the bank frame overlapping target `w2c` most (angle-penalized), `bonus` added to one candidate
    fn best(
        &self,
        candidates: &[usize],
        w2c: &Mat4,
        k: &[[f64; 3]; 3],
        bonus: Option<usize>,
    ) -> usize {
        let size = (self.width, self.height);
        let target = frustum(w2c, k, size, self.far);
        let mut best = (candidates[0], f64::NEG_INFINITY);
        for (slot, &c) in candidates.iter().enumerate() {
            let f = &self.frames[c];
            let (iou, angle) = overlap(
                &target,
                &frustum(&f.w2c, &f.intrinsics, size, self.far),
                w2c,
                &f.w2c,
            );
            let mut score = iou * (((90.0 - angle) / 180.0) * 5.0).exp().clamp(0.0, 1.0);
            if bonus == Some(slot) {
                score += SHORTCUT_BONUS;
            }
            if score > best.1 {
                best = (c, score);
            }
        }
        best.0
    }

    /// Memory frames for a trajectory `view/traj` with targets `w2cs` (the start first) and `ks`; `features` gives image
    /// features for the selector when more than eight distinct frames come back.
    pub fn retrieve(
        &self,
        view: &str,
        traj: &str,
        w2cs: &[Mat4],
        ks: &[[[f64; 3]; 3]],
        features: &mut dyn FnMut(&[&Rgb8]) -> Vec<Vec<f32>>,
    ) -> Retrieval {
        let all = (0..self.frames.len()).collect::<Vec<_>>();
        // the regular turns lean on the aerial clip of the same view (generated first)
        let shortcut = if view.starts_with("view") && (traj == "traj0" || traj == "traj1") {
            let from = self
                .frames
                .iter()
                .enumerate()
                .filter(|(_, f)| f.name.starts_with(&format!("{view}/traj2")))
                .map(|(i, _)| i)
                .collect::<Vec<_>>();
            let from = if traj == "traj0" {
                from
            } else {
                from[..from.len() / 2].to_vec()
            };
            from.into_iter().skip(3).step_by(2).collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let n = w2cs.len();
        let pick = |i: usize| -> Option<usize> {
            (!shortcut.is_empty() && SHORTCUT_TARGETS.contains(&i))
                .then(|| shortcut[i * shortcut.len() / n])
        };
        let mut per_frame = (1..n)
            .map(|i| self.best(&all, &w2cs[i], &ks[i], pick(i)))
            .collect::<Vec<_>>();
        let distinct = |per_frame: &[usize]| {
            let mut frames: Vec<usize> = Vec::new();
            let mut index = Vec::new();
            for (i, &f) in per_frame.iter().enumerate() {
                if !frames.contains(&f) {
                    frames.push(f);
                    index.push(i);
                }
            }
            (frames, index)
        };
        let (mut frames, mut index) = distinct(&per_frame);
        if frames.len() > MAX_REFERENCE {
            let images = per_frame
                .iter()
                .map(|&f| &self.frames[f].image)
                .collect::<Vec<_>>();
            let w2c = per_frame
                .iter()
                .map(|&f| self.frames[f].w2c)
                .collect::<Vec<_>>();
            let chosen = select(&w2c, &images, &features(&images), MAX_REFERENCE);
            let pool = chosen.iter().map(|&i| per_frame[i]).collect::<Vec<_>>();
            per_frame = (1..n)
                .map(|i| self.best(&pool, &w2cs[i], &ks[i], None))
                .collect();
            (frames, index) = distinct(&per_frame);
        }
        Retrieval {
            per_frame,
            frames,
            index,
        }
    }

    /// Adds the kept frames of a generated clip.
    pub fn update(
        &mut self,
        view: &str,
        traj: &str,
        frames: &[Rgb8],
        w2cs: &[Mat4],
        ks: &[[[f64; 3]; 3]],
    ) {
        for &i in KEPT_FRAMES.iter().filter(|&&i| i < frames.len()) {
            self.frames.push(MemoryFrame {
                w2c: w2cs[i],
                intrinsics: ks[i],
                image: Rgb8 {
                    width: frames[i].width,
                    height: frames[i].height,
                    pixels: frames[i].pixels.clone(),
                },
                name: format!("{view}/{traj}/{i:04}"),
            });
        }
    }
}

#[cfg(test)]
mod parity {
    use inference_tensor::{DType, Device};

    use super::super::panorama::Rgb8;
    use super::super::trajectory::{self, Mat4};
    use super::{MemoryBank, MemoryFrame};
    use crate::geometry::invert4;

    // WORLDSTEREO_SCENE_REF (trajectories) and WORLDSTEREO_RETRIEVAL_REF: the reference's retrieval on the panorama
    // bank cameras for view0's aerial clip, then, with that clip's kept frames added, for view0's right turn
    #[test]
    #[ignore = "needs the scene and retrieval reference dumps"]
    fn retrieval_matches_the_reference() -> inference_tensor::Result<()> {
        let (Some(scene_ref), Some(retrieval_ref)) = (
            std::env::var_os("WORLDSTEREO_SCENE_REF"),
            std::env::var_os("WORLDSTEREO_RETRIEVAL_REF"),
        ) else {
            return Ok(());
        };
        let s = inference_tensor::safetensors::load(scene_ref, &Device::Cpu)?;
        let r = inference_tensor::safetensors::load(retrieval_ref, &Device::Cpu)?;
        let (w, h) = trajectory::view_size();
        let k =
            trajectory::intrinsics(w, h, trajectory::VIEW_FOV_X_DEG, trajectory::VIEW_FOV_Y_DEG);
        let blank = || Rgb8 {
            width: w,
            height: h,
            pixels: vec![0; w * h * 3],
        };
        // the bank's view directions, in the planner's order
        let starts = [[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.5], [-1.0, 0.0, -0.5]];
        let mut dirs: Vec<[f64; 3]> = starts.to_vec();
        for st in starts {
            for i in 1..9 {
                let a = (40.0 * i as f64).to_radians();
                dirs.push([
                    a.cos() * st[0] + a.sin() * st[1],
                    -a.sin() * st[0] + a.cos() * st[1],
                    st[2],
                ]);
            }
        }
        let frames = dirs
            .iter()
            .enumerate()
            .map(|(i, &d)| MemoryFrame {
                w2c: invert4(&trajectory::look_at_c2w(d)),
                intrinsics: k,
                image: blank(),
                name: format!("render_results/pano_bank/{i:04}"),
            })
            .collect();
        let mut bank = MemoryBank::new(frames, (w, h), 1.010_163_1);
        for (step, traj) in ["traj2", "traj0"].iter().enumerate() {
            let c2ws = s[&format!("view0.{traj}.c2w")]
                .to_dtype(DType::F64)?
                .to_vec3::<f64>()?;
            let w2cs = c2ws
                .iter()
                .map(|m| {
                    invert4(&std::array::from_fn::<_, 4, _>(|i| {
                        std::array::from_fn(|j| m[i][j])
                    }))
                })
                .collect::<Vec<Mat4>>();
            let ks = vec![k; w2cs.len()];
            let got = bank.retrieve("view0", traj, &w2cs, &ks, &mut |imgs| {
                vec![vec![0.0]; imgs.len()]
            });
            let want = r[&format!("view0.{traj}.per_frame")]
                .to_dtype(DType::U32)?
                .to_vec1::<u32>()?;
            let want_index = r[&format!("view0.{traj}.ref_index")]
                .to_dtype(DType::U32)?
                .to_vec1::<u32>()?;
            let ours = got.per_frame.iter().map(|&i| i as u32).collect::<Vec<_>>();
            let index = got.index.iter().map(|&i| i as u32).collect::<Vec<_>>();
            println!(
                "view0/{traj}: per frame {ours:?}\n  reference {want:?}; ref_index {index:?} vs {want_index:?}"
            );
            assert_eq!(ours, want);
            assert_eq!(index, want_index);
            if step == 0 {
                bank.update(
                    "view0",
                    traj,
                    &(0..21).map(|_| blank()).collect::<Vec<_>>(),
                    &w2cs,
                    &ks,
                );
            }
        }
        Ok(())
    }
}
