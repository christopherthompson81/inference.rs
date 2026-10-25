//! Per-frame depth alignment, as the reference's memory bank runs it after generation (`PanoramaMemoryBank.alignment`):
//! each memory frame's WorldMirror depth is fitted, in inverse depth by a constrained RANSAC line, to the panorama's
//! point cloud rendered from that frame's planned camera, so every frame lands on the same metric geometry. Failed or
//! anomalous fits borrow the nearest good one in their clip; the aligned depths are unprojected with the planned
//! cameras, outliers removed per clip, and merged into one point cloud for the Gaussian training.

use std::collections::HashMap;

use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;

use super::panorama::Rgb8;
use super::render::render_features;
use super::scene::{Scene, depth_edges};
use super::trajectory::{Mat4, PointGrid};
use crate::geometry::invert4;

// a frame needs the guided depth over this share of its pixels, and this share of valid overlap, to be fitted
const VALID_THRESHOLD: f64 = 0.3;
const MIN_OVERLAP: f64 = 0.05;
// pixels whose depth rank (0..100) differs by more than this between the two depths are left out of the fit
const PERCENTILE_THRESHOLD: f32 = 20.0;
const GUIDE_POINTS_PER_PIXEL: usize = 8;
const MONO_EDGE_RTOL: f64 = 0.05;
const ALIGNED_EDGE_RTOL: f64 = 0.1;
// depths past this many times a frame's median are fitted around and not added to the cloud
const FAR_FACTOR: f32 = 8.0;
const RANSAC_MIN_SAMPLES: usize = 100;
const RANSAC_MAX_TRIALS: usize = 100;
const RANSAC_STOP_PROBABILITY: f64 = 0.995;
const RANSAC_SEED: u64 = 42;
const MIN_COEF: f64 = 1e-4;
const MAX_BIAS: f64 = 10.0;
// fits whose effect at the anchor depths deviates past this percentile of all frames' deviations are outliers
const KB_ANOMALY_PERCENTILE: f64 = 90.0;
const MIN_DEPTH: f32 = 0.05;
const MAX_DEPTH_MARGIN: f64 = 1.1;
const MIN_FINAL_PIXELS: usize = 10;
// the panorama crops are aligned in groups of this many, like a clip's kept frames
const ALIGN_NFRAME: usize = 8;
const SOR_NEIGHBOURS: usize = 10;
const SOR_STD_RATIO: f64 = 2.0;
const EXPORT_POINTS: usize = 2_000_000;
// adaptive_voxel_downsample's size for clouds already within the target
const SMALL_CLOUD_VOXEL: f64 = 0.003;
const VOXEL_SEARCH_TOL: f64 = 0.2;
const VOXEL_SEARCH_STEPS: usize = 10;
const DEPTH_EPS: f32 = 1e-6;
const PANO_PREFIX: &str = "render_results/pano_bank/";

/// One memory frame to align.
pub struct AlignFrame<'a> {
    /// `view/traj/index`, or `render_results/pano_bank/NNNN` for the panorama crops
    pub name: &'a str,
    pub image: &'a Rgb8,
    pub w2c: Mat4,
    pub intrinsics: [[f64; 3]; 3],
    /// WorldMirror's z-depth at the image's size
    pub depth: Vec<f32>,
    /// MoGe-2's validity and camera-space normals at the image's size
    pub mask: Vec<bool>,
    pub normal: Vec<[f32; 3]>,
}

/// Where a frame's final scale and shift came from.
#[derive(Clone, Debug, PartialEq)]
pub enum FitSource {
    Own,
    Borrowed(usize),
    /// every frame of its clip failed or was an outlier: the clip is left out
    Abandoned,
}

/// A frame after alignment: its final inverse-depth fit and its depth on the panorama's metric scale (0 where left
/// out), with the planned camera.
pub struct AlignedFrame {
    pub name: String,
    pub fit: Option<(f64, f64)>,
    pub source: FitSource,
    pub depth: Vec<f32>,
    pub w2c: Mat4,
    pub intrinsics: [[f64; 3]; 3],
}

pub struct Alignment {
    pub frames: Vec<AlignedFrame>,
    /// the aligned frames' points, outliers removed per clip, voxel-merged at the panorama cloud's voxel size
    pub points: Vec<[f32; 3]>,
    pub colors: Vec<[u8; 3]>,
    pub fitted: usize,
    pub borrowed: usize,
    pub abandoned: usize,
}

// a frame's first pass: its mono depth (WorldMirror's where MoGe-2 finds it valid), masks, and fit if one succeeded
struct FirstPass {
    mono: Vec<f32>,
    mono_edge_ok: Vec<bool>,
    far: Vec<bool>,
    fit: Option<(f64, f64)>,
}

/// Aligns `frames` (all the same size) to `scene`'s point cloud.
pub fn align(frames: &[AlignFrame], scene: &Scene) -> Alignment {
    let ground = (0..scene.distance.len())
        .filter(|&p| !scene.sky[p])
        .map(|p| f64::from(scene.distance[p]))
        .collect::<Vec<_>>();
    let min_d = ground.iter().copied().fold(f64::INFINITY, f64::min);
    let max_d = ground.iter().copied().fold(0.0, f64::max);
    let depth_median = lower_median(ground);

    let first = frames
        .par_iter()
        .map(|f| first_pass(f, scene))
        .collect::<Vec<_>>();

    // clips: the panorama crops in groups of ALIGN_NFRAME, then each generated clip in order
    let pano = (0..frames.len())
        .filter(|&i| frames[i].name.starts_with(PANO_PREFIX))
        .collect::<Vec<_>>();
    let mut clips: Vec<Vec<usize>> = Vec::new();
    if !pano.is_empty() {
        let splits = pano.len().div_ceil(ALIGN_NFRAME).max(1);
        let size = pano.len().div_ceil(splits);
        clips.extend(pano.chunks(size).map(<[usize]>::to_vec));
    }
    let mut by_clip: Vec<(String, Vec<usize>)> = Vec::new();
    for (i, f) in frames.iter().enumerate() {
        if f.name.starts_with(PANO_PREFIX) {
            continue;
        }
        let clip = f
            .name
            .rsplit_once('/')
            .map_or(f.name, |(c, _)| c)
            .to_string();
        match by_clip.iter_mut().find(|(c, _)| *c == clip) {
            Some((_, members)) => members.push(i),
            None => by_clip.push((clip, vec![i])),
        }
    }
    clips.extend(by_clip.into_iter().map(|(_, m)| m));

    let deviations = fit_deviations(&first, min_d, max_d, depth_median);
    let fitted_devs = deviations.iter().flatten().copied().collect::<Vec<_>>();
    let inlier_threshold = if fitted_devs.is_empty() {
        f64::INFINITY
    } else {
        percentile(&fitted_devs, KB_ANOMALY_PERCENTILE)
    };
    let mut fits = vec![None; frames.len()];
    let mut sources = vec![FitSource::Abandoned; frames.len()];
    for clip in &clips {
        let inliers = clip
            .iter()
            .enumerate()
            .map(|(local, &i)| (local, i))
            .filter(|&(_, i)| deviations[i].is_some_and(|d| d <= inlier_threshold))
            .collect::<Vec<_>>();
        if inliers.is_empty() {
            continue;
        }
        for (local, &i) in clip.iter().enumerate() {
            if inliers.iter().any(|&(_, j)| j == i) {
                fits[i] = first[i].fit;
                sources[i] = FitSource::Own;
            } else {
                // the nearest inlier in the clip; ties go to the earlier frame, as argmin does
                let &(_, j) = inliers
                    .iter()
                    .min_by_key(|&&(l, _)| l.abs_diff(local))
                    .expect("the clip has an inlier");
                fits[i] = first[j].fit;
                sources[i] = FitSource::Borrowed(j);
            }
        }
    }

    let max_depth = (max_d * MAX_DEPTH_MARGIN) as f32;
    let results = frames
        .par_iter()
        .zip(&first)
        .zip(&fits)
        .map(|((f, fp), fit)| finish(f, fp, *fit, max_depth))
        .collect::<Vec<_>>();

    let mut points = Vec::new();
    let mut colors = Vec::new();
    for clip in &clips {
        let mut clip_points = Vec::new();
        let mut clip_colors = Vec::new();
        for &i in clip {
            if let Some((_, p, c)) = &results[i] {
                clip_points.extend_from_slice(p);
                clip_colors.extend_from_slice(c);
            }
        }
        let keep = outlier_mask(&clip_points);
        points.extend(
            clip_points
                .iter()
                .zip(&keep)
                .filter(|(_, k)| **k)
                .map(|(p, _)| *p),
        );
        colors.extend(
            clip_colors
                .iter()
                .zip(&keep)
                .filter(|(_, k)| **k)
                .map(|(c, _)| *c),
        );
    }
    let voxel = adaptive_voxel_size(&scene.points, EXPORT_POINTS);
    let keep = voxel_first(&points, voxel);
    let (mut points, mut colors): (Vec<_>, Vec<_>) =
        keep.iter().map(|&i| (points[i], colors[i])).unzip();
    if points.len() > EXPORT_POINTS {
        let mut rng = StdRng::seed_from_u64(RANSAC_SEED);
        let picked = rand::seq::index::sample(&mut rng, points.len(), EXPORT_POINTS).into_vec();
        points = picked.iter().map(|&i| points[i]).collect();
        colors = picked.iter().map(|&i| colors[i]).collect();
    }

    let fitted = sources.iter().filter(|s| **s == FitSource::Own).count();
    let borrowed = sources
        .iter()
        .filter(|s| matches!(s, FitSource::Borrowed(_)))
        .count();
    let abandoned = sources
        .iter()
        .filter(|s| **s == FitSource::Abandoned)
        .count();
    let aligned = frames
        .iter()
        .zip(results)
        .zip(fits.iter().zip(sources))
        .map(|((f, r), (fit, source))| AlignedFrame {
            name: f.name.to_string(),
            fit: *fit,
            source,
            depth: r.map_or_else(|| vec![0.0; f.depth.len()], |(d, _, _)| d),
            w2c: f.w2c,
            intrinsics: f.intrinsics,
        })
        .collect();
    Alignment {
        frames: aligned,
        points,
        colors,
        fitted,
        borrowed,
        abandoned,
    }
}

fn first_pass(f: &AlignFrame, scene: &Scene) -> FirstPass {
    let (w, h) = (f.image.width, f.image.height);
    let mono = f
        .depth
        .iter()
        .zip(&f.mask)
        .map(|(&d, &m)| if m { d } else { 0.0 })
        .collect::<Vec<_>>();
    let mono_edge_ok = depth_edges(&mono, w, h, MONO_EDGE_RTOL)
        .into_iter()
        .map(|e| !e)
        .collect::<Vec<_>>();
    let positive = mono
        .iter()
        .filter(|&&d| d > 0.0)
        .map(|&d| f64::from(d))
        .collect::<Vec<_>>();
    let depth_filter = lower_median(positive) as f32 * FAR_FACTOR;
    let far = mono.iter().map(|&d| d > depth_filter).collect::<Vec<_>>();
    let mut pass = FirstPass {
        mono,
        mono_edge_ok,
        far,
        fit: None,
    };

    let guide = render_features(
        &scene.points,
        &scene.normals,
        &f.w2c,
        &f.intrinsics,
        (w, h),
        GUIDE_POINTS_PER_PIXEL,
    );
    let mut guided_ok = guide.depth.iter().map(|&d| d != -1.0).collect::<Vec<_>>();
    let both = (0..w * h)
        .map(|p| guided_ok[p] && f.mask[p])
        .collect::<Vec<_>>();
    let guided_rank = percentile_map(&guide.depth, &both);
    let mono_rank = percentile_map(&pass.mono, &both);
    for p in 0..w * h {
        if (guided_rank[p] - mono_rank[p]).abs() > PERCENTILE_THRESHOLD {
            guided_ok[p] = false;
        }
    }
    if share(&guided_ok) < VALID_THRESHOLD {
        return pass;
    }
    let c2w = invert4(&f.w2c);
    let valid = (0..w * h)
        .map(|p| {
            let n = f.normal[p];
            let world = [0, 1, 2].map(|r| {
                c2w[r][0] * f64::from(n[0])
                    + c2w[r][1] * f64::from(n[1])
                    + c2w[r][2] * f64::from(n[2])
            });
            let g = guide.features[p];
            // within 90 degrees of the guide's normal
            let facing = world[0] * f64::from(g[0])
                + world[1] * f64::from(g[1])
                + world[2] * f64::from(g[2])
                >= 0.0;
            guided_ok[p] && f.mask[p] && pass.mono_edge_ok[p] && facing
        })
        .collect::<Vec<_>>();
    if share(&valid) < MIN_OVERLAP {
        return pass;
    }
    let (x, y): (Vec<f64>, Vec<f64>) = (0..w * h)
        .filter(|&p| valid[p] && pass.mono[p] <= depth_filter)
        .map(|p| {
            (
                1.0 / f64::from(pass.mono[p]),
                1.0 / f64::from(guide.depth[p]),
            )
        })
        .unzip();
    pass.fit = ransac_line(&x, &y);
    pass
}

// each fitted frame's largest relative deviation from the median fit's effect at anchor depths spanning the scene
fn fit_deviations(first: &[FirstPass], min_d: f64, max_d: f64, median: f64) -> Vec<Option<f64>> {
    let far = (8.0 * median).min(max_d);
    let mut anchors = vec![
        min_d,
        min_d + (median - min_d) * 0.25,
        min_d + (median - min_d) * 0.5,
        min_d + (median - min_d) * 0.75,
        median,
    ];
    for m in [1.5, 2.0, 3.0, 4.0] {
        let d = median * m;
        if d >= max_d {
            break;
        }
        anchors.push(d);
    }
    let mid_far = (anchors[anchors.len() - 1] + far) / 2.0;
    if mid_far < far && mid_far < max_d {
        anchors.push(mid_far);
    }
    anchors.push(far);
    let inv = anchors.iter().map(|d| 1.0 / d).collect::<Vec<_>>();
    let effects = first
        .iter()
        .map(|f| {
            f.fit
                .map(|(k, b)| inv.iter().map(|a| k * a + b).collect::<Vec<_>>())
        })
        .collect::<Vec<_>>();
    let fitted = effects.iter().flatten().collect::<Vec<_>>();
    if fitted.is_empty() {
        return vec![None; first.len()];
    }
    let medians = (0..inv.len())
        .map(|a| numpy_median(fitted.iter().map(|e| e[a]).collect()))
        .collect::<Vec<_>>();
    effects
        .iter()
        .map(|e| {
            e.as_ref().map(|e| {
                e.iter()
                    .zip(&medians)
                    .map(|(v, m)| (v - m).abs() / (m.abs() + 1e-8))
                    .fold(0.0, f64::max)
            })
        })
        .collect()
}

// the aligned depth (0 where left out) and the points it adds, or None without a fit or enough pixels
#[allow(clippy::type_complexity)]
fn finish(
    f: &AlignFrame,
    fp: &FirstPass,
    fit: Option<(f64, f64)>,
    max_depth: f32,
) -> Option<(Vec<f32>, Vec<[f32; 3]>, Vec<[u8; 3]>)> {
    let (k, b) = fit?;
    let (w, h) = (f.image.width, f.image.height);
    let mut aligned = fp
        .mono
        .iter()
        .map(|&d| {
            let inv = (1.0 / d.max(DEPTH_EPS)) as f64 * k + b;
            1.0 / (inv as f32).max(DEPTH_EPS)
        })
        .collect::<Vec<_>>();
    let aligned_edges = depth_edges(&aligned, w, h, ALIGNED_EDGE_RTOL);
    let keep = (0..w * h)
        .map(|p| {
            fp.mono_edge_ok[p]
                && !aligned_edges[p]
                && f.mask[p]
                && aligned[p] >= MIN_DEPTH
                && aligned[p] <= max_depth
        })
        .collect::<Vec<_>>();
    for p in 0..w * h {
        if !keep[p] {
            aligned[p] = 0.0;
        }
    }
    if keep.iter().filter(|&&k| k).count() < MIN_FINAL_PIXELS {
        return None;
    }
    let c2w = invert4(&f.w2c);
    let kk = f.intrinsics;
    let (points, colors) = (0..w * h)
        .filter(|&p| keep[p] && !fp.far[p])
        .map(|p| {
            // the reference unprojects integer pixel coordinates
            let (x, y) = ((p % w) as f64, (p / w) as f64);
            let d = f64::from(aligned[p]);
            let cam = [
                (x - kk[0][2]) / kk[0][0] * d,
                (y - kk[1][2]) / kk[1][1] * d,
                d,
            ];
            let world = [0, 1, 2].map(|r| {
                (c2w[r][0] * cam[0] + c2w[r][1] * cam[1] + c2w[r][2] * cam[2] + c2w[r][3]) as f32
            });
            (world, [0, 1, 2].map(|c| f.image.pixels[p * 3 + c]))
        })
        .unzip();
    Some((aligned, points, colors))
}

// sklearn's RANSACRegressor over a constrained least-squares line (slope at least MIN_COEF, intercept within
// +-MAX_BIAS): residual threshold the median absolute deviation of y, absolute-error loss, best by inlier count then
// R^2, the final line refitted on the best inliers. The samples come from a seeded generator, not numpy's stream.
fn ransac_line(x: &[f64], y: &[f64]) -> Option<(f64, f64)> {
    let n = x.len();
    if n < RANSAC_MIN_SAMPLES {
        return None;
    }
    let med = numpy_median(y.to_vec());
    let threshold = numpy_median(y.iter().map(|v| (v - med).abs()).collect());
    let mut rng = StdRng::seed_from_u64(RANSAC_SEED);
    let mut best: Option<(usize, f64, Vec<usize>)> = None;
    let mut max_trials = RANSAC_MAX_TRIALS;
    let mut trial = 0;
    while trial < max_trials {
        trial += 1;
        let subset = rand::seq::index::sample(&mut rng, n, RANSAC_MIN_SAMPLES).into_vec();
        let Some((k, b)) = constrained_fit(subset.iter().map(|&i| (x[i], y[i]))) else {
            continue;
        };
        let inliers = (0..n)
            .filter(|&i| (y[i] - (k * x[i] + b)).abs() <= threshold)
            .collect::<Vec<_>>();
        let count = inliers.len();
        if best.as_ref().is_some_and(|(c, _, _)| count < *c) {
            continue;
        }
        let score = r2(&inliers, x, y, k, b);
        if best
            .as_ref()
            .is_some_and(|(c, s, _)| count == *c && score < *s)
        {
            continue;
        }
        max_trials = max_trials.min(dynamic_max_trials(count, n));
        best = Some((count, score, inliers));
    }
    let (_, _, inliers) = best?;
    constrained_fit(inliers.iter().map(|&i| (x[i], y[i])))
}

fn constrained_fit(samples: impl Iterator<Item = (f64, f64)>) -> Option<(f64, f64)> {
    let (mut n, mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (x, y) in samples {
        n += 1.0;
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    if n < 2.0 {
        return None;
    }
    let (mx, my) = (sx / n, sy / n);
    let var = sxx / n - mx * mx;
    let k = if var > 0.0 {
        (sxy / n - mx * my) / var
    } else {
        0.0
    };
    // the intercept comes from the unconstrained slope, then each is clamped, as ConstrainedLinearRegression does
    let b = (my - k * mx).clamp(-MAX_BIAS, MAX_BIAS);
    Some((k.max(MIN_COEF), b))
}

fn r2(idx: &[usize], x: &[f64], y: &[f64], k: f64, b: f64) -> f64 {
    if idx.is_empty() {
        return f64::NEG_INFINITY;
    }
    let mean = idx.iter().map(|&i| y[i]).sum::<f64>() / idx.len() as f64;
    let res: f64 = idx.iter().map(|&i| (y[i] - (k * x[i] + b)).powi(2)).sum();
    let tot: f64 = idx.iter().map(|&i| (y[i] - mean).powi(2)).sum();
    if tot == 0.0 { 0.0 } else { 1.0 - res / tot }
}

// sklearn's _dynamic_max_trials
fn dynamic_max_trials(inliers: usize, samples: usize) -> usize {
    let ratio = inliers as f64 / samples as f64;
    let nom = (1.0 - RANSAC_STOP_PROBABILITY).max(f64::EPSILON);
    let denom = (1.0 - ratio.powi(RANSAC_MIN_SAMPLES as i32)).max(f64::EPSILON);
    if nom == 1.0 {
        return 0;
    }
    if denom == 1.0 {
        return usize::MAX;
    }
    (nom.ln() / denom.ln()).ceil().abs() as usize
}

// each depth's rank among the masked depths (searchsorted right / n * 100), 0 outside the mask
fn percentile_map(depth: &[f32], mask: &[bool]) -> Vec<f32> {
    let mut sorted = depth
        .iter()
        .zip(mask)
        .filter(|(_, m)| **m)
        .map(|(&d, _)| d)
        .collect::<Vec<_>>();
    if sorted.is_empty() {
        return vec![0.0; depth.len()];
    }
    sorted.sort_unstable_by(f32::total_cmp);
    let n = sorted.len() as f32;
    depth
        .iter()
        .zip(mask)
        .map(|(&d, &m)| {
            if m {
                sorted.partition_point(|&v| v <= d) as f32 / n * 100.0
            } else {
                0.0
            }
        })
        .collect()
}

// statistical outlier removal: mean distance to the SOR_NEIGHBOURS nearest others within SOR_STD_RATIO deviations
fn outlier_mask(points: &[[f32; 3]]) -> Vec<bool> {
    if points.len() <= SOR_NEIGHBOURS {
        return vec![true; points.len()];
    }
    let grid = PointGrid::new(points);
    let k = SOR_NEIGHBOURS as f64;
    // mean_knn counts the point itself at distance 0
    let means = points
        .par_iter()
        .map(|p| grid.mean_knn(p.map(f64::from), SOR_NEIGHBOURS + 1) * (k + 1.0) / k)
        .collect::<Vec<_>>();
    let mean = means.iter().sum::<f64>() / means.len() as f64;
    let std = (means.iter().map(|m| (m - mean).powi(2)).sum::<f64>() / means.len() as f64).sqrt();
    let limit = mean + SOR_STD_RATIO * std;
    means.iter().map(|&m| m <= limit).collect()
}

fn voxel_key(p: [f32; 3], voxel: f64) -> i64 {
    let c = p.map(|v| (f64::from(v) / voxel).floor() as i64);
    c[0].wrapping_mul(116_101_979)
        .wrapping_add(c[1].wrapping_mul(104_729))
        .wrapping_add(c[2])
}

// voxel_downsample_fixed: the first point of each voxel, in key order as np.unique returns them
fn voxel_first(points: &[[f32; 3]], voxel: f64) -> Vec<usize> {
    let mut first: HashMap<i64, usize> = HashMap::new();
    for (i, &p) in points.iter().enumerate() {
        first.entry(voxel_key(p, voxel)).or_insert(i);
    }
    let mut keyed = first.into_iter().collect::<Vec<_>>();
    keyed.sort_unstable_by_key(|&(k, _)| k);
    keyed.into_iter().map(|(_, i)| i).collect()
}

// adaptive_voxel_downsample's voxel size: a bisection for about `target` occupied voxels
fn adaptive_voxel_size(points: &[[f32; 3]], target: usize) -> f64 {
    if points.len() <= target {
        return SMALL_CLOUD_VOXEL;
    }
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for p in points {
        for d in 0..3 {
            lo[d] = lo[d].min(p[d]);
            hi[d] = hi[d].max(p[d]);
        }
    }
    let extent = (0..3).map(|d| f64::from(hi[d] - lo[d])).fold(0.0, f64::max);
    let (mut v_lo, mut v_hi) = (1e-4, extent / 100.0);
    let mut best = (v_lo + v_hi) / 2.0;
    for _ in 0..VOXEL_SEARCH_STEPS {
        let mid = (v_lo + v_hi) / 2.0;
        let mut keys = points
            .iter()
            .map(|&p| voxel_key(p, mid))
            .collect::<Vec<_>>();
        keys.sort_unstable();
        keys.dedup();
        let count = keys.len();
        best = mid;
        if (count as f64 - target as f64).abs() / target as f64 <= VOXEL_SEARCH_TOL {
            break;
        }
        if count > target {
            v_lo = mid;
        } else {
            v_hi = mid;
        }
    }
    best
}

fn share(mask: &[bool]) -> f64 {
    mask.iter().filter(|&&m| m).count() as f64 / mask.len().max(1) as f64
}

// torch.median: the lower middle value
fn lower_median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_unstable_by(f64::total_cmp);
    v[(v.len() - 1) / 2]
}

// np.median: the mean of the two middle values
fn numpy_median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_unstable_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

// np.percentile with linear interpolation
fn percentile(values: &[f64], q: f64) -> f64 {
    let mut v = values.to_vec();
    v.sort_unstable_by(f64::total_cmp);
    let pos = q / 100.0 * (v.len() - 1) as f64;
    let (i, frac) = (pos.floor() as usize, pos.fract());
    if i + 1 < v.len() {
        v[i] * (1.0 - frac) + v[i + 1] * frac
    } else {
        v[i]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // closed-form samples (a line, uniform noise, every tenth point a gross outlier) where sklearn's RANSACRegressor
    // with the reference's estimator gives k 0.7950 to 0.7970, b 0.0561 to 0.0578 over random_state 1, 2, 3 and 42
    #[test]
    fn ransac_matches_sklearn_on_the_same_samples() {
        let frac = |v: f64| v - v.floor();
        let (mut x, mut y) = (Vec::new(), Vec::new());
        for i in 0..5000 {
            let f = f64::from(i);
            let xi = 0.1 + 2.0 * frac(f * 0.618_033_988_749_894_9);
            let yi = if i % 10 == 0 {
                3.0 * frac(f * 0.569_840_290_998_053_2)
            } else {
                0.8 * xi + 0.05 + (frac(f * 0.754_877_666_246_692_7) - 0.5) * 0.01
            };
            x.push(xi);
            y.push(yi);
        }
        let (k, b) = ransac_line(&x, &y).expect("a fit");
        assert!(
            (0.794..0.798).contains(&k) && (0.055..0.059).contains(&b),
            "{k} {b}"
        );
    }

    #[test]
    fn percentile_map_ranks_like_searchsorted_right() {
        let depth = [1.0, 2.0, 2.0, 4.0, 9.0];
        let mask = [true, true, true, true, false];
        assert_eq!(
            percentile_map(&depth, &mask),
            vec![25.0, 75.0, 75.0, 100.0, 0.0]
        );
    }
}
