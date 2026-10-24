//! A panorama made into a scene the planner and the renderer work on, as `traj_generate.py` prepares it: the merged
//! distance cleaned up (edges masked, clipped, compressed far away outdoors), its point cloud and mesh, the start views
//! and the panorama memory bank.

use super::panorama::{PanoDepth, Rgb8, quantile, split_perspective, uv_to_direction};
use super::trajectory::{self, Mat4};

// pixels whose 3 x 3 neighbourhood spans more than this relative distance are edges
const EDGE_RTOL: f64 = 0.1;
const CLIP_QUANTILE: f64 = 0.99;
// outdoors, distances past this many medians are compressed: d -> 2c - c^2 / d
const CONTRACT: f64 = 8.0;
const CONTRACT_EPS: f64 = 1e-6;
// the memory bank's views: three pitches, every 40 degrees around
const BANK_PITCHES: [f64; 3] = [0.0, 0.5, -0.5];
const BANK_STEP_DEG: f64 = 40.0;

/// A view of the scene from the panorama's centre.
pub struct View {
    pub c2w: Mat4,
    pub intrinsics: [[f64; 3]; 3],
    pub image: Rgb8,
}

pub struct Scene {
    pub width: usize,
    pub height: usize,
    /// per panorama pixel: cleaned distance, and whether it is sky
    pub distance: Vec<f32>,
    pub sky: Vec<bool>,
    /// sky or depth edge: left out of the point cloud
    pub excluded: Vec<bool>,
    pub points: Vec<[f32; 3]>,
    pub colors: Vec<[u8; 3]>,
    /// the panorama as a mesh (every pixel a vertex; quads touching sky dropped), for collision checks
    pub mesh_vertices: Vec<[f32; 3]>,
    pub mesh_triangles: Vec<[u32; 3]>,
    pub median_distance: f64,
    pub outdoor: bool,
}

// torch.median: the lower middle value
fn lower_median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_unstable_by(f64::total_cmp);
    v[(v.len() - 1) / 2]
}

// utils3d's depth_edge with rtol: (max - min) over each 3 x 3 window (outside ignored), relative to the depth
fn depth_edges(d: &[f32], width: usize, height: usize) -> Vec<bool> {
    (0..height * width)
        .map(|p| {
            let (x, y) = (p % width, p / width);
            let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
            for yy in y.saturating_sub(1)..(y + 2).min(height) {
                for xx in x.saturating_sub(1)..(x + 2).min(width) {
                    let v = d[yy * width + xx];
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
            f64::from(hi - lo) / f64::from(d[p]) > EDGE_RTOL
        })
        .collect()
}

impl Scene {
    /// From the panorama and its merged distance; `outdoor` picks the far-field compression (the reference asks a VLM),
    /// defaulting to whether any sky was found. Sky is what MoGe-2 found invalid in every view (the reference segments
    /// it with GroundingDINO and ZIM).
    pub fn new(pano: &Rgb8, depth: &PanoDepth, outdoor: Option<bool>) -> Self {
        let (w, h) = (depth.width, depth.height);
        let sky = depth.mask.iter().map(|m| !m).collect::<Vec<_>>();
        let outdoor = outdoor.unwrap_or_else(|| sky.iter().any(|&s| s));
        let mut distance = depth.distance.clone();
        let edges = depth_edges(&distance, w, h);
        let kept = (0..w * h)
            .filter(|&p| !(sky[p] || edges[p]))
            .map(|p| f64::from(distance[p]))
            .collect::<Vec<_>>();
        let max_d = quantile(&kept, CLIP_QUANTILE) as f32;
        distance.iter_mut().for_each(|d| *d = d.clamp(0.0, max_d));
        if outdoor {
            let c = lower_median(distance.iter().map(|&d| f64::from(d)).collect()) * CONTRACT;
            for d in &mut distance {
                if f64::from(*d) > c {
                    *d = (2.0 * c - c * c / (f64::from(*d) + CONTRACT_EPS)) as f32;
                }
            }
        }
        let edges = depth_edges(&distance, w, h);
        let excluded = (0..w * h).map(|p| sky[p] || edges[p]).collect::<Vec<_>>();
        let median_distance = lower_median(
            (0..w * h)
                .filter(|&p| !excluded[p])
                .map(|p| f64::from(distance[p]))
                .collect(),
        );
        let ray = |p: usize| {
            uv_to_direction(
                ((p % w) as f64 + 0.5) / w as f64,
                ((p / w) as f64 + 0.5) / h as f64,
            )
        };
        let vertex = |p: usize| ray(p).map(|r| (r * f64::from(distance[p])) as f32);
        let (points, colors) = (0..w * h)
            .filter(|&p| !excluded[p])
            .map(|p| (vertex(p), [0, 1, 2].map(|c| pano.pixels[p * 3 + c])))
            .unzip();
        let mesh_vertices = (0..w * h).map(vertex).collect::<Vec<_>>();
        let mut mesh_triangles = Vec::new();
        for y in 0..h - 1 {
            for x in 0..w {
                let (tl, tr) = (y * w + x, y * w + (x + 1) % w);
                let (bl, br) = (tl + w, tr + w);
                if sky[tl] || sky[tr] || sky[bl] || sky[br] {
                    continue;
                }
                mesh_triangles.push([tl as u32, tr as u32, bl as u32]);
                mesh_triangles.push([tr as u32, br as u32, bl as u32]);
            }
        }
        Self {
            width: w,
            height: h,
            distance,
            sky,
            excluded,
            points,
            colors,
            mesh_vertices,
            mesh_triangles,
            median_distance,
            outdoor,
        }
    }

    fn rotation(c2w: &Mat4) -> [[f64; 3]; 3] {
        // world to camera: the transpose of the camera-to-world rotation
        [0, 1, 2].map(|i| [0, 1, 2].map(|j| c2w[j][i]))
    }

    /// A view from the centre along `c2w`, at the start views' size and fields of view.
    pub fn view(&self, pano: &Rgb8, c2w: Mat4) -> View {
        let (vw, vh) = trajectory::view_size();
        let fov = (trajectory::VIEW_FOV_X_DEG, trajectory::VIEW_FOV_Y_DEG);
        View {
            c2w,
            intrinsics: trajectory::intrinsics(vw, vh, fov.0, fov.1),
            image: split_perspective(pano, &Self::rotation(&c2w), fov, (vw, vh)),
        }
    }

    /// The median z-depth the view sees of the cleaned scene, as the planner samples it (nearest panorama pixel per
    /// view pixel, excluded pixels left out), or None when it sees nothing valid.
    pub fn view_median_depth(&self, view: &View) -> Option<f64> {
        let (vw, vh) = trajectory::view_size();
        let k = view.intrinsics;
        let rot = Self::rotation(&view.c2w);
        let mut depths = Vec::with_capacity(vw * vh);
        for y in 0..vh {
            for x in 0..vw {
                let cam = [
                    (x as f64 + 0.5 - k[0][2]) / k[0][0],
                    (y as f64 + 0.5 - k[1][2]) / k[1][1],
                    1.0,
                ];
                let world = [0, 1, 2].map(|j| (0..3).map(|i| rot[i][j] * cam[i]).sum::<f64>());
                let (u, v) = super::panorama::direction_to_uv(world);
                // grid_sample(nearest, align_corners=True) at pixel (u * size - 0.5)
                let at = |t: f64, n: usize| {
                    let g = (t * n as f64 - 0.5) / n as f64 * 2.0 - 1.0;
                    (((g + 1.0) / 2.0 * (n - 1) as f64)
                        .round()
                        .clamp(0.0, (n - 1) as f64)) as usize
                };
                let p = at(v, self.height) * self.width + at(u, self.width);
                if !self.excluded[p] {
                    // distance to z-depth along the ray at the integer pixel, as split_panorama_depth converts it
                    let (rx, ry) = (
                        (x as f64 - k[0][2]) / k[0][0],
                        (y as f64 - k[1][2]) / k[1][1],
                    );
                    let len = (rx * rx + ry * ry + 1.0).sqrt();
                    depths.push(f64::from(self.distance[p]) / len);
                }
            }
        }
        let m = lower_median(depths);
        (m.is_finite() && m > 0.0).then_some(m)
    }

    /// The panorama memory bank: views at three pitches every 40 degrees around, starting from -x, in the planner's
    /// order (the three start directions, then each turned in turn).
    pub fn memory_bank(&self, pano: &Rgb8) -> Vec<View> {
        let starts = BANK_PITCHES.map(|z| [-1.0, 0.0, z]);
        let steps = (360.0 / BANK_STEP_DEG) as usize;
        let mut dirs = starts.to_vec();
        for s in starts {
            for i in 1..steps {
                let a = (BANK_STEP_DEG * i as f64).to_radians();
                // rotate_around_z_axis turns clockwise
                dirs.push([
                    a.cos() * s[0] + a.sin() * s[1],
                    -a.sin() * s[0] + a.cos() * s[1],
                    s[2],
                ]);
            }
        }
        dirs.into_iter()
            .map(|d| self.view(pano, trajectory::look_at_c2w(d)))
            .collect()
    }
}

#[cfg(test)]
mod parity {
    use inference_tensor::{DType, Device};

    use super::super::panorama::{PanoDepth, Rgb8};
    use super::super::trajectory::{self, Mat4, Obstacles, PointGrid};

    // PANO_REF (pred_pano_depth dump) and WORLDSTEREO_SCENE_REF: the reference's cleaned distance, point cloud, start
    // views and regular trajectories on it (dump_ws_scene.py)
    #[test]
    #[ignore = "needs the pano-depth and scene reference dumps"]
    fn scene_and_trajectories_match_the_reference() -> inference_tensor::Result<()> {
        let (Some(pano_ref), Some(scene_ref)) = (
            std::env::var_os("PANO_REF"),
            std::env::var_os("WORLDSTEREO_SCENE_REF"),
        ) else {
            return Ok(());
        };
        let p = inference_tensor::safetensors::load(pano_ref, &Device::Cpu)?;
        let r = inference_tensor::safetensors::load(scene_ref, &Device::Cpu)?;
        let (h, w, _) = p["pano"].dims3()?;
        let pano = Rgb8 {
            width: w,
            height: h,
            pixels: p["pano"].flatten_all()?.to_vec1::<u8>()?,
        };
        let depth = PanoDepth {
            width: w,
            height: h,
            distance: p["distance"].flatten_all()?.to_vec1::<f32>()?,
            mask: p["mask"]
                .flatten_all()?
                .to_vec1::<u8>()?
                .iter()
                .map(|&m| m > 0)
                .collect(),
        };
        let scene = super::Scene::new(&pano, &depth, Some(true));
        let want = r["distance"].flatten_all()?.to_vec1::<f32>()?;
        let max = scene
            .distance
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let excluded = r["excluded"].flatten_all()?.to_vec1::<u8>()?;
        let flips = scene
            .excluded
            .iter()
            .zip(&excluded)
            .filter(|(a, b)| **a != (**b > 0))
            .count();
        let median = r["median"].to_vec1::<f32>()?[0];
        println!(
            "distance max {max:.3e}; excluded flips {flips}; points {} vs {}; median {:.5} vs {median:.5}",
            scene.points.len(),
            r["points"].dim(0)?,
            scene.median_distance
        );
        let images = r["start_images"].flatten_all()?.to_vec1::<u8>()?;
        let grid = PointGrid::new(&scene.points);
        let obstacles = Obstacles {
            vertices: &scene.mesh_vertices,
            triangles: &scene.mesh_triangles,
            points: &grid,
        };
        for (v, c2w) in trajectory::start_views().into_iter().enumerate() {
            let view = scene.view(&pano, c2w);
            let n = view.image.pixels.len();
            let theirs = &images[v * n..(v + 1) * n];
            let diff = view
                .image
                .pixels
                .iter()
                .zip(theirs)
                .filter(|(a, b)| a != b)
                .count();
            let worst = view
                .image
                .pixels
                .iter()
                .zip(theirs)
                .map(|(a, b)| a.abs_diff(*b))
                .max()
                .unwrap_or(0);
            let md = scene.view_median_depth(&view).unwrap_or(f64::NAN);
            let want_md = r[&format!("view{v}.median")].to_vec1::<f32>()?[0];
            println!(
                "view{v}: image {:.3}% differ (max {worst}); median depth {md:.5} vs {want_md:.5}",
                100.0 * diff as f64 / n as f64
            );
            for (t, (mv, name)) in trajectory::regular_moves().into_iter().enumerate() {
                let ours = trajectory::plan(&c2w, mv, md, &obstacles);
                let theirs = r[&format!("view{v}.traj{t}.c2w")]
                    .to_dtype(DType::F64)?
                    .to_vec3::<f64>()?;
                let iters = r[&format!("view{v}.traj{t}.iterations")].to_vec1::<i64>()?[0];
                match ours {
                    Some(c2ws) => {
                        let err = c2ws
                            .iter()
                            .zip(&theirs)
                            .flat_map(|(a, b): (&Mat4, &Vec<Vec<f64>>)| {
                                (0..3).flat_map(move |i| {
                                    (0..4).map(move |j| (a[i][j] - b[i][j]).abs())
                                })
                            })
                            .fold(0f64, f64::max);
                        println!(
                            "  traj{t} {name}: {} poses, max |c2w diff| {err:.3e} (reference took {iters} shortenings)",
                            c2ws.len()
                        );
                    }
                    None => {
                        println!("  traj{t} {name}: dropped (reference took {iters} shortenings)")
                    }
                }
            }
        }
        Ok(())
    }
}
