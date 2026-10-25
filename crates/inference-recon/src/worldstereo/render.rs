//! Point-cloud renders along a trajectory, as WorldStereo's reference makes them with PyTorch3D: every point a disc
//! of radius 0.008 in NDC (the shorter side spans 2), the 20 nearest in front composited front to back with weights
//! 1 - d^2 / r^2 over a black background, and a mask of the pixels no point reaches.

use rayon::prelude::*;

use super::trajectory::Mat4;

const RADIUS_NDC: f64 = 0.008;
const POINTS_PER_PIXEL: usize = 20;
// pixels are binned in tiles of this size so each tile only sees the points that reach it
const TILE: usize = 16;

/// One rendered frame: RGB in [-1, 1] (channel-major, 3 x H x W) and 1 where no point reaches.
pub struct Render {
    pub rgb: Vec<f32>,
    pub empty: Vec<f32>,
}

/// Renders `points` (colors as 8-bit RGB) through world-to-camera `w2c` and pixel `intrinsics` at `width x height`.
pub fn render(
    points: &[[f32; 3]],
    colors: &[[u8; 3]],
    w2c: &Mat4,
    intrinsics: &[[f64; 3]; 3],
    (width, height): (usize, usize),
) -> Render {
    let radius_px = RADIUS_NDC * width.min(height) as f64 / 2.0;
    let r2 = radius_px * radius_px;
    let (tiles_x, tiles_y) = (width.div_ceil(TILE), height.div_ceil(TILE));
    // project, and bin each point into every tile its disc reaches
    let projected = points
        .par_iter()
        .enumerate()
        .filter_map(|(i, p)| {
            let p = p.map(f64::from);
            let c = [0, 1, 2]
                .map(|r| w2c[r][0] * p[0] + w2c[r][1] * p[1] + w2c[r][2] * p[2] + w2c[r][3]);
            if c[2] <= 0.0 {
                return None;
            }
            let u = intrinsics[0][0] * c[0] / c[2] + intrinsics[0][2];
            let v = intrinsics[1][1] * c[1] / c[2] + intrinsics[1][2];
            let inside = u + radius_px >= 0.0
                && v + radius_px >= 0.0
                && u - radius_px < width as f64
                && v - radius_px < height as f64;
            inside.then_some((i as u32, u, v, c[2]))
        })
        .collect::<Vec<_>>();
    let mut bins = vec![Vec::new(); tiles_x * tiles_y];
    for &(i, u, v, z) in &projected {
        let tx0 = (((u - radius_px - 0.5) / TILE as f64).floor().max(0.0)) as usize;
        let tx1 = ((((u + radius_px - 0.5) / TILE as f64).floor()) as usize).min(tiles_x - 1);
        let ty0 = (((v - radius_px - 0.5) / TILE as f64).floor().max(0.0)) as usize;
        let ty1 = ((((v + radius_px - 0.5) / TILE as f64).floor()) as usize).min(tiles_y - 1);
        for ty in ty0..=ty1 {
            for tx in tx0..=tx1 {
                bins[ty * tiles_x + tx].push((i, u, v, z));
            }
        }
    }
    let tiles = bins
        .par_iter()
        .enumerate()
        .map(|(t, bin)| {
            let (tx, ty) = (t % tiles_x, t / tiles_x);
            let (x0, y0) = (tx * TILE, ty * TILE);
            let (x1, y1) = ((x0 + TILE).min(width), (y0 + TILE).min(height));
            let mut out = Vec::with_capacity((x1 - x0) * (y1 - y0));
            let mut hits: Vec<(f64, f64, u32)> = Vec::new();
            for y in y0..y1 {
                for x in x0..x1 {
                    let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
                    hits.clear();
                    for &(i, u, v, z) in bin {
                        let d2 = (u - px).powi(2) + (v - py).powi(2);
                        if d2 < r2 {
                            hits.push((z, d2, i));
                        }
                    }
                    hits.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
                    hits.truncate(POINTS_PER_PIXEL);
                    let mut rgb = [0f64; 3];
                    let mut transmit = 1.0;
                    for &(_, d2, i) in &hits {
                        let alpha = 1.0 - d2 / r2;
                        let c = colors[i as usize];
                        for k in 0..3 {
                            rgb[k] += alpha * transmit * (f64::from(c[k]) / 255.0 * 2.0 - 1.0);
                        }
                        transmit *= 1.0 - alpha;
                    }
                    out.push((y * width + x, rgb, hits.is_empty()));
                }
            }
            out
        })
        .collect::<Vec<_>>();
    let plane = width * height;
    let mut rgb = vec![0f32; 3 * plane];
    let mut empty = vec![0f32; plane];
    for tile in tiles {
        for (p, c, e) in tile {
            for k in 0..3 {
                rgb[k * plane + p] = c[k] as f32;
            }
            empty[p] = if e { 1.0 } else { 0.0 };
        }
    }
    Render { rgb, empty }
}

#[cfg(test)]
mod parity {
    use inference_tensor::{DType, Device};

    use super::super::panorama::{PanoDepth, Rgb8};
    use super::super::scene::Scene;
    use super::super::trajectory::{self, Mat4};

    // PANO_REF, WORLDSTEREO_SCENE_REF and WORLDSTEREO_RENDER_REF: PyTorch3D renders of the scene's point cloud along two
    // of its trajectories
    #[test]
    #[ignore = "needs the pano-depth, scene and render reference dumps"]
    fn renders_match_the_reference() -> inference_tensor::Result<()> {
        let (Some(pano_ref), Some(scene_ref), Some(render_ref)) = (
            std::env::var_os("PANO_REF"),
            std::env::var_os("WORLDSTEREO_SCENE_REF"),
            std::env::var_os("WORLDSTEREO_RENDER_REF"),
        ) else {
            return Ok(());
        };
        let p = inference_tensor::safetensors::load(pano_ref, &Device::Cpu)?;
        let s = inference_tensor::safetensors::load(scene_ref, &Device::Cpu)?;
        let r = inference_tensor::safetensors::load(render_ref, &Device::Cpu)?;
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
        let scene = Scene::new(&pano, &depth, Some(true));
        let (vw, vh) = trajectory::view_size();
        let k = trajectory::intrinsics(
            vw,
            vh,
            trajectory::VIEW_FOV_X_DEG,
            trajectory::VIEW_FOV_Y_DEG,
        );
        for key in ["view0.traj2", "view1.traj0"] {
            let c2ws = s[&format!("{key}.c2w")]
                .to_dtype(DType::F64)?
                .to_vec3::<f64>()?;
            let want_rgb = r[&format!("{key}.rgb")].flatten_all()?.to_vec1::<f32>()?;
            let want_mask = r[&format!("{key}.mask")].flatten_all()?.to_vec1::<f32>()?;
            let plane = vw * vh;
            let (mut flips, mut sq, mut n, mut worst) = (0usize, 0f64, 0usize, 0f32);
            let start = std::time::Instant::now();
            for (f, c2w) in c2ws.iter().enumerate() {
                let c2w: Mat4 = std::array::from_fn(|i| std::array::from_fn(|j| c2w[i][j]));
                let w2c = crate::geometry::invert4(&c2w);
                let out = super::render(&scene.points, &scene.colors, &w2c, &k, (vw, vh));
                for p in 0..plane {
                    let m = want_mask[f * plane + p];
                    flips += usize::from((out.empty[p] > 0.5) != (m > 0.5));
                    if m < 0.5 && out.empty[p] < 0.5 {
                        for c in 0..3 {
                            let d = out.rgb[c * plane + p] - want_rgb[(f * 3 + c) * plane + p];
                            sq += f64::from(d * d);
                            worst = worst.max(d.abs());
                            n += 1;
                        }
                    }
                }
            }
            println!(
                "{key}: {:.2} s for 21 frames; mask flips {:.4}%; rgb where both cover: rms {:.3e} max {worst:.3e}",
                start.elapsed().as_secs_f64(),
                100.0 * flips as f64 / (21 * plane) as f64,
                (sq / n as f64).sqrt()
            );
        }
        Ok(())
    }
}
