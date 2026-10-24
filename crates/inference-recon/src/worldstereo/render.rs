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
