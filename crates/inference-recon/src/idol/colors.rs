//! Idol's splats as a volume of colors for TRELLIS.2's baker to sample: each Gaussian's color spread over the cells
//! within a standard deviation of it, weighted by its opacity and falloff, then grown outward so a surface just
//! outside the splats still finds a color.

use std::collections::HashMap;

use rayon::prelude::*;

use super::Gaussians;
use crate::trellis2::bake::{self, Baked, Prepared};
use crate::trellis2::glb::{Pbr, sample_attributes};

// the zeroth spherical harmonic: a splat's color is 0.5 + SH_C0 * dc
const SH_C0: f32 = 0.282_094_8;
// a Gaussian reaches this many standard deviations, and at least half a cell
const COLOR_REACH: f32 = 1.0;
const MIN_REACH_CELLS: f32 = 0.5;
const MIN_OPACITY: f32 = 0.05;
// rounds of growth into empty face neighbours, each taking their mean
const GROW_ROUNDS: usize = 3;
// matte and opaque: the splats' colors already hold their lighting
const METALLIC: f32 = 0.0;
const ROUGHNESS: f32 = 1.0;
const ALPHA: f32 = 1.0;
// a texel blends the Gaussians within this many standard deviations of it (the surface sits a little outside them)
const TEXEL_REACH: f32 = 3.0;
// the volume's color counts as this much Gaussian weight, so a texel only a few faint Gaussians reach keeps it
const VOLUME_PRIOR: f32 = 0.1;

/// The cells of a `res`^3 grid over the unit cube holding color, and their attributes as TRELLIS.2's decoder gives
/// them (base color, metallic, roughness, alpha).
pub fn color_cells(g: &Gaussians, res: usize) -> (Vec<[i32; 3]>, Vec<[f32; 6]>) {
    let mut sums: HashMap<[i32; 3], ([f32; 3], f32)> = HashMap::new();
    let cells = res as f32;
    for i in 0..g.xyz.len() {
        let opacity = g.opacity[i];
        if opacity < MIN_OPACITY {
            continue;
        }
        let color = g.dc[i].map(|c| (0.5 + SH_C0 * c).clamp(0.0, 1.0));
        let r = crate::printable::rotation(g.rotation[i]);
        let scale = g.scaling[i].map(|s| (s * cells).max(MIN_REACH_CELLS));
        let centre = g.xyz[i].map(|c| (c + 0.5) * cells);
        let half: [f32; 3] = std::array::from_fn(|j| {
            let var: f32 = (0..3).map(|m| (r[j][m] * scale[m]).powi(2)).sum();
            COLOR_REACH * var.sqrt()
        });
        let lo: [i32; 3] = std::array::from_fn(|a| (centre[a] - half[a] - 0.5).ceil() as i32);
        let hi: [i32; 3] = std::array::from_fn(|a| (centre[a] + half[a] - 0.5).floor() as i32);
        let home = centre.map(|c| c.floor() as i32);
        let lo: [i32; 3] = std::array::from_fn(|a| lo[a].min(home[a]));
        let hi: [i32; 3] = std::array::from_fn(|a| hi[a].max(home[a]));
        for z in lo[2]..=hi[2] {
            for y in lo[1]..=hi[1] {
                for x in lo[0]..=hi[0] {
                    let d = [x, y, z].map(|c| c as f32 + 0.5);
                    let d: [f32; 3] = std::array::from_fn(|a| d[a] - centre[a]);
                    let q: f32 = (0..3)
                        .map(|m| {
                            ((r[0][m] * d[0] + r[1][m] * d[1] + r[2][m] * d[2]) / scale[m]).powi(2)
                        })
                        .sum();
                    // the cell holding the centre always takes it, so a small Gaussian near a corner counts
                    if q > COLOR_REACH * COLOR_REACH && [x, y, z] != home {
                        continue;
                    }
                    let w = opacity * (-0.5 * q).exp();
                    let entry = sums.entry([x, y, z]).or_default();
                    for a in 0..3 {
                        entry.0[a] += w * color[a];
                    }
                    entry.1 += w;
                }
            }
        }
    }
    let mut colors: HashMap<[i32; 3], [f32; 3]> = sums
        .into_iter()
        .map(|(cell, (sum, w))| (cell, sum.map(|c| c / w)))
        .collect();
    for _ in 0..GROW_ROUNDS {
        let mut grown: HashMap<[i32; 3], ([f32; 3], f32)> = HashMap::new();
        for (cell, color) in &colors {
            for a in 0..3 {
                for step in [-1, 1] {
                    let mut next = *cell;
                    next[a] += step;
                    if colors.contains_key(&next) {
                        continue;
                    }
                    let entry = grown.entry(next).or_default();
                    for k in 0..3 {
                        entry.0[k] += color[k];
                    }
                    entry.1 += 1.0;
                }
            }
        }
        colors.extend(
            grown
                .into_iter()
                .map(|(cell, (sum, n))| (cell, sum.map(|c| c / n))),
        );
    }
    colors
        .into_iter()
        .map(|(cell, [r, g, b])| (cell, [r, g, b, METALLIC, ROUGHNESS, ALPHA]))
        .unzip()
}

// a Gaussian's centre, rotation and scale, in grid units
type Frame = ([f32; 3], [[f32; 3]; 3], [f32; 3]);

// the Gaussians as frames and an index of the cells each reaches
struct SplatSampler<'a> {
    g: &'a Gaussians,
    frames: Vec<Option<Frame>>,
    index: HashMap<[i32; 3], Vec<u32>>,
}

impl<'a> SplatSampler<'a> {
    fn new(g: &'a Gaussians, res: usize) -> Self {
        let cube = res as f32;
        let frames: Vec<_> = (0..g.xyz.len())
            .map(|i| {
                (g.opacity[i] >= MIN_OPACITY).then(|| {
                    let r = crate::printable::rotation(g.rotation[i]);
                    let scale = g.scaling[i].map(|s| (s * cube).max(MIN_REACH_CELLS));
                    (g.xyz[i].map(|c| (c + 0.5) * cube), r, scale)
                })
            })
            .collect();
        let mut index: HashMap<[i32; 3], Vec<u32>> = HashMap::new();
        for (i, frame) in frames.iter().enumerate() {
            let Some((centre, r, scale)) = frame else {
                continue;
            };
            let half: [f32; 3] = std::array::from_fn(|j| {
                let var: f32 = (0..3).map(|m| (r[j][m] * scale[m]).powi(2)).sum();
                TEXEL_REACH * var.sqrt()
            });
            let lo: [i32; 3] = std::array::from_fn(|a| (centre[a] - half[a]).floor() as i32);
            let hi: [i32; 3] = std::array::from_fn(|a| (centre[a] + half[a]).floor() as i32);
            for z in lo[2]..=hi[2] {
                for y in lo[1]..=hi[1] {
                    for x in lo[0]..=hi[0] {
                        index.entry([x, y, z]).or_default().push(i as u32);
                    }
                }
            }
        }
        Self { g, frames, index }
    }

    // the color at `p` (grid units) blended from the Gaussians reaching it, `prior` counting as VOLUME_PRIOR of weight
    fn color_at(&self, p: &[f32; 3], prior: [f32; 3]) -> [f32; 3] {
        let cell = p.map(|v| v.floor() as i32);
        let (mut sum, mut total) = (prior.map(|c| VOLUME_PRIOR * c), VOLUME_PRIOR);
        for &i in self.index.get(&cell).map_or(&[][..], Vec::as_slice) {
            let i = i as usize;
            let (centre, r, scale) = self.frames[i].expect("indexed Gaussians have frames");
            let d: [f32; 3] = std::array::from_fn(|a| p[a] - centre[a]);
            let q: f32 = (0..3)
                .map(|m| ((r[0][m] * d[0] + r[1][m] * d[1] + r[2][m] * d[2]) / scale[m]).powi(2))
                .sum();
            if q > TEXEL_REACH * TEXEL_REACH {
                continue;
            }
            let w = self.g.opacity[i] * (-0.5 * q).exp();
            let color = self.g.dc[i].map(|c| (0.5 + SH_C0 * c).clamp(0.0, 1.0));
            for k in 0..3 {
                sum[k] += w * color[k];
            }
            total += w;
        }
        sum.map(|c| c / total)
    }
}

/// The prepared texels' colors blended from the Gaussians reaching each (weighted by opacity and falloff), so the
/// texture holds the splats' own detail; the `res`^3 volume's trilinear color weighs in as a prior, so a texel few
/// Gaussians reach keeps it.
pub fn bake_splat_colors(
    g: &Gaussians,
    prepared: Prepared,
    res: usize,
) -> inference_tensor::Result<Baked> {
    let (cells, values) = color_cells(g, res);
    let sampler = SplatSampler::new(g, res);
    bake::finish_with(prepared, |points| {
        let fallback = sample_attributes(&cells, &values, points);
        points
            .par_iter()
            .zip(fallback)
            .map(|(p, fallback)| Pbr {
                base_color: sampler.color_at(p, fallback.base_color),
                ..fallback
            })
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_gaussians_color_fills_its_cells_and_grows_past_them() {
        // dc for pure red: (1 - 0.5) / SH_C0 in red, (0 - 0.5) / SH_C0 elsewhere
        let g = Gaussians {
            xyz: vec![[0.0; 3]],
            dc: vec![[0.5 / SH_C0, -0.5 / SH_C0, -0.5 / SH_C0]],
            scaling: vec![[0.01; 3]],
            rotation: vec![[1.0, 0.0, 0.0, 0.0]],
            opacity: vec![1.0],
        };
        let (cells, values) = color_cells(&g, 64);
        let at = |c: [i32; 3]| cells.iter().position(|&x| x == c).map(|i| values[i]);
        let red = [1.0, 0.0, 0.0, METALLIC, ROUGHNESS, ALPHA];
        // the centre (cell 32 starts at the origin; the reach covers cells 31 and 32) and a cell GROW_ROUNDS past it
        for cell in [[32, 32, 32], [32, 32, 32 + GROW_ROUNDS as i32]] {
            let v = at(cell).unwrap_or_else(|| panic!("no color at {cell:?}"));
            assert!(
                v.iter().zip(red).all(|(a, b)| (a - b).abs() < 1e-5),
                "{v:?}"
            );
        }
        assert!(at([32, 32, 40]).is_none());
    }

    #[test]
    fn a_texel_takes_the_color_of_the_gaussians_reaching_it() {
        // red and blue Gaussians 0.1 apart on x; dc for a pure color is (c - 0.5) / SH_C0
        let pure = |c: [f32; 3]| c.map(|v| (v - 0.5) / SH_C0);
        let g = Gaussians {
            xyz: vec![[-0.05, 0.0, 0.0], [0.05, 0.0, 0.0]],
            dc: vec![pure([1.0, 0.0, 0.0]), pure([0.0, 0.0, 1.0])],
            scaling: vec![[0.01; 3]; 2],
            rotation: vec![[1.0, 0.0, 0.0, 0.0]; 2],
            opacity: vec![1.0; 2],
        };
        let res = 64;
        let sampler = SplatSampler::new(&g, res);
        let at = |x: f32| [(x + 0.5) * res as f32, 0.5 * res as f32, 0.5 * res as f32];
        let gray = [0.5; 3];
        let red = sampler.color_at(&at(-0.045), gray);
        assert!(red[0] > 0.85 && red[2] < 0.1, "{red:?}");
        let blue = sampler.color_at(&at(0.045), gray);
        assert!(blue[2] > 0.85 && blue[0] < 0.1, "{blue:?}");
        // nothing reaches a point far from both: the prior alone
        assert_eq!(sampler.color_at(&at(0.3), gray), gray);
    }
}
