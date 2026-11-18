//! Idol's splats as a volume of colors for TRELLIS.2's baker to sample: each Gaussian's color spread over the cells
//! within a standard deviation of it, weighted by its opacity and falloff, then grown outward so a surface just
//! outside the splats still finds a color.

use std::collections::HashMap;

use super::Gaussians;

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
}
