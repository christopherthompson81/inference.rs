//! `adaptive_vox`: the cells a mesh's triangles touch (separating-axis triangle-box tests), on a grid that shrinks 5%
//! at a time until no more than a budget of cells is occupied.

use rayon::prelude::*;

use crate::trellis2::mesh::Mesh;

// empty cells kept around the mesh on each side
const PAD: f32 = 3.0;
const EPS: f32 = 1e-6;
const SHRINK: f64 = 0.95;
// the smallest grid with a positive scale at the padding
const MIN_RESOLUTION: usize = 2 * PAD as usize + 1;
// voxel keys pack 11 bits an axis
const KEY_BITS: u32 = 11;

fn axis_overlaps(v: [[f32; 3]; 3], a: [f32; 3]) -> bool {
    let p = v.map(|x| x[0] * a[0] + x[1] * a[1] + x[2] * a[2]);
    let lower = p[0].min(p[1]).min(p[2]);
    let upper = p[0].max(p[1]).max(p[2]);
    let radius = 0.5 * (a[0].abs() + a[1].abs() + a[2].abs());
    lower <= radius && upper >= -radius
}

// the triangle `t` (grid units) against the unit cell at integer `cell`
fn overlaps(t: [[f32; 3]; 3], cell: [i32; 3]) -> bool {
    let c = cell.map(|v| v as f32 + 0.5);
    let v = t.map(|p| [p[0] - c[0], p[1] - c[1], p[2] - c[2]]);
    for d in 0..3 {
        let lo = v[0][d].min(v[1][d]).min(v[2][d]);
        let hi = v[0][d].max(v[1][d]).max(v[2][d]);
        if lo > 0.5 || hi < -0.5 {
            return false;
        }
    }
    let edge = |a: usize, b: usize| [v[b][0] - v[a][0], v[b][1] - v[a][1], v[b][2] - v[a][2]];
    let edges = [edge(0, 1), edge(1, 2), edge(2, 0)];
    for e in edges {
        for axis in [[0.0, e[2], -e[1]], [-e[2], 0.0, e[0]], [e[1], -e[0], 0.0]] {
            if !axis_overlaps(v, axis) {
                return false;
            }
        }
    }
    let (e0, e1) = (edges[0], edges[1]);
    let n = [
        e0[1] * e1[2] - e0[2] * e1[1],
        e0[2] * e1[0] - e0[0] * e1[2],
        e0[0] * e1[1] - e0[1] * e1[0],
    ];
    let offset = -(n[0] * v[0][0] + n[1] * v[0][1] + n[2] * v[0][2]);
    let radius = 0.5 * (n[0].abs() + n[1].abs() + n[2].abs());
    offset.abs() <= radius
}

/// The cells `mesh` touches on a `resolution` grid, the mesh centred and scaled to fill it less 3 cells a side; sorted
/// as the reference's keys sort (z, then y, then x).
pub fn voxelize(mesh: &Mesh, resolution: usize) -> Vec<[i32; 3]> {
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for v in &mesh.vertices {
        for d in 0..3 {
            lo[d] = lo[d].min(v[d]);
            hi[d] = hi[d].max(v[d]);
        }
    }
    let (lo, hi) = (lo.map(|v| v - EPS), hi.map(|v| v + EPS));
    let center = [0, 1, 2].map(|d| (lo[d] + hi[d]) / 2.0);
    let range = (0..3).map(|d| hi[d] - lo[d]).fold(0f32, f32::max);
    let res = resolution as f32;
    let scale = (res - PAD * 2.0 - 2.0 * EPS) / range;
    let grid: Vec<[f32; 3]> = mesh
        .vertices
        .iter()
        .map(|v| [0, 1, 2].map(|d| (v[d] - center[d]) * scale + res / 2.0))
        .collect();
    let last = resolution as i32 - 1;
    let mut keys: Vec<u64> = mesh
        .faces
        .par_iter()
        .flat_map_iter(|f| {
            let t = f.map(|i| grid[i as usize]);
            let start =
                [0, 1, 2].map(|d| (t[0][d].min(t[1][d]).min(t[2][d]).floor() as i32).max(0));
            let end =
                [0, 1, 2].map(|d| (t[0][d].max(t[1][d]).max(t[2][d]).floor() as i32).min(last));
            let mut hits = Vec::new();
            for z in start[2]..=end[2] {
                for y in start[1]..=end[1] {
                    for x in start[0]..=end[0] {
                        if overlaps(t, [x, y, z]) {
                            hits.push(
                                ((z as u64) << (2 * KEY_BITS))
                                    | ((y as u64) << KEY_BITS)
                                    | x as u64,
                            );
                        }
                    }
                }
            }
            hits
        })
        .collect();
    keys.par_sort_unstable();
    keys.dedup();
    let mask = (1u64 << KEY_BITS) - 1;
    keys.iter()
        .map(|&k| {
            [
                (k & mask) as i32,
                ((k >> KEY_BITS) & mask) as i32,
                ((k >> (2 * KEY_BITS)) & mask) as i32,
            ]
        })
        .collect()
}

/// [`voxelize`] at `resolution`, shrunk 5% at a time until at most `budget` cells are occupied: the cells and the
/// grid they settled on.
pub fn adaptive(mesh: &Mesh, resolution: usize, budget: usize) -> Option<(Vec<[i32; 3]>, usize)> {
    let mut scale = 1.0f64;
    loop {
        let res = (resolution as f64 * scale) as usize;
        if res < MIN_RESOLUTION {
            return None;
        }
        let cells = voxelize(mesh, res);
        if cells.len() <= budget {
            return Some((cells, res));
        }
        scale *= SHRINK;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_flat_square_fills_one_layer_inside_the_padding() {
        let mesh = Mesh {
            vertices: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            faces: vec![[0, 1, 2], [0, 2, 3]],
        };
        let cells = voxelize(&mesh, 16);
        let layers: std::collections::BTreeSet<i32> = cells.iter().map(|c| c[2]).collect();
        assert_eq!(layers.len(), 1, "a flat square touches one z layer");
        let (lo, hi) = cells.iter().fold((i32::MAX, i32::MIN), |(lo, hi), c| {
            (lo.min(c[0]), hi.max(c[0]))
        });
        assert_eq!(
            (lo, hi),
            (3, 12),
            "the square spans the grid less 3 cells a side"
        );
        assert_eq!(cells.len(), 100);
    }

    #[test]
    fn the_grid_shrinks_until_the_budget_holds() {
        let mesh = Mesh {
            vertices: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            faces: vec![[0, 1, 2], [0, 2, 3]],
        };
        let (cells, res) = adaptive(&mesh, 64, 400).expect("a grid fits");
        assert!(cells.len() <= 400 && res < 64);
        assert!(
            adaptive(&mesh, 64, 0).is_none(),
            "no grid holds a surface in no cells"
        );
    }
}
