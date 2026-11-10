//! A triangle mesh, and its extraction from a flexible dual grid (o-voxel's `flexible_dual_grid_to_mesh`, inference
//! path): every cell is a vertex at its dual position; each edge a cell's flags mark as crossed makes a quad of the four
//! cells around it, when all four exist, split into two triangles along the diagonal with the larger weight product.

use std::collections::HashMap;

/// Vertices and triangles (indices into them).
#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub vertices: Vec<[f32; 3]>,
    pub faces: Vec<[u32; 3]>,
}

// the four cells around a cell's +x, +y and +z edge, the cell first
const EDGE_CELLS: [[[i32; 3]; 4]; 3] = [
    [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
    [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
    [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
];
// a quad's two splits: (0, 1, 2) + (0, 2, 3), or (0, 1, 3) + (3, 1, 2)
const SPLIT_02: [[usize; 3]; 2] = [[0, 1, 2], [0, 2, 3]];
const SPLIT_13: [[usize; 3]; 2] = [[0, 1, 3], [3, 1, 2]];

fn key([x, y, z]: [i32; 3]) -> Option<u64> {
    (x >= 0 && y >= 0 && z >= 0).then(|| ((x as u64) << 42) | ((y as u64) << 21) | z as u64)
}

/// The mesh of cells `coords` on a `grid`^3 grid filling [-0.5, 0.5]^3: their dual vertices (offsets within the cell),
/// crossed-edge flags and split weights.
pub fn dual_grid_mesh(
    coords: &[[i32; 3]],
    dual: &[[f32; 3]],
    crossed: &[[bool; 3]],
    split: &[f32],
    grid: usize,
) -> Mesh {
    let voxel = 1.0f32 / grid as f32;
    let vertices = coords
        .iter()
        .zip(dual)
        .map(|(c, d)| std::array::from_fn(|a| (c[a] as f32 + d[a]) * voxel - 0.5))
        .collect();
    let index: HashMap<u64, u32> = coords
        .iter()
        .enumerate()
        .filter_map(|(i, &c)| key(c).map(|k| (k, i as u32)))
        .collect();
    let mut faces = Vec::new();
    for (c, flags) in coords.iter().zip(crossed) {
        for (axis, _) in flags.iter().enumerate().filter(|&(_, &f)| f) {
            let quad: Option<Vec<u32>> = EDGE_CELLS[axis]
                .iter()
                .map(|o| {
                    key(std::array::from_fn(|a| c[a] + o[a])).and_then(|k| index.get(&k).copied())
                })
                .collect();
            let Some(q) = quad else { continue };
            let w = |i: usize| split[q[i] as usize];
            let tris = if w(0) * w(2) > w(1) * w(3) {
                SPLIT_02
            } else {
                SPLIT_13
            };
            faces.extend(tris.iter().map(|t| t.map(|i| q[i])));
        }
    }
    Mesh { vertices, faces }
}

impl Mesh {
    /// A binary little-endian PLY of the vertices and triangles.
    pub fn to_ply(&self) -> Vec<u8> {
        let mut out = format!(
            "ply\nformat binary_little_endian 1.0\nelement vertex {}\nproperty float x\nproperty float y\n\
             property float z\nelement face {}\nproperty list uchar uint vertex_indices\nend_header\n",
            self.vertices.len(),
            self.faces.len()
        )
        .into_bytes();
        out.reserve(self.vertices.len() * 12 + self.faces.len() * 13);
        for v in &self.vertices {
            for c in v {
                out.extend_from_slice(&c.to_le_bytes());
            }
        }
        for f in &self.faces {
            out.push(3);
            for i in f {
                out.extend_from_slice(&i.to_le_bytes());
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_crossed_edge_between_four_cells_makes_two_triangles() {
        // four cells around the +x edge of (0, 0, 0)
        let coords = [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]];
        let dual = [[0.5; 3]; 4];
        let mut crossed = [[false; 3]; 4];
        crossed[0][0] = true;
        let mesh = dual_grid_mesh(&coords, &dual, &crossed, &[1.0, 1.0, 2.0, 1.0], 4);
        // 0-2 carries the larger product
        assert_eq!(mesh.faces, [[0, 1, 2], [0, 2, 3]]);
        assert_eq!(mesh.vertices[0], [-0.375; 3]);
        // a missing neighbour leaves the edge open
        let mesh = dual_grid_mesh(&coords[..3], &dual[..3], &crossed[..3], &[1.0; 3], 4);
        assert!(mesh.faces.is_empty());
    }
}
