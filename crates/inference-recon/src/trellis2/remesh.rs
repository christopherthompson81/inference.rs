//! CuMesh's narrow-band dual-contouring remesh (`remesh_narrow_band_dc`), which `to_glb` runs when asked to remesh: the
//! level set of the unsigned distance to the mesh minus `band` voxels is found on a sparse grid refined from 32^3 to
//! `resolution` (voxels kept while that level set may pass through them), each voxel crossing it gets the mean of its
//! edge crossings as a vertex, each crossed edge the quad of the four voxels around it, split along its flatter
//! diagonal. Vertices are then pulled back toward the original surface. The level set wraps the surface on both
//! sides, so the result is a closed shell around it.

use rayon::prelude::*;

use inference_tensor::Result;

use super::bake::Distances;
use super::mesh::Mesh;
use super::vec3::{V3, cross, dot, sub};

/// `to_glb`'s band width (voxels) and pull-back toward the surface
pub const BAND: f32 = 1.0;
pub const PROJECT_BACK: f32 = 0.9;
// the coarse grid's size: the resolution halved until at most this
const COARSE: usize = 32;
// sqrt(3) / 2: a voxel's center-to-corner reach, in voxels
const HALF_DIAGONAL: f32 = 0.87;
const CORNERS: [[u32; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
];
// the four voxels around each voxel's far x, y and z edge, as offsets from it
const EDGE_VOXELS: [[[u32; 3]; 4]; 3] = [
    [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
    [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
    [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
];
// a quad's two triangle pairs per diagonal, for each crossing direction
const SPLIT_1_NEG: [usize; 6] = [0, 1, 2, 0, 2, 3];
const SPLIT_1_POS: [usize; 6] = [0, 2, 1, 0, 3, 2];
const SPLIT_2_NEG: [usize; 6] = [0, 1, 3, 3, 1, 2];
const SPLIT_2_POS: [usize; 6] = [0, 3, 1, 3, 2, 1];

/// Where the remesh's grid sits: its center, edge length, and voxels per edge.
#[derive(Clone, Copy, Debug)]
pub struct Domain {
    pub center: V3,
    pub scale: f32,
    pub resolution: u32,
}

impl Domain {
    /// `to_glb`'s domain for a mesh in [-0.5, 0.5]^3 at `resolution`: the unit cube grown by three band widths.
    pub fn unit_cube(resolution: u32) -> Self {
        let r = resolution as f32;
        Self {
            center: [0.0; 3],
            scale: (r + 3.0 * BAND) / r,
            resolution,
        }
    }

    // a grid position (in voxels at `res`) in space
    fn point(&self, g: V3, res: f32) -> V3 {
        std::array::from_fn(|k| (g[k] / res - 0.5) * self.scale + self.center[k])
    }
}

fn key([x, y, z]: [u32; 3], side: u64) -> u64 {
    (u64::from(x) * side + u64::from(y)) * side + u64::from(z)
}

/// The voxels at `domain.resolution` the shifted level set may cross, in CuMesh's order (its coarse grid's
/// x-major order, each kept voxel followed by its children in `CORNERS` order).
pub fn narrow_band(distances: &Distances, domain: &Domain, band: f32) -> Result<Vec<[u32; 3]>> {
    let eps = band * domain.scale / domain.resolution as f32;
    let mut res = domain.resolution as usize;
    while res > COARSE {
        res /= 2;
    }
    let mut coords: Vec<[u32; 3]> = (0..res * res * res)
        .map(|i| {
            [
                (i / (res * res)) as u32,
                ((i / res) % res) as u32,
                (i % res) as u32,
            ]
        })
        .collect();
    loop {
        let cell = domain.scale / res as f32;
        let points: Vec<V3> = coords
            .iter()
            .map(|c| domain.point(c.map(|v| v as f32 + 0.5), res as f32))
            .collect();
        let near = distances.unsigned_distance(&points)?;
        let kept = coords
            .iter()
            .zip(&near)
            .filter(|(_, d)| (d.0 - eps).abs() < HALF_DIAGONAL * cell);
        if res >= domain.resolution as usize {
            return Ok(kept.map(|(c, _)| *c).collect());
        }
        res *= 2;
        coords = kept
            .flat_map(|(c, _)| CORNERS.map(|o| std::array::from_fn(|k| 2 * c[k] + o[k])))
            .collect();
    }
}

/// The remeshed surface (see the module docs), with vertices moved `project_back` of the way to their closest point
/// on the mesh `distances` query.
pub fn remesh(
    distances: &Distances,
    domain: &Domain,
    band: f32,
    project_back: f32,
) -> Result<Mesh> {
    let res = domain.resolution;
    let eps = band * domain.scale / res as f32;
    let voxels = narrow_band(distances, domain, band)?;
    let side = u64::from(res) + 1;
    // the band's grid vertices, sorted by key, and the level set's value at each
    let mut corners: Vec<u64> = voxels
        .par_iter()
        .flat_map_iter(|v| CORNERS.map(|o| key(std::array::from_fn(|k| v[k] + o[k]), side)))
        .collect();
    corners.par_sort_unstable();
    corners.dedup();
    let corner_points: Vec<V3> = corners
        .iter()
        .map(|&k| {
            let g = [k / (side * side), (k / side) % side, k % side].map(|v| v as f32);
            domain.point(g, res as f32)
        })
        .collect();
    let values: Vec<f32> = distances
        .unsigned_distance(&corner_points)?
        .iter()
        .map(|d| d.0 - eps)
        .collect();
    let value = |g: [u32; 3]| {
        values[corners
            .binary_search(&key(g, side))
            .expect("a corner of a band voxel")]
    };
    // dual vertices and each voxel's far-edge crossings (+1 rising, -1 falling, 0 none)
    let dual: Vec<(V3, [i8; 3])> = voxels
        .par_iter()
        .map(|&[x, y, z]| {
            let (mut sum, mut count) = ([0f32; 3], 0u32);
            let mut crossed = [0i8; 3];
            for axis in 0..3 {
                for u in 0..2 {
                    for v in 0..2 {
                        let (a, b) = ((axis + 1) % 3, (axis + 2) % 3);
                        let mut g = [x, y, z];
                        g[a] += u;
                        g[b] += v;
                        let mut h = g;
                        h[axis] += 1;
                        let (v1, v2) = (value(g), value(h));
                        let rises = v1 < 0.0 && v2 >= 0.0;
                        let falls = v1 >= 0.0 && v2 < 0.0;
                        if rises || falls {
                            let t = -v1 / (v2 - v1);
                            for k in 0..3 {
                                sum[k] += g[k] as f32 + if k == axis { t } else { 0.0 };
                            }
                            count += 1;
                        }
                        if u == 1 && v == 1 {
                            crossed[axis] = if rises {
                                1
                            } else if falls {
                                -1
                            } else {
                                0
                            };
                        }
                    }
                }
            }
            let p = if count > 0 {
                sum.map(|s| s / count as f32)
            } else {
                [x, y, z].map(|c| c as f32 + 0.5)
            };
            (p, crossed)
        })
        .collect();
    let rside = u64::from(res);
    let mut voxel_index: Vec<(u64, u32)> = voxels
        .iter()
        .enumerate()
        .map(|(i, &v)| (key(v, rside), i as u32))
        .collect();
    voxel_index.par_sort_unstable();
    let find = |g: [u32; 3]| -> Option<u32> {
        if g.iter().any(|&c| c >= res) {
            return None;
        }
        let k = key(g, rside);
        voxel_index
            .binary_search_by_key(&k, |e| e.0)
            .ok()
            .map(|i| voxel_index[i].1)
    };
    let quads: Vec<([u32; 4], i8)> = voxels
        .iter()
        .zip(&dual)
        .flat_map(|(v, (_, crossed))| {
            (0..3)
                .filter(|&a| crossed[a] != 0)
                .map(move |a| (*v, a, crossed[a]))
        })
        .filter_map(|(v, axis, dir)| {
            let around = EDGE_VOXELS[axis].map(|o| find(std::array::from_fn(|k| v[k] + o[k])));
            around
                .iter()
                .all(Option::is_some)
                .then(|| (around.map(|i| i.expect("checked")), dir))
        })
        .collect();
    // vertices the quads use, in voxel order
    let mut used = vec![u32::MAX; voxels.len()];
    for (q, _) in &quads {
        for &i in q {
            used[i as usize] = 0;
        }
    }
    let mut vertices = Vec::new();
    for (i, u) in used.iter_mut().enumerate() {
        if *u == 0 {
            *u = vertices.len() as u32;
            let g = dual[i].0;
            vertices.push(domain.point(g, res as f32));
        }
    }
    let faces: Vec<[u32; 3]> = quads
        .par_iter()
        .flat_map_iter(|(q, dir)| {
            let q = q.map(|i| used[i as usize]);
            let (first, second) = if *dir == 1 {
                (SPLIT_1_POS, SPLIT_2_POS)
            } else {
                (SPLIT_1_NEG, SPLIT_2_NEG)
            };
            // the split whose two triangles' normals agree more
            let align = |split: [usize; 6]| {
                let p = split.map(|s| vertices[q[s] as usize]);
                let n0 = cross(sub(p[1], p[0]), sub(p[2], p[0]));
                let n1 = cross(sub(p[2], p[1]), sub(p[3], p[1]));
                dot(n0, n1).abs()
            };
            let split = if align(first) > align(second) {
                first
            } else {
                second
            };
            [
                [q[split[0]], q[split[1]], q[split[2]]],
                [q[split[3]], q[split[4]], q[split[5]]],
            ]
        })
        .collect();
    if project_back > 0.0 {
        let closest = distances.unsigned_distance(&vertices)?;
        for (v, (_, _, c)) in vertices.iter_mut().zip(closest) {
            *v = std::array::from_fn(|k| v[k] - project_back * (v[k] - c[k]));
        }
    }
    Ok(Mesh { vertices, faces })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trellis2::bake::Bvh;

    #[test]
    fn a_square_remeshes_into_a_closed_shell_around_it() {
        // a flat square in z = 0, two triangles
        let mesh = Mesh {
            vertices: vec![
                [-0.3, -0.3, 0.0],
                [0.3, -0.3, 0.0],
                [0.3, 0.3, 0.0],
                [-0.3, 0.3, 0.0],
            ],
            faces: vec![[0, 1, 2], [0, 2, 3]],
        };
        let bvh = Bvh::new(&mesh);
        let domain = Domain::unit_cube(64);
        let distances = Distances::new(&bvh, &inference_tensor::Device::Cpu).expect("a host query");
        let shell = remesh(&distances, &domain, BAND, 0.0).expect("a shell");
        assert!(!shell.faces.is_empty());
        // every vertex sits about one band width (1/64 of the domain) from the square
        let eps = BAND * domain.scale / 64.0;
        for v in &shell.vertices {
            let d = bvh.unsigned_distance(&[*v])[0].0;
            assert!((d - eps).abs() < 0.6 * eps, "{v:?} at {d}");
        }
        // closed: every edge has exactly two faces
        let mut edges = std::collections::HashMap::new();
        for f in &shell.faces {
            for (a, b) in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])] {
                *edges.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
        assert!(edges.values().all(|&n| n == 2));
    }
}
