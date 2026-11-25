//! `point_to_mesh` and `fill_mesh_holes`: the decoded cells' corner values averaged onto the grid's vertices, moved by
//! their deformations, meshed by FlexiCubes, and the boundary loops ear-clipped shut.

use std::collections::HashMap;

use inference_tensor::{D, Result};

use super::flexicubes;
use super::vae::{Decoded, SIGN_CHANNELS};
use crate::trellis2::mesh::Mesh;

const CORNERS: usize = 8;
// the deformations' denormalization
const DEFORM_SCALE: f32 = 0.5;
// the decoded cells' normalization: centroid and largest radius (plus this) of the structure's cells
const RADIUS_EPS: f32 = 1e-8;
// boundary loops longer than this stay open
const MAX_HOLE_EDGES: usize = 10_000;
// the ear clipper's guard against a zero-length edge
const TANGENT_EPS: f32 = 1e-6;
const KEY_BITS: u32 = 21;

fn key([x, y, z]: [i32; 3]) -> u64 {
    ((x as u64) << (2 * KEY_BITS)) | ((y as u64) << KEY_BITS) | z as u64
}

/// The decoded surface, in the units the structure's cells (`structure`, the latents' coordinates) normalize to:
/// their centroid at the origin and their farthest cell at radius 1.
pub fn point_to_mesh(decoded: &Decoded, structure: &[[i32; 3]]) -> Result<Mesh> {
    let n = decoded.coords.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| decoded.coords[i]);
    let signs = decoded.signs.to_vec2::<f32>()?;
    let attrs = decoded.attrs.to_vec2::<f32>()?;
    debug_assert_eq!(decoded.signs.dim(D::Minus1)?, SIGN_CHANNELS);

    // the grid's vertices (sorted, as torch.unique) and each cube's corners among them
    let mut corners: Vec<[i32; 3]> = Vec::with_capacity(n * CORNERS);
    for &i in &order {
        let c = decoded.coords[i];
        for k in 0..CORNERS {
            corners.push([
                c[0] + (k & 1) as i32,
                c[1] + ((k >> 1) & 1) as i32,
                c[2] + ((k >> 2) & 1) as i32,
            ]);
        }
    }
    let mut unique = corners.clone();
    unique.sort_unstable();
    unique.dedup();
    let index: HashMap<u64, u32> = unique
        .iter()
        .enumerate()
        .map(|(i, &c)| (key(c), i as u32))
        .collect();
    let cubes: Vec<[u32; 8]> = corners
        .chunks_exact(CORNERS)
        .map(|c| std::array::from_fn(|k| index[&key(c[k])]))
        .collect();

    // per vertex: the mean sign probability, distance and deformation over the corners that land on it
    let mut sums = vec![[0f32; 5]; unique.len()];
    let mut hits = vec![0u32; unique.len()];
    for (row, &i) in order.iter().enumerate() {
        for k in 0..CORNERS {
            let v = cubes[row][k] as usize;
            let a = &attrs[i];
            let values = [
                1.0 / (1.0 + (-signs[i][k]).exp()),
                a[k],
                a[CORNERS + 3 * k] * DEFORM_SCALE,
                a[CORNERS + 3 * k + 1] * DEFORM_SCALE,
                a[CORNERS + 3 * k + 2] * DEFORM_SCALE,
            ];
            for (s, x) in sums[v].iter_mut().zip(values) {
                *s += x;
            }
            hits[v] += 1;
        }
    }

    let count = structure.len() as f32;
    let mut center = [0f32; 3];
    for c in structure {
        for d in 0..3 {
            center[d] += c[d] as f32;
        }
    }
    let center = center.map(|v| v / count);
    let radius = structure
        .iter()
        .map(|c| {
            (0..3)
                .map(|d| (c[d] as f32 - center[d]).powi(2))
                .sum::<f32>()
                .sqrt()
        })
        .fold(0f32, f32::max)
        + RADIUS_EPS;

    let mut positions = Vec::with_capacity(unique.len());
    let mut scalar = Vec::with_capacity(unique.len());
    for (v, c) in unique.iter().enumerate() {
        let m = sums[v].map(|s| s / hits[v] as f32);
        positions.push([0, 1, 2].map(|d| (c[d] as f32 + m[2 + d] - center[d]) / radius));
        scalar.push(if m[0] >= 0.5 { m[1].abs() } else { -m[1].abs() });
    }
    let sorted: Vec<[i32; 3]> = order.iter().map(|&i| decoded.coords[i]).collect();
    let (vertices, faces) =
        flexicubes::extract(&positions, &scalar, &cubes, &sorted, decoded.resolution);
    Ok(Mesh { vertices, faces })
}

/// `fill_mesh_holes`: each boundary loop of at most 10,000 edges ear-clipped, its triangles appended.
pub fn fill_holes(mesh: &mut Mesh) {
    let mut count: HashMap<[u32; 2], u32> = HashMap::new();
    for f in &mesh.faces {
        for (a, b) in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])] {
            *count.entry([a.min(b), a.max(b)]).or_default() += 1;
        }
    }
    // the boundary's directed edges, in the reference's order (all faces' first edges, then second, then third)
    let mut boundary: Vec<[u32; 2]> = Vec::new();
    for slot in 0..3 {
        for f in &mesh.faces {
            let (a, b) = (f[slot], f[(slot + 1) % 3]);
            if count[&[a.min(b), a.max(b)]] == 1 {
                boundary.push([a, b]);
            }
        }
    }
    let mut new_faces = Vec::new();
    for lp in boundary_loops(&boundary) {
        if (3..=MAX_HOLE_EDGES).contains(&lp.len()) {
            new_faces.extend(ear_clip(&mesh.vertices, &lp));
        }
    }
    mesh.faces.extend(new_faces);
}

fn boundary_loops(boundary: &[[u32; 2]]) -> Vec<Vec<u32>> {
    let mut outgoing: HashMap<u32, Vec<usize>> = HashMap::new();
    for (i, e) in boundary.iter().enumerate() {
        outgoing.entry(e[0]).or_default().push(i);
    }
    let mut visited = vec![false; boundary.len()];
    let mut loops = Vec::new();
    for i in 0..boundary.len() {
        if visited[i] {
            continue;
        }
        visited[i] = true;
        let start = boundary[i][0];
        let mut lp = vec![start, boundary[i][1]];
        while *lp.last().unwrap() != start {
            let last = *lp.last().unwrap();
            let next = outgoing
                .get(&last)
                .and_then(|out| out.iter().copied().find(|&j| !visited[j]));
            match next {
                Some(j) => {
                    visited[j] = true;
                    lp.push(boundary[j][1]);
                }
                None => break,
            }
        }
        if lp[0] == *lp.last().unwrap() && lp.len() > 3 {
            lp.pop();
            loops.push(lp);
        }
    }
    loops
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: [f32; 3]) -> f32 {
    dot(a, a).sqrt()
}

// `_ear_clip`: repeatedly cut the vertex with the smallest positive turning angle (a collapsed one first)
fn ear_clip(vertices: &[[f32; 3]], boundary: &[u32]) -> Vec<[u32; 3]> {
    let mut idx: Vec<u32> = boundary.to_vec();
    let mut pts: Vec<[f32; 3]> = idx.iter().map(|&v| vertices[v as usize]).collect();
    let mut faces = Vec::with_capacity(idx.len());
    loop {
        let n = idx.len();
        if n == 3 {
            faces.push([idx[2], idx[1], idx[0]]);
            break;
        }
        let tangent: Vec<[f32; 3]> = (0..n)
            .map(|i| {
                let t = sub(pts[(i + 1) % n], pts[i]);
                let l = norm(t) + TANGENT_EPS;
                t.map(|v| v / l)
            })
            .collect();
        let mut center = [0f32; 3];
        for p in &pts {
            for d in 0..3 {
                center[d] += p[d];
            }
        }
        let center = center.map(|v| v / n as f32);
        let mut best = (f32::NEG_INFINITY, 0usize);
        for i in 0..n {
            let t = tangent[i];
            let t_prev = tangent[(i + n - 1) % n];
            let normal = cross(t_prev.map(|v| -v), t);
            let p = sub(pts[(i + 1) % n], center);
            let q = sub(pts[i], center);
            let sin_sign = dot(cross(p, q), normal).signum_or_zero();
            let sin = norm(normal) * sin_sign;
            let cos = -dot(t_prev, t);
            let angle = sin.atan2(cos);
            let score = if angle == 0.0 {
                f32::INFINITY
            } else {
                1.0 / angle
            };
            if score > best.0 {
                best = (score, i);
            }
        }
        let ear = best.1;
        faces.push([idx[(ear + 1) % n], idx[ear], idx[(ear + n - 1) % n]]);
        idx.remove(ear);
        pts.remove(ear);
    }
    faces
}

trait SignumOrZero {
    fn signum_or_zero(self) -> Self;
}

impl SignumOrZero for f32 {
    // np.sign: zero at zero
    fn signum_or_zero(self) -> f32 {
        if self > 0.0 {
            1.0
        } else if self < 0.0 {
            -1.0
        } else {
            0.0
        }
    }
}
