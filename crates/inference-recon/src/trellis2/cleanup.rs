//! CuMesh's hole filling and simplification on the host. Holes: boundary loops (boundary edges joined at manifold
//! boundary vertices) under a perimeter get a vertex at the mean of their edge midpoints and a triangle per edge.
//! Simplification: rounds of parallel edge collapses; each edge costs its quadric error plus length and skinny-triangle
//! terms, and collapses when it is the cheapest edge at every face around it and under a threshold that grows tenfold
//! whenever a round removes under 1% of the faces. Then the topology passes `to_glb` runs between simplifications.

use std::sync::atomic::{AtomicU64, Ordering};

use rayon::prelude::*;

use super::mesh::Mesh;
use super::vec3::{V3, add, cross, dot, norm2, scale, sub};

/// TRELLIS.2's hole perimeter limit
pub const MAX_HOLE_PERIMETER: f32 = 3e-2;
/// `to_glb`'s smallest kept connected piece (area)
pub const MIN_COMPONENT_AREA: f32 = 1e-5;
/// TRELLIS.2's face budget for the extracted mesh
pub const SIMPLIFY_TARGET: usize = 1_000_000;
pub(super) const START_THRESHOLD: f32 = 1e-8;
pub(super) const LAMBDA_EDGE_LENGTH: f32 = 1e-2;
pub(super) const LAMBDA_SKINNY: f32 = 1e-3;
// a round removing less than this share of the faces raises the threshold
pub(super) const SLOW_ROUND: f64 = 1e-2;
pub(super) const THRESHOLD_STEP: f32 = 10.0;
pub(super) const SKINNY_EPS: f32 = 1e-12;
// vertices a worker gathers edges for at a time
const EDGE_CHUNK: usize = 4096;

// edges and their face counts
type EdgeList = (Vec<(u32, u32)>, Vec<u32>);

/// Each vertex's faces, in face order (CuMesh fills them with atomics, in no fixed order).
struct VertFaces {
    offset: Vec<usize>,
    faces: Vec<u32>,
}

impl VertFaces {
    fn new(vertices: usize, faces: &[[u32; 3]]) -> Self {
        let mut count = vec![0usize; vertices + 1];
        for f in faces {
            for &v in f {
                count[v as usize + 1] += 1;
            }
        }
        for i in 0..vertices {
            count[i + 1] += count[i];
        }
        let offset = count.clone();
        let mut fill = count;
        let mut out = vec![0u32; faces.len() * 3];
        for (i, f) in faces.iter().enumerate() {
            for &v in f {
                out[fill[v as usize]] = i as u32;
                fill[v as usize] += 1;
            }
        }
        Self { offset, faces: out }
    }

    fn of(&self, v: u32) -> &[u32] {
        &self.faces[self.offset[v as usize]..self.offset[v as usize + 1]]
    }
}

/// [`edges`] from the vertices' faces: each vertex's higher neighbours, sorted, so no global sort is needed.
fn edges_by_vertex(vf: &VertFaces, faces: &[[u32; 3]]) -> EdgeList {
    let vertices = vf.offset.len() - 1;
    let chunks: Vec<EdgeList> = (0..vertices.div_ceil(EDGE_CHUNK))
        .into_par_iter()
        .map(|c| {
            let (mut edges, mut counts, mut higher) = (Vec::new(), Vec::new(), Vec::new());
            for v in (c * EDGE_CHUNK..((c + 1) * EDGE_CHUNK).min(vertices)).map(|v| v as u32) {
                higher.clear();
                higher.extend(
                    vf.of(v)
                        .iter()
                        .flat_map(|&f| faces[f as usize])
                        .filter(|&n| n > v),
                );
                higher.sort_unstable();
                for &n in &higher {
                    if edges.last() == Some(&(v, n)) {
                        *counts.last_mut().expect("a count beside each edge") += 1;
                    } else {
                        edges.push((v, n));
                        counts.push(1);
                    }
                }
            }
            (edges, counts)
        })
        .collect();
    let total = chunks.iter().map(|(e, _)| e.len()).sum();
    let (mut edges, mut counts) = (Vec::with_capacity(total), Vec::with_capacity(total));
    for (e, c) in chunks {
        edges.extend(e);
        counts.extend(c);
    }
    (edges, counts)
}

/// The unique edges `(low, high)` in sorted order, each with the number of faces using it.
fn edges(faces: &[[u32; 3]]) -> EdgeList {
    let mut all: Vec<u64> = faces
        .par_iter()
        .flat_map_iter(|f| {
            [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]
                .map(|(a, b)| (u64::from(a.min(b)) << 32) | u64::from(a.max(b)))
        })
        .collect();
    all.par_sort_unstable();
    let mut unique = Vec::new();
    let mut counts = Vec::new();
    for e in all {
        if unique.last() == Some(&e) {
            *counts.last_mut().expect("a count beside each edge") += 1;
        } else {
            unique.push(e);
            counts.push(1);
        }
    }
    (
        unique
            .into_iter()
            .map(|e| ((e >> 32) as u32, e as u32))
            .collect(),
        counts,
    )
}

/// Fills the holes whose boundary loop is shorter than `max_perimeter`.
pub fn fill_holes(mesh: &mut Mesh, max_perimeter: f32) {
    let v_count = mesh.vertices.len();
    let (edges, counts) = edges(&mesh.faces);
    let boundaries: Vec<usize> = (0..edges.len()).filter(|&e| counts[e] == 1).collect();
    if boundaries.is_empty() {
        return;
    }
    // per vertex: its boundary edges (as indices into `boundaries`) and whether it is manifold
    let mut vert_bounds: Vec<Vec<usize>> = vec![Vec::new(); v_count];
    for (b, &e) in boundaries.iter().enumerate() {
        vert_bounds[edges[e].0 as usize].push(b);
        vert_bounds[edges[e].1 as usize].push(b);
    }
    let mut manifold = vec![true; v_count];
    let mut bound_count = vec![0usize; v_count];
    for (e, &(a, b)) in edges.iter().enumerate() {
        for v in [a, b] {
            let v = v as usize;
            if counts[e] == 1 {
                bound_count[v] += 1;
                if bound_count[v] > 2 {
                    manifold[v] = false;
                }
            } else if counts[e] > 2 {
                manifold[v] = false;
            }
        }
    }
    // union the two boundary edges at each manifold boundary vertex
    let mut parent: Vec<usize> = (0..boundaries.len()).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    for (v, bs) in vert_bounds.iter().enumerate() {
        if manifold[v] && bs.len() == 2 {
            let (a, b) = (find(&mut parent, bs[0]), find(&mut parent, bs[1]));
            if a != b {
                parent[a.max(b)] = a.min(b);
            }
        }
    }
    let comp: Vec<usize> = (0..boundaries.len())
        .map(|b| find(&mut parent, b))
        .collect();
    // a component is a loop when every edge of it meets another of it at both ends
    let mut is_loop = vec![true; boundaries.len()];
    for (b, &e) in boundaries.iter().enumerate() {
        let (v0, v1) = edges[e];
        for v in [v0, v1] {
            let joined = vert_bounds[v as usize]
                .iter()
                .any(|&o| o != b && comp[o] == comp[b]);
            if !joined {
                is_loop[comp[b]] = false;
            }
        }
    }
    let mut loops: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
    for (b, &c) in comp.iter().enumerate() {
        if is_loop[c] {
            loops.entry(c).or_default().push(b);
        }
    }
    let at = |v: u32| mesh.vertices[v as usize];
    let mut new_vertices = Vec::new();
    let mut new_faces = Vec::new();
    for members in loops.values() {
        let perimeter: f32 = members
            .iter()
            .map(|&b| {
                let (a, c) = edges[boundaries[b]];
                norm2(sub(at(c), at(a))).sqrt()
            })
            .sum();
        if perimeter >= max_perimeter {
            continue;
        }
        let center = members.iter().fold([0f32; 3], |s, &b| {
            let (a, c) = edges[boundaries[b]];
            add(s, scale(add(at(a), at(c)), 0.5))
        });
        let id = (v_count + new_vertices.len()) as u32;
        new_vertices.push(scale(center, 1.0 / members.len() as f32));
        for &b in members {
            // CuMesh joins (high, low, center)
            let (low, high) = edges[boundaries[b]];
            new_faces.push([high, low, id]);
        }
    }
    mesh.vertices.extend(new_vertices);
    mesh.faces.extend(new_faces);
}

/// A quadric's upper triangle: 00 01 02 03 11 12 13 22 23 33.
#[derive(Clone, Copy, Default)]
struct Qem([f32; 10]);

impl Qem {
    fn add_plane(&mut self, [a, b, c, d]: [f32; 4]) {
        let e = &mut self.0;
        e[0] += a * a;
        e[1] += a * b;
        e[2] += a * c;
        e[3] += a * d;
        e[4] += b * b;
        e[5] += b * c;
        e[6] += b * d;
        e[7] += c * c;
        e[8] += c * d;
        e[9] += d * d;
    }

    fn plus(&self, o: &Qem) -> Qem {
        Qem(std::array::from_fn(|i| self.0[i] + o.0[i]))
    }

    fn evaluate(&self, [x, y, z]: V3) -> f32 {
        let e = &self.0;
        let mut r = 0.0f32;
        r += e[0] * x * x;
        r += 2.0 * e[1] * x * y;
        r += 2.0 * e[2] * x * z;
        r += 2.0 * e[3] * x;
        r += e[4] * y * y;
        r += 2.0 * e[5] * y * z;
        r += 2.0 * e[6] * y;
        r += e[7] * z * z;
        r += 2.0 * e[8] * z;
        r += e[9];
        r
    }
}

// a collapse's position: the boundary end when only one end is on the boundary, else the midpoint
fn collapse_point(v0: V3, v1: V3, b0: bool, b1: bool) -> V3 {
    let w0 = match (b0, b1) {
        (true, false) => 1.0,
        (false, true) => 0.0,
        _ => 0.5,
    };
    add(scale(v0, w0), scale(v1, 1.0 - w0))
}

/// One round of collapses.
fn simplify_round(mesh: &mut Mesh, threshold: f32) {
    let vf = VertFaces::new(mesh.vertices.len(), &mesh.faces);
    let (edges, counts) = edges_by_vertex(&vf, &mesh.faces);
    let mut boundary = vec![false; mesh.vertices.len()];
    for (e, &(a, b)) in edges.iter().enumerate() {
        if counts[e] == 1 {
            boundary[a as usize] = true;
            boundary[b as usize] = true;
        }
    }
    let verts = &mesh.vertices;
    let faces = &mesh.faces;
    let qems: Vec<Qem> = (0..verts.len() as u32)
        .into_par_iter()
        .map(|v| {
            let mut q = Qem::default();
            for &f in vf.of(v) {
                let [a, b, c] = faces[f as usize].map(|i| verts[i as usize]);
                let n = cross(sub(b, a), sub(c, a));
                let n = scale(n, 1.0 / norm2(n).sqrt());
                q.add_plane([n[0], n[1], n[2], -dot(n, a)]);
            }
            q
        })
        .collect();
    let costs: Vec<f32> = edges
        .par_iter()
        .map(|&(e0, e1)| {
            let (v0, v1) = (verts[e0 as usize], verts[e1 as usize]);
            let v = collapse_point(v0, v1, boundary[e0 as usize], boundary[e1 as usize]);
            let length2 = norm2(sub(v1, v0));
            let mut cost = qems[e0 as usize].plus(&qems[e1 as usize]).evaluate(v);
            cost += LAMBDA_EDGE_LENGTH * length2;
            // the skinny term only adds: an edge already over the threshold cannot collapse, and any value over it
            // compares the same against the edges that can (NaN and negative costs take the full path)
            if cost > threshold {
                return cost;
            }
            let (mut skinny, mut tris) = (0f32, 0usize);
            for (keep, other) in [(e0, e1), (e1, e0)] {
                for &f in vf.of(keep) {
                    let ids = faces[f as usize];
                    if ids.contains(&other) {
                        continue;
                    }
                    let [a, b, c] = ids.map(|i| verts[i as usize]);
                    let [na, nb, nc] =
                        [0, 1, 2].map(|k| if ids[k] == keep { v } else { [a, b, c][k] });
                    let old_n = cross(sub(b, a), sub(c, a));
                    let (e1n, e2n) = (sub(nb, na), sub(nc, na));
                    let new_n = cross(e1n, e2n);
                    if dot(old_n, new_n) < 0.0 {
                        return f32::INFINITY;
                    }
                    let area = 0.5 * norm2(new_n).sqrt();
                    let denom = (norm2(sub(nc, nb)) + norm2(e1n) + norm2(e2n)).max(SKINNY_EPS);
                    let shape = 4.0 * 3f32.sqrt() * area / denom;
                    skinny += 1.0 - shape.clamp(0.0, 1.0);
                    tris += 1;
                }
            }
            if tris > 0 {
                skinny /= tris as f32;
            }
            cost + LAMBDA_SKINNY * skinny * length2
        })
        .collect();
    let packed = |e: usize| (u64::from(costs[e].to_bits()) << 32) | e as u64;
    let best: Vec<AtomicU64> = (0..faces.len()).map(|_| AtomicU64::new(u64::MAX)).collect();
    edges.par_iter().enumerate().for_each(|(e, &(e0, e1))| {
        let p = packed(e);
        for v in [e0, e1] {
            for &f in vf.of(v) {
                best[f as usize].fetch_min(p, Ordering::Relaxed);
            }
        }
    });
    let winners: Vec<usize> = (0..edges.len())
        .into_par_iter()
        .filter(|&e| {
            // NaN costs pass this test, as on the device
            if costs[e] > threshold {
                return false;
            }
            let p = packed(e);
            let (e0, e1) = edges[e];
            [e0, e1].iter().all(|&v| {
                vf.of(v)
                    .iter()
                    .all(|&f| best[f as usize].load(Ordering::Relaxed) == p)
            })
        })
        .collect();
    let mut vertex_kept = vec![true; verts.len()];
    let mut face_kept = vec![true; faces.len()];
    let mut verts = mesh.vertices.clone();
    let mut faces = mesh.faces.clone();
    for e in winners {
        let (e0, e1) = edges[e];
        verts[e0 as usize] = collapse_point(
            verts[e0 as usize],
            verts[e1 as usize],
            boundary[e0 as usize],
            boundary[e1 as usize],
        );
        vertex_kept[e1 as usize] = false;
        for &f in vf.of(e0) {
            if faces[f as usize].contains(&e1) {
                face_kept[f as usize] = false;
            }
        }
        for &f in vf.of(e1) {
            if let Some(slot) = faces[f as usize].iter_mut().find(|i| **i == e1) {
                *slot = e0;
            }
        }
    }
    let mut remap = vec![u32::MAX; verts.len()];
    let mut new_verts = Vec::with_capacity(verts.len());
    for (v, p) in verts.iter().enumerate() {
        if vertex_kept[v] {
            remap[v] = new_verts.len() as u32;
            new_verts.push(*p);
        }
    }
    mesh.faces = faces
        .iter()
        .zip(&face_kept)
        .filter(|&(_, &k)| k)
        .map(|(f, _)| f.map(|i| remap[i as usize]))
        .collect();
    mesh.vertices = new_verts;
}

/// [`simplify`] on `device`: on the GPU for a CUDA device (same result), else on the host.
pub fn simplify_on(
    mesh: &mut Mesh,
    target: usize,
    device: &inference_tensor::Device,
) -> inference_tensor::Result<()> {
    #[cfg(feature = "cuda")]
    if let inference_tensor::Device::Cuda(dev) = device {
        return super::simplify_cuda::simplify(mesh, target, dev);
    }
    let _ = device;
    simplify(mesh, target);
    Ok(())
}

/// Collapses edges until at most `target` faces remain.
pub fn simplify(mesh: &mut Mesh, target: usize) {
    let mut threshold = START_THRESHOLD;
    let mut faces = mesh.faces.len();
    if faces <= target {
        return;
    }
    loop {
        simplify_round(mesh, threshold);
        let now = mesh.faces.len();
        if now <= target {
            break;
        }
        if ((faces - now) as f64) / (faces as f64) < SLOW_ROUND {
            threshold *= THRESHOLD_STEP;
        }
        faces = now;
    }
}

/// Drops vertices no face uses, keeping the others' order.
pub fn remove_unreferenced_vertices(mesh: &mut Mesh) {
    let mut map = vec![u32::MAX; mesh.vertices.len()];
    for &v in mesh.faces.iter().flatten() {
        map[v as usize] = 0;
    }
    let mut kept = 0u32;
    let mut vertices = Vec::new();
    for (v, m) in map.iter_mut().enumerate() {
        if *m == 0 {
            *m = kept;
            kept += 1;
            vertices.push(mesh.vertices[v]);
        }
    }
    mesh.vertices = vertices;
    for f in &mut mesh.faces {
        *f = f.map(|v| map[v as usize]);
    }
}

/// Keeps the first of each set of faces over the same three vertices (in any order).
pub fn remove_duplicate_faces(mesh: &mut Mesh) {
    let mut keys: Vec<([u32; 3], u32)> = mesh
        .faces
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let mut k = *f;
            k.sort_unstable();
            (k, i as u32)
        })
        .collect();
    keys.par_sort_unstable();
    let mut keep = vec![false; mesh.faces.len()];
    for (i, k) in keys.iter().enumerate() {
        if i == 0 || keys[i - 1].0 != k.0 {
            keep[k.1 as usize] = true;
        }
    }
    let mut i = 0;
    mesh.faces.retain(|_| {
        i += 1;
        keep[i - 1]
    });
    remove_unreferenced_vertices(mesh);
}

// union-find root, halving the path
fn find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        parent[x as usize] = parent[parent[x as usize] as usize];
        x = parent[x as usize];
    }
    x
}

fn union_low(parent: &mut [u32], a: u32, b: u32) {
    let (a, b) = (find(parent, a), find(parent, b));
    if a != b {
        parent[a.max(b) as usize] = a.min(b);
    }
}

fn corner(face: [u32; 3], v: u32) -> u32 {
    face.iter()
        .position(|&x| x == v)
        .expect("a vertex of the face") as u32
}

/// Splits vertices where faces meet only through non-manifold edges or a single point: face corners joined across
/// manifold edges become one vertex, numbered by their lowest corner.
pub fn repair_non_manifold_edges(mesh: &mut Mesh) {
    let adjacency = super::charts::manifold_adjacency(mesh);
    let faces = &mesh.faces;
    let mut parent: Vec<u32> = (0..faces.len() as u32 * 3).collect();
    for &(f0, f1, v0, v1) in &adjacency {
        let (t0, t1) = (faces[f0 as usize], faces[f1 as usize]);
        for v in [v0, v1] {
            union_low(&mut parent, 3 * f0 + corner(t0, v), 3 * f1 + corner(t1, v));
        }
    }
    let roots: Vec<u32> = (0..parent.len() as u32)
        .map(|c| find(&mut parent, c))
        .collect();
    let mut ids = vec![u32::MAX; roots.len()];
    let mut vertices = Vec::new();
    for (c, &r) in roots.iter().enumerate() {
        if r as usize == c {
            ids[c] = vertices.len() as u32;
            vertices.push(mesh.vertices[faces[c / 3][c % 3] as usize]);
        }
    }
    mesh.faces = (0..faces.len())
        .map(|f| [0, 1, 2].map(|k| ids[roots[3 * f + k] as usize]))
        .collect();
    mesh.vertices = vertices;
}

/// Drops the pieces (faces joined across manifold edges) whose area is under `min_area`.
pub fn remove_small_connected_components(mesh: &mut Mesh, min_area: f32) {
    let mut parent: Vec<u32> = (0..mesh.faces.len() as u32).collect();
    for &(f0, f1, _, _) in &super::charts::manifold_adjacency(mesh) {
        union_low(&mut parent, f0, f1);
    }
    let mut areas = vec![0f32; parent.len()];
    let roots: Vec<u32> = (0..parent.len() as u32)
        .map(|f| find(&mut parent, f))
        .collect();
    for (f, &r) in mesh.faces.iter().zip(&roots) {
        let [a, b, c] = f.map(|i| mesh.vertices[i as usize]);
        areas[r as usize] += norm2(cross(sub(b, a), sub(c, a))).sqrt() / 2.0;
    }
    let mut i = 0;
    mesh.faces.retain(|_| {
        i += 1;
        areas[roots[i - 1] as usize] >= min_area
    });
    remove_unreferenced_vertices(mesh);
}

// root and the parity of the path to it
fn find_parity(parent: &mut [(u32, bool)], mut x: u32) -> (u32, bool) {
    let mut flip = false;
    while parent[x as usize].0 != x {
        flip ^= parent[x as usize].1;
        x = parent[x as usize].0;
    }
    (x, flip)
}

/// Winds every piece like its lowest face: faces joined across a manifold edge that both run it the same way disagree,
/// and each face whose parity to its piece's lowest face is odd is flipped.
pub fn unify_face_orientations(mesh: &mut Mesh) {
    let faces = &mesh.faces;
    let mut parent: Vec<(u32, bool)> = (0..faces.len() as u32).map(|f| (f, false)).collect();
    for &(f0, f1, v0, v1) in &super::charts::manifold_adjacency(mesh) {
        let (t0, t1) = (faces[f0 as usize], faces[f1 as usize]);
        let dir = |t: [u32; 3]| (corner(t, v1) + 3 - corner(t, v0)) % 3;
        let flipped = dir(t0) == dir(t1);
        let ((r0, p0), (r1, p1)) = (find_parity(&mut parent, f0), find_parity(&mut parent, f1));
        if r0 != r1 {
            parent[r0.max(r1) as usize] = (r0.min(r1), flipped ^ p0 ^ p1);
        }
    }
    let flips: Vec<bool> = (0..faces.len() as u32)
        .map(|f| find_parity(&mut parent, f).1)
        .collect();
    for (f, flip) in mesh.faces.iter_mut().zip(flips) {
        if flip {
            *f = [f[0], f[2], f[1]];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // a w x h grid of quads in the z = 0 plane, split into triangles
    fn grid(w: usize, h: usize) -> Mesh {
        let mut mesh = Mesh::default();
        for y in 0..=h {
            for x in 0..=w {
                mesh.vertices
                    .push([x as f32 / w as f32, y as f32 / h as f32, 0.0]);
            }
        }
        let at = |x: usize, y: usize| (y * (w + 1) + x) as u32;
        for y in 0..h {
            for x in 0..w {
                mesh.faces.push([at(x, y), at(x + 1, y), at(x + 1, y + 1)]);
                mesh.faces.push([at(x, y), at(x + 1, y + 1), at(x, y + 1)]);
            }
        }
        mesh
    }

    #[test]
    fn a_small_hole_is_closed_and_a_large_one_left() {
        let mut mesh = grid(4, 4);
        // drop the two triangles of the middle quad (1, 1)
        mesh.faces.retain(|f| !f.contains(&6) || !f.contains(&12));
        let before = mesh.faces.len();
        let mut big = mesh.clone();
        // the inner hole's perimeter is 1, the grid's outer boundary 4
        fill_holes(&mut mesh, 2.0);
        assert_eq!(mesh.faces.len(), before + 4);
        assert_eq!(mesh.vertices.len(), 26);
        // the new vertex sits at the hole's center
        assert_eq!(mesh.vertices[25], [0.375, 0.375, 0.0]);
        fill_holes(&mut big, 0.5);
        assert_eq!(big.faces.len(), before);
    }

    #[test]
    fn a_flat_grid_simplifies_to_its_target_and_stays_flat() {
        let mut mesh = grid(20, 20);
        simplify(&mut mesh, 200);
        assert!(mesh.faces.len() <= 200);
        assert!(mesh.vertices.iter().all(|v| v[2] == 0.0));
    }

    #[test]
    fn topology_passes_fix_duplicates_fans_specks_and_winding() {
        let mut mesh = grid(2, 2);
        let first = mesh.faces[0];
        mesh.faces.push([first[1], first[2], first[0]]);
        remove_duplicate_faces(&mut mesh);
        assert_eq!(mesh.faces.len(), 8);
        // flip one face: it is wound back to match its piece
        let wound = mesh.faces[3];
        mesh.faces[3] = [wound[0], wound[2], wound[1]];
        unify_face_orientations(&mut mesh);
        assert_eq!(mesh.faces[3], wound);
        // a third face on the inner diagonal makes it non-manifold, so that face's corners get vertices of their own
        let extra = mesh.vertices.len() as u32;
        mesh.vertices.push([0.25, 0.25, 1.0]);
        mesh.faces.push([first[0], first[2], extra]);
        let before = mesh.vertices.len();
        repair_non_manifold_edges(&mut mesh);
        assert!(mesh.vertices.len() > before);
        // a speck far away goes, the grid stays
        let n = mesh.vertices.len() as u32;
        mesh.vertices
            .extend([[5.0, 5.0, 5.0], [5.001, 5.0, 5.0], [5.0, 5.001, 5.0]]);
        mesh.faces.push([n, n + 1, n + 2]);
        remove_small_connected_components(&mut mesh, MIN_COMPONENT_AREA);
        assert!(mesh.vertices.iter().all(|v| v[0] < 5.0));
        assert!(mesh.faces.len() >= 8);
    }
}
