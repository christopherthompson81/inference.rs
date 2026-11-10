//! CuMesh's fast chart clustering (`compute_charts`, as `to_glb` calls it: one global iteration, no refinement), the
//! clusters xatlas then charts and packs. Every face starts as its own chart; rounds of parallel merges join two
//! adjacent charts (across manifold edges) when their merged normal cone stays within the threshold and their pair is
//! the cheapest at both, the cost being the merged cone's half-angle plus area and perimeter^2/area penalties. Charts
//! are finally split into their connected pieces.

use rayon::prelude::*;

use super::mesh::Mesh;
use super::vec3::{V3, cross, dot, norm, sub};

/// `to_glb`'s defaults
pub const CONE_HALF_ANGLE: f32 = std::f32::consts::FRAC_PI_2;
pub const AREA_PENALTY: f32 = 0.1;
pub const PERIMETER_AREA_PENALTY: f32 = 1e-4;
// `remove_degenerate_faces` defaults
const DEGENERATE_ABS: f32 = 1e-24;
const DEGENERATE_REL: f32 = 1e-12;

/// Drops faces with a repeated vertex, or whose area is under `min(1e-12 * longest edge^2, 1e-24)`.
pub fn remove_degenerate_faces(mesh: &mut Mesh) {
    let v = &mesh.vertices;
    mesh.faces.retain(|f| {
        if f[0] == f[1] || f[1] == f[2] || f[2] == f[0] {
            return false;
        }
        let [a, b, c] = f.map(|i| v[i as usize]);
        let (e0, e1, e2) = (sub(b, a), sub(c, b), sub(a, c));
        let longest = norm(e0).max(norm(e1)).max(norm(e2));
        let area = norm(cross(e0, e1)) / 2.0;
        area >= (DEGENERATE_REL * longest * longest).min(DEGENERATE_ABS)
    });
}

/// Pairs of faces sharing an edge that exactly two faces use, with that edge's two vertices.
pub(super) fn manifold_adjacency(mesh: &Mesh) -> Vec<(u32, u32, u32, u32)> {
    let mut edges: Vec<(u32, u32, u32)> = mesh
        .faces
        .par_iter()
        .enumerate()
        .flat_map_iter(|(f, t)| {
            [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])].map(|(a, b)| (a.min(b), a.max(b), f as u32))
        })
        .collect();
    edges.par_sort_unstable();
    let mut out = Vec::new();
    let mut i = 0;
    while i < edges.len() {
        let mut j = i + 1;
        while j < edges.len() && edges[j].0 == edges[i].0 && edges[j].1 == edges[i].1 {
            j += 1;
        }
        if j - i == 2 {
            out.push((edges[i].2, edges[i + 1].2, edges[i].0, edges[i].1));
        }
        i = j;
    }
    out
}

// ids renumbered densely in ascending order
fn compress(ids: &mut [u32]) -> usize {
    let mut unique: Vec<u32> = ids.to_vec();
    unique.par_sort_unstable();
    unique.dedup();
    ids.par_iter_mut()
        .for_each(|id| *id = unique.binary_search(id).expect("an id it holds") as u32);
    unique.len()
}

// the merged cone of two (axis, half-angle) cones as (low, high) angles around the first's axis
fn merged_extent(a: (V3, f32), b: (V3, f32)) -> (f32, f32) {
    let angle = dot(a.0, b.0).clamp(-1.0, 1.0).acos();
    ((-a.1).min(angle - b.1), a.1.max(angle + b.1))
}

/// Each face's chart: the clusters `to_glb` hands xatlas, numbered from 0 in ascending order of their lowest face.
pub fn compute_charts(
    mesh: &Mesh,
    threshold: f32,
    area_penalty: f32,
    perimeter_penalty: f32,
) -> Vec<u32> {
    let faces = &mesh.faces;
    let verts = &mesh.vertices;
    let f_count = faces.len();
    let crosses: Vec<V3> = faces
        .par_iter()
        .map(|f| {
            let [a, b, c] = f.map(|i| verts[i as usize]);
            cross(sub(b, a), sub(c, a))
        })
        .collect();
    let areas: Vec<f32> = crosses.iter().map(|c| 0.5 * norm(*c)).collect();
    let normals: Vec<V3> = crosses
        .iter()
        .map(|c| {
            let inv = 1.0 / norm(*c);
            c.map(|v| v * inv)
        })
        .collect();
    let adjacency = manifold_adjacency(mesh);
    let mut charts: Vec<u32> = (0..f_count as u32).collect();
    let mut count = f_count;
    loop {
        // chart pairs across manifold edges, their shared boundary summed
        let mut pairs: Vec<(u64, f32)> = adjacency
            .par_iter()
            .filter_map(|&(f0, f1, v0, v1)| {
                let (c0, c1) = (charts[f0 as usize], charts[f1 as usize]);
                (c0 != c1).then(|| {
                    let key = (u64::from(c0.min(c1)) << 32) | u64::from(c0.max(c1));
                    (key, norm(sub(verts[v0 as usize], verts[v1 as usize])))
                })
            })
            .collect();
        if pairs.is_empty() {
            break;
        }
        pairs.par_sort_unstable_by_key(|p| p.0);
        let mut adj: Vec<(u32, u32)> = Vec::new();
        let mut lengths: Vec<f32> = Vec::new();
        for (key, l) in pairs {
            let pair = ((key >> 32) as u32, key as u32);
            if adj.last() == Some(&pair) {
                *lengths.last_mut().expect("a length beside each pair") += l;
            } else {
                adj.push(pair);
                lengths.push(l);
            }
        }
        let mut perims = vec![0f32; count];
        let mut chart_edges: Vec<Vec<u32>> = vec![Vec::new(); count];
        for (e, (&(c0, c1), &l)) in adj.iter().zip(&lengths).enumerate() {
            perims[c0 as usize] += l;
            perims[c1 as usize] += l;
            chart_edges[c0 as usize].push(e as u32);
            chart_edges[c1 as usize].push(e as u32);
        }
        // cones: the normalized sum of the faces' normals, and the widest angle to one of them
        let mut sums = vec![[0f32; 3]; count];
        let mut chart_areas = vec![0f32; count];
        for f in 0..f_count {
            let c = charts[f] as usize;
            for k in 0..3 {
                sums[c][k] += normals[f][k];
            }
            chart_areas[c] += areas[f];
        }
        let axes: Vec<V3> = sums
            .iter()
            .map(|s| {
                let n = norm(*s);
                if n > 0.0 { s.map(|v| v / n) } else { *s }
            })
            .collect();
        let mut half = vec![0f32; count];
        for f in 0..f_count {
            let c = charts[f] as usize;
            half[c] = half[c].max(dot(axes[c], normals[f]).clamp(-1.0, 1.0).acos());
        }
        let costs: Vec<f32> = adj
            .par_iter()
            .zip(&lengths)
            .map(|(&(c0, c1), &l)| {
                let (c0, c1) = (c0 as usize, c1 as usize);
                let (low, high) = merged_extent((axes[c0], half[c0]), (axes[c1], half[c1]));
                let area = chart_areas[c0] + chart_areas[c1];
                let perim = perims[c0] + perims[c1] - 2.0 * l;
                (high - low) * 0.5
                    + area_penalty * area
                    + perimeter_penalty * (perim * perim / area)
            })
            .collect();
        let pack = |e: u32, cost: f32| (u64::from(cost.to_bits()) << 32) | u64::from(e);
        // each chart's cheapest pair, ties to the lower edge
        let best: Vec<u64> = chart_edges
            .par_iter()
            .map(|edges| {
                let mut min = (u32::MAX, f32::MAX);
                for &e in edges {
                    let c = costs[e as usize];
                    if c < min.1 || (c == min.1 && e < min.0) {
                        min = (e, c);
                    }
                }
                pack(min.0, min.1)
            })
            .collect();
        let mut map: Vec<u32> = (0..count as u32).collect();
        let mut merged = false;
        for (e, &(c0, c1)) in adj.iter().enumerate() {
            let cost = costs[e];
            if cost > threshold {
                continue;
            }
            let p = pack(e as u32, cost);
            if best[c0 as usize] < p || best[c1 as usize] < p {
                continue;
            }
            map[c1 as usize] = c0;
            merged = true;
        }
        if !merged {
            break;
        }
        count = compress(&mut map);
        charts.par_iter_mut().for_each(|c| *c = map[*c as usize]);
    }
    // charts split into connected pieces (across manifold edges), each named by its lowest face
    let mut parent: Vec<u32> = (0..f_count as u32).collect();
    fn find(parent: &mut [u32], mut x: u32) -> u32 {
        while parent[x as usize] != x {
            parent[x as usize] = parent[parent[x as usize] as usize];
            x = parent[x as usize];
        }
        x
    }
    for &(f0, f1, _, _) in &adjacency {
        if charts[f0 as usize] == charts[f1 as usize] {
            let (a, b) = (find(&mut parent, f0), find(&mut parent, f1));
            if a != b {
                parent[a.max(b) as usize] = a.min(b);
            }
        }
    }
    let mut ids: Vec<u32> = (0..f_count as u32).map(|f| find(&mut parent, f)).collect();
    compress(&mut ids);
    ids
}

/// The charts as separate meshes: each chart's vertices (ascending source index) and its triangles over them, charts
/// in id order and their faces in mesh order.
pub fn chart_meshes(mesh: &Mesh, charts: &[u32]) -> Vec<(Vec<u32>, Vec<[u32; 3]>)> {
    let count = charts.iter().map(|&c| c as usize + 1).max().unwrap_or(0);
    let mut faces: Vec<Vec<[u32; 3]>> = vec![Vec::new(); count];
    for (f, &c) in mesh.faces.iter().zip(charts) {
        faces[c as usize].push(*f);
    }
    faces
        .into_par_iter()
        .map(|tris| {
            let mut vmap: Vec<u32> = tris.iter().flatten().copied().collect();
            vmap.sort_unstable();
            vmap.dedup();
            let local = tris
                .iter()
                .map(|t| t.map(|v| vmap.binary_search(&v).expect("a vertex of the chart") as u32))
                .collect();
            (vmap, local)
        })
        .collect()
}

/// CuMesh's `compute_vertex_normals`: the sum of the faces' cross products, normalized, or the first face's cross
/// product (unnormalized) when the sum vanishes.
pub fn vertex_normals(mesh: &Mesh) -> Vec<V3> {
    let mut sums = vec![[0f32; 3]; mesh.vertices.len()];
    let mut first: Vec<Option<V3>> = vec![None; mesh.vertices.len()];
    for f in &mesh.faces {
        let [a, b, c] = f.map(|i| mesh.vertices[i as usize]);
        let n = cross(sub(b, a), sub(c, a));
        for &v in f {
            let v = v as usize;
            for k in 0..3 {
                sums[v][k] += n[k];
            }
            first[v].get_or_insert(n);
        }
    }
    sums.iter()
        .zip(first)
        .map(|(s, first)| {
            let inv = 1.0 / norm(*s);
            let n = s.map(|x| x * inv);
            if n[0].is_nan() { first.unwrap_or(n) } else { n }
        })
        .collect()
}

/// A mesh cut along its atlas seams: each vertex's uv and source vertex, the atlas size in texels.
#[derive(Clone, Debug)]
pub struct Unwrapped {
    pub mesh: Mesh,
    pub uvs: Vec<[f32; 2]>,
    /// Each vertex's normal: CuMesh's, the normalized sum of its faces' area-weighted normals
    pub normals: Vec<V3>,
    pub source: Vec<u32>,
    pub width: u32,
    pub height: u32,
}

/// CuMesh's `uv_unwrap`: degenerate faces dropped, the clusters of [`compute_charts`] each added to xatlas as a mesh,
/// charted, parameterized and packed with its defaults, the results gathered in cluster order.
pub fn uv_unwrap(mesh: &Mesh) -> Result<Unwrapped, inference_xatlas::Error> {
    uv_unwrap_with(mesh, CONE_HALF_ANGLE, inference_xatlas::Options::default())
}

/// [`uv_unwrap`] with xatlas `options`.
pub fn uv_unwrap_with(
    mesh: &Mesh,
    cone_half_angle: f32,
    options: inference_xatlas::Options,
) -> Result<Unwrapped, inference_xatlas::Error> {
    let mut mesh = mesh.clone();
    remove_degenerate_faces(&mut mesh);
    let charts = compute_charts(&mesh, cone_half_angle, AREA_PENALTY, PERIMETER_AREA_PENALTY);
    let clusters = chart_meshes(&mesh, &charts);
    let positions: Vec<Vec<V3>> = clusters
        .iter()
        .map(|(vmap, _)| vmap.iter().map(|&v| mesh.vertices[v as usize]).collect())
        .collect();
    let inputs: Vec<inference_xatlas::MeshInput> = clusters
        .iter()
        .zip(&positions)
        .map(|((_, tris), positions)| inference_xatlas::MeshInput {
            positions,
            triangles: tris,
        })
        .collect();
    let atlas = inference_xatlas::unwrap_with(&inputs, options)?;
    let mut out = Unwrapped {
        mesh: Mesh {
            vertices: Vec::new(),
            faces: Vec::new(),
        },
        uvs: Vec::new(),
        normals: Vec::new(),
        source: Vec::new(),
        width: atlas.width,
        height: atlas.height,
    };
    for ((vmap, _), part) in clusters.iter().zip(&atlas.meshes) {
        let base = out.source.len() as u32;
        out.source
            .extend(part.xref.iter().map(|&x| vmap[x as usize]));
        out.uvs.extend_from_slice(&part.uvs);
        out.mesh
            .faces
            .extend(part.triangles.iter().map(|t| t.map(|v| v + base)));
    }
    out.mesh.vertices = out
        .source
        .iter()
        .map(|&v| mesh.vertices[v as usize])
        .collect();
    let normals = vertex_normals(&mesh);
    out.normals = out.source.iter().map(|&v| normals[v as usize]).collect();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cube_splits_where_its_faces_turn() {
        let vertices: Vec<V3> = (0..8)
            .map(|i| [(i & 1) as f32, ((i >> 1) & 1) as f32, ((i >> 2) & 1) as f32])
            .collect();
        let quads = [
            [0, 2, 3, 1],
            [4, 5, 7, 6],
            [0, 1, 5, 4],
            [2, 6, 7, 3],
            [0, 4, 6, 2],
            [1, 3, 7, 5],
        ];
        let faces: Vec<[u32; 3]> = quads
            .iter()
            .flat_map(|q: &[u32; 4]| [[q[0], q[1], q[2]], [q[0], q[2], q[3]]])
            .collect();
        let mesh = Mesh { vertices, faces };
        // a tight cone keeps the six sides apart; each side's two triangles join
        let charts = compute_charts(&mesh, 0.1, 0.0, 0.0);
        assert_eq!(charts.iter().max(), Some(&5));
        for side in 0..6 {
            assert_eq!(charts[2 * side], charts[2 * side + 1]);
        }
        // a 90 degree cone joins neighbouring sides
        let wide = compute_charts(&mesh, CONE_HALF_ANGLE, 0.0, 0.0);
        assert!(wide.iter().max() < Some(&5));
        let pieces = chart_meshes(&mesh, &charts);
        assert_eq!(pieces.len(), 6);
        assert!(
            pieces
                .iter()
                .all(|(vmap, tris)| vmap.len() == 4 && tris.len() == 2)
        );
    }
}
