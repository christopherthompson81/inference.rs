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
// the unwrap's clusters: tight enough that most flatten as one chart without overlaps
pub const UNWRAP_CONE_HALF_ANGLE: f32 = std::f32::consts::FRAC_PI_6;
// the most faces one chart may hand xatlas (its least-squares parameterization grows superlinearly; the robot's largest
// 30 degree cluster is ~12.6K faces)
pub const MAX_SINGLE_CHART_FACES: usize = 20_000;
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

// a round merging fewer than one in this many charts hands over to `merge_incrementally`
const TRICKLE_FRACTION: usize = 50;

struct ConeCost {
    area_penalty: f32,
    perimeter_penalty: f32,
    threshold: f32,
}

impl ConeCost {
    // the pair's merged cone half-angle plus the area and perimeter^2/area penalties, as a round prices it
    fn of(&self, (a0, h0): (V3, f32), (a1, h1): (V3, f32), area: f32, perim: f32) -> f32 {
        let (low, high) = merged_extent((a0, h0), (a1, h1));
        (high - low) * 0.5
            + self.area_penalty * area
            + self.perimeter_penalty * (perim * perim / area)
    }
}

// a round's view of the charts: neighbouring pairs (lower id first, ascending) with their shared boundary, and each
// chart's boundary to other charts, normal sum, area, cone axis and half-angle
struct Graph {
    adj: Vec<(u32, u32)>,
    lengths: Vec<f32>,
    perims: Vec<f32>,
    sums: Vec<V3>,
    areas: Vec<f32>,
    axes: Vec<V3>,
    half: Vec<f32>,
}

// `compute_charts`' rounds carried on from `graph` over `charts`, recomputing only the charts a round merged and their
// neighbours: the same rule (a pair merges when it is each chart's cheapest, ties to the lower pair, under the
// threshold), so a tail of one merge a round costs that merge's neighbourhood rather than a pass over the mesh
fn merge_incrementally(
    charts: &[u32],
    normals: &[V3],
    cone: &ConeCost,
    graph: Graph,
    stop: &dyn Fn() -> bool,
) -> Vec<u32> {
    let Graph {
        adj,
        lengths,
        mut perims,
        mut sums,
        mut areas,
        mut axes,
        mut half,
    } = graph;
    let count = perims.len();
    let mut faces_of: Vec<Vec<u32>> = vec![Vec::new(); count];
    for (f, &c) in charts.iter().enumerate() {
        faces_of[c as usize].push(f as u32);
    }
    let mut nbrs: Vec<Vec<(u32, f32)>> = vec![Vec::new(); count];
    for (&(c0, c1), &l) in adj.iter().zip(&lengths) {
        nbrs[c0 as usize].push((c1, l));
        nbrs[c1 as usize].push((c0, l));
    }
    let cost =
        |c: usize, d: usize, l: f32, perims: &[f32], areas: &[f32], axes: &[V3], half: &[f32]| {
            let area = areas[c] + areas[d];
            cone.of(
                (axes[c], half[c]),
                (axes[d], half[d]),
                area,
                perims[c] + perims[d] - 2.0 * l,
            )
        };
    // each chart's cheapest neighbour: (cost, lower id, higher id), so ties go to the lower pair
    let cheapest = |c: usize,
                    nbrs: &[Vec<(u32, f32)>],
                    perims: &[f32],
                    areas: &[f32],
                    axes: &[V3],
                    half: &[f32]| {
        nbrs[c]
            .iter()
            .map(|&(d, l)| {
                let k = cost(c, d as usize, l, perims, areas, axes, half);
                (k, (c as u32).min(d), (c as u32).max(d))
            })
            .min_by(|a, b| a.0.total_cmp(&b.0).then((a.1, a.2).cmp(&(b.1, b.2))))
    };
    let mut best: Vec<Option<(f32, u32, u32)>> = (0..count)
        .map(|c| cheapest(c, &nbrs, &perims, &areas, &axes, &half))
        .collect();
    // each face's chart as the merges go
    let mut owner: Vec<u32> = charts.to_vec();
    loop {
        if stop() {
            break;
        }
        let pairs: Vec<(u32, u32, f32)> = (0..count)
            .filter_map(|c| {
                let (k, lo, hi) = best[c]?;
                (lo == c as u32
                    && k <= cone.threshold
                    && best[hi as usize].map(|b| (b.1, b.2)) == Some((lo, hi)))
                .then(|| {
                    let l = nbrs[c]
                        .iter()
                        .find(|&&(d, _)| d == hi)
                        .map_or(0.0, |&(_, l)| l);
                    (lo, hi, l)
                })
            })
            .collect();
        if pairs.is_empty() {
            break;
        }
        let mut touched = Vec::new();
        for &(c, d, l) in &pairs {
            let (c, d) = (c as usize, d as usize);
            let moved = std::mem::take(&mut faces_of[d]);
            for &f in &moved {
                owner[f as usize] = c as u32;
            }
            faces_of[c].extend(moved);
            let moved_sum = sums[d];
            for (total, add) in sums[c].iter_mut().zip(moved_sum) {
                *total += add;
            }
            areas[c] += areas[d];
            perims[c] += perims[d] - 2.0 * l;
            // d's neighbours become c's, shared boundaries summed
            let gone = std::mem::take(&mut nbrs[d]);
            nbrs[c].retain(|&(n, _)| n != d as u32);
            for (n, ln) in gone {
                if n == c as u32 {
                    continue;
                }
                match nbrs[c].iter_mut().find(|e| e.0 == n) {
                    Some(e) => e.1 += ln,
                    None => nbrs[c].push((n, ln)),
                }
                let list = &mut nbrs[n as usize];
                let at_d = list
                    .iter()
                    .position(|e| e.0 == d as u32)
                    .expect("symmetric");
                let (_, ld) = list.swap_remove(at_d);
                match list.iter_mut().find(|e| e.0 == c as u32) {
                    Some(e) => e.1 += ld,
                    None => list.push((c as u32, ld)),
                }
            }
            best[d] = None;
            touched.push(c);
        }
        for &c in &touched {
            let s = sums[c];
            let n = norm(s);
            axes[c] = if n > 0.0 { s.map(|v| v / n) } else { s };
            half[c] = faces_of[c]
                .iter()
                .map(|&f| dot(axes[c], normals[f as usize]).clamp(-1.0, 1.0).acos())
                .fold(0.0, f32::max);
        }
        let mut stale: Vec<usize> = touched
            .iter()
            .flat_map(|&c| std::iter::once(c).chain(nbrs[c].iter().map(|&(n, _)| n as usize)))
            .collect();
        stale.sort_unstable();
        stale.dedup();
        for c in stale {
            best[c] = cheapest(c, &nbrs, &perims, &areas, &axes, &half);
        }
    }
    owner
}

/// Each face's chart: the clusters `to_glb` hands xatlas, numbered from 0 in ascending order of their lowest face.
pub fn compute_charts(
    mesh: &Mesh,
    threshold: f32,
    area_penalty: f32,
    perimeter_penalty: f32,
    stop: &dyn Fn() -> bool,
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
    let mut trickle = false;
    loop {
        if stop() {
            break;
        }
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
        if trickle {
            let cone = ConeCost {
                area_penalty,
                perimeter_penalty,
                threshold,
            };
            charts = merge_incrementally(
                &charts,
                &normals,
                &cone,
                Graph {
                    adj,
                    lengths,
                    perims,
                    sums,
                    areas: chart_areas,
                    axes,
                    half,
                },
                stop,
            );
            break;
        }
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
        let mut merges = 0usize;
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
            merges += 1;
        }
        if !merged {
            break;
        }
        // once few charts merge a round, each full pass costs far more than it changes: the rest goes incrementally
        trickle = merges * TRICKLE_FRACTION < count;
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
/// `charts` with every chart over `max` faces cut into connected pieces of at most `max` faces, grown breadth first
/// over the faces' shared edges. A nearly flat surface (a thin sheet from a flat drawing) is one legitimate cluster of
/// the whole mesh, and xatlas parameterizes a chart as one least-squares system, which at a million faces runs for hours.
pub fn split_large_charts(mesh: &Mesh, charts: &[u32], max: usize) -> Vec<u32> {
    let count = charts.iter().map(|&c| c as usize + 1).max().unwrap_or(0);
    let mut sizes = vec![0usize; count];
    for &c in charts {
        sizes[c as usize] += 1;
    }
    if sizes.iter().all(|&n| n <= max) {
        return charts.to_vec();
    }
    // neighbours within the same chart, as CSR over faces
    let mut degree = vec![0u32; charts.len() + 1];
    let pairs: Vec<(u32, u32)> = manifold_adjacency(mesh)
        .into_iter()
        .filter(|&(a, b, _, _)| {
            charts[a as usize] == charts[b as usize] && sizes[charts[a as usize] as usize] > max
        })
        .map(|(a, b, _, _)| (a, b))
        .collect();
    for &(a, b) in &pairs {
        degree[a as usize + 1] += 1;
        degree[b as usize + 1] += 1;
    }
    for i in 1..degree.len() {
        degree[i] += degree[i - 1];
    }
    let mut fill = degree.clone();
    let mut neighbours = vec![0u32; degree[charts.len()] as usize];
    for &(a, b) in &pairs {
        neighbours[fill[a as usize] as usize] = b;
        fill[a as usize] += 1;
        neighbours[fill[b as usize] as usize] = a;
        fill[b as usize] += 1;
    }
    let mut out = charts.to_vec();
    let mut assigned = vec![false; charts.len()];
    let mut next = count as u32;
    let mut queue = std::collections::VecDeque::new();
    for seed in 0..charts.len() {
        if sizes[charts[seed] as usize] <= max || assigned[seed] {
            continue;
        }
        // a piece: the seed's chart grown breadth first until it holds `max` faces or runs out
        let piece = next;
        next += 1;
        let mut taken = 0;
        assigned[seed] = true;
        queue.clear();
        queue.push_back(seed as u32);
        while let Some(f) = queue.pop_front() {
            out[f as usize] = piece;
            taken += 1;
            if taken == max {
                // the rest of the frontier seeds later pieces
                for &g in queue.iter() {
                    assigned[g as usize] = false;
                }
                break;
            }
            for &g in &neighbours[degree[f as usize] as usize..degree[f as usize + 1] as usize] {
                if !assigned[g as usize] {
                    assigned[g as usize] = true;
                    queue.push_back(g);
                }
            }
        }
    }
    // renumber densely
    let mut ids = vec![u32::MAX; next as usize];
    let mut dense = 0;
    for c in out.iter_mut() {
        if ids[*c as usize] == u32::MAX {
            ids[*c as usize] = dense;
            dense += 1;
        }
        *c = ids[*c as usize];
    }
    out
}

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

/// Degenerate faces dropped, the mesh clustered by [`compute_charts`] within 30 degree normal cones, and each cluster
/// taken by xatlas as one chart (split only where it does not parameterize validly), parameterized and packed. Against
/// CuMesh's settings ([`uv_unwrap_reference`]): an export at 1024 unwraps in about half the time, with less stretch, a
/// few more seams and slightly lower atlas coverage.
pub fn uv_unwrap(
    mesh: &Mesh,
    stop: &(dyn Fn() -> bool + Sync),
) -> Result<Unwrapped, inference_xatlas::Error> {
    uv_unwrap_with(
        mesh,
        UNWRAP_CONE_HALF_ANGLE,
        inference_xatlas::Options { single_chart: true },
        stop,
    )
}

/// CuMesh's `uv_unwrap`: degenerate faces dropped, the clusters of [`compute_charts`] each added to xatlas as a mesh,
/// charted, parameterized and packed with its defaults, the results gathered in cluster order.
pub fn uv_unwrap_reference(mesh: &Mesh) -> Result<Unwrapped, inference_xatlas::Error> {
    uv_unwrap_with(
        mesh,
        CONE_HALF_ANGLE,
        inference_xatlas::Options::default(),
        &|| false,
    )
}

/// [`uv_unwrap`] with xatlas `options`, giving up (with `inference_xatlas::STOPPED`) when `stop` returns true.
pub fn uv_unwrap_with(
    mesh: &Mesh,
    cone_half_angle: f32,
    options: inference_xatlas::Options,
    stop: &(dyn Fn() -> bool + Sync),
) -> Result<Unwrapped, inference_xatlas::Error> {
    let mut mesh = mesh.clone();
    remove_degenerate_faces(&mut mesh);
    let mut charts = compute_charts(
        &mesh,
        cone_half_angle,
        AREA_PENALTY,
        PERIMETER_AREA_PENALTY,
        stop,
    );
    if options.single_chart {
        charts = split_large_charts(&mesh, &charts, MAX_SINGLE_CHART_FACES);
    }
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
    let atlas = inference_xatlas::unwrap_with(&inputs, options, stop)?;
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
    fn a_flat_sheet_splits_into_connected_pieces_within_the_cap() {
        // a 100 x 100 quad grid in one plane: 20000 faces
        let n = 100u32;
        let vertices: Vec<V3> = (0..=n)
            .flat_map(|y| (0..=n).map(move |x| [x as f32, 0.0, y as f32]))
            .collect();
        let at = |x: u32, y: u32| y * (n + 1) + x;
        let faces: Vec<[u32; 3]> = (0..n)
            .flat_map(|y| {
                (0..n).flat_map(move |x| {
                    [
                        [at(x, y), at(x, y + 1), at(x + 1, y)],
                        [at(x + 1, y), at(x, y + 1), at(x + 1, y + 1)],
                    ]
                })
            })
            .collect();
        let mesh = Mesh { vertices, faces };
        // the whole sheet as one chart, as the clustering leaves a thin sheet from a flat drawing
        let charts = vec![0u32; mesh.faces.len()];
        let max = 1500;
        let pieces = split_large_charts(&mesh, &charts, max);
        let count = *pieces.iter().max().expect("faces") as usize + 1;
        assert!(count >= mesh.faces.len() / max, "{count} pieces");
        // each piece within the cap and connected through shared edges
        let adjacency = manifold_adjacency(&mesh);
        for piece in 0..count as u32 {
            let members: Vec<usize> = (0..pieces.len()).filter(|&f| pieces[f] == piece).collect();
            assert!(
                !members.is_empty() && members.len() <= max,
                "piece {piece}: {}",
                members.len()
            );
            let mut reached = std::collections::HashSet::from([members[0]]);
            let mut grew = true;
            while grew {
                grew = false;
                for &(a, b, _, _) in &adjacency {
                    let (a, b) = (a as usize, b as usize);
                    if pieces[a] == piece
                        && pieces[b] == piece
                        && reached.contains(&a) != reached.contains(&b)
                    {
                        reached.insert(a);
                        reached.insert(b);
                        grew = true;
                    }
                }
            }
            assert_eq!(reached.len(), members.len(), "piece {piece} is connected");
        }
        // charts already within the cap pass through unchanged
        assert_eq!(split_large_charts(&mesh, &charts, mesh.faces.len()), charts);
    }

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
        let charts = compute_charts(&mesh, 0.1, 0.0, 0.0, &|| false);
        assert_eq!(charts.iter().max(), Some(&5));
        for side in 0..6 {
            assert_eq!(charts[2 * side], charts[2 * side + 1]);
        }
        // a 90 degree cone joins neighbouring sides
        let wide = compute_charts(&mesh, CONE_HALF_ANGLE, 0.0, 0.0, &|| false);
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
