//! Faces from predicted edges (`edges_to_faces`): every edge triangle is a face, then chordless rings of 4 to 12 open
//! edges (fewer than two faces) are filled by a minimum-area triangulation, smallest rings first. The reference leaves
//! each face in index order; [`orient`] winds them outward.

use std::collections::{BTreeSet, HashMap};

/// The largest open ring `fill_open_rings` closes, in edges
pub const MAX_RING: usize = 12;
const MIN_RING: usize = 4;
// a triangle at most this share of the ring's squared extent counts as degenerate
const DEGENERATE: f64 = 1e-9;
// rays start this far off a face and count hits past it (the mesh spans about 1)
const RAY_START: f64 = 1e-6;
// below this a ray runs along a triangle's plane, or a face has no area
const PARALLEL: f64 = 1e-12;

fn key(a: u32, b: u32) -> (u32, u32) {
    if a < b { (a, b) } else { (b, a) }
}

/// The faces of `n` vertices at `positions` joined by `edges` (`u < v`), rings up to `max_ring` edges filled when
/// `max_ring` is at least 4.
pub fn faces(
    n: usize,
    edges: &[[u32; 2]],
    positions: &[[f32; 3]],
    max_ring: usize,
) -> Vec<[u32; 3]> {
    let mut nbr: Vec<BTreeSet<u32>> = vec![BTreeSet::new(); n];
    for &[u, v] in edges {
        nbr[u as usize].insert(v);
        nbr[v as usize].insert(u);
    }
    let mut out = Vec::new();
    for &[u, v] in edges {
        for &w in nbr[u as usize].intersection(&nbr[v as usize]) {
            if w > v {
                out.push([u, v, w]);
            }
        }
    }
    if max_ring >= MIN_RING && n >= MIN_RING {
        let fill = fill_rings(nbr, &out, positions, max_ring);
        out.extend(fill);
    }
    out
}

// `triangulate_open_rings`
fn fill_rings(
    mut nbr: Vec<BTreeSet<u32>>,
    faces: &[[u32; 3]],
    positions: &[[f32; 3]],
    max_ring: usize,
) -> Vec<[u32; 3]> {
    let n = nbr.len();
    let mut count: HashMap<(u32, u32), u32> = HashMap::new();
    for f in faces {
        for (x, y) in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])] {
            *count.entry(key(x, y)).or_default() += 1;
        }
    }
    let mut open: Vec<BTreeSet<u32>> = vec![BTreeSet::new(); n];
    let mut open_edges = Vec::new();
    for u in 0..n as u32 {
        for &v in &nbr[u as usize] {
            if v > u && count.get(&(u, v)).copied().unwrap_or(0) < 2 {
                open_edges.push((u, v));
                open[u as usize].insert(v);
                open[v as usize].insert(u);
            }
        }
    }
    let mut added = Vec::new();
    for size in MIN_RING..=max_ring {
        for &(u, v) in &open_edges {
            // a face-less edge can border two rings
            while count.get(&(u, v)).copied().unwrap_or(0) < 2 {
                let Some(ring) = shortest_ring(u, v, &open, &nbr, size) else {
                    break;
                };
                if !chordless(&ring, &nbr) {
                    break;
                }
                for tri in min_area(&ring, positions) {
                    added.push(tri);
                    for (x, y) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                        let c = count.entry(key(x, y)).or_default();
                        *c += 1;
                        nbr[x as usize].insert(y);
                        nbr[y as usize].insert(x);
                        if *c >= 2 {
                            open[x as usize].remove(&y);
                            open[y as usize].remove(&x);
                        }
                    }
                }
            }
        }
    }
    added
}

// the shortest cycle of 4..=max vertices through open edge (u, v) along open edges, avoiding vertices adjacent to both
fn shortest_ring(
    u: u32,
    v: u32,
    open: &[BTreeSet<u32>],
    nbr: &[BTreeSet<u32>],
    max: usize,
) -> Option<Vec<u32>> {
    let blocked: BTreeSet<u32> = nbr[u as usize]
        .intersection(&nbr[v as usize])
        .copied()
        .collect();
    let mut parent: HashMap<u32, Option<u32>> = HashMap::from([(v, None)]);
    let mut frontier = vec![v];
    for _ in 0..max - 1 {
        let mut reached = Vec::new();
        for &x in &frontier {
            for &y in &open[x as usize] {
                if parent.contains_key(&y) || blocked.contains(&y) {
                    continue;
                }
                if y == u {
                    if x == v {
                        continue;
                    }
                    let mut ring = vec![u];
                    let mut at = Some(x);
                    while let Some(p) = at {
                        ring.push(p);
                        at = parent[&p];
                    }
                    return Some(ring);
                }
                parent.insert(y, Some(x));
                reached.push(y);
            }
        }
        if reached.is_empty() {
            break;
        }
        frontier = reached;
    }
    None
}

fn chordless(ring: &[u32], nbr: &[BTreeSet<u32>]) -> bool {
    let n = ring.len();
    for i in 0..n.saturating_sub(2) {
        let end = if i == 0 { n - 1 } else { n };
        for &r in &ring[i + 2..end] {
            if nbr[ring[i] as usize].contains(&r) {
                return false;
            }
        }
    }
    true
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn tri_area(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    let (u, v) = (sub(b, a), sub(c, a));
    let x = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    0.5 * (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt()
}

// minimum-area triangulation of a closed ring, degenerate triangles counted first (O(n^3) DP)
fn min_area(ring: &[u32], positions: &[[f32; 3]]) -> Vec<[u32; 3]> {
    let n = ring.len();
    let p: Vec<[f64; 3]> = ring
        .iter()
        .map(|&i| positions[i as usize].map(f64::from))
        .collect();
    let extent: f64 = (0..3)
        .map(|a| {
            let (lo, hi) = p
                .iter()
                .fold((f64::MAX, f64::MIN), |(l, h), q| (l.min(q[a]), h.max(q[a])));
            (hi - lo) * (hi - lo)
        })
        .sum();
    let eps = DEGENERATE * extent;
    let mut cost = vec![vec![(0u32, 0f64); n]; n];
    let mut split = vec![vec![0usize; n]; n];
    for span in 2..n {
        for i in 0..n - span {
            let j = i + span;
            let mut best: Option<(u32, f64)> = None;
            for k in i + 1..j {
                let a = tri_area(p[i], p[k], p[j]);
                let c = (
                    cost[i][k].0 + cost[k][j].0 + u32::from(a <= eps),
                    cost[i][k].1 + cost[k][j].1 + a,
                );
                if best.is_none_or(|b| c.0 < b.0 || (c.0 == b.0 && c.1 < b.1)) {
                    best = Some(c);
                    split[i][j] = k;
                }
            }
            cost[i][j] = best.expect("a span of 2 or more has a split");
        }
    }
    let mut tris = Vec::new();
    let mut stack = vec![(0, n - 1)];
    while let Some((i, j)) = stack.pop() {
        if j - i < 2 {
            continue;
        }
        let k = split[i][j];
        tris.push([ring[i], ring[k], ring[j]]);
        stack.push((i, k));
        stack.push((k, j));
    }
    tris
}

fn cross(u: [f64; 3], w: [f64; 3]) -> [f64; 3] {
    [
        u[1] * w[2] - u[2] * w[1],
        u[2] * w[0] - u[0] * w[2],
        u[0] * w[1] - u[1] * w[0],
    ]
}

fn dot(u: [f64; 3], w: [f64; 3]) -> f64 {
    u[0] * w[0] + u[1] * w[1] + u[2] * w[2]
}

// whether a ray from `origin` along `dir` meets none of `tris` (Moller-Trumbore, hits past RAY_START)
fn escapes(origin: [f64; 3], dir: [f64; 3], tris: &[[[f64; 3]; 3]]) -> bool {
    !tris.iter().any(|[a, b, c]| {
        let (e1, e2) = (sub(*b, *a), sub(*c, *a));
        let h = cross(dir, e2);
        let det = dot(e1, h);
        if det.abs() < PARALLEL {
            return false;
        }
        let s = sub(origin, *a);
        let u = dot(s, h) / det;
        let q = cross(s, e1);
        let v = dot(dir, q) / det;
        u >= 0.0 && v >= 0.0 && u + v <= 1.0 && dot(e2, q) / det > RAY_START
    })
}

/// Winds `faces` outward. A face whose normal leads out of the mesh one way and not the other (a ray from its center
/// escapes along it and not against it, or the reverse) takes that side; the others, enclosed by other parts, take the
/// winding of a decided neighbour across an edge the two share alone. The meshes hold folds and nested shells where
/// winding spread from one face alone would turn whole regions inside out.
pub fn orient(faces: &mut [[u32; 3]], positions: &[[f32; 3]]) {
    use rayon::prelude::*;
    let tris: Vec<[[f64; 3]; 3]> = faces
        .iter()
        .map(|f| f.map(|v| positions[v as usize].map(f64::from)))
        .collect();
    let votes: Vec<i8> = tris
        .par_iter()
        .map(|t| {
            let n = cross(sub(t[1], t[0]), sub(t[2], t[0]));
            let len = dot(n, n).sqrt();
            if len < PARALLEL {
                return 0;
            }
            let n = n.map(|v| v / len);
            let c: [f64; 3] = std::array::from_fn(|k| (t[0][k] + t[1][k] + t[2][k]) / 3.0);
            let at = |s: f64| std::array::from_fn(|k| c[k] + s * RAY_START * n[k]);
            let ahead = escapes(at(1.0), n, &tris);
            let behind = escapes(at(-1.0), n.map(|v| -v), &tris);
            i8::from(ahead && !behind) - i8::from(behind && !ahead)
        })
        .collect();
    let mut by_edge: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (i, f) in faces.iter().enumerate() {
        for (x, y) in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])] {
            by_edge.entry(key(x, y)).or_default().push(i);
        }
    }
    let mut decided: Vec<bool> = votes.iter().map(|&v| v != 0).collect();
    for (f, &v) in faces.iter_mut().zip(&votes) {
        if v < 0 {
            f.swap(1, 2);
        }
    }
    // whether face `f` walks edge (x, y) from x to y
    let walks = |f: &[u32; 3], x: u32, y: u32| (0..3).any(|k| f[k] == x && f[(k + 1) % 3] == y);
    let mut queue: std::collections::VecDeque<usize> =
        (0..faces.len()).filter(|&i| decided[i]).collect();
    // faces no decided one reaches keep the winding the first of them spreads
    let mut next = 0;
    loop {
        while let Some(i) = queue.pop_front() {
            let f = faces[i];
            for (x, y) in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])] {
                let shared = &by_edge[&key(x, y)];
                if shared.len() != 2 {
                    continue;
                }
                for &g in shared {
                    if decided[g] {
                        continue;
                    }
                    if walks(&faces[g], x, y) {
                        faces[g].swap(1, 2);
                    }
                    decided[g] = true;
                    queue.push_back(g);
                }
            }
        }
        while next < faces.len() && decided[next] {
            next += 1;
        }
        if next == faces.len() {
            break;
        }
        decided[next] = true;
        queue.push_back(next);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_square_of_edges_with_a_diagonal_makes_two_faces() {
        let edges = [[0, 1], [0, 2], [0, 3], [1, 2], [2, 3]];
        let positions = [[0.0; 3]; 4];
        assert_eq!(faces(4, &edges, &positions, 0), vec![[0, 1, 2], [0, 2, 3]]);
    }

    #[test]
    fn a_tetrahedron_written_in_index_order_is_wound_outward() {
        let positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let edges = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];
        let mut tris = faces(4, &edges, &positions, 0);
        orient(&mut tris, &positions);
        for [a, b, c] in tris {
            let [p, q, r] = [a, b, c].map(|v| positions[v as usize]);
            let n = [
                (q[1] - p[1]) * (r[2] - p[2]) - (q[2] - p[2]) * (r[1] - p[1]),
                (q[2] - p[2]) * (r[0] - p[0]) - (q[0] - p[0]) * (r[2] - p[2]),
                (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]),
            ];
            // the centroid's at (0.25, 0.25, 0.25): an outward normal points away from it
            let out = (0..3)
                .map(|i| n[i] * ((p[i] + q[i] + r[i]) / 3.0 - 0.25))
                .sum::<f32>();
            assert!(out > 0.0, "face {a} {b} {c} points inward");
        }
    }

    #[test]
    fn an_open_square_is_filled_by_two_triangles() {
        let edges = [[0, 1], [1, 2], [2, 3], [0, 3]];
        let positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let got = faces(4, &edges, &positions, MAX_RING);
        assert_eq!(got.len(), 2);
    }
}
