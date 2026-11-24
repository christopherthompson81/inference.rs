//! Faces from predicted edges (`edges_to_faces`): every edge triangle is a face, then chordless rings of 4 to 12 open
//! edges (fewer than two faces) are filled by a minimum-area triangulation, smallest rings first.

use std::collections::{BTreeSet, HashMap};

/// The largest open ring `fill_open_rings` closes, in edges
pub const MAX_RING: usize = 12;
const MIN_RING: usize = 4;
// a triangle at most this share of the ring's squared extent counts as degenerate
const DEGENERATE: f64 = 1e-9;

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
