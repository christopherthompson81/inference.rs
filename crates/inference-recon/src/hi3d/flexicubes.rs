//! FlexiCubes' dual marching cubes as Hi3D runs it (no learned weights): each surface cube places its dual vertices at
//! the mean of their edges' zero crossings, ambiguous cases are resolved against their neighbour, and every surface
//! edge shared by four surface cubes becomes a quad of their dual vertices.

use std::collections::HashMap;

use super::flexicubes_tables::{CHECK, DMC, NUM_VD};

/// A cube's 12 edges as corner pairs (corner `c` at offset `(c & 1, c >> 1 & 1, c >> 2 & 1)`).
const CUBE_EDGES: [[usize; 2]; 12] = [
    [0, 1],
    [1, 5],
    [4, 5],
    [0, 4],
    [2, 3],
    [3, 7],
    [6, 7],
    [2, 6],
    [2, 0],
    [3, 1],
    [7, 5],
    [6, 4],
];
const QUAD_SPLIT: [usize; 6] = [0, 1, 3, 3, 1, 2];

/// The surface of `scalar` (negative inside) over cubes `cubes` (each its 8 corners' rows of `vertices`), at integer
/// `coords` (sorted, as the cubes are) on a `resolution` grid.
pub fn extract(
    vertices: &[[f32; 3]],
    scalar: &[f32],
    cubes: &[[u32; 8]],
    coords: &[[i32; 3]],
    resolution: usize,
) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    let inside = |v: u32| scalar[v as usize] < 0.0;
    // surface cubes and their occupancy cases
    let mut surf: Vec<usize> = Vec::new();
    let mut cases: Vec<usize> = Vec::new();
    for (i, cube) in cubes.iter().enumerate() {
        let occ = cube.iter().filter(|&&v| inside(v)).count();
        if occ > 0 && occ < 8 {
            surf.push(i);
            cases.push(
                cube.iter()
                    .enumerate()
                    .map(|(c, &v)| usize::from(inside(v)) << c)
                    .sum(),
            );
        }
    }
    if surf.is_empty() {
        return (Vec::new(), Vec::new());
    }

    // ambiguous cases flip when their neighbour in the table's direction is ambiguous too; only cubes whose own
    // neighbour lies on the grid count as ambiguous
    let res = resolution as i32;
    let neighbour = |k: usize| {
        let check = CHECK[cases[k]];
        let c = coords[surf[k]];
        let adj = [
            c[0] + check[1] as i32,
            c[1] + check[2] as i32,
            c[2] + check[3] as i32,
        ];
        (check[0] == 1 && adj.iter().all(|&v| (0..res).contains(&v))).then_some(adj)
    };
    let ambiguous: Vec<Option<[i32; 3]>> = (0..surf.len()).map(neighbour).collect();
    let problem: std::collections::HashSet<[i32; 3]> = ambiguous
        .iter()
        .enumerate()
        .filter(|(_, adj)| adj.is_some())
        .map(|(k, _)| coords[surf[k]])
        .collect();
    for (k, adj) in ambiguous.iter().enumerate() {
        if let Some(adj) = adj
            && problem.contains(adj)
        {
            cases[k] = CHECK[cases[k]][4] as usize;
        }
    }

    // the surface cubes' edges: unique by their (ordered) corner pair, counted over the cubes sharing them
    let mut edge_ids: HashMap<[u32; 2], u32> = HashMap::new();
    let mut edges: Vec<[u32; 2]> = Vec::new();
    let mut counts: Vec<u8> = Vec::new();
    let mut slots: Vec<u32> = Vec::with_capacity(surf.len() * 12);
    for &i in &surf {
        for pair in CUBE_EDGES {
            let e = [cubes[i][pair[0]], cubes[i][pair[1]]];
            let id = *edge_ids.entry(e).or_insert_with(|| {
                edges.push(e);
                counts.push(0);
                (edges.len() - 1) as u32
            });
            counts[id as usize] += 1;
            slots.push(id);
        }
    }
    let crossing = |e: u32| {
        let [a, b] = edges[e as usize];
        inside(a) != inside(b)
    };
    let zero = |e: u32| {
        let [a, b] = edges[e as usize];
        let (x0, x1) = (vertices[a as usize], vertices[b as usize]);
        let (s0, s1) = (scalar[a as usize], scalar[b as usize]);
        let den = s1 - s0;
        [0, 1, 2].map(|d| (x0[d] * s1 - x1[d] * s0) / den)
    };

    // dual vertices, and each (cube, edge) slot's: numbered as the reference groups them, cubes making one vertex
    // first, then two, three and four, each group in cube order
    let mut dual: Vec<[f32; 3]> = Vec::new();
    let mut slot_vd = vec![u32::MAX; slots.len()];
    let mut by_count: Vec<usize> = (0..cases.len()).collect();
    by_count.sort_by_key(|&k| NUM_VD[cases[k]]);
    for k in by_count {
        let case = cases[k];
        for group in DMC[case].iter().take(NUM_VD[case] as usize) {
            let local: Vec<usize> = group
                .iter()
                .take_while(|&&e| e >= 0)
                .map(|&e| e as usize)
                .collect();
            let mut sum = [0f32; 3];
            for &e in &local {
                let p = zero(slots[k * 12 + e]);
                for d in 0..3 {
                    sum[d] += p[d];
                }
            }
            let id = dual.len() as u32;
            dual.push(sum.map(|v| v / local.len() as f32));
            for &e in &local {
                slot_vd[k * 12 + e] = id;
            }
        }
    }

    // a quad per surface edge inside four surface cubes, its dual vertices in cube order
    let mut quads: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut order: Vec<u32> = Vec::new();
    for (s, &e) in slots.iter().enumerate() {
        if counts[e as usize] == 4 && crossing(e) {
            let entry = quads.entry(e).or_insert_with(|| {
                order.push(e);
                Vec::with_capacity(4)
            });
            entry.push(slot_vd[s]);
        }
    }
    // in the order of their edges' corner pairs (torch.unique's)
    order.sort_unstable_by_key(|&e| edges[e as usize]);
    // the quads whose edge starts outside first, then the others (the reference concatenates them so)
    let (flipped, plain): (Vec<u32>, Vec<u32>) = order
        .into_iter()
        .partition(|&e| scalar[edges[e as usize][0] as usize] > 0.0);
    let mut faces = Vec::with_capacity((flipped.len() + plain.len()) * 2);
    for (list, flip) in [(&flipped, true), (&plain, false)] {
        for e in list {
            let q = &quads[e];
            let quad = if flip {
                [q[0], q[1], q[3], q[2]]
            } else {
                [q[2], q[3], q[1], q[0]]
            };
            faces.push([
                quad[QUAD_SPLIT[0]],
                quad[QUAD_SPLIT[1]],
                quad[QUAD_SPLIT[2]],
            ]);
            faces.push([
                quad[QUAD_SPLIT[3]],
                quad[QUAD_SPLIT[4]],
                quad[QUAD_SPLIT[5]],
            ]);
        }
    }
    (dual, faces)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIDE: i32 = 12;
    const RADIUS: f32 = 4.3;

    // a grid's vertices, their scalar, its cubes and their coordinates
    type Grid = (Vec<[f32; 3]>, Vec<f32>, Vec<[u32; 8]>, Vec<[i32; 3]>);

    // a sphere's distance sampled on a grid's vertices, every cube of the grid present
    fn sphere() -> Grid {
        let n = SIDE + 1;
        let vid = |x: i32, y: i32, z: i32| ((x * n + y) * n + z) as u32;
        let mut vertices = Vec::new();
        let mut scalar = Vec::new();
        for x in 0..n {
            for y in 0..n {
                for z in 0..n {
                    let p = [x, y, z].map(|v| v as f32 - SIDE as f32 / 2.0);
                    vertices.push(p);
                    scalar.push((p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() - RADIUS);
                }
            }
        }
        let mut cubes = Vec::new();
        let mut coords = Vec::new();
        for x in 0..SIDE {
            for y in 0..SIDE {
                for z in 0..SIDE {
                    cubes.push(std::array::from_fn(|c| {
                        vid(
                            x + (c & 1) as i32,
                            y + ((c >> 1) & 1) as i32,
                            z + ((c >> 2) & 1) as i32,
                        )
                    }));
                    coords.push([x, y, z]);
                }
            }
        }
        (vertices, scalar, cubes, coords)
    }

    #[test]
    fn a_sphere_meshes_closed_on_its_surface_with_faces_outward() {
        let (vertices, scalar, cubes, coords) = sphere();
        let (dual, faces) = extract(&vertices, &scalar, &cubes, &coords, SIDE as usize);
        assert!(!faces.is_empty());
        let mut edges: HashMap<[u32; 2], i32> = HashMap::new();
        let mut volume = 0f32;
        for f in &faces {
            for k in 0..3 {
                *edges
                    .entry([f[k].min(f[(k + 1) % 3]), f[k].max(f[(k + 1) % 3])])
                    .or_default() += 1;
            }
            let [a, b, c] = f.map(|v| dual[v as usize]);
            // six times the signed volume of the tetrahedron to the origin
            volume += a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0]);
        }
        assert!(
            edges.values().all(|&n| n == 2),
            "the surface has open or non-manifold edges"
        );
        for v in &dual {
            let r = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            assert!((r - RADIUS).abs() < 0.3, "a vertex at radius {r}");
        }
        let enclosed = volume / 6.0;
        let ball = 4.0 / 3.0 * std::f32::consts::PI * RADIUS.powi(3);
        // the dual vertices sit on chords of the sphere, a little inside it
        assert!(
            (enclosed - ball).abs() / ball < 0.1,
            "encloses {enclosed} (a ball {ball}), faces inward when negative"
        );
    }
}
