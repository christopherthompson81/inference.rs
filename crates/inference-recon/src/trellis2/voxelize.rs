//! A mesh as a flexible dual grid (o_voxel's `mesh_to_flexible_dual_grid`, CPU path): the cells of a `res`^3 grid
//! over the unit cube whose edges the surface crosses, each with a dual vertex minimizing a quadric error built from
//! the planes of the faces crossing its edges and touching it, the mesh's open boundary edges through it, and a pull
//! toward its crossings' mean, kept within the cell. TRELLIS.2's shape encoder reads them.

use std::collections::HashMap;

use super::mesh::Mesh;

/// `Trellis2TexturingPipeline.encode_shape_slat`'s weights
pub const BOUNDARY_WEIGHT: f32 = 0.2;
pub const REGULARIZATION_WEIGHT: f32 = 1e-2;

type M4 = [[f32; 4]; 4];

/// The grid's occupied cells.
pub struct DualGrid {
    pub cells: Vec<[i32; 3]>,
    /// Each cell's dual vertex within it, in [0, 1] of the cell
    pub offsets: Vec<[f32; 3]>,
    /// Whether the surface crosses the cell's edge along x, y, z leaving its lower corner
    pub crossed: Vec<[bool; 3]>,
}

struct Cell {
    mean: [f32; 3],
    count: f32,
    crossed: [bool; 3],
    q: M4,
}

fn outer(p: [f32; 4]) -> M4 {
    std::array::from_fn(|i| std::array::from_fn(|j| p[i] * p[j]))
}

fn add_scaled(q: &mut M4, r: &M4, s: f32) {
    for i in 0..4 {
        for j in 0..4 {
            q[i][j] += s * r[i][j];
        }
    }
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

// Eigen's: a zero vector stays zero
fn normalized(a: [f32; 3]) -> [f32; 3] {
    let n = dot(a, a).sqrt();
    if n > 0.0 { a.map(|v| v / n) } else { a }
}

// the face's plane quadric, and its unit normal
fn plane(v0: [f32; 3], v1: [f32; 3], v2: [f32; 3]) -> (M4, [f32; 3]) {
    let n = normalized(cross(sub(v1, v0), sub(v2, v1)));
    (outer([n[0], n[1], n[2], -dot(n, v0)]), n)
}

fn lerp(a: f64, b: f64, t: f64, va: [f64; 2], vb: [f64; 2]) -> [f64; 2] {
    if a == b {
        return va;
    }
    let alpha = (t - a) / (b - a);
    [
        (1.0 - alpha) * va[0] + alpha * vb[0],
        (1.0 - alpha) * va[1] + alpha * vb[1],
    ]
}

struct Grid {
    res: i32,
    size: f32,
    table: HashMap<[i32; 3], usize>,
    cells: Vec<([i32; 3], Cell)>,
}

impl Grid {
    // the crossings of each grid line along `ax2` with the triangle, adding each to the four cells around the line
    fn scan(&mut self, tri: [[f32; 3]; 3], q: &M4, ax2: usize) {
        let (ax0, ax1) = ((ax2 + 1) % 3, (ax2 + 2) % 3);
        let size = f64::from(self.size);
        let hi = self.res - 1;
        let mut t: [[f64; 3]; 3] = tri.map(|v| [v[ax0], v[ax1], v[ax2]].map(f64::from));
        t.sort_by(|a, b| a[1].partial_cmp(&b[1]).unwrap_or(std::cmp::Ordering::Equal));
        let row = |y: f64| ((y / size) as i32).clamp(0, hi);
        let (start, mid, end) = (row(t[0][1]), row(t[1][1]), row(t[2][1]));
        self.half(start, mid, [t[0], t[1], t[2]], q, [ax0, ax1, ax2]);
        self.half(mid, end, [t[2], t[1], t[0]], q, [ax0, ax1, ax2]);
    }

    fn half(&mut self, rows_from: i32, rows_to: i32, t: [[f64; 3]; 3], q: &M4, ax: [usize; 3]) {
        let [ax0, ax1, ax2] = ax;
        let size = f64::from(self.size);
        let hi = self.res - 1;
        let [t0, t1, t2] = t;
        for y_idx in rows_from..rows_to {
            let y = f64::from(y_idx + 1) * size;
            let mut a = lerp(t0[1], t1[1], y, [t0[0], t0[2]], [t1[0], t1[2]]);
            let mut b = lerp(t0[1], t2[1], y, [t0[0], t0[2]], [t2[0], t2[2]]);
            if a[0] > b[0] {
                std::mem::swap(&mut a, &mut b);
            }
            let line_start = ((a[0] / size) as i32).clamp(0, hi);
            let line_end = ((b[0] / size) as i32).clamp(0, hi);
            for x_idx in line_start..line_end {
                let x = f64::from(x_idx + 1) * size;
                let z = lerp(a[0], b[0], x, [a[1], 0.0], [b[1], 0.0])[0];
                let z_idx = (z / size) as i32;
                if !(0..self.res).contains(&z_idx) {
                    continue;
                }
                let mut hit = [0f64; 3];
                (hit[ax0], hit[ax1], hit[ax2]) = (x, y, z);
                let hit = hit.map(|v| v as f32);
                for dx in 0..2 {
                    for dy in 0..2 {
                        let mut coord = [0i32; 3];
                        (coord[ax0], coord[ax1], coord[ax2]) = (x_idx + dx, y_idx + dy, z_idx);
                        let own = dx == 0 && dy == 0;
                        match self.table.get(&coord) {
                            Some(&i) => {
                                let cell = &mut self.cells[i].1;
                                for (m, h) in cell.mean.iter_mut().zip(hit) {
                                    *m += h;
                                }
                                cell.count += 1.0;
                                cell.crossed[ax2] |= own;
                                add_scaled(&mut cell.q, q, 1.0);
                            }
                            None => {
                                let mut crossed = [false; 3];
                                crossed[ax2] = own;
                                self.table.insert(coord, self.cells.len());
                                self.cells.push((
                                    coord,
                                    Cell {
                                        mean: hit,
                                        count: 1.0,
                                        crossed,
                                        q: *q,
                                    },
                                ));
                            }
                        }
                    }
                }
            }
        }
    }

    // the face's plane added to every occupied cell its triangle touches (a triangle-box overlap test)
    fn touch(&mut self, [v0, v1, v2]: [[f32; 3]; 3], q: &M4, n: [f32; 3]) {
        let s = self.size;
        let vs = [s; 3];
        let (e0, e1, e2) = (sub(v1, v0), sub(v2, v1), sub(v0, v2));
        let lo: [i32; 3] =
            std::array::from_fn(|k| ((v0[k].min(v1[k]).min(v2[k]) / s) as i32).max(0));
        let hi: [i32; 3] =
            std::array::from_fn(|k| ((v0[k].max(v1[k]).max(v2[k]) / s + 1.0) as i32).min(self.res));
        let c: [f32; 3] = std::array::from_fn(|k| if n[k] > 0.0 { s } else { 0.0 });
        let d1 = dot(n, sub(c, v0));
        let d2 = dot(n, sub(sub(vs, c), v0));
        // the triangle's edges projected on the plane of axes (a, b), oriented by the normal's other component
        let project = |a: usize, b: usize, other: usize| -> [([f32; 2], f32); 3] {
            let mul = if n[other] < 0.0 { -1.0 } else { 1.0 };
            [(e0, v0), (e1, v1), (e2, v2)].map(|(e, v)| {
                let ne = [-mul * e[b], mul * e[a]];
                let d = -(ne[0] * v[a] + ne[1] * v[b]) + ne[0].max(0.0) * s + ne[1].max(0.0) * s;
                (ne, d)
            })
        };
        let xy = project(0, 1, 2);
        let yz = project(1, 2, 0);
        let zx = project(2, 0, 1);
        let inside = |edges: &[([f32; 2], f32); 3], p: [f32; 2]| {
            edges
                .iter()
                .all(|(ne, d)| ne[0] * p[0] + ne[1] * p[1] + d >= 0.0)
        };
        for z in lo[2]..hi[2] {
            for y in lo[1]..hi[1] {
                for x in lo[0]..hi[0] {
                    let p = [x as f32 * s, y as f32 * s, z as f32 * s];
                    let np = dot(n, p);
                    if (np + d1) * (np + d2) > 0.0 {
                        continue;
                    }
                    if !inside(&xy, [p[0], p[1]])
                        || !inside(&yz, [p[1], p[2]])
                        || !inside(&zx, [p[2], p[0]])
                    {
                        continue;
                    }
                    if let Some(&i) = self.table.get(&[x, y, z]) {
                        add_scaled(&mut self.cells[i].1.q, q, 1.0);
                    }
                }
            }
        }
    }

    // an open boundary edge's line quadric, weighted, added to the occupied cells it passes through (3D DDA)
    fn boundary(&mut self, v0: [f32; 3], v1: [f32; 3], weight: f32) {
        let s = self.size;
        let dir64 = [0, 1, 2].map(|k| f64::from(v1[k]) - f64::from(v0[k]));
        let length = (dir64[0] * dir64[0] + dir64[1] * dir64[1] + dir64[2] * dir64[2]).sqrt();
        if length < 1e-6 {
            return;
        }
        let dir = dir64.map(|d| d / length);
        let a: [[f32; 3]; 3] = std::array::from_fn(|i| {
            std::array::from_fn(|j| f32::from(i == j) - (dir[i] * dir[j]) as f32)
        });
        let av0: [f32; 3] = std::array::from_fn(|i| dot(a[i], v0));
        let b = av0.map(|v| -v);
        let c = dot(v0, av0);
        let mut q = [[0f32; 4]; 4];
        for i in 0..3 {
            q[i][..3].copy_from_slice(&a[i]);
            q[i][3] = b[i];
            q[3][i] = b[i];
        }
        q[3][3] = c;
        let start: [i32; 3] = v0.map(|v| (v / s).floor() as i32);
        let step: [i32; 3] = dir.map(|d| if d > 0.0 { 1 } else { -1 });
        let mut t_max = [f64::INFINITY; 3];
        let mut t_delta = [f64::INFINITY; 3];
        for k in 0..3 {
            if dir[k] != 0.0 {
                let border = s * (start[k] + i32::from(step[k] > 0)) as f32;
                t_max[k] = f64::from(border - v0[k]) / dir[k];
                t_delta[k] = f64::from(s) / dir[k].abs();
            }
        }
        let mut current = start;
        let mut visited = vec![current];
        loop {
            let axis = if t_max[0] < t_max[1] {
                if t_max[0] < t_max[2] { 0 } else { 2 }
            } else if t_max[1] < t_max[2] {
                1
            } else {
                2
            };
            if t_max[axis] > length {
                break;
            }
            current[axis] += step[axis];
            t_max[axis] += t_delta[axis];
            visited.push(current);
        }
        for coord in visited {
            if coord.iter().any(|&v| !(0..self.res).contains(&v)) {
                continue;
            }
            if let Some(&i) = self.table.get(&coord) {
                add_scaled(&mut self.cells[i].1.q, &q, weight);
            }
        }
    }
}

// v^T Q v for v = (p, 1)
fn error(q: &M4, p: [f32; 3]) -> f32 {
    let v = [p[0], p[1], p[2], 1.0];
    (0..4)
        .map(|i| v[i] * (0..4).map(|j| q[i][j] * v[j]).sum::<f32>())
        .sum()
}

// A x = b by Gaussian elimination with partial pivoting, in f64: the reference's f32 solve of quadrics at absolute
// coordinates is noisy to ~1e-3 of a cell, which this does not try to reproduce
fn solve<const N: usize>(a: [[f32; N]; N], b: [f32; N]) -> [f32; N] {
    let x = solve64(a.map(|r| r.map(f64::from)), b.map(f64::from));
    x.map(|v| v as f32)
}

fn solve64<const N: usize>(mut a: [[f64; N]; N], mut b: [f64; N]) -> [f64; N] {
    for col in 0..N {
        let pivot = (col..N)
            .max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))
            .unwrap_or(col);
        a.swap(col, pivot);
        b.swap(col, pivot);
        if a[col][col] == 0.0 {
            continue;
        }
        let pivot_row = a[col];
        for row in col + 1..N {
            let f = a[row][col] / pivot_row[col];
            for (x, p) in a[row][col..].iter_mut().zip(&pivot_row[col..]) {
                *x -= f * p;
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = [0f64; N];
    for row in (0..N).rev() {
        let s: f64 = (row + 1..N).map(|k| a[row][k] * x[k]).sum();
        x[row] = if a[row][row] == 0.0 {
            0.0
        } else {
            (b[row] - s) / a[row][row]
        };
    }
    x
}

// the point minimizing the cell's quadric within [lo, hi]: unconstrained, else the best on its faces, edges, corners
fn dual_vertex(q: &M4, lo: [f32; 3], hi: [f32; 3]) -> [f32; 3] {
    let within = |p: [f32; 3], axes: &[usize]| axes.iter().all(|&k| p[k] >= lo[k] && p[k] <= hi[k]);
    let a3: [[f32; 3]; 3] = std::array::from_fn(|i| std::array::from_fn(|j| q[i][j]));
    let free = solve(a3, [-q[0][3], -q[1][3], -q[2][3]]);
    if within(free, &[0, 1, 2]) {
        return free;
    }
    let mut best = f32::INFINITY;
    let mut out = free;
    let mut consider = |p: [f32; 3]| {
        let e = error(q, p);
        if e < best {
            best = e;
            out = p;
        }
    };
    for fixed in 0..3 {
        let (a1, a2) = ((fixed + 1) % 3, (fixed + 2) % 3);
        let a = [[q[a1][a1], q[a1][a2]], [q[a2][a1], q[a2][a2]]];
        for bound in [lo[fixed], hi[fixed]] {
            let b = [
                -(q[a1][fixed] * bound + q[a1][3]),
                -(q[a2][fixed] * bound + q[a2][3]),
            ];
            let x = solve(a, b);
            let mut p = [0f32; 3];
            (p[fixed], p[a1], p[a2]) = (bound, x[0], x[1]);
            if within(p, &[a1, a2]) {
                consider(p);
            }
        }
    }
    for axis in 0..3 {
        let (a1, a2) = ((axis + 1) % 3, (axis + 2) % 3);
        for (b1, b2) in [
            (lo[a1], lo[a2]),
            (lo[a1], hi[a2]),
            (hi[a1], lo[a2]),
            (hi[a1], hi[a2]),
        ] {
            let x = -(q[axis][a1] * b1 + q[axis][a2] * b2 + q[axis][3]) / q[axis][axis];
            if x >= lo[axis] && x <= hi[axis] {
                let mut p = [0f32; 3];
                (p[axis], p[a1], p[a2]) = (x, b1, b2);
                consider(p);
            }
        }
    }
    for corner in 0..8 {
        consider(std::array::from_fn(|k| {
            if corner >> (2 - k) & 1 == 0 {
                hi[k]
            } else {
                lo[k]
            }
        }));
    }
    out
}

/// The flexible dual grid of `mesh` (vertices within [-0.5, 0.5]^3) at `res` cells a side.
pub fn flexible_dual_grid(mesh: &Mesh, res: usize) -> DualGrid {
    let size = 1.0 / res as f32;
    let mut grid = Grid {
        res: res as i32,
        size,
        table: HashMap::new(),
        cells: Vec::new(),
    };
    let shifted: Vec<[f32; 3]> = mesh.vertices.iter().map(|v| v.map(|c| c + 0.5)).collect();
    let tris: Vec<[[f32; 3]; 3]> = mesh
        .faces
        .iter()
        .map(|f| f.map(|i| shifted[i as usize]))
        .collect();
    let planes: Vec<(M4, [f32; 3])> = tris.iter().map(|&[a, b, c]| plane(a, b, c)).collect();
    for (tri, (q, _)) in tris.iter().zip(&planes) {
        for ax in 0..3 {
            grid.scan(*tri, q, ax);
        }
    }
    for (tri, (q, n)) in tris.iter().zip(&planes) {
        grid.touch(*tri, q, *n);
    }
    let mut edges: HashMap<(u32, u32), u32> = HashMap::new();
    for f in &mesh.faces {
        for k in 0..3 {
            let (a, b) = (f[k], f[(k + 1) % 3]);
            *edges.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    let mut open: Vec<(u32, u32)> = edges
        .into_iter()
        .filter(|&(_, n)| n == 1)
        .map(|(e, _)| e)
        .collect();
    open.sort_unstable();
    for (a, b) in open {
        grid.boundary(shifted[a as usize], shifted[b as usize], BOUNDARY_WEIGHT);
    }
    let mut out = DualGrid {
        cells: Vec::with_capacity(grid.cells.len()),
        offsets: Vec::with_capacity(grid.cells.len()),
        crossed: Vec::with_capacity(grid.cells.len()),
    };
    for (coord, cell) in grid.cells {
        let mut q = cell.q;
        let p = cell.mean.map(|m| m / cell.count);
        let mut reg = [[0f32; 4]; 4];
        for k in 0..3 {
            reg[k][k] = 1.0;
            reg[k][3] = -p[k];
            reg[3][k] = -p[k];
        }
        reg[3][3] = dot(p, p);
        add_scaled(&mut q, &reg, REGULARIZATION_WEIGHT * cell.count);
        let lo = coord.map(|c| c as f32 * size);
        let hi = coord.map(|c| (c + 1) as f32 * size);
        let v = dual_vertex(&q, lo, hi);
        out.offsets
            .push(std::array::from_fn(|k| v[k] * res as f32 - coord[k] as f32));
        out.cells.push(coord);
        out.crossed.push(cell.crossed);
    }
    out
}
