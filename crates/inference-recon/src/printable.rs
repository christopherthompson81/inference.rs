//! Printable solids: one closed shell from Gaussian splats or a shell of voxels, written as STL. The source marks a
//! grid's points occupied (the Gaussians' summed density, or the voxels); whatever a flood from the grid's border cannot
//! reach is solid, so cavities fill, and parts too thin to print are thickened. The largest solid piece is kept, its
//! indicator blurred and its surface extracted by surface nets, which closes by construction.

use std::collections::HashMap;

use rayon::prelude::*;

use crate::idol::Gaussians;
use crate::trellis2::mesh::Mesh;

/// Grid points beyond the unit cube on each side: past the blur's reach, so the border stays empty
const PAD: usize = 7;
/// The indicator's blur, in voxels, and its kernel's reach
const BLUR_SIGMA: f32 = 1.5;
const BLUR_RADIUS: usize = 5;
const ISO: f32 = 0.5;
// a point solidified to join a diagonal, just past the level
const JOINED: f32 = 0.51;
// Taubin's pass band (lambda, mu) and rounds
const SMOOTH_SHRINK: f32 = 0.5;
const SMOOTH_INFLATE: f32 = -0.53;
const SMOOTH_ROUNDS: usize = 10;
// a cell's three faces at its lower corner, their corners (x fastest) in cyclic order
const CELL_FACES: [[usize; 4]; 3] = [[0, 1, 3, 2], [0, 1, 5, 4], [0, 2, 6, 4]];
// face-neighbour steps the occupied points grow by before the flood, and the flood after, closing small gaps
const SPLAT_CLOSING: usize = 1;
const VOXEL_CLOSING: usize = 1;
// half the thinnest wall, in voxels: thinner parts grow to about twice this plus one (a TRELLIS.2 object can be an open
// shell, and its walls, rods and Idol's thin parts would not survive the blur or a printer)
const MIN_WALL: usize = 2;
/// Summed Gaussian density at which a point counts as occupied
const DENSITY_THRESHOLD: f32 = 0.5;
/// A Gaussian's reach, in standard deviations
const GAUSSIAN_REACH: f32 = 3.0;
/// The thinnest axis a Gaussian is given, in voxels: a thinner sheet slips between the grid points
const MIN_SCALE_VOXELS: f32 = 0.6;
const MIN_OPACITY: f32 = 0.05;
const STL_HEADER: &[u8] = b"inference.rs printable solid";
const STL_HEADER_LEN: usize = 80;
/// Faces a printable surface is simplified toward: about 25 MB of STL
const PRINT_FACES: usize = 500_000;
/// The longest side an export is scaled to when none is asked for, in millimetres
pub const DEFAULT_PRINT_SIZE_MM: f64 = 100.0;
/// The grids printable solids are made on: Idol's splats, TRELLIS.2's voxels
pub const SPLAT_GRID: usize = 256;
pub const VOXEL_GRID: usize = 384;

/// The solid points of a grid with `res` cells across the unit cube, framed around what it holds and padded on every
/// side.
pub struct Solid {
    res: usize,
    dims: [usize; 3],
    // the unit cube's grid coordinate (0 at -0.5, `res` at 0.5) of each axis's first point
    origin: [i64; 3],
    inside: Vec<bool>,
}

fn rotation([w, x, y, z]: [f32; 4]) -> [[f32; 3]; 3] {
    let norm = (w * w + x * x + y * y + z * z)
        .sqrt()
        .max(f32::MIN_POSITIVE);
    let [w, x, y, z] = [w, x, y, z].map(|v| v / norm);
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

impl Solid {
    // a grid over the unit-cube grid coordinates `lo..=hi` (clamped to the cube), padded
    fn framed(res: usize, lo: [f32; 3], hi: [f32; 3]) -> Self {
        let cube = |v: f32| v.clamp(0.0, res as f32);
        let origin = lo.map(|v| cube(v).floor() as i64 - PAD as i64);
        let dims =
            std::array::from_fn(|a| (cube(hi[a]).ceil() as i64 - origin[a]) as usize + PAD + 1);
        Self {
            res,
            dims,
            origin,
            inside: Vec::new(),
        }
    }

    fn len(&self) -> usize {
        self.dims.iter().product()
    }

    fn strides(&self) -> [usize; 3] {
        [1, self.dims[0], self.dims[0] * self.dims[1]]
    }

    fn index(&self, [x, y, z]: [usize; 3]) -> usize {
        (z * self.dims[1] + y) * self.dims[0] + x
    }

    fn coords(&self, i: usize) -> [usize; 3] {
        let [dx, dy, _] = self.dims;
        [i % dx, (i / dx) % dy, i / (dx * dy)]
    }

    // a position in the unit cube from a grid coordinate along axis `a`
    fn to_world(&self, g: f32, a: usize) -> f32 {
        (g + self.origin[a] as f32) / self.res as f32 - 0.5
    }

    // every point's index and coordinates, rows in parallel, in order
    fn points(&self) -> impl ParallelIterator<Item = (usize, [usize; 3])> + '_ {
        let [dx, dy, _] = self.dims;
        (0..self.len() / dx)
            .into_par_iter()
            .flat_map_iter(move |row| {
                let (y, z) = (row % dy, row / dy);
                (0..dx).map(move |x| (row * dx + x, [x, y, z]))
            })
    }

    // `f(i, [x, y, z])` for every point, into a new grid
    fn map<T: Send>(&self, f: impl Fn(usize, [usize; 3]) -> T + Sync + Send) -> Vec<T> {
        self.points().map(|(i, c)| f(i, c)).collect()
    }

    /// The solid inside Gaussians whose centres lie within [-0.5, 0.5]^3, on a `res`^3 grid.
    pub fn from_gaussians(g: &Gaussians, res: usize) -> Self {
        let floor = MIN_SCALE_VOXELS / res as f32;
        // each Gaussian's centre and reach (half extent along each axis) in the cube's grid units
        let reach: Vec<Option<([f32; 3], [f32; 3])>> = (0..g.xyz.len())
            .map(|i| {
                if g.opacity[i] < MIN_OPACITY {
                    return None;
                }
                let r = rotation(g.rotation[i]);
                let scale = g.scaling[i].map(|s| s.max(floor));
                let half = std::array::from_fn(|j| {
                    let var: f32 = (0..3).map(|m| (r[j][m] * scale[m]).powi(2)).sum();
                    GAUSSIAN_REACH * var.sqrt() * res as f32
                });
                Some((g.xyz[i].map(|c| (c + 0.5) * res as f32), half))
            })
            .collect();
        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        for (c, h) in reach.iter().flatten() {
            for a in 0..3 {
                lo[a] = lo[a].min(c[a] - h[a]);
                hi[a] = hi[a].max(c[a] + h[a]);
            }
        }
        if lo[0] > hi[0] {
            lo = [0.0; 3];
            hi = [0.0; 3];
        }
        let mut solid = Self::framed(res, lo, hi);
        let [dx, dy, _] = solid.dims;
        let last = solid.dims.map(|d| (d - 1) as f32);
        let mut density = vec![0f32; solid.len()];
        for (i, entry) in reach.iter().enumerate() {
            let Some((centre, half)) = entry else {
                continue;
            };
            let centre: [f32; 3] = std::array::from_fn(|a| centre[a] - solid.origin[a] as f32);
            if (0..3).any(|a| centre[a] + half[a] < 0.0 || centre[a] - half[a] > last[a]) {
                continue;
            }
            let lo: [usize; 3] =
                std::array::from_fn(|a| (centre[a] - half[a]).ceil().max(0.0) as usize);
            let hi: [usize; 3] =
                std::array::from_fn(|a| (centre[a] + half[a]).floor().min(last[a]) as usize);
            let r = rotation(g.rotation[i]);
            let inverse = g.scaling[i].map(|s| 1.0 / (s.max(floor) * res as f32));
            let opacity = g.opacity[i];
            for z in lo[2]..=hi[2] {
                for y in lo[1]..=hi[1] {
                    for x in lo[0]..=hi[0] {
                        let d = [
                            x as f32 - centre[0],
                            y as f32 - centre[1],
                            z as f32 - centre[2],
                        ];
                        let q: f32 = (0..3)
                            .map(|m| {
                                ((r[0][m] * d[0] + r[1][m] * d[1] + r[2][m] * d[2]) * inverse[m])
                                    .powi(2)
                            })
                            .sum();
                        if q < GAUSSIAN_REACH * GAUSSIAN_REACH {
                            density[(z * dy + y) * dx + x] += opacity * (-0.5 * q).exp();
                        }
                    }
                }
            }
        }
        let occupied: Vec<bool> = density.iter().map(|&d| d >= DENSITY_THRESHOLD).collect();
        solid.seal(occupied, SPLAT_CLOSING);
        solid
    }

    /// The solid inside a shell of cells (`cells` of a `cell_res`^3 grid over the unit cube, TRELLIS.2's sparse
    /// voxels), on a `res`^3 grid; each cell marks the grid points within it, or the nearest when it holds none.
    pub fn from_voxels(cells: &[[i32; 3]], cell_res: usize, res: usize) -> Self {
        let scale = res as f32 / cell_res as f32;
        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        for cell in cells {
            for a in 0..3 {
                lo[a] = lo[a].min(cell[a] as f32 * scale);
                hi[a] = hi[a].max((cell[a] + 1) as f32 * scale);
            }
        }
        if cells.is_empty() {
            lo = [0.0; 3];
            hi = [0.0; 3];
        }
        let mut solid = Self::framed(res, lo, hi);
        let mut occupied = vec![false; solid.len()];
        let last = solid.dims.map(|d| (d - 1) as f32);
        let to_grid = |c: f32, a: usize| c * scale - solid.origin[a] as f32;
        for cell in cells {
            let lo: [usize; 3] = std::array::from_fn(|a| {
                to_grid(cell[a] as f32, a).ceil().clamp(0.0, last[a]) as usize
            });
            let hi: [usize; 3] = std::array::from_fn(|a| {
                to_grid(cell[a] as f32 + 1.0, a).floor().clamp(0.0, last[a]) as usize
            });
            if (0..3).all(|a| lo[a] <= hi[a]) {
                for z in lo[2]..=hi[2] {
                    for y in lo[1]..=hi[1] {
                        for x in lo[0]..=hi[0] {
                            occupied[solid.index([x, y, z])] = true;
                        }
                    }
                }
            } else {
                let nearest = std::array::from_fn(|a| {
                    to_grid(cell[a] as f32 + 0.5, a).round().clamp(0.0, last[a]) as usize
                });
                occupied[solid.index(nearest)] = true;
            }
        }
        solid.seal(occupied, VOXEL_CLOSING);
        solid
    }

    // `steps` face-neighbour steps of growth
    fn dilate(&self, mut mask: Vec<bool>, steps: usize) -> Vec<bool> {
        let strides = self.strides();
        for _ in 0..steps {
            mask = self.map(|i, c| {
                mask[i]
                    || (0..3).any(|a| {
                        (c[a] + 1 < self.dims[a] && mask[i + strides[a]])
                            || (c[a] > 0 && mask[i - strides[a]])
                    })
            });
        }
        mask
    }

    // `steps` of growth along each axis in turn: a cube, which (unlike a face-neighbour diamond) fits a box's edges
    fn dilate_cube(&self, mut mask: Vec<bool>, steps: usize) -> Vec<bool> {
        let strides = self.strides();
        for a in 0..3 {
            for _ in 0..steps {
                mask = self.map(|i, c| {
                    mask[i]
                        || (c[a] + 1 < self.dims[a] && mask[i + strides[a]])
                        || (c[a] > 0 && mask[i - strides[a]])
                });
            }
        }
        mask
    }

    // marks with `label` the face-connected points `open` admits from `seed`, which it admits too; their count
    fn fill(
        &self,
        seed: usize,
        label: u32,
        labels: &mut [u32],
        open: impl Fn(usize) -> bool,
    ) -> usize {
        let strides = self.strides();
        let mut stack = vec![(seed, self.coords(seed))];
        labels[seed] = label;
        let mut size = 0;
        while let Some((i, c)) = stack.pop() {
            size += 1;
            for a in 0..3 {
                if c[a] + 1 < self.dims[a] {
                    let j = i + strides[a];
                    if labels[j] == 0 && open(j) {
                        labels[j] = label;
                        let mut d = c;
                        d[a] += 1;
                        stack.push((j, d));
                    }
                }
                if c[a] > 0 {
                    let j = i - strides[a];
                    if labels[j] == 0 && open(j) {
                        labels[j] = label;
                        let mut d = c;
                        d[a] -= 1;
                        stack.push((j, d));
                    }
                }
            }
        }
        size
    }

    // what a flood from the border through free points misses is solid, closed over gaps up to about twice `closing`
    // voxels (the occupied points grown, flooded around, and the flood grown back), its thin parts thickened; keeps the
    // largest piece
    fn seal(&mut self, occupied: Vec<bool>, closing: usize) {
        let occupied = self.dilate(occupied, closing);
        let mut labels = vec![0u32; self.len()];
        // the corner is free: the padding keeps the border empty, and it is all one piece
        self.fill(0, 1, &mut labels, |j| !occupied[j]);
        let outside: Vec<bool> = labels.iter().map(|&l| l == 1).collect();
        let outside = self.dilate(outside, closing);
        let remains: Vec<bool> = outside.iter().map(|&o| !o).collect();
        let walled = self.thicken(remains);
        self.inside = self.largest_piece(&walled);
    }

    // parts of `solid` thinner than a printable wall (what an opening by a cube of MIN_WALL steps removes: an open
    // shell, a thin rod) grown by MIN_WALL steps; thicker parts keep their size
    fn thicken(&self, solid: Vec<bool>) -> Vec<bool> {
        let empty: Vec<bool> = solid.iter().map(|&b| !b).collect();
        let core: Vec<bool> = self
            .dilate_cube(empty, MIN_WALL)
            .iter()
            .map(|&b| !b)
            .collect();
        let opened = self.dilate_cube(core, MIN_WALL);
        let thin: Vec<bool> = solid.iter().zip(&opened).map(|(&s, &o)| s && !o).collect();
        let grown = self.dilate(thin, MIN_WALL);
        solid.iter().zip(&grown).map(|(&s, &g)| s || g).collect()
    }

    // the largest face-connected piece of `mask`
    fn largest_piece(&self, mask: &[bool]) -> Vec<bool> {
        let mut labels = vec![0u32; mask.len()];
        let mut best = (0u32, 0usize);
        let mut next = 0u32;
        for seed in 0..mask.len() {
            if !mask[seed] || labels[seed] != 0 {
                continue;
            }
            next += 1;
            let size = self.fill(seed, next, &mut labels, |j| mask[j]);
            if size > best.1 {
                best = (next, size);
            }
        }
        labels.iter().map(|&l| l != 0 && l == best.0).collect()
    }

    fn is_empty(&self) -> bool {
        !self.inside.iter().any(|&b| b)
    }

    // the point a cell's corners need solidified, if any, so no two solid (or empty) points meet only diagonally
    fn diagonal_fix(&self, base: usize, field: &[f32]) -> Option<usize> {
        let corners = self.corners(base);
        let solid: [bool; 8] = std::array::from_fn(|k| field[corners[k]] > ISO);
        for face in CELL_FACES {
            let f = face.map(|k| solid[k]);
            if f[0] == f[2] && f[1] == f[3] && f[0] != f[1] {
                return Some(corners[if f[0] { face[1] } else { face[0] }]);
            }
        }
        let count = solid.iter().filter(|&&b| b).count();
        (0..4).find_map(|k| match count {
            2 if solid[k] && solid[7 - k] => Some(corners[k ^ 1]),
            6 if !solid[k] && !solid[7 - k] => Some(corners[k]),
            _ => None,
        })
    }

    fn corners(&self, base: usize) -> [usize; 8] {
        let s = self.strides();
        std::array::from_fn(|k| base + (k & 1) * s[0] + ((k >> 1) & 1) * s[1] + (k >> 2) * s[2])
    }

    // whether the point at `c` is a cell's lower corner, its other corners on the grid
    fn is_cell(&self, c: [usize; 3]) -> bool {
        (0..3).all(|a| c[a] + 1 < self.dims[a])
    }

    // the cells the surface passes: their corners on both sides of the level
    fn crossed_cells(&self, field: &[f32]) -> Vec<usize> {
        self.points()
            .filter(|&(i, c)| {
                let count = || self.corners(i).iter().filter(|&&k| field[k] > ISO).count();
                self.is_cell(c) && !matches!(count(), 0 | 8)
            })
            .map(|(i, _)| i)
            .collect()
    }

    // solid points meeting only at a cell's edge or corner get one more beside them: surface nets would pinch the
    // surface there, an edge with four faces
    fn join_diagonals(&self, field: &mut [f32]) {
        let mut pending = self.crossed_cells(field);
        while let Some(base) = pending.pop() {
            let Some(point) = self.diagonal_fix(base, field) else {
                continue;
            };
            field[point] = JOINED;
            // the eight cells around the point look again
            let c = self.coords(point);
            for k in 0..8 {
                let offset = [k & 1, (k >> 1) & 1, k >> 2];
                if (0..3).all(|a| c[a] >= offset[a]) {
                    let at = std::array::from_fn(|a| c[a] - offset[a]);
                    if self.is_cell(at) {
                        pending.push(self.index(at));
                    }
                }
            }
        }
    }

    // the solid's surface: closed, its faces wound outward
    fn surface(&self) -> Mesh {
        let mut field: Vec<f32> = self
            .inside
            .iter()
            .map(|&b| if b { 1.0 } else { 0.0 })
            .collect();
        let kernel: Vec<f32> = {
            let w: Vec<f32> = (0..=2 * BLUR_RADIUS)
                .map(|k| (-0.5 * ((k as f32 - BLUR_RADIUS as f32) / BLUR_SIGMA).powi(2)).exp())
                .collect();
            let sum: f32 = w.iter().sum();
            w.iter().map(|v| v / sum).collect()
        };
        let strides = self.strides();
        for a in 0..3 {
            let stride = strides[a];
            field = self.map(|i, c| {
                kernel
                    .iter()
                    .enumerate()
                    .filter_map(|(k, w)| {
                        let at = (c[a] + k)
                            .checked_sub(BLUR_RADIUS)
                            .filter(|&at| at < self.dims[a])?;
                        Some(w * field[i + at * stride - c[a] * stride])
                    })
                    .sum()
            });
        }
        // the blur can leave a lone point over the level beside a thin part, or pinch off the bottom of a crevice
        let over: Vec<bool> = field.iter().map(|&f| f > ISO).collect();
        let keep = self.largest_piece(&over);
        let mut labels = vec![0u32; field.len()];
        self.fill(0, 1, &mut labels, |j| !keep[j]);
        for (i, f) in field.iter_mut().enumerate() {
            if !keep[i] && labels[i] == 0 {
                *f = JOINED;
            } else if !keep[i] && *f > ISO {
                *f = 0.0;
            }
        }
        self.join_diagonals(&mut field);
        let solid = |i: usize| field[i] > ISO;
        // one vertex per cell the surface passes, at the mean of its edges' crossings
        let cells = self.crossed_cells(&field);
        let vertices: Vec<[f32; 3]> = cells
            .par_iter()
            .map(|&base| {
                let corners = self.corners(base);
                let mut sum = [0f32; 3];
                let mut crossings = 0f32;
                for (k, &i) in corners.iter().enumerate() {
                    for a in 0..3 {
                        if k & (1 << a) != 0 {
                            continue;
                        }
                        let j = corners[k | (1 << a)];
                        if solid(i) == solid(j) {
                            continue;
                        }
                        let t = (ISO - field[i]) / (field[j] - field[i]);
                        let mut p = [(k & 1) as f32, ((k >> 1) & 1) as f32, (k >> 2) as f32];
                        p[a] += t;
                        for (s, q) in sum.iter_mut().zip(p) {
                            *s += q;
                        }
                        crossings += 1.0;
                    }
                }
                let local = self.coords(base);
                std::array::from_fn(|a| self.to_world(local[a] as f32 + sum[a] / crossings, a))
            })
            .collect();
        let vertex_of: HashMap<usize, u32> = cells
            .iter()
            .enumerate()
            .map(|(v, &cell)| (cell, v as u32))
            .collect();
        // a quad of the four cells around each crossed edge, wound so its normal leaves the solid
        let (vertex_of, solid) = (&vertex_of, &solid);
        let quads: Vec<[u32; 4]> = self
            .points()
            .flat_map_iter(|(i, c)| {
                (0..3).filter_map(move |a| {
                    if c[a] + 1 >= self.dims[a] || solid(i) == solid(i + strides[a]) {
                        return None;
                    }
                    let (u, v) = (strides[(a + 1) % 3], strides[(a + 2) % 3]);
                    // the border is empty, so a crossed edge always has its four cells
                    let mut quad = [i - u - v, i - v, i, i - u].map(|cell| vertex_of[&cell]);
                    if !solid(i) {
                        quad.reverse();
                    }
                    Some(quad)
                })
            })
            .collect();
        let length = |p: u32, q: u32| -> f32 {
            let (p, q) = (vertices[p as usize], vertices[q as usize]);
            (0..3).map(|k: usize| (p[k] - q[k]).powi(2)).sum()
        };
        let mut faces = Vec::with_capacity(quads.len() * 2);
        for quad in quads {
            if length(quad[0], quad[2]) <= length(quad[1], quad[3]) {
                faces.push([quad[0], quad[1], quad[2]]);
                faces.push([quad[0], quad[2], quad[3]]);
            } else {
                faces.push([quad[0], quad[1], quad[3]]);
                faces.push([quad[1], quad[2], quad[3]]);
            }
        }
        let mut mesh = Mesh { vertices, faces };
        smooth(&mut mesh, SMOOTH_ROUNDS);
        mesh
    }
}

// Taubin's smoothing: a shrinking Laplacian step then an inflating one, so ripples go and the volume stays; vertices
// only move, so the surface stays closed
fn smooth(mesh: &mut Mesh, rounds: usize) {
    let mut neighbours: Vec<Vec<u32>> = vec![Vec::new(); mesh.vertices.len()];
    for f in &mesh.faces {
        for k in 0..3 {
            let (a, b) = (f[k], f[(k + 1) % 3]);
            neighbours[a as usize].push(b);
            neighbours[b as usize].push(a);
        }
    }
    for list in &mut neighbours {
        list.sort_unstable();
        list.dedup();
    }
    for _ in 0..rounds {
        for factor in [SMOOTH_SHRINK, SMOOTH_INFLATE] {
            let moved: Vec<[f32; 3]> = neighbours
                .par_iter()
                .zip(&mesh.vertices)
                .map(|(around, &v)| {
                    if around.is_empty() {
                        return v;
                    }
                    let mut mean = [0f32; 3];
                    for &j in around {
                        let p = mesh.vertices[j as usize];
                        for a in 0..3 {
                            mean[a] += p[a];
                        }
                    }
                    std::array::from_fn(|a| v[a] + factor * (mean[a] / around.len() as f32 - v[a]))
                })
                .collect();
            mesh.vertices = moved;
        }
    }
}

/// An STL ready to print and what it holds.
pub struct Printable {
    pub stl: Vec<u8>,
    pub faces: usize,
    pub volume_mm3: f64,
}

impl Solid {
    /// The solid's surface as an STL whose longest side is `size_mm`, simplified to about PRINT_FACES when that keeps
    /// it closed; `None` when nothing is solid.
    pub fn printable(&self, size_mm: f64) -> Option<Printable> {
        if self.is_empty() {
            return None;
        }
        let mut surface = self.surface();
        if surface.faces.len() > PRINT_FACES {
            let mut simpler = surface.clone();
            crate::trellis2::cleanup::simplify_manifold(&mut simpler, PRINT_FACES);
            let closed = is_closed(&simpler);
            if closed {
                surface = simpler;
            }
        }
        Some(Printable {
            stl: to_stl(&surface, size_mm),
            faces: surface.faces.len(),
            volume_mm3: signed_volume(&surface) * (size_mm / f64::from(extent(&surface))).powi(3),
        })
    }
}

// every directed edge once and its reverse once: closed, manifold along edges, consistently wound
fn is_closed(mesh: &Mesh) -> bool {
    let mut directed: HashMap<(u32, u32), u32> = HashMap::with_capacity(mesh.faces.len() * 3);
    for f in &mesh.faces {
        for k in 0..3 {
            let count = directed.entry((f[k], f[(k + 1) % 3])).or_default();
            *count += 1;
            if *count > 1 {
                return false;
            }
        }
    }
    directed
        .keys()
        .all(|&(a, b)| directed.contains_key(&(b, a)))
}

// the volume a closed, outward-wound mesh encloses
fn signed_volume(mesh: &Mesh) -> f64 {
    mesh.faces
        .iter()
        .map(|f| {
            let [a, b, c] = f.map(|i| mesh.vertices[i as usize].map(f64::from));
            (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0]))
                / 6.0
        })
        .sum()
}

// the corners of the mesh's bounding box
fn bounds(mesh: &Mesh) -> ([f32; 3], [f32; 3]) {
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for v in &mesh.vertices {
        for a in 0..3 {
            lo[a] = lo[a].min(v[a]);
            hi[a] = hi[a].max(v[a]);
        }
    }
    (lo, hi)
}

// the bounding box's longest side
fn extent(mesh: &Mesh) -> f32 {
    let (lo, hi) = bounds(mesh);
    (0..3)
        .map(|a| hi[a] - lo[a])
        .fold(0f32, f32::max)
        .max(f32::MIN_POSITIVE)
}

// a binary STL of `mesh` (+z up) in millimetres, its longest side `size_mm`, centred over the origin and standing on
// z = 0
fn to_stl(mesh: &Mesh, size_mm: f64) -> Vec<u8> {
    let (lo, hi) = bounds(mesh);
    let scale = size_mm as f32 / extent(mesh);
    let origin = [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, lo[2]];
    let place = |v: [f32; 3]| -> [f32; 3] { std::array::from_fn(|a| (v[a] - origin[a]) * scale) };
    let mut out = Vec::with_capacity(STL_HEADER_LEN + 4 + mesh.faces.len() * 50);
    out.extend_from_slice(STL_HEADER);
    out.resize(STL_HEADER_LEN, 0);
    out.extend_from_slice(&(mesh.faces.len() as u32).to_le_bytes());
    for f in &mesh.faces {
        let [a, b, c] = f.map(|i| place(mesh.vertices[i as usize]));
        let (e, g) = (
            [0, 1, 2].map(|k| b[k] - a[k]),
            [0, 1, 2].map(|k| c[k] - a[k]),
        );
        let cross = [
            e[1] * g[2] - e[2] * g[1],
            e[2] * g[0] - e[0] * g[2],
            e[0] * g[1] - e[1] * g[0],
        ];
        let length = cross
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt()
            .max(f32::MIN_POSITIVE);
        for v in cross
            .map(|v| v / length)
            .iter()
            .chain(&a)
            .chain(&b)
            .chain(&c)
        {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&0u16.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // every directed edge once and its reverse once: closed, manifold along edges, consistently wound
    fn assert_closed(mesh: &Mesh) {
        let mut directed: HashMap<(u32, u32), usize> = HashMap::new();
        for f in &mesh.faces {
            for k in 0..3 {
                *directed.entry((f[k], f[(k + 1) % 3])).or_default() += 1;
            }
        }
        for (&(a, b), &count) in &directed {
            assert_eq!(count, 1, "edge {a}-{b} used {count} times one way");
            assert_eq!(directed.get(&(b, a)), Some(&1), "edge {a}-{b} has no twin");
        }
    }

    fn cube(side: f32) -> Mesh {
        let h = side / 2.0;
        let vertices = (0..8)
            .map(|k| [k & 1, (k >> 1) & 1, k >> 2].map(|b| if b == 1 { h } else { -h }))
            .collect();
        // outward winding, two triangles a side
        let faces = vec![
            [0, 2, 1],
            [1, 2, 3],
            [4, 5, 6],
            [5, 7, 6],
            [0, 1, 4],
            [1, 5, 4],
            [2, 6, 3],
            [3, 6, 7],
            [0, 4, 2],
            [2, 4, 6],
            [1, 3, 5],
            [3, 7, 5],
        ];
        Mesh { vertices, faces }
    }

    // the cells on the faces of the box `lo..hi` (cell coordinates), its top left off when `open`
    fn box_shell(lo: i32, hi: i32, open: bool) -> Vec<[i32; 3]> {
        let mut cells = Vec::new();
        for z in lo..hi {
            for y in lo..hi {
                for x in lo..hi {
                    let edge = |c: i32| c == lo || c == hi - 1;
                    let top = z == hi - 1 && !edge(x) && !edge(y);
                    if (edge(x) || edge(y) || edge(z)) && !(open && top) {
                        cells.push([x, y, z]);
                    }
                }
            }
        }
        cells
    }

    #[test]
    fn a_closed_shell_of_voxels_fills_to_its_volume() {
        // cells finer than the grid (each marks its nearest point) and coarser (each marks the points within it)
        let res = 64;
        for (cell_res, lo, hi) in [(128, 26, 102), (32, 6, 26)] {
            let solid = Solid::from_voxels(&box_shell(lo, hi, false), cell_res, res);
            let surface = solid.surface();
            assert_closed(&surface);
            // the cells straddle the surface: its faces run through their centres, give or take the grid's rounding
            let face = |c: i32| (c as f32 + 0.5) / cell_res as f32 - 0.5;
            let (low, high) = bounds(&surface);
            for a in 0..3 {
                let off = (low[a] - face(lo))
                    .abs()
                    .max((high[a] - face(hi - 1)).abs())
                    * res as f32;
                // a coarse cell marks all of itself, half a cell past its centre
                let allowed = 1.0 + 0.5 * (res as f32 / cell_res as f32).max(1.0);
                assert!(off < allowed, "{cell_res}: a face {off} voxels off");
            }
            let filled = (f64::from(hi - lo - 1) / cell_res as f64).powi(3);
            assert!(
                signed_volume(&surface) > 0.9 * filled,
                "{cell_res}: not filled"
            );
        }
    }

    #[test]
    fn an_open_shell_prints_as_walls_and_a_sheet_thickens() {
        let res = 64;
        let open = Solid::from_voxels(&box_shell(26, 102, true), 128, res);
        assert_closed(&open.surface());
        let centre = std::array::from_fn(|a| (res as i64 / 2 - open.origin[a]) as usize);
        assert!(!open.inside[open.index(centre)], "an open box filled");
        // one cell thick: thinner than the blur would keep, so it grows to a wall
        let sheet: Vec<[i32; 3]> = (40..90)
            .flat_map(|x| (40..90).map(move |y| [x, y, 64]))
            .collect();
        let surface = Solid::from_voxels(&sheet, 128, res).surface();
        assert_closed(&surface);
        let (lo, hi) = bounds(&surface);
        let thickness = (hi[2] - lo[2]) * res as f32;
        assert!(thickness >= (2 * MIN_WALL) as f32, "{thickness} voxels");
    }

    #[test]
    fn solids_meeting_along_an_edge_or_at_a_corner_join_into_one_shell() {
        for other in [[1, 1, 0], [1, 1, 1]] {
            let mut solid = Solid::framed(16, [0.0; 3], [16.0; 3]);
            solid.inside = vec![false; solid.len()];
            for corner in [[0, 0, 0], other] {
                for z in 0..4 {
                    for y in 0..4 {
                        for x in 0..4 {
                            let at = [x, y, z].map(|c| c + 4 + PAD);
                            let shifted: [usize; 3] =
                                std::array::from_fn(|a| at[a] + corner[a] * 4);
                            let i = solid.index(shifted);
                            solid.inside[i] = true;
                        }
                    }
                }
            }
            assert_closed(&solid.surface());
        }
    }

    #[test]
    fn a_shell_of_flat_gaussians_fills_and_drops_a_stray_one() {
        // flat discs tiling a sphere, each turned so its thin axis (z) points outward, and one far floater
        let radius = 0.3f32;
        let mut g = Gaussians {
            xyz: Vec::new(),
            dc: Vec::new(),
            scaling: Vec::new(),
            rotation: Vec::new(),
            opacity: Vec::new(),
        };
        let golden = std::f32::consts::PI * (3.0 - 5f32.sqrt());
        let count = 6000;
        for i in 0..count {
            let z = 1.0 - 2.0 * (i as f32 + 0.5) / count as f32;
            let r = (1.0 - z * z).sqrt();
            let n = [
                r * (golden * i as f32).cos(),
                r * (golden * i as f32).sin(),
                z,
            ];
            // the quaternion taking +z to n
            let (w, axis) = (1.0 + n[2], [-n[1], n[0], 0.0]);
            g.xyz.push(n.map(|c| c * radius));
            g.dc.push([0.0; 3]);
            g.scaling.push([0.012, 0.012, 0.0005]);
            g.rotation.push([w, axis[0], axis[1], axis[2]]);
            g.opacity.push(0.9);
        }
        g.xyz.push([0.45, 0.45, 0.45]);
        g.dc.push([0.0; 3]);
        g.scaling.push([0.01; 3]);
        g.rotation.push([1.0, 0.0, 0.0, 0.0]);
        g.opacity.push(1.0);
        let res = 64;
        let solid = Solid::from_gaussians(&g, res);
        let surface = solid.surface();
        assert_closed(&surface);
        // filled, not a hollow shell, and the floater gone: every vertex near the sphere
        let filled = 4.0 / 3.0 * std::f64::consts::PI * f64::from(radius).powi(3);
        assert!(
            signed_volume(&surface) > filled,
            "{} vs {filled}",
            signed_volume(&surface)
        );
        let off: f32 = surface
            .vertices
            .iter()
            .map(|v| ((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt() - radius).abs())
            .fold(0.0, f32::max);
        println!(
            "worst distance from the sphere: {:.2} voxels",
            off * res as f32
        );
        assert!(off * (res as f32) < 2.5, "{off}");
    }

    #[test]
    fn the_stl_is_sized_and_stands_on_the_bed() {
        let stl = to_stl(&cube(0.5), 40.0);
        assert_eq!(stl.len(), STL_HEADER_LEN + 4 + 12 * 50);
        assert_eq!(u32::from_le_bytes(stl[80..84].try_into().unwrap()), 12);
        let floats: Vec<f32> = stl[84..]
            .chunks(50)
            .flat_map(|t| {
                t[12..48]
                    .chunks(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect::<Vec<_>>()
            })
            .collect();
        let zs: Vec<f32> = floats.chunks(3).map(|p| p[2]).collect();
        assert_eq!(zs.iter().cloned().fold(f32::INFINITY, f32::min), 0.0);
        assert!((zs.iter().cloned().fold(0.0, f32::max) - 40.0).abs() < 1e-4);
    }
}
