//! A mesh from trained splats: their RGB and expected depth rendered from every camera, fused into a sparse TSDF the
//! way Open3D's ScalableTSDFVolume integrates (each voxel projected to its nearest pixel, the truncated signed
//! distance averaged with unit weights), the zero level set extracted by surface nets, and only the largest connected
//! piece kept (world_gs_trainer's export_mesh).

use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

use inference_tensor::Result;
use rayon::prelude::*;

// blocks of BLOCK^3 voxels, allocated where a frame's depth lands
const BLOCK: i32 = 8;
const BLOCK_VOXELS: usize = (BLOCK * BLOCK * BLOCK) as usize;

/// One fused frame: z-depth (0 where empty) and rgb in [0, 1], both row-major `h * w`.
pub struct DepthFrame<'a> {
    pub depth: &'a [f32],
    pub rgb: &'a [[f32; 3]],
    pub width: usize,
    pub height: usize,
    /// world-to-camera, row-major, OpenCV axes
    pub w2c: [[f64; 4]; 4],
    pub k: [[f64; 3]; 3],
}

#[derive(Clone, Copy, Debug)]
pub struct TsdfSettings {
    pub voxel: f64,
    pub sdf_trunc: f64,
    pub depth_trunc: f64,
}

#[derive(Clone)]
struct Block {
    tsdf: Box<[f32; BLOCK_VOXELS]>,
    weight: Box<[f32; BLOCK_VOXELS]>,
    color: Box<[[f32; 3]; BLOCK_VOXELS]>,
}

impl Block {
    fn new() -> Self {
        Self {
            tsdf: Box::new([0.0; BLOCK_VOXELS]),
            weight: Box::new([0.0; BLOCK_VOXELS]),
            color: Box::new([[0.0; 3]; BLOCK_VOXELS]),
        }
    }
}

pub struct Tsdf {
    settings: TsdfSettings,
    index: HashMap<[i32; 3], usize>,
    blocks: Vec<Block>,
}

pub struct Mesh {
    pub vertices: Vec<[f32; 3]>,
    pub colors: Vec<[u8; 3]>,
    pub triangles: Vec<[u32; 3]>,
}

fn voxel_index(v: [i32; 3]) -> usize {
    ((v[2] * BLOCK + v[1]) * BLOCK + v[0]) as usize
}

impl Tsdf {
    pub fn new(settings: TsdfSettings) -> Self {
        Self {
            settings,
            index: HashMap::new(),
            blocks: Vec::new(),
        }
    }

    fn block_of(&self, p: [f64; 3]) -> [i32; 3] {
        let size = self.settings.voxel * f64::from(BLOCK);
        p.map(|v| (v / size).floor() as i32)
    }

    /// Fuses `frame`: allocates the blocks its depth (within the truncation band) touches, then integrates them.
    pub fn integrate(&mut self, frame: &DepthFrame) {
        let s = self.settings;
        let c2w = crate::geometry::invert4(&frame.w2c);
        let (fx, fy, cx, cy) = (frame.k[0][0], frame.k[1][1], frame.k[0][2], frame.k[1][2]);
        // touched blocks: the depth point and the band either side of it along the ray
        let mut touched = (0..frame.height)
            .into_par_iter()
            .flat_map_iter(|y| {
                let mut out = Vec::new();
                for x in 0..frame.width {
                    let d = f64::from(frame.depth[y * frame.width + x]);
                    if d <= 0.0 || d > s.depth_trunc {
                        continue;
                    }
                    let ray = [(x as f64 - cx) / fx, (y as f64 - cy) / fy, 1.0];
                    for t in [d - s.sdf_trunc, d, d + s.sdf_trunc] {
                        if t <= 0.0 {
                            continue;
                        }
                        let pc = ray.map(|r| r * t);
                        let pw = [0, 1, 2].map(|i| {
                            c2w[i][0] * pc[0] + c2w[i][1] * pc[1] + c2w[i][2] * pc[2] + c2w[i][3]
                        });
                        out.push(self.block_of(pw));
                    }
                }
                out
            })
            .collect::<Vec<_>>();
        touched.par_sort_unstable();
        touched.dedup();
        let ids = touched
            .iter()
            .map(|b| {
                let next = self.blocks.len();
                let id = *self.index.entry(*b).or_insert(next);
                if id == next {
                    self.blocks.push(Block::new());
                }
                id
            })
            .collect::<Vec<_>>();
        let w2c = frame.w2c;
        let mut work = ids.iter().copied().zip(touched).collect::<Vec<_>>();
        work.sort_unstable_by_key(|&(id, _)| id);
        // each block once per frame: split the block storage into the touched ones
        let mut slots: Vec<Option<&mut Block>> = self.blocks.iter_mut().map(Some).collect();
        let jobs = work
            .into_iter()
            .map(|(id, b)| (slots[id].take().expect("a block once per frame"), b))
            .collect::<Vec<_>>();
        jobs.into_par_iter().for_each(|(block, b)| {
            for z in 0..BLOCK {
                for y in 0..BLOCK {
                    for x in 0..BLOCK {
                        let pw = [
                            f64::from(b[0] * BLOCK + x) * s.voxel,
                            f64::from(b[1] * BLOCK + y) * s.voxel,
                            f64::from(b[2] * BLOCK + z) * s.voxel,
                        ];
                        let pc = [0, 1, 2].map(|i| {
                            w2c[i][0] * pw[0] + w2c[i][1] * pw[1] + w2c[i][2] * pw[2] + w2c[i][3]
                        });
                        if pc[2] <= 0.0 {
                            continue;
                        }
                        let u = (fx * pc[0] / pc[2] + cx).round();
                        let v = (fy * pc[1] / pc[2] + cy).round();
                        if u < 0.0 || v < 0.0 || u >= frame.width as f64 || v >= frame.height as f64
                        {
                            continue;
                        }
                        let p = v as usize * frame.width + u as usize;
                        let d = f64::from(frame.depth[p]);
                        if d <= 0.0 || d > s.depth_trunc {
                            continue;
                        }
                        let sdf = d - pc[2];
                        if sdf < -s.sdf_trunc {
                            continue;
                        }
                        let tsdf = (sdf / s.sdf_trunc).min(1.0) as f32;
                        let i = voxel_index([x, y, z]);
                        let w = block.weight[i];
                        block.tsdf[i] = (block.tsdf[i] * w + tsdf) / (w + 1.0);
                        for c in 0..3 {
                            block.color[i][c] =
                                (block.color[i][c] * w + frame.rgb[p][c]) / (w + 1.0);
                        }
                        block.weight[i] = w + 1.0;
                    }
                }
            }
        });
    }

    // the voxel at global integer coordinates: (tsdf, colour) where observed
    fn sample(&self, g: [i32; 3]) -> Option<(f32, [f32; 3])> {
        let b = g.map(|v| v.div_euclid(BLOCK));
        let l = g.map(|v| v.rem_euclid(BLOCK));
        let block = &self.blocks[*self.index.get(&b)?];
        let i = voxel_index(l);
        (block.weight[i] > 0.0).then(|| (block.tsdf[i], block.color[i]))
    }

    /// The zero level set by surface nets: a vertex per cell whose observed corners change sign (the mean of its edge
    /// crossings), a quad per sign-changing voxel edge between the four cells around it.
    pub fn extract(&self) -> Mesh {
        let voxel = self.settings.voxel as f32;
        let keys = self.index.keys().copied().collect::<Vec<_>>();
        const CORNERS: [[i32; 3]; 8] = [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ];
        const EDGES: [(usize, usize); 12] = [
            (0, 1),
            (2, 3),
            (4, 5),
            (6, 7),
            (0, 2),
            (1, 3),
            (4, 6),
            (5, 7),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ];
        let cells = keys
            .par_iter()
            .flat_map_iter(|b| {
                let mut out = Vec::new();
                for z in 0..BLOCK {
                    for y in 0..BLOCK {
                        for x in 0..BLOCK {
                            let g = [b[0] * BLOCK + x, b[1] * BLOCK + y, b[2] * BLOCK + z];
                            let mut corners = [(0f32, [0f32; 3]); 8];
                            let mut ok = true;
                            for (c, o) in CORNERS.iter().enumerate() {
                                match self.sample([g[0] + o[0], g[1] + o[1], g[2] + o[2]]) {
                                    Some(v) => corners[c] = v,
                                    None => {
                                        ok = false;
                                        break;
                                    }
                                }
                            }
                            if !ok {
                                continue;
                            }
                            let mut sum = [0f32; 3];
                            let mut col = [0f32; 3];
                            let mut n = 0f32;
                            for &(a, e) in &EDGES {
                                let (ta, tb) = (corners[a].0, corners[e].0);
                                if (ta < 0.0) == (tb < 0.0) {
                                    continue;
                                }
                                let t = ta / (ta - tb);
                                for d in 0..3 {
                                    let pa = CORNERS[a][d] as f32;
                                    let pb = CORNERS[e][d] as f32;
                                    sum[d] += pa + t * (pb - pa);
                                    col[d] +=
                                        corners[a].1[d] + t * (corners[e].1[d] - corners[a].1[d]);
                                }
                                n += 1.0;
                            }
                            if n > 0.0 {
                                let pos = [0, 1, 2].map(|d| (g[d] as f32 + sum[d] / n) * voxel);
                                out.push((
                                    g,
                                    pos,
                                    col.map(|c| (c / n * 255.0).round().clamp(0.0, 255.0) as u8),
                                ));
                            }
                        }
                    }
                }
                out
            })
            .collect::<Vec<_>>();
        let cell_index = cells
            .iter()
            .enumerate()
            .map(|(i, (g, _, _))| (*g, i as u32))
            .collect::<HashMap<_, _>>();
        // quads: each voxel edge from g along axis a with a sign change, between the four cells sharing it
        let triangles = cells
            .par_iter()
            .flat_map_iter(|&(g, _, _)| {
                let mut out = Vec::new();
                let Some((t0, _)) = self.sample(g) else {
                    return out;
                };
                for axis in 0..3 {
                    let mut next = g;
                    next[axis] += 1;
                    let Some((t1, _)) = self.sample(next) else {
                        continue;
                    };
                    if (t0 < 0.0) == (t1 < 0.0) {
                        continue;
                    }
                    let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
                    let cell = |du: i32, dv: i32| {
                        let mut c = g;
                        c[u] -= du;
                        c[v] -= dv;
                        cell_index.get(&c).copied()
                    };
                    let (Some(a), Some(b), Some(c), Some(d)) =
                        (cell(0, 0), cell(1, 0), cell(1, 1), cell(0, 1))
                    else {
                        continue;
                    };
                    // facing the outside (positive tsdf)
                    if t0 < 0.0 {
                        out.push([a, b, c]);
                        out.push([a, c, d]);
                    } else {
                        out.push([a, c, b]);
                        out.push([a, d, c]);
                    }
                }
                out
            })
            .collect::<Vec<_>>();
        Mesh {
            vertices: cells.iter().map(|c| c.1).collect(),
            colors: cells.iter().map(|c| c.2).collect(),
            triangles,
        }
    }
}

impl Mesh {
    /// Only the largest connected piece (by triangles), unreferenced vertices dropped.
    pub fn largest_component(self) -> Self {
        let n = self.vertices.len();
        let mut parent = (0..n as u32).collect::<Vec<_>>();
        fn find(parent: &mut [u32], mut x: u32) -> u32 {
            while parent[x as usize] != x {
                parent[x as usize] = parent[parent[x as usize] as usize];
                x = parent[x as usize];
            }
            x
        }
        for t in &self.triangles {
            let r0 = find(&mut parent, t[0]);
            for &v in &t[1..] {
                let r = find(&mut parent, v);
                if r != r0 {
                    parent[r as usize] = r0;
                }
            }
        }
        let mut size: HashMap<u32, usize> = HashMap::new();
        let roots = self
            .triangles
            .iter()
            .map(|t| find(&mut parent, t[0]))
            .collect::<Vec<_>>();
        for r in &roots {
            *size.entry(*r).or_default() += 1;
        }
        let Some((&best, _)) = size.iter().max_by_key(|&(_, s)| *s) else {
            return self;
        };
        let mut remap = vec![u32::MAX; n];
        let mut vertices = Vec::new();
        let mut colors = Vec::new();
        let mut triangles = Vec::new();
        for (t, r) in self.triangles.iter().zip(&roots) {
            if *r != best {
                continue;
            }
            triangles.push(t.map(|v| {
                if remap[v as usize] == u32::MAX {
                    remap[v as usize] = vertices.len() as u32;
                    vertices.push(self.vertices[v as usize]);
                    colors.push(self.colors[v as usize]);
                }
                remap[v as usize]
            }));
        }
        Self {
            vertices,
            colors,
            triangles,
        }
    }

    pub fn write_ply(&self, path: &Path) -> Result<()> {
        let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
        write!(
            out,
            "ply\nformat binary_little_endian 1.0\nelement vertex {}\nproperty float x\nproperty float y\n\
             property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nelement face {}\n\
             property list uchar int vertex_indices\nend_header\n",
            self.vertices.len(),
            self.triangles.len()
        )?;
        for (v, c) in self.vertices.iter().zip(&self.colors) {
            for x in v {
                out.write_all(&x.to_le_bytes())?;
            }
            out.write_all(c)?;
        }
        for t in &self.triangles {
            out.write_all(&[3])?;
            for i in t {
                out.write_all(&(*i as i32).to_le_bytes())?;
            }
        }
        out.flush()?;
        Ok(())
    }
}

// a symmetric 4x4 error quadric (Garland-Heckbert), upper triangle a2 ab ac ad b2 bc bd c2 cd d2
#[derive(Clone, Copy, Default)]
struct Quadric([f64; 10]);

impl Quadric {
    fn plane(n: [f64; 3], d: f64, w: f64) -> Self {
        let [a, b, c] = n;
        Self(
            [
                a * a,
                a * b,
                a * c,
                a * d,
                b * b,
                b * c,
                b * d,
                c * c,
                c * d,
                d * d,
            ]
            .map(|v| v * w),
        )
    }

    fn add(&self, o: &Self) -> Self {
        Self(std::array::from_fn(|i| self.0[i] + o.0[i]))
    }

    fn error(&self, p: [f64; 3]) -> f64 {
        let q = &self.0;
        let [x, y, z] = p;
        q[0] * x * x
            + 2.0 * q[1] * x * y
            + 2.0 * q[2] * x * z
            + 2.0 * q[3] * x
            + q[4] * y * y
            + 2.0 * q[5] * y * z
            + 2.0 * q[6] * y
            + q[7] * z * z
            + 2.0 * q[8] * z
            + q[9]
    }

    // the minimizer, when the 3x3 block is well conditioned
    fn optimum(&self) -> Option<[f64; 3]> {
        let q = &self.0;
        let m = [[q[0], q[1], q[2]], [q[1], q[4], q[5]], [q[2], q[5], q[7]]];
        let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        if det.abs() < 1e-12 {
            return None;
        }
        let b = [-q[3], -q[6], -q[8]];
        let solve = |col: usize| {
            let mut mm = m;
            for r in 0..3 {
                mm[r][col] = b[r];
            }
            (mm[0][0] * (mm[1][1] * mm[2][2] - mm[1][2] * mm[2][1])
                - mm[0][1] * (mm[1][0] * mm[2][2] - mm[1][2] * mm[2][0])
                + mm[0][2] * (mm[1][0] * mm[2][1] - mm[1][1] * mm[2][0]))
                / det
        };
        Some([solve(0), solve(1), solve(2)])
    }
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

// boundary edges are held by planes through them perpendicular to their face, this much heavier than a face plane
const BOUNDARY_WEIGHT: f64 = 1e3;

#[derive(PartialEq)]
struct Candidate {
    cost: f64,
    edge: (u32, u32),
    stamp: (u32, u32),
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}

impl Ord for Candidate {
    // a min-heap on cost
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        o.cost.total_cmp(&self.cost)
    }
}

impl Mesh {
    /// Quadric edge collapse (Garland-Heckbert) down to about `target` triangles: each vertex carries the summed
    /// planes of its faces (boundary edges held by heavy perpendicular planes), the cheapest edge collapses to its
    /// quadric's optimum (or the better endpoint or midpoint), and collapses that would flip a face are skipped.
    pub fn decimate(self, target: usize) -> Self {
        use std::collections::{BinaryHeap, HashSet};
        let n = self.vertices.len();
        let mut pos = self
            .vertices
            .iter()
            .map(|v| v.map(f64::from))
            .collect::<Vec<_>>();
        let mut col = self
            .colors
            .iter()
            .map(|c| c.map(f64::from))
            .collect::<Vec<_>>();
        let mut tris = self.triangles.clone();
        let mut alive = vec![true; tris.len()];
        let mut faces: Vec<Vec<u32>> = vec![Vec::new(); n];
        for (t, tri) in tris.iter().enumerate() {
            for &v in tri {
                faces[v as usize].push(t as u32);
            }
        }
        let mut quad = vec![Quadric::default(); n];
        let mut edge_faces: HashMap<(u32, u32), u32> = HashMap::new();
        for tri in &tris {
            let [a, b, c] = tri.map(|v| pos[v as usize]);
            let nrm = cross3(sub3(b, a), sub3(c, a));
            let len = dot3(nrm, nrm).sqrt();
            if len > 0.0 {
                let unit = nrm.map(|v| v / len);
                let q = Quadric::plane(unit, -dot3(unit, a), 1.0);
                for &v in tri {
                    quad[v as usize] = quad[v as usize].add(&q);
                }
            }
            for k in 0..3 {
                let (u, v) = (tri[k].min(tri[(k + 1) % 3]), tri[k].max(tri[(k + 1) % 3]));
                *edge_faces.entry((u, v)).or_default() += 1;
            }
        }
        for tri in &tris {
            let [a, b, c] = tri.map(|v| pos[v as usize]);
            let nrm = cross3(sub3(b, a), sub3(c, a));
            for k in 0..3 {
                let (u, v) = (tri[k], tri[(k + 1) % 3]);
                if edge_faces[&(u.min(v), u.max(v))] != 1 {
                    continue;
                }
                let e = sub3(pos[v as usize], pos[u as usize]);
                let p = cross3(e, nrm);
                let len = dot3(p, p).sqrt();
                if len > 0.0 {
                    let unit = p.map(|x| x / len);
                    let q = Quadric::plane(unit, -dot3(unit, pos[u as usize]), BOUNDARY_WEIGHT);
                    quad[u as usize] = quad[u as usize].add(&q);
                    quad[v as usize] = quad[v as usize].add(&q);
                }
            }
        }
        let mut version = vec![0u32; n];
        let mut vertex_alive = vec![true; n];
        let place = |quad: &[Quadric], pos: &[[f64; 3]], u: u32, v: u32| -> ([f64; 3], f64) {
            let q = quad[u as usize].add(&quad[v as usize]);
            let (pu, pv) = (pos[u as usize], pos[v as usize]);
            let mid = [0, 1, 2].map(|i| 0.5 * (pu[i] + pv[i]));
            let mut best = (mid, q.error(mid));
            for p in [Some(pu), Some(pv), q.optimum()].into_iter().flatten() {
                let e = q.error(p);
                if e < best.1 {
                    best = (p, e);
                }
            }
            best
        };
        let mut heap = BinaryHeap::new();
        for &(u, v) in edge_faces.keys() {
            let (_, cost) = place(&quad, &pos, u, v);
            heap.push(Candidate {
                cost,
                edge: (u, v),
                stamp: (0, 0),
            });
        }
        let mut live = tris.len();
        while live > target {
            let Some(Candidate {
                edge: (u, v),
                stamp,
                ..
            }) = heap.pop()
            else {
                break;
            };
            if !vertex_alive[u as usize]
                || !vertex_alive[v as usize]
                || stamp != (version[u as usize], version[v as usize])
            {
                continue;
            }
            let (p, _) = place(&quad, &pos, u, v);
            // faces of u and v that survive must keep their orientation
            let mut flips = false;
            for &w in &[u, v] {
                for &t in &faces[w as usize] {
                    if !alive[t as usize] {
                        continue;
                    }
                    let tri = tris[t as usize];
                    if tri.contains(&u) && tri.contains(&v) {
                        continue;
                    }
                    let before = tri.map(|x| pos[x as usize]);
                    let after = tri.map(|x| if x == w { p } else { pos[x as usize] });
                    let n0 = cross3(sub3(before[1], before[0]), sub3(before[2], before[0]));
                    let n1 = cross3(sub3(after[1], after[0]), sub3(after[2], after[0]));
                    if dot3(n0, n1) <= 0.0 {
                        flips = true;
                    }
                }
            }
            if flips {
                continue;
            }
            // collapse v into u
            pos[u as usize] = p;
            col[u as usize] = [0, 1, 2].map(|i| 0.5 * (col[u as usize][i] + col[v as usize][i]));
            quad[u as usize] = quad[u as usize].add(&quad[v as usize]);
            vertex_alive[v as usize] = false;
            let moved = std::mem::take(&mut faces[v as usize]);
            for t in moved {
                if !alive[t as usize] {
                    continue;
                }
                let tri = &mut tris[t as usize];
                if tri.contains(&u) {
                    alive[t as usize] = false;
                    live -= 1;
                    continue;
                }
                for x in tri.iter_mut() {
                    if *x == v {
                        *x = u;
                    }
                }
                faces[u as usize].push(t);
            }
            faces[u as usize].retain(|&t| alive[t as usize]);
            version[u as usize] += 1;
            let mut neighbours = HashSet::new();
            for &t in &faces[u as usize] {
                for &x in &tris[t as usize] {
                    if x != u {
                        neighbours.insert(x);
                    }
                }
            }
            for x in neighbours {
                let (a, b) = (u.min(x), u.max(x));
                let (_, cost) = place(&quad, &pos, a, b);
                heap.push(Candidate {
                    cost,
                    edge: (a, b),
                    stamp: (version[a as usize], version[b as usize]),
                });
            }
        }
        let mut remap = vec![u32::MAX; n];
        let mut out = Self {
            vertices: Vec::new(),
            colors: Vec::new(),
            triangles: Vec::new(),
        };
        for (t, tri) in tris.iter().enumerate() {
            if !alive[t] {
                continue;
            }
            out.triangles.push(tri.map(|v| {
                if remap[v as usize] == u32::MAX {
                    remap[v as usize] = out.vertices.len() as u32;
                    out.vertices.push(pos[v as usize].map(|x| x as f32));
                    out.colors
                        .push(col[v as usize].map(|c| c.round().clamp(0.0, 255.0) as u8));
                }
                remap[v as usize]
            }));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // a sphere seen from six axis cameras fuses to a closed surface at its radius
    #[test]
    fn fuses_a_sphere() {
        const R: f64 = 1.0;
        const SIZE: usize = 96;
        const DIST: f64 = 3.0;
        let mut tsdf = Tsdf::new(TsdfSettings {
            voxel: 0.05,
            sdf_trunc: 0.2,
            depth_trunc: 10.0,
        });
        let f = SIZE as f64;
        let k = [[f, 0.0, f / 2.0], [0.0, f, f / 2.0], [0.0, 0.0, 1.0]];
        for (eye, up) in [
            ([DIST, 0.0, 0.0], [0.0, 0.0, 1.0]),
            ([-DIST, 0.0, 0.0], [0.0, 0.0, 1.0]),
            ([0.0, DIST, 0.0], [0.0, 0.0, 1.0]),
            ([0.0, -DIST, 0.0], [0.0, 0.0, 1.0]),
            ([0.0, 0.0, DIST], [1.0, 0.0, 0.0]),
            ([0.0, 0.0, -DIST], [1.0, 0.0, 0.0]),
        ] {
            // look at the origin: z forward, y down
            let zc = eye.map(|v: f64| -v / DIST);
            let yc = up.map(|v: f64| -v);
            let xc = [
                yc[1] * zc[2] - yc[2] * zc[1],
                yc[2] * zc[0] - yc[0] * zc[2],
                yc[0] * zc[1] - yc[1] * zc[0],
            ];
            let mut w2c = [[0.0; 4]; 4];
            for (r, axis) in [xc, yc, zc].iter().enumerate() {
                w2c[r][..3].copy_from_slice(axis);
                w2c[r][3] = -(0..3).map(|i| axis[i] * eye[i]).sum::<f64>();
            }
            w2c[3][3] = 1.0;
            let mut depth = vec![0f32; SIZE * SIZE];
            for y in 0..SIZE {
                for x in 0..SIZE {
                    let ray = [(x as f64 - f / 2.0) / f, (y as f64 - f / 2.0) / f, 1.0];
                    let dir = [0, 1, 2].map(|i| xc[i] * ray[0] + yc[i] * ray[1] + zc[i] * ray[2]);
                    let a = dir.iter().map(|v| v * v).sum::<f64>();
                    let b = 2.0 * (0..3).map(|i| dir[i] * eye[i]).sum::<f64>();
                    let c = DIST * DIST - R * R;
                    let disc = b * b - 4.0 * a * c;
                    if disc >= 0.0 {
                        depth[y * SIZE + x] = ((-b - disc.sqrt()) / (2.0 * a)) as f32;
                    }
                }
            }
            let rgb = vec![[0.5f32; 3]; SIZE * SIZE];
            tsdf.integrate(&DepthFrame {
                depth: &depth,
                rgb: &rgb,
                width: SIZE,
                height: SIZE,
                w2c,
                k,
            });
        }
        let mesh = tsdf.extract().largest_component();
        assert!(
            mesh.triangles.len() > 1000,
            "{} triangles",
            mesh.triangles.len()
        );
        let worst = mesh
            .vertices
            .iter()
            .map(|v| ((v.iter().map(|x| x * x).sum::<f32>()).sqrt() - R as f32).abs())
            .fold(0f32, f32::max);
        assert!(worst < 0.05, "a vertex {worst} off the sphere");
    }
}
