//! The regular trajectories WorldStereo's planner makes from a panorama (Stage 1 of `traj_generate.py`): three start
//! views a third of a turn apart, each with a right turn, a left turn and an up-and-right aerial move orbiting a point
//! at the view's median depth, shortened while they pass too close to the scene.

use rayon::prelude::*;

pub type Mat4 = [[f64; 4]; 4];

/// Frames per trajectory, the start included
pub const TRAJECTORY_FRAMES: usize = 21;
/// The start views' field of view and the short side they are rendered at
pub const VIEW_FOV_X_DEG: f64 = 120.0;
pub const VIEW_FOV_Y_DEG: f64 = 90.0;
pub const VIEW_SHORT_SIDE: usize = 480;
pub const START_VIEWS: usize = 3;
const ROTATION_DEG: f64 = 120.0;
const AERIAL_UP_DEG: f64 = 45.0;
const AERIAL_RIGHT_DEG: f64 = 60.0;
// collision checks sample the path at this many poses
const CHECK_POSES: usize = 80;
const DISTANCE_THRESHOLD: f64 = 0.1;
const OBSTACLE_DECAY: f64 = 2.0 / 3.0;
const OBSTACLE_LIMIT: usize = 3;
// the re-checks after a shortening average the distances to this many nearest scene points
const RECHECK_NEIGHBOURS: usize = 5;

#[derive(Clone, Copy, Debug)]
pub enum Move {
    /// orbit by (azimuth, elevation) degrees around the point at median depth
    Normal { rotation: [f64; 2] },
    /// pitch first, then turn about the world up axis through the point in front
    Aerial { rotation: [f64; 2] },
}

/// The three regular moves, in the planner's trajectory order (traj0, traj1, traj2).
pub fn regular_moves() -> [(Move, &'static str); 3] {
    [
        (
            Move::Normal {
                rotation: [-ROTATION_DEG, 0.0],
            },
            "right-rotation",
        ),
        (
            Move::Normal {
                rotation: [ROTATION_DEG, 0.0],
            },
            "left-rotation",
        ),
        (
            Move::Aerial {
                rotation: [-AERIAL_RIGHT_DEG, -AERIAL_UP_DEG],
            },
            "up-right-aerial",
        ),
    ]
}

/// The start views' size: the short side, the long side from the fields of view, both rounded up as the planner's
/// adjust_image_size does (multiples of 16 whose token grid divides by 8).
pub fn view_size() -> (usize, usize) {
    let h = VIEW_SHORT_SIDE;
    let w = ((VIEW_FOV_X_DEG.to_radians() / 2.0).tan() / (VIEW_FOV_Y_DEG.to_radians() / 2.0).tan()
        * h as f64)
        .round() as usize;
    let h16 = h.div_ceil(16) * 16;
    let a = h16 / 16;
    let twos = a.trailing_zeros().min(3) as usize;
    let required = 1usize << (3 - twos);
    let b = w.div_ceil(16).div_ceil(required) * required;
    (b * 16, h16)
}

/// Pixel intrinsics for a view of `width x height` at the given fields of view, principal point at the centre.
pub fn intrinsics(width: usize, height: usize, fov_x_deg: f64, fov_y_deg: f64) -> [[f64; 3]; 3] {
    let fx = 1.0 / (2.0 * (fov_x_deg.to_radians() / 2.0).tan()) * width as f64;
    let fy = 1.0 / (2.0 * (fov_y_deg.to_radians() / 2.0).tan()) * height as f64;
    [
        [fx, 0.0, 0.5 * width as f64],
        [0.0, fy, 0.5 * height as f64],
        [0.0, 0.0, 1.0],
    ]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(v: [f64; 3]) -> [f64; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    v.map(|x| x / n)
}

fn mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut out = [[0.0; 4]; 4];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = (0..4).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

/// Camera-to-world (OpenCV) of a camera at the origin looking along `dir` with +z up (utils3d's look-at).
pub fn look_at_c2w(dir: [f64; 3]) -> Mat4 {
    let z = normalize(dir);
    let x = normalize(cross([0.0, 0.0, -1.0], z));
    let y = normalize(cross(z, x));
    [
        [x[0], y[0], z[0], 0.0],
        [x[1], y[1], z[1], 0.0],
        [x[2], y[2], z[2], 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// The start views' camera-to-world matrices: -x, then turned a third of a turn at a time (clockwise about +z).
pub fn start_views() -> Vec<Mat4> {
    (0..START_VIEWS)
        .map(|i| {
            let a = (360.0 / START_VIEWS as f64 * i as f64).to_radians();
            // rotate_around_z_axis's clockwise turn of (-1, 0, 0)
            look_at_c2w([-a.cos(), a.sin(), 0.0])
        })
        .collect()
}

// native_camera_rotation: orbit by azimuth phi and elevation theta about the point at depth d along the optical axis
fn native_rotation(c2w: &Mat4, d: f64, phi: f64, theta: f64) -> Mat4 {
    let el = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, theta.cos(), -theta.sin(), 0.0],
        [0.0, theta.sin(), theta.cos(), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let az = [
        [phi.cos(), 0.0, phi.sin(), 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-phi.sin(), 0.0, phi.cos(), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let back = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -d],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let mut dummy = mul(&mul(&az, &el), &back);
    dummy[2][3] += d;
    mul(c2w, &dummy)
}

// camera_rotation: a camera off the ground plane turning in azimuth rotates about the world z axis through the point
// in front of it (dropped to z = 0); otherwise the native orbit. The comparison with exactly 0 is the reference's.
fn camera_rotation(c2w: &Mat4, d: f64, phi: f64, theta: f64) -> Mat4 {
    if c2w[2][3] != 0.0 && phi != 0.0 {
        let p = [0, 1, 2].map(|i| c2w[i][0] * 0.0 + c2w[i][1] * 0.0 + c2w[i][2] * d + c2w[i][3]);
        let point = [p[0], p[1], 0.0];
        let (c, s) = ((-phi).cos(), (-phi).sin());
        let r = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
        let mut m = [[0.0; 4]; 4];
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] = r[i][j];
            }
            m[i][3] = point[i] - (0..3).map(|j| r[i][j] * point[j]).sum::<f64>();
        }
        m[3][3] = 1.0;
        mul(&m, c2w)
    } else {
        native_rotation(c2w, d, phi, theta)
    }
}

fn path(c2w_start: &Mat4, mv: Move, d: f64, n: usize) -> Vec<Mat4> {
    match mv {
        Move::Normal {
            rotation: [phi, theta],
        } => (1..=n)
            .map(|j| {
                let f = j as f64 / n as f64;
                if phi * f == 0.0 && theta * f == 0.0 {
                    *c2w_start
                } else {
                    camera_rotation(
                        c2w_start,
                        d,
                        (phi * f).to_radians(),
                        (theta * f).to_radians(),
                    )
                }
            })
            .collect(),
        Move::Aerial {
            rotation: [phi, theta],
        } => {
            let n_phi = (phi.abs() / (phi.abs() + theta.abs()) * n as f64).round() as usize;
            let n_theta = n - n_phi;
            let mut out = (1..=n_theta)
                .map(|j| {
                    camera_rotation(
                        c2w_start,
                        d,
                        0.0,
                        (j as f64 / n_theta as f64 * theta).to_radians(),
                    )
                })
                .collect::<Vec<_>>();
            let middle = *out.last().unwrap_or(c2w_start);
            out.extend((1..=n_phi).map(|j| {
                camera_rotation(
                    &middle,
                    d,
                    (j as f64 / n_phi as f64 * phi).to_radians(),
                    0.0,
                )
            }));
            out
        }
    }
}

// a rotation matrix's unit quaternion (w, x, y, z) and back, for slerp
fn to_quat(m: &Mat4) -> [f64; 4] {
    let tr = m[0][0] + m[1][1] + m[2][2];
    let q = if tr > 0.0 {
        let s = (tr + 1.0).sqrt() * 2.0;
        [
            0.25 * s,
            (m[2][1] - m[1][2]) / s,
            (m[0][2] - m[2][0]) / s,
            (m[1][0] - m[0][1]) / s,
        ]
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0;
        [
            (m[2][1] - m[1][2]) / s,
            0.25 * s,
            (m[0][1] + m[1][0]) / s,
            (m[0][2] + m[2][0]) / s,
        ]
    } else if m[1][1] > m[2][2] {
        let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0;
        [
            (m[0][2] - m[2][0]) / s,
            (m[0][1] + m[1][0]) / s,
            0.25 * s,
            (m[1][2] + m[2][1]) / s,
        ]
    } else {
        let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0;
        [
            (m[1][0] - m[0][1]) / s,
            (m[0][2] + m[2][0]) / s,
            (m[1][2] + m[2][1]) / s,
            0.25 * s,
        ]
    };
    let n = q.iter().map(|v| v * v).sum::<f64>().sqrt();
    q.map(|v| v / n)
}

fn from_quat([w, x, y, z]: [f64; 4]) -> [[f64; 3]; 3] {
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

fn slerp(a: [f64; 4], b: [f64; 4], t: f64) -> [f64; 4] {
    let mut dot: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
    let b = if dot < 0.0 {
        dot = -dot;
        b.map(|v| -v)
    } else {
        b
    };
    if dot > 1.0 - 1e-12 {
        return a;
    }
    let omega = dot.clamp(-1.0, 1.0).acos();
    let (sa, sb) = (
        ((1.0 - t) * omega).sin() / omega.sin(),
        (t * omega).sin() / omega.sin(),
    );
    [0, 1, 2, 3].map(|i| sa * a[i] + sb * b[i])
}

/// `m` poses along `poses` (at least two): rotations slerped and translations interpolated linearly over evenly spaced
/// keys, as interpolate_poses does with SciPy.
pub fn interpolate_poses(poses: &[Mat4], m: usize) -> Vec<Mat4> {
    let n = poses.len();
    let quats = poses.iter().map(to_quat).collect::<Vec<_>>();
    (0..m)
        .map(|i| {
            let t = if m == 1 {
                0.0
            } else {
                i as f64 / (m - 1) as f64
            };
            let pos = t * (n - 1) as f64;
            let k = (pos.floor() as usize).min(n - 2);
            let f = pos - k as f64;
            let r = from_quat(slerp(quats[k], quats[k + 1], f));
            let mut out = [[0.0; 4]; 4];
            for row in 0..3 {
                out[row][..3].copy_from_slice(&r[row]);
                out[row][3] = poses[k][row][3] * (1.0 - f) + poses[k + 1][row][3] * f;
            }
            out[3][3] = 1.0;
            out
        })
        .collect()
}

/// The scene the planner keeps paths away from: the panorama mesh's triangles and its point cloud.
pub struct Obstacles<'a> {
    pub mesh: &'a MeshBvh,
    pub points: &'a PointGrid,
}

// triangles per BVH leaf
const BVH_LEAF: usize = 8;
const KD_LEAF: usize = 16;

struct BvhNode {
    lo: [f64; 3],
    hi: [f64; 3],
    start: u32,
    // a leaf's triangle count, 0 for an inner node (left child next, right child at `right`)
    count: u32,
    right: u32,
}

/// A mesh's triangles in a bounding-volume hierarchy, for exact nearest-distance queries.
pub struct MeshBvh {
    triangles: Vec<[[f64; 3]; 3]>,
    nodes: Vec<BvhNode>,
}

impl MeshBvh {
    pub fn new(vertices: &[[f32; 3]], triangles: &[[u32; 3]]) -> Self {
        let mut tris = triangles
            .iter()
            .map(|t| t.map(|i| vertices[i as usize].map(f64::from)))
            .collect::<Vec<_>>();
        let mut nodes = Vec::with_capacity(2 * tris.len() / BVH_LEAF + 1);
        if !tris.is_empty() {
            Self::build(&mut tris, 0, &mut nodes);
        }
        Self {
            triangles: tris,
            nodes,
        }
    }

    // median split on the longest axis of the centroids' bounds
    fn build(tris: &mut [[[f64; 3]; 3]], start: usize, nodes: &mut Vec<BvhNode>) -> usize {
        let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        let (mut clo, mut chi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for t in tris.iter() {
            for d in 0..3 {
                let c = (t[0][d] + t[1][d] + t[2][d]) / 3.0;
                clo[d] = clo[d].min(c);
                chi[d] = chi[d].max(c);
                for v in t {
                    lo[d] = lo[d].min(v[d]);
                    hi[d] = hi[d].max(v[d]);
                }
            }
        }
        let index = nodes.len();
        nodes.push(BvhNode {
            lo,
            hi,
            start: start as u32,
            count: tris.len() as u32,
            right: 0,
        });
        if tris.len() <= BVH_LEAF {
            return index;
        }
        let axis = (0..3)
            .max_by(|&a, &b| (chi[a] - clo[a]).total_cmp(&(chi[b] - clo[b])))
            .unwrap_or(0);
        let mid = tris.len() / 2;
        let centroid = |t: &[[f64; 3]; 3]| t[0][axis] + t[1][axis] + t[2][axis];
        tris.select_nth_unstable_by(mid, |a, b| centroid(a).total_cmp(&centroid(b)));
        let (left, right) = tris.split_at_mut(mid);
        nodes[index].count = 0;
        Self::build(left, start, nodes);
        let r = Self::build(right, start + mid, nodes);
        nodes[index].right = r as u32;
        index
    }

    fn box_distance(node: &BvhNode, q: [f64; 3]) -> f64 {
        (0..3)
            .map(|d| (node.lo[d] - q[d]).max(q[d] - node.hi[d]).max(0.0).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// The distance from `q` to the nearest triangle (infinite for an empty mesh).
    pub fn distance(&self, q: [f64; 3]) -> f64 {
        let mut best = f64::INFINITY;
        let Some(root) = self.nodes.first() else {
            return best;
        };
        let mut stack = vec![(0usize, Self::box_distance(root, q))];
        while let Some((n, d)) = stack.pop() {
            if d > best {
                continue;
            }
            let node = &self.nodes[n];
            if node.count > 0 {
                let start = node.start as usize;
                for t in &self.triangles[start..start + node.count as usize] {
                    best = best.min(point_triangle_distance(q, t[0], t[1], t[2]));
                }
                continue;
            }
            let (l, r) = (n + 1, node.right as usize);
            let (dl, dr) = (
                Self::box_distance(&self.nodes[l], q),
                Self::box_distance(&self.nodes[r], q),
            );
            // nearer child popped first
            if dl <= dr {
                stack.push((r, dr));
                stack.push((l, dl));
            } else {
                stack.push((l, dl));
                stack.push((r, dr));
            }
        }
        best
    }
}

/// Points in a balanced kd-tree (median splits on the widest axis), for nearest-neighbour queries.
pub struct PointGrid {
    points: Vec<[f32; 3]>,
    // per node in heap order: split axis and value
    splits: Vec<(u8, f32)>,
}

impl PointGrid {
    pub fn new(points: &[[f32; 3]]) -> Self {
        let mut points = points.to_vec();
        let depth = (points.len() / KD_LEAF + 1)
            .next_power_of_two()
            .trailing_zeros() as usize
            + 1;
        let mut splits = vec![(0, 0.0); 1 << depth];
        kd_build(&mut points, 0, &mut splits);
        Self { points, splits }
    }

    /// The mean distance to the `k` nearest points (kd-tree query of the reference).
    pub fn mean_knn(&self, q: [f64; 3], k: usize) -> f64 {
        let best = self.knn_sq(q, k);
        best.iter().map(|d| d.sqrt()).sum::<f64>() / best.len().max(1) as f64
    }

    /// Squared distances to the `k` nearest points, ascending (a point in the set finds itself at 0).
    pub fn knn_sq(&self, q: [f64; 3], k: usize) -> Vec<f64> {
        let mut best: Vec<f64> = Vec::with_capacity(k + 1);
        self.knn(0, 0, self.points.len(), q, k, &mut best);
        best
    }

    // best holds the squared distances found so far, ascending
    fn knn(&self, node: usize, lo: usize, hi: usize, q: [f64; 3], k: usize, best: &mut Vec<f64>) {
        if hi - lo <= KD_LEAF {
            for p in &self.points[lo..hi] {
                let d = (0..3)
                    .map(|d| (f64::from(p[d]) - q[d]).powi(2))
                    .sum::<f64>();
                if best.len() < k || d < best[k - 1] {
                    let at = best.partition_point(|&b| b < d);
                    best.insert(at, d);
                    best.truncate(k);
                }
            }
            return;
        }
        let mid = (lo + hi) / 2;
        let (axis, split) = self.splits[node];
        let diff = q[axis as usize] - f64::from(split);
        let (near, far) = if diff < 0.0 {
            ((2 * node + 1, lo, mid), (2 * node + 2, mid, hi))
        } else {
            ((2 * node + 2, mid, hi), (2 * node + 1, lo, mid))
        };
        self.knn(near.0, near.1, near.2, q, k, best);
        if best.len() < k || diff * diff < best[k - 1] {
            self.knn(far.0, far.1, far.2, q, k, best);
        }
    }
}

fn kd_build(points: &mut [[f32; 3]], node: usize, splits: &mut [(u8, f32)]) {
    if points.len() <= KD_LEAF {
        return;
    }
    let mut lo = [f32::MAX; 3];
    let mut hi = [f32::MIN; 3];
    for p in points.iter() {
        for d in 0..3 {
            lo[d] = lo[d].min(p[d]);
            hi[d] = hi[d].max(p[d]);
        }
    }
    let axis = (0..3)
        .max_by(|&a, &b| (hi[a] - lo[a]).total_cmp(&(hi[b] - lo[b])))
        .expect("three axes");
    let mid = points.len() / 2;
    points.select_nth_unstable_by(mid, |a, b| a[axis].total_cmp(&b[axis]));
    splits[node] = (axis as u8, points[mid][axis]);
    let (left, right) = points.split_at_mut(mid);
    kd_build(left, 2 * node + 1, splits);
    kd_build(right, 2 * node + 2, splits);
}

// distance from p to triangle abc (Ericson's closest point on a triangle)
fn point_triangle_distance(p: [f64; 3], a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    let sub = |u: [f64; 3], v: [f64; 3]| [u[0] - v[0], u[1] - v[1], u[2] - v[2]];
    let dot = |u: [f64; 3], v: [f64; 3]| u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    let add_scaled =
        |u: [f64; 3], v: [f64; 3], s: f64| [u[0] + v[0] * s, u[1] + v[1] * s, u[2] + v[2] * s];
    let (ab, ac, ap) = (sub(b, a), sub(c, a), sub(p, a));
    let (d1, d2) = (dot(ab, ap), dot(ac, ap));
    let closest = if d1 <= 0.0 && d2 <= 0.0 {
        a
    } else {
        let bp = sub(p, b);
        let (d3, d4) = (dot(ab, bp), dot(ac, bp));
        if d3 >= 0.0 && d4 <= d3 {
            b
        } else {
            let vc = d1 * d4 - d3 * d2;
            if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
                add_scaled(a, ab, d1 / (d1 - d3))
            } else {
                let cp = sub(p, c);
                let (d5, d6) = (dot(ab, cp), dot(ac, cp));
                if d6 >= 0.0 && d5 <= d6 {
                    c
                } else {
                    let vb = d5 * d2 - d1 * d6;
                    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
                        add_scaled(a, ac, d2 / (d2 - d6))
                    } else {
                        let va = d3 * d6 - d5 * d4;
                        if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
                            add_scaled(b, sub(c, b), (d4 - d3) / ((d4 - d3) + (d5 - d6)))
                        } else {
                            let denom = 1.0 / (va + vb + vc);
                            add_scaled(add_scaled(a, ab, vb * denom), ac, vc * denom)
                        }
                    }
                }
            }
        }
    };
    let d = sub(p, closest);
    dot(d, d).sqrt()
}

/// One planned trajectory: camera-to-world per frame (the start first), or None when it collides too often.
pub fn plan(
    c2w_start: &Mat4,
    mv: Move,
    median_depth: f64,
    obstacles: &Obstacles,
) -> Option<Vec<Mat4>> {
    let n = TRAJECTORY_FRAMES - 1;
    let threshold = DISTANCE_THRESHOLD.min(median_depth * 0.1);
    let mut mv = mv;
    let mut c2ws = path(c2w_start, mv, median_depth, n);
    let dense = |c2ws: &[Mat4]| {
        if c2ws.len() < CHECK_POSES {
            interpolate_poses(c2ws, CHECK_POSES)
        } else {
            c2ws.to_vec()
        }
    };
    let centre = |m: &Mat4| [m[0][3], m[1][3], m[2][3]];
    let mut distances = dense(&c2ws)
        .par_iter()
        .map(|m| obstacles.mesh.distance(centre(m)))
        .collect::<Vec<_>>();
    let mut iteration = 0;
    while distances.iter().copied().fold(f64::INFINITY, f64::min) < threshold
        && iteration < OBSTACLE_LIMIT + 1
    {
        match &mut mv {
            Move::Normal { .. } => {
                let keep = (c2ws.len() as f64 * OBSTACLE_DECAY) as usize;
                c2ws = interpolate_poses(&c2ws[..keep.max(2)], n);
            }
            Move::Aerial { rotation } => {
                let [phi, theta] = *rotation;
                let n_dist_phi = (phi.abs() / (phi.abs() + theta.abs()) * distances.len() as f64)
                    .round() as usize;
                let n_dist_theta = distances.len() - n_dist_phi;
                if distances[..n_dist_theta].iter().any(|&d| d < threshold) {
                    rotation[1] *= OBSTACLE_DECAY;
                }
                if distances[n_dist_theta..].iter().any(|&d| d < threshold) {
                    rotation[0] *= OBSTACLE_DECAY;
                }
                c2ws = path(c2w_start, mv, median_depth, n);
            }
        }
        distances = dense(&c2ws)
            .par_iter()
            .map(|m| obstacles.points.mean_knn(centre(m), RECHECK_NEIGHBOURS))
            .collect();
        iteration += 1;
    }
    if iteration > OBSTACLE_LIMIT {
        return None;
    }
    Some(std::iter::once(*c2w_start).chain(c2ws).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    // clustered points with far stragglers, sized around a split boundary, against brute force
    #[test]
    fn point_grid_knn_matches_brute_force() {
        let mut state = 7u64;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            (state >> 40) as f32 / (1u64 << 24) as f32
        };
        for n in [5, 16, 17, 1000, 4097] {
            let points = (0..n)
                .map(|i| {
                    let spread = if i % 50 == 0 { 100.0 } else { 1.0 };
                    [next() * spread, next() * spread, next() * 0.01]
                })
                .collect::<Vec<_>>();
            let grid = PointGrid::new(&points);
            for q in points.iter().step_by(7) {
                let q = q.map(f64::from);
                let mut d = points
                    .iter()
                    .map(|p| {
                        (0..3)
                            .map(|i| (f64::from(p[i]) - q[i]).powi(2))
                            .sum::<f64>()
                            .sqrt()
                    })
                    .collect::<Vec<_>>();
                d.sort_by(f64::total_cmp);
                let k = 11.min(n);
                let want = d[..k].iter().sum::<f64>() / k as f64;
                assert!((grid.mean_knn(q, 11) - want).abs() < 1e-12, "n {n}");
            }
        }
    }

    // a deterministic scatter of triangles of mixed sizes, queried inside, outside and on the bounds
    #[test]
    fn bvh_distance_equals_brute_force() {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut vertices = Vec::new();
        let mut triangles = Vec::new();
        for i in 0..500u32 {
            let centre = [
                next() * 10.0 - 5.0,
                next() * 10.0 - 5.0,
                next() * 10.0 - 5.0,
            ];
            let size = if i % 50 == 0 { 4.0 } else { 0.3 };
            for _ in 0..3 {
                vertices.push([0, 1, 2].map(|d| (centre[d] + (next() - 0.5) * size) as f32));
            }
            triangles.push([3 * i, 3 * i + 1, 3 * i + 2]);
        }
        let bvh = MeshBvh::new(&vertices, &triangles);
        let v = |i: u32| vertices[i as usize].map(f64::from);
        for _ in 0..200 {
            let q = [
                next() * 16.0 - 8.0,
                next() * 16.0 - 8.0,
                next() * 16.0 - 8.0,
            ];
            let brute = triangles
                .iter()
                .map(|t| point_triangle_distance(q, v(t[0]), v(t[1]), v(t[2])))
                .fold(f64::INFINITY, f64::min);
            assert_eq!(bvh.distance(q), brute, "{q:?}");
        }
    }
}
