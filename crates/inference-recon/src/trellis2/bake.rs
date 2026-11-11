//! TRELLIS.2's textured export (o-voxel's `to_glb`, optionally with its remesh): the mesh cleaned and simplified as CuMesh
//! does, unwrapped into an atlas, and the PBR volume baked into it. Each texel a face covers is placed in 3D on the
//! simplified mesh, moved to the closest point of the mesh as extracted, and given the volume's trilinear attributes;
//! the texels no face covers are inpainted (OpenCV's Telea) so seams don't bleed black.

use std::sync::atomic::{AtomicU32, Ordering};

use inference_tensor::{Device, Result};
use inference_vision::{TeleaSchedule, telea_schedules};
use rayon::prelude::*;

use super::charts::{Unwrapped, uv_unwrap, vertex_normals};
use super::cleanup;
use super::glb::{Pbr, sample_attributes};
use super::mesh::Mesh;
use super::remesh::{BAND, Domain, PROJECT_BACK, remesh};
use super::vec3::{V3, dot, mad, sub};
use crate::progress::CANCELLED;

/// `to_glb`'s defaults
pub const DECIMATION_TARGET: usize = 1_000_000;
pub const TEXTURE_SIZE: usize = 2048;
// the first simplification aims at this multiple of the target
const FIRST_PASS_FACTOR: usize = 3;
const COLOR_RADIUS: f64 = 3.0;
const MATERIAL_RADIUS: f64 = 1.0;
const LEAF_SIZE: usize = 4;
const RASTER_CHUNK: usize = 4096;
const STACK_DEPTH: usize = 128;

#[derive(Clone, Copy, Debug)]
pub struct BakeOptions {
    /// Faces the mesh is simplified to
    pub decimation_target: usize,
    /// Width and height of the textures
    pub texture_size: usize,
    /// Rebuild the surface by narrow-band dual contouring before simplifying (a closed shell around it)
    pub remesh: bool,
}

impl Default for BakeOptions {
    fn default() -> Self {
        Self {
            decimation_target: DECIMATION_TARGET,
            texture_size: TEXTURE_SIZE,
            remesh: false,
        }
    }
}

/// The baked mesh: its unwrapped geometry, the base color + alpha texture (RGBA) and the glTF metallic-roughness
/// texture (R unused, G roughness, B metallic), both `size` x `size`, row 0 at v = 0.
pub struct Baked {
    pub unwrapped: Unwrapped,
    pub base_color: Vec<u8>,
    pub metallic_roughness: Vec<u8>,
    pub size: usize,
    /// Texels a face covers
    pub covered: usize,
    /// Whether the faces' back sides are drawn (not after a remesh, whose shell is closed)
    pub double_sided: bool,
}

struct Node {
    min: V3,
    max: V3,
    // leaf: first triangle and count; inner: the right child (the left follows the node)
    start: u32,
    count: u32,
}

/// Closest-point queries over a [`Bvh`], on a CUDA device when given one (with the host's results).
pub struct Distances<'a> {
    bvh: &'a Bvh,
    #[cfg(feature = "cuda")]
    device: Option<super::mesh_cuda::DeviceBvh>,
}

impl<'a> Distances<'a> {
    pub fn new(bvh: &'a Bvh, device: &Device) -> Result<Self> {
        let _ = device;
        Ok(Self {
            bvh,
            #[cfg(feature = "cuda")]
            device: match device {
                Device::Cuda(dev) => Some(super::mesh_cuda::DeviceBvh::new(bvh, dev)?),
                _ => None,
            },
        })
    }

    /// [`Bvh::unsigned_distance`], on the device when there is one.
    pub fn unsigned_distance(&self, points: &[V3]) -> Result<Vec<(f32, u32, V3)>> {
        #[cfg(feature = "cuda")]
        if let Some(device) = &self.device {
            return device.unsigned_distance(points);
        }
        Ok(self.bvh.unsigned_distance(points))
    }
}

/// A bounding volume hierarchy over a mesh's triangles, for closest-point queries.
pub struct Bvh {
    nodes: Vec<Node>,
    // each triangle with its face index
    tris: Vec<([V3; 3], u32)>,
}

impl Bvh {
    pub fn new(mesh: &Mesh) -> Self {
        let mut tris: Vec<([V3; 3], u32)> = mesh
            .faces
            .iter()
            .enumerate()
            .map(|(i, f)| (f.map(|v| mesh.vertices[v as usize]), i as u32))
            .collect();
        let mut nodes = Vec::with_capacity(2 * tris.len() / LEAF_SIZE + 1);
        let len = tris.len();
        Self::build(&mut nodes, &mut tris, 0, len);
        Self { nodes, tris }
    }

    fn build(
        nodes: &mut Vec<Node>,
        tris: &mut [([V3; 3], u32)],
        start: usize,
        end: usize,
    ) -> usize {
        let (mut min, mut max) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
        for (t, _) in &tris[start..end] {
            for p in t {
                for k in 0..3 {
                    min[k] = min[k].min(p[k]);
                    max[k] = max[k].max(p[k]);
                }
            }
        }
        let id = nodes.len();
        nodes.push(Node {
            min,
            max,
            start: start as u32,
            count: (end - start) as u32,
        });
        if end - start <= LEAF_SIZE {
            return id;
        }
        let axis = (0..3)
            .max_by(|&a, &b| (max[a] - min[a]).total_cmp(&(max[b] - min[b])))
            .expect("three axes");
        let mid = (start + end) / 2;
        let centroid = |(t, _): &([V3; 3], u32)| t[0][axis] + t[1][axis] + t[2][axis];
        tris[start..end]
            .select_nth_unstable_by(mid - start, |a, b| centroid(a).total_cmp(&centroid(b)));
        Self::build(nodes, tris, start, mid);
        let right = Self::build(nodes, tris, mid, end);
        nodes[id].start = right as u32;
        nodes[id].count = 0;
        id
    }

    fn box_distance2(node: &Node, p: V3) -> f32 {
        (0..3)
            .map(|k| {
                (node.min[k] - p[k])
                    .max(p[k] - node.max[k])
                    .max(0.0)
                    .powi(2)
            })
            .sum()
    }

    /// The tree as flat arrays for a device: boxes (min xyz, max xyz), links (start, count), triangles (9 floats),
    /// and each triangle's face.
    #[cfg(feature = "cuda")]
    pub(super) fn flat(&self) -> (Vec<f32>, Vec<u32>, Vec<f32>, Vec<u32>) {
        let boxes = self
            .nodes
            .iter()
            .flat_map(|n| [n.min, n.max].concat())
            .collect();
        let links = self.nodes.iter().flat_map(|n| [n.start, n.count]).collect();
        let tris = self.tris.iter().flat_map(|(t, _)| t.concat()).collect();
        let faces = self.tris.iter().map(|(_, f)| *f).collect();
        (boxes, links, tris, faces)
    }

    /// The point of the mesh closest to `p`.
    pub fn closest(&self, p: V3) -> V3 {
        self.closest_face(p).1
    }

    /// The face closest to `p` and the point of it nearest `p`.
    pub fn closest_face(&self, p: V3) -> (u32, V3) {
        let mut best = (f32::INFINITY, p, 0);
        // median splits keep the tree about log2(faces / LEAF_SIZE) deep, two entries pushed per level
        let mut stack = [0u32; STACK_DEPTH];
        let mut top = 1;
        while top > 0 {
            top -= 1;
            let n = stack[top] as usize;
            let node = &self.nodes[n];
            if Self::box_distance2(node, p) >= best.0 {
                continue;
            }
            if node.count > 0 {
                for (t, face) in &self.tris[node.start as usize..(node.start + node.count) as usize]
                {
                    let q = closest_on_triangle(p, t);
                    let d = sub(q, p);
                    let d2 = dot(d, d);
                    if d2 < best.0 {
                        best = (d2, q, *face);
                    }
                }
                continue;
            }
            let (l, r) = (n + 1, node.start as usize);
            let (dl, dr) = (
                Self::box_distance2(&self.nodes[l], p),
                Self::box_distance2(&self.nodes[r], p),
            );
            // the nearer child is searched first
            let (far, near) = if dl < dr { (r, l) } else { (l, r) };
            stack[top] = far as u32;
            stack[top + 1] = near as u32;
            top += 2;
        }
        (best.2, best.1)
    }

    /// Each point's distance to the mesh, its closest face and the closest point, queried in parallel.
    pub fn unsigned_distance(&self, points: &[V3]) -> Vec<(f32, u32, V3)> {
        points
            .par_iter()
            .map(|&p| {
                let (face, q) = self.closest_face(p);
                let d = sub(q, p);
                (dot(d, d).sqrt(), face, q)
            })
            .collect()
    }
}

// Ericson, Real-Time Collision Detection 5.1.5
fn closest_on_triangle(p: V3, [a, b, c]: &[V3; 3]) -> V3 {
    let (ab, ac, ap) = (sub(*b, *a), sub(*c, *a), sub(p, *a));
    let (d1, d2) = (dot(ab, ap), dot(ac, ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return *a;
    }
    let bp = sub(p, *b);
    let (d3, d4) = (dot(ab, bp), dot(ac, bp));
    if d3 >= 0.0 && d4 <= d3 {
        return *b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return mad(*a, ab, d1 / (d1 - d3));
    }
    let cp = sub(p, *c);
    let (d5, d6) = (dot(ab, cp), dot(ac, cp));
    if d6 >= 0.0 && d5 <= d6 {
        return *c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return mad(*a, ac, d2 / (d2 - d6));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        return mad(*b, sub(*c, *b), (d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let denom = 1.0 / (va + vb + vc);
    mad(mad(*a, ab, vb * denom), ac, vc * denom)
}

// barycentric weights of `p` in the uv triangle, None outside it
fn barycentric(p: [f64; 2], [a, b, c]: [[f64; 2]; 3]) -> Option<[f64; 3]> {
    let edge = |p: [f64; 2], q: [f64; 2], r: [f64; 2]| {
        (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    };
    let area = edge(a, b, c);
    if area == 0.0 {
        return None;
    }
    let w = [
        edge(b, c, p) / area,
        edge(c, a, p) / area,
        edge(a, b, p) / area,
    ];
    w.iter().all(|&x| x >= 0.0).then_some(w)
}

/// Each texel's covering face (+1, 0 for none): texel (x, y) centered at uv ((x + 0.5) / size, (y + 0.5) / size), a
/// texel two faces cover going to the later one.
pub(crate) fn rasterize(faces: &[[u32; 3]], uvs: &[[f32; 2]], size: usize) -> Vec<u32> {
    let texels: Vec<AtomicU32> = (0..size * size).map(|_| AtomicU32::new(0)).collect();
    let s = size as f64;
    faces
        .par_chunks(RASTER_CHUNK)
        .enumerate()
        .for_each(|(chunk, part)| {
            for (k, f) in part.iter().enumerate() {
                let id = (chunk * RASTER_CHUNK + k) as u32 + 1;
                let tri = f.map(|v| uvs[v as usize].map(|c| f64::from(c) * s));
                let lo = |a: usize| tri.iter().map(|p| p[a]).fold(f64::INFINITY, f64::min);
                let hi = |a: usize| tri.iter().map(|p| p[a]).fold(f64::NEG_INFINITY, f64::max);
                let span = |a: usize| {
                    let first = (lo(a) - 0.5).ceil().max(0.0) as usize;
                    let last = ((hi(a) - 0.5).floor().min(s - 1.0)).max(-1.0);
                    (first, last)
                };
                let ((x0, x1), (y0, y1)) = (span(0), span(1));
                if x1 < 0.0 || y1 < 0.0 {
                    continue;
                }
                for y in y0..=y1 as usize {
                    for x in x0..=x1 as usize {
                        if barycentric([x as f64 + 0.5, y as f64 + 0.5], tri).is_some() {
                            texels[y * size + x].fetch_max(id, Ordering::Relaxed);
                        }
                    }
                }
            }
        });
    texels.into_iter().map(AtomicU32::into_inner).collect()
}

// clip(x * 255, 0, 255).astype(uint8)
fn to_u8(x: f32) -> u8 {
    (x * 255.0).clamp(0.0, 255.0) as u8
}

/// `to_glb`'s cleanup chain before unwrapping: holes filled, simplified to three times the target, duplicate faces,
/// non-manifold edges and specks fixed, holes filled again, simplified to the target and fixed again, faces wound
/// consistently. Returns the cleaned mesh and the hole-filled one the texels are projected onto.
pub fn clean(mesh: &Mesh, target: usize, device: &Device) -> Result<(Mesh, Mesh)> {
    let mut m = mesh.clone();
    cleanup::fill_holes(&mut m, cleanup::MAX_HOLE_PERIMETER);
    let original = m.clone();
    for target in [target * FIRST_PASS_FACTOR, target] {
        cleanup::simplify_on(&mut m, target, device)?;
        cleanup::remove_duplicate_faces(&mut m);
        cleanup::repair_non_manifold_edges(&mut m);
        cleanup::remove_small_connected_components(&mut m, cleanup::MIN_COMPONENT_AREA);
        cleanup::fill_holes(&mut m, cleanup::MAX_HOLE_PERIMETER);
    }
    cleanup::unify_face_orientations(&mut m);
    Ok((m, original))
}

/// The mesh (in [-0.5, 0.5]^3) cleaned, unwrapped and baked from the attribute volume (`coords` / `values` at
/// `resolution`), simplifying on `device`.
pub fn bake(
    mesh: &Mesh,
    coords: &[[i32; 3]],
    values: &[[f32; 6]],
    resolution: usize,
    options: &BakeOptions,
    device: &Device,
) -> Result<Baked> {
    finish(
        prepare(mesh, resolution, options, device, &|| false)?,
        coords,
        values,
    )
}

/// The part of a bake that needs only the shape: the unwrapped mesh, its covered texels and their points in the
/// volume's grid, and the inpainting marches over the texels left uncovered.
pub struct Prepared {
    unwrapped: Unwrapped,
    size: usize,
    color_march: TeleaSchedule,
    material_march: TeleaSchedule,
    covered: Vec<usize>,
    points: Vec<V3>,
    double_sided: bool,
}

/// [`bake`]'s shape half (cleanup, unwrap, rasterization and projection), which can run while the texture is made;
/// it gives up (an error naming [`CANCELLED`]) between its steps, in the unwrap's rounds and in xatlas when `stop`
/// returns true.
// each texel's covering face (1-based, 0 for none), the covered texels, and where each lies on the unwrapped mesh
fn texels(unwrapped: &Unwrapped, size: usize) -> (Vec<u32>, Vec<usize>, Vec<V3>) {
    let texel_faces = rasterize(&unwrapped.mesh.faces, &unwrapped.uvs, size);
    let covered: Vec<usize> = (0..size * size).filter(|&t| texel_faces[t] > 0).collect();
    let positions = covered
        .par_iter()
        .map(|&t| {
            let f = unwrapped.mesh.faces[texel_faces[t] as usize - 1];
            let tri = f.map(|v| unwrapped.uvs[v as usize].map(|c| f64::from(c) * size as f64));
            let p = [(t % size) as f64 + 0.5, (t / size) as f64 + 0.5];
            let w = barycentric(p, tri).expect("a covered texel lies in its face");
            (0..3).fold([0f32; 3], |acc, k| {
                mad(acc, unwrapped.mesh.vertices[f[k] as usize], w[k] as f32)
            })
        })
        .collect();
    (texel_faces, covered, positions)
}

// one inpainting march per radius, shared by the planes that fill with it
fn marches(texel_faces: &[u32], size: usize) -> [TeleaSchedule; 2] {
    let unknown: Vec<bool> = texel_faces.iter().map(|&f| f == 0).collect();
    telea_schedules(size, size, &unknown, [COLOR_RADIUS, MATERIAL_RADIUS])
}

/// [`prepare`] for a mesh taken as it is, the texturing pipeline's: no cleanup, its own `uvs` (row 0 at v = 0) when
/// given and otherwise unwrapped, each texel sampled where it lies on it.
pub fn prepare_given(
    mesh: &Mesh,
    uvs: Option<Vec<[f32; 2]>>,
    resolution: usize,
    texture_size: usize,
    stop: &(dyn Fn() -> bool + Sync),
) -> Result<Prepared> {
    let unwrapped = match uvs {
        Some(uvs) => Unwrapped {
            normals: vertex_normals(mesh),
            source: (0..mesh.vertices.len() as u32).collect(),
            width: texture_size as u32,
            height: texture_size as u32,
            mesh: mesh.clone(),
            uvs,
        },
        None => uv_unwrap(mesh, stop).map_err(|e| inference_tensor::Error::Msg(e.to_string()))?,
    };
    if stop() {
        inference_tensor::bail!("{CANCELLED}");
    }
    let (texel_faces, covered, positions) = texels(&unwrapped, texture_size);
    let res = resolution as f32;
    let points = positions
        .iter()
        .map(|p| p.map(|v| (v + 0.5) * res))
        .collect();
    let [color_march, material_march] = marches(&texel_faces, texture_size);
    Ok(Prepared {
        unwrapped,
        size: texture_size,
        color_march,
        material_march,
        covered,
        points,
        double_sided: true,
    })
}

pub fn prepare(
    mesh: &Mesh,
    resolution: usize,
    options: &BakeOptions,
    device: &Device,
    stop: &(dyn Fn() -> bool + Sync),
) -> Result<Prepared> {
    let check = || -> Result<()> {
        if stop() {
            inference_tensor::bail!("{CANCELLED}");
        }
        Ok(())
    };
    let (cleaned, bvh) = if options.remesh {
        let mut filled = mesh.clone();
        cleanup::fill_holes(&mut filled, cleanup::MAX_HOLE_PERIMETER);
        let bvh = Bvh::new(&filled);
        let mut shell = remesh(
            &Distances::new(&bvh, device)?,
            &Domain::unit_cube(resolution as u32),
            BAND,
            PROJECT_BACK,
        )?;
        cleanup::simplify_on(&mut shell, options.decimation_target, device)?;
        (shell, bvh)
    } else {
        let (cleaned, original) = clean(mesh, options.decimation_target, device)?;
        (cleaned, Bvh::new(&original))
    };
    check()?;
    let unwrapped =
        uv_unwrap(&cleaned, stop).map_err(|e| inference_tensor::Error::Msg(e.to_string()))?;
    check()?;
    let size = options.texture_size;
    let (texel_faces, covered, positions) = texels(&unwrapped, size);
    let res = resolution as f32;
    let points: Vec<V3> = Distances::new(&bvh, device)?
        .unsigned_distance(&positions)?
        .into_iter()
        .map(|(_, _, c)| c.map(|v| (v + 0.5) * res))
        .collect();
    let [color_march, material_march] = marches(&texel_faces, size);
    Ok(Prepared {
        unwrapped,
        size,
        color_march,
        material_march,
        covered,
        points,
        double_sided: !options.remesh,
    })
}

/// [`bake`]'s texture half: the volume (`coords` / `values`) sampled at the prepared texels and inpainted.
pub fn finish(prepared: Prepared, coords: &[[i32; 3]], values: &[[f32; 6]]) -> Result<Baked> {
    let Prepared {
        unwrapped,
        size,
        color_march,
        material_march,
        covered,
        points,
        double_sided,
    } = prepared;
    let attrs = sample_attributes(coords, values, &points);
    // planes of the covered texels' values, zero elsewhere
    let plane = |channels: usize, get: &(dyn Fn(&Pbr) -> [f32; 3] + Sync)| {
        let mut plane = vec![0u8; size * size * channels];
        for (&t, a) in covered.iter().zip(&attrs) {
            let v = get(a);
            for c in 0..channels {
                plane[t * channels + c] = to_u8(v[c]);
            }
        }
        plane
    };
    let fill = |march: &TeleaSchedule, channels: usize, get: &(dyn Fn(&Pbr) -> [f32; 3] + Sync)| {
        let mut p = plane(channels, get);
        march.inpaint(&mut p, channels);
        p
    };
    let ((color, metallic), (roughness, alpha)) = rayon::join(
        || {
            rayon::join(
                || fill(&color_march, 3, &|p| p.base_color),
                || fill(&material_march, 1, &|p| [p.metallic; 3]),
            )
        },
        || {
            rayon::join(
                || fill(&material_march, 1, &|p| [p.roughness; 3]),
                || fill(&material_march, 1, &|p| [p.alpha; 3]),
            )
        },
    );
    let base_color = (0..size * size)
        .flat_map(|t| [color[3 * t], color[3 * t + 1], color[3 * t + 2], alpha[t]])
        .collect();
    let metallic_roughness = (0..size * size)
        .flat_map(|t| [0, roughness[t], metallic[t]])
        .collect();
    Ok(Baked {
        unwrapped,
        base_color,
        metallic_roughness,
        size,
        covered: covered.len(),
        double_sided,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_bvh_finds_the_closest_point_on_a_mesh() {
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
        let faces = quads
            .iter()
            .flat_map(|q: &[u32; 4]| [[q[0], q[1], q[2]], [q[0], q[2], q[3]]])
            .collect();
        let bvh = Bvh::new(&Mesh { vertices, faces });
        assert_eq!(bvh.closest([0.5, 0.5, 3.0]), [0.5, 0.5, 1.0]);
        assert_eq!(bvh.closest([2.0, 2.0, 2.0]), [1.0, 1.0, 1.0]);
        let inside = bvh.closest([0.5, 0.5, 0.2]);
        assert!((inside[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn a_baked_cube_writes_a_glb_whose_accessors_fit_their_views() {
        let vertices: Vec<V3> = (0..8)
            .map(|i| {
                [
                    (i & 1) as f32 - 0.5,
                    ((i >> 1) & 1) as f32 - 0.5,
                    ((i >> 2) & 1) as f32 - 0.5,
                ]
                .map(|c| c * 0.5)
            })
            .collect();
        let quads = [
            [0, 2, 3, 1],
            [4, 5, 7, 6],
            [0, 1, 5, 4],
            [2, 6, 7, 3],
            [0, 4, 6, 2],
            [1, 3, 7, 5],
        ];
        let faces = quads
            .iter()
            .flat_map(|q: &[u32; 4]| [[q[0], q[1], q[2]], [q[0], q[2], q[3]]])
            .collect();
        let mesh = Mesh { vertices, faces };
        let res = 8i32;
        let coords: Vec<[i32; 3]> = (0..res * res * res)
            .map(|i| [i / (res * res), (i / res) % res, i % res])
            .collect();
        let values = vec![[0.5f32; 6]; coords.len()];
        let options = BakeOptions {
            texture_size: 64,
            ..BakeOptions::default()
        };
        let baked = bake(
            &mesh,
            &coords,
            &values,
            res as usize,
            &options,
            &Device::Cpu,
        )
        .expect("a cube bakes");
        assert_eq!(
            baked.unwrapped.normals.len(),
            baked.unwrapped.mesh.vertices.len()
        );
        let glb = super::super::glb::to_textured_glb(&baked);
        let json_len = u32::from_le_bytes(glb[12..16].try_into().expect("4 bytes")) as usize;
        let gltf: serde_json::Value =
            serde_json::from_slice(&glb[20..20 + json_len]).expect("the JSON chunk");
        let components = |t: &str| match t {
            "SCALAR" => 1,
            "VEC2" => 2,
            "VEC3" => 3,
            _ => 4,
        };
        for accessor in gltf["accessors"].as_array().expect("accessors") {
            let view =
                &gltf["bufferViews"][accessor["bufferView"].as_u64().expect("a view") as usize];
            let bytes = accessor["count"].as_u64().expect("a count")
                * 4
                * components(accessor["type"].as_str().expect("a type"));
            assert_eq!(
                bytes,
                view["byteLength"].as_u64().expect("a length"),
                "{accessor}"
            );
        }
    }

    #[test]
    fn a_triangle_covers_the_texels_whose_centers_it_holds() {
        let uvs = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let texels = rasterize(&[[0, 1, 2]], &uvs, 4);
        // the lower-left half: x + y + 1 <= 4 in texel units
        let covered: Vec<usize> = (0..16).filter(|&t| texels[t] == 1).collect();
        assert_eq!(covered, [0, 1, 2, 3, 4, 5, 6, 8, 9, 12]);
    }
}
