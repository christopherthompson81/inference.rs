//! Mesh passes on a CUDA device (`kernels/cuda/mesh.cu`), each giving its host port's result: [`simplify`], whose mesh
//! stays on the GPU for every round, and closest-point queries over a [`Bvh`].

use inference_tensor::Result;
use inference_tensor::cuda_backend::cudarc::driver::{
    CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg,
};
use inference_tensor::cuda_backend::{CudaDevice, WrapErr};

use super::bake::Bvh;
use super::cleanup::{
    LAMBDA_EDGE_LENGTH, LAMBDA_SKINNY, SKINNY_EPS, SLOW_ROUND, START_THRESHOLD, THRESHOLD_STEP,
};
use super::mesh::Mesh;
use super::vec3::V3;

const MODULE: &str = "inference_recon_mesh";
static IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mesh.fatbin"));
// values one `scan_block_u32` block scans; `SCAN_BLOCK` in the kernel source
const SCAN_BLOCK: u32 = 1024;
// threads per block for the per-element kernels
const BLOCK: u32 = 256;

fn grid(n: u32, block: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(block).max(1), 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    }
}

// launches kernel `$name` with `$cfg` and the arguments in its signature's order
macro_rules! launch {
    ($dev:expr, $name:expr, $cfg:expr, $($arg:expr),* $(,)?) => {{
        let func = $dev.get_or_load_custom_image($name, MODULE, IMAGE)?;
        let mut builder = func.builder();
        $( builder.arg($arg); )*
        // SAFETY: the arguments follow the kernel's signature, with buffers sized for its counts
        unsafe { builder.launch($cfg) }.w()?;
    }};
}

// launches `$name` over `$n` elements, nothing when there are none
macro_rules! launch_n {
    ($dev:expr, $name:expr, $n:expr, $($arg:expr),* $(,)?) => {{
        let n: u32 = $n;
        if n > 0 {
            launch!($dev, $name, grid(n, BLOCK), $($arg),*);
        }
    }};
}

fn alloc<T: DeviceRepr>(dev: &CudaDevice, n: usize) -> Result<CudaSlice<T>> {
    // SAFETY: every buffer is written before it is read
    unsafe { dev.alloc::<T>(n.max(1)) }
}

// `n` copies of `value`
fn filled(dev: &CudaDevice, n: u32, value: u32) -> Result<CudaSlice<u32>> {
    let mut p = alloc::<u32>(dev, n as usize)?;
    launch_n!(dev, "fill_u32", n, &mut p, &value, &n);
    Ok(p)
}

// `n` flags set, then a zero the scan of them ends on
fn flags(dev: &CudaDevice, n: u32) -> Result<CudaSlice<u32>> {
    let mut p = filled(dev, n + 1, 1)?;
    dev.memcpy_htod(&[0u32], &mut p.slice_mut(n as usize..))?;
    Ok(p)
}

// exclusive prefix sum of `input`'s first `n` values
fn scan(dev: &CudaDevice, input: &CudaSlice<u32>, n: u32) -> Result<CudaSlice<u32>> {
    let mut out = alloc::<u32>(dev, n as usize)?;
    let blocks = n.div_ceil(SCAN_BLOCK).max(1);
    let mut sums = alloc::<u32>(dev, blocks as usize)?;
    let cfg = grid(n, SCAN_BLOCK);
    launch!(dev, "scan_block_u32", cfg, input, &mut out, &mut sums, &n);
    if blocks > 1 {
        let offsets = scan(dev, &sums, blocks)?;
        launch!(dev, "scan_add_u32", cfg, &mut out, &offsets, &n);
    }
    Ok(out)
}

fn last(dev: &CudaDevice, p: &CudaSlice<u32>, n: u32) -> Result<u32> {
    let mut v = [0u32];
    dev.memcpy_dtoh(&p.slice(n as usize - 1..n as usize), &mut v)?;
    Ok(v[0])
}

// one round on the device; the new vertex and face counts
fn round(
    dev: &CudaDevice,
    verts: &mut CudaSlice<f32>,
    faces: &mut CudaSlice<u32>,
    (v_count, f_count): (u32, u32),
    threshold: f32,
) -> Result<(u32, u32)> {
    // vertex -> faces, each list ascending
    let mut count = filled(dev, v_count + 1, 0)?;
    launch_n!(dev, "vf_count", f_count, &*faces, &f_count, &mut count);
    let offsets = scan(dev, &count, v_count + 1)?;
    let mut fill = alloc::<u32>(dev, v_count as usize + 1)?;
    dev.memcpy_dtod(&offsets, &mut fill)?;
    let mut list = alloc::<u32>(dev, 3 * f_count as usize)?;
    launch_n!(
        dev, "vf_fill", f_count, &*faces, &f_count, &mut fill, &mut list
    );
    launch_n!(dev, "vf_sort", v_count, &offsets, &mut list, &v_count);
    // edges (v, higher neighbour), vertex-major and neighbour-ascending, with their face counts
    let mut edge_counts = filled(dev, v_count + 1, 0)?;
    launch_n!(
        dev,
        "edge_count",
        v_count,
        &offsets,
        &list,
        &*faces,
        &v_count,
        &mut edge_counts
    );
    let edge_offsets = scan(dev, &edge_counts, v_count + 1)?;
    let e_count = last(dev, &edge_offsets, v_count + 1)?;
    let mut edges = alloc::<u32>(dev, 2 * e_count as usize)?;
    let mut face_counts = alloc::<u32>(dev, e_count as usize)?;
    launch_n!(
        dev,
        "edge_fill",
        v_count,
        &offsets,
        &list,
        &*faces,
        &v_count,
        &edge_offsets,
        &mut edges,
        &mut face_counts
    );
    let mut boundary = filled(dev, v_count, 0)?;
    launch_n!(
        dev,
        "boundary_mark",
        e_count,
        &edges,
        &face_counts,
        &e_count,
        &mut boundary
    );
    let mut qems = alloc::<f32>(dev, 10 * v_count as usize)?;
    launch_n!(
        dev,
        "vertex_qem",
        v_count,
        &offsets,
        &list,
        &*faces,
        &*verts,
        &v_count,
        &mut qems
    );
    let mut costs = alloc::<f32>(dev, e_count as usize)?;
    launch_n!(
        dev,
        "edge_cost",
        e_count,
        &edges,
        &e_count,
        &*verts,
        &*faces,
        &offsets,
        &list,
        &boundary,
        &qems,
        &threshold,
        &LAMBDA_EDGE_LENGTH,
        &LAMBDA_SKINNY,
        &SKINNY_EPS,
        &mut costs
    );
    let mut best = alloc::<u64>(dev, f_count as usize)?;
    launch_n!(dev, "fill_u64", f_count, &mut best, &u64::MAX, &f_count);
    launch_n!(
        dev,
        "edge_best",
        e_count,
        &edges,
        &e_count,
        &costs,
        &offsets,
        &list,
        &mut best
    );
    let mut vertex_kept = flags(dev, v_count)?;
    let mut face_kept = flags(dev, f_count)?;
    launch_n!(
        dev,
        "edge_collapse",
        e_count,
        &edges,
        &e_count,
        &costs,
        &threshold,
        &offsets,
        &list,
        &best,
        &boundary,
        &mut *verts,
        &mut *faces,
        &mut vertex_kept,
        &mut face_kept
    );
    // compaction, order kept
    let remap = scan(dev, &vertex_kept, v_count + 1)?;
    let positions = scan(dev, &face_kept, f_count + 1)?;
    let (new_v, new_f) = (
        last(dev, &remap, v_count + 1)?,
        last(dev, &positions, f_count + 1)?,
    );
    let mut out_verts = alloc::<f32>(dev, 3 * new_v as usize)?;
    launch_n!(
        dev,
        "compact_vertices",
        v_count,
        &*verts,
        &vertex_kept,
        &remap,
        &v_count,
        &mut out_verts
    );
    let mut out_faces = alloc::<u32>(dev, 3 * new_f as usize)?;
    launch_n!(
        dev,
        "compact_faces",
        f_count,
        &*faces,
        &face_kept,
        &positions,
        &remap,
        &f_count,
        &mut out_faces
    );
    *verts = out_verts;
    *faces = out_faces;
    Ok((new_v, new_f))
}

/// [`super::cleanup::simplify`] on `dev`.
pub fn simplify(mesh: &mut Mesh, target: usize, dev: &CudaDevice) -> Result<()> {
    let mut f_count = mesh.faces.len() as u32;
    if f_count as usize <= target {
        return Ok(());
    }
    let mut v_count = mesh.vertices.len() as u32;
    let flat_v: Vec<f32> = mesh.vertices.iter().flatten().copied().collect();
    let flat_f: Vec<u32> = mesh.faces.iter().flatten().copied().collect();
    let mut verts = dev.clone_htod(&flat_v)?;
    let mut faces = dev.clone_htod(&flat_f)?;
    let mut threshold = START_THRESHOLD;
    loop {
        let before = f_count;
        (v_count, f_count) = round(dev, &mut verts, &mut faces, (v_count, f_count), threshold)?;
        if f_count as usize <= target {
            break;
        }
        if f64::from(before - f_count) / f64::from(before) < SLOW_ROUND {
            threshold *= THRESHOLD_STEP;
        }
    }
    let flat_v = dev.clone_dtoh(&verts.slice(..3 * v_count as usize))?;
    let flat_f = dev.clone_dtoh(&faces.slice(..3 * f_count as usize))?;
    mesh.vertices = flat_v.as_chunks::<3>().0.to_vec();
    mesh.faces = flat_f.as_chunks::<3>().0.to_vec();
    Ok(())
}

// points one query launch takes, bounding the device buffers
const QUERY_CHUNK: usize = 1 << 22;

/// A [`Bvh`] on a device, for [`Bvh::unsigned_distance`] queries there.
pub struct DeviceBvh {
    dev: CudaDevice,
    boxes: CudaSlice<f32>,
    links: CudaSlice<u32>,
    tris: CudaSlice<f32>,
    faces: CudaSlice<u32>,
}

impl DeviceBvh {
    pub fn new(bvh: &Bvh, dev: &CudaDevice) -> Result<Self> {
        let (boxes, links, tris, faces) = bvh.flat();
        Ok(Self {
            boxes: dev.clone_htod(&boxes)?,
            links: dev.clone_htod(&links)?,
            tris: dev.clone_htod(&tris)?,
            faces: dev.clone_htod(&faces)?,
            dev: dev.clone(),
        })
    }

    /// Each point's distance to the mesh, its closest face and the closest point, as [`Bvh::unsigned_distance`].
    pub fn unsigned_distance(&self, points: &[V3]) -> Result<Vec<(f32, u32, V3)>> {
        let dev = &self.dev;
        let mut out = Vec::with_capacity(points.len());
        for chunk in points.chunks(QUERY_CHUNK) {
            let n = chunk.len() as u32;
            let flat: Vec<f32> = chunk.iter().flatten().copied().collect();
            let p = dev.clone_htod(&flat)?;
            let mut distance = alloc::<f32>(dev, chunk.len())?;
            let mut face = alloc::<u32>(dev, chunk.len())?;
            let mut closest = alloc::<f32>(dev, 3 * chunk.len())?;
            launch_n!(
                dev,
                "bvh_closest",
                n,
                &self.boxes,
                &self.links,
                &self.tris,
                &self.faces,
                &p,
                &n,
                &mut distance,
                &mut face,
                &mut closest
            );
            let (d, f, c) = (
                dev.clone_dtoh(&distance)?,
                dev.clone_dtoh(&face)?,
                dev.clone_dtoh(&closest)?,
            );
            let c = c.as_chunks::<3>().0;
            out.extend((0..chunk.len()).map(|i| (d[i], f[i], c[i])));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    inference_tensor::gpu_cases!(cuda_cases, [the_device_simplifies_as_the_host_does]);

    // a bumpy w x h grid, so collapse costs differ and many rounds run
    fn bumpy_grid(w: usize, h: usize) -> Mesh {
        let mut mesh = Mesh::default();
        for y in 0..=h {
            for x in 0..=w {
                let z = ((x * 7 + y * 13) % 11) as f32 * 1e-3;
                mesh.vertices
                    .push([x as f32 / w as f32, y as f32 / h as f32, z]);
            }
        }
        let at = |x: usize, y: usize| (y * (w + 1) + x) as u32;
        for y in 0..h {
            for x in 0..w {
                mesh.faces.push([at(x, y), at(x + 1, y), at(x + 1, y + 1)]);
                mesh.faces.push([at(x, y), at(x + 1, y + 1), at(x, y + 1)]);
            }
        }
        mesh
    }

    fn the_device_simplifies_as_the_host_does() -> Result<()> {
        // no device here: nothing to compare
        let Ok(inference_tensor::Device::Cuda(dev)) = inference_tensor::Device::new_cuda(0) else {
            return Ok(());
        };
        let mut host = bumpy_grid(60, 40);
        let mut device = host.clone();
        super::super::cleanup::simplify(&mut host, 900);
        simplify(&mut device, 900, &dev)?;
        assert!(host.faces.len() <= 900);
        assert_eq!(host.faces, device.faces);
        assert_eq!(host.vertices, device.vertices);
        Ok(())
    }
}
