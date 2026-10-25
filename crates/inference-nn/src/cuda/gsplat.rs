//! 3D Gaussian splatting for one pinhole camera (see `kernels/cuda/gsplat.cu`): render RGB plus accumulated z-depth
//! from raw Gaussian parameters, and the gradients of those parameters from the render's.

use std::ffi::c_void;

use inference_tensor::cuda_backend::CudaStorageSlice;
use inference_tensor::cuda_backend::cudarc::driver::DevicePtr;
use inference_tensor::{DType, Device, Result, Storage, Tensor};

/// Channels of a render: rgb, then z accumulated by weight.
pub const CHANNELS: usize = 4;

/// Gaussians as optimized: centres `[n, 3]`, quaternions (w, x, y, z) `[n, 4]` (normalized inside), log scales
/// `[n, 3]`, opacity logits `[n]`, degree-0 SH `[n, 3]`; all contiguous f32 on one CUDA device.
pub struct Gaussians<'a> {
    pub means: &'a Tensor,
    pub quats: &'a Tensor,
    pub log_scales: &'a Tensor,
    pub logit_opacities: &'a Tensor,
    pub sh0: &'a Tensor,
}

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    /// world-to-camera, row-major, OpenCV axes
    pub viewmat: [f32; 16],
    /// intrinsics, row-major
    pub k: [f32; 9],
    pub width: usize,
    pub height: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct RasterSettings {
    /// added to the projected covariance's diagonal
    pub eps2d: f32,
    /// opacity scaled by the blur's area change (gsplat's antialiased mode), else left as is (classic)
    pub antialiased: bool,
    pub near_plane: f32,
    pub far_plane: f32,
    pub radius_clip: f32,
    pub tile_size: usize,
}

impl Default for RasterSettings {
    fn default() -> Self {
        Self {
            eps2d: 0.3,
            antialiased: true,
            near_plane: 0.01,
            far_plane: 1e10,
            radius_clip: 0.0,
            tile_size: 16,
        }
    }
}

/// One render's inputs: `gate` (`[n]`, MaskGaussian) multiplies each Gaussian's alpha in compositing.
pub struct Frame<'a> {
    pub gaussians: Gaussians<'a>,
    pub gate: Option<&'a Tensor>,
    pub camera: Camera,
    pub settings: RasterSettings,
}

/// A render and what its backward pass reads.
pub struct Render {
    /// `[h, w, CHANNELS]`: rgb and z accumulated by weight (divide by alpha for the expected depth)
    pub image: Tensor,
    /// `[h, w]`
    pub alpha: Tensor,
    /// `[n, 2]` i32 pixel radii, 0 where culled
    pub radii: Tensor,
    /// `[n, 2]` projected centres
    pub means2d: Tensor,
    pub depths: Tensor,
    conics: Tensor,
    compensations: Tensor,
    opacities: Tensor,
    colors: Tensor,
    offsets: Tensor,
    flatten_ids: Tensor,
    last_ids: Tensor,
    n_isects: usize,
}

/// Gradients of the raw parameters, plus those of the projected centres the densification reads.
pub struct Grads {
    pub means: Tensor,
    pub quats: Tensor,
    pub log_scales: Tensor,
    pub logit_opacities: Tensor,
    pub sh0: Tensor,
    /// set when the render had a gate
    pub gate: Option<Tensor>,
    pub means2d: Tensor,
    /// summed absolute per-pixel gradients of the projected centres, when asked for
    pub means2d_abs: Option<Tensor>,
}

fn status(code: i32, what: &str) -> Result<()> {
    if code != 0 {
        inference_tensor::bail!("{what} failed with CUDA status {code}");
    }
    Ok(())
}

fn i32_of(v: usize) -> Result<i32> {
    i32::try_from(v).map_err(inference_tensor::Error::wrap)
}

fn usize_of(v: i64) -> Result<usize> {
    usize::try_from(v).map_err(inference_tensor::Error::wrap)
}

// Device pointers of `tensors` (null for None) at their layout offsets, held for the duration of `run`.
fn with_ptrs(
    device: &Device,
    tensors: &[Option<&Tensor>],
    run: impl FnOnce(&[*mut c_void], i64) -> i32,
    what: &str,
) -> Result<()> {
    let Device::Cuda(dev) = device else {
        inference_tensor::bail!("{what} needs CUDA");
    };
    let stream = dev.cuda_stream();
    let storages = tensors
        .iter()
        .map(|t| t.map(Tensor::storage_and_layout))
        .collect::<Vec<_>>();
    let mut guards = Vec::new();
    let mut ptrs = Vec::with_capacity(tensors.len());
    for entry in &storages {
        let Some((storage, layout)) = entry else {
            ptrs.push(std::ptr::null_mut());
            continue;
        };
        let Storage::Cuda(s) = &**storage else {
            inference_tensor::bail!("{what} takes CUDA tensors");
        };
        let (base, size) = match &s.slice {
            CudaStorageSlice::F32(v) => {
                let (p, g) = v.device_ptr(&stream);
                guards.push(g);
                (p, 4)
            }
            CudaStorageSlice::I32(v) => {
                let (p, g) = v.device_ptr(&stream);
                guards.push(g);
                (p, 4)
            }
            CudaStorageSlice::I64(v) => {
                let (p, g) = v.device_ptr(&stream);
                guards.push(g);
                (p, 8)
            }
            _ => inference_tensor::bail!("{what} takes f32, i32 and i64 tensors"),
        };
        ptrs.push((base + (layout.start_offset() * size) as u64) as *mut c_void);
    }
    let code = run(&ptrs, stream.cu_stream() as i64);
    drop(guards);
    status(code, what)
}

fn check(t: &Tensor, dims: &[usize], what: &str) -> Result<()> {
    if t.dtype() != DType::F32 || !t.is_contiguous() || t.dims() != dims {
        inference_tensor::bail!(
            "{what} must be contiguous f32 {dims:?}, got {:?} {:?}",
            t.dtype(),
            t.dims()
        );
    }
    Ok(())
}

impl Gaussians<'_> {
    fn validate(&self, gate: Option<&Tensor>) -> Result<usize> {
        let n = self.means.dim(0)?;
        check(self.means, &[n, 3], "means")?;
        check(self.quats, &[n, 4], "quats")?;
        check(self.log_scales, &[n, 3], "log scales")?;
        check(self.logit_opacities, &[n], "opacity logits")?;
        check(self.sh0, &[n, 3], "sh0")?;
        if let Some(gate) = gate {
            check(gate, &[n], "gate")?;
        }
        Ok(n)
    }
}

pub fn render(frame: &Frame) -> Result<Render> {
    let (gaussians, gate, camera, settings) =
        (&frame.gaussians, frame.gate, &frame.camera, &frame.settings);
    use super::ffi::{
        gs_project_fwd, gs_raster_fwd, gs_tiles_count, gs_tiles_sort, gs_tiles_temp_bytes,
    };
    let n = gaussians.validate(gate)?;
    let device = gaussians.means.device().clone();
    let (w, h, ts) = (camera.width, camera.height, settings.tile_size);
    let (wi, hi, tsi, ni) = (i32_of(w)?, i32_of(h)?, i32_of(ts)?, i32_of(n)?);
    let zeros = |dims: &[usize], dtype: DType| Tensor::zeros(dims, dtype, &device);
    let radii = zeros(&[n, 2], DType::I32)?;
    let means2d = zeros(&[n, 2], DType::F32)?;
    let depths = zeros(&[n], DType::F32)?;
    let conics = zeros(&[n, 3], DType::F32)?;
    let compensations = zeros(&[n], DType::F32)?;
    let opacities = zeros(&[n], DType::F32)?;
    let colors = zeros(&[n, CHANNELS], DType::F32)?;
    let g = gaussians;
    with_ptrs(
        &device,
        &[
            Some(g.means),
            Some(g.quats),
            Some(g.log_scales),
            Some(g.logit_opacities),
            Some(g.sh0),
            Some(&radii),
            Some(&means2d),
            Some(&depths),
            Some(&conics),
            Some(&compensations),
            Some(&opacities),
            Some(&colors),
        ],
        |p, s| unsafe {
            // SAFETY: shapes checked above; the camera arrays are read on the host at launch
            gs_project_fwd(
                p[0],
                p[1],
                p[2],
                p[3],
                p[4],
                camera.viewmat.as_ptr(),
                camera.k.as_ptr(),
                ni,
                wi,
                hi,
                settings.eps2d,
                i32::from(settings.antialiased),
                settings.near_plane,
                settings.far_plane,
                settings.radius_clip,
                p[5],
                p[6],
                p[7],
                p[8],
                p[9],
                p[10],
                p[11],
                s,
            )
        },
        "gaussian projection",
    )?;

    let tiles_per_gauss = zeros(&[n], DType::I64)?;
    let cum_tiles = zeros(&[n], DType::I64)?;
    // SAFETY: a pure size query
    let scan_bytes = unsafe { gs_tiles_temp_bytes(ni, 0, wi, hi, tsi) };
    let temp = zeros(&[usize_of(scan_bytes)?.div_ceil(8).max(1)], DType::I64)?;
    with_ptrs(
        &device,
        &[
            Some(&radii),
            Some(&means2d),
            Some(&tiles_per_gauss),
            Some(&cum_tiles),
            Some(&temp),
        ],
        |p, s| unsafe {
            // SAFETY: n entries each, temp sized by the query
            gs_tiles_count(p[0], p[1], ni, wi, hi, tsi, p[2], p[3], p[4], scan_bytes, s)
        },
        "gaussian tile count",
    )?;
    let n_isects = if n == 0 {
        0
    } else {
        usize_of(cum_tiles.get(n - 1)?.to_scalar::<i64>()?)?
    };
    let (tw, th) = (w.div_ceil(ts), h.div_ceil(ts));
    // SAFETY: a pure size query
    let sort_bytes = unsafe { gs_tiles_temp_bytes(ni, n_isects as i64, wi, hi, tsi) };
    let temp = zeros(&[usize_of(sort_bytes)?.div_ceil(8).max(1)], DType::I64)?;
    let isects = n_isects.max(1);
    let keys = zeros(&[isects], DType::I64)?;
    let values = zeros(&[isects], DType::I32)?;
    let keys_sorted = zeros(&[isects], DType::I64)?;
    let flatten_ids = zeros(&[isects], DType::I32)?;
    let offsets = zeros(&[th, tw], DType::I32)?;
    with_ptrs(
        &device,
        &[
            Some(&radii),
            Some(&means2d),
            Some(&depths),
            Some(&cum_tiles),
            Some(&keys),
            Some(&values),
            Some(&keys_sorted),
            Some(&flatten_ids),
            Some(&offsets),
            Some(&temp),
        ],
        |p, s| unsafe {
            // SAFETY: n_isects entries in the key and value buffers, one offset per tile
            gs_tiles_sort(
                p[0],
                p[1],
                p[2],
                p[3],
                ni,
                wi,
                hi,
                tsi,
                n_isects as i64,
                p[4],
                p[5],
                p[6],
                p[7],
                p[8],
                p[9],
                sort_bytes,
                s,
            )
        },
        "gaussian tile sort",
    )?;

    let image = zeros(&[h, w, CHANNELS], DType::F32)?;
    let alpha = zeros(&[h, w], DType::F32)?;
    let last_ids = zeros(&[h, w], DType::I32)?;
    with_ptrs(
        &device,
        &[
            Some(&means2d),
            Some(&conics),
            Some(&colors),
            Some(&opacities),
            gate,
            Some(&offsets),
            Some(&flatten_ids),
            Some(&image),
            Some(&alpha),
            Some(&last_ids),
        ],
        |p, s| unsafe {
            // SAFETY: per-Gaussian buffers of n, per-pixel buffers of h * w
            gs_raster_fwd(
                p[0],
                p[1],
                p[2],
                p[3],
                p[4],
                p[5],
                p[6],
                wi,
                hi,
                tsi,
                n_isects as i64,
                p[7],
                p[8],
                p[9],
                s,
            )
        },
        "gaussian rasterization",
    )?;
    Ok(Render {
        image,
        alpha,
        radii,
        means2d,
        depths,
        conics,
        compensations,
        opacities,
        colors,
        offsets,
        flatten_ids,
        last_ids,
        n_isects,
    })
}

/// The parameter gradients given the gradients of `render.image` and `render.alpha`.
pub fn render_backward(
    frame: &Frame,
    render: &Render,
    v_image: &Tensor,
    v_alpha: &Tensor,
    abs_grad: bool,
) -> Result<Grads> {
    use super::ffi::{gs_project_bwd, gs_raster_bwd};
    let (gaussians, gate, camera, settings) =
        (&frame.gaussians, frame.gate, &frame.camera, &frame.settings);
    let n = gaussians.validate(gate)?;
    let (w, h) = (camera.width, camera.height);
    check(v_image, &[h, w, CHANNELS], "image gradient")?;
    check(v_alpha, &[h, w], "alpha gradient")?;
    let device = gaussians.means.device().clone();
    let (wi, hi, tsi, ni) = (
        i32_of(w)?,
        i32_of(h)?,
        i32_of(settings.tile_size)?,
        i32_of(n)?,
    );
    let zeros = |dims: &[usize]| Tensor::zeros(dims, DType::F32, &device);
    let v_means2d = zeros(&[n, 2])?;
    let v_means2d_abs = abs_grad.then(|| zeros(&[n, 2])).transpose()?;
    let v_conics = zeros(&[n, 3])?;
    let v_colors = zeros(&[n, CHANNELS])?;
    let v_opacities = zeros(&[n])?;
    let v_gate = gate.map(|_| zeros(&[n])).transpose()?;
    let r = render;
    with_ptrs(
        &device,
        &[
            Some(&r.means2d),
            Some(&r.conics),
            Some(&r.colors),
            Some(&r.opacities),
            gate,
            Some(&r.offsets),
            Some(&r.flatten_ids),
            Some(&r.alpha),
            Some(&r.last_ids),
            Some(v_image),
            Some(v_alpha),
            Some(&v_means2d),
            v_means2d_abs.as_ref(),
            Some(&v_conics),
            Some(&v_colors),
            Some(&v_opacities),
            v_gate.as_ref(),
        ],
        |p, s| unsafe {
            // SAFETY: the buffers of the forward pass this render came from, gradients zeroed
            gs_raster_bwd(
                p[0],
                p[1],
                p[2],
                p[3],
                p[4],
                p[5],
                p[6],
                wi,
                hi,
                tsi,
                r.n_isects as i64,
                p[7],
                p[8],
                p[9],
                p[10],
                p[11],
                p[12],
                p[13],
                p[14],
                p[15],
                p[16],
                s,
            )
        },
        "gaussian rasterization backward",
    )?;
    let v_means = zeros(&[n, 3])?;
    let v_quats = zeros(&[n, 4])?;
    let v_log_scales = zeros(&[n, 3])?;
    let v_logit_opacities = zeros(&[n])?;
    let v_sh0 = zeros(&[n, 3])?;
    let g = gaussians;
    with_ptrs(
        &device,
        &[
            Some(g.means),
            Some(g.quats),
            Some(g.log_scales),
            Some(g.logit_opacities),
            Some(g.sh0),
            Some(&r.radii),
            Some(&r.conics),
            Some(&r.compensations),
            Some(&v_means2d),
            Some(&v_conics),
            Some(&v_colors),
            Some(&v_opacities),
            Some(&v_means),
            Some(&v_quats),
            Some(&v_log_scales),
            Some(&v_logit_opacities),
            Some(&v_sh0),
        ],
        |p, s| unsafe {
            // SAFETY: n entries each, every output written
            gs_project_bwd(
                p[0],
                p[1],
                p[2],
                p[3],
                p[4],
                camera.viewmat.as_ptr(),
                camera.k.as_ptr(),
                ni,
                wi,
                hi,
                settings.eps2d,
                i32::from(settings.antialiased),
                p[5],
                p[6],
                p[7],
                p[8],
                p[9],
                p[10],
                p[11],
                p[12],
                p[13],
                p[14],
                p[15],
                p[16],
                s,
            )
        },
        "gaussian projection backward",
    )?;
    Ok(Grads {
        means: v_means,
        quats: v_quats,
        log_scales: v_log_scales,
        logit_opacities: v_logit_opacities,
        sh0: v_sh0,
        gate: v_gate,
        means2d: v_means2d,
        means2d_abs: v_means2d_abs,
    })
}

/// Adds one render's DefaultStrategy statistics: `grad2d += |v_means2d * (w/2, h/2)|`, `count += 1` and
/// `max_radii = max(max_radii, radius / max(w, h))` over the Gaussians the render saw (all three `[n]` f32).
pub fn densify_stats(
    render: &Render,
    v_means2d: &Tensor,
    camera: &Camera,
    grad2d: &Tensor,
    count: &Tensor,
    max_radii: &Tensor,
) -> Result<()> {
    let n = render.radii.dim(0)?;
    check(v_means2d, &[n, 2], "means2d gradient")?;
    for (t, what) in [
        (grad2d, "grad2d"),
        (count, "count"),
        (max_radii, "max radii"),
    ] {
        check(t, &[n], what)?;
    }
    let (ni, wi, hi) = (i32_of(n)?, i32_of(camera.width)?, i32_of(camera.height)?);
    with_ptrs(
        render.radii.device(),
        &[
            Some(v_means2d),
            Some(&render.radii),
            Some(grad2d),
            Some(count),
            Some(max_radii),
        ],
        |p, s| unsafe {
            // SAFETY: n entries each, updated in place
            super::ffi::gs_densify_stats(p[0], p[1], ni, wi, hi, p[2], p[3], p[4], s)
        },
        "gaussian densify stats",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE_ENV: &str = "INFERENCE_GSPLAT_FIXTURE";

    // a little-endian C-order f4 or f8 .npy, as f32 on the CPU
    #[allow(clippy::cast_possible_truncation)]
    fn read_npy(path: &std::path::Path) -> Result<Tensor> {
        let bytes = std::fs::read(path)?;
        let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        let header = std::str::from_utf8(&bytes[10..10 + header_len])
            .map_err(inference_tensor::Error::wrap)?;
        let shape_text = &header[header.find("'shape': (").expect("shape") + 10..];
        let shape = shape_text[..shape_text.find(')').expect("shape end")]
            .split(',')
            .filter_map(|d| d.trim().parse::<usize>().ok())
            .collect::<Vec<_>>();
        assert!(
            header.contains("'fortran_order': False"),
            "{}: C order only",
            path.display()
        );
        let body = &bytes[10 + header_len..];
        let values = if header.contains("'<f8'") {
            body.as_chunks::<8>()
                .0
                .iter()
                .map(|c| f64::from_le_bytes(*c) as f32)
                .collect::<Vec<_>>()
        } else {
            body.as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect()
        };
        Tensor::from_vec(values, shape, &Device::Cpu)
    }

    fn max_rel(got: &Tensor, want: &Tensor) -> Result<(f32, f32)> {
        let got = got.to_dtype(DType::F32)?.flatten_all()?;
        let want = want.to_dtype(DType::F32)?.flatten_all()?;
        let err = (&got - &want)?.abs()?;
        let scale = want.abs()?.max_all()?.to_scalar::<f32>()?.max(1e-12);
        let rms = (err.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?)
            / (want.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?).max(1e-12);
        Ok((err.max_all()?.to_scalar::<f32>()? / scale, rms))
    }

    // analytic gradients of a weighted sum of the render against central differences, on a small random scene
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn gradients_match_finite_differences() -> Result<()> {
        skip_without_cuda!();
        const N: usize = 24;
        const W: usize = 64;
        const H: usize = 48;
        const STEP: f32 = 2e-3;
        const MIN_COSINE: f64 = 0.99;
        let dev = Device::new_cuda(0)?;
        let mut state = 12_345u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 40) as f32 / (1u64 << 24) as f32
        };
        let mut params: Vec<Vec<f32>> = vec![
            (0..N)
                .flat_map(|_| [next() * 2.0 - 1.0, next() * 1.4 - 0.7, 3.0 + next() * 2.0])
                .collect(),
            (0..N * 4).map(|_| next() * 2.0 - 1.0).collect(),
            (0..N * 3).map(|_| -2.2 + next() * 0.8).collect(),
            (0..N).map(|_| next() * 2.0 - 0.5).collect(),
            (0..N * 3).map(|_| next() * 2.0 - 1.0).collect(),
            (0..N).map(|_| 0.3 + next() * 0.7).collect(),
        ];
        let dims: [&[usize]; 6] = [&[N, 3], &[N, 4], &[N, 3], &[N], &[N, 3], &[N]];
        let names = [
            "means",
            "quats",
            "log scales",
            "opacity logits",
            "sh0",
            "gate",
        ];
        let weights_image = (0..H * W * CHANNELS).map(|_| next()).collect::<Vec<_>>();
        let weights_alpha = (0..H * W).map(|_| next()).collect::<Vec<_>>();
        let camera = Camera {
            viewmat: [
                1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
            ],
            k: [60., 0., W as f32 / 2.0, 0., 60., H as f32 / 2.0, 0., 0., 1.],
            width: W,
            height: H,
        };
        let tensors = |params: &[Vec<f32>]| -> Result<Vec<Tensor>> {
            params
                .iter()
                .zip(dims)
                .map(|(p, d)| Tensor::from_vec(p.clone(), d, &dev))
                .collect()
        };
        let frame = |t: &[Tensor]| -> Result<Render> {
            render(&Frame {
                gaussians: Gaussians {
                    means: &t[0],
                    quats: &t[1],
                    log_scales: &t[2],
                    logit_opacities: &t[3],
                    sh0: &t[4],
                },
                gate: Some(&t[5]),
                camera,
                settings: RasterSettings::default(),
            })
        };
        let loss = |r: &Render| -> Result<f64> {
            let image = r.image.flatten_all()?.to_vec1::<f32>()?;
            let alpha = r.alpha.flatten_all()?.to_vec1::<f32>()?;
            let a = image
                .iter()
                .zip(&weights_image)
                .map(|(x, w)| f64::from(x * w))
                .sum::<f64>();
            Ok(a + alpha
                .iter()
                .zip(&weights_alpha)
                .map(|(x, w)| f64::from(x * w))
                .sum::<f64>())
        };
        let t = tensors(&params)?;
        let r = frame(&t)?;
        let g = render_backward(
            &Frame {
                gaussians: Gaussians {
                    means: &t[0],
                    quats: &t[1],
                    log_scales: &t[2],
                    logit_opacities: &t[3],
                    sh0: &t[4],
                },
                gate: Some(&t[5]),
                camera,
                settings: RasterSettings::default(),
            },
            &r,
            &Tensor::from_vec(weights_image.clone(), (H, W, CHANNELS), &dev)?,
            &Tensor::from_vec(weights_alpha.clone(), (H, W), &dev)?,
            false,
        )?;
        let analytic = [
            &g.means,
            &g.quats,
            &g.log_scales,
            &g.logit_opacities,
            &g.sh0,
            g.gate.as_ref().expect("gated"),
        ];
        for (k, name) in names.iter().enumerate() {
            let want = analytic[k].flatten_all()?.to_vec1::<f32>()?;
            let mut numeric = Vec::with_capacity(want.len());
            for j in 0..params[k].len() {
                let base = params[k][j];
                params[k][j] = base + STEP;
                let up = loss(&frame(&tensors(&params)?)?)?;
                params[k][j] = base - STEP;
                let down = loss(&frame(&tensors(&params)?)?)?;
                params[k][j] = base;
                numeric.push((up - down) / (2.0 * f64::from(STEP)));
            }
            let dot = want
                .iter()
                .zip(&numeric)
                .map(|(a, b)| f64::from(*a) * b)
                .sum::<f64>();
            let na = want
                .iter()
                .map(|a| f64::from(*a).powi(2))
                .sum::<f64>()
                .sqrt();
            let nb = numeric.iter().map(|b| b.powi(2)).sum::<f64>().sqrt();
            let cosine = dot / (na * nb).max(1e-30);
            println!("{name:15} cosine {cosine:.5}  |analytic| {na:.4e}  |numeric| {nb:.4e}");
            assert!(cosine > MIN_COSINE, "{name}: cosine {cosine}");
        }
        Ok(())
    }

    // the reference fork's forward outputs and gradients for one camera of a trained scene (scripts/gsplat_parity.sh)
    #[test]
    #[ignore = "reads a gsplat fixture"]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn matches_gsplat_fixture() -> Result<()> {
        let dir = std::path::PathBuf::from(std::env::var(FIXTURE_ENV).expect("fixture dir"));
        let dev = Device::new_cuda(0)?;
        let load = |name: &str| -> Result<Tensor> {
            read_npy(&dir.join(format!("{name}.npy")))?.to_device(&dev)
        };
        let means = load("means")?;
        let n = means.dim(0)?;
        let (quats, log_scales, logits) = (
            load("quats")?,
            load("log_scales")?,
            load("logit_opacities")?,
        );
        let sh0 = load("sh0")?.reshape((n, 3))?;
        let gate = load("gate")?;
        let host = |name: &str| -> Result<Vec<f32>> {
            read_npy(&dir.join(format!("{name}.npy")))?
                .flatten_all()?
                .to_vec1::<f32>()
        };
        let size = read_npy(&dir.join("size.npy"))?.to_vec1::<f32>()?;
        let viewmat: [f32; 16] = host("viewmat")?.try_into().expect("4x4");
        let k: [f32; 9] = host("K")?.try_into().expect("3x3");
        let frame = Frame {
            gaussians: Gaussians {
                means: &means,
                quats: &quats,
                log_scales: &log_scales,
                logit_opacities: &logits,
                sh0: &sh0,
            },
            gate: Some(&gate),
            camera: Camera {
                viewmat,
                k,
                width: size[0] as usize,
                height: size[1] as usize,
            },
            settings: RasterSettings {
                near_plane: size[2],
                far_plane: size[3],
                ..RasterSettings::default()
            },
        };
        let r = render(&frame)?;
        let alpha = r.alpha.unsqueeze(2)?;
        // the fixture's fourth channel is the expected depth
        let safe = alpha.clamp(1e-10f32, f32::MAX)?;
        let rgb = r.image.narrow(2, 0, 3)?;
        let depth = r.image.narrow(2, 3, 1)?;
        let ours = Tensor::cat(&[&rgb, &depth.broadcast_div(&safe)?], 2)?;
        let want = load("render")?;
        let report = |what: &str, got: &Tensor, want: &Tensor| -> Result<f32> {
            let (max, rms) = max_rel(got, want)?;
            println!("{what:14} max rel {max:.2e}  rms rel {rms:.2e}");
            Ok(rms)
        };
        let radii_match = r
            .radii
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .eq(&read_npy(&dir.join("radii.npy"))?)?
            .to_dtype(DType::F32)?
            .mean_all()?
            .to_scalar::<f32>()?;
        println!("radii equal for {:.6} of gaussians", radii_match);
        let mut worst = 0f32;
        worst = worst.max(report(
            "rgb",
            &ours.narrow(2, 0, 3)?,
            &want.narrow(2, 0, 3)?,
        )?);
        worst = worst.max(report(
            "depth",
            &ours.narrow(2, 3, 1)?,
            &want.narrow(2, 3, 1)?,
        )?);
        worst = worst.max(report("alpha", &alpha, &load("alpha")?)?);

        // d/d(render * wr + alpha * wa) through the expected-depth division
        let (wr, wa) = (load("weight_render")?, load("weight_alpha")?);
        let w_depth = wr.narrow(2, 3, 1)?;
        let live = alpha.ge(1e-10f32)?.to_dtype(DType::F32)?;
        let v_depth = w_depth.broadcast_div(&safe)?;
        let v_alpha = (wa
            - (w_depth * depth)?
                .broadcast_div(&safe.sqr()?)?
                .broadcast_mul(&live)?)?;
        let v_image = Tensor::cat(&[&wr.narrow(2, 0, 3)?, &v_depth], 2)?.contiguous()?;
        let g = render_backward(
            &frame,
            &r,
            &v_image,
            &v_alpha.squeeze(2)?.contiguous()?,
            false,
        )?;
        worst = worst.max(report("v_means2d", &g.means2d, &load("v_means2d")?)?);
        worst = worst.max(report("grad means", &g.means, &load("grad_means")?)?);
        worst = worst.max(report("grad quats", &g.quats, &load("grad_quats")?)?);
        worst = worst.max(report(
            "grad scales",
            &g.log_scales,
            &load("grad_log_scales")?,
        )?);
        worst = worst.max(report(
            "grad opacity",
            &g.logit_opacities,
            &load("grad_logit_opacities")?,
        )?);
        worst = worst.max(report(
            "grad sh0",
            &g.sh0,
            &load("grad_sh0")?.reshape((n, 3))?,
        )?);
        worst = worst.max(report(
            "grad gate",
            g.gate.as_ref().expect("gated"),
            &load("grad_gate")?,
        )?);
        assert!(
            radii_match > 0.999,
            "radii differ for {:.4}",
            1.0 - radii_match
        );
        assert!(worst < 1e-3, "rms rel {worst}");
        Ok(())
    }
}
