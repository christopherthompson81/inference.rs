//! One image to Idol's Gaussians: preprocess, encode, sample the latent and camera with guidance, place anchors with
//! the octree decoder and decode 32 Gaussians each. Each model loads for its stage and is dropped after it.

use std::path::{Path, PathBuf};

use inference_tensor::nn::VarBuilder;
use inference_tensor::{DType, Device, Result, Tensor};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};

use super::conditioning::{ImageEncoders, VaeLatent};
use super::decoder::{
    GAUSSIANS_PER_ANCHOR, GaussianDecoder, Gaussians, IDOL_GAUSSIAN_SETTINGS, IDOL_GAUSSIANS,
    IDOL_OCTREE, OctreeDecoder, decode_in_batches,
};
use super::dit::{Dit, IDOL_DIT};
use super::preprocess::{IMAGE_SIZE, MARGIN_SYNTHETIC, prepare};
use crate::birefnet::BiRefNet;
use crate::jovox::Rgba;
use crate::progress::{Progress, check};

/// The checkpoints Idol reads.
pub struct IdolPaths {
    /// `speridlabs/idol`'s folder (`denoiser`, `decoder`, `decoder_gs`, `birefnet` safetensors)
    pub idol: PathBuf,
    /// DINOv3 ViT-H+/16's `model.safetensors`
    pub dinov3: PathBuf,
    /// a FLUX.2 VAE's `diffusion_pytorch_model.safetensors`
    pub vae: PathBuf,
}

/// The octree's depth (`max_voxel_level`): anchors land in 256^3 cells.
const OCTREE_LEVEL: u32 = 8;
/// The Gaussian counts the reference accepts.
pub const MIN_GAUSSIANS: usize = 32_768;
pub const MAX_GAUSSIANS: usize = 1_048_576;
/// Sampling steps past which a run is refused.
pub const MAX_STEPS: usize = 200;
// anchors the Gaussian decoder takes at once (each becomes 32 Gaussians)
const DECODE_BATCH: usize = 4096;
// the time the denoiser reads is t x 1000
const TIME_SCALE: f64 = 1000.0;

/// How a run samples; the defaults are the reference's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IdolSettings {
    pub seed: u64,
    pub steps: usize,
    /// guided velocity = (1 + w) v(cond) - w v(uncond)
    pub guidance: f64,
    /// shifts the time grid toward t = 1 (more steps at high noise)
    pub rescale_t: f64,
    /// rounded down to a multiple of 32
    pub gaussians: usize,
    /// crop side over the object's bounding box ([`super::MARGIN_SYNTHETIC`], [`super::MARGIN_REAL`])
    pub margin: f64,
}

impl Default for IdolSettings {
    fn default() -> Self {
        Self {
            seed: 42,
            steps: 50,
            guidance: 3.0,
            rescale_t: 3.0,
            gaussians: 262_144,
            margin: MARGIN_SYNTHETIC,
        }
    }
}

/// The input's predicted camera.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    /// radians
    pub yaw: f64,
    /// radians
    pub pitch: f64,
    pub radius: f64,
    /// vertical field of view, degrees
    pub fov: f64,
}

// `decode_camera`: a direction, an inverse distance (orthographic near 0) and the half width of the view
fn decode_camera(latent: &[f32]) -> Camera {
    const ORTHOGRAPHIC_INVERSE: f64 = 1e-3;
    const ORTHOGRAPHIC_RADIUS: f64 = 1000.0;
    let d = [latent[0] as f64, latent[1] as f64, latent[2] as f64];
    let (inverse, half_width) = (latent[3] as f64, latent[4] as f64);
    let norm = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    let [dx, dy, dz] = if norm > 1e-6 {
        d.map(|v| v / norm)
    } else {
        [0.0, 0.0, 1.0]
    };
    let (radius, fov) = if inverse < ORTHOGRAPHIC_INVERSE {
        (
            ORTHOGRAPHIC_RADIUS,
            (2.0 * (half_width / ORTHOGRAPHIC_RADIUS).atan()).to_degrees(),
        )
    } else {
        (
            1.0 / inverse,
            (2.0 * (half_width * inverse).atan()).to_degrees(),
        )
    };
    Camera {
        yaw: dx.atan2(dy),
        pitch: dz.clamp(-1.0, 1.0).asin(),
        radius,
        fov,
    }
}

/// A run's result.
pub struct IdolOutput {
    pub gaussians: Gaussians,
    pub camera: Camera,
    /// the RGB image the encoders saw, `IDOL_IMAGE_SIZE`^2
    pub prepared: Vec<u8>,
}

/// The encoders' input side.
pub const IDOL_IMAGE_SIZE: usize = IMAGE_SIZE;

fn weights(path: &Path, device: &Device) -> Result<VarBuilder<'static>> {
    // SAFETY: the checkpoint files are not modified while the model is loaded
    unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device) }
}

fn standard_normal(
    rng: &mut StdRng,
    shape: (usize, usize, usize),
    device: &Device,
) -> Result<Tensor> {
    let n = shape.0 * shape.1 * shape.2;
    let v: Vec<f32> = (0..n).map(|_| StandardNormal.sample(rng)).collect();
    Tensor::from_vec(v, shape, device)
}

/// The object in `image` as Idol's Gaussians. The models run as shipped on a GPU (encoders bf16, blocks fp16) and in
/// f32 on the CPU.
pub fn generate(
    paths: &IdolPaths,
    image: &Rgba,
    settings: &IdolSettings,
    progress: &dyn Progress,
    device: &Device,
) -> Result<IdolOutput> {
    let s = *settings;
    if !(1..=MAX_STEPS).contains(&s.steps) {
        inference_tensor::bail!("steps must be 1 to {MAX_STEPS}, got {}", s.steps);
    }
    if !(MIN_GAUSSIANS..=MAX_GAUSSIANS).contains(&s.gaussians) {
        inference_tensor::bail!(
            "gaussians must be {MIN_GAUSSIANS} to {MAX_GAUSSIANS}, got {}",
            s.gaussians
        );
    }
    let (encoder_dtype, block_dtype) = if device.is_cpu() {
        (DType::F32, DType::F32)
    } else {
        (DType::BF16, DType::F16)
    };
    let mut rng = StdRng::seed_from_u64(s.seed);

    progress.stage("preprocess", 0);
    let prepared = prepare(image, s.margin, &mut || {
        BiRefNet::load_file(&paths.idol.join("birefnet.safetensors"), device)
    })?;
    check(progress)?;

    progress.stage("condition", 0);
    let pixels: Vec<f32> = prepared.iter().map(|&v| f32::from(v) / 255.0).collect();
    let pixels = Tensor::from_vec(pixels, (1, IMAGE_SIZE, IMAGE_SIZE, 3), device)?
        .permute((0, 3, 1, 2))?
        .contiguous()?;
    let (feature1, feature2) = {
        let encoders = ImageEncoders::load(
            &paths.dinov3,
            &paths.vae,
            encoder_dtype,
            encoder_dtype,
            device,
        )?;
        encoders.encode(&pixels, VaeLatent::Sample, &mut rng)?
    };
    check(progress)?;

    progress.stage("sample", s.steps);
    let dit = Dit::load(
        IDOL_DIT,
        block_dtype,
        weights(&paths.idol.join("denoiser.safetensors"), device)?,
    )?;
    let guided = s.guidance > 0.0;
    let condition = if guided {
        let both1 = Tensor::cat(&[&feature1, &feature1.zeros_like()?], 0)?;
        let both2 = Tensor::cat(&[&feature2, &feature2.zeros_like()?], 0)?;
        dit.condition(&both1, &both2)?
    } else {
        dit.condition(&feature1, &feature2)?
    };
    drop((feature1, feature2));
    let mut latent = standard_normal(
        &mut rng,
        (1, IDOL_DIT.latent_tokens, IDOL_DIT.latent_channels),
        device,
    )?;
    let mut camera = standard_normal(&mut rng, (1, 1, IDOL_DIT.camera_channels), device)?;
    let grid: Vec<f64> = (0..=s.steps)
        .map(|i| {
            let t = 1.0 - i as f64 / s.steps as f64;
            s.rescale_t * t / (1.0 + (s.rescale_t - 1.0) * t)
        })
        .collect();
    for step in 0..s.steps {
        let (t, t_prev) = (grid[step], grid[step + 1]);
        let (v_latent, v_camera) = if guided {
            let x = Tensor::cat(&[&latent, &latent], 0)?;
            let c = Tensor::cat(&[&camera, &camera], 0)?;
            let time = Tensor::new(&[(t * TIME_SCALE) as f32; 2], device)?;
            let v = dit.forward(&x, &c, &time, &condition)?;
            let mix = |both: &Tensor| -> Result<Tensor> {
                let cond = both.narrow(0, 0, 1)?;
                let uncond = both.narrow(0, 1, 1)?;
                (cond * (1.0 + s.guidance))? - (uncond * s.guidance)?
            };
            (mix(&v.latent)?, mix(&v.camera)?)
        } else {
            let time = Tensor::new(&[(t * TIME_SCALE) as f32], device)?;
            let v = dit.forward(&latent, &camera, &time, &condition)?;
            (v.latent, v.camera)
        };
        latent = (latent - (v_latent * (t - t_prev))?)?;
        camera = (camera - (v_camera * (t - t_prev))?)?;
        progress.advance(step + 1);
        check(progress)?;
    }
    drop((dit, condition));
    let camera = decode_camera(&camera.flatten_all()?.to_vec1::<f32>()?);

    progress.stage("decode", 0);
    let anchors = {
        let octree = OctreeDecoder::load(
            IDOL_OCTREE,
            block_dtype,
            weights(&paths.idol.join("decoder.safetensors"), device)?,
        )?;
        octree.sample(
            &latent,
            (s.gaussians / GAUSSIANS_PER_ANCHOR).max(1),
            OCTREE_LEVEL,
            &mut rng,
        )?
    };
    check(progress)?;
    let decoder = GaussianDecoder::load(
        IDOL_GAUSSIANS,
        IDOL_GAUSSIAN_SETTINGS,
        block_dtype,
        weights(&paths.idol.join("decoder_gs.safetensors"), device)?,
    )?;
    let gaussians = decode_in_batches(&decoder, &anchors, &latent, DECODE_BATCH)?;
    Ok(IdolOutput {
        gaussians,
        camera,
        prepared,
    })
}
