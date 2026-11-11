//! TRELLIS.2's texturing pipeline (`Trellis2TexturingPipeline.run`): a given mesh painted from an image. The mesh is
//! centered in the unit cube (z up), turned into a flexible dual grid and encoded to shape latents; the texture flow
//! samples PBR latents beside them under the image's DINOv3 tokens, and the texture decoder unfolds them back onto the
//! mesh's own surface cells, where the bake reads them.

use inference_tensor::{DType, Device, Result, Tensor};
use rand::SeedableRng;
use rand::rngs::StdRng;

use super::crop_object;
use super::decoder::{ShapeDecoder, TEXTURE_DECODER};
use super::encoder::{Hierarchy, SHAPE_ENCODER, ShapeEncoder};
use super::mesh::Mesh;
use super::pipeline::{
    SamplerChoice, TEXTURE_512_FILE, TEXTURE_1024_FILE, TEXTURE_DECODER_FILE, Trellis2Paths,
    condition, noise, stats, weights,
};
use super::shape_flow::{ShapeFlow, TEXTURE_FLOW, TEXTURE_SAMPLER, denormalize, normalize};
use super::voxelize::flexible_dual_grid;
use crate::dinov3::{Dinov3, Dinov3Spec};
use crate::jovox::Rgba;
use crate::progress::{Progress, check};

pub const ENCODER_FILE: &str = "ckpts/shape_enc_next_dc_f16c32_fp16.safetensors";
/// The TRELLIS.2-4B files texturing reads beyond the shape stage's
pub const FILES: [&str; 5] = [
    super::PIPELINE_FILE,
    ENCODER_FILE,
    TEXTURE_512_FILE,
    TEXTURE_1024_FILE,
    TEXTURE_DECODER_FILE,
];
/// The grid resolutions texturing runs at; the reference's default is the higher
pub const RESOLUTIONS: [usize; 2] = [512, 1024];
// `preprocess_mesh`: the longest side just inside the unit cube
const FIT: f32 = 0.99999;
const PBR: usize = 6;

/// What a texturing run samples.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TexturingSettings {
    /// 512 or 1024
    pub resolution: usize,
    pub seed: u64,
    pub texture: SamplerChoice,
}

impl Default for TexturingSettings {
    fn default() -> Self {
        Self {
            resolution: 1024,
            seed: 0,
            texture: SamplerChoice::default(),
        }
    }
}

/// `preprocess_mesh`: `mesh` centered in the unit cube, its longest side just inside it, y up turned to z up.
pub fn fit_mesh(mesh: &mut Mesh) {
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for v in &mesh.vertices {
        for k in 0..3 {
            lo[k] = lo[k].min(v[k]);
            hi[k] = hi[k].max(v[k]);
        }
    }
    let extent = (0..3).map(|k| hi[k] - lo[k]).fold(0f32, f32::max);
    let scale = if extent > 0.0 { FIT / extent } else { 1.0 };
    let center: [f32; 3] = std::array::from_fn(|k| (lo[k] + hi[k]) / 2.0);
    for v in &mut mesh.vertices {
        let p: [f32; 3] = std::array::from_fn(|k| (v[k] - center[k]) * scale);
        *v = [p[0], -p[2], p[1]];
    }
}

/// The texture decoder's PBR attributes (in [0, 1]) on the mesh's surface cells.
pub struct Painted {
    pub voxels: Vec<[i32; 3]>,
    pub attrs: Vec<[f32; PBR]>,
    pub surface_cells: usize,
    pub latent_cells: usize,
}

/// `mesh` (already fitted, [`fit_mesh`]) painted from the object in `image` (its background already removed). The
/// flow runs in bf16 and the encoder and decoder in f16 on a GPU, all in f32 on the CPU.
pub fn paint(
    paths: &Trellis2Paths,
    image: &Rgba,
    mesh: &Mesh,
    settings: &TexturingSettings,
    progress: &dyn Progress,
    device: &Device,
) -> Result<Painted> {
    let res = settings.resolution;
    if !RESOLUTIONS.contains(&res) {
        inference_tensor::bail!("texturing runs at 512 or 1024, not {res}");
    }
    let (flow_dtype, net_dtype) = if device.is_cpu() {
        (DType::F32, DType::F32)
    } else {
        (DType::BF16, DType::F16)
    };
    let stats = stats(&paths.trellis2)?;

    progress.stage("condition", 0);
    let crop = crop_object(image)?;
    let cond = {
        let encoder = Dinov3::load(
            Dinov3Spec::vitl16(DType::F32),
            weights(&paths.dinov3.join("model.safetensors"), device)?,
        )?;
        condition(&encoder, &crop, res, device)?
    };
    check(progress)?;

    progress.stage("encode", 0);
    let grid = flexible_dual_grid(mesh, res);
    if grid.cells.is_empty() {
        inference_tensor::bail!("the mesh crosses no cell of the grid");
    }
    check(progress)?;
    let hierarchy = Hierarchy::new(&grid.cells, device)?;
    let shape = {
        let encoder = ShapeEncoder::load(
            SHAPE_ENCODER,
            net_dtype,
            weights(&paths.trellis2.join(ENCODER_FILE), device)?,
        )?;
        encoder.forward(&grid, &hierarchy, device)?.unsqueeze(0)?
    };
    let latent_cells = hierarchy.coarsest().len();
    check(progress)?;

    // the texture latents, the normalized shape latents beside the noise
    let cells: Vec<[usize; 3]> = hierarchy
        .coarsest()
        .iter()
        .map(|c| c.map(|v| v as usize))
        .collect();
    let params = settings.texture.apply(TEXTURE_SAMPLER);
    let tex = {
        let file = if res == 512 {
            TEXTURE_512_FILE
        } else {
            TEXTURE_1024_FILE
        };
        let flow = ShapeFlow::load(
            TEXTURE_FLOW,
            flow_dtype,
            weights(&paths.trellis2.join(file), device)?,
        )?;
        let kv = flow.condition(&Tensor::cat(&[&cond, &cond.zeros_like()?], 0)?)?;
        let rope = flow.rope(&cells)?;
        let shape_norm = normalize(&shape, &stats.shape.0, &stats.shape.1)?;
        let mut rng = StdRng::seed_from_u64(settings.seed);
        let x = noise(&mut rng, cells.len(), TEXTURE_FLOW.out_channels, device)?;
        progress.stage("texture", params.steps);
        let tex = flow.sample(&x, &rope, &kv, Some(&shape_norm), &params, &mut |step| {
            progress.advance(step);
            check(progress)
        })?;
        denormalize(&tex, &stats.texture.0, &stats.texture.1)?
    };

    progress.stage("decode", 0);
    let decoder = ShapeDecoder::load(
        TEXTURE_DECODER,
        net_dtype,
        weights(&paths.trellis2.join(TEXTURE_DECODER_FILE), device)?,
    )?;
    let (out, _) = decoder.forward(
        hierarchy.coarsest(),
        &tex.squeeze(0)?,
        Some(&hierarchy.subdivisions),
        &mut |_, _| {},
    )?;
    let attrs = ((out.feats * 0.5)? + 0.5)?
        .to_vec2::<f32>()?
        .iter()
        .map(|r| std::array::from_fn(|k| r[k]))
        .collect();
    check(progress)?;
    Ok(Painted {
        voxels: out.coords,
        attrs,
        surface_cells: grid.cells.len(),
        latent_cells,
    })
}
