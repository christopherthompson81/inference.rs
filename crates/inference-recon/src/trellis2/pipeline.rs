//! TRELLIS.2 end to end (`Trellis2ImageTo3DPipeline.run`): DINOv3 conditions, the sparse structure (a dense flow and
//! TRELLIS's structure decoder), the shape latents (at 512 or 1024, or 512 then 1024 or 1536 on the decoder's
//! upsampled cells), the texture latents beside them, and their decoding into a mesh with PBR attributes at its
//! vertices. Each model is loaded for its stage and dropped after it.

use std::path::{Path, PathBuf};

use inference_tensor::nn::VarBuilder;
use inference_tensor::{DType, Device, Result, Tensor};
use inference_vision::PilFilter;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};

use super::bake;
use super::decoder::{SHAPE_DECODER, ShapeDecoder, TEXTURE_DECODER, dual_grid};
use super::glb::{Pbr, sample_attributes};
use super::mesh::{Mesh, dual_grid_mesh};
use super::shape_flow::{
    SHAPE_FLOW_1024, SHAPE_SAMPLER, STRUCTURE_FLOW, STRUCTURE_SAMPLER, SamplerParams, ShapeFlow,
    ShapeFlowSpec, TEXTURE_FLOW, TEXTURE_SAMPLER, denormalize, normalize,
};
use super::structure::{STRUCTURE_DECODER, StructureDecoder, grid_cells, occupied};
use super::{PIPELINE_FILE, crop_object};
use crate::dinov3::{Dinov3, Dinov3Spec};
use crate::jovox::Rgba;
use crate::progress::{Progress, check};

const STRUCTURE_FLOW_FILE: &str = "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors";
const SHAPE_512_FILE: &str = "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors";
const SHAPE_1024_FILE: &str = "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors";
const TEXTURE_512_FILE: &str = "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors";
const TEXTURE_1024_FILE: &str = "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors";
const SHAPE_DECODER_FILE: &str = "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors";
const TEXTURE_DECODER_FILE: &str = "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors";
/// TRELLIS-image-large's structure decoder, which TRELLIS.2 reuses
pub const STRUCTURE_DECODER_FILE: &str = "ckpts/ss_dec_conv3d_16l8_fp16.safetensors";
/// The TRELLIS.2-4B files the whole pipeline reads.
pub const FILES: [&str; 8] = [
    PIPELINE_FILE,
    STRUCTURE_FLOW_FILE,
    SHAPE_512_FILE,
    SHAPE_1024_FILE,
    TEXTURE_512_FILE,
    TEXTURE_1024_FILE,
    SHAPE_DECODER_FILE,
    TEXTURE_DECODER_FILE,
];
// the structure latent's grid, and the decoder's four doublings
const LATENT_SIDE: usize = 16;
const UPSAMPLES: usize = 4;
const LOW: usize = 512;
const HIGH: usize = 1024;
const HIGHEST: usize = 1536;
/// A cascade past 1024 steps its resolution down by [`RESOLUTION_STEP`] until its cells number fewer than this
pub const MAX_TOKENS: usize = 49152;
const RESOLUTION_STEP: usize = 128;
// the latents' cells sit at the mesh resolution over 16
const CELL_SCALE: usize = 16;
const PBR: usize = 6;

/// Which resolutions the shape and texture run at (the reference's `pipeline_type`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    /// Everything at 512 (a 32^3 structure)
    Res512,
    /// Everything at 1024, on a 64^3 structure
    Res1024,
    /// The reference's default: shape at 512, upsampled, again at 1024; texture at 1024
    Cascade1024,
    /// Shape at 512, upsampled, again at 1536 (lower when that is [`MAX_TOKENS`] cells or more); texture at 1024
    Cascade1536,
}

impl Variant {
    pub const ALL: [Self; 4] = [
        Self::Res512,
        Self::Res1024,
        Self::Cascade1024,
        Self::Cascade1536,
    ];

    /// The reference's name for it
    pub const fn name(self) -> &'static str {
        match self {
            Self::Res512 => "512",
            Self::Res1024 => "1024",
            Self::Cascade1024 => "1024_cascade",
            Self::Cascade1536 => "1536_cascade",
        }
    }

    /// From its name, with `-` or `_`
    pub fn parse(name: &str) -> Option<Self> {
        let name = name.replace('-', "_");
        Self::ALL.into_iter().find(|v| v.name() == name)
    }

    // the occupancy grid the structure is pooled to
    fn structure_side(self) -> usize {
        match self {
            Self::Res1024 => HIGH / CELL_SCALE,
            _ => LOW / CELL_SCALE,
        }
    }

    // the resolution a cascade samples its shape at again, after 512
    fn cascade(self) -> Option<usize> {
        match self {
            Self::Cascade1024 => Some(HIGH),
            Self::Cascade1536 => Some(HIGHEST),
            Self::Res512 | Self::Res1024 => None,
        }
    }
}

// the 512 latents' upsampled cells at `target` (lower by RESOLUTION_STEP while MAX_TOKENS or more, down to 1024)
fn cascade_cells(fine: &[[i32; 3]], target: usize) -> (Vec<[usize; 3]>, usize) {
    let mut resolution = target;
    loop {
        let side = (resolution / CELL_SCALE) as f32;
        let mut cells: Vec<[usize; 3]> = fine
            .iter()
            .map(|c| c.map(|v| ((v as f32 + 0.5) / LOW as f32 * side) as usize))
            .collect();
        cells.sort_unstable();
        cells.dedup();
        if cells.len() < MAX_TOKENS || resolution <= HIGH {
            return (cells, resolution);
        }
        resolution -= RESOLUTION_STEP;
    }
}

/// Where the weights are.
#[derive(Clone, Debug)]
pub struct Trellis2Paths {
    /// TRELLIS.2-4B's folder (`pipeline.json`, `ckpts/`)
    pub trellis2: PathBuf,
    /// TRELLIS-image-large's folder (its `ckpts/` structure decoder)
    pub structure_decoder: PathBuf,
    /// DINOv3 ViT-L/16's folder
    pub dinov3: PathBuf,
}

/// The mesh as extracted and the texture decoder's attribute volume.
pub struct Generated {
    pub mesh: Mesh,
    /// The texture decoder's cells and their PBR attributes in [0, 1]
    pub voxels: Vec<[i32; 3]>,
    pub voxel_attrs: Vec<[f32; PBR]>,
    pub structure_cells: usize,
    pub latent_cells: usize,
    pub resolution: usize,
}

impl Generated {
    /// `mesh` (from [`vertex_mesh`]) with the volume's attributes at each vertex.
    pub fn colored(&self, mesh: Mesh) -> (Mesh, Vec<Pbr>) {
        let res = self.resolution as f32;
        let points: Vec<[f32; 3]> = mesh
            .vertices
            .iter()
            .map(|v| v.map(|c| (c + 0.5) * res))
            .collect();
        let attrs = sample_attributes(&self.voxels, &self.voxel_attrs, &points);
        (mesh, attrs)
    }
}

/// The extracted `mesh` as the vertex-colored export takes it: the export's cleanup ([`bake::clean`]), simplifying to
/// `faces` on `device`; it needs only the shape, so it can run beside the texture stage.
pub fn vertex_mesh(mesh: &Mesh, faces: usize, device: &Device) -> Result<Mesh> {
    Ok(bake::clean(mesh, faces, device)?.0)
}

struct Stats {
    shape: (Vec<f32>, Vec<f32>),
    texture: (Vec<f32>, Vec<f32>),
}

fn stats(dir: &Path) -> Result<Stats> {
    let pipeline: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join(PIPELINE_FILE))?)
            .map_err(inference_tensor::Error::wrap)?;
    let read = |group: &str, k: &str| -> Result<Vec<f32>> {
        pipeline["args"][group][k]
            .as_array()
            .ok_or_else(|| {
                inference_tensor::Error::Msg(format!("{PIPELINE_FILE} has no {group}.{k}"))
            })?
            .iter()
            .map(|v| {
                v.as_f64().map(|v| v as f32).ok_or_else(|| {
                    inference_tensor::Error::Msg(format!("{PIPELINE_FILE}: {group}.{k}"))
                })
            })
            .collect()
    };
    Ok(Stats {
        shape: (
            read("shape_slat_normalization", "mean")?,
            read("shape_slat_normalization", "std")?,
        ),
        texture: (
            read("tex_slat_normalization", "mean")?,
            read("tex_slat_normalization", "std")?,
        ),
    })
}

fn weights(path: &Path, device: &Device) -> Result<VarBuilder<'static>> {
    // SAFETY: the checkpoint files are not modified while the model is loaded
    unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device) }
}

fn noise(rng: &mut StdRng, cells: usize, channels: usize, device: &Device) -> Result<Tensor> {
    let v: Vec<f32> = (0..cells * channels)
        .map(|_| StandardNormal.sample(rng))
        .collect();
    Tensor::from_vec(v, (1, cells, channels), device)
}

// a flow stage: its progress name, checkpoint, widths and sampler
struct FlowStage {
    name: &'static str,
    file: &'static str,
    spec: ShapeFlowSpec,
    params: SamplerParams,
}

struct Run<'a> {
    paths: &'a Trellis2Paths,
    device: &'a Device,
    flow_dtype: DType,
    decoder_dtype: DType,
    progress: &'a dyn Progress,
}

impl Run<'_> {
    // one flow's sampling on `cells` under `cond`, the model loaded for it
    fn flow(
        &self,
        stage: FlowStage,
        cond: &Tensor,
        cells: &[[usize; 3]],
        x: &Tensor,
        concat: Option<&Tensor>,
    ) -> Result<Tensor> {
        let FlowStage {
            name,
            file,
            spec,
            params,
        } = stage;
        let flow = ShapeFlow::load(
            spec,
            self.flow_dtype,
            weights(&self.paths.trellis2.join(file), self.device)?,
        )?;
        let rope = flow.rope(cells)?;
        let kv = flow.condition(&Tensor::cat(&[cond, &cond.zeros_like()?], 0)?)?;
        self.progress.stage(name, params.steps);
        flow.sample(x, &rope, &kv, concat, &params, &mut |step| {
            self.progress.advance(step);
            check(self.progress)
        })
    }

    fn decoder(&self, file: &str, spec: super::decoder::DecoderSpec) -> Result<ShapeDecoder> {
        ShapeDecoder::load(
            spec,
            self.decoder_dtype,
            weights(&self.paths.trellis2.join(file), self.device)?,
        )
    }
}

/// DINOv3's tokens `[1, tokens, 1024]` for the cropped object `(rgb, size)` resized (PIL Lanczos) to `res`^2.
fn condition(
    encoder: &Dinov3,
    crop: &(Vec<u8>, (usize, usize)),
    res: usize,
    device: &Device,
) -> Result<Tensor> {
    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];
    let rgb = inference_vision::pil_resize(&crop.0, 3, crop.1, (res, res), PilFilter::Lanczos);
    let plane = res * res;
    let mut data = vec![0f32; 3 * plane];
    for (i, px) in rgb.as_chunks::<3>().0.iter().enumerate() {
        for c in 0..3 {
            data[c * plane + i] = (f32::from(px[c]) / 255.0 - MEAN[c]) / STD[c];
        }
    }
    encoder.forward(&Tensor::from_vec(data, (1, 3, res, res), device)?)
}

/// Sampling steps past which a flow's sampler is refused
pub const MAX_STEPS: usize = 100;

/// Changes to one flow's sampler; unset fields keep `pipeline.json`'s values.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SamplerChoice {
    pub steps: Option<usize>,
    /// Classifier-free guidance strength (1: none)
    pub guidance_strength: Option<f64>,
    /// How far the guided prediction is rescaled toward the conditioned one's spread, 0 to 1
    pub guidance_rescale: Option<f64>,
    /// The timestep schedule's shift (1: uniform)
    pub rescale_t: Option<f64>,
}

impl SamplerChoice {
    /// What is wrong with it, if anything
    pub fn problem(&self) -> Option<String> {
        if let Some(steps) = self.steps
            && !(1..=MAX_STEPS).contains(&steps)
        {
            return Some(format!("steps must be 1 to {MAX_STEPS}, got {steps}"));
        }
        if let Some(g) = self.guidance_strength
            && !(g.is_finite() && g >= 0.0)
        {
            return Some(format!("guidance strength must be 0 or more, got {g}"));
        }
        if let Some(r) = self.guidance_rescale
            && !(0.0..=1.0).contains(&r)
        {
            return Some(format!("guidance rescale must be 0 to 1, got {r}"));
        }
        if let Some(t) = self.rescale_t
            && !(t.is_finite() && t > 0.0)
        {
            return Some(format!("rescale_t must be above 0, got {t}"));
        }
        None
    }

    fn apply(&self, base: SamplerParams) -> SamplerParams {
        SamplerParams {
            steps: self.steps.unwrap_or(base.steps),
            guidance_strength: self.guidance_strength.unwrap_or(base.guidance_strength),
            guidance_rescale: self.guidance_rescale.unwrap_or(base.guidance_rescale),
            rescale_t: self.rescale_t.unwrap_or(base.rescale_t),
            ..base
        }
    }
}

/// What a run samples: the variant, the seed that draws every stage's noise, and each flow's sampler changes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Settings {
    pub variant: Variant,
    pub seed: u64,
    pub structure: SamplerChoice,
    pub shape: SamplerChoice,
    pub texture: SamplerChoice,
}

impl Settings {
    /// `variant` and `seed` with the samplers as `pipeline.json` has them
    pub fn new(variant: Variant, seed: u64) -> Self {
        Self {
            variant,
            seed,
            structure: SamplerChoice::default(),
            shape: SamplerChoice::default(),
            texture: SamplerChoice::default(),
        }
    }
}

/// The object in `image` (with transparency: its background already removed) as a mesh with PBR vertex attributes.
/// The flows run in bf16 and the decoders in f16 on a GPU, all in f32 on the CPU.
pub fn generate(
    paths: &Trellis2Paths,
    image: &Rgba,
    settings: &Settings,
    progress: &dyn Progress,
    device: &Device,
) -> Result<Generated> {
    generate_with(paths, image, settings, progress, device, &mut |_, _| {})
}

/// [`generate`], handing `shape_ready` the extracted mesh and its resolution before the texture stage, so work on the
/// shape alone can run beside it.
pub fn generate_with(
    paths: &Trellis2Paths,
    image: &Rgba,
    settings: &Settings,
    progress: &dyn Progress,
    device: &Device,
    shape_ready: &mut dyn FnMut(&Mesh, usize),
) -> Result<Generated> {
    let (flow_dtype, decoder_dtype) = if device.is_cpu() {
        (DType::F32, DType::F32)
    } else {
        (DType::BF16, DType::F16)
    };
    let run = Run {
        paths,
        device,
        flow_dtype,
        decoder_dtype,
        progress,
    };
    let stats = stats(&paths.trellis2)?;
    let variant = settings.variant;
    let mut rng = StdRng::seed_from_u64(settings.seed);

    progress.stage("condition", 0);
    let crop = crop_object(image)?;
    let (cond_low, cond_high) = {
        let encoder = Dinov3::load(
            Dinov3Spec::vitl16(DType::F32),
            weights(&paths.dinov3.join("model.safetensors"), device)?,
        )?;
        let low = condition(&encoder, &crop, LOW, device)?;
        let high = (variant != Variant::Res512)
            .then(|| condition(&encoder, &crop, HIGH, device))
            .transpose()?;
        (low, high)
    };

    // the structure: a dense 16^3 latent, decoded to 64^3 occupancy and pooled to the latents' grid
    let grid: Vec<[usize; 3]> = grid_cells(LATENT_SIDE)
        .iter()
        .map(|c| c.map(|v| v as usize))
        .collect();
    let x = noise(&mut rng, grid.len(), STRUCTURE_FLOW.out_channels, device)?;
    let z = run.flow(
        FlowStage {
            name: "structure",
            file: STRUCTURE_FLOW_FILE,
            spec: STRUCTURE_FLOW,
            params: settings.structure.apply(STRUCTURE_SAMPLER),
        },
        &cond_low,
        &grid,
        &x,
        None,
    )?;
    let logits = {
        let decoder = StructureDecoder::load(
            STRUCTURE_DECODER,
            LATENT_SIDE,
            decoder_dtype,
            weights(
                &paths.structure_decoder.join(STRUCTURE_DECODER_FILE),
                device,
            )?,
        )?;
        decoder.forward(&z.squeeze(0)?)?.to_vec1::<f32>()?
    };
    let full = 4 * LATENT_SIDE;
    let mut cells = occupied(&logits, full, variant.structure_side());
    if cells.is_empty() {
        inference_tensor::bail!("the structure came out empty");
    }
    let structure_cells = cells.len();
    check(progress)?;

    // the shape latents at 512 or 1024; a cascade upsamples the 512 ones' cells and samples again on them
    let shape_stage = |file| FlowStage {
        name: "shape",
        file,
        spec: SHAPE_FLOW_1024,
        params: settings.shape.apply(SHAPE_SAMPLER),
    };
    let (resolution, cond) = match (&cond_high, variant) {
        (Some(high), Variant::Res1024) => (HIGH, high),
        _ => (LOW, &cond_low),
    };
    let first = if resolution == HIGH {
        SHAPE_1024_FILE
    } else {
        SHAPE_512_FILE
    };
    let x = noise(&mut rng, cells.len(), SHAPE_FLOW_1024.out_channels, device)?;
    let mut slat = run.flow(shape_stage(first), cond, &cells, &x, None)?;
    slat = denormalize(&slat, &stats.shape.0, &stats.shape.1)?;
    let (resolution, cond) = match (&cond_high, variant.cascade()) {
        (Some(high), Some(target)) => {
            let fine = {
                let decoder = run.decoder(SHAPE_DECODER_FILE, SHAPE_DECODER)?;
                let coords: Vec<[i32; 3]> = cells.iter().map(|c| c.map(|v| v as i32)).collect();
                decoder.upsample(&coords, &slat.squeeze(0)?, UPSAMPLES)?
            };
            let resolution;
            (cells, resolution) = cascade_cells(&fine, target);
            let x = noise(&mut rng, cells.len(), SHAPE_FLOW_1024.out_channels, device)?;
            slat = run.flow(shape_stage(SHAPE_1024_FILE), high, &cells, &x, None)?;
            slat = denormalize(&slat, &stats.shape.0, &stats.shape.1)?;
            (resolution, high)
        }
        _ => (resolution, cond),
    };
    check(progress)?;

    // the shape decoded and its surface extracted, handed on before the texture is made; its subdivisions guide the
    // texture decoder onto the same cells
    progress.stage("decode", 0);
    let coords: Vec<[i32; 3]> = cells.iter().map(|c| c.map(|v| v as i32)).collect();
    let (shape_out, subs) = run.decoder(SHAPE_DECODER_FILE, SHAPE_DECODER)?.forward(
        &coords,
        &slat.squeeze(0)?,
        None,
        &mut |_, _| {},
    )?;
    let grid = dual_grid(&shape_out.feats)?;
    let mesh = dual_grid_mesh(
        &shape_out.coords,
        &grid.vertices,
        &grid.crossed,
        &grid.split,
        resolution,
    );
    drop((shape_out, grid));
    shape_ready(&mesh, resolution);
    check(progress)?;

    // the texture latents, the normalized shape latents beside the noise
    let shape_norm = normalize(&slat, &stats.shape.0, &stats.shape.1)?;
    let x = noise(&mut rng, cells.len(), TEXTURE_FLOW.out_channels, device)?;
    let texture_file = if resolution == LOW {
        TEXTURE_512_FILE
    } else {
        TEXTURE_1024_FILE
    };
    let tex = run.flow(
        FlowStage {
            name: "texture",
            file: texture_file,
            spec: TEXTURE_FLOW,
            params: settings.texture.apply(TEXTURE_SAMPLER),
        },
        cond,
        &cells,
        &x,
        Some(&shape_norm),
    )?;
    let tex = denormalize(&tex, &stats.texture.0, &stats.texture.1)?;

    let (tex_out, _) = run
        .decoder(TEXTURE_DECODER_FILE, TEXTURE_DECODER)?
        .forward(&coords, &tex.squeeze(0)?, Some(&subs), &mut |_, _| {})?;
    let values: Vec<[f32; PBR]> = ((tex_out.feats * 0.5)? + 0.5)?
        .to_vec2::<f32>()?
        .iter()
        .map(|r| std::array::from_fn(|k| r[k]))
        .collect();
    check(progress)?;
    Ok(Generated {
        mesh,
        voxels: tex_out.coords,
        voxel_attrs: values,
        structure_cells,
        latent_cells: cells.len(),
        resolution,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variants_parse_from_their_names_with_either_separator() {
        for v in Variant::ALL {
            assert_eq!(Variant::parse(v.name()), Some(v));
            assert_eq!(Variant::parse(&v.name().replace('_', "-")), Some(v));
        }
        assert_eq!(Variant::parse("2048"), None);
    }

    #[test]
    fn a_cascade_steps_its_resolution_down_until_its_cells_fit() {
        // a 192^3 block of 512 cells: 36^3 = 46656 cells at 1536, under the cap; every 4th along each axis reaches
        // every cell up to 1536 (16 / 3 apart)
        let block = |side: i32| -> Vec<[i32; 3]> {
            let axis: Vec<i32> = (0..side).step_by(4).chain([side - 1]).collect();
            let mut cells = Vec::new();
            for &x in &axis {
                for &y in &axis {
                    cells.extend(axis.iter().map(|&z| [x, y, z]));
                }
            }
            cells
        };
        let (cells, res) = cascade_cells(&block(192), HIGHEST);
        assert_eq!((cells.len(), res), (36 * 36 * 36, HIGHEST));
        // 224^3: 42^3 at 1536 and 39^3 at 1408 are over it, 35^3 at 1280 under
        let fine = block(224);
        let (cells, res) = cascade_cells(&fine, HIGHEST);
        assert_eq!((cells.len(), res), (35 * 35 * 35, 1280));
        // at 1024 the cap never applies
        let (cells, res) = cascade_cells(&fine, HIGH);
        assert_eq!((cells.len(), res), (28 * 28 * 28, HIGH));
    }
}
