//! TRELLIS.2 end to end (`Trellis2ImageTo3DPipeline.run`): DINOv3 conditions, the sparse structure (a dense flow and
//! TRELLIS's structure decoder), the shape latents (at 512, or 512 then 1024 on the decoder's upsampled cells), the
//! texture latents beside them, and their decoding into a mesh with PBR attributes at its vertices. Each model is
//! loaded for its stage and dropped after it.

use std::path::{Path, PathBuf};

use inference_tensor::nn::VarBuilder;
use inference_tensor::{DType, Device, Result, Tensor};
use inference_vision::PilFilter;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};

use super::cleanup;
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
// the latents' cells sit at the mesh resolution over 16
const CELL_SCALE: usize = 16;
const PBR: usize = 6;

/// Which resolutions the shape and texture run at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    /// Everything at 512 (a 32^3 structure)
    Res512,
    /// The reference's default: shape at 512, upsampled, again at 1024; texture at 1024
    Cascade1024,
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
    /// The mesh with small holes filled, simplified (on `device`) to TRELLIS.2's face budget, and the volume's
    /// attributes at each vertex.
    pub fn vertex_colored(&self, device: &Device) -> Result<(Mesh, Vec<Pbr>)> {
        let mut mesh = self.mesh.clone();
        cleanup::fill_holes(&mut mesh, cleanup::MAX_HOLE_PERIMETER);
        cleanup::simplify_on(&mut mesh, cleanup::SIMPLIFY_TARGET, device)?;
        let res = self.resolution as f32;
        let points: Vec<[f32; 3]> = mesh
            .vertices
            .iter()
            .map(|v| v.map(|c| (c + 0.5) * res))
            .collect();
        let attrs = sample_attributes(&self.voxels, &self.voxel_attrs, &points);
        Ok((mesh, attrs))
    }
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

/// The object in `image` (with transparency: its background already removed) as a mesh with PBR vertex attributes.
/// The flows run in bf16 and the decoders in f16 on a GPU, all in f32 on the CPU; `seed` draws every stage's noise.
pub fn generate(
    paths: &Trellis2Paths,
    image: &Rgba,
    variant: Variant,
    seed: u64,
    progress: &dyn Progress,
    device: &Device,
) -> Result<Generated> {
    generate_with(
        paths,
        image,
        variant,
        seed,
        progress,
        device,
        &mut |_, _| {},
    )
}

/// [`generate`], handing `shape_ready` the extracted mesh and its resolution before the texture stage, so work on the
/// shape alone can run beside it.
pub fn generate_with(
    paths: &Trellis2Paths,
    image: &Rgba,
    variant: Variant,
    seed: u64,
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
    let mut rng = StdRng::seed_from_u64(seed);

    progress.stage("condition", 0);
    let crop = crop_object(image)?;
    let (cond_low, cond_high) = {
        let encoder = Dinov3::load(
            Dinov3Spec::vitl16(DType::F32),
            weights(&paths.dinov3.join("model.safetensors"), device)?,
        )?;
        let low = condition(&encoder, &crop, LOW, device)?;
        let high = (variant == Variant::Cascade1024)
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
            params: STRUCTURE_SAMPLER,
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
    let mut cells = occupied(&logits, full, LOW / CELL_SCALE);
    if cells.is_empty() {
        inference_tensor::bail!("the structure came out empty");
    }
    let structure_cells = cells.len();
    check(progress)?;

    // the shape latents at 512; the cascade upsamples their cells and samples again at 1024
    let x = noise(&mut rng, cells.len(), SHAPE_FLOW_1024.out_channels, device)?;
    let mut slat = run.flow(
        FlowStage {
            name: "shape",
            file: SHAPE_512_FILE,
            spec: SHAPE_FLOW_1024,
            params: SHAPE_SAMPLER,
        },
        &cond_low,
        &cells,
        &x,
        None,
    )?;
    slat = denormalize(&slat, &stats.shape.0, &stats.shape.1)?;
    let (resolution, cond) = match (&cond_high, variant) {
        (Some(high), Variant::Cascade1024) => {
            let fine = {
                let decoder = run.decoder(SHAPE_DECODER_FILE, SHAPE_DECODER)?;
                let coords: Vec<[i32; 3]> = cells.iter().map(|c| c.map(|v| v as i32)).collect();
                decoder.upsample(&coords, &slat.squeeze(0)?, UPSAMPLES)?
            };
            let side = (HIGH / CELL_SCALE) as f32;
            let mut quantized: Vec<[usize; 3]> = fine
                .iter()
                .map(|c| c.map(|v| ((v as f32 + 0.5) / LOW as f32 * side) as usize))
                .collect();
            quantized.sort_unstable();
            quantized.dedup();
            cells = quantized;
            let x = noise(&mut rng, cells.len(), SHAPE_FLOW_1024.out_channels, device)?;
            slat = run.flow(
                FlowStage {
                    name: "shape",
                    file: SHAPE_1024_FILE,
                    spec: SHAPE_FLOW_1024,
                    params: SHAPE_SAMPLER,
                },
                high,
                &cells,
                &x,
                None,
            )?;
            slat = denormalize(&slat, &stats.shape.0, &stats.shape.1)?;
            (HIGH, high)
        }
        _ => (LOW, &cond_low),
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
    let texture_file = if resolution == HIGH {
        TEXTURE_1024_FILE
    } else {
        TEXTURE_512_FILE
    };
    let tex = run.flow(
        FlowStage {
            name: "texture",
            file: texture_file,
            spec: TEXTURE_FLOW,
            params: TEXTURE_SAMPLER,
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
