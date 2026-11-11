//! TRELLIS.2's second stage: the shape flow fills a voxel structure's cells with latents under the image, the decoder
//! turns them into a 1024^3 flexible dual grid, and the grid into a mesh, its small holes filled and its faces reduced.

pub mod bake;
pub mod charts;
pub mod cleanup;
pub mod decoder;
pub mod dit;
#[cfg(feature = "cuda")]
mod dit_cuda;
pub mod encoder;
pub mod glb;
pub mod mesh;
#[cfg(feature = "cuda")]
mod mesh_cuda;
#[cfg(test)]
mod parity;
pub mod pipeline;
pub mod remesh;
pub mod shape_flow;
pub mod sparse;
#[cfg(feature = "cuda")]
mod sparse_cuda;
pub mod structure;
pub mod vec3;
pub mod voxelize;

use std::path::Path;

use inference_tensor::nn::VarBuilder;
use inference_tensor::{DType, Device, Result, Tensor};
use inference_vision::PilFilter;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};

use crate::jovox::Rgba;
use crate::progress::{Progress, check};
use decoder::{SHAPE_DECODER, ShapeDecoder, dual_grid};
use mesh::{Mesh, dual_grid_mesh};
use shape_flow::{SHAPE_FLOW_1024, SHAPE_SAMPLER, ShapeFlow};

pub(crate) const PIPELINE_FILE: &str = "pipeline.json";
const FLOW_FILE: &str = "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors";
const DECODER_FILE: &str = "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors";
/// The files of TRELLIS.2-4B the second stage reads.
pub const FILES: [&str; 3] = [PIPELINE_FILE, FLOW_FILE, DECODER_FILE];
/// The condition's and the mesh grid's resolution
pub const RESOLUTION: usize = 1024;
// alpha above this is the object, as `preprocess_image` crops it
const FOREGROUND_ALPHA: u8 = 204;
const RGBA: usize = 4;

/// The image as TRELLIS.2's `preprocess_image` and `get_cond(.., 1024)` read it: [`crop_object`], Lanczos to 1024^2.
/// RGB bytes.
pub fn condition_image(image: &Rgba) -> Result<Vec<u8>> {
    let (rgb, size) = crop_object(image)?;
    Ok(inference_vision::pil_resize(
        &rgb,
        3,
        size,
        (RESOLUTION, RESOLUTION),
        PilFilter::Lanczos,
    ))
}

/// `preprocess_image` on an image with transparency: the square around the object (alpha above 204, no margin; parts
/// past the image transparent), color times alpha, as RGB bytes and its size.
pub fn crop_object(image: &Rgba) -> Result<(Vec<u8>, (usize, usize))> {
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
    for (i, px) in image.pixels.as_chunks::<RGBA>().0.iter().enumerate() {
        if px[3] > FOREGROUND_ALPHA {
            let (x, y) = (i % image.width, i / image.width);
            (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
        }
    }
    if x0 == usize::MAX {
        inference_tensor::bail!(
            "the image has no foreground (no pixel with alpha above {FOREGROUND_ALPHA})"
        );
    }
    let (cx, cy) = ((x0 + x1) as f64 / 2.0, (y0 + y1) as f64 / 2.0);
    let half = ((x1 - x0).max(y1 - y0) / 2) as f64;
    // PIL's crop rounds the box half to even
    let left = (cx - half).round_ties_even() as i64;
    let top = (cy - half).round_ties_even() as i64;
    let side = ((cx + half).round_ties_even() as i64 - left) as usize;
    let side_h = ((cy + half).round_ties_even() as i64 - top) as usize;
    let mut rgb = vec![0u8; side * side_h * 3];
    for y in 0..side_h {
        for x in 0..side {
            let (sx, sy) = (left + x as i64, top + y as i64);
            if sx < 0 || sy < 0 || sx >= image.width as i64 || sy >= image.height as i64 {
                continue;
            }
            let px = &image.pixels[(sy as usize * image.width + sx as usize) * RGBA..][..RGBA];
            let alpha = f32::from(px[3]) / 255.0;
            for c in 0..3 {
                rgb[(y * side + x) * 3 + c] = ((f32::from(px[c]) / 255.0 * alpha) * 255.0) as u8;
            }
        }
    }
    Ok((rgb, (side, side_h)))
}

/// The shape flow and decoder with the latent statistics they share.
pub struct ShapeStage {
    flow: ShapeFlow,
    decoder: ShapeDecoder,
    mean: Vec<f32>,
    std: Vec<f32>,
    device: Device,
}

impl ShapeStage {
    /// Loads them from a TRELLIS.2-4B folder (`pipeline.json` and the `ckpts/` files in [`FILES`]); the flow runs in
    /// bf16 and the decoder in f16 on a GPU, both in f32 on the CPU.
    pub fn load(dir: &Path, device: &Device) -> Result<Self> {
        let pipeline: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join(PIPELINE_FILE))?)
                .map_err(inference_tensor::Error::wrap)?;
        let stat = |k: &str| -> Result<Vec<f32>> {
            pipeline["args"]["shape_slat_normalization"][k]
                .as_array()
                .ok_or_else(|| {
                    inference_tensor::Error::Msg(format!("{PIPELINE_FILE} has no shape {k}"))
                })?
                .iter()
                .map(|v| {
                    v.as_f64().map(|v| v as f32).ok_or_else(|| {
                        inference_tensor::Error::Msg(format!("{PIPELINE_FILE}: a shape {k} entry"))
                    })
                })
                .collect()
        };
        let (flow_dtype, decoder_dtype) = if device.is_cpu() {
            (DType::F32, DType::F32)
        } else {
            (DType::BF16, DType::F16)
        };
        // SAFETY: the checkpoint files are not modified while the model is loaded
        let (flow_vb, decoder_vb) = unsafe {
            (
                VarBuilder::from_mmaped_safetensors(&[dir.join(FLOW_FILE)], DType::F32, device)?,
                VarBuilder::from_mmaped_safetensors(&[dir.join(DECODER_FILE)], DType::F32, device)?,
            )
        };
        Ok(Self {
            flow: ShapeFlow::load(SHAPE_FLOW_1024, flow_dtype, flow_vb)?,
            decoder: ShapeDecoder::load(SHAPE_DECODER, decoder_dtype, decoder_vb)?,
            mean: stat("mean")?,
            std: stat("std")?,
            device: device.clone(),
        })
    }

    /// The mesh of the structure `cells` (64^3) under `cond` (DINOv3's tokens for [`condition_image`],
    /// `[1, tokens, 1024]`), its noise drawn from `seed`: raw, before hole filling and simplification.
    pub fn mesh(
        &self,
        cond: &Tensor,
        cells: &[[usize; 3]],
        seed: u64,
        progress: &dyn Progress,
    ) -> Result<Mesh> {
        let latent = SHAPE_FLOW_1024.out_channels;
        let mut rng = StdRng::seed_from_u64(seed);
        let noise: Vec<f32> = (0..cells.len() * latent)
            .map(|_| StandardNormal.sample(&mut rng))
            .collect();
        let noise = Tensor::from_vec(noise, (1, cells.len(), latent), &self.device)?;
        let rope = self.flow.rope(cells)?;
        let kv = self
            .flow
            .condition(&Tensor::cat(&[cond, &cond.zeros_like()?], 0)?)?;
        progress.stage("shape", SHAPE_SAMPLER.steps);
        let x = self
            .flow
            .sample(&noise, &rope, &kv, None, &SHAPE_SAMPLER, &mut |step| {
                progress.advance(step);
                check(progress)
            })?;
        drop(kv);
        let latents = shape_flow::denormalize(&x, &self.mean, &self.std)?.squeeze(0)?;
        progress.stage("decode", 0);
        let coords: Vec<[i32; 3]> = cells.iter().map(|c| c.map(|v| v as i32)).collect();
        let (out, _) = self
            .decoder
            .forward(&coords, &latents, None, &mut |_, _| {})?;
        check(progress)?;
        let grid = dual_grid(&out.feats)?;
        Ok(dual_grid_mesh(
            &out.coords,
            &grid.vertices,
            &grid.crossed,
            &grid.split,
            RESOLUTION,
        ))
    }
}
