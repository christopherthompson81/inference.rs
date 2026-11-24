//! LATO.2 (LoHhhha/LATO.2, MIT): a mesh with artist-like topology from one image, in factorized flows. TRELLIS's
//! sparse-structure flow lays a 64^3 voxel scaffold under DINOv2 features of the image; V-Flow places latents on the
//! scaffold that the vertex VAE decodes to vertices on a 1024^3 grid, about as many as asked; T-Flow places a latent
//! on each vertex that the topology VAE decodes into edges, and the edges' triangles (small open rings filled) are the
//! faces.

pub mod faces;
#[cfg(test)]
mod parity;
pub mod structure;
pub mod topology;
pub mod transformer;
pub mod vertex_flow;
pub mod vertex_vae;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use inference_tensor::nn::{GroupNorm, VarBuilder};
use inference_tensor::pickle::PthTensors;
use inference_tensor::{DType, Device, Result, Tensor};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};

use crate::dinov2::{Dinov2, Dinov2Spec, PosInterp};
use crate::jovox::Rgba;
use crate::progress::{Progress, check};
use crate::trellis2::mesh::Mesh;
use crate::trellis2::structure::{STRUCTURE_DECODER, StructureDecoder};
use structure::{
    LATENT_CHANNELS, LATENT_SIDE, STRUCTURE_SAMPLER, StructureFlow, StructureSampler, y_up_voxels,
};
use topology::{TOPO_LATENT, TOPO_STEPS, TopologyDecoder, TopologyFlow, VoxelConditioner};
use vertex_flow::{VERTEX_SAMPLER, VertexFlow, VertexSampler};
use vertex_vae::{THRESHOLD, VertexDecoder};

/// The LATO.2 checkpoints the image pipeline reads (`0x4c48/LATO.2`).
pub const FILES: [&str; 6] = [
    "vflow.pt",
    "vvae.pt",
    "offset_head.pt",
    "tflow.pt",
    "tvae.pt",
    "voxel_encoder.pt",
];
/// TRELLIS-image-large's structure flow and decoder
pub const STRUCTURE_FLOW_FILE: &str = "ckpts/ss_flow_img_dit_L_16l8_fp16.safetensors";
pub const STRUCTURE_DECODER_FILE: &str = "ckpts/ss_dec_conv3d_16l8_fp16.safetensors";
/// The conditioning image's side, and DINOv2's
pub const IMAGE_SIZE: usize = 518;
/// The vertex counts V-Flow takes (the reference clamps a requested count to this range)
pub const VERTICES_RANGE: std::ops::RangeInclusive<usize> = 200..=5000;
pub const DEFAULT_VERTICES: usize = 2000;
// the crop's side over the object's bounding box
const CROP_PADDING: f64 = 1.2;
// alpha above 0.8 is the object
const FOREGROUND_ALPHA: f32 = 0.8 * 255.0;
const RGBA: usize = 4;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];
// the condition's affine-free LayerNorm (`F.layer_norm` default)
const COND_EPS: f64 = 1e-5;
const DINO_SPEC: Dinov2Spec = Dinov2Spec {
    dim: 1024,
    depth: 24,
    heads: 16,
    patch: 14,
    registers: 4,
    pos_interp: PosInterp::Bicubic { offset: 0.1 },
    dtype: DType::F32,
};
// T-Flow needs at least a triangle's worth of vertices
const MIN_TOPOLOGY_VERTICES: usize = 3;

/// AbsolutePositionEmbedder: each coordinate's `[sin(x w_i), cos(x w_i)]`, `w_i = 10000^(-i / f)` for `f = channels /
/// 3 / 2` frequencies, the axes side by side and zeros to `channels`: `[n, channels]` (f32).
pub fn ape(coords: &[[f32; 3]], channels: usize, device: &Device) -> Result<Tensor> {
    let freq = channels / 3 / 2;
    let freqs: Vec<f32> = (0..freq)
        .map(|i| 1.0 / 10000f32.powf(i as f32 / freq as f32))
        .collect();
    let mut out = vec![0f32; coords.len() * channels];
    for (row, c) in out.chunks_mut(channels).zip(coords) {
        for (a, &x) in c.iter().enumerate() {
            let at = &mut row[a * 2 * freq..(a + 1) * 2 * freq];
            for (i, &f) in freqs.iter().enumerate() {
                at[i] = (x * f).sin();
                at[freq + i] = (x * f).cos();
            }
        }
    }
    Tensor::from_vec(out, (coords.len(), channels), device)
}

/// A LATO.2 `.pt` checkpoint (`{"config", "state_dict"}`) or a flat torch state dict (`key` none) as a var builder.
pub fn pt_weights(path: &Path, key: Option<&str>, device: &Device) -> Result<VarBuilder<'static>> {
    let pth = PthTensors::new(path, key)?;
    let tensors = pth
        .tensor_infos()
        .keys()
        .map(|name| {
            let t = pth
                .get(name)?
                .ok_or_else(|| inference_tensor::Error::Msg(format!("{name} vanished")))?;
            Ok((name.clone(), t))
        })
        .collect::<Result<HashMap<_, _>>>()?;
    Ok(VarBuilder::from_tensors(tensors, DType::F32, device))
}

/// `nn.GroupNorm(groups, c, eps)` from its weight and bias.
fn load_group_norm(groups: usize, c: usize, eps: f64, vb: VarBuilder) -> Result<GroupNorm> {
    GroupNorm::new(vb.get(c, "weight")?, vb.get(c, "bias")?, c, groups, eps)
}

fn safetensors(path: &Path, device: &Device) -> Result<VarBuilder<'static>> {
    // SAFETY: the checkpoint files are not modified while the model is loaded
    unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device) }
}

/// `crop_foreground`: the square around the object (alpha above 0.8), its side 1.2 times the box's longer side, PIL
/// Lanczos to 518^2 (premultiplied, parts past the image transparent), then color times alpha on black. RGB bytes.
pub fn condition_image(image: &Rgba) -> Result<Vec<u8>> {
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
    for (i, px) in image.pixels.as_chunks::<RGBA>().0.iter().enumerate() {
        if f32::from(px[3]) > FOREGROUND_ALPHA {
            let (x, y) = (i % image.width, i / image.width);
            (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
        }
    }
    if x0 == usize::MAX {
        inference_tensor::bail!("the image has no foreground (no pixel with alpha above 0.8)");
    }
    let (cx, cy) = ((x0 + x1) as f64 / 2.0, (y0 + y1) as f64 / 2.0);
    let side = ((x1 - x0).max(y1 - y0) as f64 * CROP_PADDING) as i64;
    if side < 2 {
        inference_tensor::bail!("the foreground is too small to crop");
    }
    // PIL's crop rounds its float box half to even
    let half = (side / 2) as f64;
    let left = (cx - half).round_ties_even() as i64;
    let top = (cy - half).round_ties_even() as i64;
    let (w, h) = (
        ((cx + half).round_ties_even() as i64 - left) as usize,
        ((cy + half).round_ties_even() as i64 - top) as usize,
    );
    let mut crop = vec![0u8; w * h * RGBA];
    for y in 0..h {
        for x in 0..w {
            let (sx, sy) = (left + x as i64, top + y as i64);
            if sx < 0 || sy < 0 || sx >= image.width as i64 || sy >= image.height as i64 {
                continue;
            }
            let src = (sy as usize * image.width + sx as usize) * RGBA;
            crop[(y * w + x) * RGBA..][..RGBA].copy_from_slice(&image.pixels[src..src + RGBA]);
        }
    }
    let resized =
        inference_vision::pil_resize_rgba_lanczos(&crop, (w, h), (IMAGE_SIZE, IMAGE_SIZE));
    Ok(resized
        .as_chunks::<RGBA>()
        .0
        .iter()
        .flat_map(|px| {
            let alpha = f32::from(px[3]) / 255.0;
            [0, 1, 2].map(|c| ((f32::from(px[c]) / 255.0 * alpha) * 255.0) as u8)
        })
        .collect())
}

/// The conditioning image's DINOv2 tokens as LATO.2 and TRELLIS read them: every token before the final norm, then
/// an affine-free LayerNorm, `[1, 1374, 1024]` (f32).
pub(crate) fn condition_tokens(dino: &Dinov2, rgb: &[u8], device: &Device) -> Result<Tensor> {
    let plane = IMAGE_SIZE * IMAGE_SIZE;
    let mut data = vec![0f32; 3 * plane];
    for (i, px) in rgb.as_chunks::<3>().0.iter().enumerate() {
        for c in 0..3 {
            data[c * plane + i] = (f32::from(px[c]) / 255.0 - MEAN[c]) / STD[c];
        }
    }
    let images = Tensor::from_vec(data, (1, 3, IMAGE_SIZE, IMAGE_SIZE), device)?;
    transformer::plain_norm(&dino.forward_prenorm(&images)?, COND_EPS)
}

/// Loads DINOv2 ViT-L/14 with registers from `facebook/dinov2-with-registers-large`'s `model.safetensors` (the torch
/// hub checkpoint's weights under transformers' names), renamed to the hub's layout.
pub(crate) fn load_dinov2(path: &Path, device: &Device) -> Result<Dinov2> {
    let hf = inference_tensor::safetensors::load(path, &Device::Cpu)?;
    let take = |k: &str| {
        hf.get(k)
            .cloned()
            .ok_or_else(|| inference_tensor::Error::Msg(format!("{}: no {k}", path.display())))
    };
    let mut hub = HashMap::new();
    for (to, from) in [
        ("cls_token", "embeddings.cls_token"),
        ("register_tokens", "embeddings.register_tokens"),
        ("pos_embed", "embeddings.position_embeddings"),
        (
            "patch_embed.proj.weight",
            "embeddings.patch_embeddings.projection.weight",
        ),
        (
            "patch_embed.proj.bias",
            "embeddings.patch_embeddings.projection.bias",
        ),
        ("norm.weight", "layernorm.weight"),
        ("norm.bias", "layernorm.bias"),
    ] {
        hub.insert(to.to_string(), take(from)?);
    }
    for i in 0..DINO_SPEC.depth {
        let (h, f) = (format!("blocks.{i}."), format!("encoder.layer.{i}."));
        for part in ["weight", "bias"] {
            let qkv = ["query", "key", "value"]
                .map(|n| take(&format!("{f}attention.attention.{n}.{part}")))
                .into_iter()
                .collect::<Result<Vec<_>>>()?;
            hub.insert(format!("{h}attn.qkv.{part}"), Tensor::cat(&qkv, 0)?);
            for (to, from) in [
                ("attn.proj", "attention.output.dense"),
                ("norm1", "norm1"),
                ("norm2", "norm2"),
                ("mlp.fc1", "mlp.fc1"),
                ("mlp.fc2", "mlp.fc2"),
            ] {
                hub.insert(
                    format!("{h}{to}.{part}"),
                    take(&format!("{f}{from}.{part}"))?,
                );
            }
        }
        hub.insert(
            format!("{h}ls1.gamma"),
            take(&format!("{f}layer_scale1.lambda1"))?,
        );
        hub.insert(
            format!("{h}ls2.gamma"),
            take(&format!("{f}layer_scale2.lambda1"))?,
        );
    }
    Dinov2::load(DINO_SPEC, VarBuilder::from_tensors(hub, DType::F32, device))
}

/// Where the weights are.
#[derive(Clone, Debug)]
pub struct Lato2Paths {
    /// LATO.2's folder (the `.pt` files in [`FILES`])
    pub lato2: PathBuf,
    /// TRELLIS-image-large's folder (its `ckpts/` structure flow and decoder)
    pub trellis_large: PathBuf,
    /// `facebook/dinov2-with-registers-large`'s `model.safetensors`
    pub dinov2: PathBuf,
}

/// The sampling settings.
#[derive(Clone, Copy, Debug)]
pub struct Lato2Options {
    pub vertices: usize,
    pub seed: u64,
    pub structure: StructureSampler,
    pub vertex: VertexSampler,
    pub topology_steps: usize,
    /// Open rings up to this many edges are filled (below 4: none)
    pub max_ring: usize,
    /// Boundary loops (holes and cracks) of up to this many edges left after the ring filling are closed, adding no
    /// vertices (below 3: none). Not in the reference, whose meshes keep them
    pub close_loops: usize,
}

impl Default for Lato2Options {
    fn default() -> Self {
        Self {
            vertices: DEFAULT_VERTICES,
            seed: 0,
            structure: STRUCTURE_SAMPLER,
            vertex: VERTEX_SAMPLER,
            topology_steps: TOPO_STEPS,
            max_ring: faces::MAX_RING,
            close_loops: faces::MAX_LOOP,
        }
    }
}

/// A generated mesh (y up, in [-0.5, 0.5]) and what led to it.
pub struct Generated {
    pub mesh: Mesh,
    /// The conditioning image (518^2 RGB)
    pub condition: Vec<u8>,
    /// The scaffold's 64^3 cells
    pub scaffold: Vec<[i32; 3]>,
    pub edges: usize,
}

fn noise(rng: &mut StdRng, rows: usize, channels: usize, device: &Device) -> Result<Tensor> {
    let v: Vec<f32> = (0..rows * channels)
        .map(|_| StandardNormal.sample(rng))
        .collect();
    Tensor::from_vec(v, (1, rows, channels), device)
}

// the stages' dtypes: the reference's autocast and fp16 blocks on a GPU, f32 on the CPU
fn half(device: &Device, dtype: DType) -> DType {
    if device.is_cpu() { DType::F32 } else { dtype }
}

/// The mesh for `image`, an object whose background is transparent (matted). Each model is loaded for its stage and
/// dropped after it.
pub fn generate(
    paths: &Lato2Paths,
    image: &Rgba,
    options: &Lato2Options,
    progress: &dyn Progress,
    device: &Device,
) -> Result<Generated> {
    if !VERTICES_RANGE.contains(&options.vertices) {
        inference_tensor::bail!(
            "vertices must be in {}..={}",
            VERTICES_RANGE.start(),
            VERTICES_RANGE.end()
        );
    }
    let mut rng = StdRng::seed_from_u64(options.seed);
    let lato = |f: &str| pt_weights(&paths.lato2.join(f), Some("state_dict"), device);
    let condition = condition_image(image)?;

    progress.stage("condition", 0);
    let cond = {
        let dino = load_dinov2(&paths.dinov2, device)?;
        condition_tokens(&dino, &condition, device)?
    };
    let both = Tensor::cat(&[&cond, &cond.zeros_like()?], 0)?;
    check(progress)?;

    let scaffold = {
        let flow = StructureFlow::load(
            half(device, DType::F16),
            safetensors(&paths.trellis_large.join(STRUCTURE_FLOW_FILE), device)?,
        )?;
        let kv = flow.condition(&both)?;
        let cells = LATENT_SIDE * LATENT_SIDE * LATENT_SIDE;
        let x = noise(&mut rng, cells, LATENT_CHANNELS, device)?;
        progress.stage("scaffold", options.structure.steps);
        let latent = flow.sample(&x, &kv, &options.structure, &mut |i| {
            progress.advance(i);
            check(progress)
        })?;
        drop((flow, kv));
        let decoder = StructureDecoder::load(
            STRUCTURE_DECODER,
            LATENT_SIDE,
            half(device, DType::F16),
            safetensors(&paths.trellis_large.join(STRUCTURE_DECODER_FILE), device)?,
        )?;
        let logits = decoder.forward(&latent.squeeze(0)?)?.to_vec1::<f32>()?;
        y_up_voxels(&logits, 4 * LATENT_SIDE)
    };
    if scaffold.is_empty() {
        inference_tensor::bail!("the scaffold came out empty");
    }
    check(progress)?;

    let vertices = {
        let flow = VertexFlow::load(half(device, DType::BF16), lato("vflow.pt")?)?;
        let kv = flow.condition(&both)?;
        let rope = flow.rope(&scaffold)?;
        let x = noise(
            &mut rng,
            scaffold.len(),
            vertex_flow::LATENT_CHANNELS,
            device,
        )?;
        progress.stage("vertices", options.vertex.steps);
        let latent = flow.sample(
            &x,
            options.vertices,
            &rope,
            &kv,
            &options.vertex,
            &mut |i| {
                progress.advance(i);
                check(progress)
            },
        )?;
        drop((flow, kv, rope));
        progress.stage("vertex decoding", 0);
        let decoder = VertexDecoder::load(
            half(device, DType::BF16),
            lato("vvae.pt")?,
            lato("offset_head.pt")?,
        )?;
        decoder.decode(&latent.squeeze(0)?, &scaffold, THRESHOLD, &mut |_, _, _| {})?
    };
    check(progress)?;
    let positions = vertices.positions();
    let n = vertices.cells.len();
    if !(MIN_TOPOLOGY_VERTICES..=topology::MAX_VERTICES).contains(&n) {
        inference_tensor::bail!(
            "{n} vertices came out; topology takes 3 to {}",
            topology::MAX_VERTICES
        );
    }

    let edges = {
        let voxel_cond = VoxelConditioner::load(lato("voxel_encoder.pt")?)?.forward(&scaffold)?;
        let flow = TopologyFlow::load(DType::F32, lato("tflow.pt")?)?;
        let x = noise(&mut rng, n, TOPO_LATENT, device)?;
        progress.stage("topology", options.topology_steps);
        let z = flow.sample(
            &x,
            &vertices.cells,
            &voxel_cond,
            options.topology_steps,
            &mut |i| {
                progress.advance(i);
                check(progress)
            },
        )?;
        drop(flow);
        let decoder = TopologyDecoder::load(half(device, DType::BF16), lato("tvae.pt")?)?;
        let feats = decoder.features(&z, &vertices.cells)?;
        decoder.edges(&feats, topology::EDGE_THRESHOLD)?
    };
    let cells: Vec<[f32; 3]> = vertices.cells.iter().map(|c| c.map(|v| v as f32)).collect();
    let mut mesh = Mesh {
        vertices: positions,
        faces: faces::faces(n, &edges, &cells, options.max_ring),
    };
    if options.close_loops >= 3 {
        let closing = faces::close_loops(&mesh.faces, &cells, options.close_loops);
        mesh.faces.extend(closing);
    }
    faces::orient(&mut mesh.faces, &mesh.vertices);
    Ok(Generated {
        mesh,
        condition,
        scaffold,
        edges: edges.len(),
    })
}
