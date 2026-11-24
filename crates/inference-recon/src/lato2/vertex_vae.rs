//! The vertex VAE's decoder (`VertexVAE.decode`) and the offset head: the flow's 64^3 latents are expanded by a
//! windowed transformer, the cells that hold a vertex kept, then four levels each subdivide the kept cells, convolve,
//! keep the children their pruning head scores (at least one a parent) and attend back to the latents, down to one
//! 1024^3 cell a vertex; the offset head places each vertex within its cell.

use inference_tensor::nn::{GroupNorm, Module, VarBuilder};
use inference_tensor::{DType, Device, Result, Tensor};

use super::transformer::{AttnMode, Positions, Stack, StackSpec, plain_norm};
use crate::trellis2::sparse::{Neighbors, SparseConv};
use crate::worldmirror::layers::AmpLinear;

const LATENT: usize = 32;
const EXPANDED: usize = 512;
const CONTEXT: usize = 512;
const HEADS: usize = 8;
const WINDOW: i32 = 8;
const GROUPS: usize = 32;
const GROUP_EPS: f64 = 1e-5;
// the decoder block's F.layer_norm default
const LEVEL_EPS: f64 = 1e-5;
const CHILDREN: usize = 8;
// the shell-to-vertex step keeps at least this many cells when none passes
const MIN_VERTICES: usize = 2;
/// The pruning heads' probability threshold (`inference_threshold`)
pub const THRESHOLD: f32 = 0.5;
/// The grid the decoded vertices sit on
pub const RESOLUTION: usize = 1024;

// (in, out, model) channels of each level's upsample and its cross stack
const LEVELS: [(usize, usize, usize); 4] = [
    (1024, 768, 768),
    (768, 512, 512),
    (512, 512, 512),
    (512, 512, 512),
];
// SparseDecoderUpsampleBlock's own cross transformer width
const UPSAMPLE_MODEL: usize = 512;
const LEVEL_BLOCKS: usize = 4;

/// A pruning head: Linear, tanh-GELU, Linear to one logit.
struct OccHead {
    fc1: AmpLinear,
    fc2: AmpLinear,
}

impl OccHead {
    fn load(c: usize, dtype: DType, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("mlp");
        Ok(Self {
            fc1: AmpLinear::load(c, 4 * c, true, dtype, vb.pp(0))?,
            fc2: AmpLinear::load(4 * c, 1, true, dtype, vb.pp(2))?,
        })
    }

    /// Each row's probability, on the host.
    fn probs(&self, xs: &Tensor) -> Result<Vec<f32>> {
        let logits = self.fc2.forward(&self.fc1.forward(xs)?.gelu()?)?;
        inference_tensor::nn::ops::sigmoid(&logits.to_dtype(DType::F32)?)?
            .flatten_all()?
            .to_vec1()
    }
}

/// `SparseGroupNorm32` over one sample's cells `[n, c]`, in f32.
fn group_norm(norm: &GroupNorm, xs: &Tensor) -> Result<Tensor> {
    let (n, c) = xs.dims2()?;
    norm.forward(&xs.to_dtype(DType::F32)?.t()?.reshape((1, c, n))?)?
        .reshape((c, n))?
        .t()?
        .contiguous()
}

/// `SparseSubdivide`: each cell's eight children, parent-major, child `k` at offset `(k >> 2, k >> 1 & 1, k & 1)`.
fn subdivide(coords: &[[i32; 3]]) -> Vec<[i32; 3]> {
    coords
        .iter()
        .flat_map(|&[x, y, z]| {
            (0..CHILDREN as i32)
                .map(move |k| [2 * x + (k >> 2), 2 * y + (k >> 1 & 1), 2 * z + (k & 1)])
        })
        .collect()
}

/// `SparseDecoderUpsampleBlock` and the cross stack after it.
struct Level {
    norm_in: GroupNorm,
    conv1: SparseConv,
    norm_mid: GroupNorm,
    conv2: SparseConv,
    skip: Option<AmpLinear>,
    proj_ctx: AmpLinear,
    ca: Stack,
    proj_out: AmpLinear,
    prune: OccHead,
    latent_proj: AmpLinear,
    cross: Stack,
}

impl Level {
    fn load(
        (cin, cout, model): (usize, usize, usize),
        dtype: DType,
        vb: VarBuilder,
        ca_vb: VarBuilder,
        proj_vb: VarBuilder,
    ) -> Result<Self> {
        let up = vb.pp("upsample");
        let gn = |c: usize, vb: VarBuilder| super::load_group_norm(GROUPS, c, GROUP_EPS, vb);
        let layers = up.pp("out_layers");
        Ok(Self {
            norm_in: gn(cin, up.pp("act_layers").pp(0))?,
            conv1: SparseConv::load(cin, cout, DType::F32, layers.pp(0).pp("conv"))?,
            norm_mid: gn(cout, layers.pp(1))?,
            conv2: SparseConv::load(cout, cout, DType::F32, layers.pp(3).pp("conv"))?,
            skip: (cin != cout)
                .then(|| {
                    let vb = up.pp("skip_connection").pp("conv");
                    // spconv's 1x1 path multiplies by `weight.view(in, out)`, whatever the declared layout says
                    let w = vb
                        .get((cout, 1, 1, 1, cin), "weight")?
                        .reshape((cin, cout))?
                        .t()?
                        .contiguous()?;
                    let b = vb.get(cout, "bias")?;
                    AmpLinear::from_parts(w, Some(b), DType::F32)
                })
                .transpose()?,
            proj_ctx: AmpLinear::load(cout, UPSAMPLE_MODEL, true, dtype, up.pp("proj_ctx"))?,
            ca: Stack::load(
                StackSpec {
                    input: cout,
                    channels: UPSAMPLE_MODEL,
                    context: Some(UPSAMPLE_MODEL),
                    blocks: LEVEL_BLOCKS,
                    heads: HEADS,
                    positions: Positions::Absolute,
                    attn: AttnMode::Full,
                },
                dtype,
                up.pp("ca"),
            )?,
            proj_out: AmpLinear::load(UPSAMPLE_MODEL, cout, true, dtype, up.pp("proj_out"))?,
            prune: OccHead::load(cout, dtype, up.pp("pruning_head"))?,
            latent_proj: AmpLinear::load(LATENT, CONTEXT, true, dtype, proj_vb)?,
            cross: Stack::load(
                StackSpec {
                    input: cout,
                    channels: model,
                    context: Some(CONTEXT),
                    blocks: LEVEL_BLOCKS,
                    heads: HEADS,
                    positions: Positions::Rope,
                    attn: AttnMode::Full,
                },
                dtype,
                ca_vb,
            )?,
        })
    }

    /// The kept children's coordinates and features `[kept, model]` (f32) of cells `x` `[n, in]` at `coords`, the
    /// latents `latent` `[m, 32]` at `latent_coords` their context.
    fn forward(
        &self,
        x: &Tensor,
        coords: &[[i32; 3]],
        latent: &Tensor,
        latent_coords: &[[i32; 3]],
        threshold: f32,
        device: &Device,
    ) -> Result<(Vec<[i32; 3]>, Tensor)> {
        let x = plain_norm(x, LEVEL_EPS)?;
        let n = coords.len();
        let children = subdivide(coords);
        let parent_rows: Vec<u32> = (0..n as u32).flat_map(|p| [p; CHILDREN]).collect();
        let parent_rows = Tensor::from_vec(parent_rows, n * CHILDREN, device)?;
        let h = group_norm(&self.norm_in, &x)?
            .silu()?
            .index_select(&parent_rows, 0)?;
        let x_sub = x.index_select(&parent_rows, 0)?;
        let neighbors = Neighbors::new(&children, device)?;
        let h = self.conv1.forward(&h, &neighbors)?;
        let h = group_norm(&self.norm_mid, &h)?.silu()?;
        let h = self.conv2.forward(&h, &neighbors)?;
        let skip = match &self.skip {
            Some(skip) => skip.forward(&x_sub)?,
            None => x_sub,
        };
        let h = (h + skip)?;
        let ctx = self.proj_ctx.forward(&h)?;
        let h = self
            .proj_out
            .forward(&self.ca.forward(&h, &children, Some((&ctx, &children)))?)?
            .to_dtype(DType::F32)?;
        let probs = self.prune.probs(&h)?;
        let mut keep = Vec::new();
        for (p, group) in probs.chunks(CHILDREN).enumerate() {
            let before = keep.len();
            keep.extend(
                group
                    .iter()
                    .enumerate()
                    .filter(|&(_, &s)| s >= threshold)
                    .map(|(k, _)| (p * CHILDREN + k) as u32),
            );
            if keep.len() == before {
                // torch.topk(k=1) takes the first of equal maxima
                let best = group
                    .iter()
                    .enumerate()
                    .fold(0, |b, (k, &s)| if s > group[b] { k } else { b });
                keep.push((p * CHILDREN + best) as u32);
            }
        }
        let kept: Vec<[i32; 3]> = keep.iter().map(|&r| children[r as usize]).collect();
        let rows = Tensor::from_vec(keep, kept.len(), device)?;
        let h = h.index_select(&rows, 0)?;
        let context = self.latent_proj.forward(latent)?;
        let h = self
            .cross
            .forward(&h, &kept, Some((&context, latent_coords)))?;
        Ok((kept, h))
    }
}

pub struct VertexDecoder {
    expander: Stack,
    vtx_prune: OccHead,
    vtx_proj: AmpLinear,
    levels: Vec<Level>,
    offset: (AmpLinear, AmpLinear),
    device: Device,
}

/// Sees each decoder level's index, kept cells and features.
pub type LevelInspect<'a> = dyn FnMut(usize, &[[i32; 3]], &Tensor) + 'a;

/// The decoded vertices: their 1024^3 cells and offsets within them in (-1, 1) (half a cell).
pub struct Vertices {
    pub cells: Vec<[i32; 3]>,
    pub offsets: Vec<[f32; 3]>,
}

impl Vertices {
    /// Positions in [-0.5, 0.5]: the cell's corner plus half a cell per unit of offset (`export_vertex`).
    pub fn positions(&self) -> Vec<[f32; 3]> {
        let res = RESOLUTION as f64;
        self.cells
            .iter()
            .zip(&self.offsets)
            .map(|(c, o)| {
                std::array::from_fn(|i| {
                    (f64::from(c[i]) / res - 0.5 + f64::from(o[i]) / (2.0 * res)) as f32
                })
            })
            .collect()
    }
}

impl VertexDecoder {
    /// From `vvae.pt`'s and `offset_head.pt`'s state dicts; linears and attention run in `dtype` (bf16, as the
    /// reference's autocast), convolutions and norms in f32.
    pub fn load(dtype: DType, vvae: VarBuilder, offset_head: VarBuilder) -> Result<Self> {
        let device = vvae.device().clone();
        let levels = LEVELS
            .iter()
            .enumerate()
            .map(|(i, &dims)| {
                Level::load(
                    dims,
                    dtype,
                    vvae.pp("decoder_vtx").pp(i),
                    vvae.pp("decoder_vtx_ca").pp(i),
                    vvae.pp("latent_proj").pp(i),
                )
            })
            .collect::<Result<_>>()?;
        let head = offset_head.pp("mlp");
        let last = LEVELS[LEVELS.len() - 1].2;
        Ok(Self {
            expander: Stack::load(
                StackSpec {
                    input: LATENT,
                    channels: EXPANDED,
                    context: None,
                    blocks: 8,
                    heads: HEADS,
                    positions: Positions::Rope,
                    attn: AttnMode::Swin { window: WINDOW },
                },
                dtype,
                vvae.pp("latent_expander"),
            )?,
            vtx_prune: OccHead::load(EXPANDED, dtype, vvae.pp("vtx_pruning_head"))?,
            vtx_proj: AmpLinear::load(EXPANDED, LEVELS[0].0, true, dtype, vvae.pp("vtx_proj"))?,
            levels,
            offset: (
                AmpLinear::load(last, 4 * last, true, dtype, head.pp(0))?,
                AmpLinear::load(4 * last, 3, true, dtype, head.pp(2))?,
            ),
            device,
        })
    }

    /// The vertices of latents `latent` `[cells, 32]` (f32) on the scaffold's `cells`; `inspect` sees each level's
    /// kept cells and features.
    pub fn decode(
        &self,
        latent: &Tensor,
        cells: &[[i32; 3]],
        threshold: f32,
        inspect: &mut LevelInspect,
    ) -> Result<Vertices> {
        let expanded = self.expander.forward(latent, cells, None)?;
        let probs = self.vtx_prune.probs(&expanded)?;
        let mut keep: Vec<u32> = (0..probs.len() as u32)
            .filter(|&i| probs[i as usize] >= threshold)
            .collect();
        if keep.is_empty() {
            let mut order: Vec<u32> = (0..probs.len() as u32).collect();
            order.sort_by(|&a, &b| probs[b as usize].total_cmp(&probs[a as usize]));
            keep = order.into_iter().take(MIN_VERTICES).collect();
            keep.sort_unstable();
        }
        let mut coords: Vec<[i32; 3]> = keep.iter().map(|&i| cells[i as usize]).collect();
        inspect(0, &coords, &expanded);
        let rows = Tensor::from_vec(keep, coords.len(), &self.device)?;
        let mut h = self
            .vtx_proj
            .forward(&expanded.index_select(&rows, 0)?)?
            .to_dtype(DType::F32)?;
        for (i, level) in self.levels.iter().enumerate() {
            (coords, h) = level.forward(&h, &coords, latent, cells, threshold, &self.device)?;
            inspect(i + 1, &coords, &h);
        }
        let offsets = self
            .offset
            .1
            .forward(&self.offset.0.forward(&h)?.gelu()?)?
            .tanh()?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?;
        Ok(Vertices {
            cells: coords,
            offsets: offsets.into_iter().map(|o| [o[0], o[1], o[2]]).collect(),
        })
    }
}
