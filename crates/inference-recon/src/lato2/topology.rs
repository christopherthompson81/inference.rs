//! The topology stage: the scaffold's 64^3 occupancy encoded to 8^3 condition tokens (`VoxelFieldConditioner`),
//! T-Flow (`TopologySiTFlow`, a dense DiT over the vertices: RoPE on their 1024^3 cells, cross-attention to the
//! scaffold tokens) denoising 16-channel per-vertex latents, and the topology VAE's decoder: a transformer over the
//! latents, then edge logits for every vertex pair from a symmetrized pair MLP.

use inference_tensor::nn::{GroupNorm, Linear, Module, VarBuilder};
use inference_tensor::{D, DType, Device, Result, Tensor};

use super::transformer::plain_norm;
use crate::trellis2::dit::{Rope, TimestepMlp, sdpa};
use crate::trellis2::sparse::{Neighbors, SparseConv};
use crate::trellis2::structure::grid_cells;
use crate::worldmirror::layers::AmpLinear;

const SINCOS_BASE: f64 = 10000.0;
// the blocks' LayerNorm(elementwise_affine=False, eps=1e-6)
const NORM_EPS: f64 = 1e-6;
const MLP_RATIO: usize = 4;

/// The voxel conditioner's input grid and output tokens' width
pub const VOXEL_GRID: usize = 64;
const COND_CHANNELS: usize = 256;
const BASE_CHANNELS: usize = 32;
const DOWNSAMPLES: usize = 3;
const BLOCKS_PER_LEVEL: usize = 3;
const COND_GROUPS: usize = 8;
const GROUP_EPS: f64 = 1e-5;

/// T-Flow's latent width, the scale its samples are divided by, and the grid its vertices sit on
pub const TOPO_LATENT: usize = 16;
pub const LATENT_SCALE: f64 = 0.25;
pub const NUM_DISCRETE: usize = 1024;
/// T-Flow's longest vertex sequence (its index embedding's length)
pub const MAX_VERTICES: usize = 10000;
const FLOW_WIDTH: usize = 768;
const FLOW_DEPTH: usize = 16;
const FLOW_HEADS: usize = 12;
const FLOW_RMS_EPS: f64 = 1e-5;
/// T-Flow's Euler steps (t from 0 to 1, no guidance)
pub const TOPO_STEPS: usize = 50;
// adaLN chunks with the cross gate: shift, scale, gate (attention), cross gate, shift, scale, gate (MLP)
const ADALN_CHUNKS: usize = 7;

const VAE_WIDTH: usize = 768;
const VAE_HEADS: usize = 8;
const VAE_STAGES: usize = 2;
const VAE_PLAIN_PER_STAGE: usize = 3;
const PAIR_HIDDEN: usize = 256;
/// Rows of the pair logits computed at once
const PAIR_ROWS: usize = 64;
/// The edge logit threshold
pub const EDGE_THRESHOLD: f32 = 0.0;

// [sin(p * w_i), cos(p * w_i)] for w_i = 10000^(-i / half), positions 0..n: `[n, dim]`
fn sincos_1d(n: usize, dim: usize) -> Vec<Vec<f32>> {
    let half = dim / 2;
    let omega: Vec<f32> = (0..half)
        .map(|i| (1.0 / SINCOS_BASE.powf(i as f64 / half as f64)) as f32)
        .collect();
    (0..n)
        .map(|p| {
            let a: Vec<f32> = omega.iter().map(|w| p as f32 * w).collect();
            a.iter()
                .map(|v| v.sin())
                .chain(a.iter().map(|v| v.cos()))
                .collect()
        })
        .collect()
}

// `_safe_group_norm`: the most groups up to 8 that divide the channels
fn safe_groups(c: usize) -> usize {
    (1..=COND_GROUPS.min(c))
        .rev()
        .find(|&g| c.is_multiple_of(g))
        .unwrap_or(1)
}

fn group_norm(norm: &GroupNorm, xs: &Tensor) -> Result<Tensor> {
    let (n, c) = xs.dims2()?;
    norm.forward(&xs.t()?.reshape((1, c, n))?)?
        .reshape((c, n))?
        .t()?
        .contiguous()
}

/// A dense `nn.Conv3d` (kernel 3, padding 1, stride 1 or 2) on a full grid's cells `[side^3, c]` (x-major).
struct Conv3 {
    conv: SparseConv,
    stride: usize,
}

impl Conv3 {
    fn load(cin: usize, cout: usize, stride: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            conv: SparseConv::load_dense_unbiased(cin, cout, DType::F32, vb)?,
            stride,
        })
    }

    fn forward(&self, xs: &Tensor, side: usize, neighbors: &Neighbors) -> Result<Tensor> {
        let out = self.conv.forward(xs, neighbors)?;
        if self.stride == 1 {
            return Ok(out);
        }
        // a stride-2 conv is the stride-1 one read at the even cells
        let half = side / 2;
        let rows: Vec<u32> = grid_cells(half)
            .iter()
            .map(|c| {
                ((2 * c[0] as usize * side + 2 * c[1] as usize) * side + 2 * c[2] as usize) as u32
            })
            .collect();
        out.index_select(&Tensor::from_vec(rows, half * half * half, xs.device())?, 0)
    }
}

struct ResBlock {
    norm1: GroupNorm,
    conv1: Conv3,
    norm2: GroupNorm,
    conv2: Conv3,
}

impl ResBlock {
    fn load(c: usize, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("block");
        let g = safe_groups(c);
        Ok(Self {
            norm1: super::load_group_norm(g, c, GROUP_EPS, vb.pp(0))?,
            conv1: Conv3::load(c, c, 1, vb.pp(2))?,
            norm2: super::load_group_norm(g, c, GROUP_EPS, vb.pp(3))?,
            conv2: Conv3::load(c, c, 1, vb.pp(5))?,
        })
    }

    fn forward(&self, xs: &Tensor, side: usize, neighbors: &Neighbors) -> Result<Tensor> {
        let h = self
            .conv1
            .forward(&group_norm(&self.norm1, xs)?.silu()?, side, neighbors)?;
        let h = self
            .conv2
            .forward(&group_norm(&self.norm2, &h)?.silu()?, side, neighbors)?;
        xs + h
    }
}

/// `VoxelFieldConditioner`: the scaffold's occupancy to `[1, 512, 256]` condition tokens (f32).
pub struct VoxelConditioner {
    stem: (Conv3, GroupNorm),
    levels: Vec<(Vec<ResBlock>, Conv3, GroupNorm)>,
    tail: Vec<ResBlock>,
    proj: Linear,
    pos: Tensor,
    device: Device,
}

impl VoxelConditioner {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let mut levels = Vec::new();
        let mut c = BASE_CHANNELS;
        for i in 0..DOWNSAMPLES {
            let net = vb.pp("down_blocks").pp(i).pp("net");
            let blocks = (0..BLOCKS_PER_LEVEL)
                .map(|j| ResBlock::load(c, net.pp(j)))
                .collect::<Result<_>>()?;
            let down = Conv3::load(c, 2 * c, 2, net.pp(BLOCKS_PER_LEVEL))?;
            let norm = super::load_group_norm(
                safe_groups(2 * c),
                2 * c,
                GROUP_EPS,
                net.pp(BLOCKS_PER_LEVEL + 1),
            )?;
            levels.push((blocks, down, norm));
            c *= 2;
        }
        let side = VOXEL_GRID >> DOWNSAMPLES;
        let proj = vb.pp("proj");
        let pos = conditioner_positions(side, COND_CHANNELS, &device)?;
        Ok(Self {
            stem: (
                Conv3::load(1, BASE_CHANNELS, 1, vb.pp("stem").pp(0))?,
                super::load_group_norm(
                    safe_groups(BASE_CHANNELS),
                    BASE_CHANNELS,
                    GROUP_EPS,
                    vb.pp("stem").pp(1),
                )?,
            ),
            levels,
            tail: (0..BLOCKS_PER_LEVEL)
                .map(|j| ResBlock::load(c, vb.pp("tail_blocks").pp(j)))
                .collect::<Result<_>>()?,
            proj: Linear::new(
                proj.get((COND_CHANNELS, c, 1, 1, 1), "weight")?
                    .reshape((COND_CHANNELS, c))?,
                Some(proj.get(COND_CHANNELS, "bias")?),
            ),
            pos,
            device,
        })
    }

    /// The tokens for the occupied `cells` of the 64^3 grid.
    pub fn forward(&self, cells: &[[i32; 3]]) -> Result<Tensor> {
        let n = VOXEL_GRID;
        let mut field = vec![0f32; n * n * n];
        for c in cells {
            field[(c[0] as usize * n + c[1] as usize) * n + c[2] as usize] = 1.0;
        }
        let mut side = n;
        let mut neighbors = Neighbors::new(&grid_cells(side), &self.device)?;
        let xs = Tensor::from_vec(field, (n * n * n, 1), &self.device)?;
        let mut xs =
            group_norm(&self.stem.1, &self.stem.0.forward(&xs, side, &neighbors)?)?.silu()?;
        for (blocks, down, norm) in &self.levels {
            for block in blocks {
                xs = block.forward(&xs, side, &neighbors)?;
            }
            xs = group_norm(norm, &down.forward(&xs, side, &neighbors)?)?.silu()?;
            side /= 2;
            neighbors = Neighbors::new(&grid_cells(side), &self.device)?;
        }
        for block in &self.tail {
            xs = block.forward(&xs, side, &neighbors)?;
        }
        self.proj
            .forward(&xs)?
            .broadcast_add(&self.pos)?
            .unsqueeze(0)
    }
}

// `_get_3d_sincos_embed`: each axis's 1D sincos over (dim / 3) rounded down to even, padded with zeros
fn conditioner_positions(n: usize, dim: usize, device: &Device) -> Result<Tensor> {
    let axis = dim / 3 / 2 * 2;
    let e = sincos_1d(n, axis);
    let mut out = vec![0f32; n * n * n * dim];
    for (i, row) in out.chunks_mut(dim).enumerate() {
        let (d, h, w) = (i / (n * n), i / n % n, i % n);
        row[..axis].copy_from_slice(&e[d]);
        row[axis..2 * axis].copy_from_slice(&e[h]);
        row[2 * axis..3 * axis].copy_from_slice(&e[w]);
    }
    Tensor::from_vec(out, (n * n * n, dim), device)
}

/// `RMSNorm(head_dim)`: `x / sqrt(mean(x^2) + eps) * weight`.
struct HeadRms {
    weight: Tensor,
}

impl HeadRms {
    fn load(d: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get(d, "weight")?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let x = xs.to_dtype(DType::F32)?;
        let rms = (x.sqr()?.mean_keepdim(D::Minus1)? + FLOW_RMS_EPS)?.sqrt()?;
        x.broadcast_div(&rms)?
            .broadcast_mul(&self.weight)?
            .to_dtype(dtype)
    }
}

fn modulate(xs: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    xs.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(shift)
}

// `[1, n, h, d]` heads to `sdpa`'s `[1, h, n, d]` and back to `[1, n, h * d]`
fn attend(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let (b, n, h, d) = q.dims4()?;
    sdpa(
        &q.transpose(1, 2)?,
        &k.transpose(1, 2)?,
        &v.transpose(1, 2)?,
    )?
    .transpose(1, 2)?
    .reshape((b, n, h * d))
}

struct FlowBlock {
    modulation: AmpLinear,
    qkv: AmpLinear,
    norm_q: HeadRms,
    norm_k: HeadRms,
    proj_out: AmpLinear,
    q_ca: AmpLinear,
    kv_ca: AmpLinear,
    norm_q_ca: HeadRms,
    norm_k_ca: HeadRms,
    proj_ca_out: AmpLinear,
    fc1: AmpLinear,
    fc2: AmpLinear,
}

impl FlowBlock {
    fn load(dtype: DType, vb: VarBuilder) -> Result<Self> {
        let c = FLOW_WIDTH;
        let d = c / FLOW_HEADS;
        Ok(Self {
            modulation: AmpLinear::load(
                c,
                ADALN_CHUNKS * c,
                true,
                dtype,
                vb.pp("adaLN_modulation").pp(1),
            )?,
            qkv: AmpLinear::load(c, 3 * c, true, dtype, vb.pp("qkv"))?,
            norm_q: HeadRms::load(d, vb.pp("norm_q"))?,
            norm_k: HeadRms::load(d, vb.pp("norm_k"))?,
            proj_out: AmpLinear::load(c, c, true, dtype, vb.pp("proj_out"))?,
            q_ca: AmpLinear::load(c, c, true, dtype, vb.pp("q_ca"))?,
            kv_ca: AmpLinear::load(c, 2 * c, true, dtype, vb.pp("kv_ca"))?,
            norm_q_ca: HeadRms::load(d, vb.pp("norm_q_ca"))?,
            norm_k_ca: HeadRms::load(d, vb.pp("norm_k_ca"))?,
            proj_ca_out: AmpLinear::load(c, c, true, dtype, vb.pp("proj_ca_out"))?,
            fc1: AmpLinear::load(c, MLP_RATIO * c, true, dtype, vb.pp("mlp").pp(0))?,
            fc2: AmpLinear::load(MLP_RATIO * c, c, true, dtype, vb.pp("mlp").pp(2))?,
        })
    }

    // `xs` `[1, n, c]` f32; `c_emb` `[1, c]`; `cond` `[1, m, c]`
    fn forward(&self, xs: &Tensor, c_emb: &Tensor, rope: &Rope, cond: &Tensor) -> Result<Tensor> {
        let (b, n, c) = xs.dims3()?;
        let heads = FLOW_HEADS;
        let m = self
            .modulation
            .forward(&c_emb.silu()?)?
            .to_dtype(DType::F32)?;
        let part = |i: usize| m.narrow(1, i * c, c)?.unsqueeze(1);
        let h = modulate(&plain_norm(xs, NORM_EPS)?, &part(0)?, &part(1)?)?;
        let qkv = self.qkv.forward(&h)?.reshape((b, n, 3, heads, ()))?;
        let q = rope.apply(&self.norm_q.forward(&qkv.narrow(2, 0, 1)?.squeeze(2)?)?)?;
        let k = rope.apply(&self.norm_k.forward(&qkv.narrow(2, 1, 1)?.squeeze(2)?)?)?;
        let v = qkv.narrow(2, 2, 1)?.squeeze(2)?;
        let attn = self
            .proj_out
            .forward(&attend(&q, &k, &v)?)?
            .to_dtype(DType::F32)?;
        let xs = (xs + attn.broadcast_mul(&part(2)?)?)?;

        let mk = cond.dim(1)?;
        let q = self.norm_q_ca.forward(
            &self
                .q_ca
                .forward(&plain_norm(&xs, NORM_EPS)?)?
                .reshape((b, n, heads, ()))?,
        )?;
        let kv = self.kv_ca.forward(cond)?.reshape((b, mk, 2, heads, ()))?;
        let k = self.norm_k_ca.forward(&kv.narrow(2, 0, 1)?.squeeze(2)?)?;
        let v = kv.narrow(2, 1, 1)?.squeeze(2)?;
        let attn = self
            .proj_ca_out
            .forward(&attend(&q, &k, &v)?)?
            .to_dtype(DType::F32)?;
        let xs = (xs + attn.broadcast_mul(&part(3)?)?)?;

        let h = modulate(&plain_norm(&xs, NORM_EPS)?, &part(4)?, &part(5)?)?;
        let mlp = self
            .fc2
            .forward(&self.fc1.forward(&h)?.gelu()?)?
            .to_dtype(DType::F32)?;
        xs + mlp.broadcast_mul(&part(6)?)?
    }
}

/// Unit-cube coordinates of 1024^3 cells (`(v + 0.5) / 1024 * 2 - 1`), `[1, n, 3]`.
fn unit_coords(cells: &[[i32; 3]], device: &Device) -> Result<Tensor> {
    let flat: Vec<f32> = cells
        .iter()
        .flat_map(|c| c.map(|v| (v as f32 + 0.5) / NUM_DISCRETE as f32 * 2.0 - 1.0))
        .collect();
    Tensor::from_vec(flat, (1, cells.len(), 3), device)
}

fn cell_rope(cells: &[[i32; 3]], head_dim: usize, device: &Device) -> Result<Rope> {
    let positions: Vec<[f32; 3]> = cells.iter().map(|c| c.map(|v| v as f32)).collect();
    Rope::new(&positions, head_dim, device)
}

/// `TopologySiTFlow`.
pub struct TopologyFlow {
    input: Linear,
    coord: (Linear, Linear),
    t_embed: TimestepMlp,
    cond_proj: (Linear, Linear),
    index_pos: Tensor,
    blocks: Vec<FlowBlock>,
    final_mod: Linear,
    final_out: Linear,
    device: Device,
}

impl TopologyFlow {
    /// The blocks' linears run in `dtype` (f32 in the reference), the rest in f32.
    pub fn load(dtype: DType, vb: VarBuilder) -> Result<Self> {
        let c = FLOW_WIDTH;
        let device = vb.device().clone();
        let lin = |i: usize, o: usize, vb: VarBuilder| inference_tensor::nn::linear(i, o, vb);
        let index_pos: Vec<f32> = sincos_1d(MAX_VERTICES, c).into_iter().flatten().collect();
        Ok(Self {
            input: lin(TOPO_LATENT, c, vb.pp("input_proj"))?,
            coord: (
                lin(3, c, vb.pp("coord_embed").pp(0))?,
                lin(c, c, vb.pp("coord_embed").pp(2))?,
            ),
            t_embed: TimestepMlp::load(c, vb.pp("t_embedder"))?,
            cond_proj: (
                lin(COND_CHANNELS, c, vb.pp("cond_proj").pp(0))?,
                lin(c, c, vb.pp("cond_proj").pp(2))?,
            ),
            index_pos: Tensor::from_vec(index_pos, (MAX_VERTICES, c), &device)?,
            blocks: (0..FLOW_DEPTH)
                .map(|i| FlowBlock::load(dtype, vb.pp("blocks").pp(i)))
                .collect::<Result<_>>()?,
            final_mod: lin(c, 2 * c, vb.pp("final_layer").pp("adaLN_modulation").pp(1))?,
            final_out: lin(c, TOPO_LATENT, vb.pp("final_layer").pp("linear"))?,
            device,
        })
    }

    /// The velocity `[1, n, 16]` at time `t` (0 to 1) for latents `x` `[1, n, 16]` on vertex `cells`, under the
    /// projected condition `cond` `[1, m, 768]`.
    fn forward(
        &self,
        x: &Tensor,
        t: f32,
        base: &Tensor,
        rope: &Rope,
        cond: &Tensor,
    ) -> Result<Tensor> {
        let mut h = self.input.forward(x)?.broadcast_add(base)?;
        let c_emb = self.t_embed.forward(t, &self.device)?;
        for block in &self.blocks {
            h = block.forward(&h, &c_emb, rope, cond)?;
        }
        let m = self.final_mod.forward(&c_emb.silu()?)?;
        let c = FLOW_WIDTH;
        let (shift, scale) = (
            m.narrow(1, 0, c)?.unsqueeze(1)?,
            m.narrow(1, c, c)?.unsqueeze(1)?,
        );
        self.final_out
            .forward(&modulate(&plain_norm(&h, NORM_EPS)?, &shift, &scale)?)
    }

    /// Euler from `noise` `[1, n, 16]` at t = 0 to t = 1 over `steps`, for vertex `cells` under `voxel_cond`
    /// `[1, 512, 256]`: the latents, divided by the latent scale.
    pub fn sample(
        &self,
        noise: &Tensor,
        cells: &[[i32; 3]],
        voxel_cond: &Tensor,
        steps: usize,
        progress: &mut dyn FnMut(usize) -> Result<()>,
    ) -> Result<Tensor> {
        let n = cells.len();
        if n > MAX_VERTICES {
            inference_tensor::bail!("{n} vertices, past T-Flow's {MAX_VERTICES}");
        }
        let coords = unit_coords(cells, &self.device)?;
        let base = (self.index_pos.narrow(0, 0, n)?.unsqueeze(0)?
            + self
                .coord
                .1
                .forward(&self.coord.0.forward(&coords)?.silu()?)?)?;
        let cond = self
            .cond_proj
            .1
            .forward(&self.cond_proj.0.forward(voxel_cond)?.silu()?)?;
        let rope = cell_rope(cells, FLOW_WIDTH / FLOW_HEADS, &self.device)?;
        let mut x = noise.clone();
        for i in 0..steps {
            let (t, t_next) = (i as f64 / steps as f64, (i + 1) as f64 / steps as f64);
            let v = self.forward(&x, t as f32, &base, &rope, &cond)?;
            x = (x + (v * (t_next - t))?)?;
            progress(i + 1)?;
        }
        x / LATENT_SCALE
    }
}

/// `FlashVarlenTransformerBlock` (and `GraphAttnVarlenBlock` without adjacency): learned per-channel residual scales.
struct PlainBlock {
    qkv: AmpLinear,
    proj_out: AmpLinear,
    fc1: AmpLinear,
    fc2: AmpLinear,
    scale_msa: Tensor,
    scale_mlp: Tensor,
}

impl PlainBlock {
    fn load(dtype: DType, vb: VarBuilder) -> Result<Self> {
        let c = VAE_WIDTH;
        Ok(Self {
            qkv: AmpLinear::load(c, 3 * c, true, dtype, vb.pp("qkv"))?,
            proj_out: AmpLinear::load(c, c, true, dtype, vb.pp("proj_out"))?,
            fc1: AmpLinear::load(c, MLP_RATIO * c, true, dtype, vb.pp("ffn").pp(0))?,
            fc2: AmpLinear::load(MLP_RATIO * c, c, true, dtype, vb.pp("ffn").pp(2))?,
            scale_msa: vb.get(c, "scale_msa")?,
            scale_mlp: vb.get(c, "scale_mlp")?,
        })
    }

    fn forward(&self, xs: &Tensor, rope: Option<&Rope>) -> Result<Tensor> {
        let (b, n, _) = xs.dims3()?;
        let qkv =
            self.qkv
                .forward(&plain_norm(xs, NORM_EPS)?)?
                .reshape((b, n, 3, VAE_HEADS, ()))?;
        let part = |i: usize| -> Result<Tensor> {
            let x = qkv.narrow(2, i, 1)?.squeeze(2)?;
            match (rope, i < 2) {
                (Some(rope), true) => rope.apply(&x),
                _ => Ok(x),
            }
        };
        let attn = self
            .proj_out
            .forward(&attend(&part(0)?, &part(1)?, &part(2)?)?)?
            .to_dtype(DType::F32)?;
        let xs = (xs + attn.broadcast_mul(&self.scale_msa)?)?;
        let mlp = self
            .fc2
            .forward(&self.fc1.forward(&plain_norm(&xs, NORM_EPS)?)?.gelu()?)?
            .to_dtype(DType::F32)?;
        xs + mlp.broadcast_mul(&self.scale_mlp)?
    }
}

/// The topology VAE's decoder and edge predictor.
pub struct TopologyDecoder {
    input: AmpLinear,
    backbone: Vec<PlainBlock>,
    connection: Vec<PlainBlock>,
    // the pair MLP's first layer split into its two halves `[768, 256]` each, its bias, its second layer
    pair_u: Tensor,
    pair_v: Tensor,
    pair_bias: Tensor,
    pair_out: Tensor,
    pair_out_bias: f32,
    device: Device,
}

impl TopologyDecoder {
    /// From `tvae.pt`'s state dict; the transformer's linears in `dtype` (bf16 as the reference's autocast), the pair
    /// MLP in f32.
    pub fn load(dtype: DType, vb: VarBuilder) -> Result<Self> {
        let dec = vb.pp("decoder");
        let mut backbone = Vec::new();
        for s in 0..VAE_STAGES {
            let stage = dec.pp("backbone").pp("stages").pp(s);
            backbone.push(PlainBlock::load(dtype, stage.pp("graph_block"))?);
            for f in 0..VAE_PLAIN_PER_STAGE {
                backbone.push(PlainBlock::load(dtype, stage.pp("flash_blocks").pp(f))?);
            }
        }
        let connections = VAE_STAGES * (1 + VAE_PLAIN_PER_STAGE);
        let mlp = vb.pp("connection_predictor").pp("mlp");
        let w1 = mlp.pp(0).get((PAIR_HIDDEN, 2 * VAE_WIDTH), "weight")?;
        Ok(Self {
            input: AmpLinear::load(TOPO_LATENT, VAE_WIDTH, true, dtype, dec.pp("input_proj"))?,
            backbone,
            connection: (0..connections)
                .map(|i| PlainBlock::load(dtype, vb.pp("connection_transformer_blocks").pp(i)))
                .collect::<Result<_>>()?,
            pair_u: w1.narrow(1, 0, VAE_WIDTH)?.t()?.contiguous()?,
            pair_v: w1.narrow(1, VAE_WIDTH, VAE_WIDTH)?.t()?.contiguous()?,
            pair_bias: mlp.pp(0).get(PAIR_HIDDEN, "bias")?,
            pair_out: mlp
                .pp(2)
                .get((1, PAIR_HIDDEN), "weight")?
                .t()?
                .contiguous()?,
            pair_out_bias: mlp.pp(2).get(1, "bias")?.to_vec1::<f32>()?[0],
            device: vb.device().clone(),
        })
    }

    /// Each vertex's features `[n, 768]` (f32) for latents `z` `[1, n, 16]` on vertex `cells`.
    pub fn features(&self, z: &Tensor, cells: &[[i32; 3]]) -> Result<Tensor> {
        let mut h = self.input.forward(z)?.to_dtype(DType::F32)?;
        for block in &self.backbone {
            h = block.forward(&h, None)?;
        }
        let rope = cell_rope(cells, VAE_WIDTH / VAE_HEADS, &self.device)?;
        for block in &self.connection {
            h = block.forward(&h, Some(&rope))?;
        }
        h.squeeze(0)
    }

    /// The pairs `(u, v)`, `u < v`, whose symmetrized logit passes `threshold`, row-major.
    pub fn edges(&self, feats: &Tensor, threshold: f32) -> Result<Vec<[u32; 2]>> {
        let n = feats.dim(0)?;
        let a = feats.matmul(&self.pair_u)?.broadcast_add(&self.pair_bias)?;
        let b = feats.matmul(&self.pair_v)?;
        let mut rows = Vec::with_capacity(n.div_ceil(PAIR_ROWS));
        let mut start = 0;
        while start < n {
            let r = PAIR_ROWS.min(n - start);
            // gelu_erf(a[u] + b[v]) . w2 for each u in this block and every v
            let hidden = a
                .narrow(0, start, r)?
                .unsqueeze(1)?
                .broadcast_add(&b.unsqueeze(0)?)?
                .gelu_erf()?;
            rows.push(
                hidden
                    .reshape((r * n, PAIR_HIDDEN))?
                    .matmul(&self.pair_out)?
                    .reshape((r, n))?,
            );
            start += r;
        }
        let m = Tensor::cat(&rows, 0)?;
        let logits = ((&m + m.t()?)? * 0.5)?;
        let logits = (logits + f64::from(self.pair_out_bias))?.to_vec2::<f32>()?;
        let mut edges = Vec::new();
        for (u, row) in logits.iter().enumerate() {
            for (v, &l) in row.iter().enumerate().skip(u + 1) {
                if l > threshold {
                    edges.push([u as u32, v as u32]);
                }
            }
        }
        Ok(edges)
    }
}
