//! `Sparc3DTransformer`: sparse residual blocks that pool the cells 2x2x2, TRELLIS cross blocks over the pooled cells
//! with RoPE, and mirrored residual blocks that scatter back to the cells, under shared timestep modulation.

use std::collections::BTreeMap;

use inference_tensor::nn::{LayerNorm, Linear, Module, VarBuilder};
use inference_tensor::{DType, Device, Result, Tensor};

use crate::trellis2::dit::{BLOCK_EPS, Block, CrossKv, DitSpec, Rope, TimestepMlp};
use crate::trellis2::sparse::{Neighbors, SparseConv};

pub const LATENT_CHANNELS: usize = 32;
const IO_CHANNELS: usize = 256;
const MODEL_CHANNELS: usize = 2112;
const HEADS: usize = 22;
const DIT: DitSpec = DitSpec {
    channels: MODEL_CHANNELS,
    cond_channels: 2048,
    blocks: 24,
    heads: HEADS,
    mlp: 4 * MODEL_CHANNELS,
    share_mod: true,
    qk_norm: true,
    cross_qk_norm: false,
    rope_first: true,
};
// the final F.layer_norm's default
const OUT_EPS: f64 = 1e-5;
// RoPE's training grid: longer token grids rescale its frequencies
const ROPE_TRAIN_LENGTH: f32 = 64.0;
// wavelengths (in train lengths) above which a frequency is left as trained
const ROPE_KEEP_WAVELENGTH: f32 = 24.0;

/// `RotaryPositionEmbedder._update_freqs`: frequencies interpolated toward `train / infer` by wavelength.
fn rescaled_freqs(freqs: &[f32], infer_length: f32) -> Vec<f32> {
    let squeeze = ROPE_TRAIN_LENGTH / infer_length;
    freqs
        .iter()
        .map(|&f| {
            let wavelength = f * ROPE_TRAIN_LENGTH / (2.0 * std::f32::consts::PI);
            let ratio = if wavelength > ROPE_KEEP_WAVELENGTH {
                1.0
            } else if wavelength < 1.0 {
                squeeze
            } else {
                let r = (wavelength - 1.0) / (ROPE_KEEP_WAVELENGTH - 1.0);
                r + (1.0 - r) * squeeze
            };
            f * ratio
        })
        .collect()
}

/// The cells a pass samples on, with what every forward over them shares: neighbour tables at both resolutions, the
/// 2x2x2 pooling and the pooled cells' rotations.
pub struct Grid {
    full: Neighbors,
    half: Neighbors,
    // each cell's pooled row `[cells]` (u32) and each pooled row's divisor `[pooled, 1]` (f32)
    inverse: Tensor,
    counts: Tensor,
    pooled: usize,
    rope: Rope,
}

impl Grid {
    pub fn new(coords: &[[i32; 3]], device: &Device) -> Result<Self> {
        // SparseDownsample's order: unique pooled coordinates sorted
        let mut rows: BTreeMap<[i32; 3], u32> =
            coords.iter().map(|c| (c.map(|v| v / 2), 0)).collect();
        for (i, row) in rows.values_mut().enumerate() {
            *row = i as u32;
        }
        let pooled: Vec<[i32; 3]> = rows.keys().copied().collect();
        let inverse: Vec<u32> = coords.iter().map(|c| rows[&c.map(|v| v / 2)]).collect();
        // scatter_reduce's mean includes the zeros it reduces into: one more than the cells
        let mut counts = vec![1f32; pooled.len()];
        for &r in &inverse {
            counts[r as usize] += 1.0;
        }
        let head_dim = MODEL_CHANNELS / HEADS;
        let mut freqs = Rope::freqs(head_dim);
        let longest = pooled.iter().flatten().copied().max().unwrap_or(0) as f32;
        if longest >= ROPE_TRAIN_LENGTH {
            // `infer_length = (64 + max) // 2`
            freqs = rescaled_freqs(&freqs, ((ROPE_TRAIN_LENGTH + longest) / 2.0).floor());
        }
        let positions: Vec<[f32; 3]> = pooled.iter().map(|c| c.map(|v| v as f32)).collect();
        Ok(Self {
            full: Neighbors::new(coords, device)?,
            half: Neighbors::new(&pooled, device)?,
            inverse: Tensor::from_vec(inverse, coords.len(), device)?,
            counts: Tensor::from_vec(counts, (pooled.len(), 1), device)?,
            pooled: pooled.len(),
            rope: Rope::with_freqs(&positions, head_dim, &freqs, device)?,
        })
    }

    // `SparseDownsample(2)`: each pooled cell its cells' sum over one more than their count, in f32
    fn down(&self, feats: &Tensor) -> Result<Tensor> {
        let dtype = feats.dtype();
        let feats = feats.to_dtype(DType::F32)?;
        let sum = Tensor::zeros((self.pooled, feats.dim(1)?), DType::F32, feats.device())?
            .index_add(&self.inverse, &feats, 0)?;
        sum.broadcast_div(&self.counts)?.to_dtype(dtype)
    }

    // `SparseUpsample(2)`: each cell its pooled cell's features
    fn up(&self, feats: &Tensor) -> Result<Tensor> {
        feats.index_select(&self.inverse, 0)
    }
}

enum Resample {
    None,
    Down,
    Up,
}

fn linear(cin: usize, cout: usize, dtype: DType, vb: VarBuilder) -> Result<Linear> {
    Ok(Linear::new(
        vb.get((cout, cin), "weight")?.to_dtype(dtype)?,
        Some(vb.get(cout, "bias")?.to_dtype(dtype)?),
    ))
}

fn plain_norm(c: usize, eps: f64, device: &Device) -> Result<LayerNorm> {
    Ok(LayerNorm::new(
        Tensor::ones(c, DType::F32, device)?,
        Tensor::zeros(c, DType::F32, device)?,
        eps,
    ))
}

// a LayerNorm in f32, back to `dtype`
fn norm(norm: &LayerNorm, xs: &Tensor, dtype: DType) -> Result<Tensor> {
    norm.forward(&xs.to_dtype(DType::F32)?)?.to_dtype(dtype)
}

/// `SparseResBlock3d`: LN, SiLU, conv, modulated LN, SiLU, conv, SiLU, linear, plus the (projected) input.
struct ResBlock {
    norm1: LayerNorm,
    norm2: LayerNorm,
    conv1: SparseConv,
    conv2: SparseConv,
    emb: Linear,
    linear: Linear,
    skip: Option<Linear>,
    hidden: usize,
    resample: Resample,
    dtype: DType,
}

impl ResBlock {
    fn load(
        cin: usize,
        cout: usize,
        resample: Resample,
        dtype: DType,
        vb: VarBuilder,
    ) -> Result<Self> {
        let hidden = cout / 2;
        let n1 = vb.pp("norm1");
        Ok(Self {
            norm1: LayerNorm::new(n1.get(cin, "weight")?, n1.get(cin, "bias")?, BLOCK_EPS),
            norm2: plain_norm(hidden, BLOCK_EPS, vb.device())?,
            conv1: SparseConv::load(cin, hidden, dtype, vb.pp("conv1").pp("conv"))?,
            conv2: SparseConv::load(hidden, hidden, dtype, vb.pp("conv2").pp("conv"))?,
            emb: linear(MODEL_CHANNELS, 2 * hidden, dtype, vb.pp("emb_layers").pp(1))?,
            linear: linear(hidden, cout, dtype, vb.pp("linear"))?,
            skip: if cin != cout {
                Some(linear(cin, cout, dtype, vb.pp("skip_connection"))?)
            } else {
                None
            },
            hidden,
            resample,
            dtype,
        })
    }

    // `x` `[cells, in]` on its grid; `emb` the timestep embedding `[1, 2112]` in the block's dtype
    fn forward(&self, x: &Tensor, emb: &Tensor, grid: &Grid) -> Result<Tensor> {
        let mods = self.emb.forward(&emb.silu()?)?;
        let (scale, shift) = (
            mods.narrow(1, 0, self.hidden)?,
            mods.narrow(1, self.hidden, self.hidden)?,
        );
        let (x, neighbors) = match self.resample {
            Resample::None => (x.clone(), &grid.full),
            Resample::Down => (grid.down(x)?, &grid.half),
            Resample::Up => (grid.up(x)?, &grid.full),
        };
        let h = norm(&self.norm1, &x, self.dtype)?.silu()?;
        let h = self.conv1.forward(&h, neighbors)?;
        let h = norm(&self.norm2, &h, self.dtype)?
            .broadcast_mul(&(scale + 1.0)?)?
            .broadcast_add(&shift)?
            .silu()?;
        let h = self.conv2.forward(&h, neighbors)?.silu()?;
        let skip = match &self.skip {
            Some(skip) => skip.forward(&x)?,
            None => x,
        };
        self.linear.forward(&h)? + skip
    }
}

pub struct Transformer {
    input: Linear,
    t_embed: TimestepMlp,
    shared_mod: Linear,
    input_blocks: Vec<ResBlock>,
    blocks: Vec<Block>,
    out_blocks: Vec<ResBlock>,
    out_norm: LayerNorm,
    out: Linear,
    dtype: DType,
    device: Device,
}

impl Transformer {
    /// The residual and transformer blocks in `dtype` (f16 in the reference); the input, timestep, shared modulation
    /// and output layers in f32.
    pub fn load(dtype: DType, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let c = MODEL_CHANNELS;
        let (ib, ob) = (vb.pp("input_blocks"), vb.pp("out_blocks"));
        Ok(Self {
            input: linear(
                LATENT_CHANNELS,
                IO_CHANNELS,
                DType::F32,
                vb.pp("input_layer"),
            )?,
            t_embed: TimestepMlp::load(c, vb.pp("t_embedder"))?,
            shared_mod: linear(c, 6 * c, DType::F32, vb.pp("adaLN_modulation").pp(1))?,
            input_blocks: vec![
                ResBlock::load(IO_CHANNELS, IO_CHANNELS, Resample::None, dtype, ib.pp(0))?,
                ResBlock::load(IO_CHANNELS, c, Resample::Down, dtype, ib.pp(1))?,
            ],
            blocks: (0..DIT.blocks)
                .map(|i| {
                    Block::load(
                        DIT,
                        plain_norm(c, BLOCK_EPS, &device)?,
                        dtype,
                        vb.pp("blocks").pp(i),
                    )
                })
                .collect::<Result<_>>()?,
            out_blocks: vec![
                ResBlock::load(2 * c, IO_CHANNELS, Resample::Up, dtype, ob.pp(0))?,
                ResBlock::load(
                    2 * IO_CHANNELS,
                    IO_CHANNELS,
                    Resample::None,
                    dtype,
                    ob.pp(1),
                )?,
            ],
            out_norm: plain_norm(IO_CHANNELS, OUT_EPS, &device)?,
            out: linear(IO_CHANNELS, LATENT_CHANNELS, DType::F32, vb.pp("out_layer"))?,
            dtype,
            device,
        })
    }

    /// Each block's cross-attention keys and values over `cond` `[1, tokens, 2048]`.
    pub fn condition(&self, cond: &Tensor) -> Result<Vec<CrossKv>> {
        let cond = cond.to_dtype(self.dtype)?;
        self.blocks.iter().map(|b| b.cross_kv(&cond)).collect()
    }

    /// The velocity `[cells, 32]` (f32) of latents `x` `[cells, 32]` on `grid` at timestep `t` (already `1000 *`).
    pub fn forward(&self, x: &Tensor, t: f32, grid: &Grid, kv: &[CrossKv]) -> Result<Tensor> {
        self.forward_inspect(x, t, grid, kv, &mut |_, _| {})
    }

    // `forward`, handing `inspect` the input layer's, each residual block's and the first and last DiT blocks' outputs
    pub(crate) fn forward_inspect(
        &self,
        x: &Tensor,
        t: f32,
        grid: &Grid,
        kv: &[CrossKv],
        inspect: &mut dyn FnMut(&str, &Tensor),
    ) -> Result<Tensor> {
        let h = self.input.forward(x)?;
        inspect("input_layer", &h);
        let h = h.to_dtype(self.dtype)?;
        let t_emb = self.t_embed.forward(t, &self.device)?;
        let emb = t_emb.to_dtype(self.dtype)?;
        let mut skips = Vec::with_capacity(self.input_blocks.len());
        let mut h = h;
        for (i, block) in self.input_blocks.iter().enumerate() {
            h = block.forward(&h, &emb, grid)?;
            inspect(&format!("in{i}"), &h);
            skips.push(h.clone());
        }
        let shared = self
            .shared_mod
            .forward(&t_emb.silu()?)?
            .to_dtype(self.dtype)?;
        let mut tokens = h.unsqueeze(0)?;
        let last = self.blocks.len() - 1;
        for (i, (block, kv)) in self.blocks.iter().zip(kv).enumerate() {
            tokens = block.forward(&tokens, &shared, Some(&grid.rope), kv)?;
            if i == 0 || i == last {
                inspect(&format!("block{i}"), &tokens.squeeze(0)?);
            }
        }
        let mut h = tokens.squeeze(0)?;
        for (i, (block, skip)) in self.out_blocks.iter().zip(skips.iter().rev()).enumerate() {
            h = block.forward(&Tensor::cat(&[&h, skip], 1)?, &emb, grid)?;
            inspect(&format!("out{i}"), &h);
        }
        let h = norm(&self.out_norm, &h, self.dtype)?.to_dtype(DType::F32)?;
        self.out.forward(&h)
    }

    /// The first DiT block on pooled tokens `x` `[pooled, 2112]`, unfused, handing `inspect` its parts.
    #[cfg(test)]
    pub(crate) fn block0_inspect(
        &self,
        x: &Tensor,
        t: f32,
        grid: &Grid,
        kv: &CrossKv,
        inspect: &mut dyn FnMut(&str, &Tensor),
    ) -> Result<Tensor> {
        let t_emb = self.t_embed.forward(t, &self.device)?;
        let shared = self
            .shared_mod
            .forward(&t_emb.silu()?)?
            .to_dtype(self.dtype)?;
        let x = x.to_dtype(self.dtype)?.unsqueeze(0)?;
        let fused = self.blocks[0].forward(&x, &shared, Some(&grid.rope), kv)?;
        inspect("fused", &fused.squeeze(0)?);
        self.blocks[0].forward_inspect(&x, &shared, Some(&grid.rope), kv, inspect)
    }

    /// The first residual block's norm, SiLU and first convolution on input-layer outputs `[cells, 256]` (f32).
    #[cfg(test)]
    pub(crate) fn in0_conv1(&self, h: &Tensor, grid: &Grid) -> Result<Tensor> {
        let b = &self.input_blocks[0];
        let h = norm(&b.norm1, &h.to_dtype(self.dtype)?, self.dtype)?.silu()?;
        b.conv1.forward(&h, &grid.full)
    }
}
