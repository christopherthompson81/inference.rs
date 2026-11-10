//! JoVox's flow: TRELLIS.2's sparse DiT over the sampler's tokens, each embedded from its slots, its position and its
//! level, predicting the 256 values of each of its 8 slots. The blocks run in `dtype` (bf16, as the reference), the
//! embedding and output layers in f32.

use inference_tensor::nn::{LayerNorm, Linear, Module, VarBuilder};
use inference_tensor::{DType, Device, Result, Tensor};

use super::sampler::{MASK, SLOTS, Tokens};
use crate::trellis2::dit::{BLOCK_EPS, Block, CrossKv, DitSpec, Rope, TimestepMlp};

pub const CLASSES: usize = 256;
// the progress ratios are scaled to diffusion timesteps
const TIME_SCALE: f32 = 1000.0;
// the final F.layer_norm's default
const OUT_EPS: f64 = 1e-5;
// rope runs on unified coordinates scaled to this grid
const ROPE_RES: f32 = 256.0;
// the 4^3 and 16^3 cells' sides, which position a token in the unit cube
const LEVEL_SIDES: [f32; 2] = [4.0, 16.0];
const POINT_FREQS: usize = 10;
// each slot is 0, 1 or unknown, in the child and grandchild embeddings
const TRI: usize = 3;
const AXES: usize = 3;
// JoVox's MLP ratio
const MLP_RATIO: usize = 4;

/// The flow's widths, as the checkpoint's config sets them.
#[derive(Clone, Copy, Debug)]
pub struct FlowSpec {
    pub channels: usize,
    pub cond_channels: usize,
    pub blocks: usize,
    pub heads: usize,
}

pub(crate) struct Flow {
    // child-slot rows (`slot * 3 + state`) then grandchild rows folded through `grand_proj` (`(slot * 8 + bit) * 3 +
    // state`), so a token's slot embedding is its row counts times this
    slot_table: Tensor,
    base: Tensor,
    point_fc1: Linear,
    point_fc2: Linear,
    level_embed: Tensor,
    t_embed: [TimestepMlp; 2],
    blocks: Vec<Block>,
    out_norm: LayerNorm,
    out: Linear,
    head_dim: usize,
    dtype: DType,
    device: Device,
}

impl Flow {
    pub(crate) fn load(spec: FlowSpec, dtype: DType, vb: VarBuilder) -> Result<Self> {
        let c = spec.channels;
        let dit = DitSpec {
            channels: c,
            cond_channels: spec.cond_channels,
            blocks: spec.blocks,
            heads: spec.heads,
            mlp: MLP_RATIO * c,
            share_mod: false,
        };
        let device = vb.device().clone();
        let tri = vb
            .get((SLOTS, TRI, c), "shared_tri_embed")?
            .reshape((SLOTS * TRI, c))?;
        let grand = vb.pp("grand_proj").get((c, SLOTS * c), "weight")?;
        let mut rows = vec![tri.clone()];
        for slot in 0..SLOTS {
            rows.push(tri.matmul(&grand.narrow(1, slot * c, c)?.t()?.contiguous()?)?);
        }
        let ones = Tensor::ones(c, DType::F32, &device)?;
        let zeros = Tensor::zeros(c, DType::F32, &device)?;
        let plain = LayerNorm::new(ones.clone(), zeros.clone(), BLOCK_EPS);
        let point = vb.pp("point_embed").pp("mlp");
        let point_in = AXES * (1 + 2 * POINT_FREQS);
        let t = vb.pp("t_embedder");
        Ok(Self {
            slot_table: Tensor::cat(&rows, 0)?,
            base: vb.get(c, "base_embed")?,
            point_fc1: inference_tensor::nn::linear(point_in, c, point.pp(0))?,
            point_fc2: inference_tensor::nn::linear(c, c, point.pp(2))?,
            level_embed: vb.pp("level_embed").get((2, c), "weight")?,
            t_embed: [
                TimestepMlp::load(c, t.pp("r0"))?,
                TimestepMlp::load(c, t.pp("r1"))?,
            ],
            blocks: (0..spec.blocks)
                .map(|i| Block::load(dit, plain.clone(), dtype, vb.pp("blocks").pp(i)))
                .collect::<Result<_>>()?,
            out_norm: LayerNorm::new(ones, zeros, OUT_EPS),
            out: inference_tensor::nn::linear(c, SLOTS * CLASSES, vb.pp("out_layer"))?,
            head_dim: c / spec.heads,
            dtype,
            device,
        })
    }

    /// Each block's cross-attention keys and values over `cond` `[batch, tokens, cond_channels]`, which every step
    /// shares.
    pub(crate) fn condition(&self, cond: &Tensor) -> Result<Vec<CrossKv>> {
        let cond = cond.to_dtype(self.dtype)?;
        self.blocks.iter().map(|b| b.cross_kv(&cond)).collect()
    }

    // the tokens' f32 embeddings `[tokens, channels]`: slots, position, level
    fn embed(&self, tokens: &Tokens, units: &[[f32; 3]]) -> Result<Tensor> {
        let n = tokens.slots.len();
        let rows = SLOTS * TRI * (1 + SLOTS);
        let mut counts = vec![0f32; n * rows];
        for (t, slots) in tokens.slots.iter().enumerate() {
            let row = &mut counts[t * rows..][..rows];
            for (s, &v) in slots.iter().enumerate() {
                let (child, bits) = if v == MASK {
                    (2, [2; SLOTS])
                } else {
                    (
                        usize::from(v > 0),
                        std::array::from_fn(|j| usize::from((v >> j) & 1)),
                    )
                };
                row[s * TRI + child] += 1.0;
                for (j, &bit) in bits.iter().enumerate() {
                    row[SLOTS * TRI * (1 + s) + j * TRI + bit] += 1.0;
                }
            }
        }
        let counts = Tensor::from_vec(counts, (n, rows), &self.device)?;
        let freqs: Vec<f32> = (0..POINT_FREQS)
            .map(|k| (2.0 * std::f32::consts::PI) * (1u32 << k) as f32)
            .collect();
        let width = AXES * (1 + 2 * POINT_FREQS);
        let mut point = Vec::with_capacity(n * width);
        for u in units {
            point.extend_from_slice(u);
            for f in [f32::sin, f32::cos] {
                for &v in u {
                    point.extend(freqs.iter().map(|&w| f(v * w)));
                }
            }
        }
        let point = Tensor::from_vec(point, (n, width), &self.device)?;
        let point = self
            .point_fc2
            .forward(&self.point_fc1.forward(&point)?.silu()?)?;
        let levels: Vec<u32> = tokens.levels.iter().map(|&l| u32::from(l)).collect();
        let levels = Tensor::from_vec(levels, n, &self.device)?;
        counts
            .matmul(&self.slot_table)?
            .broadcast_add(&self.base)?
            .add(&point)?
            .add(&self.level_embed.index_select(&levels, 0)?)
    }

    /// Logits `[batch, tokens, 8, 256]` (f32) for `tokens`, each batch row under its own condition in `kv`.
    pub(crate) fn forward(&self, tokens: &Tokens, kv: &[CrossKv]) -> Result<Tensor> {
        let units: Vec<[f32; 3]> = tokens
            .coords
            .iter()
            .zip(&tokens.levels)
            .map(|(xyz, &l)| xyz.map(|v| (v as f32 + 0.5) / LEVEL_SIDES[usize::from(l)]))
            .collect();
        let batch = kv[0].batch()?;
        let n = units.len();
        let feats = self.embed(tokens, &units)?.to_dtype(self.dtype)?;
        let mut xs = feats.unsqueeze(0)?.repeat((batch, 1, 1))?;
        let t_emb = (self.t_embed[0].forward(tokens.ratios[0] * TIME_SCALE, &self.device)?
            + self.t_embed[1].forward(tokens.ratios[1] * TIME_SCALE, &self.device)?)?
        .to_dtype(self.dtype)?;
        let scaled: Vec<[f32; 3]> = units.iter().map(|u| u.map(|v| v * ROPE_RES)).collect();
        let rope = Rope::new(&scaled, self.head_dim, &self.device)?;
        for (block, kv) in self.blocks.iter().zip(kv) {
            xs = block.forward(&xs, &t_emb, &rope, kv)?;
        }
        let xs = self.out_norm.forward(&xs.to_dtype(DType::F32)?)?;
        self.out.forward(&xs)?.reshape((batch, n, SLOTS, CLASSES))
    }
}
