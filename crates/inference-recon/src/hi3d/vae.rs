//! `SparconvVAE`'s decoder (`PTv3MeshDecoder`): latents on the structure's cells to per-corner signed distances and
//! deformations at 8x the resolution. Serialized patch attention mixes the coarsest cells; each stage's unpooling keeps
//! the children its head predicts.

use inference_tensor::nn::{LayerNorm, Linear, Module, VarBuilder};
use inference_tensor::{D, DType, Device, Result, Tensor};

use super::transformer::LATENT_CHANNELS;
use crate::trellis2::dit::sdpa;
use crate::trellis2::sparse::{Neighbors, SparseConv};

const MID_CHANNELS: usize = 512;
// channels and residual blocks per stage, coarsest last
const STAGE_CHANNELS: [usize; 4] = [128, 256, 512, 512];
const STAGE_DEPTH: usize = 2;
const ATTN_HEADS: usize = 32;
const PATCH: usize = 48;
const MLP_RATIO: usize = 4;
const GROUPS: usize = 32;
const MIN_GROUP_CHANNELS: usize = 4;
const GN_EPS: f64 = 1e-6;
const LN_EPS: f64 = 1e-5;
/// The sign logits ahead of the attributes in the decoder's output.
pub const SIGN_CHANNELS: usize = 8;
const ATTR_CHANNELS: usize = 32;
// residual blocks pass larger point sets through untouched (the reference's memory guard, weights measured with it)
const RESNET_SKIP_CELLS: usize = 12_000_000;
const CHILDREN: usize = 8;
// rows a group norm takes in f32 at once
const NORM_CHUNK_BYTES: usize = 256 << 20;

/// Decoded cells: their coordinates at the decoded resolution, sign logits `[cells, 8]` and activated attributes
/// `[cells, 32]` (8 corner distances, 8 corner deformations x3, tanh), f32.
pub struct Decoded {
    pub coords: Vec<[i32; 3]>,
    pub resolution: usize,
    pub signs: Tensor,
    pub attrs: Tensor,
}

/// `PointGroupNorm`: per-group statistics over every cell, `weights` and `bias` `[1, c]`.
struct GroupNorm {
    weight: Tensor,
    bias: Tensor,
    groups: usize,
}

impl GroupNorm {
    fn load(c: usize, vb: VarBuilder) -> Result<Self> {
        let groups = if MIN_GROUP_CHANNELS * GROUPS > c {
            c / MIN_GROUP_CHANNELS
        } else {
            GROUPS
        };
        Ok(Self {
            weight: vb.get((1, c), "weights")?,
            bias: vb.get((1, c), "bias")?,
            groups,
        })
    }

    // statistics in f32 over row chunks, the output in the input's dtype
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (n, c) = x.dims2()?;
        let per = c / self.groups;
        let chunk = (NORM_CHUNK_BYTES / (c * 4)).max(1);
        let device = x.device();
        let mut sum = Tensor::zeros(self.groups, DType::F32, device)?;
        let mut sq = Tensor::zeros(self.groups, DType::F32, device)?;
        for start in (0..n).step_by(chunk) {
            let g = x
                .narrow(0, start, chunk.min(n - start))?
                .to_dtype(DType::F32)?
                .reshape(((), self.groups, per))?;
            sum = (sum + g.sum(0)?.sum(1)?)?;
            sq = (sq + g.sqr()?.sum(0)?.sum(1)?)?;
        }
        let count = (n * per) as f64 + GN_EPS;
        let mean = (sum / count)?;
        let var = ((sq / count)? - mean.sqr()?)?;
        let inv = (var + GN_EPS)?.sqrt()?.recip()?;
        let (mean, inv) = (
            mean.reshape((1, self.groups, 1))?,
            inv.reshape((1, self.groups, 1))?,
        );
        let mut parts = Vec::with_capacity(n.div_ceil(chunk));
        for start in (0..n).step_by(chunk) {
            let rows = chunk.min(n - start);
            parts.push(
                x.narrow(0, start, rows)?
                    .to_dtype(DType::F32)?
                    .reshape((rows, self.groups, per))?
                    .broadcast_sub(&mean)?
                    .broadcast_mul(&inv)?
                    .reshape((rows, c))?
                    .broadcast_mul(&self.weight)?
                    .broadcast_add(&self.bias)?
                    .to_dtype(x.dtype())?,
            );
        }
        Tensor::cat(&parts, 0)
    }
}

fn linear(cin: usize, cout: usize, bias: bool, dtype: DType, vb: VarBuilder) -> Result<Linear> {
    Ok(Linear::new(
        vb.get((cout, cin), "weight")?.to_dtype(dtype)?,
        if bias {
            Some(vb.get(cout, "bias")?.to_dtype(dtype)?)
        } else {
            None
        },
    ))
}

/// `SparseResnetBlock`: GroupNorm, Swish, conv, GroupNorm, Swish, conv, plus the (projected) input.
struct ResnetBlock {
    norm1: GroupNorm,
    conv1: SparseConv,
    norm2: GroupNorm,
    conv2: SparseConv,
    res: Option<Linear>,
}

impl ResnetBlock {
    fn load(cin: usize, cout: usize, dtype: DType, vb: VarBuilder) -> Result<Self> {
        let b = vb.pp("block");
        Ok(Self {
            norm1: GroupNorm::load(cin, b.pp(0))?,
            conv1: SparseConv::load(cin, cout, dtype, b.pp(2))?,
            norm2: GroupNorm::load(cout, b.pp(3))?,
            conv2: SparseConv::load(cout, cout, dtype, b.pp(5))?,
            res: if cin != cout {
                Some(linear(cin, cout, true, dtype, vb.pp("res_conv"))?)
            } else {
                None
            },
        })
    }

    fn forward(&self, x: &Tensor, neighbors: &Neighbors) -> Result<Tensor> {
        if x.dim(0)? > RESNET_SKIP_CELLS {
            return Ok(x.clone());
        }
        let h = self
            .conv1
            .forward(&self.norm1.forward(x)?.silu()?, neighbors)?;
        let h = self
            .conv2
            .forward(&self.norm2.forward(&h)?.silu()?, neighbors)?;
        let shortcut = match &self.res {
            Some(res) => res.forward(&x.to_dtype(h.dtype())?)?,
            None => x.to_dtype(h.dtype())?,
        };
        shortcut + h
    }
}

/// Z-order keys (x at bit 3i + 2, y at 3i + 1, z at 3i over `depth` bits).
fn z_key([x, y, z]: [i32; 3], depth: u32) -> u64 {
    let mut key = 0u64;
    for i in 0..depth {
        let bit = |v: i32| ((v as u64) >> i) & 1;
        key |= bit(x) << (3 * i + 2) | bit(y) << (3 * i + 1) | bit(z) << (3 * i);
    }
    key
}

/// Patch attention over a serialization: the cells in curve order cut into windows of 48, the last window the last
/// 48 cells (it overlaps the one before), each cell's output from its own window.
struct Serialization {
    // `[windows * 48]` cell rows (u32) and each cell's row in the windows' output `[cells]` (u32)
    gather: Tensor,
    scatter: Tensor,
    windows: usize,
    len: usize,
}

impl Serialization {
    fn new(coords: &[[i32; 3]], transposed: bool, device: &Device) -> Result<Self> {
        let n = coords.len();
        let longest = coords.iter().flatten().copied().max().unwrap_or(0).max(0) as u32;
        let depth = u32::BITS - longest.leading_zeros();
        let mut order: Vec<u32> = (0..n as u32).collect();
        let key = |i: u32| {
            let [x, y, z] = coords[i as usize];
            z_key(if transposed { [y, x, z] } else { [x, y, z] }, depth)
        };
        order.sort_by_key(|&i| key(i));
        let len = n.min(PATCH);
        let windows = n.div_ceil(PATCH);
        let tail = n - n % PATCH;
        let mut gather = Vec::with_capacity(windows * len);
        for w in 0..windows {
            let start = if n > PATCH {
                (w * PATCH).min(n - PATCH)
            } else {
                0
            };
            gather.extend_from_slice(&order[start..start + len]);
        }
        let mut scatter = vec![0u32; n];
        for (j, &cell) in order.iter().enumerate() {
            let row = if n.is_multiple_of(PATCH) || j < tail || n <= PATCH {
                j
            } else {
                (windows - 1) * PATCH + j - (n - PATCH)
            };
            scatter[cell as usize] = row as u32;
        }
        Ok(Self {
            gather: Tensor::from_vec(gather, windows * len, device)?,
            scatter: Tensor::from_vec(scatter, n, device)?,
            windows,
            len,
        })
    }
}

/// `SparseAttentionBlock`: conditional position encoding (conv, linear, LayerNorm), serialized patch attention and a
/// GELU MLP, each residual.
struct AttentionBlock {
    cpe_conv: SparseConv,
    cpe_linear: Linear,
    cpe_norm: LayerNorm,
    norm1: LayerNorm,
    qkv: Linear,
    proj: Linear,
    norm2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    transposed: bool,
}

fn layer_norm(c: usize, vb: VarBuilder) -> Result<LayerNorm> {
    Ok(LayerNorm::new(
        vb.get(c, "weight")?,
        vb.get(c, "bias")?,
        LN_EPS,
    ))
}

impl AttentionBlock {
    fn load(c: usize, transposed: bool, vb: VarBuilder) -> Result<Self> {
        let cpe = vb.pp("cpe");
        let mlp = vb.pp("mlp").pp(0);
        Ok(Self {
            cpe_conv: SparseConv::load(c, c, DType::F32, cpe.pp(0))?,
            cpe_linear: linear(c, c, true, DType::F32, cpe.pp(1))?,
            cpe_norm: layer_norm(c, cpe.pp(2))?,
            norm1: layer_norm(c, vb.pp("norm1").pp(0))?,
            qkv: linear(c, 3 * c, true, DType::F32, vb.pp("attn").pp("qkv"))?,
            proj: linear(c, c, true, DType::F32, vb.pp("attn").pp("proj"))?,
            norm2: layer_norm(c, vb.pp("norm2").pp(0))?,
            fc1: linear(c, MLP_RATIO * c, true, DType::F32, mlp.pp("fc1"))?,
            fc2: linear(MLP_RATIO * c, c, true, DType::F32, mlp.pp("fc2"))?,
            transposed,
        })
    }

    fn forward(&self, x: &Tensor, neighbors: &Neighbors, serial: &Serialization) -> Result<Tensor> {
        let cpe = self.cpe_norm.forward(
            &self
                .cpe_linear
                .forward(&self.cpe_conv.forward(x, neighbors)?)?,
        )?;
        let x = (x + cpe)?;
        let (n, c) = x.dims2()?;
        let d = c / ATTN_HEADS;
        let qkv = self
            .qkv
            .forward(&self.norm1.forward(&x)?)?
            .index_select(&serial.gather, 0)?
            .reshape((serial.windows, serial.len, 3, ATTN_HEADS, d))?;
        let part = |i: usize| {
            qkv.narrow(2, i, 1)?
                .squeeze(2)?
                .transpose(1, 2)?
                .contiguous()
        };
        let attn = sdpa(&part(0)?, &part(1)?, &part(2)?)?
            .transpose(1, 2)?
            .reshape((serial.windows * serial.len, c))?
            .index_select(&serial.scatter, 0)?;
        debug_assert_eq!(attn.dim(0)?, n);
        let x = (&x + self.proj.forward(&attn)?)?;
        let h = self
            .fc2
            .forward(&self.fc1.forward(&self.norm2.forward(&x)?)?.gelu_erf()?)?;
        x + h
    }
}

/// `SerializedUnpooling`: each cell's 8 children (child j at `2 * cell + (j >> 2 & 1, j >> 1 & 1, j & 1)`) projected
/// from its features, kept where the class head's logit is positive.
struct Unpool {
    head_norm: GroupNorm,
    head_conv: SparseConv,
    proj: Linear,
}

impl Unpool {
    fn load(c: usize, dtype: DType, vb: VarBuilder) -> Result<Self> {
        let up = vb.pp("upsample");
        let fc = up.pp("cls_head").pp("fc");
        Ok(Self {
            head_norm: GroupNorm::load(c, fc.pp(0))?,
            head_conv: SparseConv::load(c, CHILDREN, DType::F32, fc.pp(2))?,
            proj: linear(c, CHILDREN * c, false, dtype, up.pp("proj"))?,
        })
    }

    // each kept child's features: its parent's row through its slot's block of the projection, `[children, c]`
    fn forward(
        &self,
        x: &Tensor,
        coords: &[[i32; 3]],
        neighbors: &Neighbors,
    ) -> Result<(Tensor, Vec<[i32; 3]>, Neighbors)> {
        let c = x.dim(1)?;
        let logits = self
            .head_conv
            .forward(&self.head_norm.forward(x)?.silu()?, neighbors)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mut children = Vec::new();
        // the neighbour table's slots order children x-bit first
        let mut table_rows = Vec::new();
        let mut by_slot: [Vec<u32>; CHILDREN] = Default::default();
        let mut slot_of = Vec::new();
        for (i, p) in coords.iter().enumerate() {
            for j in 0..CHILDREN {
                if logits[i * CHILDREN + j] > 0.0 {
                    let (dx, dy, dz) = ((j >> 2) & 1, (j >> 1) & 1, j & 1);
                    children.push([
                        2 * p[0] + dx as i32,
                        2 * p[1] + dy as i32,
                        2 * p[2] + dz as i32,
                    ]);
                    table_rows.push((i * CHILDREN + (dx | dy << 1 | dz << 2)) as u32);
                    slot_of.push((j, by_slot[j].len()));
                    by_slot[j].push(i as u32);
                }
            }
        }
        let device = x.device();
        let weight = self.proj.weight();
        let x = x.to_dtype(weight.dtype())?;
        let mut blocks = Vec::with_capacity(CHILDREN);
        let mut offsets = [0usize; CHILDREN];
        let mut total = 0;
        for (j, parents) in by_slot.iter().enumerate() {
            offsets[j] = total;
            total += parents.len();
            if parents.is_empty() {
                continue;
            }
            let rows = x.index_select(&Tensor::from_slice(parents, parents.len(), device)?, 0)?;
            blocks.push(rows.matmul(&weight.narrow(0, j * c, c)?.t()?)?);
        }
        let order: Vec<u32> = slot_of
            .iter()
            .map(|&(j, k)| (offsets[j] + k) as u32)
            .collect();
        let feats = Tensor::cat(&blocks, 0)?
            .index_select(&Tensor::from_vec(order, children.len(), device)?, 0)?;
        let device_rows = Tensor::from_slice(&table_rows, table_rows.len(), device)?;
        let next = neighbors.subdivided(&children, &table_rows, &device_rows)?;
        Ok((feats, children, next))
    }
}

struct Stage {
    blocks: Vec<ResnetBlock>,
    up: Option<Unpool>,
}

pub struct Decoder {
    z_to_f: SparseConv,
    block_in: ResnetBlock,
    attention: Vec<AttentionBlock>,
    block_out: ResnetBlock,
    stages: Vec<Stage>,
    out_norm: GroupNorm,
    out_conv: SparseConv,
}

impl Decoder {
    /// The coarsest level (to the first unpooling's projection) in f32; the finer ones, where the cells multiply, in
    /// `fine` (f16 on a GPU, as the reference's autocast keeps them).
    pub fn load(fine: DType, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("decoder");
        let mid = vb.pp("mid");
        let top = STAGE_CHANNELS.len() - 1;
        let stages = (0..STAGE_CHANNELS.len())
            .rev()
            .map(|s| {
                let dec = vb.pp("dec").pp(format!("dec{s}"));
                let out = STAGE_CHANNELS[s.saturating_sub(1)];
                let dtype = if s == top { DType::F32 } else { fine };
                let mut cin = STAGE_CHANNELS[s];
                let blocks = (0..STAGE_DEPTH)
                    .map(|i| {
                        let block = ResnetBlock::load(cin, out, dtype, dec.pp(format!("block{i}")));
                        cin = out;
                        block
                    })
                    .collect::<Result<_>>()?;
                let up = if s > 0 {
                    Some(Unpool::load(out, fine, dec.pp("up"))?)
                } else {
                    None
                };
                Ok(Stage { blocks, up })
            })
            .collect::<Result<_>>()?;
        let last = STAGE_CHANNELS[0];
        let conv_out = vb.pp("conv_out").pp("conv");
        Ok(Self {
            z_to_f: SparseConv::load(
                LATENT_CHANNELS,
                MID_CHANNELS,
                DType::F32,
                vb.pp("z_to_f").pp("conv").pp(0),
            )?,
            block_in: ResnetBlock::load(
                MID_CHANNELS,
                MID_CHANNELS,
                DType::F32,
                mid.pp("block_in"),
            )?,
            attention: (0..2)
                .map(|i| {
                    AttentionBlock::load(MID_CHANNELS, i % 2 == 1, mid.pp(format!("block{i}")))
                })
                .collect::<Result<_>>()?,
            block_out: ResnetBlock::load(
                MID_CHANNELS,
                MID_CHANNELS,
                DType::F32,
                mid.pp("block_out"),
            )?,
            stages,
            out_norm: GroupNorm::load(last, conv_out.pp(0))?,
            out_conv: SparseConv::load(last, SIGN_CHANNELS + ATTR_CHANNELS, fine, conv_out.pp(2))?,
        })
    }

    /// Denormalized latents `[cells, 32]` (f32) on `coords` at `resolution` to the decoded cells.
    pub fn forward(
        &self,
        latents: &Tensor,
        coords: &[[i32; 3]],
        resolution: usize,
    ) -> Result<Decoded> {
        let device = latents.device();
        let mut neighbors = Neighbors::new(coords, device)?;
        let mut coords = coords.to_vec();
        let mut resolution = resolution;
        let mut x = self.z_to_f.forward(latents, &neighbors)?;
        x = self.block_in.forward(&x, &neighbors)?;
        let serials = [
            Serialization::new(&coords, false, device)?,
            Serialization::new(&coords, true, device)?,
        ];
        for block in &self.attention {
            x = block.forward(&x, &neighbors, &serials[usize::from(block.transposed)])?;
        }
        x = self.block_out.forward(&x, &neighbors)?;
        for stage in &self.stages {
            for block in &stage.blocks {
                x = block.forward(&x, &neighbors)?;
            }
            if let Some(up) = &stage.up {
                let (feats, children, next) = up.forward(&x, &coords, &neighbors)?;
                (x, coords, neighbors) = (feats, children, next);
                resolution *= 2;
            }
        }
        let out = self
            .out_conv
            .forward(&self.out_norm.forward(&x)?.silu()?, &neighbors)?;
        let out = out.to_dtype(DType::F32)?;
        Ok(Decoded {
            coords,
            resolution,
            signs: out.narrow(D::Minus1, 0, SIGN_CHANNELS)?,
            attrs: out
                .narrow(D::Minus1, SIGN_CHANNELS, ATTR_CHANNELS)?
                .tanh()?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(t: &Tensor) -> Vec<u32> {
        t.to_vec1::<u32>().unwrap()
    }

    #[test]
    fn the_last_window_takes_the_last_cells_and_answers_for_the_tail() {
        // 100 cells on a line along z: z order is their order
        let coords: Vec<[i32; 3]> = (0..100).map(|z| [0, 0, z]).collect();
        let s = Serialization::new(&coords, false, &Device::Cpu).unwrap();
        assert_eq!((s.windows, s.len), (3, PATCH));
        let gather = rows(&s.gather);
        assert_eq!(
            gather[2 * PATCH],
            52,
            "the last window starts 48 from the end"
        );
        let scatter = rows(&s.scatter);
        assert_eq!(
            scatter[97],
            (2 * PATCH + 97 - 52) as u32,
            "a tail cell answers from the last window"
        );
        assert_eq!(scatter[40], 40);
        assert_eq!(scatter[60], 60, "a cell in a full window answers from it");
    }

    #[test]
    fn a_short_set_is_one_window() {
        let coords: Vec<[i32; 3]> = (0..10).map(|z| [z, 0, 0]).collect();
        let s = Serialization::new(&coords, true, &Device::Cpu).unwrap();
        assert_eq!((s.windows, s.len), (1, 10));
    }

    #[test]
    fn z_order_puts_x_highest() {
        assert!(z_key([1, 0, 0], 2) > z_key([0, 1, 1], 2));
        assert_eq!(z_key([1, 1, 1], 1), 7);
    }
}
