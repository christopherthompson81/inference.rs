//! TRELLIS's sparse-structure decoder (`ss_dec_conv3d_16l8`, from TRELLIS-image-large): dense 3^3 convolutions with
//! channel LayerNorms, from an 8-channel 16^3 latent to occupancy logits at 64^3 through two conv + 3D pixel shuffle
//! upsamplings. A dense grid is a sparse one with every cell active, so the sparse convolution runs it.

use inference_tensor::nn::{LayerNorm, Module, VarBuilder};
use inference_tensor::{DType, Device, Result, Tensor};

use super::sparse::{Neighbors, SparseConv};

// nn.LayerNorm's default, as ChannelLayerNorm32 builds it
const EPS: f64 = 1e-5;
const SHUFFLE: usize = 8;

/// `ss_dec_conv3d_16l8_fp16.json`
pub const STRUCTURE_DECODER: StructureDecoderSpec = StructureDecoderSpec {
    latent: 8,
    channels: [512, 128, 32],
    res_blocks: 2,
    middle_blocks: 2,
};

#[derive(Clone, Copy, Debug)]
pub struct StructureDecoderSpec {
    pub latent: usize,
    pub channels: [usize; 3],
    pub res_blocks: usize,
    pub middle_blocks: usize,
}

/// Every cell of an `n`^3 grid, x-major (as torch flattens `[C, X, Y, Z]`).
pub fn grid_cells(n: usize) -> Vec<[i32; 3]> {
    (0..n * n * n)
        .map(|i| [(i / (n * n)) as i32, (i / n % n) as i32, (i % n) as i32])
        .collect()
}

// ChannelLayerNorm32: each cell's channels, in f32
fn norm(ln: &LayerNorm, xs: &Tensor) -> Result<Tensor> {
    ln.forward(&xs.to_dtype(DType::F32)?)?.to_dtype(xs.dtype())
}

struct ResBlock {
    norm1: LayerNorm,
    conv1: SparseConv,
    norm2: LayerNorm,
    conv2: SparseConv,
}

impl ResBlock {
    fn load(c: usize, dtype: DType, vb: VarBuilder) -> Result<Self> {
        let ln = |vb: VarBuilder| -> Result<LayerNorm> {
            Ok(LayerNorm::new(
                vb.get(c, "weight")?,
                vb.get(c, "bias")?,
                EPS,
            ))
        };
        Ok(Self {
            norm1: ln(vb.pp("norm1"))?,
            conv1: SparseConv::load_dense(c, c, dtype, vb.pp("conv1"))?,
            norm2: ln(vb.pp("norm2"))?,
            conv2: SparseConv::load_dense(c, c, dtype, vb.pp("conv2"))?,
        })
    }

    fn forward(&self, xs: &Tensor, neighbors: &Neighbors) -> Result<Tensor> {
        let h = self
            .conv1
            .forward(&norm(&self.norm1, xs)?.silu()?, neighbors)?;
        let h = self
            .conv2
            .forward(&norm(&self.norm2, &h)?.silu()?, neighbors)?;
        xs + h
    }
}

pub struct StructureDecoder {
    input: SparseConv,
    middle: Vec<ResBlock>,
    // per resolution: its blocks, then the upsampling conv to the next
    stages: Vec<(Vec<ResBlock>, Option<SparseConv>)>,
    out_norm: LayerNorm,
    out_conv: SparseConv,
    latent_side: usize,
    dtype: DType,
    device: Device,
}

impl StructureDecoder {
    /// The input and output layers run in f32, the blocks in `dtype` (f16 as the reference).
    pub fn load(
        spec: StructureDecoderSpec,
        latent_side: usize,
        dtype: DType,
        vb: VarBuilder,
    ) -> Result<Self> {
        let ch = spec.channels;
        let mut index = 0;
        let mut stages = Vec::new();
        for (i, &c) in ch.iter().enumerate() {
            let blocks = (0..spec.res_blocks)
                .map(|_| {
                    let b = ResBlock::load(c, dtype, vb.pp("blocks").pp(index));
                    index += 1;
                    b
                })
                .collect::<Result<Vec<_>>>()?;
            let up = if i + 1 < ch.len() {
                let conv = SparseConv::load_dense(
                    c,
                    ch[i + 1] * SHUFFLE,
                    dtype,
                    vb.pp("blocks").pp(index).pp("conv"),
                )?;
                index += 1;
                Some(conv)
            } else {
                None
            };
            stages.push((blocks, up));
        }
        let last = ch[ch.len() - 1];
        let out = vb.pp("out_layer");
        Ok(Self {
            input: SparseConv::load_dense(spec.latent, ch[0], DType::F32, vb.pp("input_layer"))?,
            middle: (0..spec.middle_blocks)
                .map(|i| ResBlock::load(ch[0], dtype, vb.pp("middle_block").pp(i)))
                .collect::<Result<_>>()?,
            stages,
            out_norm: LayerNorm::new(
                out.pp(0).get(last, "weight")?,
                out.pp(0).get(last, "bias")?,
                EPS,
            ),
            out_conv: SparseConv::load_dense(last, 1, DType::F32, out.pp(2))?,
            latent_side,
            dtype,
            device: vb.device().clone(),
        })
    }

    /// Occupancy logits `[cells]` (f32) on the full grid four times the latent's side, x-major, for a latent
    /// `[latent_side^3, 8]` (f32, x-major).
    pub fn forward(&self, latent: &Tensor) -> Result<Tensor> {
        let mut side = self.latent_side;
        let mut neighbors = Neighbors::new(&grid_cells(side), &self.device)?;
        let mut h = self
            .input
            .forward(latent, &neighbors)?
            .to_dtype(self.dtype)?;
        for block in &self.middle {
            h = block.forward(&h, &neighbors)?;
        }
        for (blocks, up) in &self.stages {
            for block in blocks {
                h = block.forward(&h, &neighbors)?;
            }
            if let Some(up) = up {
                h = pixel_shuffle(&up.forward(&h, &neighbors)?, side)?;
                side *= 2;
                neighbors = Neighbors::new(&grid_cells(side), &self.device)?;
            }
        }
        let h = norm(&self.out_norm, &h.to_dtype(DType::F32)?)?.silu()?;
        self.out_conv.forward(&h, &neighbors)?.squeeze(1)
    }
}

// 3D pixel shuffle on a full `side`^3 grid's cells `[side^3, c * 8]` (x-major): channel `c * 8 + 4a + 2b + d` goes to
// child (2x + a, 2y + b, 2z + d) of the grid twice the side, returned x-major
fn pixel_shuffle(x: &Tensor, side: usize) -> Result<Tensor> {
    let (n, c8) = x.dims2()?;
    let c = c8 / SHUFFLE;
    let rows = x
        .reshape((n, c, SHUFFLE))?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((n * SHUFFLE, c))?;
    let big = 2 * side;
    let index: Vec<u32> = grid_cells(big)
        .iter()
        .map(|&[x, y, z]| {
            let (x, y, z) = (x as usize, y as usize, z as usize);
            let parent = (x / 2 * side + y / 2) * side + z / 2;
            (parent * SHUFFLE + (x % 2) * 4 + (y % 2) * 2 + z % 2) as u32
        })
        .collect();
    let len = index.len();
    rows.index_select(&Tensor::from_vec(index, len, x.device())?, 0)
}

/// The occupied cells of `logits` (positive) on an `n`^3 grid, max-pooled down to `side`^3 when that is smaller (a cell
/// is occupied when any of its children is), x-major.
pub fn occupied(logits: &[f32], n: usize, side: usize) -> Vec<[usize; 3]> {
    let ratio = n / side;
    let mut keep = vec![false; side * side * side];
    for (i, &l) in logits.iter().enumerate() {
        if l > 0.0 {
            let (x, y, z) = (i / (n * n), i / n % n, i % n);
            keep[((x / ratio) * side + y / ratio) * side + z / ratio] = true;
        }
    }
    keep.iter()
        .enumerate()
        .filter(|&(_, &k)| k)
        .map(|(i, _)| [i / (side * side), i / side % side, i % side])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pixel_shuffle_places_channels_as_torch_does() -> Result<()> {
        // one cell, one output channel: channel j lands at child (j >> 2 & 1, j >> 1 & 1, j & 1)
        let x = Tensor::arange(0f32, 8.0, &Device::Cpu)?.reshape((1, 8))?;
        let out = pixel_shuffle(&x, 1)?.flatten_all()?.to_vec1::<f32>()?;
        // x-major children: (0,0,0) (0,0,1) (0,1,0) ... -> channels 0, 1, 2, ...
        assert_eq!(out, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        Ok(())
    }

    #[test]
    fn occupancy_pools_any_child() {
        let mut logits = vec![-1.0f32; 64];
        // cell (3, 0, 1) of the 4^3 grid sits in (1, 0, 0) of the 2^3 one
        logits[3 * 16 + 1] = 2.0;
        assert_eq!(occupied(&logits, 4, 4), [[3, 0, 1]]);
        assert_eq!(occupied(&logits, 4, 2), [[1, 0, 0]]);
    }
}
