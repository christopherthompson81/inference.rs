//! TRELLIS.2's shape encoder (`FlexiDualGridVaeEncoder`, `shape_enc_next_dc_f16c32`): a mesh's flexible dual grid at
//! 1024^3 (or 512^3) down to shape latents on its 64^3 (32^3) cells, through ConvNeXt blocks and four space-to-channel
//! blocks, the decoder's mirror. The blocks run in f16, as the reference; norms in f32. The levels it pools through
//! are what the texture decoder unfolds back onto.

use std::collections::HashMap;

use inference_tensor::nn::{LayerNorm, Linear, Module, VarBuilder};
use inference_tensor::{DType, Device, Result, Tensor};

use super::decoder::{CHILDREN, ConvNext, OUT_EPS, Subdivision, layer_norm, norm_silu};
use super::sparse::{Neighbors, SparseConv, subdivide};
use super::voxelize::DualGrid;

const IN_CHANNELS: usize = 6;
const LEVELS: usize = 4;

/// `shape_enc_next_dc_f16c32_fp16.json`
pub const SHAPE_ENCODER: EncoderSpec = EncoderSpec {
    channels: [64, 128, 256, 512, 1024],
    blocks: [0, 4, 8, 16, 4],
    latent: 32,
};

#[derive(Clone, Copy, Debug)]
pub struct EncoderSpec {
    pub channels: [usize; 5],
    pub blocks: [usize; 5],
    pub latent: usize,
}

// one level of the hierarchy, coarse to fine: its cells, their neighbours, and for each the parent slot (row * 8 +
// child) it fills
struct Level {
    coords: Vec<[i32; 3]>,
    neighbors: Neighbors,
    parent_slots: Vec<u32>,
}

/// The cells from 64^3 down to the grid's own, each level in the order the decoder's subdivision makes them.
pub struct Hierarchy {
    levels: Vec<Level>,
    /// The subdivisions taking each level to the next: the texture decoder's guide
    pub subdivisions: Vec<Subdivision>,
    /// Each finest cell's index in the dual grid
    pub order: Vec<usize>,
}

impl Hierarchy {
    pub fn new(cells: &[[i32; 3]], device: &Device) -> Result<Self> {
        let mut sets: Vec<Vec<[i32; 3]>> = vec![cells.to_vec()];
        for _ in 0..LEVELS {
            let mut up: Vec<[i32; 3]> = sets[0].iter().map(|c| c.map(|v| v >> 1)).collect();
            up.sort_unstable();
            up.dedup();
            sets.insert(0, up);
        }
        let coarsest = sets[0].clone();
        let mut levels = vec![Level {
            neighbors: Neighbors::new(&coarsest, device)?,
            coords: coarsest,
            parent_slots: Vec::new(),
        }];
        let mut subdivisions = Vec::with_capacity(LEVELS);
        for finer in &sets[1..] {
            let present: std::collections::HashSet<[i32; 3]> = finer.iter().copied().collect();
            let parent = levels.last().expect("a coarsest level");
            let keep: Subdivision = parent
                .coords
                .iter()
                .map(|c| {
                    std::array::from_fn(|j| {
                        let j = j as i32;
                        present.contains(&[
                            2 * c[0] + (j & 1),
                            2 * c[1] + ((j >> 1) & 1),
                            2 * c[2] + ((j >> 2) & 1),
                        ])
                    })
                })
                .collect();
            let (children, rows) = subdivide(&parent.coords, &keep);
            let device_rows = Tensor::from_slice(&rows, rows.len(), device)?;
            let neighbors = parent
                .neighbors
                .subdivided(&children, &rows, &device_rows)?;
            subdivisions.push(keep);
            levels.push(Level {
                coords: children,
                neighbors,
                parent_slots: rows,
            });
        }
        let index: HashMap<[i32; 3], usize> =
            cells.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        let order = levels[LEVELS].coords.iter().map(|c| index[c]).collect();
        Ok(Self {
            levels,
            subdivisions,
            order,
        })
    }

    /// The latents' cells
    pub fn coarsest(&self) -> &[[i32; 3]] {
        &self.levels[0].coords
    }
}

// `[fine, c]` rows into their parents' child slots, `[parents, 8 * c]`, zero where a child is absent
fn to_parents(xs: &Tensor, slots: &[u32], parents: usize) -> Result<Tensor> {
    let (fine, c) = xs.dims2()?;
    let mut table = vec![fine as u32; parents * CHILDREN];
    for (k, &s) in slots.iter().enumerate() {
        table[s as usize] = k as u32;
    }
    let table = Tensor::from_vec(table, parents * CHILDREN, xs.device())?;
    let padded = Tensor::cat(&[xs, &Tensor::zeros((1, c), xs.dtype(), xs.device())?], 0)?;
    padded
        .index_select(&table, 0)?
        .reshape((parents, CHILDREN * c))
}

/// `SparseResBlockS2C3d`: a conv to an eighth of the output's channels, its children gathered into their parents'
/// channels, a second conv there; the skip averages the input's gathered channels in groups.
struct SpaceToChannel {
    norm1: LayerNorm,
    conv1: SparseConv,
    norm2: LayerNorm,
    conv2: SparseConv,
    out: usize,
}

impl SpaceToChannel {
    fn load(c: usize, out: usize, dtype: DType, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        Ok(Self {
            norm1: layer_norm(c, Some(vb.pp("norm1")), &device)?,
            conv1: SparseConv::load(c, out / CHILDREN, dtype, vb.pp("conv1"))?,
            norm2: layer_norm(out, None, &device)?,
            conv2: SparseConv::load(out, out, dtype, vb.pp("conv2"))?,
            out,
        })
    }

    fn forward(&self, xs: &Tensor, fine: &Level, coarse: &Level) -> Result<Tensor> {
        let parents = coarse.coords.len();
        let c = xs.dim(1)?;
        let h = self
            .conv1
            .forward(&norm_silu(&self.norm1, xs)?, &fine.neighbors)?;
        let h = to_parents(&h, &fine.parent_slots, parents)?;
        let h = self
            .conv2
            .forward(&norm_silu(&self.norm2, &h)?, &coarse.neighbors)?;
        let skip = to_parents(xs, &fine.parent_slots, parents)?
            .reshape((parents, self.out, CHILDREN * c / self.out))?
            .to_dtype(DType::F32)?
            .mean(2)?
            .to_dtype(h.dtype())?;
        h + skip
    }
}

/// `SparseUnetVaeEncoder` with the dual grid's six inputs a cell.
pub struct ShapeEncoder {
    input: Linear,
    stages: Vec<(Vec<ConvNext>, Option<SpaceToChannel>)>,
    out_norm: LayerNorm,
    to_latent: Linear,
    latent: usize,
    dtype: DType,
}

impl ShapeEncoder {
    pub fn load(spec: EncoderSpec, dtype: DType, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let ch = spec.channels;
        let stages = (0..ch.len())
            .map(|i| {
                let stage = vb.pp("blocks").pp(i);
                let blocks = (0..spec.blocks[i])
                    .map(|j| ConvNext::load(ch[i], dtype, stage.pp(j)))
                    .collect::<Result<Vec<_>>>()?;
                let down = (i + 1 < ch.len())
                    .then(|| {
                        SpaceToChannel::load(ch[i], ch[i + 1], dtype, stage.pp(spec.blocks[i]))
                    })
                    .transpose()?;
                Ok((blocks, down))
            })
            .collect::<Result<_>>()?;
        let last = ch[ch.len() - 1];
        Ok(Self {
            input: inference_tensor::nn::linear(IN_CHANNELS, ch[0], vb.pp("input_layer"))?,
            stages,
            out_norm: LayerNorm::new(
                Tensor::ones(last, DType::F32, &device)?,
                Tensor::zeros(last, DType::F32, &device)?,
                OUT_EPS,
            ),
            to_latent: inference_tensor::nn::linear(last, 2 * spec.latent, vb.pp("to_latent"))?,
            latent: spec.latent,
            dtype,
        })
    }

    /// The latents' means `[cells, latent]` (f32) on `hierarchy`'s coarsest cells, for `grid` (whose cells it was
    /// built from).
    pub fn forward(
        &self,
        grid: &DualGrid,
        hierarchy: &Hierarchy,
        device: &Device,
    ) -> Result<Tensor> {
        let inputs: Vec<f32> = hierarchy
            .order
            .iter()
            .flat_map(|&i| {
                let (o, x) = (grid.offsets[i], grid.crossed[i]);
                [
                    o[0] - 0.5,
                    o[1] - 0.5,
                    o[2] - 0.5,
                    f32::from(u8::from(x[0])) - 0.5,
                    f32::from(u8::from(x[1])) - 0.5,
                    f32::from(u8::from(x[2])) - 0.5,
                ]
            })
            .collect();
        let rows = hierarchy.order.len();
        let mut h = self
            .input
            .forward(&Tensor::from_vec(inputs, (rows, IN_CHANNELS), device)?)?
            .to_dtype(self.dtype)?;
        let levels = &hierarchy.levels;
        for (i, (blocks, down)) in self.stages.iter().enumerate() {
            let level = &levels[LEVELS - i];
            for block in blocks {
                h = block.forward(&h, &level.neighbors)?;
            }
            if let Some(down) = down {
                h = down.forward(&h, level, &levels[LEVELS - i - 1])?;
            }
        }
        let h = self.out_norm.forward(&h.to_dtype(DType::F32)?)?;
        self.to_latent.forward(&h)?.narrow(1, 0, self.latent)
    }
}
