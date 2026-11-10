//! TRELLIS.2's shape decoder (`FlexiDualGridVaeDecoder`, `shape_dec_next_dc_f16c32`): ConvNeXt blocks around 3^3
//! submanifold convolutions, and four channel-to-space blocks that each keep the children their cell predicts, from
//! the latents' 64^3 cells to 1024^3; each final cell holds a dual vertex, three edge-crossing logits and a quad split
//! weight. The blocks run in f16, as the reference; norms in f32.

use inference_tensor::nn::{LayerNorm, Linear, Module, VarBuilder};
use inference_tensor::{DType, Device, Result, Tensor};

use super::sparse::{Neighbors, Sparse, SparseConv, subdivide};
use crate::worldmirror::layers::AmpLinear;

const EPS: f64 = 1e-6;
// the output's final F.layer_norm default
const OUT_EPS: f64 = 1e-5;
const CHILDREN: usize = 8;
// a dual vertex sits within its cell grown by this margin on each side
const VOXEL_MARGIN: f64 = 0.5;
pub const OUT_CHANNELS: usize = 7;
const SOFTPLUS_THRESHOLD: f32 = 20.0;

/// `shape_dec_next_dc_f16c32_fp16.json`
pub const SHAPE_DECODER: DecoderSpec = DecoderSpec {
    channels: [1024, 512, 256, 128, 64],
    blocks: [4, 16, 8, 4, 0],
    latent: 32,
};

#[derive(Clone, Copy, Debug)]
pub struct DecoderSpec {
    pub channels: [usize; 5],
    pub blocks: [usize; 5],
    pub latent: usize,
}

fn layer_norm(c: usize, affine: Option<VarBuilder>, device: &Device) -> Result<LayerNorm> {
    Ok(match affine {
        Some(vb) => LayerNorm::new(vb.get(c, "weight")?, vb.get(c, "bias")?, EPS),
        None => LayerNorm::new(
            Tensor::ones(c, DType::F32, device)?,
            Tensor::zeros(c, DType::F32, device)?,
            EPS,
        ),
    })
}

// LayerNorm32: in f32, back to the stream's dtype
fn norm(ln: &LayerNorm, xs: &Tensor) -> Result<Tensor> {
    ln.forward(&xs.to_dtype(DType::F32)?)?.to_dtype(xs.dtype())
}

/// `SparseConvNeXtBlock3d`: conv, LayerNorm, Linear-SiLU-Linear, residual.
struct ConvNext {
    conv: SparseConv,
    norm: LayerNorm,
    fc1: AmpLinear,
    fc2: AmpLinear,
}

impl ConvNext {
    fn load(c: usize, dtype: DType, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        Ok(Self {
            conv: SparseConv::load(c, c, dtype, vb.pp("conv"))?,
            norm: layer_norm(c, Some(vb.pp("norm")), &device)?,
            fc1: AmpLinear::load(c, 4 * c, true, dtype, vb.pp("mlp").pp(0))?,
            fc2: AmpLinear::load(4 * c, c, true, dtype, vb.pp("mlp").pp(2))?,
        })
    }

    fn forward(&self, xs: &Tensor, neighbors: &Neighbors) -> Result<Tensor> {
        let h = norm(&self.norm, &self.conv.forward(xs, neighbors)?)?;
        let h = self.fc2.forward(&self.fc1.forward(&h)?.silu()?)?;
        xs + h
    }
}

/// `SparseResBlockC2S3d`: predicts which children each cell keeps, then a conv to 8 children's channels, rearranged
/// onto the kept children, and a second conv there; the skip repeats the input's per-child channels.
struct ChannelToSpace {
    to_subdiv: AmpLinear,
    norm1: LayerNorm,
    conv1: SparseConv,
    norm2: LayerNorm,
    conv2: SparseConv,
    out: usize,
}

impl ChannelToSpace {
    fn load(c: usize, out: usize, dtype: DType, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        Ok(Self {
            to_subdiv: AmpLinear::load(c, CHILDREN, true, dtype, vb.pp("to_subdiv"))?,
            norm1: layer_norm(c, Some(vb.pp("norm1")), &device)?,
            conv1: SparseConv::load(c, out * CHILDREN, dtype, vb.pp("conv1"))?,
            norm2: layer_norm(out, None, &device)?,
            conv2: SparseConv::load(out, out, dtype, vb.pp("conv2"))?,
            out,
        })
    }

    fn forward(&self, x: &Sparse, neighbors: &Neighbors) -> Result<(Sparse, Neighbors)> {
        let n = x.coords.len();
        let c = x.feats.dim(1)?;
        let subdiv = self
            .to_subdiv
            .forward(&x.feats)?
            .gt(0.0)?
            .to_dtype(DType::U8)?
            .flatten_all()?
            .to_vec1::<u8>()?;
        let keep: Vec<[bool; CHILDREN]> = subdiv
            .as_chunks::<CHILDREN>()
            .0
            .iter()
            .map(|k| std::array::from_fn(|j| k[j] != 0))
            .collect();
        let (children, rows) = subdivide(&x.coords, &keep);
        let device = x.feats.device();
        let rows = Tensor::from_vec(rows, children.len(), device)?;
        let h = self
            .conv1
            .forward(&norm(&self.norm1, &x.feats)?.silu()?, neighbors)?;
        let h = h
            .reshape((n * CHILDREN, self.out))?
            .index_select(&rows, 0)?;
        let skip = x
            .feats
            .reshape((n * CHILDREN, c / CHILDREN))?
            .index_select(&rows, 0)?;
        let m = children.len();
        let repeat = self.out / (c / CHILDREN);
        let skip = skip
            .unsqueeze(2)?
            .broadcast_as((m, c / CHILDREN, repeat))?
            .reshape((m, self.out))?;
        let neighbors = Neighbors::new(&children, device)?;
        let h = self
            .conv2
            .forward(&norm(&self.norm2, &h)?.silu()?, &neighbors)?;
        Ok((
            Sparse {
                coords: children,
                feats: (h + skip)?,
            },
            neighbors,
        ))
    }
}

/// What the decoder hands each stage's end to, for inspection.
pub type Inspect<'a> = &'a mut dyn FnMut(usize, &Sparse);

pub struct ShapeDecoder {
    from_latent: Linear,
    stages: Vec<(Vec<ConvNext>, Option<ChannelToSpace>)>,
    out_norm: LayerNorm,
    output: Linear,
    dtype: DType,
}

impl ShapeDecoder {
    pub fn load(spec: DecoderSpec, dtype: DType, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let ch = spec.channels;
        let stages = (0..ch.len())
            .map(|i| {
                let stage = vb.pp("blocks").pp(i);
                let blocks = (0..spec.blocks[i])
                    .map(|j| ConvNext::load(ch[i], dtype, stage.pp(j)))
                    .collect::<Result<Vec<_>>>()?;
                let up = (i + 1 < ch.len())
                    .then(|| {
                        ChannelToSpace::load(ch[i], ch[i + 1], dtype, stage.pp(spec.blocks[i]))
                    })
                    .transpose()?;
                Ok((blocks, up))
            })
            .collect::<Result<_>>()?;
        let last = ch[ch.len() - 1];
        Ok(Self {
            from_latent: inference_tensor::nn::linear(spec.latent, ch[0], vb.pp("from_latent"))?,
            stages,
            out_norm: LayerNorm::new(
                Tensor::ones(last, DType::F32, &device)?,
                Tensor::zeros(last, DType::F32, &device)?,
                OUT_EPS,
            ),
            output: inference_tensor::nn::linear(last, OUT_CHANNELS, vb.pp("output_layer"))?,
            dtype,
        })
    }

    /// The finest cells and their 7 outputs `[cells, 7]` (f32) for latents on `coords` (`[cells, latent]`, f32);
    /// `inspect` sees each stage's first block's input and its end.
    pub fn forward(
        &self,
        coords: &[[i32; 3]],
        latents: &Tensor,
        inspect: Inspect,
    ) -> Result<Sparse> {
        let device = latents.device().clone();
        let mut x = Sparse {
            coords: coords.to_vec(),
            feats: self.from_latent.forward(latents)?.to_dtype(self.dtype)?,
        };
        let mut neighbors = Neighbors::new(&x.coords, &device)?;
        for (i, (blocks, up)) in self.stages.iter().enumerate() {
            for block in blocks {
                x = x.replace(block.forward(&x.feats, &neighbors)?);
            }
            if let Some(up) = up {
                (x, neighbors) = up.forward(&x, &neighbors)?;
            }
            inspect(i, &x);
        }
        let h = self.out_norm.forward(&x.feats.to_dtype(DType::F32)?)?;
        Ok(x.replace(self.output.forward(&h)?))
    }
}

/// The decoder's outputs as the mesh extraction reads them.
pub struct DualGrid {
    /// Each cell's dual vertex within it (`2 sigmoid - 0.5`, the margin)
    pub vertices: Vec<[f32; 3]>,
    /// Which of its x/y/z edges the surface crosses
    pub crossed: Vec<[bool; 3]>,
    /// Its quad split weight (softplus)
    pub split: Vec<f32>,
}

/// [`DualGrid`] from the decoder's 7 outputs a cell.
pub fn dual_grid(out: &Tensor) -> Result<DualGrid> {
    let scale = (1.0 + 2.0 * VOXEL_MARGIN) as f32;
    let margin = VOXEL_MARGIN as f32;
    let rows = out.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
    let vertices = rows
        .iter()
        .map(|r| std::array::from_fn(|k| scale * sigmoid(r[k]) - margin))
        .collect();
    let crossed = rows
        .iter()
        .map(|r| std::array::from_fn(|k| r[3 + k] > 0.0))
        .collect();
    let split = rows.iter().map(|r| softplus(r[6])).collect();
    Ok(DualGrid {
        vertices,
        crossed,
        split,
    })
}

// torch's softplus (beta 1, threshold 20): log1p(exp(x)), x itself above 20
fn softplus(x: f32) -> f32 {
    if x > SOFTPLUS_THRESHOLD {
        x
    } else {
        x.exp().ln_1p()
    }
}
