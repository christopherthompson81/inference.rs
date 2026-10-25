//! The DINOv2 ViT-L/14 (4 registers) WorldMirror embeds each frame with: all 24 blocks run and the normed patch
//! tokens are kept.

use inference_tensor::nn::VarBuilder;
use inference_tensor::{DType, Device, Result, Tensor};

use super::layers::{Block, BlockSpec, F32LayerNorm};

// DINOv2's own LayerNorm epsilon (the trunk's is 1e-5)
const DINO_EPS: f64 = 1e-6;
// tokens before the patches: cls and the registers
const CLS_TOKENS: usize = 1;
// the cubic kernel's coefficient for antialiased bicubic resampling (PyTorch's AA path, not the -0.75 of plain bicubic)
const AA_CUBIC_A: f64 = -0.5;

pub(crate) struct Dinov2 {
    proj_weight: Tensor,
    proj_bias: Tensor,
    cls_token: Tensor,
    register_tokens: Tensor,
    pos_embed: Tensor,
    blocks: Vec<Block>,
    norm: F32LayerNorm,
    patch: usize,
    dtype: DType,
}

impl Dinov2 {
    pub(crate) fn load(
        dim: usize,
        depth: usize,
        heads: usize,
        patch: usize,
        registers: usize,
        dtype: DType,
        vb: VarBuilder,
    ) -> Result<Self> {
        let grid = 518 / patch;
        let spec = BlockSpec {
            dim,
            heads,
            mlp_ratio: 4,
            eps: DINO_EPS,
            qk_norm: false,
            dtype,
        };
        let proj = vb.pp("patch_embed").pp("proj");
        Ok(Self {
            proj_weight: proj
                .get((dim, 3, patch, patch), "weight")?
                .to_dtype(dtype)?,
            proj_bias: proj.get(dim, "bias")?.to_dtype(dtype)?,
            cls_token: vb.get((1, 1, dim), "cls_token")?,
            register_tokens: vb.get((1, registers, dim), "register_tokens")?,
            pos_embed: vb.get((1, 1 + grid * grid, dim), "pos_embed")?,
            blocks: (0..depth)
                .map(|i| Block::load(spec, vb.pp("blocks").pp(i)))
                .collect::<Result<_>>()?,
            norm: F32LayerNorm::load(dim, DINO_EPS, vb.pp("norm"))?,
            patch,
            dtype,
        })
    }

    /// The normed patch tokens `[frames, hp * wp, dim]` (f32) of normalized images `[frames, 3, H, W]` (f32).
    pub(crate) fn forward(&self, images: &Tensor) -> Result<Tensor> {
        let (frames, _, h, w) = images.dims4()?;
        let (hp, wp) = (h / self.patch, w / self.patch);
        let patches = images
            .to_dtype(self.dtype)?
            .conv2d(&self.proj_weight, 0, self.patch, 1, 1)?
            .broadcast_add(&self.proj_bias.reshape((1, (), 1, 1))?)?
            .flatten_from(2)?
            .transpose(1, 2)?
            .to_dtype(DType::F32)?;
        let dim = patches.dim(2)?;
        let cls = self.cls_token.broadcast_as((frames, 1, dim))?;
        let pos = self.pos_embed(hp, wp)?;
        let tokens = Tensor::cat(&[&cls, &patches], 1)?.broadcast_add(&pos)?;
        let registers =
            self.register_tokens
                .broadcast_as((frames, self.register_tokens.dim(1)?, dim))?;
        let mut xs = Tensor::cat(
            &[
                &tokens.narrow(1, 0, CLS_TOKENS)?,
                &registers,
                &tokens.narrow(1, CLS_TOKENS, hp * wp)?,
            ],
            1,
        )?;
        for block in &self.blocks {
            xs = block.forward(&xs, None)?;
        }
        let specials = CLS_TOKENS + self.register_tokens.dim(1)?;
        self.norm.forward(&xs)?.narrow(1, specials, hp * wp)
    }

    // The position embedding for an `hp x wp` patch grid: cls kept, the 37 x 37 grid resampled with antialiased bicubic
    fn pos_embed(&self, hp: usize, wp: usize) -> Result<Tensor> {
        let (_, n, dim) = self.pos_embed.dims3()?;
        let grid = ((n - 1) as f64).sqrt() as usize;
        if hp == grid && wp == grid {
            return Ok(self.pos_embed.clone());
        }
        let table = self
            .pos_embed
            .narrow(1, 1, grid * grid)?
            .reshape((grid * grid, dim))?
            .to_device(&Device::Cpu)?
            .to_vec2::<f32>()?;
        let (rows, cols) = (aa_bicubic_weights(grid, hp), aa_bicubic_weights(grid, wp));
        let mut out = vec![0f32; hp * wp * dim];
        for (oy, row) in rows.iter().enumerate() {
            for (ox, col) in cols.iter().enumerate() {
                let cell = &mut out[(oy * wp + ox) * dim..][..dim];
                for &(iy, wy) in row {
                    for &(ix, wx) in col {
                        let weight = (wy * wx) as f32;
                        for (o, v) in cell.iter_mut().zip(&table[iy * grid + ix]) {
                            *o += weight * v;
                        }
                    }
                }
            }
        }
        let resampled = Tensor::from_vec(out, (1, hp * wp, dim), self.pos_embed.device())?;
        Tensor::cat(&[&self.pos_embed.narrow(1, 0, 1)?, &resampled], 1)
    }
}

// The cubic convolution kernel PyTorch's antialiased bicubic uses
fn aa_cubic(x: f64) -> f64 {
    let (a, x) = (AA_CUBIC_A, x.abs());
    if x < 1.0 {
        ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a
    } else {
        0.0
    }
}

// Each output index's (input index, weight) taps for resampling `input` samples to `output` with antialiased bicubic,
// as `F.interpolate(mode="bicubic", antialias=True, align_corners=False)` computes them
fn aa_bicubic_weights(input: usize, output: usize) -> Vec<Vec<(usize, f64)>> {
    let scale = input as f64 / output as f64;
    let (support, inv) = if scale >= 1.0 {
        (2.0 * scale, 1.0 / scale)
    } else {
        (2.0, 1.0)
    };
    (0..output)
        .map(|i| {
            let center = scale * (i as f64 + 0.5);
            let lo = ((center - support + 0.5) as isize).max(0) as usize;
            let hi = ((center + support + 0.5) as usize).min(input);
            let taps = (lo..hi)
                .map(|j| (j, aa_cubic((j as f64 - center + 0.5) * inv)))
                .collect::<Vec<_>>();
            let total: f64 = taps.iter().map(|t| t.1).sum();
            taps.into_iter().map(|(j, w)| (j, w / total)).collect()
        })
        .collect()
}
