//! Swin Transformer v1 as BiRefNet's backbone builds it: windows padded at the bottom-right edge, the shifted-window
//! mask made over the padded grid (also when the grid is a single window), each stage's output through its own norm.

use inference_tensor::nn::{LayerNorm, Linear, Module, VarBuilder};
use inference_tensor::{D, Device, Result, Tensor};

const PATCH: usize = 4;
const MLP_RATIO: usize = 4;
const EPS: f64 = 1e-5;
// the shifted-window mask's value between windows of different regions
const MASK_FILL: f32 = -100.0;

#[derive(Clone, Copy, Debug)]
pub(crate) struct SwinSpec {
    pub embed_dim: usize,
    pub depths: [usize; 4],
    pub heads: [usize; 4],
    pub window: usize,
}

/// Swin-L at window 12, BiRefNet's default backbone.
pub(crate) const SWIN_L: SwinSpec = SwinSpec {
    embed_dim: 192,
    depths: [2, 2, 18, 2],
    heads: [6, 12, 24, 48],
    window: 12,
};

struct Block {
    norm1: LayerNorm,
    qkv: Linear,
    proj: Linear,
    // `[heads, window^2, window^2]`
    bias: Tensor,
    norm2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    heads: usize,
    shift: usize,
}

// x `[B, H, W, C]` rolled by `-shift` (or back by `shift`) along H and W
fn roll(xs: &Tensor, shift: usize, back: bool) -> Result<Tensor> {
    let mut xs = xs.clone();
    for dim in [1, 2] {
        let n = xs.dim(dim)?;
        let s = if back { n - shift % n } else { shift % n };
        xs = Tensor::cat(&[xs.narrow(dim, s, n - s)?, xs.narrow(dim, 0, s)?], dim)?;
    }
    Ok(xs)
}

// `[B, Hp, Wp, C]` into `[B * windows, window^2, C]`, windows row-major
fn partition(xs: &Tensor, window: usize) -> Result<Tensor> {
    let (b, h, w, c) = xs.dims4()?;
    xs.reshape((b, h / window, window, w / window, window, c))?
        .permute((0, 1, 3, 2, 4, 5))?
        .reshape((b * (h / window) * (w / window), window * window, c))
}

fn reverse(windows: &Tensor, window: usize, (b, h, w): (usize, usize, usize)) -> Result<Tensor> {
    let c = windows.dim(D::Minus1)?;
    windows
        .reshape((b, h / window, w / window, window, window, c))?
        .permute((0, 1, 3, 2, 4, 5))?
        .reshape((b, h, w, c))
}

impl Block {
    fn load(dim: usize, heads: usize, window: usize, shift: usize, vb: VarBuilder) -> Result<Self> {
        let attn = vb.pp("attn");
        let span = 2 * window - 1;
        let table = attn.get((span * span, heads), "relative_position_bias_table")?;
        let n = window * window;
        let index: Vec<u32> = (0..n)
            .flat_map(|i| {
                (0..n).map(move |j| {
                    let (dy, dx) = (
                        i / window + window - 1 - j / window,
                        i % window + window - 1 - j % window,
                    );
                    (dy * span + dx) as u32
                })
            })
            .collect();
        let index = Tensor::from_vec(index, n * n, table.device())?;
        let bias = table
            .index_select(&index, 0)?
            .reshape((n, n, heads))?
            .permute((2, 0, 1))?
            .contiguous()?;
        Ok(Self {
            norm1: inference_tensor::nn::layer_norm(dim, EPS, vb.pp("norm1"))?,
            qkv: inference_tensor::nn::linear(dim, 3 * dim, attn.pp("qkv"))?,
            proj: inference_tensor::nn::linear(dim, dim, attn.pp("proj"))?,
            bias,
            norm2: inference_tensor::nn::layer_norm(dim, EPS, vb.pp("norm2"))?,
            fc1: inference_tensor::nn::linear(dim, MLP_RATIO * dim, vb.pp("mlp").pp("fc1"))?,
            fc2: inference_tensor::nn::linear(MLP_RATIO * dim, dim, vb.pp("mlp").pp("fc2"))?,
            heads,
            shift,
        })
    }

    // `[B, H * W, C]` through the block; `mask` is the shifted windows' `[windows, n, n]`
    fn forward(
        &self,
        xs: &Tensor,
        (h, w): (usize, usize),
        window: usize,
        mask: &Tensor,
    ) -> Result<Tensor> {
        let (b, _, c) = xs.dims3()?;
        let (hp, wp) = (h.div_ceil(window) * window, w.div_ceil(window) * window);
        let mut x = self
            .norm1
            .forward(xs)?
            .reshape((b, h, w, c))?
            .pad_with_zeros(1, 0, hp - h)?
            .pad_with_zeros(2, 0, wp - w)?;
        if self.shift > 0 {
            x = roll(&x, self.shift, false)?;
        }
        let windows = partition(&x, window)?;
        let (bw, n, _) = windows.dims3()?;
        let hd = c / self.heads;
        let qkv = self
            .qkv
            .forward(&windows)?
            .reshape((bw, n, 3, self.heads, hd))?
            .permute((2, 0, 3, 1, 4))?;
        let q = (qkv.get(0)? * (hd as f64).powf(-0.5))?.contiguous()?;
        let (k, v) = (qkv.get(1)?.contiguous()?, qkv.get(2)?.contiguous()?);
        let mut attn = q.matmul(&k.t()?)?.broadcast_add(&self.bias.unsqueeze(0)?)?;
        if self.shift > 0 {
            let nw = mask.dim(0)?;
            attn = attn
                .reshape((bw / nw, nw, self.heads, n, n))?
                .broadcast_add(&mask.unsqueeze(1)?.unsqueeze(0)?)?
                .reshape((bw, self.heads, n, n))?;
        }
        let attn = inference_tensor::nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?.transpose(1, 2)?.reshape((bw, n, c))?;
        let mut x = reverse(&self.proj.forward(&out)?, window, (b, hp, wp))?;
        if self.shift > 0 {
            x = roll(&x, self.shift, true)?;
        }
        let x = x.narrow(1, 0, h)?.narrow(2, 0, w)?.reshape((b, h * w, c))?;
        let xs = (xs + x)?;
        let mlp = self
            .fc2
            .forward(&self.fc1.forward(&self.norm2.forward(&xs)?)?.gelu_erf()?)?;
        xs + mlp
    }
}

// The shifted windows' mask `[windows, n, n]` over an `hp x wp` padded grid: -100 between tokens from different
// regions of the roll
fn shift_mask(
    hp: usize,
    wp: usize,
    window: usize,
    shift: usize,
    device: &Device,
) -> Result<Tensor> {
    let region = |i: usize, n: usize| {
        if i < n.saturating_sub(window) {
            0
        } else if i < n - shift {
            1
        } else {
            2
        }
    };
    let (nh, nw) = (hp / window, wp / window);
    let n = window * window;
    let mut out = Vec::with_capacity(nh * nw * n * n);
    for wy in 0..nh {
        for wx in 0..nw {
            let labels: Vec<usize> = (0..n)
                .map(|t| {
                    let (y, x) = (wy * window + t / window, wx * window + t % window);
                    3 * region(y, hp) + region(x, wp)
                })
                .collect();
            for &a in &labels {
                out.extend(labels.iter().map(|&b| if a == b { 0.0 } else { MASK_FILL }));
            }
        }
    }
    Tensor::from_vec(out, (nh * nw, n, n), device)
}

struct Merge {
    norm: LayerNorm,
    reduction: Linear,
}

impl Merge {
    fn forward(&self, xs: &Tensor, (h, w): (usize, usize)) -> Result<Tensor> {
        let (b, _, c) = xs.dims3()?;
        let x = xs
            .reshape((b, h, w, c))?
            .pad_with_zeros(1, 0, h % 2)?
            .pad_with_zeros(2, 0, w % 2)?;
        let (h2, w2) = (h.div_ceil(2), w.div_ceil(2));
        let x = x.reshape((b, h2, 2, w2, 2, c))?;
        let pick = |dy: usize, dx: usize| {
            x.narrow(2, dy, 1)?
                .narrow(4, dx, 1)?
                .reshape((b, h2, w2, c))
        };
        let x = Tensor::cat(&[pick(0, 0)?, pick(1, 0)?, pick(0, 1)?, pick(1, 1)?], 3)?
            .reshape((b, h2 * w2, 4 * c))?;
        self.reduction.forward(&self.norm.forward(&x)?)
    }
}

struct Stage {
    blocks: Vec<Block>,
    merge: Option<Merge>,
    norm: LayerNorm,
}

pub(crate) struct Swin {
    patch_weight: Tensor,
    patch_bias: Tensor,
    patch_norm: LayerNorm,
    stages: Vec<Stage>,
    spec: SwinSpec,
}

impl Swin {
    pub(crate) fn load(spec: SwinSpec, vb: VarBuilder) -> Result<Self> {
        let e = spec.embed_dim;
        let pe = vb.pp("patch_embed");
        let stages = (0..4)
            .map(|i| {
                let dim = e << i;
                let layer = vb.pp("layers").pp(i);
                let blocks = (0..spec.depths[i])
                    .map(|j| {
                        let shift = if j % 2 == 0 { 0 } else { spec.window / 2 };
                        Block::load(
                            dim,
                            spec.heads[i],
                            spec.window,
                            shift,
                            layer.pp("blocks").pp(j),
                        )
                    })
                    .collect::<Result<_>>()?;
                let merge = if i < 3 {
                    let ds = layer.pp("downsample");
                    Some(Merge {
                        norm: inference_tensor::nn::layer_norm(4 * dim, EPS, ds.pp("norm"))?,
                        reduction: Linear::new(
                            ds.pp("reduction").get((2 * dim, 4 * dim), "weight")?,
                            None,
                        ),
                    })
                } else {
                    None
                };
                Ok(Stage {
                    blocks,
                    merge,
                    norm: inference_tensor::nn::layer_norm(dim, EPS, vb.pp(format!("norm{i}")))?,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            patch_weight: pe.pp("proj").get((e, 3, PATCH, PATCH), "weight")?,
            patch_bias: pe.pp("proj").get(e, "bias")?,
            patch_norm: inference_tensor::nn::layer_norm(e, EPS, pe.pp("norm"))?,
            stages,
            spec,
        })
    }

    /// The four stages' normed outputs `[B, C, H, W]` for images `[B, 3, H, W]` (sides a multiple of 4).
    pub(crate) fn forward(&self, images: &Tensor) -> Result<Vec<Tensor>> {
        let x = images
            .conv2d(&self.patch_weight, 0, PATCH, 1, 1)?
            .broadcast_add(&self.patch_bias.reshape((1, (), 1, 1))?)?;
        let (b, c, mut h, mut w) = x.dims4()?;
        let mut x = self
            .patch_norm
            .forward(&x.flatten_from(2)?.transpose(1, 2)?)?;
        let window = self.spec.window;
        let mut outs = Vec::with_capacity(4);
        let mut dim = c;
        for stage in &self.stages {
            let (hp, wp) = (h.div_ceil(window) * window, w.div_ceil(window) * window);
            let mask = shift_mask(hp, wp, window, window / 2, x.device())?.to_dtype(x.dtype())?;
            for block in &stage.blocks {
                x = block.forward(&x, (h, w), window, &mask)?;
            }
            outs.push(
                stage
                    .norm
                    .forward(&x)?
                    .reshape((b, h, w, dim))?
                    .permute((0, 3, 1, 2))?
                    .contiguous()?,
            );
            if let Some(merge) = &stage.merge {
                x = merge.forward(&x, (h, w))?;
                (h, w, dim) = (h.div_ceil(2), w.div_ceil(2), 2 * dim);
            }
        }
        Ok(outs)
    }
}
