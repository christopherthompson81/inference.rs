use inference_tensor::nn::{
    Conv2d, Conv2dConfig, LayerNorm, Linear, Module, VarBuilder, conv2d, layer_norm, linear, ops,
};
use inference_tensor::{D, DType, Device, Result, Tensor};

use super::config::EncoderConfig;

const LAYER_NORM_EPS: f64 = 1e-5;
const BATCH_NORM_EPS: f64 = 1e-5;
const POSITION_BASE: f64 = 10000.0;
// the conformer's macaron feed-forwards each contribute half a residual
const FEED_FORWARD_WEIGHT: f64 = 0.5;
// attention rows per block, about 40 s of audio at 80 ms a frame
const QUERY_BLOCK: usize = 512;

fn linear_maybe_bias(i: usize, o: usize, bias: bool, vb: VarBuilder) -> Result<Linear> {
    if bias {
        linear(i, o, vb)
    } else {
        Ok(Linear::new(vb.get((o, i), "weight")?, None))
    }
}

// NeMo's dw_striding: conv, then (depthwise, pointwise) pairs, each stride-2 conv halving time and frequency
struct Subsampling {
    first: Conv2d,
    pairs: Vec<(Conv2d, Conv2d)>,
    linear: Linear,
}

impl Subsampling {
    fn new(cfg: &EncoderConfig, vb: VarBuilder) -> Result<Self> {
        let (k, ch) = (
            cfg.subsampling_conv_kernel_size,
            cfg.subsampling_conv_channels,
        );
        let strided = Conv2dConfig {
            padding: (k - 1) / 2,
            stride: cfg.subsampling_conv_stride,
            ..Default::default()
        };
        let layers = vb.pp("layers");
        let first = conv2d(1, ch, k, strided, layers.pp(0))?;
        let steps = cfg.subsampling_factor.ilog2() as usize;
        // module indices: 0 conv, 1 relu, then (depthwise, pointwise, relu) per further step
        let pairs = (1..steps)
            .map(|i| {
                let depthwise = Conv2dConfig {
                    groups: ch,
                    ..strided
                };
                Ok((
                    conv2d(ch, ch, k, depthwise, layers.pp(3 * i - 1))?,
                    conv2d(ch, ch, 1, Default::default(), layers.pp(3 * i))?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let freq = cfg.num_mel_bins / cfg.subsampling_conv_stride.pow(steps as u32);
        let linear = linear(ch * freq, cfg.hidden_size, vb.pp("linear"))?;
        Ok(Self {
            first,
            pairs,
            linear,
        })
    }

    /// `(1, frames, mels)` to `(1, frames / factor, hidden)`.
    fn forward(&self, features: &Tensor) -> Result<Tensor> {
        let mut xs = self.first.forward(&features.unsqueeze(1)?)?.relu()?;
        for (depthwise, pointwise) in &self.pairs {
            xs = pointwise.forward(&depthwise.forward(&xs)?)?.relu()?;
        }
        let (b, c, t, f) = xs.dims4()?;
        let xs = xs.transpose(1, 2)?.reshape((b, t, c * f))?;
        self.linear.forward(&xs)
    }
}

struct FeedForward {
    up: Linear,
    down: Linear,
}

impl FeedForward {
    fn new(cfg: &EncoderConfig, vb: VarBuilder) -> Result<Self> {
        let bias = cfg.attention_bias;
        Ok(Self {
            up: linear_maybe_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                bias,
                vb.pp("linear1"),
            )?,
            down: linear_maybe_bias(
                cfg.intermediate_size,
                cfg.hidden_size,
                bias,
                vb.pp("linear2"),
            )?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.down.forward(&ops::silu(&self.up.forward(xs)?)?)
    }
}

// Transformer-XL attention: content scores against the keys with bias_u, position scores against the projected
// relative embeddings with bias_v, shifted so each query lines up with its own offsets
struct RelPositionAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    pos: Linear,
    bias_u: Tensor,
    bias_v: Tensor,
    heads: usize,
    head_dim: usize,
}

impl RelPositionAttention {
    fn new(cfg: &EncoderConfig, vb: VarBuilder) -> Result<Self> {
        let (h, bias) = (cfg.hidden_size, cfg.attention_bias);
        let heads = cfg.num_attention_heads;
        let head_dim = h / heads;
        Ok(Self {
            q: linear_maybe_bias(h, h, bias, vb.pp("q_proj"))?,
            k: linear_maybe_bias(h, h, bias, vb.pp("k_proj"))?,
            v: linear_maybe_bias(h, h, bias, vb.pp("v_proj"))?,
            o: linear_maybe_bias(h, h, bias, vb.pp("o_proj"))?,
            pos: linear_maybe_bias(h, h, false, vb.pp("relative_k_proj"))?,
            bias_u: vb
                .get((heads, head_dim), "bias_u")?
                .reshape((1, heads, 1, head_dim))?,
            bias_v: vb
                .get((heads, head_dim), "bias_v")?
                .reshape((1, heads, 1, head_dim))?,
            heads,
            head_dim,
        })
    }

    fn heads(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, t, _) = xs.dims3()?;
        // contiguous per head, so a transposed key is a layout the GPU's batched matmul takes
        xs.reshape((b, t, self.heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
    }

    /// `positions` is `(1, 2T - 1, hidden)`, offsets T-1 down to -(T-1).
    fn forward(&self, xs: &Tensor, positions: &Tensor) -> Result<Tensor> {
        let (b, t, _) = xs.dims3()?;
        let scale = (self.head_dim as f64).powf(-0.5);
        let q = self.heads(&self.q.forward(xs)?)?;
        let k = self.heads(&self.k.forward(xs)?)?;
        let v = self.heads(&self.v.forward(xs)?)?;
        let p = self.heads(&self.pos.forward(positions)?)?;
        let q_content = q.broadcast_add(&self.bias_u)?;
        let q_position = q.broadcast_add(&self.bias_v)?;
        // query blocks bound the score tensors on long audio; rows a..b read the position rows from T - b on
        let mut blocks = Vec::new();
        for start in (0..t).step_by(QUERY_BLOCK) {
            let rows = QUERY_BLOCK.min(t - start);
            let content = q_content.narrow(2, start, rows)?.matmul(&k.t()?)?;
            let window = p.narrow(2, t - start - rows, t + rows - 1)?;
            let position = q_position.narrow(2, start, rows)?.matmul(&window.t()?)?;
            let scores = ((content + rel_shift(&position, t)?)? * scale)?;
            blocks.push(ops::softmax_last_dim(&scores)?.matmul(&v)?);
        }
        let attn = Tensor::cat(&blocks, 2)?;
        let attn = attn
            .transpose(1, 2)?
            .reshape((b, t, self.heads * self.head_dim))?;
        self.o.forward(&attn)
    }
}

// a block's (Q, T+Q-1) offset scores to (Q, T) key scores, row r at column Q-1-r+j: Transformer-XL's pad-and-reshape
fn rel_shift(scores: &Tensor, t: usize) -> Result<Tensor> {
    let (b, h, q, p) = scores.dims4()?;
    scores
        .pad_with_zeros(D::Minus1, 1, 0)?
        .reshape((b, h, p + 1, q))?
        .narrow(2, 1, p)?
        .reshape((b, h, q, p))?
        .narrow(3, 0, t)
}

// pointwise, GLU, depthwise (with the batch norm folded in), SiLU, pointwise
struct ConvModule {
    pointwise1: Linear,
    depthwise: Tensor,
    depthwise_bias: Tensor,
    pointwise2: Linear,
    kernel: usize,
}

impl ConvModule {
    fn new(cfg: &EncoderConfig, vb: VarBuilder) -> Result<Self> {
        let (h, k, bias) = (cfg.hidden_size, cfg.conv_kernel_size, cfg.convolution_bias);
        let pointwise = |i: usize, o: usize, name: &str| -> Result<Linear> {
            let w = vb.pp(name).get((o, i, 1), "weight")?.squeeze(2)?;
            let b = bias.then(|| vb.pp(name).get(o, "bias")).transpose()?;
            Ok(Linear::new(w, b))
        };
        let pointwise1 = pointwise(h, 2 * h, "pointwise_conv1")?;
        let pointwise2 = pointwise(h, h, "pointwise_conv2")?;
        let dw = vb
            .pp("depthwise_conv")
            .get((h, 1, k), "weight")?
            .squeeze(1)?;
        let dw_bias = if bias {
            vb.pp("depthwise_conv").get(h, "bias")?
        } else {
            Tensor::zeros(h, dw.dtype(), dw.device())?
        };
        let norm = vb.pp("norm");
        let gamma = norm.get(h, "weight")?;
        let beta = norm.get(h, "bias")?;
        let mean = norm.get(h, "running_mean")?;
        let var = norm.get(h, "running_var")?;
        let inv_std = (var + BATCH_NORM_EPS)?.sqrt()?.recip()?;
        let gain = (gamma * inv_std)?;
        let depthwise = dw.broadcast_mul(&gain.unsqueeze(1)?)?;
        let depthwise_bias = (((dw_bias - mean)? * gain)? + beta)?;
        Ok(Self {
            pointwise1,
            depthwise,
            depthwise_bias,
            pointwise2,
            kernel: k,
        })
    }

    /// `(B, T, C)` in and out.
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gated = self.pointwise1.forward(xs)?;
        let c = gated.dim(D::Minus1)? / 2;
        let xs =
            (gated.narrow(D::Minus1, 0, c)? * ops::sigmoid(&gated.narrow(D::Minus1, c, c)?)?)?;
        let t = xs.dim(1)?;
        let half = (self.kernel - 1) / 2;
        let padded = xs.pad_with_zeros(1, half, half)?;
        // a depthwise conv as kernel-many shifted products: one grouped conv per channel is slow off CUDA
        let mut acc = self
            .depthwise_bias
            .reshape((1, 1, c))?
            .broadcast_as(xs.shape())?
            .contiguous()?;
        for k in 0..self.kernel {
            let tap = self.depthwise.narrow(1, k, 1)?.reshape((1, 1, c))?;
            acc = (acc + padded.narrow(1, k, t)?.broadcast_mul(&tap)?)?;
        }
        self.pointwise2.forward(&ops::silu(&acc)?)
    }
}

struct Block {
    norm_ff1: LayerNorm,
    ff1: FeedForward,
    norm_attn: LayerNorm,
    attn: RelPositionAttention,
    norm_conv: LayerNorm,
    conv: ConvModule,
    norm_ff2: LayerNorm,
    ff2: FeedForward,
    norm_out: LayerNorm,
}

impl Block {
    fn new(cfg: &EncoderConfig, vb: VarBuilder) -> Result<Self> {
        let norm = |name: &str| layer_norm(cfg.hidden_size, LAYER_NORM_EPS, vb.pp(name));
        Ok(Self {
            norm_ff1: norm("norm_feed_forward1")?,
            ff1: FeedForward::new(cfg, vb.pp("feed_forward1"))?,
            norm_attn: norm("norm_self_att")?,
            attn: RelPositionAttention::new(cfg, vb.pp("self_attn"))?,
            norm_conv: norm("norm_conv")?,
            conv: ConvModule::new(cfg, vb.pp("conv"))?,
            norm_ff2: norm("norm_feed_forward2")?,
            ff2: FeedForward::new(cfg, vb.pp("feed_forward2"))?,
            norm_out: norm("norm_out")?,
        })
    }

    fn forward(&self, xs: &Tensor, positions: &Tensor) -> Result<Tensor> {
        let xs = (xs + (self.ff1.forward(&self.norm_ff1.forward(xs)?)? * FEED_FORWARD_WEIGHT)?)?;
        let xs = (&xs
            + self
                .attn
                .forward(&self.norm_attn.forward(&xs)?, positions)?)?;
        let xs = (&xs + self.conv.forward(&self.norm_conv.forward(&xs)?)?)?;
        let xs = (&xs + (self.ff2.forward(&self.norm_ff2.forward(&xs)?)? * FEED_FORWARD_WEIGHT)?)?;
        self.norm_out.forward(&xs)
    }
}

/// The FastConformer encoder: `(1, frames, mels)` features to `(1, frames / factor, hidden)`.
pub struct Encoder {
    subsampling: Subsampling,
    blocks: Vec<Block>,
    input_scale: f64,
    hidden: usize,
}

impl Encoder {
    pub fn new(cfg: &EncoderConfig, vb: VarBuilder) -> Result<Self> {
        let blocks = (0..cfg.num_hidden_layers)
            .map(|i| Block::new(cfg, vb.pp("layers").pp(i)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            subsampling: Subsampling::new(cfg, vb.pp("subsampling"))?,
            blocks,
            input_scale: if cfg.scale_input {
                (cfg.hidden_size as f64).sqrt()
            } else {
                1.0
            },
            hidden: cfg.hidden_size,
        })
    }

    pub fn forward(&self, features: &Tensor) -> Result<Tensor> {
        let xs = (self.subsampling.forward(features)? * self.input_scale)?;
        let positions = relative_positions(xs.dim(1)?, self.hidden, xs.dtype(), xs.device())?;
        let mut xs = xs;
        for block in &self.blocks {
            xs = block.forward(&xs, &positions)?;
        }
        Ok(xs)
    }
}

// sinusoids over offsets T-1 down to -(T-1), sin and cos interleaved per frequency
fn relative_positions(t: usize, hidden: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    let half = hidden / 2;
    let mut table = Vec::with_capacity((2 * t - 1) * hidden);
    for i in 0..2 * t - 1 {
        let offset = (t as f64 - 1.0) - i as f64;
        for f in 0..half {
            let angle = offset / POSITION_BASE.powf(2.0 * f as f64 / hidden as f64);
            table.push(angle.sin() as f32);
            table.push(angle.cos() as f32);
        }
    }
    Tensor::from_vec(table, (1, 2 * t - 1, hidden), device)?.to_dtype(dtype)
}
