use inference_tensor::nn::{
    Conv2d, Conv2dConfig, LayerNorm, Linear, Module, VarBuilder, conv2d, layer_norm, linear, ops,
};
use inference_tensor::{D, DType, Device, Result, Tensor};

use super::config::{ChunkedContext, EncoderConfig};

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

// NeMo's dw_striding, each stride-2 conv halving time and frequency; a streaming one pads (k - 1, s - 1), causal
struct Subsampling {
    first: Conv2d,
    pairs: Vec<(Conv2d, Conv2d)>,
    linear: Linear,
    causal_pad: Option<(usize, usize)>,
}

impl Subsampling {
    fn new(cfg: &EncoderConfig, vb: VarBuilder) -> Result<Self> {
        let (k, s, ch) = (
            cfg.subsampling_conv_kernel_size,
            cfg.subsampling_conv_stride,
            cfg.subsampling_conv_channels,
        );
        let causal_pad = cfg.is_streaming().then_some((k - 1, s - 1));
        let strided = Conv2dConfig {
            padding: if causal_pad.is_some() { 0 } else { (k - 1) / 2 },
            stride: s,
            ..Default::default()
        };
        let depthwise = Conv2dConfig {
            groups: ch,
            ..strided
        };
        let steps = cfg.subsampling_factor.ilog2() as usize;
        let (first, pairs) = if causal_pad.is_some() {
            let layers = vb.pp("layers");
            let pairs = (0..steps - 1)
                .map(|i| {
                    let layer = layers.pp(i);
                    Ok((
                        conv2d(ch, ch, k, depthwise, layer.pp("depthwise_conv"))?,
                        conv2d(ch, ch, 1, Default::default(), layer.pp("pointwise_conv"))?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            (conv2d(1, ch, k, strided, vb.pp("conv_in"))?, pairs)
        } else {
            let layers = vb.pp("layers");
            // module indices: 0 conv, 1 relu, then (depthwise, pointwise, relu) per further step
            let pairs = (1..steps)
                .map(|i| {
                    Ok((
                        conv2d(ch, ch, k, depthwise, layers.pp(3 * i - 1))?,
                        conv2d(ch, ch, 1, Default::default(), layers.pp(3 * i))?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            (conv2d(1, ch, k, strided, layers.pp(0))?, pairs)
        };
        let freq = match causal_pad {
            Some((before, after)) => {
                (0..steps).fold(cfg.num_mel_bins, |f, _| (f + before + after - k) / s + 1)
            }
            None => cfg.num_mel_bins / s.pow(steps as u32),
        };
        let linear = linear(ch * freq, cfg.hidden_size, vb.pp("linear"))?;
        Ok(Self {
            first,
            pairs,
            linear,
            causal_pad,
        })
    }

    fn pad(&self, xs: Tensor) -> Result<Tensor> {
        match self.causal_pad {
            Some((before, after)) => xs
                .pad_with_zeros(3, before, after)?
                .pad_with_zeros(2, before, after),
            None => Ok(xs),
        }
    }

    /// `(1, frames, mels)` to `(1, frames / factor, hidden)`.
    fn forward(&self, features: &Tensor) -> Result<Tensor> {
        let mut xs = self
            .first
            .forward(&self.pad(features.unsqueeze(1)?)?)?
            .relu()?;
        for (depthwise, pointwise) in &self.pairs {
            xs = pointwise
                .forward(&depthwise.forward(&self.pad(xs)?)?)?
                .relu()?;
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

/// A block of query rows and the keys it may see: all of them, or a chunked-limited window and its mask.
pub struct QueryBlock {
    start: usize,
    rows: usize,
    keys_from: usize,
    keys: usize,
    // (1, 1, rows, keys) additive, 0 where a key is in reach and -inf where not
    mask: Option<Tensor>,
}

/// Query blocks over `t` frames: full attention, or each block's chunked-limited key window, so attention over
/// long audio grows linearly with it.
pub fn query_blocks(
    t: usize,
    context: Option<ChunkedContext>,
    dtype: DType,
    device: &Device,
) -> Result<Vec<QueryBlock>> {
    let mut blocks = Vec::new();
    for start in (0..t).step_by(QUERY_BLOCK) {
        let rows = QUERY_BLOCK.min(t - start);
        let Some(ctx) = context else {
            blocks.push(QueryBlock {
                start,
                rows,
                keys_from: 0,
                keys: t,
                mask: None,
            });
            continue;
        };
        // a query sees its own chunk and `left / chunk` whole chunks before it, never one after
        let chunk = ctx.right + 1;
        let left_chunks = ctx.left / chunk;
        let keys_from = (start / chunk).saturating_sub(left_chunks) * chunk;
        let keys_to = t.min(((start + rows - 1) / chunk + 1) * chunk);
        let keys = keys_to - keys_from;
        let mut mask = Vec::with_capacity(rows * keys);
        for q in start..start + rows {
            for k in keys_from..keys_to {
                let behind = (q / chunk) as isize - (k / chunk) as isize;
                let seen = (0..=left_chunks as isize).contains(&behind);
                mask.push(if seen { 0f32 } else { f32::NEG_INFINITY });
            }
        }
        let mask = Tensor::from_vec(mask, (1, 1, rows, keys), device)?.to_dtype(dtype)?;
        blocks.push(QueryBlock {
            start,
            rows,
            keys_from,
            keys,
            mask: Some(mask),
        });
    }
    Ok(blocks)
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
    fn forward(&self, xs: &Tensor, positions: &Tensor, blocks: &[QueryBlock]) -> Result<Tensor> {
        let (b, t, _) = xs.dims3()?;
        let scale = (self.head_dim as f64).powf(-0.5);
        let q = self.heads(&self.q.forward(xs)?)?;
        let k = self.heads(&self.k.forward(xs)?)?;
        let v = self.heads(&self.v.forward(xs)?)?;
        let p = self.heads(&self.pos.forward(positions)?)?;
        let q_content = q.broadcast_add(&self.bias_u)?;
        let q_position = q.broadcast_add(&self.bias_v)?;
        // rows a..b over keys c..c+K read position rows from T - b + c: offsets b-1-c down to a-(c+K-1)
        let mut outs = Vec::with_capacity(blocks.len());
        for b in blocks {
            let keys = k.narrow(2, b.keys_from, b.keys)?;
            let values = v.narrow(2, b.keys_from, b.keys)?;
            let content = q_content.narrow(2, b.start, b.rows)?.matmul(&keys.t()?)?;
            let window = p.narrow(2, t - b.start - b.rows + b.keys_from, b.rows + b.keys - 1)?;
            let position = q_position
                .narrow(2, b.start, b.rows)?
                .matmul(&window.t()?)?;
            let mut scores = ((content + rel_shift(&position, b.keys)?)? * scale)?;
            if let Some(mask) = &b.mask {
                scores = scores.broadcast_add(mask)?;
            }
            outs.push(ops::softmax_last_dim(&scores)?.matmul(&values)?);
        }
        let attn = Tensor::cat(&outs, 2)?;
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

// pointwise, GLU, depthwise, norm, SiLU, pointwise: centred with batch norm folded in, or causal with a layer norm
struct ConvModule {
    pointwise1: Linear,
    depthwise: Tensor,
    depthwise_bias: Tensor,
    norm: Option<LayerNorm>,
    pointwise2: Linear,
    kernel: usize,
    causal: bool,
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
        if cfg.is_streaming() {
            return Ok(Self {
                pointwise1,
                depthwise: dw,
                depthwise_bias: dw_bias,
                norm: Some(layer_norm(h, LAYER_NORM_EPS, vb.pp("norm"))?),
                pointwise2,
                kernel: k,
                causal: true,
            });
        }
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
            norm: None,
            pointwise2,
            kernel: k,
            causal: false,
        })
    }

    /// `(B, T, C)` in and out.
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gated = self.pointwise1.forward(xs)?;
        let c = gated.dim(D::Minus1)? / 2;
        let xs =
            (gated.narrow(D::Minus1, 0, c)? * ops::sigmoid(&gated.narrow(D::Minus1, c, c)?)?)?;
        let t = xs.dim(1)?;
        let (before, after) = if self.causal {
            (self.kernel - 1, 0)
        } else {
            let half = (self.kernel - 1) / 2;
            (half, half)
        };
        let padded = xs.pad_with_zeros(1, before, after)?;
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
        if let Some(norm) = &self.norm {
            acc = norm.forward(&acc)?;
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

    fn forward(&self, xs: &Tensor, positions: &Tensor, blocks: &[QueryBlock]) -> Result<Tensor> {
        let xs = (xs + (self.ff1.forward(&self.norm_ff1.forward(xs)?)? * FEED_FORWARD_WEIGHT)?)?;
        let xs = (&xs
            + self
                .attn
                .forward(&self.norm_attn.forward(&xs)?, positions, blocks)?)?;
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
    context: Option<ChunkedContext>,
}

impl Encoder {
    pub fn new(cfg: &EncoderConfig, vb: VarBuilder) -> Result<Self> {
        if cfg.is_streaming() && cfg.chunked_context().is_none() {
            return Err(inference_tensor::Error::Msg(
                "a streaming encoder config needs `sliding_window` and `default_num_lookahead_tokens`".into(),
            ));
        }
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
            context: cfg.chunked_context(),
        })
    }

    pub fn forward(&self, features: &Tensor) -> Result<Tensor> {
        let xs = self.subsample(features)?;
        let positions = relative_positions(xs.dim(1)?, self.hidden, xs.dtype(), xs.device())?;
        self.encode(&xs, &positions)
    }

    /// The pre-encode alone: `(1, frames, mels)` features to `(1, frames / factor, hidden)`, before input scaling.
    pub fn subsample(&self, features: &Tensor) -> Result<Tensor> {
        self.subsampling.forward(features)
    }

    /// The conformer blocks over `subsample`'s output, `positions` being `relative_positions` for its length.
    pub fn encode(&self, xs: &Tensor, positions: &Tensor) -> Result<Tensor> {
        let blocks = query_blocks(xs.dim(1)?, self.context, xs.dtype(), xs.device())?;
        let mut xs = (xs * self.input_scale)?;
        for block in &self.blocks {
            xs = block.forward(&xs, positions, &blocks)?;
        }
        Ok(xs)
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden
    }
}

/// `(1, 2T - 1, hidden)` sinusoids over offsets T-1 down to -(T-1), sin and cos interleaved per frequency; the
/// table for a shorter length is this one's middle `2t - 1` rows.
pub fn relative_positions(
    t: usize,
    hidden: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
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

#[cfg(test)]
mod tests {
    use super::*;

    // a query sees its own 2-frame chunk and the two before it
    const CONTEXT: ChunkedContext = ChunkedContext { left: 4, right: 1 };

    fn seen(block: &QueryBlock, q: usize) -> Result<Vec<usize>> {
        let mask = block
            .mask
            .as_ref()
            .expect("a chunked block")
            .squeeze(0)?
            .squeeze(0)?
            .to_vec2::<f32>()?;
        Ok(mask[q - block.start]
            .iter()
            .enumerate()
            .filter(|(_, m)| **m == 0.)
            .map(|(i, _)| block.keys_from + i)
            .collect())
    }

    #[test]
    fn chunked_blocks_see_their_chunk_and_the_left_context_only() -> Result<()> {
        let blocks = query_blocks(10, Some(CONTEXT), DType::F32, &Device::Cpu)?;
        assert_eq!(blocks.len(), 1);
        assert_eq!(seen(&blocks[0], 5)?, [0, 1, 2, 3, 4, 5]);
        assert_eq!(seen(&blocks[0], 9)?, [4, 5, 6, 7, 8, 9]);
        assert_eq!(seen(&blocks[0], 0)?, [0, 1]);

        // a later block's keys start at its first chunk's left context, not at frame 0
        let t = QUERY_BLOCK + 88;
        let blocks = query_blocks(t, Some(CONTEXT), DType::F32, &Device::Cpu)?;
        let second = &blocks[1];
        assert_eq!(
            (second.start, second.keys_from),
            (QUERY_BLOCK, QUERY_BLOCK - 4)
        );
        assert_eq!(second.keys_from + second.keys, t);
        assert_eq!(
            seen(second, QUERY_BLOCK + 1)?,
            (QUERY_BLOCK - 4..QUERY_BLOCK + 2).collect::<Vec<_>>()
        );

        // a chunk of 3 that neither the block size nor the 5-frame left context divides: one whole chunk back
        let unaligned = ChunkedContext { left: 5, right: 2 };
        let blocks = query_blocks(t, Some(unaligned), DType::F32, &Device::Cpu)?;
        assert_eq!(seen(&blocks[0], 7)?, [3, 4, 5, 6, 7, 8]);
        let second = &blocks[1];
        // the block opens mid-chunk (512 = 3 * 170 + 2), so its keys reach back to the chunk before that one
        assert_eq!(second.keys_from, 3 * 169);
        assert_eq!(
            seen(second, QUERY_BLOCK)?,
            (3 * 169..3 * 171).collect::<Vec<_>>()
        );

        let full = query_blocks(t, None, DType::F32, &Device::Cpu)?;
        assert!(
            full.iter()
                .all(|b| b.keys_from == 0 && b.keys == t && b.mask.is_none())
        );
        Ok(())
    }
}
