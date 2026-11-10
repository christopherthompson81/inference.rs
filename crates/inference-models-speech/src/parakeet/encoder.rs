use inference_nn::attention::fast_exp;
use inference_tensor::cpu::kernels::VecOps;
use inference_tensor::nn::{
    Conv2d, Conv2dConfig, LayerNorm, Linear, Module, VarBuilder, conv2d, layer_norm, linear, ops,
};
use inference_tensor::vulkan::RelPosAttn;
use inference_tensor::{CpuStorage, CustomOp2, D, DType, Device, Layout, Result, Shape, Tensor};
use rayon::prelude::*;

use super::config::{ChunkedContext, EncoderConfig};

const LAYER_NORM_EPS: f64 = 1e-5;
const BATCH_NORM_EPS: f64 = 1e-5;
const POSITION_BASE: f64 = 10000.0;
// the conformer's macaron feed-forwards each contribute half a residual
const FEED_FORWARD_WEIGHT: f64 = 0.5;
// the offline features refuse shorter audio
const MIN_FEATURE_FRAMES: usize = 2;
// attention rows per block, about 40 s of audio at 80 ms a frame
const QUERY_BLOCK: usize = 512;
// feature frames an offline subsampling takes at a time (~10 s): its first conv's output over 10 minutes is 2 GB, and
// the chunks a GPU has in flight each hold their own (a 780M over 10 minutes: 4 GB at 4096 frames, 2.9 GB at 1024, and
// 1.4 GB once a batch also submits by the bytes it allocates)
const SUBSAMPLING_CHUNK: usize = 1024;
// feed-forward rows a GPU takes at a time: over 10 minutes the (t, 4 hidden) projection and its SiLU are 123 MB each
const FEED_FORWARD_ROWS: usize = 2048;

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
    kernel: usize,
    stride: usize,
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
            kernel: k,
            stride: s,
        })
    }

    // a streaming (causal) stride step's frames out for `x` frames in, as `forward` pads them
    fn step_frames(&self, x: usize, (before, after): (usize, usize)) -> usize {
        (x + before + after - self.kernel) / self.stride + 1
    }

    fn steps(&self) -> usize {
        self.pairs.len() + 1
    }

    fn factor(&self) -> usize {
        self.stride.pow(self.steps() as u32)
    }

    // the leading outputs of a window that read its zero padding where the audio before it had frames
    fn streaming_drop(&self, (before, _): (usize, usize)) -> usize {
        (0..self.steps()).fold(0, |valid, _| (valid + before).div_ceil(self.stride))
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
        if self.causal_pad.is_some() || features.dim(1)? <= SUBSAMPLING_CHUNK {
            return self.forward_whole(features);
        }
        self.forward_chunked(features, SUBSAMPLING_CHUNK)
    }

    // the offline (symmetric) subsampling over time chunks of `chunk` frames (a multiple of the factor): an output
    // frame reads the inputs within `reach` of factor times its index, so a window from a multiple of the factor and
    // `halo` frames wider each side computes its chunk's outputs as the whole pass does
    fn forward_chunked(&self, features: &Tensor, chunk: usize) -> Result<Tensor> {
        let (f, frames) = (self.factor(), features.dim(1)?);
        let pad = (self.kernel - 1) / 2;
        let reach = pad * (f - 1) / (self.stride - 1);
        let halo = reach.next_multiple_of(f);
        let total =
            (0..self.steps()).fold(frames, |x, _| (x + 2 * pad - self.kernel) / self.stride + 1);
        let mut outs = Vec::with_capacity(total.div_ceil(chunk / f));
        for o0 in (0..total).step_by(chunk / f) {
            let o1 = (o0 + chunk / f).min(total);
            let start = (o0 * f).saturating_sub(halo);
            let end = (o1 * f + halo).min(frames);
            let window = self.convs(&features.narrow(1, start, end - start)?)?;
            outs.push(window.narrow(1, (o0 * f - start) / f, o1 - o0)?);
        }
        // the projection over every chunk's frames at once: a chunk's ~130 rows had run it at a fifth of the rate
        self.linear.forward(&Tensor::cat(&outs, 1)?)
    }

    fn forward_whole(&self, features: &Tensor) -> Result<Tensor> {
        self.linear.forward(&self.convs(features)?)
    }

    // the convs, to `(1, frames / factor, channels x frequency)` ahead of the projection
    fn convs(&self, features: &Tensor) -> Result<Tensor> {
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
        xs.transpose(1, 2)?.reshape((b, t, c * f))
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
        let t = xs.dim(1)?;
        if xs.device().is_cpu() || t <= FEED_FORWARD_ROWS {
            return self.rows(xs);
        }
        let parts = (0..t)
            .step_by(FEED_FORWARD_ROWS)
            .map(|start| self.rows(&xs.narrow(1, start, FEED_FORWARD_ROWS.min(t - start))?))
            .collect::<Result<Vec<_>>>()?;
        Tensor::cat(&parts, 1)
    }

    fn rows(&self, xs: &Tensor) -> Result<Tensor> {
        let hidden = self.up.forward(xs)?;
        // the CPU's SiLU over the fresh projection in place: a second (t, 4 hidden) buffer was 123 MB over 10 minutes
        if inference_tensor::cpu::silu_in_place(&hidden)? {
            return self.down.forward(&hidden);
        }
        self.down.forward(&ops::silu(&hidden)?)
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
        let (q, k, v) = (
            self.q.forward(xs)?,
            self.k.forward(xs)?,
            self.v.forward(xs)?,
        );
        // rows a..b over keys c..c+K read position rows from T - b + c: offsets b-1-c down to a-(c+K-1)
        let first = |b: &QueryBlock| t - b.start - b.rows + b.keys_from;
        let band = |b: &QueryBlock| first(b)..first(b) + b.rows + b.keys - 1;
        // only the rows some block reads: all 2T-1 for full attention, a band set by the context when chunked
        let lo = blocks.iter().map(|b| band(b).start).min().unwrap_or(0);
        let hi = blocks.iter().map(|b| band(b).end).max().unwrap_or(0);
        let p = self.pos.forward(&positions.narrow(1, lo, hi - lo)?)?;
        let full = blocks
            .iter()
            .all(|b| b.mask.is_none() && b.keys_from == 0 && b.keys == t);
        // on the CPU (bf16 pairs inside the model's bf16 scope, else f32 on AVX2): fused over the projections as they
        // are, biased and scaled as it packs them, so no (rows, keys) scores, per-head copies or scaled queries exist
        if full
            && xs.device().is_cpu()
            && let Some(attn) =
                inference_tensor::cpu::rel_pos_attention(&inference_tensor::cpu::RelPosOperands {
                    q: &q,
                    bias_u: &self.bias_u,
                    bias_v: &self.bias_v,
                    scale: scale as f32,
                    k: &k,
                    v: &v,
                    positions: &p,
                    heads: self.heads,
                    pos_base: t - 1 - lo,
                })?
        {
            return self.o.forward(&attn);
        }
        // each projection moved in and dropped once split: a shadowed one would live beside its copy to the end
        let split = |x: Tensor| self.heads(&x);
        let (q, k, v, p) = (split(q)?, split(k)?, split(v)?, split(p)?);
        // full attention fused on an iGPU that has it: no (rows, keys) scores in memory, so no blocks either; a discrete
        // GPU's matrix cores run the blocks' products faster (a 3090 over 10 minutes: 2.6 s against 7.0 fused)
        if full
            && xs.device().is_vulkan()
            && !inference_tensor::vulkan::is_discrete(xs.device())
            && let Some(attn) = self.fused_f32(&q, &k, &v, &p, scale, t - 1 - lo)?
        {
            let attn = attn
                .transpose(1, 2)?
                .reshape((b, t, self.heads * self.head_dim))?
                .to_dtype(q.dtype())?;
            return self.o.forward(&attn);
        }
        // scaled before the products, so the (rows, keys) scores take no extra pass
        let q_content = (q.broadcast_add(&self.bias_u)? * scale)?;
        let q_position = (q.broadcast_add(&self.bias_v)? * scale)?;
        let mut outs = Vec::with_capacity(blocks.len());
        for b in blocks {
            let keys = k.narrow(2, b.keys_from, b.keys)?;
            let values = v.narrow(2, b.keys_from, b.keys)?;
            let content = q_content.narrow(2, b.start, b.rows)?.matmul(&keys.t()?)?;
            let window = p.narrow(2, first(b) - lo, b.rows + b.keys - 1)?;
            let rows = q_position.narrow(2, b.start, b.rows)?;
            // one position table serves every item of a batch
            let position = if rows.dim(0)? == 1 {
                rows.matmul(&window.t()?)?
            } else {
                rows.broadcast_matmul(&window.t()?)?
            };
            let probs = match &b.mask {
                None if content.device().is_cpu() => {
                    content.apply_op2_no_bwd(&position, &RelShiftSoftmax)?
                }
                mask => {
                    let mut scores = (content + rel_shift(&position, b.keys)?)?;
                    if let Some(mask) = mask {
                        scores = scores.broadcast_add(mask)?;
                    }
                    ops::softmax_last_dim(&scores)?
                }
            };
            outs.push(probs.matmul(&values)?);
        }
        let attn = Tensor::cat(&outs, 2)?;
        let attn = attn
            .transpose(1, 2)?
            .reshape((b, t, self.heads * self.head_dim))?;
        self.o.forward(&attn)
    }
}

impl RelPositionAttention {
    // the fused kernel is f32: a bf16 model widens its (t, hidden) operands, far less than bf16 (rows, keys) scores
    // cost, and the scores keep f32's precision (a 780M over 10 minutes: bf16 asr 16.4 -> 14.6 s, peak 2.57 -> 2.44 GB)
    fn fused_f32(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        positions: &Tensor,
        scale: f64,
        pos_base: usize,
    ) -> Result<Option<Tensor>> {
        let wide = |x: &Tensor| x.to_dtype(DType::F32);
        let q = wide(q)?;
        inference_tensor::vulkan::rel_pos_attn(&RelPosAttn {
            q_content: &(q.broadcast_add(&wide(&self.bias_u)?)? * scale)?,
            q_position: &(q.broadcast_add(&wide(&self.bias_v)?)? * scale)?,
            k: &wide(k)?,
            v: &wide(v)?,
            positions: &wide(positions)?,
            pos_base,
        })
    }

    /// `(1, heads, 2W - 1, head_dim)` projected positions for a streaming window of `w` frames.
    fn window_positions(
        &self,
        w: usize,
        hidden: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let table = relative_positions(w, hidden, dtype, device)?;
        self.heads(&self.pos.forward(&table)?)
    }

    /// One streamed chunk: its rows attend to themselves and the cached keys, all in reach, and the cache keeps
    /// the last `keep` frames.
    fn step(
        &self,
        xs: &Tensor,
        cache: &mut Option<(Tensor, Tensor)>,
        positions: &Tensor,
        keep: usize,
    ) -> Result<Tensor> {
        let (b, rows, _) = xs.dims3()?;
        let scale = (self.head_dim as f64).powf(-0.5);
        let q = self.heads(&self.q.forward(xs)?)?;
        let mut k = self.heads(&self.k.forward(xs)?)?;
        let mut v = self.heads(&self.v.forward(xs)?)?;
        if let Some((past_k, past_v)) = cache.as_ref() {
            k = Tensor::cat(&[past_k, &k], 2)?;
            v = Tensor::cat(&[past_v, &v], 2)?;
        }
        let keys = k.dim(2)?;
        let content = (q.broadcast_add(&self.bias_u)? * scale)?.matmul(&k.t()?)?;
        // the table's rows run from offset W-1 down; this window needs keys-1 down to -(rows-1)
        let w = positions.dim(2)?.div_ceil(2);
        let window = positions.narrow(2, w - keys, keys + rows - 1)?;
        let position = (q.broadcast_add(&self.bias_v)? * scale)?.matmul(&window.t()?)?;
        let scores = (content + rel_shift(&position, keys)?)?;
        let attn = ops::softmax_last_dim(&scores)?.matmul(&v)?;
        let kept = keep.min(keys);
        *cache = Some((
            k.narrow(2, keys - kept, kept)?.contiguous()?,
            v.narrow(2, keys - kept, kept)?.contiguous()?,
        ));
        let attn = attn
            .transpose(1, 2)?
            .reshape((b, rows, self.heads * self.head_dim))?;
        self.o.forward(&attn)
    }
}

// (Q, T+Q-1) offset scores to (Q, T) key scores: (r, j) is flat element Q-1 + r(P-1) + j, a view plus one copy
// the softmax of each row of (rows, keys) content scores plus its rel-shifted position scores (row r reads columns
// rows - 1 - r on), in one pass on the CPU, as `softmax_last_dim(content + rel_shift(position, keys))` computes it
// but for its exp: the sum and the shift's strided read had each taken a pass over every head's scores
struct RelShiftSoftmax;

impl CustomOp2 for RelShiftSoftmax {
    fn name(&self) -> &'static str {
        "rel-shift-softmax"
    }

    fn cpu_fwd(
        &self,
        content: &CpuStorage,
        content_l: &Layout,
        position: &CpuStorage,
        position_l: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (CpuStorage::F32(content), CpuStorage::F32(position)) = (content, position) else {
            inference_tensor::bail!("rel-shift-softmax takes f32 scores")
        };
        let (Some((c0, c1)), Some((p0, p1))) = (
            content_l.contiguous_offsets(),
            position_l.contiguous_offsets(),
        ) else {
            inference_tensor::bail!("rel-shift-softmax takes contiguous scores")
        };
        let (content, position) = (&content[c0..c1], &position[p0..p1]);
        let dims = content_l.dims();
        let (rows, keys) = (dims[dims.len() - 2], dims[dims.len() - 1]);
        let wide = position_l.dims()[dims.len() - 1];
        if wide + 1 < rows + keys {
            inference_tensor::bail!("{wide} position columns for {rows} rows over {keys} keys")
        }
        let mut out = Vec::with_capacity(content.len());
        // SAFETY: f32 takes any bit pattern, and every row is written below before it is read
        #[allow(clippy::uninit_vec)]
        unsafe {
            out.set_len(content.len())
        };
        out.par_chunks_mut(keys).enumerate().for_each(|(row, dst)| {
            let shift = rows - 1 - row % rows;
            let pos = &position[row * wide + shift..][..keys];
            for ((d, c), p) in dst.iter_mut().zip(&content[row * keys..]).zip(pos) {
                *d = c + p;
            }
            let mut max = f32::NEG_INFINITY;
            // SAFETY: `dst` holds `keys` values
            unsafe { f32::vec_reduce_max(dst.as_ptr(), &mut max, keys) };
            // Cephes' exp (~1e-7 relative) vectorizes where libm's is a call an element: a tenth of the encoder
            for d in dst.iter_mut() {
                *d = fast_exp(*d - max);
            }
            let mut sum = 0f32;
            // SAFETY: as above
            unsafe { f32::vec_reduce_sum(dst.as_ptr(), &mut sum, keys) };
            for d in dst.iter_mut() {
                *d /= sum;
            }
        });
        Ok((CpuStorage::F32(out), Shape::from_dims(dims)))
    }
}

fn rel_shift(scores: &Tensor, t: usize) -> Result<Tensor> {
    let (b, h, q, p) = scores.dims4()?;
    // a lone row reads columns 0..T, one wider than the (P-1)-wide rows of the view
    if q == 1 {
        return scores.narrow(3, 0, t);
    }
    scores
        .flatten_from(2)?
        .narrow(2, q - 1, q * (p - 1))?
        .reshape((b, h, q, p - 1))?
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
        // llama.cpp's converter folds the running statistics into the scale and shift
        let (gain, shift) = match norm.contains_tensor("running_mean") {
            true => {
                let mean = norm.get(h, "running_mean")?;
                let var = norm.get(h, "running_var")?;
                let gain = (gamma * (var + BATCH_NORM_EPS)?.sqrt()?.recip()?)?;
                (gain.clone(), (beta - (mean * gain)?)?)
            }
            false => (gamma, beta),
        };
        let depthwise = dw.broadcast_mul(&gain.unsqueeze(1)?)?;
        let depthwise_bias = ((dw_bias * gain)? + shift)?;
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

    fn gate(&self, xs: &Tensor) -> Result<Tensor> {
        let gated = self.pointwise1.forward(xs)?;
        let c = gated.dim(D::Minus1)? / 2;
        gated.narrow(D::Minus1, 0, c)? * ops::sigmoid(&gated.narrow(D::Minus1, c, c)?)?
    }

    /// `(B, T, C)` in and out; `frames` `(B, T, 1)` is 1 on each item's own frames, 0 on a batch's padding.
    fn forward(&self, xs: &Tensor, frames: Option<&Tensor>) -> Result<Tensor> {
        let mut xs = self.gate(xs)?;
        // the padding reads as the zeros a lone window's edge pads with
        if let Some(frames) = frames {
            xs = xs.broadcast_mul(frames)?;
        }
        let (before, after) = if self.causal {
            (self.kernel - 1, 0)
        } else {
            let half = (self.kernel - 1) / 2;
            (half, half)
        };
        let t = xs.dim(1)?;
        self.depthwise_tail(&xs.pad_with_zeros(1, before, after)?, t)
    }

    /// A causal module over one streamed chunk: `cache` holds the last `kernel - 1` gated frames before it,
    /// zeros at the start as the offline pass pads.
    fn step(&self, xs: &Tensor, cache: &mut Option<Tensor>) -> Result<Tensor> {
        let xs = self.gate(xs)?;
        let t = xs.dim(1)?;
        let past = match cache.take() {
            Some(past) => past,
            None => Tensor::zeros(
                (xs.dim(0)?, self.kernel - 1, xs.dim(2)?),
                xs.dtype(),
                xs.device(),
            )?,
        };
        let padded = Tensor::cat(&[&past, &xs], 1)?;
        *cache = Some(padded.narrow(1, t, self.kernel - 1)?.contiguous()?);
        self.depthwise_tail(&padded, t)
    }

    // the depthwise conv over `padded` (`t + kernel - 1` frames), then norm, SiLU and the last pointwise
    fn depthwise_tail(&self, padded: &Tensor, t: usize) -> Result<Tensor> {
        let c = padded.dim(D::Minus1)?;
        let xs_shape = (padded.dim(0)?, t, c);
        // a depthwise conv as kernel-many shifted products, reading (B, T, C) as it lies rather than channel-major
        let mut acc = self
            .depthwise_bias
            .reshape((1, 1, c))?
            .broadcast_as(xs_shape)?
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

    // each residual replaces the last: shadowing ones kept all five (t, hidden) copies to the end of the block
    fn forward(
        &self,
        xs: Tensor,
        positions: &Tensor,
        blocks: &[QueryBlock],
        frames: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut xs =
            (&xs + (self.ff1.forward(&self.norm_ff1.forward(&xs)?)? * FEED_FORWARD_WEIGHT)?)?;
        xs = (&xs
            + self
                .attn
                .forward(&self.norm_attn.forward(&xs)?, positions, blocks)?)?;
        xs = (&xs + self.conv.forward(&self.norm_conv.forward(&xs)?, frames)?)?;
        xs = (&xs + (self.ff2.forward(&self.norm_ff2.forward(&xs)?)? * FEED_FORWARD_WEIGHT)?)?;
        self.norm_out.forward(&xs)
    }
}

impl Block {
    fn step(
        &self,
        xs: &Tensor,
        state: &mut LayerState,
        positions: &Tensor,
        keep: usize,
    ) -> Result<Tensor> {
        let xs = (xs + (self.ff1.forward(&self.norm_ff1.forward(xs)?)? * FEED_FORWARD_WEIGHT)?)?;
        let xs = (&xs
            + self.attn.step(
                &self.norm_attn.forward(&xs)?,
                &mut state.attention,
                positions,
                keep,
            )?)?;
        let xs = (&xs
            + self
                .conv
                .step(&self.norm_conv.forward(&xs)?, &mut state.conv)?)?;
        let xs = (&xs + (self.ff2.forward(&self.norm_ff2.forward(&xs)?)? * FEED_FORWARD_WEIGHT)?)?;
        self.norm_out.forward(&xs)
    }
}

#[derive(Default)]
struct LayerState {
    attention: Option<(Tensor, Tensor)>,
    conv: Option<Tensor>,
}

/// A streamed encoder's carried state: each layer's cached keys, values and conv frames, the mel frames the next
/// chunk's subsampling window reaches back over, and how far the stream has got.
pub struct EncoderStream {
    layers: Vec<LayerState>,
    positions: Vec<Tensor>,
    // row-major mel frames from global frame `mel_start` on
    mel: Vec<f32>,
    mel_start: usize,
    mel_frames: usize,
    emitted: usize,
}

/// The FastConformer encoder: `(1, frames, mels)` features to `(1, frames / factor, hidden)`.
pub struct Encoder {
    subsampling: Subsampling,
    blocks: Vec<Block>,
    input_scale: f64,
    hidden: usize,
    context: Option<ChunkedContext>,
}

// `layer` with its weight kept in f16 where `Linear::to_half` keeps it
fn halve(layer: &mut Linear) -> Result<()> {
    *layer = layer.to_half()?;
    Ok(())
}

impl Encoder {
    /// Its projections' weights kept in f16 where the device's matrix cores take them (`Linear::to_half`).
    pub fn halve_weights(&mut self) -> Result<()> {
        halve(&mut self.subsampling.linear)?;
        for block in &mut self.blocks {
            let attn = &mut block.attn;
            for layer in [
                &mut block.ff1.up,
                &mut block.ff1.down,
                &mut attn.q,
                &mut attn.k,
                &mut attn.v,
                &mut attn.o,
                &mut attn.pos,
                &mut block.conv.pointwise1,
                &mut block.conv.pointwise2,
                &mut block.ff2.up,
                &mut block.ff2.down,
            ] {
                halve(layer)?;
            }
        }
        Ok(())
    }
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

    // the features and the subsampled frames go once used, rather than live through every block
    pub fn forward(&self, features: Tensor) -> Result<Tensor> {
        let xs = self.subsample(&features)?;
        drop(features);
        let positions = relative_positions(xs.dim(1)?, self.hidden, xs.dtype(), xs.device())?;
        self.encode_owned(xs, &positions)
    }

    /// The pre-encode alone: `(1, frames, mels)` features to `(1, frames / factor, hidden)`, before input scaling.
    pub fn subsample(&self, features: &Tensor) -> Result<Tensor> {
        self.subsampling.forward(features)
    }

    /// The conformer blocks over `subsample`'s output, `positions` being `relative_positions` for its length.
    pub fn encode(&self, xs: &Tensor, positions: &Tensor) -> Result<Tensor> {
        self.encode_owned(xs.clone(), positions)
    }

    fn encode_owned(&self, xs: Tensor, positions: &Tensor) -> Result<Tensor> {
        let blocks = query_blocks(xs.dim(1)?, self.context, xs.dtype(), xs.device())?;
        let mut xs = if self.input_scale == 1.0 {
            xs
        } else {
            (xs * self.input_scale)?
        };
        for block in &self.blocks {
            xs = block.forward(xs, positions, &blocks, None)?;
        }
        Ok(xs)
    }

    /// `forward` over several `(1, frames, mels)` windows at once, each output as `forward` gives it alone: the
    /// subsampled windows padded to the longest, the padding masked from attention and zeroed ahead of each conv.
    /// Offline encoders only.
    pub fn forward_batch(&self, features: &[Tensor]) -> Result<Vec<Tensor>> {
        if self.context.is_some() || self.subsampling.causal_pad.is_some() {
            return Err(inference_tensor::Error::Msg(
                "a streaming encoder's windows run one at a time".into(),
            ));
        }
        let parts = features
            .iter()
            .map(|f| self.subsample(f))
            .collect::<Result<Vec<_>>>()?;
        let lengths = parts.iter().map(|p| p.dim(1)).collect::<Result<Vec<_>>>()?;
        let longest = lengths.iter().copied().max().unwrap_or(0);
        let Some(first) = parts.first() else {
            return Ok(Vec::new());
        };
        let (dtype, device) = (first.dtype(), first.device().clone());
        let padded = parts
            .into_iter()
            .zip(&lengths)
            .map(|(p, &len)| p.pad_with_zeros(1, 0, longest - len))
            .collect::<Result<Vec<_>>>()?;
        let xs = Tensor::cat(&padded, 0)?;
        drop(padded);
        let batch = lengths.len();
        let own = |len: usize, on: f32, off: f32| {
            (0..longest).map(move |t| if t < len { on } else { off })
        };
        let frames: Vec<f32> = lengths.iter().flat_map(|&len| own(len, 1.0, 0.0)).collect();
        let frames = Tensor::from_vec(frames, (batch, longest, 1), &device)?.to_dtype(dtype)?;
        let keys: Vec<f32> = lengths
            .iter()
            .flat_map(|&len| own(len, 0.0, f32::NEG_INFINITY))
            .collect();
        let keys = Tensor::from_vec(keys, (batch, 1, 1, longest), &device)?.to_dtype(dtype)?;
        let blocks: Vec<QueryBlock> = query_blocks(longest, None, dtype, &device)?
            .into_iter()
            .map(|b| QueryBlock {
                mask: Some(keys.clone()),
                ..b
            })
            .collect();
        let positions = relative_positions(longest, self.hidden, dtype, &device)?;
        let mut xs = if self.input_scale == 1.0 {
            xs
        } else {
            (xs * self.input_scale)?
        };
        for block in &self.blocks {
            xs = block.forward(xs, &positions, &blocks, Some(&frames))?;
        }
        lengths
            .iter()
            .enumerate()
            .map(|(i, &len)| xs.narrow(0, i, 1)?.narrow(1, 0, len))
            .collect()
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden
    }

    pub fn can_stream(&self) -> bool {
        self.context.is_some()
    }

    // the streaming chunk and the left context it keeps, in encoder frames, and the causal subsampling pad
    fn stream_geometry(&self) -> Result<(usize, usize, (usize, usize))> {
        let (Some(ctx), Some(pad)) = (self.context, self.subsampling.causal_pad) else {
            return Err(inference_tensor::Error::Msg(
                "only a cache-aware streaming encoder (chunked-limited attention) can stream"
                    .into(),
            ));
        };
        let chunk = ctx.right + 1;
        Ok((chunk, ctx.left / chunk * chunk, pad))
    }

    /// A fresh stream, its encoder frames equal to `forward`'s over the whole audio.
    pub fn stream(&self, dtype: DType, device: &Device) -> Result<EncoderStream> {
        let (chunk, keep, _) = self.stream_geometry()?;
        let positions = self
            .blocks
            .iter()
            .map(|b| {
                b.attn
                    .window_positions(keep + chunk, self.hidden, dtype, device)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(EncoderStream {
            layers: self.blocks.iter().map(|_| LayerState::default()).collect(),
            positions,
            mel: Vec::new(),
            mel_start: 0,
            mel_frames: 0,
            emitted: 0,
        })
    }

    /// Appends `(frames, mels)` row-major features; returns each whole chunk they complete, `(1, frames, hidden)`,
    /// with the index of its first frame.
    pub fn stream_push(
        &self,
        stream: &mut EncoderStream,
        mel: &[f32],
        mels: usize,
    ) -> Result<Vec<(usize, Tensor)>> {
        stream.mel.extend_from_slice(mel);
        stream.mel_frames += mel.len() / mels;
        let (chunk, _, _) = self.stream_geometry()?;
        let factor = self.subsampling.factor();
        let mut outs = Vec::new();
        // a chunk's last frame reads mel frames up to `factor` times its index
        while stream.mel_frames > factor * (stream.emitted + chunk - 1) {
            let end = factor * (stream.emitted + chunk - 1) + 1;
            outs.push(self.stream_chunk(stream, end, chunk, mels)?);
        }
        Ok(outs)
    }

    /// The chunks left once the audio ends, as `stream_push` returns them; too little audio fails as offline.
    pub fn stream_finish(
        &self,
        stream: &mut EncoderStream,
        mels: usize,
    ) -> Result<Vec<(usize, Tensor)>> {
        let (chunk, _, pad) = self.stream_geometry()?;
        if stream.mel_frames < MIN_FEATURE_FRAMES {
            return Err(inference_tensor::Error::Msg(format!(
                "audio is too short: it gives {} feature frames",
                stream.mel_frames
            )));
        }
        let total = (0..self.subsampling.steps()).fold(stream.mel_frames, |x, _| {
            self.subsampling.step_frames(x, pad)
        });
        let mut outs = Vec::new();
        while stream.emitted < total {
            let rows = chunk.min(total - stream.emitted);
            outs.push(self.stream_chunk(stream, stream.mel_frames, rows, mels)?);
        }
        Ok(outs)
    }

    // the next chunk's `rows` frames from a subsampling window of mel frames up to (not including) `end`
    fn stream_chunk(
        &self,
        stream: &mut EncoderStream,
        end: usize,
        rows: usize,
        mels: usize,
    ) -> Result<(usize, Tensor)> {
        let (_, keep, pad) = self.stream_geometry()?;
        let factor = self.subsampling.factor();
        let f0 = stream.emitted;
        // a window from frame 0 reads the same zero pad the offline pass does, so needs no drop
        let drop = self.subsampling.streaming_drop(pad).min(f0);
        let start = factor * (f0 - drop);
        let window =
            &stream.mel[(start - stream.mel_start) * mels..(end - stream.mel_start) * mels];
        let device = stream.positions[0].device();
        let features = Tensor::from_slice(window, (1, end - start, mels), device)?
            .to_dtype(stream.positions[0].dtype())?;
        let mut xs = (self.subsample(&features)?.narrow(1, drop, rows)? * self.input_scale)?;
        for ((block, state), positions) in self
            .blocks
            .iter()
            .zip(stream.layers.iter_mut())
            .zip(&stream.positions)
        {
            xs = block.step(&xs, state, positions, keep)?;
        }
        stream.emitted += rows;
        let next = stream.emitted;
        let next_start = factor * (next - self.subsampling.streaming_drop(pad).min(next));
        let cut = next_start - stream.mel_start;
        stream.mel.drain(..cut * mels);
        stream.mel_start = next_start;
        Ok((f0, xs))
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
    // a row a thread: a long recording's table is millions of sinusoids, ahead of every layer
    let divisors: Vec<f64> = (0..hidden / 2)
        .map(|f| POSITION_BASE.powf(2.0 * f as f64 / hidden as f64))
        .collect();
    let mut table = vec![0f32; (2 * t - 1) * hidden];
    table
        .par_chunks_mut(hidden)
        .enumerate()
        .for_each(|(i, row)| {
            let offset = (t as f64 - 1.0) - i as f64;
            for (pair, divisor) in row.as_chunks_mut::<2>().0.iter_mut().zip(&divisors) {
                let (sin, cos) = (offset / divisor).sin_cos();
                pair[0] = sin as f32;
                pair[1] = cos as f32;
            }
        });
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

    // llama.cpp's GGUF stores the conv's batch norm folded into a scale and shift; both forms compute the same
    // the fused CPU softmax gives what the shift, the sum and the softmax give apart, to its exp's rounding
    #[test]
    fn the_fused_rel_shift_softmax_matches_its_composite() -> Result<()> {
        const FAST_EXP_TOLERANCE: f32 = 1e-6;
        let (heads, rows, keys) = (2, 5, 9);
        let content = Tensor::randn(0f32, 2., (1, heads, rows, keys), &Device::Cpu)?;
        let position = Tensor::randn(0f32, 2., (1, heads, rows, rows + keys - 1), &Device::Cpu)?;
        let fused = content.apply_op2_no_bwd(&position, &RelShiftSoftmax)?;
        let apart = ops::softmax_last_dim(&(&content + rel_shift(&position, keys)?)?)?;
        let off = (fused - apart)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(off < FAST_EXP_TOLERANCE, "{off}");
        Ok(())
    }

    // the offline subsampling over time chunks gives the whole pass's frames (to the CPU gemm's summation order, which
    // follows the matrix's size): three stride-2 steps, a partial last chunk
    #[test]
    fn a_chunked_subsampling_matches_the_whole_pass() -> Result<()> {
        const CHUNKED_TOLERANCE: f32 = 1e-5;
        let (ch, mels, hidden) = (4, 16, 8);
        let cfg: EncoderConfig = serde_json::from_value(serde_json::json!({
            "hidden_size": hidden, "intermediate_size": 8, "num_hidden_layers": 1, "num_attention_heads": 1,
            "num_mel_bins": mels, "conv_kernel_size": 3, "subsampling_conv_channels": ch,
            "subsampling_conv_kernel_size": 3, "subsampling_conv_stride": 2, "subsampling_factor": 8,
        }))
        .map_err(|e| inference_tensor::Error::Msg(e.to_string()))?;
        let dev = Device::Cpu;
        let mut weights = std::collections::HashMap::new();
        let mut add = |name: &str, shape: &[usize]| -> Result<()> {
            weights.insert(name.to_string(), Tensor::randn(0f32, 0.5, shape, &dev)?);
            Ok(())
        };
        add("layers.0.weight", &[ch, 1, 3, 3])?;
        add("layers.0.bias", &[ch])?;
        for (dw, pw) in [(2, 3), (5, 6)] {
            add(&format!("layers.{dw}.weight"), &[ch, 1, 3, 3])?;
            add(&format!("layers.{dw}.bias"), &[ch])?;
            add(&format!("layers.{pw}.weight"), &[ch, ch, 1, 1])?;
            add(&format!("layers.{pw}.bias"), &[ch])?;
        }
        add("linear.weight", &[hidden, ch * mels / 8])?;
        add("linear.bias", &[hidden])?;
        let sub = Subsampling::new(&cfg, VarBuilder::from_tensors(weights, DType::F32, &dev))?;
        let features = Tensor::randn(0f32, 1., (1, 300, mels), &dev)?;
        let whole = sub.forward_whole(&features)?;
        let chunked = sub.forward_chunked(&features, 64)?;
        assert_eq!(chunked.dims(), whole.dims());
        let off = (chunked - whole)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(off < CHUNKED_TOLERANCE, "{off}");
        Ok(())
    }

    #[test]
    fn a_folded_conv_norm_matches_its_running_statistics() -> Result<()> {
        let (h, k) = (4, 3);
        let cfg: EncoderConfig = serde_json::from_value(serde_json::json!({
            "hidden_size": h, "intermediate_size": 8, "num_hidden_layers": 1, "num_attention_heads": 1,
            "num_mel_bins": 8, "conv_kernel_size": k, "subsampling_conv_channels": 2,
            "subsampling_conv_kernel_size": 3, "subsampling_conv_stride": 2, "subsampling_factor": 2,
            "convolution_bias": true,
        }))
        .map_err(|e| inference_tensor::Error::Msg(e.to_string()))?;
        let dev = Device::Cpu;
        let values = |n: usize, seed: f32| {
            (0..n)
                .map(|i| (i as f32 * 0.7 + seed).sin())
                .collect::<Vec<_>>()
        };
        let t =
            |n: usize, seed: f32, shape: &[usize]| Tensor::from_vec(values(n, seed), shape, &dev);
        let mut shared = std::collections::HashMap::new();
        shared.insert(
            "pointwise_conv1.weight".to_string(),
            t(2 * h * h, 0.1, &[2 * h, h, 1])?,
        );
        shared.insert("pointwise_conv1.bias".to_string(), t(2 * h, 0.2, &[2 * h])?);
        shared.insert(
            "pointwise_conv2.weight".to_string(),
            t(h * h, 0.3, &[h, h, 1])?,
        );
        shared.insert("pointwise_conv2.bias".to_string(), t(h, 0.4, &[h])?);
        shared.insert(
            "depthwise_conv.weight".to_string(),
            t(h * k, 0.5, &[h, 1, k])?,
        );
        shared.insert("depthwise_conv.bias".to_string(), t(h, 0.6, &[h])?);
        let (gamma, beta) = (t(h, 0.7, &[h])?, t(h, 0.8, &[h])?);
        let mean = t(h, 0.9, &[h])?;
        let var = (t(h, 1.0, &[h])?.abs()? + 0.5)?;
        let mut stats = shared.clone();
        stats.insert("norm.weight".to_string(), gamma.clone());
        stats.insert("norm.bias".to_string(), beta.clone());
        stats.insert("norm.running_mean".to_string(), mean.clone());
        stats.insert("norm.running_var".to_string(), var.clone());
        let gain = (gamma * (var + BATCH_NORM_EPS)?.sqrt()?.recip()?)?;
        let mut folded = shared;
        folded.insert("norm.bias".to_string(), (beta - (mean * &gain)?)?);
        folded.insert("norm.weight".to_string(), gain);
        let module =
            |weights| ConvModule::new(&cfg, VarBuilder::from_tensors(weights, DType::F32, &dev));
        let xs = t(5 * h, 1.1, &[1, 5, h])?;
        let a = module(stats)?
            .forward(&xs, None)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let b = module(folded)?
            .forward(&xs, None)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for (a, b) in a.iter().zip(&b) {
            assert!((a - b).abs() < 1e-5, "{a} against {b}");
        }
        Ok(())
    }

    #[test]
    fn rel_shift_reads_row_r_from_column_q_minus_one_minus_r() -> Result<()> {
        let (q, t) = (5, 7);
        let p = q + t - 1;
        let scores =
            Tensor::arange(0f32, (2 * q * p) as f32, &Device::Cpu)?.reshape((1, 2, q, p))?;
        let shifted = rel_shift(&scores, t)?.squeeze(0)?.to_vec3::<f32>()?;
        let scores = scores.squeeze(0)?.to_vec3::<f32>()?;
        for (h, rows) in shifted.iter().enumerate() {
            for (r, row) in rows.iter().enumerate() {
                for (j, v) in row.iter().enumerate() {
                    assert_eq!(*v, scores[h][r][q - 1 - r + j], "head {h} ({r}, {j})");
                }
            }
        }
        Ok(())
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
