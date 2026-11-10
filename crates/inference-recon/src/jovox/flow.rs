//! JoVox's flow: TRELLIS.2's sparse DiT (AdaLN self- and cross-attention blocks with per-head RMS q/k norms) over the
//! sampler's tokens, each embedded from its slots, its position and its level, predicting the 256 values of each of
//! its 8 slots. The blocks run in `dtype` (bf16, as the reference), the embedding and output layers in f32.

use inference_nn::attention::{AttentionMask, FlashParams, Sdpa, SdpaParams};
use inference_tensor::nn::{LayerNorm, Linear, Module, VarBuilder};
use inference_tensor::{D, DType, Device, Result, Tensor};

use super::sampler::{MASK, SLOTS, Tokens};
use crate::worldmirror::layers::AmpLinear;

pub const CLASSES: usize = 256;
// TimestepEmbedder's sinusoid width and period
const FREQ_EMBED: usize = 256;
const MAX_PERIOD: f64 = 10000.0;
// the progress ratios are scaled to diffusion timesteps
const TIME_SCALE: f32 = 1000.0;
const BLOCK_EPS: f64 = 1e-6;
// the final F.layer_norm's default
const OUT_EPS: f64 = 1e-5;
const RMS_EPS: f64 = 1e-12;
const ROPE_BASE: f32 = 10000.0;
// rope runs on unified coordinates scaled to this grid
const ROPE_RES: f32 = 256.0;
// the 4^3 and 16^3 cells' sides, which position a token in the unit cube
const LEVEL_SIDES: [f32; 2] = [4.0, 16.0];
const POINT_FREQS: usize = 10;
// each slot is 0, 1 or unknown, in the child and grandchild embeddings
const TRI: usize = 3;
const AXES: usize = 3;

/// The flow's widths, as the checkpoint's config sets them.
#[derive(Clone, Copy, Debug)]
pub struct FlowSpec {
    pub channels: usize,
    pub cond_channels: usize,
    pub blocks: usize,
    pub heads: usize,
}

struct TimestepMlp {
    fc1: Linear,
    fc2: Linear,
}

impl TimestepMlp {
    fn load(channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: inference_tensor::nn::linear(FREQ_EMBED, channels, vb.pp("mlp").pp(0))?,
            fc2: inference_tensor::nn::linear(channels, channels, vb.pp("mlp").pp(2))?,
        })
    }

    // `t` scaled to a timestep, `[cos, sin]` of it at the sinusoid's frequencies, then the MLP
    fn forward(&self, t: f32, device: &Device) -> Result<Tensor> {
        let half = FREQ_EMBED / 2;
        let t = t * TIME_SCALE;
        let args: Vec<f32> = (0..half)
            .map(|i| t * (-MAX_PERIOD.ln() * i as f64 / half as f64).exp() as f32)
            .collect();
        let emb: Vec<f32> = args
            .iter()
            .map(|a| a.cos())
            .chain(args.iter().map(|a| a.sin()))
            .collect();
        let emb = Tensor::from_vec(emb, (1, FREQ_EMBED), device)?;
        self.fc2.forward(&self.fc1.forward(&emb)?.silu()?)
    }
}

struct RmsNorm {
    // gamma * sqrt(head_dim), `[heads, head_dim]`
    scale: Tensor,
}

impl RmsNorm {
    fn load(heads: usize, head_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            scale: (vb.get((heads, head_dim), "gamma")? * (head_dim as f64).sqrt())?,
        })
    }

    // F.normalize over the head dim in f32, then the gain; `xs` is `[..., heads, head_dim]`
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let xs = xs.to_dtype(DType::F32)?;
        let norm = xs.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?.maximum(RMS_EPS)?;
        xs.broadcast_div(&norm)?
            .broadcast_mul(&self.scale)?
            .to_dtype(dtype)
    }
}

/// Per-token rotations `[1, tokens, 1, head_dim / 2, 1]` (f32), pairs of adjacent channels rotated together.
pub(crate) struct Rope {
    cos: Tensor,
    sin: Tensor,
}

impl Rope {
    fn apply(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, n, h, d) = xs.dims4()?;
        let dtype = xs.dtype();
        let xs = xs.to_dtype(DType::F32)?.reshape((b, n, h, d / 2, 2))?;
        let (even, odd) = (xs.narrow(4, 0, 1)?, xs.narrow(4, 1, 1)?);
        let re = (even.broadcast_mul(&self.cos)? - odd.broadcast_mul(&self.sin)?)?;
        let im = (even.broadcast_mul(&self.sin)? + odd.broadcast_mul(&self.cos)?)?;
        Tensor::cat(&[re, im], 4)?
            .reshape((b, n, h, d))?
            .to_dtype(dtype)
    }
}

struct Block {
    modulation: AmpLinear,
    qkv: AmpLinear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    self_out: AmpLinear,
    norm2: LayerNorm,
    cross_q: AmpLinear,
    cross_kv: AmpLinear,
    cross_q_norm: RmsNorm,
    cross_k_norm: RmsNorm,
    cross_out: AmpLinear,
    fc1: AmpLinear,
    fc2: AmpLinear,
    plain: LayerNorm,
    heads: usize,
    dtype: DType,
}

/// A block's cross-attention keys and values over the condition, `[batch, heads, tokens, head_dim]`.
pub(crate) struct CrossKv {
    k: Tensor,
    v: Tensor,
}

fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let head_dim = q.dim(D::Minus1)?;
    let params = SdpaParams {
        n_kv_groups: 1,
        softcap: None,
        softmax_scale: 1.0 / (head_dim as f32).sqrt(),
        sliding_window: None,
        sinks: None,
        chunk: None,
    };
    Sdpa.run_attention(
        &q.contiguous()?,
        &k.contiguous()?,
        &v.contiguous()?,
        &AttentionMask::None,
        Some(&FlashParams::empty(false)),
        &params,
    )
}

impl Block {
    fn load(spec: FlowSpec, plain: LayerNorm, dtype: DType, vb: VarBuilder) -> Result<Self> {
        let (c, heads) = (spec.channels, spec.heads);
        let head_dim = c / heads;
        let (sa, ca) = (vb.pp("self_attn"), vb.pp("cross_attn"));
        let norm2 = vb.pp("norm2");
        Ok(Self {
            modulation: AmpLinear::load(c, 6 * c, true, dtype, vb.pp("adaLN_modulation").pp(1))?,
            qkv: AmpLinear::load(c, 3 * c, true, dtype, sa.pp("to_qkv"))?,
            q_norm: RmsNorm::load(heads, head_dim, sa.pp("q_rms_norm"))?,
            k_norm: RmsNorm::load(heads, head_dim, sa.pp("k_rms_norm"))?,
            self_out: AmpLinear::load(c, c, true, dtype, sa.pp("to_out"))?,
            norm2: LayerNorm::new(norm2.get(c, "weight")?, norm2.get(c, "bias")?, BLOCK_EPS),
            cross_q: AmpLinear::load(c, c, true, dtype, ca.pp("to_q"))?,
            cross_kv: AmpLinear::load(spec.cond_channels, 2 * c, true, dtype, ca.pp("to_kv"))?,
            cross_q_norm: RmsNorm::load(heads, head_dim, ca.pp("q_rms_norm"))?,
            cross_k_norm: RmsNorm::load(heads, head_dim, ca.pp("k_rms_norm"))?,
            cross_out: AmpLinear::load(c, c, true, dtype, ca.pp("to_out"))?,
            fc1: AmpLinear::load(c, 4 * c, true, dtype, vb.pp("mlp").pp("mlp").pp(0))?,
            fc2: AmpLinear::load(4 * c, c, true, dtype, vb.pp("mlp").pp("mlp").pp(2))?,
            plain,
            heads,
            dtype,
        })
    }

    fn cross_kv(&self, cond: &Tensor) -> Result<CrossKv> {
        let (b, l, _) = cond.dims3()?;
        let kv = self
            .cross_kv
            .forward(cond)?
            .reshape((b, l, 2, self.heads, ()))?;
        let k = self
            .cross_k_norm
            .forward(&kv.narrow(2, 0, 1)?.squeeze(2)?)?;
        let v = kv.narrow(2, 1, 1)?.squeeze(2)?;
        Ok(CrossKv {
            k: k.transpose(1, 2)?.contiguous()?,
            v: v.transpose(1, 2)?.contiguous()?,
        })
    }

    // the plain LayerNorm in f32, back to the stream's dtype
    fn norm(&self, norm: &LayerNorm, xs: &Tensor) -> Result<Tensor> {
        norm.forward(&xs.to_dtype(DType::F32)?)?
            .to_dtype(self.dtype)
    }

    fn forward(&self, xs: &Tensor, t_emb: &Tensor, rope: &Rope, kv: &CrossKv) -> Result<Tensor> {
        let (b, n, c) = xs.dims3()?;
        let m = self.modulation.forward(&t_emb.silu()?)?;
        let part = |i: usize| m.narrow(1, i * c, c)?.unsqueeze(1);
        let (shift_msa, scale_msa, gate_msa) = (part(0)?, part(1)?, part(2)?);
        let (shift_mlp, scale_mlp, gate_mlp) = (part(3)?, part(4)?, part(5)?);

        let h = self
            .norm(&self.plain, xs)?
            .broadcast_mul(&(scale_msa + 1.0)?)?
            .broadcast_add(&shift_msa)?;
        let qkv = self.qkv.forward(&h)?.reshape((b, n, 3, self.heads, ()))?;
        let q = rope.apply(&self.q_norm.forward(&qkv.narrow(2, 0, 1)?.squeeze(2)?)?)?;
        let k = rope.apply(&self.k_norm.forward(&qkv.narrow(2, 1, 1)?.squeeze(2)?)?)?;
        let v = qkv.narrow(2, 2, 1)?.squeeze(2)?;
        let attn = sdpa(
            &q.transpose(1, 2)?,
            &k.transpose(1, 2)?,
            &v.transpose(1, 2)?,
        )?;
        let attn = self
            .self_out
            .forward(&attn.transpose(1, 2)?.reshape((b, n, c))?)?;
        let xs = (xs + attn.broadcast_mul(&gate_msa)?)?;

        let h = self.norm(&self.norm2, &xs)?;
        let q = self
            .cross_q_norm
            .forward(&self.cross_q.forward(&h)?.reshape((b, n, self.heads, ()))?)?;
        let attn = sdpa(&q.transpose(1, 2)?, &kv.k, &kv.v)?;
        let attn = self
            .cross_out
            .forward(&attn.transpose(1, 2)?.reshape((b, n, c))?)?;
        let xs = (xs + attn)?;

        let h = self
            .norm(&self.plain, &xs)?
            .broadcast_mul(&(scale_mlp + 1.0)?)?
            .broadcast_add(&shift_mlp)?;
        let mlp = self.fc2.forward(&self.fc1.forward(&h)?.gelu()?)?;
        xs + mlp.broadcast_mul(&gate_mlp)?
    }
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
                .map(|i| Block::load(spec, plain.clone(), dtype, vb.pp("blocks").pp(i)))
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

    fn rope(&self, units: &[[f32; 3]]) -> Result<Rope> {
        let freq_dim = self.head_dim / 2 / AXES;
        let pairs = self.head_dim / 2;
        let freqs: Vec<f32> = (0..freq_dim)
            .map(|i| 1.0 / ROPE_BASE.powf(i as f32 / freq_dim as f32))
            .collect();
        let mut cos = vec![1f32; units.len() * pairs];
        let mut sin = vec![0f32; units.len() * pairs];
        for (t, u) in units.iter().enumerate() {
            for (a, &v) in u.iter().enumerate() {
                let x = v * ROPE_RES;
                for (f, &freq) in freqs.iter().enumerate() {
                    let angle = x * freq;
                    cos[t * pairs + a * freq_dim + f] = angle.cos();
                    sin[t * pairs + a * freq_dim + f] = angle.sin();
                }
            }
        }
        let shape = (1, units.len(), 1, pairs, 1);
        Ok(Rope {
            cos: Tensor::from_vec(cos, shape, &self.device)?,
            sin: Tensor::from_vec(sin, shape, &self.device)?,
        })
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
        let batch = kv[0].k.dim(0)?;
        let n = units.len();
        let feats = self.embed(tokens, &units)?.to_dtype(self.dtype)?;
        let mut xs = feats.unsqueeze(0)?.repeat((batch, 1, 1))?;
        let t_emb = (self.t_embed[0].forward(tokens.ratios[0], &self.device)?
            + self.t_embed[1].forward(tokens.ratios[1], &self.device)?)?
        .to_dtype(self.dtype)?;
        let rope = self.rope(&units)?;
        for (block, kv) in self.blocks.iter().zip(kv) {
            xs = block.forward(&xs, &t_emb, &rope, kv)?;
        }
        let xs = self.out_norm.forward(&xs.to_dtype(DType::F32)?)?;
        self.out.forward(&xs)?.reshape((batch, n, SLOTS, CLASSES))
    }
}
