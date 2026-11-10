//! TRELLIS.2's sparse DiT pieces: the timestep embedder, per-head RMS q/k norms, 3D RoPE and the AdaLN cross-attention
//! block, shared by JoVox's flow and the shape flow. Blocks run in their `dtype` (bf16 in the references), norms and
//! the q/k rotations in f32.

use inference_nn::attention::{AttentionMask, FlashParams, Sdpa, SdpaParams};
use inference_tensor::nn::{LayerNorm, Linear, Module, VarBuilder};
use inference_tensor::{D, DType, Device, Result, Tensor};

use crate::worldmirror::layers::AmpLinear;

// TimestepEmbedder's sinusoid width and period
const FREQ_EMBED: usize = 256;
const MAX_PERIOD: f64 = 10000.0;
pub(crate) const BLOCK_EPS: f64 = 1e-6;
const RMS_EPS: f64 = 1e-12;
const ROPE_BASE: f32 = 10000.0;
const AXES: usize = 3;

pub(crate) struct TimestepMlp {
    fc1: Linear,
    fc2: Linear,
}

impl TimestepMlp {
    pub(crate) fn load(channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: inference_tensor::nn::linear(FREQ_EMBED, channels, vb.pp("mlp").pp(0))?,
            fc2: inference_tensor::nn::linear(channels, channels, vb.pp("mlp").pp(2))?,
        })
    }

    /// `[cos, sin]` of timestep `t` at the sinusoid's frequencies, then the MLP: `[1, channels]` (f32).
    pub(crate) fn forward(&self, t: f32, device: &Device) -> Result<Tensor> {
        let half = FREQ_EMBED / 2;
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

pub(crate) struct RmsNorm {
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
pub struct Rope {
    cos: Tensor,
    sin: Tensor,
}

impl Rope {
    /// Each token's 3D position, 21 frequencies an axis for head dim 128 (the last pairs unrotated), as TRELLIS.2's
    /// `SparseRotaryPositionEmbedder` phases them.
    pub(crate) fn new(positions: &[[f32; 3]], head_dim: usize, device: &Device) -> Result<Self> {
        let freq_dim = head_dim / 2 / AXES;
        let pairs = head_dim / 2;
        let freqs: Vec<f32> = (0..freq_dim)
            .map(|i| 1.0 / ROPE_BASE.powf(i as f32 / freq_dim as f32))
            .collect();
        let mut cos = vec![1f32; positions.len() * pairs];
        let mut sin = vec![0f32; positions.len() * pairs];
        for (t, p) in positions.iter().enumerate() {
            for (a, &x) in p.iter().enumerate() {
                for (f, &freq) in freqs.iter().enumerate() {
                    let angle = x * freq;
                    cos[t * pairs + a * freq_dim + f] = angle.cos();
                    sin[t * pairs + a * freq_dim + f] = angle.sin();
                }
            }
        }
        let shape = (1, positions.len(), 1, pairs, 1);
        Ok(Self {
            cos: Tensor::from_vec(cos, shape, device)?,
            sin: Tensor::from_vec(sin, shape, device)?,
        })
    }

    pub(crate) fn apply(&self, xs: &Tensor) -> Result<Tensor> {
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

/// Where a block's six AdaLN vectors come from: its own SiLU-linear of the timestep embedding, or its learned offset on
/// the model's shared one (`share_mod`).
enum Modulation {
    Own(AmpLinear),
    Shared(Tensor),
}

/// The widths of a TRELLIS.2 sparse DiT.
#[derive(Clone, Copy, Debug)]
pub struct DitSpec {
    pub channels: usize,
    pub cond_channels: usize,
    pub blocks: usize,
    pub heads: usize,
    pub mlp: usize,
    pub share_mod: bool,
}

/// `ModulatedSparseTransformerCrossBlock`: AdaLN self-attention with RoPE, cross-attention to the condition, AdaLN MLP.
pub(crate) struct Block {
    modulation: Modulation,
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
pub struct CrossKv {
    k: Tensor,
    v: Tensor,
}

impl CrossKv {
    pub(crate) fn batch(&self) -> Result<usize> {
        self.k.dim(0)
    }

    /// Batch row `i` alone.
    #[cfg(test)]
    pub(crate) fn row(&self, i: usize) -> Result<Self> {
        Ok(Self {
            k: self.k.narrow(0, i, 1)?,
            v: self.v.narrow(0, i, 1)?,
        })
    }

    /// The first `n` batch rows.
    pub(crate) fn first(&self, n: usize) -> Result<Self> {
        Ok(Self {
            k: self.k.narrow(0, 0, n)?,
            v: self.v.narrow(0, 0, n)?,
        })
    }
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
    pub(crate) fn load(
        spec: DitSpec,
        plain: LayerNorm,
        dtype: DType,
        vb: VarBuilder,
    ) -> Result<Self> {
        let (c, heads) = (spec.channels, spec.heads);
        let head_dim = c / heads;
        let (sa, ca) = (vb.pp("self_attn"), vb.pp("cross_attn"));
        let norm2 = vb.pp("norm2");
        let modulation = if spec.share_mod {
            Modulation::Shared(vb.get(6 * c, "modulation")?)
        } else {
            Modulation::Own(AmpLinear::load(
                c,
                6 * c,
                true,
                dtype,
                vb.pp("adaLN_modulation").pp(1),
            )?)
        };
        Ok(Self {
            modulation,
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
            fc1: AmpLinear::load(c, spec.mlp, true, dtype, vb.pp("mlp").pp("mlp").pp(0))?,
            fc2: AmpLinear::load(spec.mlp, c, true, dtype, vb.pp("mlp").pp("mlp").pp(2))?,
            plain,
            heads,
            dtype,
        })
    }

    pub(crate) fn cross_kv(&self, cond: &Tensor) -> Result<CrossKv> {
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

    /// `t_emb` is the timestep embedding (in the stream's dtype), or with `share_mod` the model's shared modulation.
    pub(crate) fn forward(
        &self,
        xs: &Tensor,
        t_emb: &Tensor,
        rope: &Rope,
        kv: &CrossKv,
    ) -> Result<Tensor> {
        self.forward_inspect(xs, t_emb, rope, kv, &mut |_, _| {})
    }

    // `forward`, handing `inspect` the first norm, the rotated q and the attention and MLP outputs
    pub(crate) fn forward_inspect(
        &self,
        xs: &Tensor,
        t_emb: &Tensor,
        rope: &Rope,
        kv: &CrossKv,
        inspect: &mut dyn FnMut(&str, &Tensor),
    ) -> Result<Tensor> {
        let (b, n, c) = xs.dims3()?;
        let m = match &self.modulation {
            Modulation::Own(linear) => linear.forward(&t_emb.silu()?)?,
            // the f32 offset plus the shared vectors, back to their dtype
            Modulation::Shared(offset) => t_emb
                .to_dtype(DType::F32)?
                .broadcast_add(offset)?
                .to_dtype(t_emb.dtype())?,
        };
        let part = |i: usize| m.narrow(1, i * c, c)?.unsqueeze(1);
        let (shift_msa, scale_msa, gate_msa) = (part(0)?, part(1)?, part(2)?);
        let (shift_mlp, scale_mlp, gate_mlp) = (part(3)?, part(4)?, part(5)?);

        let normed = self.norm(&self.plain, xs)?;
        inspect("norm1", &normed);
        let h = normed
            .broadcast_mul(&(scale_msa + 1.0)?)?
            .broadcast_add(&shift_msa)?;
        let qkv = self.qkv.forward(&h)?.reshape((b, n, 3, self.heads, ()))?;
        let q = rope.apply(&self.q_norm.forward(&qkv.narrow(2, 0, 1)?.squeeze(2)?)?)?;
        let k = rope.apply(&self.k_norm.forward(&qkv.narrow(2, 1, 1)?.squeeze(2)?)?)?;
        let v = qkv.narrow(2, 2, 1)?.squeeze(2)?;
        inspect("q_rope", &q);
        let attn = sdpa(
            &q.transpose(1, 2)?,
            &k.transpose(1, 2)?,
            &v.transpose(1, 2)?,
        )?;
        let attn = self
            .self_out
            .forward(&attn.transpose(1, 2)?.reshape((b, n, c))?)?;
        inspect("self_attn", &attn);
        let xs = (xs + attn.broadcast_mul(&gate_msa)?)?;

        let h = self.norm(&self.norm2, &xs)?;
        let q = self
            .cross_q_norm
            .forward(&self.cross_q.forward(&h)?.reshape((b, n, self.heads, ()))?)?;
        let attn = sdpa(&q.transpose(1, 2)?, &kv.k, &kv.v)?;
        let attn = self
            .cross_out
            .forward(&attn.transpose(1, 2)?.reshape((b, n, c))?)?;
        inspect("cross_attn", &attn);
        let xs = (xs + attn)?;

        let h = self
            .norm(&self.plain, &xs)?
            .broadcast_mul(&(scale_mlp + 1.0)?)?
            .broadcast_add(&shift_mlp)?;
        let mlp = self.fc2.forward(&self.fc1.forward(&h)?.gelu()?)?;
        inspect("mlp", &mlp);
        xs + mlp.broadcast_mul(&gate_mlp)?
    }
}
