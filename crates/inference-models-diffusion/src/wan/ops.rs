//! Pieces the Wan transformer and its ControlNet share, in the reference's mixed precision: matmuls in bf16, norms and
//! modulation in f32, the residual stream in bf16.

use inference_nn::attention::{AttentionMask, FlashParams, Sdpa, SdpaParams};
use inference_quant::ShardedVarBuilder;
use inference_tensor::{D, DType, Device, Module, Result, Tensor};

use crate::qlinear::MaybeQuantLinear;

pub(crate) const ACT_DTYPE: DType = DType::BF16;
const ROPE_THETA: f64 = 10_000.0;

/// A linear whose input is cast to the activation dtype first.
pub(crate) struct Linear(MaybeQuantLinear);

impl Linear {
    pub(crate) fn load(in_dim: usize, out_dim: usize, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self(MaybeQuantLinear::new(in_dim, out_dim, true, vb)?))
    }

    pub(crate) fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.0.forward(&xs.to_dtype(ACT_DTYPE)?)
    }

    pub(crate) fn to_device(&self, device: &Device) -> Result<Self> {
        Ok(Self(self.0.to_device(device)?))
    }
}

/// LayerNorm computed in f32, with or without affine terms; the output stays f32.
pub(crate) struct LayerNorm {
    weight: Option<Tensor>,
    bias: Option<Tensor>,
    eps: f64,
}

impl LayerNorm {
    pub(crate) fn plain(eps: f64) -> Self {
        Self {
            weight: None,
            bias: None,
            eps,
        }
    }

    pub(crate) fn load(dim: usize, eps: f64, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            weight: Some(vb.get(dim, "weight")?.to_dtype(DType::F32)?),
            bias: Some(vb.get(dim, "bias")?.to_dtype(DType::F32)?),
            eps,
        })
    }

    pub(crate) fn to_device(&self, device: &Device) -> Result<Self> {
        Ok(Self {
            weight: self
                .weight
                .as_ref()
                .map(|w| w.to_device(device))
                .transpose()?,
            bias: self
                .bias
                .as_ref()
                .map(|b| b.to_device(device))
                .transpose()?,
            eps: self.eps,
        })
    }

    /// The norm (and affine) of `xs`, then `* (1 + scale) + shift` when given, in the activation dtype.
    pub(crate) fn forward_act(
        &self,
        xs: &Tensor,
        modulation: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if xs.device().is_cuda() && xs.dtype() == ACT_DTYPE {
            let (shift, scale) = match modulation {
                Some((shift, scale)) => (Some(shift.contiguous()?), Some(scale.contiguous()?)),
                None => (None, None),
            };
            return inference_nn::cuda::dit::layer_norm(
                xs,
                self.weight.as_ref(),
                self.bias.as_ref(),
                scale.as_ref(),
                shift.as_ref(),
                self.eps as f32,
            );
        }
        match modulation {
            Some((shift, scale)) => modulate(&self.forward(xs)?, shift, scale),
            None => self.forward(xs)?.to_dtype(ACT_DTYPE),
        }
    }

    pub(crate) fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x = xs.to_dtype(DType::F32)?;
        let mean = x.mean_keepdim(D::Minus1)?;
        let centered = x.broadcast_sub(&mean)?;
        let var = centered.sqr()?.mean_keepdim(D::Minus1)?;
        let mut out = centered.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        if let Some(w) = &self.weight {
            out = out.broadcast_mul(w)?;
        }
        if let Some(b) = &self.bias {
            out = out.broadcast_add(b)?;
        }
        Ok(out)
    }
}

/// RMSNorm over the whole (all heads) feature dim, computed in f32, output in the activation dtype.
pub(crate) struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub(crate) fn load(dim: usize, eps: f64, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get(dim, "weight")?.to_dtype(DType::F32)?,
            eps,
        })
    }

    pub(crate) fn to_device(&self, device: &Device) -> Result<Self> {
        Ok(Self {
            weight: self.weight.to_device(device)?,
            eps: self.eps,
        })
    }

    /// The norm of `xs` `[b, tokens, heads * head_dim]` split into `[b, heads, tokens, head_dim]`, rotated by `rope`.
    pub(crate) fn heads(
        &self,
        xs: &Tensor,
        n_heads: usize,
        rope: Option<&RopeTables>,
    ) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if xs.device().is_cuda() && xs.dtype() == ACT_DTYPE {
            return inference_nn::cuda::dit::rms_norm_rope(
                xs,
                &self.weight,
                rope.map(|r| (&r.cos, &r.sin)),
                n_heads,
                self.eps as f32,
            );
        }
        let split = heads(&self.forward(xs)?, n_heads)?;
        match rope {
            Some(rope) => rope.apply(&split),
            None => Ok(split),
        }
    }

    pub(crate) fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x = xs.to_dtype(DType::F32)?;
        let rms = (x.sqr()?.mean_keepdim(D::Minus1)? + self.eps)?.sqrt()?;
        x.broadcast_div(&rms)?
            .broadcast_mul(&self.weight)?
            .to_dtype(ACT_DTYPE)
    }
}

/// `norm(x) * (1 + scale) + shift` in f32, back to the activation dtype.
pub(crate) fn modulate(normed: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    normed
        .broadcast_mul(&(scale + 1.0)?)?
        .broadcast_add(shift)?
        .to_dtype(ACT_DTYPE)
}

/// `x + y * gate` in f32, back to the activation dtype.
pub(crate) fn gated_residual(xs: &Tensor, ys: &Tensor, gate: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if xs.device().is_cuda() && xs.dtype() == ACT_DTYPE && ys.dtype() == ACT_DTYPE {
        return inference_nn::cuda::dit::gated_residual(xs, ys, &gate.contiguous()?);
    }
    (xs.to_dtype(DType::F32)? + ys.to_dtype(DType::F32)?.broadcast_mul(gate)?)?.to_dtype(ACT_DTYPE)
}

/// Per-token rotary tables `(cos, sin)`, each `[tokens, head_dim / 2]` (f32): Wan's 3D RoPE splits a head into frame,
/// row and column parts and rotates adjacent pairs.
pub(crate) struct RopeTables {
    pub cos: Tensor,
    pub sin: Tensor,
}

impl RopeTables {
    /// Tables for tokens at `(frame, row, column)` positions.
    pub(crate) fn new(head_dim: usize, positions: &[[usize; 3]], device: &Device) -> Result<Self> {
        let hw = 2 * (head_dim / 6);
        let dims = [head_dim - 2 * hw, hw, hw];
        let half = head_dim / 2;
        let mut cos = Vec::with_capacity(positions.len() * half);
        let mut sin = Vec::with_capacity(positions.len() * half);
        for pos in positions {
            for (axis, &dim) in dims.iter().enumerate() {
                for i in 0..dim / 2 {
                    let freq = 1.0 / ROPE_THETA.powf((2 * i) as f64 / dim as f64);
                    let angle = pos[axis] as f64 * freq;
                    cos.push(angle.cos() as f32);
                    sin.push(angle.sin() as f32);
                }
            }
        }
        Ok(Self {
            cos: Tensor::from_vec(cos, (positions.len(), half), device)?,
            sin: Tensor::from_vec(sin, (positions.len(), half), device)?,
        })
    }

    /// Tables for a grid of `frames x rows x cols` tokens in frame-major, row-major order.
    pub(crate) fn grid(
        head_dim: usize,
        (f, h, w): (usize, usize, usize),
        device: &Device,
    ) -> Result<Self> {
        let positions = (0..f)
            .flat_map(|t| (0..h).flat_map(move |y| (0..w).map(move |x| [t, y, x])))
            .collect::<Vec<_>>();
        Self::new(head_dim, &positions, device)
    }

    pub(crate) fn cat(parts: &[&RopeTables]) -> Result<Self> {
        Ok(Self {
            cos: Tensor::cat(&parts.iter().map(|p| &p.cos).collect::<Vec<_>>(), 0)?,
            sin: Tensor::cat(&parts.iter().map(|p| &p.sin).collect::<Vec<_>>(), 0)?,
        })
    }

    /// `x` `[b, heads, tokens, head_dim]` rotated pairwise, computed in f32, returned in the activation dtype.
    pub(crate) fn apply(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, h, n, d) = xs.dims4()?;
        let pairs = xs.to_dtype(DType::F32)?.reshape((b, h, n, d / 2, 2))?;
        let x1 = pairs.narrow(4, 0, 1)?.squeeze(4)?;
        let x2 = pairs.narrow(4, 1, 1)?.squeeze(4)?;
        let cos = self.cos.reshape((1, 1, n, d / 2))?;
        let sin = self.sin.reshape((1, 1, n, d / 2))?;
        let out1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let out2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;
        Tensor::stack(&[out1, out2], 4)?
            .reshape((b, h, n, d))?
            .to_dtype(ACT_DTYPE)
    }
}

/// Bidirectional attention over `[b, heads, tokens, head_dim]` tensors, output `[b, tokens, heads * head_dim]`.
pub(crate) fn attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let (b, h, n, d) = q.dims4()?;
    let params = SdpaParams {
        n_kv_groups: 1,
        softcap: None,
        softmax_scale: 1.0 / (d as f32).sqrt(),
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
    )?
    .transpose(1, 2)?
    .reshape((b, n, h * d))
}

/// `[b, tokens, heads * head_dim]` to `[b, heads, tokens, head_dim]`.
pub(crate) fn heads(xs: &Tensor, n_heads: usize) -> Result<Tensor> {
    let (b, n, c) = xs.dims3()?;
    xs.reshape((b, n, n_heads, c / n_heads))?.transpose(1, 2)
}

/// A conv whose kernel equals its stride, `[o, i, kt, kh, kw]` with kt = 1, as a matmul over patches: frames
/// `[c, f, h, w]` to tokens `[f * (h / kh) * (w / kw), o]` in frame-major, row-major order.
pub(crate) struct PatchConv {
    weight: Tensor,
    bias: Tensor,
    kh: usize,
    kw: usize,
}

impl PatchConv {
    pub(crate) fn load(vb: &ShardedVarBuilder) -> Result<Self> {
        let weight = vb.get_unchecked("weight")?;
        let (o, i, _, kh, kw) = weight.dims5()?;
        Ok(Self {
            weight: weight.reshape((o, i * kh * kw))?.to_dtype(ACT_DTYPE)?,
            bias: vb.get_unchecked("bias")?.to_dtype(ACT_DTYPE)?,
            kh,
            kw,
        })
    }

    pub(crate) fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (c, f, h, w) = xs.dims4()?;
        let (kh, kw) = (self.kh, self.kw);
        let patches = xs
            .to_dtype(ACT_DTYPE)?
            .reshape((c, f, h / kh, kh, w / kw, kw))?
            .permute((1, 2, 4, 0, 3, 5))?
            .reshape((f * (h / kh) * (w / kw), c * kh * kw))?;
        patches.matmul(&self.weight.t()?)?.broadcast_add(&self.bias)
    }
}

/// tanh-approximated GELU in f32, back to the activation dtype.
pub(crate) fn gelu_tanh(xs: &Tensor) -> Result<Tensor> {
    // the CUDA bf16 kernel computes in f32 already
    if xs.device().is_cuda() && xs.dtype() == ACT_DTYPE {
        return xs.gelu();
    }
    xs.to_dtype(DType::F32)?.gelu()?.to_dtype(ACT_DTYPE)
}
