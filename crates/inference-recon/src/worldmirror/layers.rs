//! Blocks shared by WorldMirror's backbone, trunk and camera head, in the reference's mixed precision: matmuls in
//! `dtype` (bf16 for the trunk, as its autocast runs them; f32 for the heads), norms and residuals in f32.

use inference_nn::attention::{AttentionMask, FlashParams, Sdpa, SdpaParams};
use inference_tensor::nn::{LayerNorm, Linear, Module, VarBuilder};
use inference_tensor::{DType, Result, Tensor};

use super::rope::RopeTable;

/// A linear layer whose weights and matmul run in `dtype`; its output stays in `dtype`, as under autocast.
pub(crate) struct AmpLinear {
    inner: Linear,
    dtype: DType,
}

impl AmpLinear {
    pub(crate) fn load(
        in_dim: usize,
        out_dim: usize,
        bias: bool,
        dtype: DType,
        vb: VarBuilder,
    ) -> Result<Self> {
        let weight = vb.get((out_dim, in_dim), "weight")?.to_dtype(dtype)?;
        let bias = if bias {
            Some(vb.get(out_dim, "bias")?.to_dtype(dtype)?)
        } else {
            None
        };
        Ok(Self {
            inner: Linear::new(weight, bias),
            dtype,
        })
    }

    pub(crate) fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.inner.forward(&xs.to_dtype(self.dtype)?.contiguous()?)
    }
}

/// LayerNorm in f32 whatever the input dtype, as autocast runs it.
pub(crate) struct F32LayerNorm(LayerNorm);

impl F32LayerNorm {
    pub(crate) fn load(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self(inference_tensor::nn::layer_norm(size, eps, vb)?))
    }

    pub(crate) fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.0.forward(&xs.to_dtype(DType::F32)?)
    }
}

/// fc1, exact GELU, fc2.
pub(crate) struct Mlp {
    fc1: AmpLinear,
    fc2: AmpLinear,
}

impl Mlp {
    pub(crate) fn load(dim: usize, hidden: usize, dtype: DType, vb: VarBuilder) -> Result<Self> {
        Self::load_with_hidden(dim, hidden, dim, dtype, vb)
    }

    pub(crate) fn load_with_hidden(
        dim: usize,
        hidden: usize,
        out: usize,
        dtype: DType,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            fc1: AmpLinear::load(dim, hidden, true, dtype, vb.pp("fc1"))?,
            fc2: AmpLinear::load(hidden, out, true, dtype, vb.pp("fc2"))?,
        })
    }

    pub(crate) fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.fc2.forward(&self.fc1.forward(xs)?.gelu_erf()?)
    }
}

/// Multi-head self-attention, bidirectional, with optional per-head q/k LayerNorm and 2D RoPE.
pub(crate) struct Attention {
    qkv: AmpLinear,
    proj: AmpLinear,
    qk_norm: Option<(F32LayerNorm, F32LayerNorm)>,
    heads: usize,
    dtype: DType,
}

impl Attention {
    pub(crate) fn load(
        dim: usize,
        heads: usize,
        qk_norm: bool,
        dtype: DType,
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim = dim / heads;
        let qk_norm = if qk_norm {
            Some((
                F32LayerNorm::load(head_dim, 1e-5, vb.pp("q_norm"))?,
                F32LayerNorm::load(head_dim, 1e-5, vb.pp("k_norm"))?,
            ))
        } else {
            None
        };
        Ok(Self {
            qkv: AmpLinear::load(dim, 3 * dim, true, dtype, vb.pp("qkv"))?,
            proj: AmpLinear::load(dim, dim, true, dtype, vb.pp("proj"))?,
            qk_norm,
            heads,
            dtype,
        })
    }

    pub(crate) fn forward(
        &self,
        xs: &Tensor,
        rope: Option<(&RopeTable, &Tensor)>,
    ) -> Result<Tensor> {
        self.forward_inspect(xs, rope, &mut |_, _| {})
    }

    // `forward`, handing `inspect` the normed q and the rotated q
    pub(crate) fn forward_inspect(
        &self,
        xs: &Tensor,
        rope: Option<(&RopeTable, &Tensor)>,
        inspect: &mut dyn FnMut(&str, &Tensor),
    ) -> Result<Tensor> {
        let (b, n, c) = xs.dims3()?;
        let head_dim = c / self.heads;
        let qkv = self
            .qkv
            .forward(xs)?
            .reshape((b, n, 3, self.heads, head_dim))?
            .permute((2, 0, 3, 1, 4))?;
        let (mut q, mut k, v) = (qkv.get(0)?, qkv.get(1)?, qkv.get(2)?.contiguous()?);
        if let Some((q_norm, k_norm)) = &self.qk_norm {
            q = q_norm.forward(&q)?.to_dtype(self.dtype)?;
            k = k_norm.forward(&k)?.to_dtype(self.dtype)?;
        }
        inspect("q_norm", &q);
        if let Some((table, cells)) = rope {
            q = table.apply(&q, cells)?;
            k = table.apply(&k, cells)?;
        }
        inspect("q_rope", &q);
        let params = SdpaParams {
            n_kv_groups: 1,
            softcap: None,
            softmax_scale: 1.0 / (head_dim as f32).sqrt(),
            sliding_window: None,
            sinks: None,
            chunk: None,
        };
        let out = Sdpa.run_attention(
            &q.contiguous()?,
            &k.contiguous()?,
            &v,
            &AttentionMask::None,
            Some(&FlashParams::empty(false)),
            &params,
        )?;
        self.proj.forward(&out.transpose(1, 2)?.reshape((b, n, c))?)
    }
}

/// Pre-norm transformer block with LayerScale: `x + ls1 * attn(norm1(x))`, then `x + ls2 * mlp(norm2(x))`, the
/// residual in f32.
pub(crate) struct Block {
    norm1: F32LayerNorm,
    attn: Attention,
    ls1: Tensor,
    norm2: F32LayerNorm,
    mlp: Mlp,
    ls2: Tensor,
}

/// How a block family is built: its width, heads, norm epsilon, q/k norm and matmul dtype.
#[derive(Clone, Copy)]
pub(crate) struct BlockSpec {
    pub dim: usize,
    pub heads: usize,
    pub mlp_ratio: usize,
    pub eps: f64,
    pub qk_norm: bool,
    pub dtype: DType,
}

impl Block {
    pub(crate) fn load(spec: BlockSpec, vb: VarBuilder) -> Result<Self> {
        let BlockSpec {
            dim,
            heads,
            mlp_ratio,
            eps,
            qk_norm,
            dtype,
        } = spec;
        Ok(Self {
            norm1: F32LayerNorm::load(dim, eps, vb.pp("norm1"))?,
            attn: Attention::load(dim, heads, qk_norm, dtype, vb.pp("attn"))?,
            ls1: vb.pp("ls1").get(dim, "gamma")?,
            norm2: F32LayerNorm::load(dim, eps, vb.pp("norm2"))?,
            mlp: Mlp::load(dim, dim * mlp_ratio, dtype, vb.pp("mlp"))?,
            ls2: vb.pp("ls2").get(dim, "gamma")?,
        })
    }

    pub(crate) fn forward(
        &self,
        xs: &Tensor,
        rope: Option<(&RopeTable, &Tensor)>,
    ) -> Result<Tensor> {
        self.forward_inspect(xs, rope, &mut |_, _| {})
    }

    // `forward`, handing `inspect` the norm1 output, the attention's insides and its output
    pub(crate) fn forward_inspect(
        &self,
        xs: &Tensor,
        rope: Option<(&RopeTable, &Tensor)>,
        inspect: &mut dyn FnMut(&str, &Tensor),
    ) -> Result<Tensor> {
        let normed = self.norm1.forward(xs)?;
        inspect("norm1", &normed);
        let attn = self.attn.forward_inspect(&normed, rope, inspect)?;
        inspect("attn_out", &attn);
        let xs = (xs + attn.to_dtype(DType::F32)?.broadcast_mul(&self.ls1)?)?;
        let normed = self.norm2.forward(&xs)?;
        inspect("norm2", &normed);
        let mlp = self.mlp.forward(&normed)?;
        inspect("mlp_out", &mlp);
        xs + mlp.to_dtype(DType::F32)?.broadcast_mul(&self.ls2)?
    }
}
