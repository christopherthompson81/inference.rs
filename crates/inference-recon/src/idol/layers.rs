//! The pieces Idol's denoiser and decoders share: f32 LayerNorms around low-precision blocks, per-head RMS norms of
//! queries and keys, tanh-GELU feed-forwards and non-causal attention.

use inference_nn::attention::{AttentionMask, FlashParams, Sdpa, SdpaParams};
use inference_tensor::nn::VarBuilder;
use inference_tensor::{D, DType, Result, Tensor};

use crate::worldmirror::layers::AmpLinear;

/// The blocks' LayerNorms (eps 1e-6); the rotary layers' and the heads' use torch's default.
pub(super) const BLOCK_EPS: f32 = 1e-6;
pub(super) const DEFAULT_EPS: f32 = 1e-5;
// F.normalize's floor on the norm
pub(super) const NORMALIZE_EPS: f64 = 1e-12;

/// A LayerNorm computed in f32 (`LayerNorm32`), with or without its affine, returned in the input's dtype.
pub(super) struct Norm32 {
    weight: Tensor,
    bias: Tensor,
    eps: f32,
    // read by the fused CUDA path only
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    affine: bool,
}

impl Norm32 {
    pub(super) fn load(channels: usize, affine: bool, eps: f32, vb: VarBuilder) -> Result<Self> {
        let (weight, bias) = if affine {
            (vb.get(channels, "weight")?, vb.get(channels, "bias")?)
        } else {
            (
                Tensor::ones(channels, DType::F32, vb.device())?,
                Tensor::zeros(channels, DType::F32, vb.device())?,
            )
        };
        Ok(Self {
            weight,
            bias,
            eps,
            affine,
        })
    }

    /// The f32 weight and bias, when it has them.
    #[cfg(feature = "cuda")]
    pub(super) fn affine(&self) -> Option<(&Tensor, &Tensor)> {
        self.affine.then_some((&self.weight, &self.bias))
    }

    #[cfg(feature = "cuda")]
    pub(super) fn eps(&self) -> f32 {
        self.eps
    }

    pub(super) fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        inference_tensor::nn::ops::layer_norm(
            &xs.to_dtype(DType::F32)?,
            &self.weight,
            &self.bias,
            self.eps,
        )?
        .to_dtype(dtype)
    }
}

/// `F.layer_norm` with no affine, in f32.
pub(super) fn plain_norm(xs: &Tensor, eps: f32) -> Result<Tensor> {
    let c = xs.dim(D::Minus1)?;
    let ones = Tensor::ones(c, DType::F32, xs.device())?;
    let zeros = Tensor::zeros(c, DType::F32, xs.device())?;
    inference_tensor::nn::ops::layer_norm(&xs.to_dtype(DType::F32)?, &ones, &zeros, eps)
}

/// `MultiHeadRMSNorm`: each head's vector normalized, times its gain and sqrt(head_dim), on `[b, l, heads, d]`.
pub(super) struct HeadRmsNorm {
    gain: Tensor,
}

impl HeadRmsNorm {
    pub(super) fn load(heads: usize, dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gain: (vb.get((heads, dim), "gamma")? * (dim as f64).sqrt())?,
        })
    }

    /// `gamma * sqrt(dim)` `[heads, dim]`, f32.
    #[cfg(feature = "cuda")]
    pub(super) fn gain(&self) -> &Tensor {
        &self.gain
    }

    pub(super) fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let xs = xs.to_dtype(DType::F32)?;
        let norm = xs
            .sqr()?
            .sum_keepdim(D::Minus1)?
            .sqrt()?
            .maximum(NORMALIZE_EPS)?;
        xs.broadcast_div(&norm)?
            .broadcast_mul(&self.gain)?
            .to_dtype(dtype)
    }
}

/// Linear, tanh GELU, linear.
pub(super) struct FeedForward {
    up: AmpLinear,
    down: AmpLinear,
}

impl FeedForward {
    pub(super) fn load(
        channels: usize,
        hidden: usize,
        dtype: DType,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            up: AmpLinear::load(channels, hidden, true, dtype, vb.pp("0"))?,
            down: AmpLinear::load(hidden, channels, true, dtype, vb.pp("2"))?,
        })
    }

    pub(super) fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.down.forward(&self.up.forward(xs)?.gelu()?)
    }

    #[cfg(feature = "cuda")]
    pub(super) fn linears(&self) -> (&AmpLinear, &AmpLinear) {
        (&self.up, &self.down)
    }
}

/// Non-causal attention of `[b, l, heads, d]` queries over `[b, s, heads, d]` keys and values, as `[b, l, heads, d]`.
/// The softmax-weighted values sum in f32: fp16 sums over Idol's 12K tokens cost the velocity 4% (rms) against the
/// reference. On CUDA that is fattn's `f32_values`; elsewhere the inputs go to f32.
pub(super) fn attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let d = q.dim(D::Minus1)?;
    let scale = 1.0 / (d as f32).sqrt();
    #[cfg(feature = "cuda")]
    if q.device().is_cuda() {
        let opts = inference_fattn::FattnOptions {
            scale,
            f32_values: true,
            ..Default::default()
        };
        let (q, k, v) = (q.contiguous()?, k.contiguous()?, v.contiguous()?);
        if inference_fattn::supported(&q, &k, &v, &opts)? {
            return inference_fattn::flash_attn(&q, &k, &v, &opts);
        }
    }
    let params = SdpaParams {
        n_kv_groups: 1,
        softcap: None,
        softmax_scale: scale,
        sliding_window: None,
        sinks: None,
        chunk: None,
    };
    let heads_first = |t: &Tensor| t.to_dtype(DType::F32)?.transpose(1, 2)?.contiguous();
    Sdpa.run_attention(
        &heads_first(q)?,
        &heads_first(k)?,
        &heads_first(v)?,
        &AttentionMask::None,
        Some(&FlashParams::empty(false)),
        &params,
    )?
    .transpose(1, 2)?
    .to_dtype(q.dtype())
}

/// `SinusoidalEmbedder`: a scalar per batch element (time, octree resolution) through cos/sin and a two-layer MLP.
pub(super) struct ScalarEmbedder {
    fc1: AmpLinear,
    fc2: AmpLinear,
    max_period: f64,
    two_pi: bool,
}

// the embedder's sinusoid width
const FREQUENCY_CHANNELS: usize = 256;

impl ScalarEmbedder {
    pub(super) fn load(
        channels: usize,
        max_period: f64,
        two_pi: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            fc1: AmpLinear::load(
                FREQUENCY_CHANNELS,
                channels,
                true,
                DType::F32,
                vb.pp("mlp.0"),
            )?,
            fc2: AmpLinear::load(channels, channels, true, DType::F32, vb.pp("mlp.2"))?,
            max_period,
            two_pi,
        })
    }

    /// `values` `[b]` f32 -> `[b, channels]` f32.
    pub(super) fn forward(&self, values: &Tensor) -> Result<Tensor> {
        let half = FREQUENCY_CHANNELS / 2;
        let freqs: Vec<f32> = (0..half)
            .map(|i| (-(self.max_period.ln()) * i as f64 / half as f64).exp() as f32)
            .collect();
        let freqs = Tensor::from_vec(freqs, (1, half), values.device())?;
        let mut args = values
            .to_dtype(DType::F32)?
            .unsqueeze(1)?
            .broadcast_mul(&freqs)?;
        if self.two_pi {
            args = (args * (2.0 * std::f64::consts::PI))?;
        }
        let emb = Tensor::cat(&[args.cos()?, args.sin()?], 1)?;
        self.fc2.forward(&self.fc1.forward(&emb)?.silu()?)
    }
}
