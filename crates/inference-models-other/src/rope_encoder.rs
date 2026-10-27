//! Bidirectional self-attention with a fused QKV projection and rotary positions, shared by the RoPE encoders.
#![allow(clippy::cast_precision_loss)]

use std::sync::Arc;

use inference_quant::{QuantMethod, ShardedVarBuilder};
use inference_tensor::{DType, Device, Result, Tensor};

use crate::{
    attention::{AttentionMask, FlashParams, SdpaParams},
    layers::{RotaryEmbedding, Sdpa, apply_rotary_preselected_qk},
};

/// The cos / sin tables for `head_dim`, built once per rope base.
pub fn rotary(
    base: f32,
    head_dim: usize,
    max_positions: usize,
    device: &Device,
    dtype: DType,
) -> Result<Arc<RotaryEmbedding>> {
    // NeoX-style halves, as transformers' rotate_half applies them
    Ok(Arc::new(RotaryEmbedding::new(
        base,
        head_dim,
        max_positions,
        device,
        true,
        dtype,
    )?))
}

pub struct RopeSelfAttention {
    qkv: Arc<dyn QuantMethod>,
    out: Arc<dyn QuantMethod>,
    rotary: Arc<RotaryEmbedding>,
    num_heads: usize,
    head_dim: usize,
}

impl RopeSelfAttention {
    /// `qkv` holds q, k and v stacked along its output rows.
    pub fn new(
        hidden: usize,
        num_heads: usize,
        bias: bool,
        rotary: Arc<RotaryEmbedding>,
        qkv: ShardedVarBuilder,
        out: ShardedVarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            qkv: inference_quant::linear_b(hidden, 3 * hidden, bias, &None, qkv)?,
            out: inference_quant::linear_b(hidden, hidden, bias, &None, out)?,
            rotary,
            num_heads,
            head_dim: hidden / num_heads,
        })
    }

    pub fn forward(&self, xs: &Tensor, mask: &AttentionMask) -> Result<Tensor> {
        let (b_sz, seq_len, hidden) = xs.dims3()?;
        let qkv =
            self.qkv
                .forward(xs)?
                .reshape((b_sz, seq_len, 3, self.num_heads, self.head_dim))?;
        let part = |i| {
            qkv.narrow(2, i, 1)?
                .squeeze(2)?
                .transpose(1, 2)?
                .contiguous()
        };
        let (q, k, v) = (part(0)?, part(1)?, part(2)?);
        let (cos, sin) = self.rotary.get_cos_sin()?;
        let (q, k) = apply_rotary_preselected_qk(
            &q,
            &k,
            &cos.narrow(0, 0, seq_len)?,
            &sin.narrow(0, 0, seq_len)?,
            true,
        )?;
        let params = SdpaParams {
            n_kv_groups: 1,
            sliding_window: None,
            softcap: None,
            softmax_scale: 1.0 / (self.head_dim as f32).sqrt(),
            sinks: None,
            chunk: None,
        };
        let attn =
            Sdpa.run_attention(&q, &k, &v, mask, Some(&FlashParams::empty(false)), &params)?;
        self.out
            .forward(&attn.transpose(1, 2)?.reshape((b_sz, seq_len, hidden))?)
    }
}
