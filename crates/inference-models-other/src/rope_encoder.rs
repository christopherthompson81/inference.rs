//! Bidirectional self-attention with a fused QKV projection and rotary positions, shared by the RoPE encoders.
#![allow(clippy::cast_precision_loss)]

use std::sync::Arc;

use inference_quant::{QuantMethod, ShardedVarBuilder};
use inference_tensor::{DType, Device, Result, Tensor};

use crate::{
    attention::{AttentionMask, FlashParams, SdpaParams},
    embedding::{TaskLora, TaskRows},
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
    Ok(Arc::new(RotaryEmbedding::new(
        base,
        head_dim,
        max_positions,
        device,
        true,
        dtype,
    )?))
}

/// How the fused projection orders its output rows.
#[derive(Clone, Copy)]
pub enum QkvLayout {
    /// All of q, then k, then v.
    Stacked,
    /// q, k and v of head 0, then of head 1, and so on.
    PerHead,
}

/// Which dims a rotation pairs.
#[derive(Clone, Copy)]
pub enum RopeStyle {
    /// The two halves of the head (transformers' rotate_half).
    NeoX,
    /// Adjacent pairs (Llama 1's complex rotation).
    Interleaved,
}

pub struct RopeSelfAttention {
    qkv: Arc<dyn QuantMethod>,
    out: Arc<dyn QuantMethod>,
    rotary: Arc<RotaryEmbedding>,
    num_heads: usize,
    head_dim: usize,
    layout: QkvLayout,
    style: RopeStyle,
    // per-task adapters on the fused qkv and the output projection (Jina v3)
    lora: Option<(TaskLora, TaskLora)>,
}

/// The shape of one attention block.
#[derive(Clone, Copy)]
pub struct RopeAttentionShape {
    pub hidden: usize,
    pub num_heads: usize,
    pub bias: bool,
    pub layout: QkvLayout,
    pub style: RopeStyle,
}

impl RopeSelfAttention {
    pub fn new(
        shape: &RopeAttentionShape,
        rotary: Arc<RotaryEmbedding>,
        qkv: ShardedVarBuilder,
        out: ShardedVarBuilder,
    ) -> Result<Self> {
        let hidden = shape.hidden;
        Ok(Self::from_projections(
            shape,
            rotary,
            inference_quant::linear_b(hidden, 3 * hidden, shape.bias, &None, qkv)?,
            inference_quant::linear_b(hidden, hidden, shape.bias, &None, out)?,
            None,
        ))
    }

    /// Over projections the caller built, with optional task adapters on each.
    pub fn from_projections(
        shape: &RopeAttentionShape,
        rotary: Arc<RotaryEmbedding>,
        qkv: Arc<dyn QuantMethod>,
        out: Arc<dyn QuantMethod>,
        lora: Option<(TaskLora, TaskLora)>,
    ) -> Self {
        Self {
            qkv,
            out,
            rotary,
            num_heads: shape.num_heads,
            head_dim: shape.hidden / shape.num_heads,
            layout: shape.layout,
            style: shape.style,
            lora,
        }
    }

    pub fn forward(&self, xs: &Tensor, mask: &AttentionMask) -> Result<Tensor> {
        self.forward_tasks(xs, mask, None)
    }

    /// As `forward`, adding each row's task adapter where the block has them.
    pub fn forward_tasks(
        &self,
        xs: &Tensor,
        mask: &AttentionMask,
        tasks: Option<&TaskRows>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, hidden) = xs.dims3()?;
        let adapted = |proj: &Arc<dyn QuantMethod>, lora: Option<&TaskLora>, x: &Tensor| {
            let y = proj.forward(x)?;
            match (lora, tasks) {
                (Some(lora), Some(rows)) => y + lora.delta(x, rows)?,
                _ => Ok(y),
            }
        };
        let (qkv_lora, out_lora) = match &self.lora {
            Some((qkv, out)) => (Some(qkv), Some(out)),
            None => (None, None),
        };
        let qkv = adapted(&self.qkv, qkv_lora, xs)?;
        let (h, d) = (self.num_heads, self.head_dim);
        // (b, seq, 3, heads, d) or (b, seq, heads, 3, d), sliced on the `3` axis
        let (qkv, axis) = match self.layout {
            QkvLayout::Stacked => (qkv.reshape((b_sz, seq_len, 3, h, d))?, 2),
            QkvLayout::PerHead => (qkv.reshape((b_sz, seq_len, h, 3, d))?, 3),
        };
        let part = |i| {
            qkv.narrow(axis, i, 1)?
                .squeeze(axis)?
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
            matches!(self.style, RopeStyle::NeoX),
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
        adapted(
            &self.out,
            out_lora,
            &attn.transpose(1, 2)?.reshape((b_sz, seq_len, hidden))?,
        )
    }
}
