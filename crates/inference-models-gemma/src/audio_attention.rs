#![allow(clippy::cast_possible_truncation)]
//! The chunked local self-attention of the Gemma 3n and Gemma 4 audio conformers, with its relative position bias.
//! Each model keeps its projections, query and key scaling and timescales, and hands the scaled states here.

use std::sync::Arc;

use inference_quant::QuantMethod;
use inference_tensor::{D, DType, Device, Result, Tensor, bail};

pub struct RelativePositionEmbedding {
    num_heads: usize,
    head_dim: usize,
    pub pos_proj: Arc<dyn QuantMethod>,
    pub(crate) inv_timescales: Tensor,
    pub(crate) pos_indices: Tensor,
}

impl RelativePositionEmbedding {
    /// `inv_timescales` is `(1, 1, channels / 2)`; positions run from `max_backward` down to `-max_forward`.
    pub fn new(
        num_heads: usize,
        head_dim: usize,
        pos_proj: Arc<dyn QuantMethod>,
        inv_timescales: Tensor,
        max_backward: usize,
        max_forward: usize,
    ) -> Result<Self> {
        let pos_values = (-(max_forward as i64)..=max_backward as i64)
            .rev()
            .collect::<Vec<_>>();
        let pos_indices = Tensor::from_vec(
            pos_values,
            (1, max_backward + max_forward + 1),
            inv_timescales.device(),
        )?;
        Ok(Self {
            num_heads,
            head_dim,
            pos_proj,
            inv_timescales,
            pos_indices,
        })
    }

    fn timing_signal(&self, position: &Tensor, dtype: DType) -> Result<Tensor> {
        let position = position.to_dtype(DType::F32)?.unsqueeze(D::Minus1)?;
        let inv_timescales = self.inv_timescales.to_device(position.device())?;
        let scaled_time = position.broadcast_mul(&inv_timescales)?;
        Tensor::cat(&[scaled_time.sin()?, scaled_time.cos()?], D::Minus1)?.to_dtype(dtype)
    }

    // `[B, N, U, W, span]` position logits shifted onto the `context` keys each query block sees
    fn relative_shift(term_bd: &Tensor, context: usize) -> Result<Tensor> {
        let (b, n, u, w, span) = term_bd.dims5()?;
        term_bd
            .pad_with_zeros(D::Minus1, 0, (context + 1) - span)?
            .reshape((b, n, u, w * (context + 1)))?
            .narrow(D::Minus1, 0, w * context)?
            .reshape((b, n, u, w, context))
    }

    /// Content plus position logits `[B, N, U, W, C]` of queries `[B, U, W, N, H]` over keys `[B, U, C, N, H]`.
    pub fn forward(&self, queries: &Tensor, keys: &Tensor) -> Result<Tensor> {
        let (batch_size, num_query_blocks, query_block_size, num_heads, head_dim) =
            queries.dims5()?;
        let keys = match keys.dim(1)? {
            n if n == num_query_blocks => keys.clone(),
            n if n > num_query_blocks => keys.narrow(1, 0, num_query_blocks)?,
            n => bail!("Keys have fewer blocks than queries: {n} < {num_query_blocks}"),
        };
        let key_context_size = keys.dim(2)?;
        let pos_indices = self.pos_indices.to_device(queries.device())?;
        let max_span_plus_1 = pos_indices.dim(1)?;

        let timing_signal = self.timing_signal(&pos_indices, queries.dtype())?;
        let sin_emb = self
            .pos_proj
            .forward(&timing_signal)?
            .reshape((1, max_span_plus_1, self.num_heads, self.head_dim))?
            .squeeze(0)?
            .to_dtype(queries.dtype())?;

        let blocks = batch_size * num_heads * num_query_blocks;
        let queries_3d = queries
            .transpose(1, 3)?
            .transpose(2, 3)?
            .contiguous()?
            .reshape((blocks, query_block_size, head_dim))?;
        let keys_3d = keys
            .transpose(1, 3)?
            .transpose(2, 3)?
            .transpose(3, 4)?
            .contiguous()?
            .reshape((blocks, head_dim, key_context_size))?;
        let term_ac = queries_3d.matmul(&keys_3d)?.reshape((
            batch_size,
            num_heads,
            num_query_blocks,
            query_block_size,
            key_context_size,
        ))?;

        let q_reshaped = queries.transpose(1, 3)?.transpose(2, 3)?.reshape((
            batch_size * num_heads,
            num_query_blocks * query_block_size,
            head_dim,
        ))?;
        let s_broadcast = sin_emb
            .transpose(0, 2)?
            .transpose(0, 1)?
            .unsqueeze(0)?
            .broadcast_as((batch_size, num_heads, head_dim, max_span_plus_1))?
            .reshape((batch_size * num_heads, head_dim, max_span_plus_1))?
            .contiguous()?;
        let term_bd = q_reshaped.contiguous()?.matmul(&s_broadcast)?.reshape((
            batch_size,
            num_heads,
            num_query_blocks,
            query_block_size,
            max_span_plus_1,
        ))?;

        term_ac.broadcast_add(&Self::relative_shift(&term_bd, key_context_size)?)
    }
}

// `[chunk, context]`: 1 where key `j` of a block's context window is within reach of query `i`
fn local_causal_valid_mask(
    chunk_size: usize,
    context_size: usize,
    diag_offset: usize,
    device: &Device,
) -> Result<Tensor> {
    let mut mask = vec![0u8; chunk_size * context_size];
    for i in 0..chunk_size {
        for j in i..context_size.min(i + diag_offset + 1) {
            mask[i * context_size + j] = 1;
        }
    }
    Tensor::from_vec(mask, (chunk_size, context_size), device)
}

/// Attention of each `chunk_size` block of queries over its block plus `context_left - 1` frames back and
/// `context_right` ahead, with logits soft-capped and padded or out-of-reach keys masked.
pub struct LocalAttention {
    num_heads: usize,
    head_dim: usize,
    chunk_size: usize,
    max_future_horizon: usize,
    max_past_horizon: usize,
    context_size: usize,
    pub(crate) local_causal_valid_mask: Tensor,
    softcap: f64,
    pub(crate) invalid_logits_tensor: Tensor,
}

impl LocalAttention {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num_heads: usize,
        head_dim: usize,
        chunk_size: usize,
        context_left: usize,
        context_right: usize,
        softcap: f64,
        invalid_logits_value: f64,
        device: &Device,
    ) -> Result<Self> {
        let max_past_horizon = context_left.saturating_sub(1);
        let max_future_horizon = context_right;
        let context_size = chunk_size + max_past_horizon + max_future_horizon;
        Ok(Self {
            num_heads,
            head_dim,
            chunk_size,
            max_future_horizon,
            max_past_horizon,
            context_size,
            local_causal_valid_mask: local_causal_valid_mask(
                chunk_size,
                context_size,
                max_past_horizon + max_future_horizon,
                device,
            )?,
            softcap,
            invalid_logits_tensor: Tensor::new(invalid_logits_value as f32, device)?,
        })
    }

    fn convert_to_block(&self, x: &Tensor) -> Result<Tensor> {
        let shape = x.dims();
        let (b, t) = (shape[0], shape[1]);
        let num_blocks = t.div_ceil(self.chunk_size);
        let x = x.pad_with_zeros(1, 0, num_blocks * self.chunk_size - t)?;
        let mut new_shape = vec![b, num_blocks, self.chunk_size];
        new_shape.extend_from_slice(&shape[2..]);
        x.reshape(new_shape)
    }

    // `[B, U, context, ...]`: each block's context window over the time axis
    fn extract_block_context(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.pad_with_zeros(
            1,
            self.max_past_horizon,
            self.max_future_horizon + self.chunk_size - 1,
        )?;
        let num_windows = (x.dim(1)? - self.context_size) / self.chunk_size + 1;
        let mut windows = Vec::with_capacity(num_windows);
        for i in 0..num_windows {
            windows.push(x.narrow(1, i * self.chunk_size, self.context_size)?);
        }
        Tensor::stack(&windows, 1)
    }

    /// Scaled query, key and value states `[B, T, N, H]` and the padding mask `[B, T]` (nonzero where padded) to
    /// context vectors `[B, T, N, H]`.
    pub fn forward(
        &self,
        relative_position_embedding: &RelativePositionEmbedding,
        query_states: &Tensor,
        key_states: &Tensor,
        value_states: &Tensor,
        mask: &Tensor,
    ) -> Result<Tensor> {
        let device = query_states.device();
        let local_causal_valid_mask = self.local_causal_valid_mask.to_device(device)?;
        let invalid_logits_tensor = self.invalid_logits_tensor.to_device(device)?;
        let (batch_size, q_time) = (query_states.dim(0)?, query_states.dim(1)?);

        let query_blocks = self.convert_to_block(query_states)?;
        let mut key_blocks = self.extract_block_context(key_states)?;
        let mut value_blocks = self.extract_block_context(value_states)?;
        let num_query_blocks = query_blocks.dim(1)?;
        let num_key_blocks = key_blocks.dim(1)?;
        if num_key_blocks < num_query_blocks {
            bail!("Keys have fewer blocks than queries: {num_key_blocks} < {num_query_blocks}");
        }
        if num_key_blocks > num_query_blocks {
            key_blocks = key_blocks.narrow(1, 0, num_query_blocks)?;
            value_blocks = value_blocks.narrow(1, 0, num_query_blocks)?;
        }

        // [B, U, C] validity of each block's keys, against the [W, C] reach of each query
        let valid_mask_blocks = self.extract_block_context(&mask.eq(0.0)?.to_dtype(DType::U8)?)?;
        let condition = valid_mask_blocks
            .unsqueeze(1)?
            .unsqueeze(3)?
            .broadcast_mul(
                &local_causal_valid_mask
                    .unsqueeze(0)?
                    .unsqueeze(0)?
                    .unsqueeze(0)?,
            )?;

        let logits = relative_position_embedding.forward(&query_blocks, &key_blocks)?;
        let logits = ((logits / self.softcap)?.tanh()? * self.softcap)?;
        let condition = condition.broadcast_as(logits.shape())?;
        let invalid_logits = invalid_logits_tensor
            .broadcast_as(logits.shape())?
            .to_dtype(logits.dtype())?;
        let probabilities = inference_tensor::nn::ops::softmax_last_dim(
            &condition
                .where_cond(&logits, &invalid_logits)?
                .to_dtype(DType::F32)?,
        )?
        .to_dtype(value_blocks.dtype())?;

        let (b_dim, n_dim, u_dim, w_dim, c_dim) = probabilities.dims5()?;
        let h_dim = value_blocks.dim(D::Minus1)?;
        let probabilities = probabilities.permute((0, 2, 1, 3, 4))?.reshape((
            b_dim * u_dim * n_dim,
            w_dim,
            c_dim,
        ))?;
        let value_blocks = value_blocks.permute((0, 1, 3, 2, 4))?.reshape((
            b_dim * u_dim * n_dim,
            c_dim,
            h_dim,
        ))?;
        probabilities
            .matmul(&value_blocks)?
            .reshape((b_dim, u_dim, n_dim, w_dim, h_dim))?
            .permute((0, 1, 3, 2, 4))?
            .reshape((
                batch_size,
                num_query_blocks * self.chunk_size,
                self.num_heads,
                self.head_dim,
            ))?
            .narrow(1, 0, q_time)
    }
}

#[cfg(test)]
pub(crate) mod pin {
    use inference_nn::testing::fill;
    use inference_tensor::{Result, Tensor};

    const FRAMES: usize = 11;
    const PADDED_FRAMES: usize = 3;
    const TOL: f32 = 1e-5;

    /// Output probes, sum and L2 norm of an attention `forward(x, mask)`.
    pub(crate) struct Expected {
        pub probes: [f32; 4],
        pub sum: f32,
        pub l2: f32,
    }

    fn close(actual: f32, expected: f32) -> bool {
        (actual - expected).abs() <= TOL * expected.abs().max(1.0)
    }

    // an 11-frame clip whose last 3 frames are padding
    pub(crate) fn assert_padded_clip(
        hidden_size: usize,
        forward: impl Fn(&Tensor, &Tensor) -> Result<Tensor>,
        expected: &Expected,
    ) -> anyhow::Result<()> {
        let x = fill("input", &[1, FRAMES, hidden_size])?;
        let mask = Tensor::from_vec(
            (0..FRAMES)
                .map(|t| u8::from(t >= FRAMES - PADDED_FRAMES))
                .collect::<Vec<_>>(),
            (1, FRAMES),
            x.device(),
        )?;
        let out = forward(&x, &mask)?;
        let flat = out.flatten_all()?.to_vec1::<f32>()?;
        let probes = [flat[0], flat[37], flat[100], flat[flat.len() - 1]];
        let sum = flat.iter().sum::<f32>();
        let l2 = flat.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            probes
                .iter()
                .zip(expected.probes)
                .all(|(a, e)| close(*a, e))
                && close(sum, expected.sum)
                && close(l2, expected.l2),
            "attention moved: {:?} {probes:?} {sum:?} {l2:?}",
            out.dims()
        );
        Ok(())
    }
}
