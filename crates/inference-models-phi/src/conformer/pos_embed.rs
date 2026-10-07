#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use inference_quant::ShardedVarBuilder;
use inference_tensor::nn::{Embedding, Module};
use inference_tensor::{DType, Result, Tensor};

use crate::layers;

pub struct T5RelativeAttentionLogitBias {
    bias_values: Embedding,
    skip_bucketing: bool,
    max_distance: usize,
    symmetric: bool,
}

impl T5RelativeAttentionLogitBias {
    pub fn new(
        num_heads: usize,
        num_buckets: Option<usize>,
        max_distance: usize,
        symmetric: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let skip_bucketing = num_buckets.is_none();
        let mut num_buckets = num_buckets.unwrap_or(max_distance);
        if !symmetric {
            num_buckets *= 2;
        }

        Ok(Self {
            bias_values: layers::dense_embedding(
                num_buckets,
                num_heads,
                vb.pp("bias_values"),
                &None,
            )?,
            skip_bucketing,
            symmetric,
            max_distance,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let maxpos = x.dim(1)?;
        let device = x.device();

        // Create position matrices
        let context_position = Tensor::arange(0f32, maxpos as f32, device)?.unsqueeze(1)?;
        let memory_position = Tensor::arange(0f32, maxpos as f32, device)?.unsqueeze(0)?;

        // Calculate relative positions
        let relative_position = memory_position.broadcast_sub(&context_position)?;

        // Clip to max distance (equivalent to Python's masked_fill)
        let max_dist = self.max_distance as i64;
        let relative_position = relative_position.clamp(-max_dist, max_dist - 1)?;

        // Map to bias indices
        let bias_idx = if self.skip_bucketing {
            relative_position
        } else {
            unimplemented!("require skip_bucketing");
        };

        let bias_idx = if self.symmetric {
            bias_idx.abs()?
        } else {
            let offset = (self.bias_values.embeddings().dim(0)? / 2) as i64;
            (bias_idx + offset as f64)?
        };

        // Ensure bias_idx is the right type for embedding lookup
        let bias_idx = bias_idx.to_dtype(DType::U32)?;

        // Get bias values
        let t5_rel_att_bias = self.bias_values.forward(&bias_idx)?; // [L, L, H]
        t5_rel_att_bias.permute((2, 0, 1))?.unsqueeze(0) // [1, H, L, L]
    }
}
