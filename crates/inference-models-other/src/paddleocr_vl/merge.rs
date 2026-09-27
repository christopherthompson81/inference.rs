//! Token embed + `masked_scatter(input_ids == image_token_id, image_embeds)`, as in transformers.

use std::sync::Arc;

use crate::layers::embedding;
use crate::utils::unvarbuilder::UnVarBuilder;
use candle_core::{DType, Result, Tensor};
use inference_quant::{QuantMethod, ShardedVarBuilder};

pub struct Merger {
    embed_tokens: Arc<dyn QuantMethod>,
    dtype: DType,
    image_token_id: i64,
}

impl Merger {
    pub fn load(
        vb: ShardedVarBuilder,
        vocab: usize,
        hidden: usize,
        image_token_id: i64,
    ) -> Result<Self> {
        Ok(Self {
            embed_tokens: embedding(vocab, hidden, vb.pp("embed_tokens"), &None)?,
            dtype: vb.dtype(),
            image_token_id,
        })
    }

    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens
            .embedding_forward(&input_ids.to_dtype(DType::U32)?, self.dtype)
    }

    // masked_scatter without a scatter op: index_select over cat([text ; image_embeds]), row-major fill order.
    pub fn forward(&self, input_ids: &Tensor, image_embeds: &Tensor) -> Result<Tensor> {
        let ids = input_ids.to_dtype(DType::I64)?.to_vec1::<i64>()?;
        let s = ids.len();
        let ids_u32: Vec<u32> = ids.iter().map(|&v| v as u32).collect();
        let idx_emb = Tensor::from_vec(ids_u32, s, input_ids.device())?;
        let text = self.embed_tokens.embedding_forward(&idx_emb, self.dtype)?;

        let mut gather = Vec::with_capacity(s);
        let mut img = 0u32;
        for (j, &id) in ids.iter().enumerate() {
            if id == self.image_token_id {
                gather.push(s as u32 + img);
                img += 1;
            } else {
                gather.push(j as u32);
            }
        }
        let combined = Tensor::cat(&[&text, image_embeds], 0)?;
        let gather = Tensor::from_vec(gather, s, input_ids.device())?;
        combined.index_select(&gather, 0)
    }

    pub fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.pp("model").pp("embed_tokens").add(&self.embed_tokens);
        uvb.to_safetensors()
    }
}
