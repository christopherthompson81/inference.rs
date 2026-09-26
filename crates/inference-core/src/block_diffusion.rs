//! Block-diffusion text generation support (e.g. DiffusionGemma): models that commit a
//! whole denoised block of tokens per engine step instead of sampling one token from logits.

pub use crate::model::BlockDiffusionMixin;
use std::sync::Arc;

use tokenizers::Tokenizer;
use tokio::sync::mpsc::Sender;

use crate::{response::BlockDenoisingProgress, sequence::Sequence, Response};

#[derive(Clone)]
pub(crate) struct BlockDenoisingProgressEmitter {
    batch_index: usize,
    response_index: usize,
    tokenizer: Arc<Tokenizer>,
    response: Sender<Response>,
}

impl BlockDenoisingProgressEmitter {
    pub(crate) fn emit(
        &self,
        step: usize,
        total_steps: usize,
        tokens: &[u32],
        finished: bool,
        final_block: bool,
    ) {
        let text = self
            .tokenizer
            .decode(tokens, false)
            .map(|text| text.replace('\u{2581}', " "))
            .unwrap_or_default();
        let _ = self
            .response
            .try_send(Response::BlockDenoisingProgress(BlockDenoisingProgress {
                index: self.response_index,
                step,
                total_steps,
                tokens: tokens.to_vec(),
                text,
                finished,
                final_block,
            }));
    }

    pub(crate) fn batch_index(&self) -> usize {
        self.batch_index
    }
}

pub(crate) fn block_denoising_progress_emitters(
    tokenizer: Option<Arc<Tokenizer>>,
    input_seqs: &[&mut Sequence],
    seq_indices: &[usize],
    return_raw_logits: bool,
) -> Option<Vec<BlockDenoisingProgressEmitter>> {
    if return_raw_logits {
        return None;
    }
    let tokenizer = tokenizer?;
    let emitters = seq_indices
        .iter()
        .enumerate()
        .filter_map(|(batch_index, seq_idx)| {
            let seq = input_seqs.get(*seq_idx)?;
            if !seq.get_mut_group().is_streaming {
                return None;
            }
            Some(BlockDenoisingProgressEmitter {
                batch_index,
                response_index: seq.get_response_index(),
                tokenizer: tokenizer.clone(),
                response: seq.responder(),
            })
        })
        .collect::<Vec<_>>();
    (!emitters.is_empty()).then_some(emitters)
}
