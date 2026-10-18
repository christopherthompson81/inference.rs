use inference_tensor::Result;

use super::Engine;
use crate::{
    get_mut_arcmutex,
    paged_attention::block_hash::compute_block_hashes,
    request::{NormalRequest, Request, RequestMessage},
    sampler::SamplingParams,
};

// One greedy token: the warm-up is for its prefill, which snapshots the reply's last full block
const WARMUP_MAX_TOKENS: usize = 1;
// Responses a warm-up sends before its channel closes; they are drained and dropped
const WARMUP_RESPONSE_CAPACITY: usize = 4;

impl Engine {
    /// Prefills each finished reply on top of its prompt's snapshot, while the client acts on the reply (an agent runs
    /// its tool), so the next turn resumes after the reply instead of recomputing it.
    pub(super) fn queue_reply_warmups(&self, replies: Vec<(Vec<u32>, usize)>, block_size: usize) {
        for (tokens, prompt_tokens) in replies {
            if let Err(error) = self.queue_reply_warmup(tokens, prompt_tokens, block_size) {
                tracing::debug!("Skipped prefilling a finished reply: {error}");
            }
        }
    }

    fn queue_reply_warmup(
        &self,
        tokens: Vec<u32>,
        prompt_tokens: usize,
        block_size: usize,
    ) -> Result<()> {
        // only a prefill of the reply alone is worth its GPU time: the prompt's snapshot has to be on the GPU
        let hashes = compute_block_hashes(&tokens, block_size, &[], &[]);
        let prompt_blocks = prompt_tokens.saturating_sub(1) / block_size;
        let held = get_mut_arcmutex!(self.prefix_cacher)
            .peek_longest_paged_recurrent_prefix(&hashes, prompt_blocks)
            .map_or(0, |(n_blocks, _)| n_blocks);
        if prompt_blocks == 0 || held < prompt_blocks {
            return Ok(());
        }
        let (response, mut drain) = tokio::sync::mpsc::channel(WARMUP_RESPONSE_CAPACITY);
        let mut sampling = SamplingParams::deterministic();
        sampling.max_len = Some(WARMUP_MAX_TOKENS);
        let mut request = NormalRequest::new_simple(
            RequestMessage::CompletionTokens(tokens),
            sampling,
            response,
            0,
            None,
            None,
        );
        request.prefix_warmup = true;
        self.tx
            .try_send(Request::Normal(Box::new(request)))
            .map_err(|error| inference_tensor::Error::msg(format!("engine queue: {error}")))?;
        tokio::spawn(async move { while drain.recv().await.is_some() {} });
        metrics::counter!("inference_prefix_reply_warmups_total").increment(1);
        Ok(())
    }
}
