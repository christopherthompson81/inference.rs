//! What an MTP head's drafter forward takes and gives, eager or replayed from a graph.

use inference_tensor::Tensor;

use crate::paged_attention::PagedAttentionInputMetadata;

/// One MTP forward's outputs: what `lm_head` reads, and what the next chained draft takes as its hidden state.
pub struct MtpDraft {
    pub logits_hidden: Tensor,
    pub chain_hidden: Tensor,
}

/// One drafter forward's device inputs: `[b, rows]` tokens, the head's own positions when it stages them (Qwen3.5's
/// `[3, b, rows]` M-RoPE), and `[b, rows, hidden]` target hidden states, with the rows' paged metadata and starts.
pub struct MtpForwardInputs<'a> {
    pub tokens: &'a Tensor,
    pub positions: Option<&'a Tensor>,
    pub target_hidden: &'a Tensor,
    pub metadata: &'a PagedAttentionInputMetadata,
    pub starts: &'a [usize],
}
