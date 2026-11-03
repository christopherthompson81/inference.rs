/// One `PagedAttention::attend_selected` call: `(b, heads, s, head_dim)` queries, the `(b * s, n_kv)` additive f16
/// mask (causality included), and each sequence's cached rows, `(b,)` u32.
pub struct SelectedAttention<'a> {
    pub query: &'a inference_tensor::Tensor,
    pub mask: &'a inference_tensor::Tensor,
    pub key_cache: &'a inference_tensor::Tensor,
    pub value_cache: &'a inference_tensor::Tensor,
    pub input_metadata: &'a crate::paged_attention::PagedAttentionInputMetadata,
    pub seq_lens: &'a inference_tensor::Tensor,
    pub sdpa_params: &'a crate::attention::SdpaParams,
}

/// One `PagedAttention::attend_gathered` call: `(rows, heads, 1, head_dim)` queries, each over the leading `kept[row]`
/// of its row of `cells` (`(rows, width)` u32 positions, read through `block_table`'s row), all `(rows,)` u32 for
/// `kept`; `compact_table` is the `(rows, pages)` u32 identity table of the compact cache the cells go into.
pub struct GatheredAttention<'a> {
    pub query: &'a inference_tensor::Tensor,
    pub cells: &'a inference_tensor::Tensor,
    pub kept: &'a inference_tensor::Tensor,
    pub block_table: &'a inference_tensor::Tensor,
    pub compact_table: &'a inference_tensor::Tensor,
    pub key_cache: &'a inference_tensor::Tensor,
    pub value_cache: &'a inference_tensor::Tensor,
    pub sdpa_params: &'a crate::attention::SdpaParams,
}

pub mod paged_attention;
pub use paged_attention::PagedAttention;
