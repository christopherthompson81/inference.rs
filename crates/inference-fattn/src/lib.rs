//! llama.cpp's FlashAttention kernels (`ggml-cuda/fattn*`): mma, tile and vector variants behind one entry point.

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "cuda")]
pub use cuda::{
    causal_mask, flash_attn, flash_attn_paged, flash_attn_paged_varlen, flash_attn_varlen,
    mma_available, paged_causal_mask, paged_kv_len, supported, supported_paged,
    supported_paged_varlen, supported_varlen, varlen_causal_mask, varlen_kv_len,
};

/// Options for `flash_attn` beyond the operands.
#[derive(Debug, Clone, Default)]
pub struct FattnOptions {
    /// Applied to `q @ k^T` before the softmax.
    pub scale: f32,
    /// `softcap * tanh(x / softcap)` on the scores; 0 disables it.
    pub softcap: f32,
    /// Additive f16 mask `(batch | 1, seq_q, seq_kv)`; `-inf` hides a position. `causal` needs no tensor.
    pub mask: Option<candle_core::Tensor>,
    /// Per-head f32 attention sinks `(n_head,)`: an extra logit that takes softmax mass but contributes no value.
    pub sinks: Option<candle_core::Tensor>,
    /// Dequantization scales of fp8 e4m3 K and V (`x * scale`); `None` is 1.0.
    pub kv_scales: Option<KvScales>,
    /// Causal masking without a mask tensor: each sequence's queries are its last positions, so every sequence needs
    /// at least as many keys as queries (a sequence with no visible key comes out NaN). Excludes `mask`. Runs on the
    /// mma kernel, except a single query with no window, which needs no mask at all.
    pub causal: bool,
    /// With `causal`, each query also sees only the `window_left` keys before it (FA2's `window_size_left`); tiles
    /// wholly before the window are still computed.
    pub window_left: Option<usize>,
}

/// Per-tensor scales of an fp8 K/V cache, in (0, 146]: dequantized values must stay inside f16's range.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KvScales {
    pub k: f32,
    pub v: f32,
}

impl Default for KvScales {
    fn default() -> Self {
        Self { k: 1., v: 1. }
    }
}

/// A paged K/V cache read in place: `(num_blocks, n_head_kv, block_size, head_dim)` blocks, as FlashInfer's HND layout.
#[derive(Debug, Clone, Copy)]
pub struct PagedKv<'a> {
    pub k_cache: &'a candle_core::Tensor,
    pub v_cache: &'a candle_core::Tensor,
    /// `(batch, max_blocks)` u32: each sequence's blocks in order; entries covering its rows must be `< num_blocks`.
    pub block_table: &'a candle_core::Tensor,
    /// `(batch,)` u32: the rows each sequence holds, at least 1 (unused rows read the sequence's first row).
    pub seq_lens: &'a candle_core::Tensor,
}

/// Sequences packed along dim 0 of a `(total, heads, dim)` tensor.
#[derive(Debug, Clone, Copy)]
pub struct Packed<'a> {
    /// `(batch + 1,)` u32: sequence `i` holds rows `cu_seqlens[i]..cu_seqlens[i + 1]`, from 0 up to the total. For
    /// K/V every sequence holds at least one row (rows past a sequence's end read its first row).
    pub cu_seqlens: &'a candle_core::Tensor,
    /// The longest sequence, at least each sequence's length (it sizes the kernel's grid).
    pub max_len: usize,
}
