//! llama.cpp's FlashAttention kernels (`ggml-cuda/fattn*`): mma, tile and vector variants behind one entry point.

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "cuda")]
pub use cuda::{causal_mask, flash_attn, supported};

/// Options for `flash_attn` beyond the operands.
#[derive(Debug, Clone, Default)]
pub struct FattnOptions {
    /// Applied to `q @ k^T` before the softmax.
    pub scale: f32,
    /// `softcap * tanh(x / softcap)` on the scores; 0 disables it.
    pub softcap: f32,
    /// Additive f16 mask `(batch | 1, seq_q, seq_kv)`; `-inf` hides a position. Causal attention passes one too.
    pub mask: Option<candle_core::Tensor>,
    /// Per-head f32 attention sinks `(n_head,)`: an extra logit that takes softmax mass but contributes no value.
    pub sinks: Option<candle_core::Tensor>,
}
