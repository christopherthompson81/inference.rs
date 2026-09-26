# Third-party provenance

Kernel sources in this crate that come from other projects. They are adapted and maintained here, so they live in
`kernels/`; this file records where they came from.

| in-tree path | upstream | revision | license | status |
|---|---|---|---|---|
| `kernels/cuda/` (FlashAttention-2 headers, per-hdim `flash_fwd_*.cu`) | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) v2, via [candle-flash-attn](https://github.com/huggingface/candle) | not recorded | BSD-3-Clause | adapted: paged split-KV (`block_table`), hdim 512, `FLASHATTENTION_DISABLE_*` builds |
| `kernels/cuda/flash_api.cu`, `kernels.h`, `kernel_helpers.h`, `error.h` | [huggingface/candle](https://github.com/huggingface/candle) `candle-flash-attn` | not recorded | MIT OR Apache-2.0 | adapted: paged and softcap arguments |
