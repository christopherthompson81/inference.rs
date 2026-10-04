# inference-fattn

llama.cpp's CUDA FlashAttention kernels (`fattn`: mma, tile and vector variants) behind a candle-facing API. The
kernels and their dispatch are vendored nearly verbatim over a small ggml compatibility layer, so they can be
re-synced with upstream; see `third_party/README.md`.
