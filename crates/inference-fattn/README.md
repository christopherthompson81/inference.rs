# inference-fattn

llama.cpp's CUDA FlashAttention kernels (`fattn`: mma, tile and vector variants) behind a candle-facing API. The
kernels and their dispatch are vendored over a small ggml compatibility layer, with local changes kept small enough to
re-sync with upstream; `third_party/README.md` lists them.
