# Inkling 2 Mini on CUDA

Inkling 2 Mini (`inkling2_loop`, #1035 brought it up on Vulkan) on the 3090: ggml-org's IQ4_NL GGUF (17.6 GB; 24 heads
over 8 KV heads of 128, 48 cache slots from the looped core, bias extent and sliding window 1024), against llama.cpp
master (44c20b1c0, `build_cuda`).

## Run 1 - 2026-11-21 07:30

First load on CUDA failed: `dtype mismatch in matmul, lhs: F32, rhs: BF16`. The router and the loop controllers take
f32 inputs, but their weights (stored F32 in the GGUF) were loaded in the model's dtype (BF16 on CUDA; Vulkan ran
F32). Both now load as f32.

Attention: a fused CUDA kernel (`kernels/cuda/biased_attention.cu`): the bias looked up by distance from the
`(b, h, t, extent + 1)` table, the causal mask and the window computed in the kernel, the sink starting each
softmax; a prefill kernel (a block a tile of 8 queries, K/V through shared memory 32 keys at a time) and a decode
kernel (a block a query, its 8 warps splitting the keys, merged in shared memory). It replaces materializing the bias
per (head, query, key) and the mask; `RelativeInputs` is no longer built on CUDA. Tests: six shapes against the
materialized form (prefill, after a past, windowed, decode, windowed decode, short multi-row decode), and the tiny
model's decode-vs-prefill and CPU-vs-CUDA long prompt (head size raised 16 -> 32 so the kernel takes it).

Greedy, raw prompt "The history of the printing press begins in", 64 tokens: identical text to `llama-completion
--temp 0 --top-k 1`.

```
                         ours (serve/bench, --paged-attn off)    llama.cpp (llama-bench)
pp512                    1310 tok/s                              1486 tok/s
tg128 (short context)    16.4 tok/s                              61.1 tok/s
```

nsys over a decode window: 12,022 kernels a token, the GPU busy 32% of it (101 ms a token under the profiler, 33 ms
of kernels). Top: the experts' IQ4_NL `mul_mat_q` 5.1 ms (131 a token), `bmul_f32` 4.1 ms (1,420), the Q8_0
projections' mmvq 2.2 ms, `affine_f32` 2.0 ms (1,504), `cast_bf16_f32` 1.9 ms (756), `bdiv_f32` 1.6 ms, the fused
attention 1.5 ms (48), `badd_f32` 1.5 ms. Decode is launch-bound: ~250 small f32 ops a slot (L2 clips, unit RMS,
the depth mixes' softmax, short convolutions, controllers, casts). CUDA graphs in the engine ride the paged cache,
which Inkling does not take yet.
