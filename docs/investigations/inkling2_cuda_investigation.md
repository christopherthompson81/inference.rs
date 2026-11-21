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

## Run 2 - 2026-11-21 09:00

Decode was launch-bound (12,022 kernels a token). Each step fused a chain into one launch (`kernels/cuda/inkling.cu`,
`inference_nn::cuda::inkling`), measured with `inference bench` (pp512, tg128) and the kernels a token under nsys:

```
step                                                         tg128 tok/s   pp512 tok/s   kernels/token
Run 1                                                        16.4          1310          12,022
L2 clip, unit RMS, depth mix, short conv each one launch     41.0          1780           5,400
decode attention: lanes split the head (coalesced K/V, 4     40.1          1769
  keys a step); depth mix a warp a state
block stacks cached by tensor ids (the mix's states)         43.0          1813           5,041
controllers: unit-norm + concat in, tanh/forget/gain out     ~49           ...            3,985
  (the message is f32, x bf16: the kernels take each dtype)
softcapped SwiGLU fused (rows strided: gate/up halves)       50.9          2095           3,707
mix writes the model dtype; q/table/sinks read token-major   54.0          2266           3,292
  (no transposes); conv writes the model dtype; the
  attention's sigmoid gate fused with its cast
depth mix: each thread a few columns of every state, one     54.5          2200
  block reduction; decode K/V reshaped, not copied
experts' decode on the indexed matvec: SoftcapSilu has no    65.3          2206
  fused GLU type, so the matvec ran grouped mmq at decode
  (5.2 ms a token); now the gate's and up's rows come from
  their own matvecs and the activation runs between
decode attention 32 warps a query (from 8)                   65.5
llama.cpp (llama-bench)                                      61.1          1486
```

At 4,096 positions: tg128 59.4 against llama.cpp's 59.2 (56.5 with 8 decode warps).

Greedy against `llama-completion` after the steps: the code prompt (269 chars) and a science prompt (524 chars)
identical; the printing-press prompt splits at char 95 (a near-tie). Before the fusing, the unfused binary split from
llama.cpp on the science prompt at char 384, and the fused one first took a 0.012-logprob tie the other way on the
printing-press prompt; both are ties, not drift.

Tests: `unfused_activation_indexed_decode_matches_gather` (the gated-pair indexed decode with SoftcapSilu against the
gather path, Q4_0/Q5_0 and Q4_K/Q8_0; GgmlDType has no IQ4_NL to build test weights from), the attention cases with
token-major inputs, the tiny model on CUDA.

Left: the projections each quantize their input to Q8_1 (q, r, gate and sink read the same input: ~3 extra
quantizes a slot), the conv state rebuilt by a 4-way concat a slot, the f32 router/controller gemvs on cuBLAS, and
CUDA graphs (the engine's ride the paged cache, which Inkling does not take).
