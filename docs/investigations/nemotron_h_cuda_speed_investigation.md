# Nemotron-H CUDA speed

Nemotron 3.5 Transcribe's text GGUF (IQ4_NL up, Q5_0 down, Q8_0 rest, 19.30 GiB) on an RTX 3090, against llama.cpp
970b51ce7 on the same file. Builds are the dev profile with debug assertions and overflow checks off.

## Run 1 - 2026-11-02 ~03:20

**Question:** where do we stand on the GPU?

**Commands:** `llama-bench -m <text gguf> -p 512,2048 -n 256 -ngl 99 -fa 1`; ours `inference serve --max-seq-len 8192
--format gguf -m <dir> -f <text gguf>` with a script sending a 539-token prompt three times, a 2436-token one twice
(max_tokens 1) and 256 decoded tokens twice (`ignore_eos`).

**llama.cpp:** pp512 4342 +- 136, pp2048 4345 +- 42, tg256 198.2 tok/s.

**Ours:** 539-token prompt 4856 tok/s warm (651 cold). The 2436-token prompt was refused: "CUDA memory pressure
prevented prompt admission requiring 379584512 bytes of transient workspace", with 23.7 GiB in use.

**Why:** Granite's model reported plain `ModelConfigMetadata`, so the paged cache (a) held KV for all 29 layers, the 23
Mamba ones included (2775 MB for 97,952 tokens, ~29 KB a token where the 6 attention layers need ~6 KB), and (b) knew
no prefix-prefill features, so a prompt reusing a cached prefix (the chat template's first block) planned the gather
path, whose workspace grows with query x context. The planner also reserved only 64 MB for activations (the GGUF
estimate counts one attention chunk's scores), leaving no room for it.

## Run 2 - 2026-11-02 ~03:50

**Change:** Granite (and the Transcribe, Granite Speech and Granite 4 Vision wrappers) report a
`HybridPagedKvCacheConfig`: only attention layers paged, their recurrent state bytes, and uniform prefix-prefill
features when no layer has sinks.

**Result:** the same 2669 MB now holds 455,456 tokens. 539-token prompt 4856 tok/s; 2436-token prompt 721 / 724
tok/s (admitted, but 6.7x slower per token than the short one and 6x slower than llama.cpp's pp2048); decode 55.2 /
55.1 tok/s (llama.cpp 198). The 64 MB activation estimate is unchanged.

**Correction (Run 3's profile):** the 539-token rate (4856 tok/s) was a prefix-cache hit, the script sending the same
text with a different last line. A fresh 959-token prompt runs at 642 tok/s: prefill was ~650-720 tok/s at every
length, 6-7x behind llama.cpp. Later runs open every prompt with a distinct line (`gpu_fresh.py`).

## Run 3 - 2026-11-02 ~04:30

**Question:** where does a prompt's GPU time go?

**Command:** `nsys launch --trace=cuda,nvtx --cuda-graph-trace=node inference serve ...`, then `nsys start` / `stop`
around one request each (a plain `nsys profile` stopped by SIGINT kept only the first 1.26 s of CUDA records), and
`nsys export --type sqlite`.

**959-token fresh prompt (1.52 s of kernels, 1.25 s busy):** `indexed_moe_forward_q5_0_q8_1` (the Q5_0 down
projection) 45.5 % at 8.2 ms a layer per chunk, IQ4_NL up as `gather_rows_u8` + `dequantize_superblocks` + BF16 gemm
per expert 21 %, elementwise (bmul, copy2d, fast_sum) 13 %, the SSM scan 3.8 %. The 2436-token prompt looks the
same over six prefill chunks.

**Why:** ungated experts on CUDA always took the gather path: the indexed matvec per (token, expert) pair, decode's
kernel, at every batch size. Gated experts have had grouped mmq for prefill.

## Run 4 - 2026-11-02 ~05:00

**Change:** ungated experts take grouped mmq for prefill (32+ tokens, no LoRA, not calibrating): routes sorted by
expert on the GPU, up through `grouped_moe_mmq`, the activation, down through `grouped_moe_mmq`, the weighted reduce.
The dispatch-table setup is shared with the gated path (`grouped_dispatch`). GPU case
`ungated_grouped_prefill_matches_gather` (Q4_0 / Q5_0 and Q4_K / Q8_0 against the gather path) passes.

**Result (`gpu_fresh.py`, no prefix reuse):** prompts of 592 / 530 / 1149 / 2321 / 2345 / 4398 tokens at 1298 / 1468
/ 1526 / 1721 / 1757 / 1788 tok/s (was ~650-720); decode 54.0 / 54.7 tok/s (unchanged). llama.cpp: 4342 / 198.

## Run 5 - 2026-11-02 ~05:40

**Question:** does a tiny Nemotron-H score on CUDA as its decode does (the CPU tests never ran it on the GPU)?

**Finding:** no, by up to 2 nats, with or without quantization, prefix caching or the grouped path. Bisecting the
tiny model's layer list: attention-only models agree with the CPU to 0.001, Mamba ones differ by up to 1.25 nats.
`ssm_scan_kernel` gives each warp lane `ceil(d_state / 32)` state items with no bounds check, so a `d_state` below 32
(the tiny model's 8) has lanes 8-31 read and write past the head's state row and add other heads' B and C into the
sum. The real model's 128 never hit it. With the guard: Mamba-only models match the CPU to 1.3e-6, the whole tiny
model to 2.5e-4.

Quantized (Q8_0) on CUDA, prefill and decode differ by 0.002-0.035 nats whether or not the experts run grouped (the
dense projections' prefill dequantizes to cuBLAS while decode quantizes the activations to Q8_1), so the CUDA case
`a_long_prompt_on_cuda_scores_as_its_decode` is unquantized at 1e-3; grouped mmq is pinned against the gather path by
the experts' GPU case.

**Real model (`audio_kld.py cmp`, 24 clips, 2801 positions, GPU pair against the BF16 CPU reference):** mean KLD
0.005451, same top-1 99.536 % (before Runs 2 and 4: 0.004948, 99.357 %). IQ4_NL up now quantizes its activations to
Q8_1 inside mmq, as llama.cpp does, where it dequantized the weights to BF16 for cuBLAS before.

## Run 6 - 2026-11-02 ~06:30

**Question:** what is left after grouped mmq (2436-token prompt, 1.61 s of kernels, 1.12 s busy)?

**Profile:** `mul_mat_q` 39 % (experts and dense), then 56,028 each of `bmul_bf16`, `fast_sum_small_bf16` and
`copy2d_bf16` (~406 per Mamba layer per chunk, one per token): on CUDA the Mamba conv ran per timestep (narrow the
window, multiply by the weight, sum), only the CPU having a conv kernel. The scan 10 %.

**Change:** GDN's fused conv kernels (prompt, width-4 tiled, decode step) take an optional bias before their SiLU, and
Granite's Mamba conv runs them on CUDA for prompts and decode steps. GPU case `causal_conv1d_bias_matches_the_cpu_conv`
(widths 3 and 4, prompt and step); tiny model `a_long_prompt_scores_on_cuda_as_on_the_cpu` (the same checkpoint's
logprobs on both devices, 1e-3).

**Result (`gpu_fresh.py`):** 592 / 530 / 1149 / 2321 / 2345 / 4398 tokens at 2092 / 3252 / 2623 / 3158 / 3326 / 3302
tok/s (Run 4: ~1300-1790); decode 55.6 / 55.8 (unchanged). llama.cpp pp512 4342, tg256 198.

## Run 7 - 2026-11-02 ~07:10

**Question:** where does a decode step go (64 tokens, 1.61 s of kernels, 0.98 s busy)?

**Profile:** the IQ4_NL up ran `gather_rows_u8` + `dequantize_superblocks` + a BF16 gemm once per routed expert per
layer (~158 a token, 31 % of busy time): the gather path, since IQ4_NL (raw storage) has no `indexed_moe_forward`
kernel. Q5_0 down took `indexed_moe_forward_q5_0_q8_1`. The lm head `mmvq_gguf_q8_0_f32_plain` 0.43 ms a token, then
~1.7 ms idle before the next step's embedding (`get_rows_q8_0`).

**Change:** `fast_mmvq::moe_decode_ungated`: the expert-indexed matvec gated decode uses, with the up projection
alone (no fused GLU), the activation as tensor ops on its F32 rows, Q8_1 quantization, then down; `moe_decode` and it
share one body. Ungated experts take it for up to 8 tokens. GPU case `ungated_indexed_decode_matches_gather`.

**Result (`gpu_fresh.py`):** decode 91.0 / 91.4 tok/s (was 55.6 / 55.8); prefill unchanged (2027-3317). llama.cpp
tg256 198.

## Run 8 - 2026-11-02 ~07:40

**Question:** the decode profile after Run 7: 1157 ms over 64 tokens under nsys, 625 ms busy, ~1750 kernels a token.
What are they?

**Profile:** `bmul_f32` 160 a token, `cast_bf16_f32` 208, `uexp_f32`, `badd_f32`...: the Mamba decode step ran as ~20
F32 tensor ops a layer (dA, dB, the state update, the dot with C, the D skip), the scan kernel only taking prompts.

**Change:** a decode step runs the scan kernel as a one-token chunk (B and C expanded to the heads as for prompts),
at decode's dt floor (`time_step_min`; prompts clamp at `prefill_time_step_min`).

**Result (`gpu_fresh.py`):** decode 114.5 / 113.5 tok/s (was 91); 4398-token prompt 3294 tok/s. Tiny-model CUDA cases
pass (decode against one prefill, CPU against CUDA).

## Run 9 - 2026-11-02 ~07:55

**Question:** what is left of a decode step at 114 tok/s (8.8 ms a token; llama.cpp ~5 ms)?

**Profile (64 tokens under nsys, so slower):** per token 12.1 ms wall, 6.6 ms of kernels, 1348 kernels. Of the kernel
time: dense Q8_0 matvecs (`mmvq_gguf_q8_0_bf16_plain`, ~96 a token at 21 us: Mamba in/out, shared experts,
attention) 2.0 ms, the experts' indexed matvecs (Q5_0 down, IQ4_NL up) 1.1 ms, the F32 lm head 0.43 ms, ~230 BF16-F32
casts 0.5 ms, then small elementwise ops and Q8_1 quantizations. Between steps ~1.6 ms idle from the lm head to the
next embedding (the logits' copy to the host, 0.6 ms a call including the wait, then sampling and scheduling).

**Implications:** the GPU idles ~45 % of a step under nsys (less without it): launch overhead over ~1350 kernels.
llama.cpp runs its decode as a CUDA graph. Granite does not opt into decode graphs (`supports_cuda_decode_graphs`),
and its Mamba cache replaces its conv and SSM state tensors every step where a captured graph needs fixed addresses,
so graphs need the recurrent state updated in place in a pool, as Qwen3.5's GDN layers do. Short of that: fewer
casts (the Mamba step's F32 round trips) and fusing the elementwise ops.
