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

## Run 10 - 2026-11-02 ~09:00

**Question:** do Granite's hybrid decode steps capture as CUDA graphs as they are?

**Finding:** yes. On CUDA the recurrent pool's gather is `index_select` and its scatter the `copy_rows` kernel, both
by the device slot indices the graph machinery swaps, so opting in (`supports_cuda_decode_graphs`) was the whole
change for the real model: 4 graphs (buckets 1-4) captured in 194 ms, decode 163.5 / 163.7 tok/s (was 114).
200-token greedy completions of three prompts are identical with graphs on and off (ids and logprobs to 3 decimals);
three concurrent requests diverge from single ones at near-ties as much with graphs off (batched BF16 rounding).
Four "CUDA error with no caller" logs after capture are evicted, never-replayed graph outputs whose free fails as
`drop_cuda_graph_entry_output` expects: the first prompt's memory check evicts the precaptured graphs (the KV cache
leaves the planner's 64 MB activation estimate free, Run 1), and they are captured again lazily.

**Opt-in limits:** a tiny checkpoint's capture failed on a host copy: unquantized ungated experts group their routes
on the host, as do Granite's own experts (`GraniteMoE`). Granite opts in only when every MoE layer's experts decode on
the device (`MoEExperts::decode_is_device_only`: quantized ungated stacks the indexed matvec reads, or any gated
backend).

**Two flakes on the way:**
- The fused MoE router, with fewer experts than a warp's 32 lanes, scored the padding lanes as sigmoid(-inf) = 0;
  a selection bias leaving every expert below 0 made a padding lane the pick (expert 4 of 4), about one tiny run in
  six. Fixed in the kernel, with a GPU case.
- Decode-against-prefill on the CPU (Q8_0) differed by up to 1.4e-3 about once in 30 full-group runs, never alone:
  the scoring request could reuse the prefix cache's state from the decode steps. The checks now run without a
  prefix cache: 0 failures in 40 runs of the group.

**Tests:** `a_long_prompt_scores_as_its_graph_decode` (BF16, Q8_0 experts, head dim 64 so FlashInfer takes the
attention) replays graphs and scores within 0.5 of one prefill; over 30 random checkpoints the gap is p90 0.054,
worst 0.29 with graphs and p90 0.056, worst 0.18 without, so it bounds stale-state replays, not rounding.

**Result (`gpu_fresh.py`):** prompts 2107-3307 tok/s, decode 163.7 tok/s (llama.cpp pp512 4342, tg256 198).

## Run 11 - 2026-11-02 ~10:00

**Question:** why are the precaptured decode graphs evicted (Run 10), and only for this model?

**Finding (`/metrics` after load, before any prompt):** `inference_cuda_memory_pressure_total{level="reserve"}` 5,
`inference_cuda_graph_evictions_total{reason="memory_pressure"}` 4, 350 MB available against a required 512 MB. The
default paged cache holds one sequence at the model's context, or what is free when that is less; for Nemotron-H's
context it is less, so the cache took everything but the cap's 512 MB reserve, and what loaded after it (graphs,
workspaces) left free memory under the engine's floor (total / 128 within 512 MB-2 GB, the same 512 MB on a 24 GB
card). Every idle check and prompt then reclaimed the graphs. Qwen2.5-Coder-3B's full context fits in 1152 MB, so its
graphs stay.

**Change:** a default-sized cache on CUDA reserves the engine's free floor as a post-load component
(`cuda_free_floor_bytes`, now shared by the planner and the engine's thresholds). A cache that fits beside it, or one
sized by the user, is unchanged.

**Result:** 512 MB reserved, the cache at 2157 MB (368,096 tokens, was 455,456), 880 MB available after the run;
4 captures, 510 replays, no pressure events or evictions over `gpu_fresh.py`. Decode 167.3 tok/s, prompts unchanged.

## Run 12 - 2026-11-02 ~11:00

**Question:** what does a decode step's Mamba layer launch now (167 tok/s)?

**Profile (one layer between two scans, from Run 9's trace):** 57 kernels a Mamba + MoE block. Before the scan:
`a_log` cast, exp and neg, `D` and `dt_bias` casts every step; B and C cast and copied out to every head; the state
gathered from the recurrent pool (`is_u32_f32`, 11.8 us) and scattered back after (`indexed_row_copy`), the conv
state too. After it, the grouped gated RMSNorm as ~11 elementwise ops.

**Change:** the scan kernel reads B and C by group and takes optional slot indices, updating those rows of a
recurrent pool in place (a negative slot is a padding row: no state write, output 0); its wrapper takes the CPU scan's
`SelectiveScan`. Granite keeps the scan's F32 constants from load. A CUDA decode step with an F32 state pool and conv
pool in the activation dtype runs the conv update and the scan on the pool's rows, without the gather and scatter.
GPU case `cuda_selective_scan_matches_the_cpu_scan_gathered_and_in_a_pool` (40-wide state, a padding row, untouched
rows checked).

**Result:** decode 200.8 / 200.8 tok/s (was 167; llama.cpp tg256 198); prompts 3463-3469 at 2.3-4.4K tokens. Greedy
completions of the three prompts identical to Run 10's eager reference (ids and logprobs to 3 decimals).

**Found on the way, not fixed:** `a_long_prompt_scores_as_its_graph_decode` fails about once in 20-60 runs, master
too (1 in 60). Scoring the sequence twice gives the same prefill both times, so the decode is off, and only at one
step (step 1, 2 or 4 of five, by 0.5-1.9 nats, the steps after it within 0.03), every step a graph replay (4 replays,
cache hits). A one-step error that does not carry into later steps is not the recurrent state or the KV cache; it
looks like the step's reported logprob, possibly read after the next replay (the engine runs one step ahead) has
overwritten the graph's logits. The tokens match, so greedy decoding is unaffected.

## Run 13 - 2026-11-02 ~12:00

**Question:** is Run 12's one-step decode error a graph-replay bug or test noise?

**Code reading (subagent, read-only):** with logprobs requested no lookahead step launches: `Sampler::cuda_sampling_plan`
returns `None` when `return_logprobs` is set (inference-nn/src/sampler.rs ~937), so `try_submit_cuda_token_batch`
declines and the step samples eagerly, the logits copied to the host (`to_vec1`, a blocking copy on the graph's
stream) and the token and logprob both taken from that copy. The graph's logits buffer is reused across replays, but
the next replay cannot launch before that copy returns. So it is not the logprob read racing the next step.

**Command:** the graph case with `INFERENCE_RS_CUDA_GRAPHS=0`, 100 runs (replay assertion skipped).

**Result:** 0 failures in 100 with graphs off, against about 1 in 30 over the loops with graphs on (master included):
at that rate 100 clean runs happen ~3 % of the time, so replays are implicated. Eager decode and replays run the same
kernels, which argues against routing near-ties. A step that is off by 0.5-1.9 nats while the steps after it are
within 0.03 fits a replay reading one stale piece of per-step attention metadata (a position, context length or slot)
better than stale recurrent indices, which would carry into every later step. Not yet tested; next is a debug mode
that runs each replayed step eagerly too and compares the logits, to catch the bad step and diff its inputs.

## Run 14 - 2026-11-02 ~13:00

**Question:** what does a failing replay read wrong?

**Probes (temporary):** after each replay, synchronize and read back the graph's inputs (token ids, positions, state
indices, and every metadata buffer the staging wrote: slot mappings, block tables, context lengths, FlashInfer
indptr / indices / tile plans, rope positions) against the values staged, and the step's top token and logprob from
the graph's own logits; the capture and replay threads. A failing run: every input as staged, capture and replays on
one thread, all kernels and copies on one stream, and the graph's own logits already give the wrong logprob (so not a
later read of them).

**Decisive check:** the test keeps a failing checkpoint; rerun on it, it fails 10 of 10 with identical values, and with
`INFERENCE_RS_CUDA_GRAPHS=0` it fails identically, bit for bit (step 5: decode -2.5467, prefill -3.6523). Run 13's "0 in
100 with graphs off" was an unlucky draw of checkpoints: the test made new random weights every run, and about one draw
in 30 has a step where decode (indexed matvec, Q8_1-quantized activations) and prefill (grouped mmq) land on opposite
sides of a near-tie, most likely in the experts' routing. Not a graph bug; Run 13's conclusion is withdrawn.

**Change:** the tiny Nemotron-H's weights come from a seeded `StdRng`, as `support/recording.rs` seeds its
checkpoints, so every run checks the same weights. On them the BF16 graph case differs from one prefill by 0.023, and
its tolerance goes from 0.5 to 0.1; the other cases: CUDA F32 1.4e-6, CPU quantized 2.4e-7, CPU 4.0e-5, CPU against
CUDA 4.4e-4. 0 failures in 10 runs of the Nemotron-H and Granite tiny group.

## Run 15 - 2026-11-02 06:05 (local clock; earlier runs' times were approximate)

**Question:** where does prefill trail llama.cpp now (2436-token fresh prompt, `fresh_long.py`, nsys)?

**Profile (ours, one pass):** 0.715 s busy: `mul_mat_q` 462 ms, `ssm_scan_kernel` 110 ms (138 calls, 794 us each:
one warp per (head, head_dim row) walking ~400 tokens in order), then elementwise (`cast_bf16_f32` 18 ms, `bmul_f32`
11, `usqr_f32` 10, `ucopy_bf16` 10, `bdiv_f32` 9, `urelu_f32` 7). **llama.cpp** (`llama-bench -fa 1 -p 2436 -n 0
-r 1 -ub 512` under nsys, halved for its warm-up pass): ~550 ms busy: `mul_mat_q` ~434 ms, and its scan is chunked SSD
(#22675: per 256-token chunk, cuBLAS for C.B, the intra-chunk output, the state's output and the state update, beside
three small kernels), ~38 ms.

**Change:** a prompt of 128+ tokens (llama.cpp's crossover; a window costs ~200 us fixed against the scan's ~2 us a
token) with no pool slots runs the scan as chunked SSD in F32: per sequence, windows of two 256-token chunks (scratch
~40 MB a chunk at 64 heads); one kernel prepares dt (bias, softplus, clamp, summed within the chunk), x dt and x dt
decayed to the chunk's end, and B and C by group; cuBLAS C.B for all chunks; a kernel masks and decays it; per chunk,
cuBLAS the chunk's own outputs (per head) and its state increment (per group, the group's heads side by side); one
kernel carries the state from chunk to chunk; cuBLAS the carried state's outputs (per group); a kernel adds the two
outputs, the decay to each position and D x. Unlike llama.cpp's, dt is summed within each chunk (no long-prompt
cancellation in the decays), every matmul is F32, and the chunk's matmuls run batched rather than chunk by chunk. GPU
case `cuda_chunked_ssd_scan_matches_the_cpu_scan` (700 tokens, batch 2, 3 heads a group, the dt clamp) within 1e-4 of
the output's scale.

**Result:** the scan's 110 ms is ~51 ms of SSD (SGEMMs 31, mask 7.6, prepare 6.3, finish 4.8, state 1.8; the prepare
kernel was 122 us a window until each position's decay was computed once and the group's heads shared the B/C copy).
Busy 0.665 s. `gpu_fresh.py`: 592 / 530 / 1149 / 2321 / 2345 / 4398 tokens at 2129 / 3442 / 2887 / 3517 / 3740 / 3705
tok/s (Run 12: 3463-3469 at 2.3-4.4K); decode 200.3 / 200.8. Short-prompt greedy completions identical to Run 10's.

**Long prompts:** 64-token greedy completions after 235 / 1301 / 2852-token prompts against the same build with SSD
off: first-token logprobs differ by 0.036-0.048, one completion matches in full, two diverge (steps 24 and 2). A
temporary probe ran both scans on every layer's real inputs: the largest difference in y is 1e-6 to 2e-5 of the
layer's largest output (A down to -11,076, x up to 62), F32 rounding of two summation orders. The gap is BF16
rounding of that noise, carried through 52 layers of a 4-bit model (Run 4 saw 0.002-0.035 between two expert matmul
paths).

**KLD** (the perplexity example's `--kld-base` against the BF16 base `bf16-wiki-c512-ch40.kld`, ik_llama.cpp's
`llama-perplexity` tokens, 40 chunks of 512, 10,200 positions): SSD on PPL 12.7827, KLD mean 0.07516, median 0.02778,
p99 0.689, same top-1 88.57 %; SSD off PPL 12.6964, KLD mean 0.07435, median 0.02699, p99 0.629, same top-1 89.02 %
(base PPL 12.1989). SSD costs ~0.0008 KLD (1 %). Both are 3x llama.cpp's KLD for the same GGUF (0.0250, top-1 93.2 %,
nemotron_transcribe_investigation.md): our CUDA path loses accuracy somewhere else, before this change.

**Next:** find the 3x KLD gap against llama.cpp (CPU against CUDA first, then per stage); the remaining ~115 ms: `mul_mat_q` (+28 ms against llama.cpp), F16 or TF32 tensor cores for the SSD
matmuls (~31 ms of SGEMM; llama.cpp runs them in F16), the BF16/F32 casts and the gated RMSNorm's elementwise chain.

## Run 16 - 2026-11-02 08:50 (local clock)

**Question:** after the dt fix (#772), where does a 2438-token fresh prompt trail llama.cpp, measured on the same
prompt? (Run 15's llama.cpp figures came from `llama-bench`'s random tokens, which route experts differently.)

**Commands:** scratch `nsys/prefill_profile.sh <tag>` (serve under nsys, two warm-up prompts, one profiled
`fresh_long.py` prompt, three timed); llama.cpp `970b51ce7` `llama-server -fa on -ub 512 --cache-ram 0` under nsys
with the same request; `mmq_breakdown.py` groups `mul_mat_q` by template instance and grid.

**Baseline (master `d08d42934`):** span 0.724 s, busy 0.689 s, 10,217 kernels, ~3,850 tok/s. llama-server: span
0.672 s, busy 0.614 s, 7,820 kernels. By kernel, against llama-server:

| | ours | llama.cpp |
|---|---|---|
| `mul_mat_q` | 485.5 ms | 461.9 ms |
| Q5_0 expert down, full ubatch / last | 1931 / 1881 us | 1767 / 1596 us |
| IQ4_NL expert up, full / last | 1608 / 1541 us | 1641 / 1496 us |
| Q8_0 dense | 104 / 300 us | 101 / 288 us |
| SSD scan (SGEMM + 4 kernels; llama.cpp `Kernel2` + 4) | ~57 ms | ~40 ms |
| elementwise | gated RMSNorm as ~11 F32 ops, relu^2 as `urelu` + `usqr` | `rms_norm_f32` 6.9, `unary_op_kernel` 7.9 |

The kernels match llama.cpp's (same templates, grids, 254 registers); only the Q5_0 down is slower (~22 ms over the
pass). llama.cpp's mmq now picks tiles from per-architecture tables (`mmq-config-ampere.cuh`), but its Ampere Q5_0 row
is the 128 / 8 warps we launch, so that is not the difference.

**Change 1, TF32 for the SSD matmuls** (`CUBLAS_COMPUTE_32F_FAST_TF32`, operands still F32): SGEMMs ~35 -> ~19 ms,
busy 0.689 -> 0.681 s. KLD 0.02507 (F32: 0.02672, both noise around llama.cpp's 0.02498). The SSD GPU case's
output error rises to 4.6e-4 of the output's scale (F32: under 1e-4); its tolerance is now 2e-3.

**Change 2, a fused gated RMSNorm:** Mamba's `x * silu(z)` normalized over each group (8 of 512 here) times the weight
ran as ~11 F32 tensor ops (two casts, silu, a multiply, square, mean, add, sqrt, divide, a cast back, the weight
multiply), each a pass over 2438 x 4096 values. `mamba_gated_rmsnorm_kernel` (ssm.cu) does it in one pass, one block
a (row, group), reading the scan's F32 output and the bf16 gate in place from the input projection's slice, writing
the gate's dtype. Unlike GDN's gated norm it gates before normalizing. GPU case
`cuda_mamba_gated_rmsnorm_matches_the_composite` (gate a slice of a wider bf16 tensor, groups of 96).

**Result:** span 0.665 s (llama-server 0.672), busy 0.636 s, 8,699 kernels; the norm 3.2 ms for 138 calls.
`fresh_long.py` ~4,330 tok/s (was ~3,850). `gpu_fresh.py`: 592 / 530 / 1149 / 2321 / 2345 / 4398 tokens at 2349 /
3759 / 3024 / 3774 / 3928 / 3909 tok/s; decode 216.0 / 222.6 tok/s (Run 15: 200.3 / 200.8; the norm runs in decode
too). KLD 0.02490, median 0.00966, p99 0.238, same top-1 93.34 % (llama.cpp 0.02498 / 0.00936 / 93.22 %). Greedy
200-token completions with the fused norm against the same build without it: one identical, two diverge at tokens 66
and 94, logprobs within 0.16 before that (rounding: rsqrt and the summation order).

**Next:** the Q5_0 expert down (~22 ms against llama.cpp, same kernel); relu^2 as one pass (~7 ms); `cast_bf16_f32`
(12 ms, 691 calls) and `ucopy_bf16` (10 ms).

## Run 17 - 2026-11-02 10:19 (local clock)

**Question:** why is the Q5_0 expert down slower than llama.cpp's with the same template, grid and registers (Run 16:
1931 against 1767 us)?

**Commands:** `ncu --set detailed` on two full-ubatch launches of `mul_mat_q<Q5_0, 128, false>` in our server and in
llama-server (`970b51ce7`), same request; scratch `ncu/ophist.py` compares executed SASS by opcode.

**Finding:** at ncu's base clocks 1.84 / 1.86 ms against 1.68 / 1.72 ms, the same DRAM traffic (~615 MB read), but
475M instructions against 449M: +16.7M `LOP3`, +10.5M `PRMT`. Our Q5_0 tile loader subtracts the 16 offset with
`__vsubss4` (saturating), which Ampere has no instruction for; llama.cpp's #29478 (2026-09-29) replaced every
`__vsubss4` in its CUDA code with `__vsub4`, since no operand can saturate (4-6 bit values minus their offset: Q4_0,
Q5_0, Q3_K, Q6_K and the K-quant scales in mmq and mmvq, the Q4_0 / Q5_0 KV dequantizers in fattn).

**Change:** the same in our copies: `mmq_gguf.cuh`, `mmq_vecdotq.cuh`, `mmvq_gguf.cuh`, `indexed_moe.cu`, and fattn's
`vecdotq.cuh` and `fattn-common.cuh` (20 sites).

**Result:** the Q5_0 down at base clocks 1.73 / 1.75 ms, 449M / 456M instructions (llama.cpp 449M / 453M); what is
left is branch syncs (`BSSY`/`BSYNC` +1.3M each) and the write-back. Greedy completions bit-identical to Run 16's
build. End to end the gain is within the card's power-cap noise: nsys `mul_mat_q` 482.4 ms (Run 16: 482.6; Q5_0 down
1863 against 1922 us, IQ4_NL up, unchanged code, 1651 against 1586), SM clock 1,695-1,980 MHz with the power-cap
reason set, `fresh_long.py` 4,324-4,371 tok/s.

## Run 18 - 2026-11-04 15:25 (local clock)

**Question:** Nemotron 3.5 Lightning on the 3090, CUDA: what can the card do, as a target for the Vulkan backend
(Vulkan Runs 30-35: 512 tokens ~2200 tok/s, 2048 ~2900, decode 111, a 244K prompt 715 s with a q8 cache)?

**Build fix first:** master's CUDA build failed. `device_location_sort_key` (`pipeline/cuda_graph.rs`) had no arm for
`DeviceLocation::Vulkan`, which #795 added; `local_ci.sh --vulkan` never builds `cuda`. Added the arm. With CUDA
building again, `nemotron_h_tiny::a_long_prompt_scores_on_cuda_as_on_the_cpu` (0.0026 off) and
`a_long_prompt_on_cuda_scores_as_its_decode` (0.0011) failed their 1e-3 tolerance. Bisected to #795, which raised the
tiny checkpoint's Mamba state from 8 to 128 for the Vulkan scan; the chunked SSD's cuBLAS matmuls run
`CUBLAS_COMPUTE_32F_FAST_TF32`. With `CUBLAS_COMPUTE_32F` all long-prompt tests pass at 1e-3, so the CUDA tolerance
is 5e-3 now (TF32 stays: it is the production path).

**Commands:** `llama-bench -m Lightning-exps-q4_0.gguf -p 512,2048 -n 128 -ngl 99 -fa 1 -r 3` (`970b51ce7`); ours
`inference bench --format gguf -f Lightning-exps-q4_0.gguf --prompt-len P --gen-len G` (dev build, debug assertions
and overflow checks off), with `--paged-attn on --pa-context-len 262144 --max-seq-len 262144 [--pa-cache-type q8]
--depth D` for the long runs. Note `--depth` sets the decode depth only; TTFT is always a fresh prompt.

**Short (Q4_0 experts, 18.2 GiB):**

| | llama.cpp CUDA | ours CUDA | ours Vulkan (Run 30) |
|---|---|---|---|
| 512-token prompt | 4142 +- 280 tok/s | 4285 +- 51 | ~2200 |
| 2048-token prompt | 4327 +- 86 | 4247 +- 29 | ~2900 |
| decode | 197.3 | 208.6 | 111 |

**Decode at depth (q8 cache):** 180.7 tok/s at 32K, 142.0 at 128K.

**Long prompts, one fresh prompt each:**

| prompt | q8 cache | bf16 cache (1536 MB at 262K) | Vulkan, q8 (Run 35) |
|---|---|---|---|
| 131072 | 84.6 s (1549 tok/s) | | ~150 s at 105K |
| 245760 | 246.1 s (999 tok/s) | 115.0 s (2137 tok/s) | 715.3 s at 244K |

**Finding:** the q8 runs fit `T = aN + bN^2/2` exactly (a = 1/4250 s, b = 6.26e-9 s). At 245K that's 58 s of
per-token work and 188 s of attention. Causal attention over the 6 attention layers (32 heads, head 128) is
49152 N^2 FLOPs, 2.97 PFLOP at 245K: 15.8 TFLOPS through the q8 cache, ~52 TFLOPS through bf16 (115 - 58 s).
The quantized-cache prefill attention costs 2.1x the whole prompt on CUDA; bf16's is near what the card's f16 tensor
cores sustain. Vulkan's 715 s leaves ~630 s of attention after ~85 s of per-token work: ~4.7 TFLOPS.

**Next:** two separate levers. On CUDA, the q8 cache's prefill attention (dequantize into the flash tiles rather than
whatever it does now) is worth ~130 s at 245K. On Vulkan, prompt attention over the paged cache runs at ~1/10 of
bf16 CUDA's; at 245K the card's demonstrated floor is ~115 s.

## Run 19 - 2026-11-04 17:30 (local clock)

**Question:** Run 18's q8-cache prompt attention ran at ~16 TFLOPS against bf16's ~52. Why, and how much comes back?

**Commands:** nsys over a 65536-token prompt (q8 and bf16 caches); a new bench, `bench::chunk_over_prefix` (an
8192-row chunk over a 8K / 32K / 128K paged prefix, 32/2 heads, head 128, causal); `ncu --set detailed` and
`--section SourceCounters` on one launch at 32K; e2e `inference bench --prompt-len 245760` as in Run 18.

**Findings, in order:**
- nsys: everything but `flash_attn_ext_f16` matched; it took 13973 ms with q8 against 4157 with bf16 (3.4x), the same
  tile (`<128, 128, 8, 8>`, 128 threads, 255 registers).
- q8's staging (raw rows copied by `cp.async`, then dequantized from smem) put the block at 52.4 KB of smem
  against bf16's 33.9: one block an SM where bf16 fits two. q4 (80-byte rows, 44 KB) fit two, and ran 1.5x bf16
  where q8 ran 3.1x. The dequantized K tile (Q.K) and V tile (P.V) live between barriers, never at once, so they now
  share one buffer: q8 at 36 KB, two blocks an SM. Bench, 128K prefix: 842 -> 563 ms.
- Byte-permute dequant of 8-bit codes (3 ops two values, bit-identical): 563 -> 544 ms. Not the limiter.
- ncu at 32K: same occupancy and IPC (1.40 / 1.48), but 26.2G warp instructions against 12.5G. Source counters: 49%
  in `fattn_stage_quant_rows`, 27 instructions a `cp.async`. A q8 row is 136 bytes, not whole 16-byte units, so it
  copied 34 4-byte units a row, each recomputing its row (a divide by 34, the block table). Each row now goes to a
  power-of-two group of threads that looks it up once, and copies 8 bytes a unit (every head a multiple of 32 has
  8-byte rows): 544 -> 422 ms; q4 420 -> 340.
- The tile's staging and dequant are a fixed cost a block, shared by its columns. A 128-column tile (16 positions x
  8 heads, 256 threads, one block an SM; one new instance, head 128 only, quantized caches with >= 8-head groups):
  q8 422 -> 376 ms (45 TFLOPS); q4 unchanged (342).

**Bench (`chunk_over_prefix`, us; bf16 275-298K at 128K):**

| step | q8 8K | q8 32K | q8 128K | q4 128K |
|---|---|---|---|---|
| before | 28201 | 190197 | 842398 | 407481 |
| shared K/V buffer | 18706 | 125496 | 562810 | 424507 |
| byte-permute dequant | 17801 | 121972 | 544425 | 420502 |
| row-wise staging | 14087 | 95333 | 422195 | 339579 |
| 128 columns | 12669 | 84644 | 375678 | 342124 |

**Result (e2e, one fresh 245760-token prompt, Q4_0 experts):** q8 cache 246.1 -> 141.4 s, q4 157 -> 137.1 s, bf16
115.0 / 116.3 s. Decode at depth with q8 unchanged (181.4 tok/s at 32K, 140.8 at 128K).

**Tests:** `prefix_prefill_over_a_quantized_cache_matches_attention_over_its_values` gains 700/500-row sequences
(many tiles through the staged kernels) and GQA-16 cases (32/2 heads) that take the 128-column tile.

**Next:** q8 prompts sit 1.22x bf16 end to end. Item (1) of Run 18: Vulkan's prompt attention over the paged cache.

## Run 20 - 2026-11-05 12:20 (local clock)

**Question:** does Lightning's built-in MTP head speed up CUDA decode, as it did on the CPU (Lightning CPU Run 6: MTP 2
+19 %)?

**Commands:** dev build with `cuda vulkan` at f77f76fcf. `inference serve --max-seqs 1 --format gguf -f
Lightning-exps-q4_0.gguf` plain, with `--paged-attn on --pa-context-len 8192`, and with `--mtp --mtp-n-predict
{1,2,3}` added. Four chat prompts (code, a list, prose, counting), greedy, thinking off, 256 tokens; a warm-up pass,
then a measured one, reading `usage.avg_compl_tok_per_sec` (a harness in the session scratchpad).

**Result (decode tok/s per prompt, mean):**

| config | prompts | mean | MTP accept (len) |
|---|---|---|---|
| plain | 243.6 / 245.4 / 247.1 / 247.6 | 245.9 | |
| paged | 248.8 / 250.2 / 248.7 / 249.8 | 249.4 | |
| MTP 1 | 225.7 / 199.8 / 211.0 / 244.7 | 220.3 | 87.1 % (1.87) |
| MTP 2 | 226.7 / 204.3 / 192.2 / 271.5 | 223.7 | 78.2 % (2.56) |
| MTP 3 | 244.3 / 198.9 / 165.6 / 306.2 | 228.7 | 67.3 % (3.02) |

MTP loses on CUDA despite the CPU's acceptance. MTP 1's step (1.87 tokens at 220 tok/s) is ~8.5 ms against plain's
4.0: 2.1x a decode step.

**Finding:** with MTP the startup capture fails: "CUDA decode graph precapture failed, graphs will be captured
lazily: CUDA graph captured forward failed". With debug logging, cudarc reports `CUDA_ERROR_INVALID_VALUE` with no
caller ("kept for this context's next call"), and a step then fails ("step - Model failed with error:
DriverError(CUDA_ERROR_INVALID_VALUE)"). Plain and paged runs capture their decode graph at startup (~100 ms).

**Next:** the failing call under capture. User note: DFlash and DSpark drafter heads for Lightning are on Hugging
Face (NVFP4).

## Run 21 - 2026-11-05 12:47 (local clock)

**Question:** Run 20's failing capture, and whether MTP's output on CUDA is right.

**Correctness first:** MTP's text departed from plain decoding early in 3 of 4 prompts at every depth (characters
38-111). A new test, `an_mtp_head_drafts_on_cuda_without_changing_greedy_output` (the graph checkpoint: BF16, Q8_0
experts, paged), failed the same way. An F32 checkpoint failed too, as did one with graphs off, so neither graphs nor
TF32 cause it; full-f32 cuBLAS for the SSD and the GEMMs changed nothing. One draft or two, the same token
departed (the third).

**Localized:** a per-layer checksum of row 0 in each forward, plain run against MTP run, step by step. Step 2 matched
to 2e-7; at step 3 layer 0's input (the embedding) matched and its output (Mamba) was 10 % off. So the recurrent state
was right after the first commit and wrong after the second. Printing layer 0's gathered pre-state around the verify
pass: SSM norm 55.84 before, 41.67 after, read through the stash's own tensor. `selective_scan_cuda` advances the
state it is handed in place, and the stash shared that storage (`to_dtype` to the same dtype is a view), so every
commit's replay started from the already-advanced state. The CPU's scan writes a new tensor, and Vulkan's branch
copies first.

**Fix:** the verify pass stashes copies of the gathered conv and SSM states, taken before the pass (one state copy a
Mamba layer a verify step, ~48 MB for Lightning, ~50 us). A copy taken when the stash was pushed, after the pass,
had not helped.

**Result (Run 20's prompts, after the fix):**

| config | mean tok/s | MTP accept (len) | text against plain, per prompt |
|---|---|---|---|
| paged | 246.2 | | |
| MTP 1 | 213.5 | 88.4 % (1.88) | departs at char 248 / 222 / 1023 / same |
| MTP 2 | 229.4 | 82.4 % (2.65) | departs at 38 / 55 / 944 / same |
| MTP 3 | 233.9 | 72.1 % (3.16) | departs at 38 / 55 / 944 / same |

MTP 1's departures moved from 60-104 to 222-1023. MTP 2 and 3 still part at 38 and 55; whether those are near-ties a
3-4 row forward rounds the other way (the CPU's Run 6 case) is not yet checked.

**The graph capture (open):** a backtrace hook in cudarc's `record_err` (removed) showed the "no caller" errors are
frees of the wrong memory around a capture. Freeing an eager allocation inside a capture is invalid, and so is freeing
a graph allocation whose graph never ran.
- The MTP head's `record` replaced the warm-up's capture inside the captured forward. For hybrid models
  (`nonmutating_transition_capture`) the decoder doesn't reinstall the warm state afterwards, so the head is left as
  the only owner of the graph's tensors. ERNIE's and Qwen3.5's MTP graph tests take the other path and pass.
- Granite's `mamba_replay` stash has the same two problems, and a third: after a replay the commit needs exactly the
  graph's stash buffers. It has to become part of the speculative graph state handed to the entry, as the MTP
  capture is; Qwen3.5's hybrid MTP rolls back by checkpoints and has no stash.
- A separate head slot for captures made during a capture fixed the head's half but not the stash's (the tiny test
  still failed); reverted.

Until then MTP runs eagerly on CUDA, and loses to graph-replayed plain decoding (4.0 ms a step against MTP 1's ~8.5).

**Next:** the Mamba stash as graph state, or checkpoint-based rollback for Nemotron-H's verify graphs. The DFlash and
DSpark drafters for Lightning: the codebase already runs both kinds of drafter for Qwen3.5 (`tiny_dflash`,
`tiny_dspark`).

## Run 22 — 2026-11-05

**Question:** with Granite's verify steps skipping graphs anyway ("SpeculativeConflict", no speculative graph-state
hooks), does turning decode-graph capture off while the MTP head drafts remove the capture errors, and what does MTP
cost against eager and graphed plain decoding?

**Change:** `supports_cuda_decode_graphs` returns false while the MTP head has drafts (`n_predict() > 0`), so no decode
graph is captured that would record the head's state.

**Command:** `mtp/run.sh LABEL 256 --paged-attn on --pa-context-len 8192 [--mtp N]` (scratch harness, 4 generic
prompts, greedy, thinking off); `INFERENCE_RS_CUDA_GRAPHS=0` for the eager row.

| config | mean tok/s |
|---|---|
| paged, graphs | 244.8 |
| paged, eager | 169.6 (170.0 / 168.8 / 171.8 / 167.9) |
| MTP 2, eager (graphs off by the change) | 229.6 (accept 82.4 %, len 2.65) |

No `CUDA_ERROR_INVALID_VALUE` and no graph warnings in the log; the 15 nemotron_h_tiny tests pass.

**Implication:** MTP 2 already beats eager plain decoding by 35 %, so verify steps replayed as graphs should take it
past graphed plain decoding. Next: Granite's `take_speculative_graph_state` / `install_speculative_graph_state`
carrying the MTP capture and the Mamba replay stash.

## Run 23 — 2026-11-05

**Question:** with the Mamba replay stash and the MTP head's capture handed to the graph entry as speculative graph
state, do verify steps replay as graphs, does the text stay what eager MTP produced, and does MTP now beat graphed
plain decoding?

**Change:** Granite's `take_speculative_graph_state` / `install_speculative_graph_state` carry a
`GraniteSpecGraphState` (the head's `MtpGraphState` and each Mamba layer's pre-pass conv/SSM state, conv input and
dt). The device slot indices stay the graph's buffer, refreshed by each replay; installed replays drop their host
slots (the CUDA scatter reads device slots only). Run 22's opt-out of decode graphs is reverted: holding the warm
state across the capture is what keeps the head's eager capture from being freed inside it.

- First try refreshed the host slots from the cache's live table and bailed when it was short. Precapture has no live
  sequence, so it failed ("Mamba replay has 1 rows, but the live slot table has fewer"). The failed capture's
  never-run graph memory then freed with the `CUDA_ERROR_INVALID_VALUE` "no caller" errors.
- After the fix: "Captured 2 CUDA decode graphs" (widths 1 and 3), and no errors or lazy-capture warnings in any log.

**Command:** `mtp/run.sh LABEL 256 --paged-attn on --pa-context-len 8192 [--mtp --mtp-n-predict N]`, one session.

| config | per prompt tok/s | mean | MTP accept (len) |
|---|---|---|---|
| paged | 247.8 / 247.6 / 240.1 / 241.5 | 244.3 | |
| MTP 1 | 253.7 / 223.0 / 224.8 / 264.5 | 241.5 | 88.6 % (1.89) |
| MTP 2 | 266.1 / 228.4 / 228.2 / 301.5 | 256.1 | 80.5 % (2.61) |
| MTP 3 | 281.3 / 225.6 / 196.4 / 348.3 | 262.9 | 70.6 % (3.12) |
| MTP 4 | 276.5 / 189.9 / 178.0 / 361.6 | 251.5 | 65.7 % (3.63) |

MTP 2's text against plain decoding departs at chars 38 / 55 / 944 / same, exactly as Run 21's eager MTP 2, so the
graphed verify replays what the eager one computes. The tiny CUDA MTP test now runs with graphs and asserts that only
prompts skipped them (it would see `speculative_conflict` skips without the hooks); 15/15 nemotron_h_tiny pass.

**Implication:** MTP 3 is +7.6 % over plain decoding. A step at MTP 3 costs ~11.9 ms (3.12 tokens at 262.9 tok/s)
against plain decoding's 4.1 ms a token, so the verify, the drafts and the commit's replay together cost ~2.9 plain
steps. Next: where that step time goes (nsys), since a 4-row MoE verify reads up to 4 tokens' distinct experts.

## Run 24 — 2026-11-05

**Question:** where does an MTP 3 step's time go now that verify steps replay as graphs?

**Command:** `mtp/prof.sh p3 --paged-attn on --pa-context-len 8192 --mtp --mtp-n-predict 3` (nsys
`--cuda-graph-trace=node` over one 194-token prose request after a warm-up, sqlite export; kernels grouped by
correlation id per graph launch), and `p0` without MTP for the 1-row decode graph.

Per step (75 steps; nsys slows the run, 171 against 195 tok/s for this prompt):

| part | GPU time |
|---|---|
| verify graph (4 rows, 1026 kernels) | 9.5 ms |
| drafter graph (48 kernels) x 3 | 0.24 ms each |
| draft lm_head, Q8_0 `mmvq_gguf_q8_0_bf16_plain_cuda1`, eager x 3 | 0.44 ms each, 1.3 ms |
| commit's Mamba replay and other eager kernels | ~1.4 ms |
| GPU idle | ~3.2 ms |

Verify graph against the 1-row decode graph (4.8 ms under nsys), per launch:

| kernel | verify | decode |
|---|---|---|
| `mmvq_gguf_q4_0_moe_cuda` x 46 | 3793 us | 1023 us |
| Q8_0 dense matvecs x 52 | 1472 | 1400 |
| Q5_0 dense matvecs x 46 | 648 | 459 |
| casts x 116 | 524 | 241 |
| Mamba gather / copy / scatter / `gemmSN_TN` | ~1010 | none (in place) |
| `ssm_scan_kernel` x 23 | 237 | 134 |

The dense matvecs stay flat at 4 rows (the DFlash 2 lesson, `dflash2_drafter_investigation.md` Runs 4-5, already
holds for Q8_0 here). The experts do not: the MoE grid is (928, 24), one block row per (token, expert) pair, each
pair reading its expert's rows by itself. Verify attention already runs per sequence (`flash_attn_ext_f16` with
`ncols1 = 4`, DFlash 2 Run 12's fix), so it is not a verify cost at 8K.

**Overlap (temporary probe, removed):** distinct experts per 4-token MoE call, eager MTP 3 run, 14298 calls: mean
16.45 of 24 pairs (10-19). Reading each distinct expert once would cut the expert traffic to ~69 %.

**Draft head (`--mtp-draft-isq`):**

| config | mean tok/s | accept (len) |
|---|---|---|
| MTP 3, Q8_0 draft head | 262.9 | 70.6 % (3.12) |
| MTP 3, q4_0 draft head | 275.7 | 67.6 % (3.03) |
| MTP 3, q4k draft head | 271.8 | 66.5 % (3.00) |

**Implication:** two levers, the cheap one already there (a 4-bit draft head). Next: an expert matvec that reads an
expert once for the tokens routed to it.

## Run 25 — 2026-11-05

**Question:** can the MoE matvec read a shared expert once for its tokens?

**Try 1, grouped blocks (reverted):** a pair leading its expert's group (up to 8 pairs) dotted each weight block
against every member's activations, the rest returning. Correct for every type (the indexed MoE test with shared,
repeated and over-full groups), but 167 registers and 128 bytes of stack for the Q4_0 GLU kernel against 56, so
~3 blocks per SM instead of ~8 in a latency-bound kernel:

| config | mean tok/s |
|---|---|
| plain | 246.3 |
| MTP 3 | 153.6 |
| MTP 3, q4_0 head | 157.3 |

**Try 2, pairs fast (kept):** the same one-pair blocks, but with several tokens the grid is (pairs, row blocks), so
every pair's block over one row block runs together and the pairs sharing an expert read its rows from L2 after the
first. No register change; one-token decode keeps the old order (its pairs are distinct experts). The launchers take
a `grouped` flag, which `moe_decode_with` sets for more than one token.

| config | mean tok/s | accept (len) |
|---|---|---|
| MTP 3 | 272.8 (was 262.9) | 70.2 % (3.11) |
| MTP 2, q4_0 head | 280.2 | 82.3 % (2.65) |
| MTP 3, q4_0 head | 295.0 (was 275.7) | 72.7 % (3.18) |
| MTP 4, q4_0 head | 288.3 | 65.2 % (3.61) |

Profile (`p3b`, MTP 3, q4_0 head): the verify's expert matvec 3793 -> 3099 us per launch (ideal from Run 24's overlap
~2600), the verify graph 9.5 -> 8.8 ms.

**Implication:** MTP 3 with a q4_0 draft head is 295 tok/s against plain decoding's 246 (+20 %). Left, by size: the
GPU idle time between graphs (the eager commit and host work), the verify's Mamba gather / copy / scatter (~1 ms,
in place as the decode step would save the copy and scatter), the 116 casts, and the last ~0.5 ms of expert reuse.
