# A Vulkan backend for UMA iGPUs

Target: the Radeon 780M (RDNA3, gfx1103, Mesa RADV) beside a Ryzen 7 7840U, sharing dual-channel DDR5-5600. Goal:
decode and prefill on the iGPU for models the CPU path runs today, starting with Nemotron 3.5 Lightning
(`nemotron_lightning_cpu_investigation.md`, Runs 13-14).

## Why

- The CPU's reads cap at 58-59 GB/s (its fabric link); the iGPU reads 73-79 GB/s from a model-sized buffer.
- llama.cpp's Vulkan backend runs Lightning recipe A at 22.2 tok/s decode and 411 tok/s prefill, against our CPU
  path's 17.8 plain / 20.4 with MTP and ~106 prefill.

## Decisions

- **Its own crate, `inference-vulkan`**, on `ash` (the loader opened at runtime, so a build needs no Vulkan SDK to
  link), with no dependency on `inference-tensor`, so the tensor crate can take it as its backend later. Everything
  is behind its `vulkan` feature; without it the crate is empty, so `--workspace` builds need neither `glslc` nor a
  driver.
- **Shaders compile ahead of time**: `build.rs` runs `glslc` over `shaders/*.comp` and embeds the SPIR-V.
- **Zero-copy weights**: the iGPU reads host memory at near carve-out speed, so weights the CPU path already holds are
  imported (`VK_EXT_external_memory_host`) rather than copied: 73 GB/s imported against 79 GB/s for GTT allocations,
  for no second copy of the model. GTT must be raised from the 4 GiB default (`amdgpu.gttsize`, `ttm.pages_limit`).
- **Bindings stay under 4 GiB** (a storage binding past it reads zeros), and one submission stays well under the
  GPU's job timeout (~2 s of work loses the device).

## Phases

1. Device, memory (import, GTT allocations), pipelines from embedded SPIR-V, and one kernel: a Q8_0 matrix-vector
   product, checked against a CPU reference and measured at full bandwidth on an `lm_head`-sized weight.
2. The rest of a decode step's kernels (Q5_0 / IQ4_NL matvec, routed experts, RMSNorm and elementwise ops, the
   Mamba conv and scan, attention over the KV cache, the router), each against the CPU path.
3. Whole layers recorded into one command buffer per decode step, so a token costs one submit; the model's forward
   on the iGPU, then MTP and prefill.

## Runs

### Run 1 - 2026-11-03 ~16:00

**Change:** `inference-vulkan`: a context (the integrated GPU first; push descriptors, host-memory import, 16-bit
storage), host-visible buffers (cached), zero-copy imports rounded out to the import alignment, pipelines from the
embedded SPIR-V, and `mul_mat_vec_q8_0` (two rows a 64-thread workgroup, the threads striding over a row's blocks,
a shared-memory reduction; f32 activations, u16 loads of the blocks).

**Tests:** `a_q8_0_matvec_matches_the_cpu` (333 x 2688, a weight starting mid-page, offset activations and
outputs, against an f64 dequantized reference); `scripts/local_ci.sh --vulkan`.

**Result:** an `lm_head`-sized weight (131072 x 2688 Q8_0, 374 MB) imported from host memory, twenty matvecs a
submit: 76.1 GB/s, 4.9 ms each (the raw-read test's imported ceiling was 73-79 GB/s).

### Run 2 - 2026-11-03 ~20:50

**Change:** `inference-vulkan` records into a stream: dispatches append to an open command buffer, submitted without
waiting every 64 dispatches, each followed by a barrier, with the resources a batch uses kept alive until its fence
signals. `Device::Vulkan` / `Storage::Vulkan` / `QStorage::Vulkan` in `inference-tensor` (a stub without the
feature): storage in host-visible buffers; every op, custom op and in-place op runs the CPU backend's kernel on a
copy and uploads the result; quantized weights keep CPU blocks. Model code that branches by device sends Vulkan
down the CPU's branch (Mamba's scan / conv / pool step, the MoE experts from their blocks, the mapped-CPU expert
backend), and CPU kernels' outputs follow their input's device. `--vulkan` / `vulkan:0` select it; F32 activations.

**Measured:** a submit-and-wait costs ~57 us on the 780M and a batched dispatch ~10-15 us, so the design submits only
when the CPU reads.

**Tests:** the backend's ops against the CPU's (`vulkan_backend::tests`); `nemotron_h_tiny::
a_long_prompt_scores_on_vulkan_as_on_the_cpu` (plain and Q8_0 experts, with the CPU's KV cache in f32: its f16 default
was the only difference, which the quantized checkpoint amplified to 0.015).

**Result (Lightning recipe A, 64-token prompt, 32 decoded):** prefill 23.3 tok/s, decode 3.0 tok/s (332 ms a token):
~0.5 ms an op of fallback (a synchronize, two copies, a fresh Vulkan allocation), the baseline the kernels replace.

### Run 3 - 2026-11-03 ~21:10

**Change:** host buffers pool by capacity (powers of two from 256 B to 256 MiB, up to 2 GiB idle): a dropped buffer
returns to its class, which the stream's retention makes safe, and a new one reuses it rather than allocating.

**Result (as Run 2):** prefill 23.3 -> 61.2 tok/s, decode 3.0 -> 8.1 tok/s (124 ms a token).

### Run 4 - 2026-11-03 ~21:40

**Change:** `map_f32` (unary, binary with broadcasting, affine, powf, elu, fill over strided f32 operands of up to 6
dims, by op code) and `copy_b32` (copies and fills of any 4-byte element bit for bit); `VulkanStorage` runs its unary,
binary, affine, powf, elu, strided and 2-D copies, same-dtype `to_dtype`, `try_clone` and `const_set` on them.

**Count first:** a decode step had ~285 fallback calls a token: binary ops, strided copies (`copy2d`), unary ops,
gathers, affine, f32 matmuls (router, attention), reductions, then the custom ops and in-place pool updates.

**Result (as Run 2):** prefill 61.2 -> 80.9 tok/s; decode 8.1 -> 6.6 tok/s. A native op leaves work in flight that
the next fallback op waits for (~60 us a submit-and-wait), so decode pays for every boundary until a layer's ops are
all on the GPU.

### Run 5 - 2026-11-03 ~23:00

**Change:** `mul_mat_vec_q5_0` beside Q8_0's; `CustomOp1::vulkan_fwd` (an op's Vulkan kernel, or `None` for the CPU's
on a copy); `QStorage::Vulkan` weights run decode and verify rows (up to 8) on the GPU from a driver-allocated copy of
their blocks, made on first use.

**Finding: imports cost every submit.** Weights first ran from zero-copy imports: decode fell to 3.4 tok/s, with a
submit of one small dispatch taking 1.3 ms. A tiny dispatch's submit with imports alive: 60 us with none, 181 us with
150 MB, 1547 us with 1.8 GB (~0.8 us a MB: RADV validates userptr pages each submit). With driver allocations alive:
36 us at 1.8 GB and 35 us at 6 GB. Imports suit one buffer streamed in long submits, not a model's weights; the GPU
clock (forced `high`) and the CPU's spinning workers were ruled out first.

**Kernels at Nemotron's shapes (synced / batched):** Q8_0 8448x2688 410 / 343 us (70 GB/s), Q5_0 3712x2688 157 / 100
us (69 GB/s), Q8_0 2688x4096 222 / 156 us (75 GB/s).

**Result (as Run 2):** decode 6.6 -> 7.1 tok/s (a weight copy each: +1.9 GB for recipe A's dense layers). Prefill
fell to 45.6 tok/s with the Vulkan quantized storage alone (the CPU path's tiles are a likely cause, to look at with
GPU prefill).

### Run 6 - 2026-11-04 ~00:00

**Change:** `rows_f32` (a workgroup a row: RMS norm, softmax, sum / max / min / argmax), `index_b32` (index_select and
gather over 32-bit words), `matmul_f32` (batched, every operand strided) and a sigmoid map op;
`CustomOp2::vulkan_fwd`, and the RMS norm, softmax and sigmoid ops' Vulkan kernels.

**Result (as Run 2):** prefill 45.6 -> 82.5 tok/s, decode 7.1 -> 6.8 tok/s: what still syncs a decode step are the
model's CPU branches (the experts, the router's sort, Mamba's conv and scan), which read their inputs back each layer.

### Run 7 - 2026-11-04 ~00:40

**Change:** `ssm_conv_step` and `ssm_scan_step` kernels update Mamba's conv and state pools in place at the batch's
slot rows, which stay on the GPU (the CPU kernels read them back to the host); granite takes them for a decode step on
Vulkan. `nemotron_h_tiny::a_vulkan_decode_scores_as_one_prefill` checks a Vulkan decode against the model's own
prefill, plain (1e-3) and with Q8_0 weights (0.05: the prompt's last row reaches `lm_head` alone, through Vulkan's
f32-activation matvec, where a whole-prompt score runs the CPU's int8-activation kernel).

**Result (as Run 2):** decode 6.8 -> 8.9 tok/s (112 ms a token), prefill 81 tok/s.

### Run 8 - 2026-11-04 ~01:20

**Change:** `moe_router` (a token a workgroup: scores, the selection bias, a stable top-k, the weights renormalized
and scaled, as the CPU router) and `moe_mat_vec_q5_0` (a Q5_0 expert stack's rows for each route, the route's
expert id read on the GPU); the ungated experts run a decode step's routes (up to 64) as up, relu^2, down and the
k-way weighted sum (a batched matmul) on Vulkan, and `moe_router_topk` routes on it.

**Tests:** `the_vulkan_router_routes_as_the_cpu_does` (every score and weight function, bias and clip);
`a_vulkan_decode_scores_as_one_prefill` with Q5_0 weights too.

**Result (as Run 2):** decode 8.9 -> 16.8 tok/s (59.5 ms a token); prefill 71 tok/s. Recipe A's expert stacks are
copied once into driver memory (~14 GB of GTT) on first use.

### Run 9 - 2026-11-04 ~02:10

**Change:** `INFERENCE_VULKAN_PROFILE` times every dispatch with GPU timestamps and prints each kernel's share every
5 s. It showed the weight reads at bandwidth (Q8_0 matvecs ~79 GB/s, the Q5_0 experts ~78 GB/s) and three kernels
far off it: the scan step (a thread a 512-byte state row, uncoalesced: 275 us), the router (its top-k on one thread:
171 us) and the f32 matmuls (a thread an output over a strided row: ~6.9 ms a token). The scan now gives a state row
32 lanes of 4 states summed by a clustered subgroup add (57 us), the router's scores and each pick's argmax run over
the workgroup (12.5 us), and a matmul with a k of 64 or more gives each output a workgroup splitting its dot product.

**Result (as Run 2):** decode 16.8 -> 22.8 tok/s (43.8 ms a token; llama.cpp's Vulkan backend: 22.2), prefill 79 tok/s.
Per token now: Q8_0 matvecs ~20 ms, Q5_0 experts ~14 ms, the shared expert ~5 ms, everything else ~5 ms.

### Run 10 - 2026-11-04 ~02:50

**Change:** `paged_cache_write` (reshape_and_cache over f32 vLLM-layout caches, the i64 slots read on the GPU) and
`paged_decode` (a workgroup a (sequence, head): scores into shared memory, softmax, then a value dimension a thread);
the CPU-only build's paged-attention ops take them on Vulkan, the prompt's gather runs the CPU's on host copies, and
`--paged-attn on` enables paged attention on Vulkan, so the MTP head attaches.

**Tests:** `vulkan_paged_attention_matches_the_cpu` (cache bytes equal, decode within 1e-4);
`an_mtp_head_drafts_on_vulkan_without_changing_greedy_output`.

**Result (`--pa-context-len 8192`):** paged decode 22.8 tok/s, as non-paged; MTP 2 15.5 and MTP 3 15.4 tok/s. The
GPU is busy 67 % of an MTP step: Mamba's multi-token path (verify and replay) still runs on the CPU, and each verify row
is its own matvec dispatch, reading the weights once a row. Without `--pa-context-len` the cache is fitted to free memory
and one layer's buffer passes the 4 GiB binding limit.

### Run 11 - 2026-11-04 ~04:00

**Change:** multi-row Q8_0/Q5_0 matvecs (`mul_mat_vecs_*`, up to 8 rows, each weight block decoded once for all rows;
one row keeps the single-row kernels, 76 vs 61 GB/s). The scan and conv step kernels take a sequence (`seq` steps a
dispatch, the state held across them), so granite's multi-token path (prompt chunks, MTP verify and replay) runs them
on a copy of the gathered state. Three host round trips went: the pool's slot clamp (a u32 `minimum` the backend runs
on the CPU; the row select clamps its ids instead), the replay's scatter indices (stashed from the step's host
indices), and the sampler's full-vocabulary filter and `WeightedIndex` for top-1 (it takes the argmax). A batch is
submitted after 8 dispatches when nothing is in flight, so the GPU starts soon after a synchronize.

**Tests:** the scan test runs 3 steps and checks the advanced state; the tiny Nemotron-H takes 128 states, so its
Vulkan tests run the scan kernels; `top_1_sampling_at_a_temperature_takes_the_argmax`.

**Result (`--pa-context-len 8192`):** MTP 2 15.5 -> 25.7 tok/s, MTP 3 15.4 -> 25.2, plain decode 22.8 -> 23.7, prefill
79 -> 103 tok/s (llama.cpp's Vulkan backend: 22.2 decode). The GPU is busy ~62 % of an MTP step before the sampler
change; prefill's matmuls and experts past 8 rows still run on the CPU.

**KLD (recipe A on Vulkan, `perplexity --device vulkan:0 --kld-base` over Run 3's BF16 base, 40 chunks of 512):** mean
0.00885 (median 0.00355, p99 0.078), top-1 96.22 %, PPL 6.5454 against the base's 6.5329; the CPU scored 0.0090.

### Run 12 - 2026-11-04 ~05:00

**Change:** a verify step's experts: 77 % of its 3 x 6 routes are distinct experts, so `moe_mat_vecs_q5_0` has the
first slot of each expert's run compute every slot routed to it (up to 8), the others leaving (766 -> 697 us a
dispatch). Paged decode splits the context into 128-position chunks, a workgroup a (chunk, kv head, sequence) scoring
all 16 of the kv head's query heads from vec4 key tiles in shared memory and summing values straight from the cache
(a value vec4 serving 8 heads), and `paged_decode_reduce` combines the chunks: before, each query head re-read its kv
head's cache, 345 -> 1038 us a call over a 512-token run, now 60 -> 110 us, with no context limit. f32 matmuls of 16 or
more rows and columns take a 64 x 64 tiled kernel (a long prompt's attention had launched a workgroup an output), and a
batch submits once its matmuls pass 4e9 multiply-adds: a 4096-token prompt had tripped amdgpu's job timeout.

**Tests:** `vulkan_expert_matvecs_match_the_dequantized_experts` (one token, shared experts, an expert past 8 slots);
`vulkan_paged_attention_splits_long_contexts` (32 over 2 heads, head 128, 700 and 257 positions, shuffled blocks);
`vulkan_tiled_matmuls_match_the_cpu`.

**Result (Run 6's four prompts, greedy, 256 tokens, `serve --vulkan --pa-context-len 8192 --paged-attn on`):**

| config | code | list | prose | counting | mean |
|---|---|---|---|---|---|
| plain | 24.47 | 24.47 | 24.42 | 24.40 | 24.44 |
| MTP 2, Q5_0 draft head | 25.33 | 25.04 | 22.21 | 28.91 | 25.37 |
| MTP 3, Q5_0 draft head | 25.43 | 23.99 | 19.44 | 30.77 | 24.91 |

MTP's text is byte-identical to plain's on all four. A verify step costs ~2.4 plain tokens (its experts ~2.3x, the
draft head's Q5_0 lm_head 3.3 ms a draft), so MTP gains 4 % on real text against the bench's synthetic prompt (26.5).
`bench --depth 4096`: plain 22.6 tok/s.

### Run 13 - 2026-11-04 ~05:40

**Change:** greedy sampling with no logprobs, penalties or bias asked takes the argmax on the Vulkan device and reads
back the token alone, as CUDA's top-1 does, instead of the vocabulary's logits and their softmax (the MTP draft rows
and plain decode take it; the verifier's target rows still score on the CPU).

**Result (Run 12's prompts):** plain 24.44 -> 25.05 tok/s, MTP 2 25.37 -> 26.52, text unchanged. Plain decode now
reads ~3.3 GB a token at ~82 GB/s, the iGPU's measured peak: only drafting gets past it.

### Run 14 - 2026-11-04 ~07:00

**Question:** the verifier's lesson from the DFlash 2 runs (a wide verify step should read its weights at the 1-row
kernel's bandwidth): how far off are the multi-row matvecs, and what closes the gap?

**Finding (`matvec_bandwidth`, a 131072 x 2688 weight, GB/s):** the f32 kernels were compute-bound past one row, each
block's 32 scalar activation loads per row and the decoded block in registers dominating.

| rows | Q8_0 f32 | Q8_0 int8 | Q5_0 f32 | Q5_0 int8 |
|---|---|---|---|---|
| 1 | 77.3 (1-row kernel) | 80.4 | 75.7 (1-row kernel) | 82.2 |
| 2 | 59.3 | 81.0 | 40.0 | 80.6 |
| 3 | 52.8 | 80.7 | 33.1 | 77.9 |
| 4 | 47.1 | 79.1 | 27.1 | 73.6 |
| 8 | 31.2 | 66.6 | 16.6 | 60.0 |

(vec4 f32 loads and 4 weight rows a workgroup first took 3 rows to 68.5 / 52.2; staging the activations in shared
memory dropped occupancy and fell to 21 / 13.)

**Change:** activations are quantized to int8 blocks (`quantize_q8_1`: d = max |x| / 127, the CPU's Q8_0 rounding,
with an f32 scale and the block's d * sum q), and every matvec packs a weight block's values to int8 once and takes
`dotPacked4x8EXT` against each row's (`VK_KHR_shader_integer_dot_product`, now required; RDNA3's `v_dot4`). Q5_0's
values go in as 0..31, the 16 offset taken off through the sum. The expert matvec does the same, one kernel for decode
and verify. The f32 matvec kernels are gone. `matvecs_round_only_their_activations` holds each output to its blocks'
int8 rounding bound.

**Result (Run 12's prompts):**

| config | code | list | prose | counting | mean |
|---|---|---|---|---|---|
| plain | 25.55 | 25.38 | 25.55 | 25.74 | 25.55 |
| MTP 2 | 33.17 | 33.92 | 28.40 | 38.74 | 33.56 |
| MTP 3 | 35.17 | 31.73 | 26.12 | 43.50 | 34.13 |

MTP's text is byte-identical to plain's. (With int8 dense matvecs but f32 experts, MTP 2 measured 28.74 and two texts
split at near-ties of 0.142 and 0.162 nats.)

### Run 15 - 2026-11-04 ~08:30

**Question:** after Run 14, does MTP pay at long context, and how deep should it draft?

**Depth (Run 12's prompts):** MTP 4 32.30 and MTP 5 31.39 tok/s (accept 2.6-2.9), auto-depth up to 5 31.54: MTP 3
stays best. At `bench --depth 4096` MTP 3 led plain by 26.2 to 23.0, its 4-row verify attention ~4 ms a call.

**Change:** paged decode's chunks are 64 positions (twice the workgroups) and its values are staged a tile at a time
in shared memory with coalesced vec4 loads (the cache's (dimension, position) rows had been read a dimension a
thread, 128 bytes apart); the shared tiles take heads up to 128. GPU time at 4100 positions: 1 query 337 -> 279 us,
4 queries 1120 -> 760 us: still latency-bound (~30 GB/s; each tile waits on a block-table read, then its K/V). An MTP
head reading a whole prompt calls it with a query a prompt row: those run in pieces whose partials stay within 256 MiB
(64-position chunks put a 4096-token prompt's at 4.4 GB, past the 4 GiB binding limit).

**Result:** `bench --depth 4096` MTP 3 26.7 tok/s, plain 23.4; Run 12's prompts plain 25.97, MTP 3 34.41 (text as
plain's from the same build; against Run 14's plain, two texts split at near-ties under the new summation order).
