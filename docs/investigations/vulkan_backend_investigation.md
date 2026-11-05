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

### Run 16 - 2026-11-04 ~22:00

**Question:** prefill: a prompt's matmuls (more than 8 rows) and experts (more than 64 routes) ran the CPU's kernels,
~100 tok/s against llama.cpp's 411 on Vulkan.

**Change:**
- `mul_mm_q8_0/q5_0`: a tiled int8 matmul over `quantize_q8_1`'s blocks (64 weight rows x 64 tokens a workgroup, 4
  blocks a step through shared memory, 4 x 4 outputs a thread, `dotPacked4x8EXT`): 4.4 int8 TOPS from 64 to 512
  tokens.
- Experts: `moe_sort` groups a step's routes by expert into a tile list, and `moe_mm_q5_0` takes it. The tiled
  kernel's weight side tops out near 26 GB/s (load, unpack, shared store, barrier), flat from 24 to 64 tokens, and an
  expert sees ~24 tokens of a 512-token step: 17-20 ms a call, whatever the tile shape (64 x 32, 128 x 32, 8 blocks a
  step, coalesced weight reads, a register prefetch that cost occupancy, wave32: all within 15-23 ms). Removing the
  weight reads took it to 11 ms. So `mat_mul_q_rows.glsl` streams the weights as a matvec does, a thread a weight row
  unpacking each block once into registers, while the step's token blocks (up to 64) sit in shared memory and every
  thread reads them at once; it stops at the tile's real tokens. The isolated benchmark (`expert_matmul_throughput`:
  3072 routes over 128 experts of 1856 x 2688) went 17.1 -> 13.8 ms, 11.4 ms a call in the model. Dense prompt matmuls
  stay tiled (row streaming wins at 24-32 rows, 65 GB/s, and loses at 64 to 3.2 TOPS).
- The type includes split loading a block (`read_raw`) from unpacking it (`unpack_raw`).

**Tests:** `matvecs_round_only_their_activations` takes 70 rows (two token tiles); the expert test a 150-route prompt
(an expert past a tile); `a_long_quantized_prompt_on_vulkan_scores_as_its_decode` (Q8_0 and Q5_0: Vulkan's tiled
prefill against its own decode, 0.0000 apart). Against the CPU an ISQ checkpoint's long prompt is 0.01 apart on average
(int8 activations rounded with an f32 scale here, f16 there), its Mamba state carrying a few positions to 0.37, so
that comparison checks the mean.

**Result:** `bench` TTFT for 512 tokens: 1.69 s (302 tok/s) with the tiled experts, 1.29 s (397 tok/s) streamed.
Prefill kernels per 512 tokens: experts 53 %, dense matmuls 28 %, elementwise 9 %, the scan 6 %. Decode unchanged
(Run 12's prompts: plain 24.7, MTP 3 34.1, with a browser running). A ~30-token chat turn prefills at ~110 tok/s: its
routes touch nearly every expert, so a step reads most of the 20 GB of expert weights.

### Run 17 - 2026-11-03 ~21:00 (desktop: RTX 3090, NVIDIA 615.71, no iGPU)

**Question:** the backend's buffers are host-visible memory, which a discrete GPU reads over PCIe (this 3090 has a
256 MiB BAR). Can a discrete GPU be a target as well, with weights and activations in its own memory?

**Change:** `HostBuffer` became `Buffer`. On a unified device (the iGPU, or a CPU device such as llvmpipe) it stays
mapped host RAM. On a discrete one it is device-local memory, which the host writes and reads through 64 MiB staging
copies recorded on the stream; fills use `cmd_fill_buffer`. The pool keeps the two placements apart. The backend's
uploads, readbacks and zero fills now go through `write` / `read` / `fill` rather than the mapping.

**Build note:** Ubuntu 24.04's `glslc` (shaderc 2023.8) has no `GL_EXT_integer_dot_product`. `GLSLC` in
`~/.cargo/config.toml` points at the LunarG SDK's (1.4.363.1, shaderc 2026.4) instead.

**Tests:** `scripts/local_ci.sh --vulkan` passes on the 3090 (18 tests). `buffers_take_writes_fills_and_reads_wherever_
they_live` covers both placements, past one staging buffer.

**Measured on the 3090:**
- Device-to-device bandwidth: the copy engine 740 GB/s (`copy_bandwidth`), a shader copy 828 GB/s
  (`copy_b32_bandwidth`).
- A fresh allocation costs more than on the iGPU:

| size | device | host |
|---|---|---|
| 4 KiB | 78 us | 57 us |
| 1 MiB | 92 us | 115 us |
| 16 MiB | 277 us | 907 us |
| 128 MiB | 1.5 ms | 15 ms |

A run of unsynchronized ops that cannot reuse pooled buffers (they return to the pool only once their batch's fence
signals) pays these costs; with a synchronize between runs, the pool serves them.

### Run 18 - 2026-11-03 ~21:20

**Question:** the prompt's Mamba scan (`ssm_scan_step` over `seq` steps) is one serial loop a state row. What does a
Lightning layer's scan cost on the 3090?

**Bench:** `prompt_scan_time` (`#[ignore]`): one batch row, 64 heads x 64 x 128 states, 8 groups, synchronized runs.
Serial: 128 tokens 0.297 ms, 512 tokens 1.051 ms, 2048 tokens 4.108 ms (~2 us a token a layer, about 24 ms a 512-token
prompt over 23 layers).

### Run 19 - 2026-11-03 ~21:40

**Change:** chunked SSD for prompts, in three passes over 32-position chunks (`ssd_head.glsl`):
1. `ssd_chunk_state`: each chunk's inclusive dt * a sums, and its own state from zero, in register tiles (an n quad
   by up to 16 p rows a thread).
2. `ssd_state_pass`: a thread per state item carries the state across chunks, eight chunks' loads at a time. It
   leaves in place the state each chunk starts from and advances the pool row.
3. `ssd_chunk_out`: the carried state's output (start-state columns staged 16 at a time), the decayed causal C.B
   times x, and D x, in 4 x 4 tiles.

`ScanStep` takes optional scratch; the backend gives it scratch for 128 tokens or more, and shorter scans step in
order.

**Dead ends:**
- The first version (scalar, two shared loads a multiply-add) took 1.38 ms at 512 tokens, slower than the serial
  kernel.
- Register tiles took that to 1.86 ms. Bisecting the chunk-state pass pinned it: the prefix sums alone 0.045 ms, the
  loads and stores 0.27 ms, the multiply-adds the remaining 0.9 ms. The store loop ran to a runtime row count, which
  indexes the accumulator array dynamically, so the driver kept the array in local memory. Looping to the constant
  bound and breaking fixed it.
- Earlier per-pass timings, from unsynchronized runs and from seds that missed rustfmt's line splits, were
  allocation-bound and misleading.

**Result (synchronized):**

| tokens | serial | chunked | chunk state | state pass | chunk out |
|---|---|---|---|---|---|
| 128 | 0.297 ms | 0.296 ms | 0.158 | 0.079 | 0.117 |
| 512 | 1.051 ms | 0.762 ms | 0.371 | 0.144 | 0.324 |
| 2048 | 4.108 ms | 2.644 ms | 1.178 | 0.430 | 1.089 |

`a_chunked_vulkan_scan_matches_its_steps` checks a 150-token scan (head 80, a partial last chunk, a padded row)
against the same scan in serial pieces, outputs and pool (2e-4). The chunk-state pass is still well off bandwidth
(57 MB in 0.37 ms). Whether it is worth more depends on its share of a real prefill, still to be measured.

### Run 20 - 2026-11-03 ~22:00 (laptop: Radeon 780M)

**Question:** a Vulkan weight was held twice: its blocks in host memory for the CPU's kernels, and a driver-allocated
copy for the GPU's, the copy in a pool size class rounded up to a power of two.

**Change:** a Q8_0 or Q5_0 weight on Vulkan keeps its blocks only in a driver buffer allocated to size and never
pooled (`Buffer::exact`), copied there when the storage is made; its CPU-side `QuantizedType` is `MappedBlocks`,
the same `QuantizedType` methods as a `Vec` of blocks (one macro now writes both) over the mapped pages, so the CPU's
fallback kernels, embedding lookups and quantizing read or write the very memory the GPU reads. Other types keep host
blocks and their lazy GPU copy, now also exact. A device whose buffers are not mapped (a discrete GPU) keeps host
blocks for every type.

**Test:** `a_vulkan_weight_holds_its_blocks_once` (the blocks and the GPU buffer share a pointer; bytes and
dequantization as the CPU's).

**Result (recipe A served, after one request):** system memory in use 47 -> 27 GB, the process's RSS 29 -> 20 GiB
(mostly the shared mapped pages), GTT 21 -> 22 GiB. Prefill 395 tok/s and decode 24.4 tok/s as before.

### Run 21 - 2026-11-04 ~03:30 (desktop: RTX 3090)

**Question:** which GGUF types beyond Q8_0 / Q5_0 can the int8 kernels take without a main per type?

**Change:** a type now unpacks one 32-value sub-block of its stored block (32 or 256 values) to int8 halves (values
0..15 and 16..31) and (s_lo, s_hi, o_lo, o_hi), with w = s q + o over each half. `block_at.glsl` finds a sub-block
inside a superblock for every main (the matvec, the tiled and row-streaming prompt matmuls, the expert matvec).
`quantize_q8_1` writes (d, d * sum_lo, d * sum_hi, 0) for each activation block, so the offsets need no per-type
code. New matvec and prompt-matmul kernels: Q4_0, Q4_1, Q5_1, Q4_K, Q5_K, Q6_K (Q6_K is the one type whose halves
take their own scales).

**Tests:** `vulkan_quantized_matmuls_match_the_dequantized_weight` covers all eight types at 1, 3, 8 and 70 rows
(97 x 2816), against the CPU's dequantized weight. A deliberately broken Q4_K decoder fails it (132.5 vs 1.51).

**Dead ends:**
- The first A/B showed the expert prompt matmul 5.6x slower (11.4 against 2.0 ms). It was the GPU at idle clocks:
  the expert bench is ~20 ms of work, and it ran first. Run warm, a bench's second run is the one to read.
- Warm, the general form still cost the tiled matmul 16 % (26.3 against 31.7 TOPS at 512 tokens), from vec4
  coefficients and scales in its shared tiles where master kept a float and a vec2.

**Fix:** a type declares what it needs: `SPLIT_HALVES` (vec4s), a free offset (vec2s), `OFFSET_RATIO` (Q5_0's -16,
Q4_0's -8: s alone), or `NO_OFFSET` (Q8_0: s, and d alone for the tokens). The dot is one 8-term `dotPacked4x8EXT`
sum, as before.

**Result (`matvec_bandwidth`, `expert_matmul_throughput`, warm; master 6460fe532 -> branch):**

| kernel | master | branch |
|---|---|---|
| Q8_0 x 1 / 3 / 4 / 8 rows | 876 / 723 / 584 / 416 GB/s | 874 / 651 / 573 / 412 GB/s |
| Q5_0 x 1 / 3 / 4 / 8 rows | 860 / 693 / 625 / 402 GB/s | 861 / 697 / 613 / 397 GB/s |
| Q8_0 / Q5_0 x 512 rows | 31.7 / 34.4 TOPS | 31.6 / 33.8 TOPS |
| experts, 512 tokens x 6 of 128 | 1.90 ms | 1.84 ms |

Q8_0 at 3 rows stays 10 % behind, with the same unpack and dot: left for the 780M A/B to judge.

**Model (Nemotron Nano 9B v2 Q4_K_M, `bench --vulkan`, 512-token prompt, 64 decoded):** prefill 26.8 -> 1275 tok/s,
decode 20.7 -> 59.8 tok/s. Its Q4_K and Q6_K weights had run through the CPU fallback before.

### Run 22 - 2026-11-04 ~04:30

**Question:** the 780M A/B of Run 21 (laptop): KLD 0.00891 against master's 0.00897, but 512-token prefill 388 -> 366
tok/s (-5.7 %), `mul_mm` at 512 rows -3.7 %, the 8-row Q5_0 matvec -7 %. Greedy text also moved at near-ties: the
offset as d * sum_lo + d * sum_hi rounds differently from master's d * sum q.

**Change:** the activation scales are master's (d, d * sum q) again, 8 bytes a block. Only Q6_K needs the half-sums,
and it takes them from the activation's int8 words (a packed dot with 0x01010101). Q8_0 and Q5_0 now compute exactly
master's expressions.

**Result (3090, warm; master -> this):** Q8_0 x 3 rows 723 -> 725 GB/s, x 8 rows 416 -> 415; Q5_0 x 3 rows 693 -> 696,
x 8 rows 402 -> 399; prompt matmuls at 512 rows 31.7 / 34.4 -> 31.9 / 34.3 TOPS; experts 1.90 -> 1.85 ms.

### Run 23 - 2026-11-04 ~04:30 (desktop: RTX 3090)

**Question:** Lightning recipe A (21.7 GiB) on a 24 GiB card that also drives the display. What runs, and how fast?

**First finding:** the prompt step failed with `ERROR_OUT_OF_DEVICE_MEMORY` on a 64 MiB buffer while the pool held
723 MB idle. The weight copies (made on first use) had filled the card.

**Changes, in order:**
1. A device allocation that fails frees the pool's idle device buffers and retries, then falls back to host memory.
   Result: it ran, 512-token prefill 1882 tok/s and decode 106 +- 18 tok/s; the noise was hot activations landing in
   host memory.
2. Exact (weight) buffers stay in device memory while they leave `EXACT_RESERVE_BYTES` (2 GiB) of the device's free
   memory at startup (`VK_EXT_memory_budget`: 23218 MiB here), then go to host memory. Result: 2070 tok/s, decode
   83.9 +- 0.2.
3. Pool classes past 1 MiB are eighths of a power of two, and buffers up to 1 GiB are pooled. A 2048-token prompt
   had made 190 fresh 256 MiB allocations, since a 132 MB activation took a 256 MiB class. Result: 512 tokens 2280,
   2048 tokens 1477 tok/s, still spilling.
4. Logging the live buffers at 512 tokens showed nothing large, only about 1.5 GB of 32-64 MiB temporaries. The CPU
   records a whole forward ahead of the GPU, and a dropped temporary returns to the pool only once its batch's fence
   signals.
   - Synchronizing on out-of-memory, before draining the pool, ended the spill, but stalled the overlap: 512 tokens
     1676 tok/s.
   - Instead, a pool miss on a discrete GPU now waits for the oldest submitted batch and tries the pool again before
     allocating. That gives 512 tokens 2628 tok/s and decode 108 tok/s, with nothing spilled.

**Dead ends:**
- Paged attention with 512-token prompt chunks did not help long prompts: 2048 tokens 1489 +- 540, 4096 tokens
  934 tok/s.
- A tiled-main expert kernel for discrete GPUs: 2.56 ms against the row-streaming 1.85 ms at 512 tokens, and
  5.31 against 5.46 ms at 2048. Not kept.

**Profile (2048-token prefill, a steady 5 s window):**

| kernel | dispatches | GPU time | each |
|---|---|---|---|
| `moe_mm_q5_0` | 155 | 1529 ms | 9.9 ms |
| `mul_mm_q8_0` | 234 | 822 ms | 3.5 ms |
| `ssd_chunk_state` | 77 | 348 ms | 4.5 ms |
| `matmul_f32_tiled` | 158 | 193 ms | 1.2 ms |
| `mul_mm_q5_0` | 154 | 171 ms | 1.1 ms |

The int8 dp4a kernels run at about 15-33 TOPS on the 3090, a fraction of its int8 tensor cores.
`VK_KHR_cooperative_matrix` and `VK_NV_cooperative_matrix2` are both offered: the next lever for prompts on this
card.

### Run 24 - 2026-11-04 ~05:30 (desktop: RTX 3090)

**Question:** after Run 23, a 2048-token prompt on recipe A swung from about 1 s to 19 s an iteration. What makes long
prompts erratic, and how close is llama.cpp?

**References (llama.cpp 970b51ce7 on recipe A, this 3090):**

| backend | pp512 | pp2048 | tg64 |
|---|---|---|---|
| CUDA | 4157 | 4278 | 200 |
| Vulkan, NV coopmat2 | 3845 | 3856 | 161 |
| Vulkan, KHR coopmat | 2840 | 2826 | |
| Vulkan, no matrix cores | 1506 | 1490 | |

**Findings:**
- Master runs Nemotron Nano 9B v2 Q4_K_M (6.5 GB) out of device memory at 2048 tokens. The CPU records a whole
  forward ahead of the GPU, and the batches' temporaries take more than 17 GB.
- On Run 23's branch the 9B took about 28 GB of fresh device allocations per 2048-token prompt. The idle pool's
  2 GiB cap evicted buffers of 40, 128 and 320 MiB that the next layer wanted again, and each fresh allocation is
  zeroed by the driver (system time 1.35 s at 512 tokens, 6.7 s at 2048).
- With the cap lifted, recipe A hit out-of-memory 14 times an iteration. Each one synchronized and drained the whole
  pool, and there were about 425 oldest-batch waits too. No buffer spilled, yet iterations still took 6.5, 12.5 and
  19 s at times: allocating up to the edge of the card's memory stalls the driver.
- A large CPU fallback was ruled out: the only one at 1 MiB or more was a 10 MiB `where_cond`, four times.

**Changes:**
- A discrete GPU's stream lets at most 3 submitted batches run unfinished. A submit past that waits for the oldest,
  so the CPU no longer records a whole forward ahead. The pool-miss wait of Run 23 is gone.
- The context counts every device-local byte, live or pooled. Before a fresh device allocation, idle device buffers
  are freed, largest first, until the total leaves `ROOM_MARGIN_BYTES` (512 MiB) of the startup budget, with one
  synchronize when the pool runs dry. Only then come the out-of-memory retry and the host spill.
- Idle device memory is not capped (a 4 GiB cap measured worse: recipe A median 1.26 against 1.02 s an
  iteration). Idle host memory keeps its 2 GiB cap, which is all an iGPU uses.

**Result (median seconds an iteration of a 2048-token prompt plus 4 decoded, 6 after warmup):**

| model | Run 23 | now |
|---|---|---|
| recipe A | 1.05 typical, 6.5 / 12.5 / 19.3 outliers | 1.02 (1.01-1.71) |
| Nano 9B Q4_K_M | 1.45 typical, 4.2 outlier | 1.50 (1.45-4.19, the first measured run) |

`bench`, recipe A: 512 tokens 2554 tok/s with decode 125 tok/s (Run 23: 2628 / 108). At 4096 tokens it still
spills (737 tok/s): a single 4096-token pass's working set passes the 2 GiB the weights leave. The 9B decodes at
88 tok/s after a 512-token prompt. The next gap to llama.cpp is the matrix cores (coopmat2 3845 against our 2554 at
512 tokens).

### Run 25 - 2026-11-04 ~08:00 (desktop: RTX 3090)

**Question:** do the 3090's matrix cores through portable `VK_KHR_cooperative_matrix` beat the int8 dp4a prompt
matmuls, and what do they do to accuracy? (llama.cpp on recipe A, Run 24: KHR coopmat 2840 against no matrix cores
1506 tok/s at 512 tokens.)

**Driver's shapes (`get_physical_device_cooperative_matrix_properties`):** 16 x 16 x 16 and 16 x 8 x 16 f16 (f16
or f32 accumulate), bf16 16 x 16 x 16 into f32, int8 / uint8 16 x 16 x 32 and 16 x 8 x 32 into int32, all
subgroup scope.
- Int8 at K = 32 fits a GGUF block, but each block's int32 product needs its own row-by-column rescale. KHR exposes
  no element's row or column, so that rescale goes through shared memory.
- So the kernels take f16 instead: weights dequantized through the type's `unpack_sub` into shared memory, f16
  activations, f32 accumulators across all of K. That is llama.cpp's KHR path.

**Change:**
- `mul_mmc_*`: a 64 x 64 tile, 4 subgroups each 32 x 32 (2 x 2 accumulators), 4 blocks a step.
- `convert_f16` puts the tokens into the f16 scratch, padded to whole tiles with zeros: a tile's last matrices read
  past its tokens and past a row's end, against zero weights, and an Inf there would make NaN.
- `moe_mmc_q5_0` with `moe_gather_f16` does the experts on the same main through tile hooks (`begin_tile`,
  `y_row`). The routed activations are gathered as f16 in `moe_sort`'s order.
- The pipelines are built only where the device has the shape. They are picked only off unified devices, from 9
  rows for the dense matmuls and from 48 routes an expert for the experts. `INFERENCE_VULKAN_NO_COOPMAT` (the
  tensor crate's `NO_MATRIX_CORES_ENV`) turns them off.
- The expert scratch grows for the gathered rows only where coopmat may run.

**Dead end:** the first version staged both operands in shared memory a 16-bit scalar at a time, two blocks a step:
13 TFLOPS against dp4a's 32-34 TOPS. Loading the f16 activations straight from memory and stepping four blocks gave
36-40 TFLOPS.

**Result (warm):**

| case | dp4a | KHR coopmat |
|---|---|---|
| Q8_0 / Q5_0 dense, 512 rows | 31.8 / 34.4 TOPS | 36.0 / 39.8 TFLOPS |
| experts, 512 tokens x 6 of 128 (24 routes an expert) | 1.97 ms | 2.37 ms |
| experts, 2048 tokens (96 routes an expert) | 5.55 ms | 5.16 ms |
| Lightning recipe A, 512-token prompt (median iteration) | 0.24 s | 0.23 s |
| Lightning recipe A, 2048-token prompt (median iteration) | 1.03 s | 1.00 s |
| Nano 9B Q4_K_M, 512 tokens (`bench`) | 1525 tok/s | 1538 tok/s |

The coopmat expert kernel loses where its 64-token tiles are mostly empty, hence the 48-routes threshold.

**KLD (recipe A against the BF16 base, 40 chunks of 512; the base made here with llama-perplexity, PPL 6.5503):**

| path | KLD mean | median | p99 | same top-1 | PPL |
|---|---|---|---|---|---|
| KHR coopmat | 0.00846 | 0.00325 | 0.0811 | 96.26 % | 6.5454 |
| dp4a | 0.00924 | 0.00363 | 0.0846 | 96.03 % | 6.5402 |

**Tests:**
- The tiny Nemotron-H's int8 parity tests (Vulkan against the CPU, Vulkan's prefill against its decode) now set
  `NO_MATRIX_CORES_ENV`, so their int8 tolerances stay as tight as before.
- `a_long_quantized_prompt_on_vulkan_matrix_cores_scores_near_its_decode` covers the f16 prompt path at 0.2 a
  step. It measured 0.113 at one Q5_0 step.

**Next:** NV coopmat2 (`VK_NV_cooperative_matrix2`), where llama.cpp reaches 3845 tok/s.

### Run 26 - 2026-11-04 ~09:00 (laptop: Radeon 780M)

**Question:** where recipe A's time goes on the 780M at master 5fda98d97 (`INFERENCE_VULKAN_PROFILE`).

**Decode (24.3 tok/s, the GPU busy 93 %):** 38 ms of GPU time a token. Q8_0 matvecs read ~1.49 GB a token (`ssm_in`
677 MB, `output` 374 MB, `ssm_out` 269 MB, attention 174 MB) in 18 ms, and the Q5_0 experts and shared expert ~1.32 GB
in 15.7 ms: both ~83 GB/s, the bandwidth ceiling. The rest is the scan 1.3 ms, ~350 elementwise dispatches 0.9 ms and
small kernels; ~3 ms a token the GPU waits.

**Prefill (512 tokens, 367 tok/s):** experts 41 %, dense matmuls 26 %, chunked SSD 12 % (182 ms), elementwise 8 %.
The serial scan was 6 % in Run 16, and prefill had slipped from 397 to ~367 tok/s since chunked SSD (Run 19).

**Finding (`prompt_scan_time`, a Lightning layer, now timing both paths):**

| tokens | serial | chunked |
|---|---|---|
| 512 | 4.4 ms | 7.3 ms |
| 1024 | 7.1 ms | 13.6 ms |
| 2048 | 19.2 ms | 28.7 ms |
| 3072 | 38.1 ms | 43.3 ms |
| 4096 | 60.9 ms | 57.4 ms |

On the iGPU chunked SSD grows linearly at ~14 ms a 1k tokens, while the serial scan grows faster than that and
crosses it near 3.5k tokens.

**Change:** a unified device scans serially below 3584 tokens (`UNIFIED_CHUNKED_SCAN_MIN_SEQ`); a discrete one
keeps chunked SSD from 128. `a_chunked_vulkan_scan_matches_its_steps` forces the chunked path, so it still runs here.

**Result (`bench`, recipe A):** 512 tokens 366.6 -> 403.6 tok/s (417 in one run), 2048 tokens 345.3 -> 380.3,
4096 tokens 311.6 -> 307.7 (chunked both, within noise). Decode unchanged.

**Next:** at 4096 tokens prompt attention's unfused scores (`matmul_f32_tiled`, `map_f32`, `rows_f32` over seq x seq)
take ~4 s of a 13.3 s prompt; a fused attention kernel is the long-prompt lever.

### Run 27 - 2026-11-04 ~10:30 (laptop: Radeon 780M)

**Question:** Run 26's long-prompt lever: prompt attention ran unfused (`run_attention_noflash`: the q k^T matmul,
scale and mask maps, softmax rows, the v matmul), its (positions x keys) scores written and read a pass each, ~4 s of
a 4096-token prompt.

**Change:** `flash_attn.comp`, prompt attention fused with an online softmax over 32-key tiles staged in shared
memory. A workgroup's 128 rows are every query head of a kv head's group times a few positions (Lightning: 16 heads x
8 positions), so a tile serves the whole group. A row is 4 threads, each holding a quarter of the head in registers
and summing its scores with a clustered subgroup add. It applies the caller's additive (positions, keys) mask as it
is, and, when the caller says the mask is causal, skips the tiles past each position's diagonal. Calls run in pieces
of positions within one submit's work. `run_attention_noflash` takes it on Vulkan for f32 heads up to 128 whose mask is
shared by every batch and head, and falls back to the unfused path otherwise.

**Tests:** `fused_vulkan_attention_matches_its_definition` (GQA 16, a cached prefix, partial tiles, head sizes 64, 80
and 128, masked and not; 1e-4 against the CPU's definition).

**Result (`bench`, recipe A, against Run 26):** 512 tokens 403.6 -> 421.8 tok/s, 2048 tokens 380.3 -> 410.8, 4096
tokens 307.7 -> 369.6. At 4096 tokens the attention costs ~1.1 s a prompt, where the unfused path cost ~4 s; it runs
at ~0.85 TFLOPS (64-row workgroups 361.5 tok/s, 128 rows 369.6). KLD 0.00898 (Run 26: 0.00897), top-1 96.41 %. MTP 3
text matches plain's on all four prompts. Against master's, two split at near-ties.

### Run 28 - 2026-11-04 ~08:30 (desktop: RTX 3090)

**Question:** NVIDIA's `VK_NV_cooperative_matrix2` (llama.cpp's fastest path here: 3845 tok/s at 512 tokens against
2840 for KHR). Does it take the prompt matmuls past the KHR kernels?

**Setup:**
- ash 0.38 (Vulkan 1.3.281) predates the extension, so its name and features struct are declared from the registry
  (structure type 1000593000, extension 594).
- The instance asks for Vulkan 1.3. The `*_cm2` shaders build for SPIR-V 1.6. The device enables workgroup scope,
  flexible dimensions, per-element operations, tensor addressing, block loads and buffer device addresses, which
  the decode functions' buffer references need.

**Kernel (`mul_mm_cm2_main.glsl`, Q8_0 and Q5_0):**
- A 128 x 128 workgroup-scope tile (weight rows x tokens), 32 values a step.
- W loads through a per-element decode on its block structs; the f16 tokens load transposed.
- Clamped tensor layouts give zeros and drop writes past every edge, so it needs no padding.

**Result (512 rows, warm):**

| kernel | Q8_0 | Q5_0 |
|---|---|---|
| KHR coopmat, f32 accumulation (Run 25) | 36.0 | 39.8 |
| cm2, f32 accumulation | 20.5 | 31.4 |
| cm2, BK 64 | 26.3 | 18.3 |
| cm2, f16 accumulation over all of K | ~48 | ~42 |
| cm2, f16 over 256-value spans folded into f32 | 40.3 | 31.0 |

(TFLOPS. At 64 rows the cm2 kernels were 10-20, half of a 128-token tile empty.)

**Finding:**
- Recipe A end to end is unchanged by cm2: 0.23 s an iteration at 512 tokens and about 1.0 s at 2048, as with KHR
  coopmat and nearly as with dp4a (0.24 / 1.03 s).
- llama.cpp's 3845 tok/s is about 0.13 s a 512-token prompt; our prefill is about 0.20 s of the 0.23. The dense
  matmuls are not what holds the 3090 back.
- f16 accumulation over the whole K would need a KLD check before use.

**Next:** a GPU breakdown of a whole 512-token prompt that adds up to its wall time. The per-dispatch profiler has
over-reported on NVIDIA, so the experts, the Mamba conv and scan, the f32 attention matmuls, the elementwise ops and
CPU-side gaps have to be measured before more kernel work.

### Run 29 - 2026-11-04 ~09:00 (desktop: RTX 3090)

**Question:** Run 28's next step. Where does a 512-token prompt's GPU time go on recipe A?

**Profile (`INFERENCE_VULKAN_PROFILE`, 512 tokens; a window's GPU time agrees with wall time here):**
- `moe_mm_q5_0` took 59 % of the GPU time, at 6.4 ms a call against 1.7 ms in Run 23.
- `mat_vecs_q8_0` at the last token took 44 ms a call.
- Everything else was small: `moe_sort` 2.3 %, the Mamba conv and scan about 7 %, the f32 attention matmuls about
  3 %, elementwise ops about 2 %.

**Finding:** some weights live in host memory.
- Logging the exact buffers that miss the budget: the startup budget was 21396 MiB (other processes held about
  1.8 GiB more than in Runs 23-24), the weights held 19.7 GB, and five 418 MiB expert stacks went to host memory.
- A prompt reads every expert stack, so about 1.6 GB of it crosses PCIe on each prompt.
- Weight copies are placed in first-use order, so whatever comes last (the last layers' experts, `lm_head`) is
  what misses.
- On this card the prompt's main cost is VRAM, not a kernel: llama.cpp keeps all 21.7 GiB on it.

**Dead end:** `EXACT_RESERVE_BYTES` at 1 GiB, with 2.7 GiB held by other processes: 512 tokens 1.39 s, 2048 tokens
8.1 s, 4096 tokens 48.8 s an iteration (median). The activations spilled instead. Kept at 2 GiB.

**Environment:** another process (1.7 GB of VRAM) then started on the GPU. With it, 2 GiB gives 0.78 s at 512 tokens
and 8-23 s at 2048. Measurements here need the card to themselves.

**Next:**
- Place weights by how often they are read, so that host memory gets only rarely read experts, never `lm_head` or
  the dense layers.
- When VRAM is short, take the routed experts' host share through a streaming path (as the CUDA offload does)
  rather than reading host memory in the matmul.

### Run 30 - 2026-11-04 ~10:00 (desktop: RTX 3090)

**Question:** recipe A (21.7 GiB) cannot fit the 3090 beside the desktop and a long context. Which recipe does, at
what KLD? (The Qwen recipes for this card keep 14-18 GiB of weights at a KLD of 0.017-0.022.)

**Candidates (routed and shared experts changed, the rest Q8_0 as recipe A; bartowski's imatrix; KLD against BF16,
40 chunks of 512, llama.cpp CUDA):**

| file | size | KLD mean | median | p99 | PPL vs BF16 |
|---|---|---|---|---|---|
| recipe A (experts Q5_0) | 21.7 GiB | 0.0093 | 0.0036 | 0.088 | -0.06 % |
| experts IQ4_NL | 18.2 GiB | 0.0172 | 0.0067 | 0.158 | +0.92 % |
| experts Q4_0 | 18.2 GiB | 0.0192 | 0.0073 | 0.180 | +0.91 % |

The k-quants do not apply: the experts' rows are 2688 and 1856 wide, neither a multiple of 256. IQ4_NL has no Vulkan
kernel yet, so Q4_0.

**Finding:** the Q4_0 file first ran 3 s a 512-token prompt and decoded at 21 tok/s. The expert kernels were Q5_0
only, so Q4_0 experts ran on the CPU.

**Change:** the expert kernels (`moe_mat_vecs_*`, `moe_mm_*`, `moe_mmc_*`) take all eight block types over the
shared type interface, and `MoeMatVec` carries its `BlockType`.

**Result (Q4_0 experts, no paged cache, median iteration):** 512 tokens 0.23 s, 2048 tokens 0.70 s (recipe A: 0.24 /
1.57 s, spilling), decode 111 tok/s (recipe A 24.7).

### Run 31 - 2026-11-04 ~11:00 (desktop: RTX 3090)

**Question:** a context of 256K tokens resident on the 3090 with the Q4_0 recipe, nothing spilled.

**Findings, in order:**
- A paged prompt chunk after the first gathered the whole K/V cache to the host every chunk and layer, about 3.7 GB
  over PCIe a 512-token chunk at a 131K cache, then scored the chunk against the full context. A 16K-token prompt
  stalled for 15+ minutes.
- `paged_prefill` (new) reads the paged cache in place:
  - A workgroup takes (context split, kv head, 8 rows) of one sequence, each (row, query head) in a pair of threads.
    32-position key and value tiles are staged in shared memory for every row.
  - Flash-style running sums over the split; each row is causal over its own length; `paged_decode_reduce` combines
    the splits.
  - Two bugs on the way. Writing single components of shared vec4s raced, so the value tile is floats now. And -inf
    masked scores leaked weight through exp on NVIDIA (all-ones values came out 0.72), so masked scores are a finite
    -1e30. The same bug makes master's `flash_attn.comp` score 0.023 off the CPU on this card; the laptop is landing
    its fix separately.
- Vulkan prompts were never chunked: the pipeline step chunked on CUDA only. A 49K-token prompt asked for a 9.8 GB
  mask. Vulkan now chunks as CUDA does.
- Weights were still being placed against a free-memory figure read once at startup:
  - The figure missed the paged cache allocated before the weights, so activations spilled at 15K tokens.
  - A server started while another was releasing its VRAM saw 1147 MiB free and placed almost everything in host
    memory.
  - Placement and room-making now ask the driver for free memory at each decision (`VK_EXT_memory_budget`: other
    processes and our own allocations both counted).
- An unquantized paged cache on Vulkan is now f16, half f32's memory. Its key rows hold 8 dimensions a position
  (16 bytes, as vLLM's layout has it), which the decode and prefill loaders map.

**Result (Q4_0 experts, `serve --paged-attn on`, completions over wiki text):**

| setup | 3.5K tokens | 15K | 50K | 112K |
|---|---|---|---|---|
| 131K f32 cache, before | 1.2 s | 49 s | 9.8 GB request, failed | |
| 131K f32 cache, chunked and in place | 1.2 s | 7.7 s | 39.5 s | |
| 262K f16 cache | 2.1 s | | 48.2 s | 146 s |

At 262K the server loads at 21.5 GB and peaks at 23.5 GB; one buffer spilled during the 112K prompt. A quantized
cache (the engine's q8/q4 format) would free 0.9-1.3 GB more: next.

### Run 32 - 2026-11-04 ~12:30 (laptop: Radeon 780M)

**Question:** with the model's default sampling (temperature 1.0, top_p 0.95, what the web UI sends), MTP 3 decoded
at ~12.6 tok/s against ~27 greedy. Where does a speculative step go?

**Finding:** the GPU sat idle ~95 ms of a step. Each step drains the GPU ~14 times, and ~10 of those read a full
131072-token logits row and run the CPU filter, whose top-p with no top-k sorts the whole vocabulary. Those rows are
the verify rows, the draft candidate rows in the verifier, and the draft samples. The other drains are the draft
embedding's ids and the causal mask's `where_cond`, which round-trips through the host.

**Change:** `ranked_topk.comp`, a workgroup a row, takes each row's 128 largest logits by radix select on
order-preserving keys (ties to the lower index), sorts them, and sums the softmax normalizer. It writes CUDA's packed
top-k layout. The sampler reads that back (1 KB a row, not 512 KB) and applies the CPU's top-k, top-p and min-p
rules over the 128, in `sample` (decode and drafts) and `speculative_probs` (the verifier's target and candidate
rows). Where a nucleus or min-p cut would keep a token past the 128, it takes the full CPU path. Penalties, bias,
processors, DRY and logprob requests keep the CPU path too.

**Tests:** `ranked_top_k_matches_a_sort` (131072 and 1000 tokens with ties and masked tokens: the same values and
indices as a sort, the normalizer within 1e-4); `a_ranked_row_filters_as_the_cpu_does` (top-k, top-p, min-p and
mixes give the CPU's filtered, normalized distribution within 1e-5; a nucleus wider than 128 is refused).

**Result (a served MTP 3 chat turn, 256 tokens, three seeds):**

| sampling | before | after |
|---|---|---|
| temperature 1.0, top_p 0.95 | 12.2 / 12.9 / 12.9 tok/s | 25.9 / 23.7 / 23.8 |
| greedy | 27.4 / 27.2 / 27.2 | 27.9 / 27.9 / 27.9 |
| top_k 1 | 24.6 / 23.8 / 23.8 | 24.8 / 24.7 / 24.7 |

**Next:** a step still drains the GPU at each draft (its token, then its embedding's ids) and at the causal mask.

### Run 33 - 2026-11-04 ~14:30 (laptop: Radeon 780M)

**Question:** after Run 32, a chat-sampled MTP 3 step still drained the GPU ~17 times. What can go?

**Changes:**
- The quantized embedding gathers its rows on the device (`embed_<type>.comp`, a thread a 32-value block through
  each type's `unpack`). It used to read the ids back and gather on the CPU. That was 4 drains a step (3 drafts and
  the verify forward).
- A Vulkan causal mask is built as the additive f32 matrix on the host and uploaded. `masked_fill`'s `where_cond` has
  no Vulkan kernel, so it round-tripped through the host: 3 a step.
- The stochastic verifier ranks every verify row in one dispatch and one readback
  (`Sampler::ranked_speculative_rows`). A Vulkan draft's logits become the verifier's sparse proposal rows the same
  way, so `probability_row` no longer touches the device. That took ~5.5 drains a step to 2.

**Tests:** `a_vulkan_embedding_gathers_as_the_cpu_does` (all eight kernel types against the CPU's gather) and
`ranked_speculative_rows_match_the_cpu`. Greedy text is identical to master's on Run 12's prompts, and MTP 3's
matches plain's.

**Result (a served MTP 3 chat turn at temperature 1.0, top_p 0.95, three seeds):** 25.0 / 22.9 / 22.9 ->
27.3 / 24.7 / 25.1 tok/s. The GPU is 89 % busy (Run 32: ~82 %).

**Left:** each draft's sample is read back (3 a step). Chaining drafts on the device needs the pick there, with its
uniform from the host (as CUDA's draft sampler does). The Mamba state copies take ~420 small dispatches, ~7 ms a
step. `ranked_topk` runs a workgroup a row, ~0.4 ms each.

### Run 34 - 2026-11-04 ~11:30 (desktop: RTX 3090)

**Question:** does the engine's quantized KV cache, ported to Vulkan, let Lightning (Q4_0 experts) hold 256K of
context on the 3090 without spilling?

**Port:**
- The row format is CUDA's (`kv_quant`): HND rows of bytes. Each 32-value group is rotated by a 32-wide Hadamard,
  quantized on a midpoint grid against its absmax, and given an f16 scale.
- `paged_cache_write_quant` uses one thread per (token, kv head, K or V) row.
- `paged_decode` and `paged_prefill` get `CACHE_QUANT` builds whose tile loaders dequantize into the same shared
  tiles the float builds fill.
- `hadamard_32` rotates the query into the cache's space and the output back out. The backend wrappers do both, so
  callers pass U8 caches as they are.
- The cache engine allocates HND quant rows on Vulkan and `validate_quant` admits Vulkan layers.
- Test `vulkan_quantized_cache_scores_as_its_dequantized_rows` checks q8/q8, q4/q4 and q3/q5 against a host
  fake-quantizer through CPU attention: max diff within 1e-3, prefill and decode. Giving the reference one more K bit
  fails it (0.0019).

**Run:**
```
srv.sh start --vulkan --paged-attn on --pa-context-len 262144 --pa-cache-type q8 --format gguf \
  -m /mnt/data/models/nemotron-lightning-gguf -f Lightning-exps-q4_0.gguf
longprompt.py 18765 <words>   # wiki text completions, nvidia-smi sampled each second
```

| prompt | time | peak VRAM |
|---|---|---|
| 3.0K tokens | 2.2 s | 23.3 GB |
| 45.9K | 45.7 s | 23.6 GB |
| 104.7K | 134.6 s | 24.1 GB |
| 244.5K | 669.6 s | 24.1 GB |

- The cache is 816 MB at q8, against about 1.6 GB at f16, and the server loads at 20.8 GB (f16: 21.5 GB).
- Speed matches f16 (50K: 48.2 s before) and 244K tokens completes.
- It still spilled: "device memory is full" printed during a prompt of 112K tokens or less. That message prints only
  after the idle pool is drained and an allocation still fails, so live buffers, not pooled ones, filled the card.
  The saved cache memory went to whatever grows with the context.

**Next:** find what live memory grows with context. The first suspect is `paged_prefill`'s partials: up to 256 MiB a
call, retained per layer across up to 3 batches in flight.

### Run 35 - 2026-11-04 ~12:38 (desktop: RTX 3090)

**Question:** what filled the card in Run 34, given that the idle pool was already drained?

**Finding:**
- The spill message now reports what was asked, held and free: `1600 MiB asked, 20949 MiB held, 1472 MiB free`, at
  a 105K-token prompt.
- A backtrace at the spill (a temporary env hook, since removed) pointed at `CausalMasker::make_causal_mask`, called
  from Granite's `forward_embeds`. The mask covered chunk rows x the whole context, rebuilt for every chunk.
- Granite then discarded it for every chunk but the first, since later chunks attend through the paged cache.
  Qwen3-Next, Qwen3.5 text, Voxtral and Hunyuan MoE did the same; Qwen3.5's multimodal entry already skipped it.
- All five now build the mask for the first chunk only.
- `paged_prefill`'s partials budget is now its own 32 MiB, down from the shared 256 MiB. A prompt chunk's rows
  already fill the GPU, and each layer's partials stay live until its batch completes.

**Run** (same server as Run 34, q8 cache at 262K):

| prompt | time | peak VRAM | spilled |
|---|---|---|---|
| 46K, after a 3K prompt | 36.7 s (Run 34: 45.7) | 23.6 GB | no |
| 105K, after the 46K one | 112.5 s (Run 34: 134.6) | 23.6 GB | no |
| 244K, fresh server | 715.3 s | 23.6 GB | no |
| 105K, fresh server, 32 MiB partials | 149.6 s | 23.6 GB | no |
| 105K, fresh server, 256 MiB partials | 151.1 s | 23.6 GB | no |

- Each longer prompt starts with the shorter one's text, so a run after a shorter prompt reuses its cached prefix.
  The ladder rows compare against Run 34's ladder; fresh-server rows compare only with each other.
- Run 34's 244K-token figure (669.6 s) came after its 105K prompt, so it isn't comparable with 715.3 s either.
- An exact repeat took 0.4 s.
- The partials cap is speed-neutral (149.6 against 151.1 s), so it stays at 32 MiB.

**Result:** Lightning Q4_0 with a 262K q8 cache holds 244K tokens of context on the 3090 without spilling, peaking at
23.6 GB.

### Run 36 - 2026-11-04 ~16:00 (laptop: Radeon 780M)

**Question:** the web UI's chats decoded slower than the benches, the GPU ~80 % busy. The UI's sampling defaults are
temperature 0.7, top_p 0.9, top_k 40 and a repetition penalty of 1.1. Any penalty or logit bias kept Run 32's device
top-K off, so each row went back to the full-vocabulary readback and CPU filter.

**Change:** `penalize.comp` applies the CPU sampler's frequency, presence and repetition penalties and logit bias to
a copy of the logits on the device, through a sparse list of (row, token, generated count, bias) entries, one a
distinct context or biased token of a row. Frequency and presence count generated tokens, repetition every context
token, then the bias, as `apply_freq_pres_rep_penalty` and `apply_logits_bias` do. The ranked top-K then reads the
penalized rows. A verify step's row `i` is penalized over the context plus the drafts before it, as the CPU's
verifier sees it once those are accepted, and a draft row the same way.

**Test:** `device_penalties_rank_as_the_cpu_does` (repetition, frequency and presence penalties and a bias, over a
context and drafts: each row's filtered distribution as the CPU's within 1e-5).

**Result (a served MTP 3 chat turn, 256 tokens, three seeds):** the UI's defaults went 22.8 / 20.3 / 22.5 ->
27.1 / 26.7 / 26.7 tok/s. Temperature 1.0 with top_p 0.95 is unchanged: 28.5 / 26.2 / 27.4 -> 28.6 / 26.3 / 27.2.

### Run 37 - 2026-11-04 ~17:45 (desktop: RTX 3090)

**Question:** Lightning on CUDA (Nemotron-H CUDA Runs 18-19) puts a 245K-token prompt at 116 s (bf16 cache) to 141 s
(q8), where Vulkan's took 715 s, almost all of it prompt attention over the paged cache at ~4.7 TFLOPS. Can the
3090's matrix cores take it?

**Change:** `paged_prefill_cm_main.glsl`, prompt attention on `VK_KHR_cooperative_matrix` (f16 16 x 16 x 16 into f32)
with `paged_prefill`'s bindings, push constants and partials, so `paged_decode_reduce` and the three cache types
(f32, f16, quantized) are unchanged around it:
- A workgroup (4 subgroups of 32) takes 64 query rows (positions x the kv head's query heads), Q in f16 A matrices.
- It stages 32 positions of K and V as f16, through the scalar kernel's index math (f32 and f16 caches). A quantized
  tile goes a thread a 32-value group: the row, its scale and its words read once.
- Each subgroup's 16 x 32 scores go through shared memory, a pair of lanes a row for the softmax. P V adds into
  accumulators that stay in matrix registers; a row's reference max moves only when a tile passes it by 8 nats (P
  stays below e^8 in f16), the one time its accumulators round-trip shared memory to be rescaled.
- ~32 KB of shared memory. Built only for 32-lane subgroups (`CoopKernels::paged_prefill`); heads a multiple of 16
  up to 128, groups up to 64.

**Tests:** `vulkan_paged_prefill_matches_each_row_scored_alone` now pins the f32 kernel (`NO_MATRIX_CORES_ENV`) at
1e-4; `vulkan_matrix_core_paged_prefill_matches_each_row_scored_alone` runs the matrix cores at 1e-3 relative
(1.0e-4 seen: f16 Q, K, V and P). `vulkan_quantized_cache_scores_as_its_dequantized_rows` passes at its 1e-3.

**Result (`inference bench --vulkan`, Q4_0 experts, one fresh prompt each):**

| prompt | cache | scalar | matrix cores, per-value quant loads | matrix cores |
|---|---|---|---|---|
| 32768 | q8 | 24.3 s | 18.0 s | 12.5 s |
| 32768 | f16 | 20.7 s | | 13.1 s |
| 131072 | q8 | | 154.0 s | 77.4 s |
| 245760 | q8 | 715.3 s (Run 35, 244K) | | 240.8 s |

(The 32K scalar rows ran with `INFERENCE_VULKAN_NO_COOPMAT`, which also takes the matmuls off the matrix cores; Run
25 measured those at 0.23 against 0.24 s a 512-token prompt.) Decode after the prompt 127-132 tok/s.

**Finding:** at 32K the f16 cache's attention is down to ~2 s of 13 (~29 TFLOPS); the per-value quantized loads had
cost the q8 cache ~5.5 s more, now gone (q8 reads half f16's bytes and is the faster cache). At 245K, against ~85 s
of per-token work, attention is ~156 s (~19 TFLOPS); CUDA's q8 is 141 s in all.

**Next:** item (3) of Nemotron-H CUDA Run 18: short prompts and decode at half CUDA's rate.

### Run 38 - 2026-11-04 ~19:30 (laptop: Radeon 780M)

**Question:** do the matrix-core kernels (`mul_mmc_*`, `moe_mmc_*`, `paged_prefill_cm*`) speed up the 780M's prompts?
They are laid out for 4 subgroups of 32, and RADV runs the 780M's compute at 64 by default (`subgroupSize` 64,
`minSubgroupSize` 32, `subgroupSizeControl` with compute in `requiredSubgroupSizeStages`).

**Finding:** at RADV's default width the matrix-core prompt path fails (the tiny Nemotron-H's matrix-core test errs).
So any wave64-default device had broken matmuls there since Run 25, not only the 780M.

**Change:**
- The matrix-core kernels are built only where compute subgroups are 32 wide or can be made so
  (`VK_EXT_subgroup_size_control`). Every one of them is created with `requiredSubgroupSize` 32 where needed
  (`Pipeline::with_subgroup`).
- `MatrixCores` (`Off`, `Discrete`, `Any`) replaces the bool; `IGPU_MATRIX_CORES_ENV` (`INFERENCE_VULKAN_IGPU_COOPMAT`)
  picks `Any`, and the two matrix-core tests set it, so an iGPU with matrix cores runs them.

**Result (Lightning recipe A, `bench --vulkan --paged-attn on --pa-context-len 8192`, interleaved, 2 rounds):**

| prompt | dp4a, scalar attention | matrix cores |
|---|---|---|
| 512 | 413-419 tok/s | 396-397 tok/s |
| 2048 | 403-404 tok/s | 372-374 tok/s |

So an iGPU keeps the matrix cores off by default.

**Trap:** `--paged-attn on` without `--pa-context-len` sizes the cache from free memory (12 GB, 2M tokens of
Lightning's ~7 KB a token here) and the device map moves 5 layers to the CPU: 250 tok/s at 512.

### Run 39 - 2026-11-04 ~18:30 (desktop: RTX 3090)

**Question:** item (3) of Nemotron-H CUDA Run 18. Vulkan decodes Lightning (Q4_0 experts) at 115-130 tok/s where CUDA
does 209. Where does a decode step go?

**Commands:** `INFERENCE_VULKAN_PROFILE=1 inference bench --vulkan --prompt-len 16 --gen-len 1024`; perf and
`/proc/<pid>/task/*/stat` over 5-8 s of steady decode; `nvidia-smi dmon -s pu`; a probe timing 1000 back-to-back
`map_f32` dispatches (2688 floats) with and without the per-dispatch barrier; llama.cpp `970b51ce7` Vulkan
`llama-bench -p 512 -n 128 -fa 1` on the same file.

**Findings:**
- The timestamp profile counts ~375 dispatches a token: ~146 `map_f32`, ~68 `quantize_q8_1`, ~32 `rows_f32`, ~29
  `mat_vecs_q8_0`, 19 each of the Q4_0 experts and Q5_0 matvecs, ~10 each of router, scan, conv and the f32 matmuls.
  It summed only ~3.2 ms of GPU a ~10 ms token, a dead end: `nvidia-smi` shows the SMs 98-99 % busy at ~280 W with
  memory at 32-45 %. Decode is GPU-bound, and the per-dispatch timestamps undercount on NVIDIA (as Run 29 found).
- The main thread is on the CPU 26 % of a step (1.31 s user in 5 s); the driver's `[vkrt] Analysis` thread spins
  ~1.8 s of 5 in the kernel by itself, unrelated to our recording.
- The probe: 7.1-9.2 us a dispatch with the barrier each dispatch records, 1.3 us without. Each barrier drains the
  GPU, ~7 us of busy-but-idle SMs; at ~375 a token, ~2.6 ms of the step.
- llama.cpp's Vulkan backend on the same card and file: pp512 4074 tok/s, tg128 176 tok/s (ours ~2200 and 115-130).

**Next:** fewer drains a token. Two levers: fuse the elementwise maps and the activation quantizes into their
neighbours (fewer dispatches), and record a barrier only where a dispatch reads or overwrites what an earlier,
unbarriered one wrote (which needs each pipeline's written bindings).

### Run 40 - 2026-11-04 ~20:30 (desktop: RTX 3090)

**Question:** Run 39's decode step, with the card to itself: what moves it?

**Note:** a Godot editor had been holding 1.7 GB of VRAM and ~37 % of the GPU during Run 39's later checks; it was
closed for these. Decode still swings ~+-15 tok/s within a 2-iteration bench, so A/Bs here alternate runs.

**Changes and results (Lightning Q4_0 experts, `--paged-attn on --pa-cache-type q8 --pa-context-len 32768`):**
- `gated_rmsnorm.comp`: Mamba-2's gated RMSNorm (silu gate, per-group RMS, weight) in one dispatch, where the
  composite took nine a Mamba layer (`a_fused_gated_rmsnorm_matches_its_composite`). Alternating six pairs, fused
  won all six: 114-121 against 109-115 tok/s (~4 %). The maps it removes were ~0.4 ms of ~8 a token by timestamps.
- Decode profile (paged, steady window, ~563 tokens): Q8_0 dense matvecs 2.51 ms a token (71 calls, ~530 GB/s),
  Q4_0 expert matvecs 1.80 ms (46), Q5_0 shared experts 0.79 ms (46), router 0.48 ms, `quantize_q8_1` 0.47 ms (163),
  `ranked_topk` 0.29 ms (one workgroup for the one row: one SM of 82), paged attention ~0.35 ms.
- `matvec_bandwidth_at_decode_shapes` and `expert_matvec_bandwidth_at_decode` (new benches): the lm_head shape
  reads at 871 GB/s, but decode's shapes fall off: Mamba in 713, Mamba out 579, attention q 553, attention k 98, the
  shared expert 465-534, the routed experts 479-502. Skipping the activation quantize saves 2.5-3 us a call.
- Experts at 8 rows a workgroup: spills (down 296 us). At 32 threads a workgroup (one subgroup, 4 rows): up 35.2 ->
  32.4 us, down 33.6 -> 29.6; 2 rows: up 31.1, down 32.8. Dense at 32 threads: decode shapes 1-13 % faster, lm_head
  unchanged, 8 activation rows Q8_0 410 -> 472 GB/s but Q5_0 393 -> 362.
- Both at 32 threads, 4 rows: decode 116.7 -> 129-133 tok/s; prompts unchanged (512 tokens 226 ms, 2048 724 ms).

**Next:** the fixed cost a dispatch (~3 us beyond its work, ~600 a token) and the remaining matvec gap to the card's
~870 GB/s; `ranked_topk` over more of the GPU for a single row.
