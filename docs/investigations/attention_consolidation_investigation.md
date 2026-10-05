# Attention consolidation investigation

Goal (#270 step 3): one CUDA attention family for prefill, decode, paged and non-paged, in place of today's Dao FA2,
FlashInfer decode (with its FP8-KV copy), vLLM PagedAttention v1/v2, the sinks kernel and the eager fallbacks.
Context: `docs/investigations/cuda_binary_size_investigation.md` Runs 14-16 (attention is ~34 MiB of the shipped
GPU code; llama.cpp's single fattn family is 5.9 MiB).

## Run 1 - 2026-10-04 12:57

Question: which implementation does each configuration reach, and what does each cover?

Command: a linker map of the Qwen-only release library
(`RUSTFLAGS="-C link-arg=-Wl,-Map=ffi.map" cargo build --release -p inference-ffi --no-default-features --features
cuda,flash-attn,code-execution,models-qwen`), summing `.nv_fatbin` input sections per archive member; then a code
read of the dispatch in `inference-nn/src/attention/` and `inference-nn/src/paged_attention/`.

Finding, linked GPU code: FA2 dense 14.30 MiB (25 objects), FA2 paged split-kv 5.78 (16), FlashInfer decode
f32/f16/bf16 3.75/3.73/3.72 (5 objects each: base + 4 FP8-KV head dims), PagedAttention v1/v2 2.33 (6), sinks 0.07,
FlashInfer MLA decode 0.06. Total linked `.nv_fatbin` 52.89 MiB, of which attention 33.8.

Dispatch:
- Layout. A layer gets the FlashInfer cache layout when K and V head dims match, the head dim is 64/128/256/512,
  the GQA group is 1..8 or 16, and `INFERENCE_RS_FLASHINFER_DECODE` is not 0 (`flashinfer/mod.rs:531-547`).
  GPT-OSS, Llama 4 and vLlama4 force the Standard (vLLM) layout; MLA has its own.
- Prefill without a cache hit, paged or not, goes through `Sdpa.run_attention`: FA2 (FA3 on sm90 for
  64/128/256/512) for f16/bf16. These go eager instead: custom masks, f32, sliding window or softcap above head
  dim 256, MLA's absorbed 576-dim q, and every prefill in a `cuda`-only build. Sinks always take `flash_attn_sinks`.
- Prefill after a prefix hit (`PrefixPrefillPlan::choose`, `plan.rs:46-97`), tried in order:
  1. FA3 FP8 paged on sm90, with an FP8 cache and head dim 256;
  2. FA2 paged split-kv (FlashInfer layout, cache dtype = activation dtype, no sinks or alibi);
  3. otherwise gather the cache, then SDPA (FA2 varlen or eager with a host-built mask).
- Decode (`DecodePlan::choose`, `plan.rs:671-690`): FA3 decode (sm90, FP8, 256) or FlashInfer batch decode on the
  FlashInfer layout. vLLM v1/v2 run on the Standard layout: GPT-OSS sinks, Llama 4, head dims 80/96/112/192, odd GQA
  groups. MLA uses `flashinfer_mla_decode`. Custom masks gather, then SDPA.
- FP8 KV is opt-in only (`--pa-cache-type f8e4m3`; default `auto` = activation dtype).
- Dead: alibi everywhere (every caller passes None), `context_attention_fwd_mla`, `kv_scale_update`,
  CUDA `swap_blocks`.

Coverage: FA2 is the most complete prefill kernel (paged, varlen, window/softcap up to 256, media-prefix spans) but
has no sinks, FP8 KV, f32 or MLA. The vendored FlashInfer is decode only (no `prefill.cuh`). No single
implementation covers sinks + MLA + window/softcap at 512.

Decision (user, 2026-10-04): port llama.cpp's fattn (mma/tile/vec) as the single family and extend
it with what it lacks: native bf16 in the mma path (upstream converts bf16 K/V to f16 into scratch on every call),
paged K/V through block tables, FP8 e4m3 KV, varlen batches and media-prefix spans. fattn already covers sinks,
softcap, sliding window via the mask, alibi, head dims 40-576 including MLA's 576/512, f16/bf16/f32 and explicit
masks.

Plan:
1. Prototype: crate `inference-fattn` with fattn vendored nearly verbatim over a minimal ggml compat shim (ggml's
   public headers vendored, plus the few ggml functions it calls), a raw-pointer C entry point and a Rust API. GPU
   parity tests against an eager reference, and benchmarks against FA2 (prefill) and FlashInfer (decode) at Qwen3.5
   and Gemma shapes. Not wired into the engine.
2. Extensions, one PR each: bf16 mma; paged block tables; FP8 KV; varlen and media-prefix spans.
3. Migrate call sites path by path behind parity tests.
4. Delete FA2, FlashInfer decode, vLLM v1/v2 and the sinks kernel.

## Run 2 - 2026-10-04 13:12

Prototype: `crates/inference-fattn`.
- llama.cpp `ggml/src/ggml-cuda/` fattn at `4617ccc1a`, vendored unchanged: `fattn*.cu*`, `common.cuh`, `mma.cuh`,
  `convert.cuh`, `vecdotq.cuh`, `cp-async.cuh`, and the 21 mma, 12 tile and 2 vector (f16-f16, bf16-bf16)
  instances.
- ggml's public headers are vendored for the declarations.
- `ggml_compat.cu` holds the 19 functions the objects link against, found by compiling every object and diffing
  `nm -u` against what they define: `ggml_nbytes` and the type-size helpers, logging and abort, `ggml_cuda_info`
  from `cudaGetDeviceProperties`, set/get device, a `cudaMallocAsync` pool on the caller's stream, and f32/bf16 to
  f16 converters.
- `entry.cu` wraps raw pointers in ggml tensor descriptors and runs `ggml_cuda_flash_attn_ext` on candle's stream.
- The build defines llama.cpp's `GGML_CUDA_FA_<K>_<V>` 0/1 for the 49 K/V pairs and passes its `-use_fast_math
  -extended-lambda`.
- Rust: `flash_attn(q, k, v, &FattnOptions { scale, softcap, mask, sinks })` over `(batch, seq, heads, dim)`
  tensors as strided views. Q goes to F32 (fattn's Q type), the output is F32 and is cast back to Q's dtype, and
  the destination is sized by `ggml_cuda_flash_attn_ext_get_alloc_size`.

Question: does it compute the right thing, and how fast is it next to FA2?

Command:
- `cargo nextest run -p inference-fattn --features cuda --profile cuda`: 7 parity tests against an F32 reference
  with GQA repeat, additive causal mask, softcap and sinks; f16 and bf16 K/V; batch 2, 8 heads.
- `--features bench-fa2 --run-ignored only -E 'test(/bench::/)' --no-capture`: `bench::fattn_vs_fa2`, bf16, batch 1, 5 warmup and 50
  timed iterations between `synchronize()`.
- nsys on the bench.

Finding, parity: 7/7 pass on the first full run (after fixing the reference to use candle-nn's composed softmax, since
`softmax_last_dim` has no CUDA kernel in this dev-dependency). Covered:
- decode with KV padded to 256 and unpadded, at head dims 64/128/256 (both ran the mma kernel, see Run 3);
- prefill at head dims 64/80/96/128/256;
- 37 queries over 512 keys (prefix-cache shape);
- head dim 512 with GQA;
- softcap 30, and sinks.
Tolerances: f16 < 1e-2, bf16 < 4e-2.

Finding, speed (fattn / FA2, us per call):

| shape | q x kv | fattn | FA2 | ratio |
|---|---|---|---|---|
| Qwen3.5-0.8B full attn (8 heads, 2 KV, hd 256) | 512 x 512 | 129.8 | 53.5 | 2.43 |
| | 2048 x 2048 | 526.5 | 407.4 | 1.29 |
| | 8192 x 8192 | 4162.2 | 4559.0 | 0.91 |
| | 1 x 512 | 21.6 | 45.9 | 0.47 |
| | 1 x 4096 | 56.4 | 317.3 | 0.18 |
| | 1 x 16384 | 176.2 | 1214.8 | 0.15 |
| Llama-8B (32 heads, 8 KV, hd 128) | 512 x 512 | 160.1 | 76.9 | 2.08 |
| | 2048 x 2048 | 832.7 | 589.6 | 1.41 |
| | 8192 x 8192 | 8198.3 | 8187.5 | 1.00 |
| | 1 x 512 | 43.3 | 48.2 | 0.90 |
| | 1 x 4096 | 95.3 | 333.0 | 0.29 |
| | 1 x 16384 | 328.1 | 1254.6 | 0.26 |

nsys over the whole bench: the attention kernels dominate both sides (FA2 `flash_fwd_kernel` ~934 ms in total,
fattn `flash_attn_ext_f16` ~618 ms). fattn's wrapping adds `cast_bf16_f32` (Q, 78.5 ms / 660 calls),
`cast_f32_bf16` (output, 42.5 ms) and `convert_to_f16<bf16>` (K/V for the f16 mma path, 37.7 ms / 1320 calls), plus
`flash_attn_mask_to_KV_max`. There is host work per call too: the scratch `cudaMallocAsync` and the candle casts.

Implication:
- Decode beats FA2 by 1.1-6.9x, but FA2 is not built for decode. The real decode comparison is FlashInfer's paged
  decode, which needs the paged extension first.
- Long prefill is at parity.
- Short prefill is 1.3-2.4x slower; the cause is not established (Run 3 rules out one explanation). The
  conversions above are real per-call work, and the planned bf16 extensions remove them, but the decode timings
  (21.6 us end to end with the same wrapping) show the wrapping alone cannot explain 75-300 us.
- Next: those, then paged K/V. Re-run this bench after each.

## Run 3 - 2026-10-04 13:28

Review follow-up on the prototype.

Changes:
- **Validation.** fattn's checks are `GGML_ASSERT`s that abort the process (a GQA ratio that does not divide aborts
  even inside `supported()`), so `src/cuda.rs` now validates operands before building any descriptor:
  - q/k/v head dims and batch/seq/heads agree, and `n_head % n_head_kv == 0`;
  - K and V are f16 or bf16, of one dtype;
  - the mask is f16 with a contiguous last dim and shape `(b | divisor of b, seq_q, seq_kv)`;
  - sinks are contiguous f32 of length `n_head`;
  - the device index is below 16, and the stream is not null.
- **Guards.** The cudarc device-pointer guards stay alive across the launch, and dst uses `device_ptr_mut`.
- **Memory.** An F32-Q result is copied, so the f16 K/V scratch is not kept alive with it.
- **Smaller fixes.** `supported()` builds descriptors from layouts without a GPU cast; `causal_mask` rejects
  `seq_q > seq_kv`; the build adds `-DNDEBUG` (llama.cpp's Release build does); a CONT log line takes the level of
  the line it continues.

New finding, the 576/512 (MLA) contract: with an independent V, 576/512 returned wrong values (max diff 0.53 on a
peak of 0.38). The mma kernel hard-codes `V_is_K_view = DKQ == 576` (`fattn-mma-f16.cuh:1986`) and reads V out of
K's tiles, as llama.cpp only uses that pair for DeepSeek's absorbed attention, where V is the leading 512 dims of K.
`validate` now requires V to share K's storage, start offset and row strides at head dim 576, and the test passes V
as `k.narrow(3, 0, 512)`.

Tests (12, `cargo nextest run -p inference-fattn --features cuda --profile cuda`, all pass):
- Tolerance is now relative, max |diff| / max |reference| <= 4e-3 (f16) and 1e-2 (bf16).
- The kernel claim in Run 2 was wrong: with batch 2 and GQA 4, every decode case ran mma, not vec. On Ampere, vec
  is chosen for decode only when the GQA optimisation does not apply, so `decode_vec` uses batch 1 with GQA 1.
- New cases: the tile kernel at head dims 40 and 72; 112; no-mask (encoder); MLA-shaped 192/128 and 576/512; a
  strided (transposed) Q; F32 Q; and the rejection of mismatched operands (including the GQA ratio that used to
  abort).
- The FA2 bench moved behind a `bench-fa2` feature. As a plain dev-dependency with `cuda` on, it switched candle's
  CUDA on for the workspace's CPU test build: three inference-quant HQQ tests then picked the GPU without
  inference-quant's CUDA kernels and failed (`no cuda implementation for bitwise`).

Hypothesis tested: the review suggested the short-prefill gap is fattn skipping fully masked KV tiles
(`flash_attn_mask_to_KV_max`) only for seq_q >= 1024 or batch > 1, so a batch-1 512 prefill computes the whole
square. The bench now has batch-2 rows:

| shape | b x q x kv | fattn us | FA2 us | ratio |
|---|---|---|---|---|
| Qwen3.5-0.8B | 1 x 512 x 512 | 130.6 | 53.4 | 2.45 |
| | 2 x 512 x 512 | 161.5 | 86.9 | 1.86 |
| | 1 x 2048 x 2048 | 519.0 | 407.5 | 1.27 |
| | 2 x 2048 x 2048 | 995.5 | 678.7 | 1.47 |
| Llama-8B | 1 x 512 x 512 | 159.0 | 76.4 | 2.08 |
| | 2 x 512 x 512 | 263.3 | 96.0 | 2.74 |
| | 1 x 2048 x 2048 | 815.3 | 591.4 | 1.38 |
| | 2 x 2048 x 2048 | 1917.5 | 1120.1 | 1.71 |

Batch 2 does not close the gap, so tile skipping is not the explanation. The cause of the prefill gap is open. Next
measurement: per-call kernel time against wrapper time at 512 (nsys trace, one shape), before choosing between
kernel-side work and wrapper-side work.

Recorded for wiring later: the pool allocates with `cudaMallocAsync` during calls, and graphs with memory nodes have
restrictions (`cudaGraphExecUpdate`, concurrent instantiation). That needs checking against `cuda_graph.rs` before
fattn runs inside captured decode graphs.


## Run 4 - 2026-10-04 19:07

Question: is the short-prefill gap (Run 3: fattn 1.9-2.7x FA2 at 512) in the kernel or in the wrapper around it?

Command: the bench gained a `FATTN_BENCH_FILTER` env var (substring of the row label), so one row runs under nsys:
`FATTN_BENCH_FILTER="llama-8b                 b 2 q   512" nsys profile -t cuda --export sqlite <bench binary> --ignored --exact bench::fattn_vs_fa2`,
then per-kernel averages from `CUPTI_ACTIVITY_KIND_KERNEL`. RTX 3090 (SM86, no perf-counter access, so no ncu).

Per-call breakdown before this run's changes (us, 55 calls averaged):

| row | fattn kernel | stream-k fixup | candle casts (Q to f32, out to bf16) | K/V to f16 | FA2 kernel |
|---|---|---|---|---|---|
| Qwen3.5-0.8B 1 x 512 | 54.8 | 35.8 (general) | 22.2 + 10.0 | 7.3 | 50.3 |
| Llama-8B 1 x 512 | 84.8 | - | 44.5 + 20.7 | 11.3 | 65.3 |
| Llama-8B 2 x 512 | 120.3 | - | 89.6 + 46.2 | 18.6 | 102.9 |
| Llama-8B 1 x 2048 | 533.3 | - | 180.1 + 90.9 | 33.2 | 608.1 |

Findings:
- The wrapper was the bulk of the gap. Candle's dtype casts around the call cost more than the attention at 512
  (155 us of 302 at 2 x 512), and at 2048 the fattn kernel alone already beats FA2 (533 vs 608 us).
- The Qwen row's general stream-k fixup (36 us for 164 blocks over 64 output tiles) is upstream behaviour: llama.cpp's
  own `llama-bench -p 512 -fa 1` on Qwen3.5-0.8B-Q8_0 under nsys shows the same kernel at 60.8 us and
  `flash_attn_stream_k_fixup_general` at 36.0 us.
- Dead end, kept for later: raising `max_efficiency_loss_percent` (stream-k rounding, fattn-common.cuh) from 5 to 25
  rounds 164 blocks down to 128, which takes the uniform fixup. The kernel then runs in 45.6 us and the fixup in
  14.3 us, so the row goes from 136 to 101 us. 50 and 100 measured the same, and no other bench row moved. That is too
  narrow a basis for changing upstream's heuristic (Ada and newer always use stream-k), so it was reverted.

Change: native bf16 Q and output.
- Every fattn kernel (mma, tile, vec) and the combine/fixup kernels take `Q_bf16` / `dst_bf16` runtime flags. Q
  loads and final stores go through `fattn_load_q*` / `fattn_store_dst*`. Runtime flags rather than template
  parameters, so the instance count (and the binary) does not double; both are taken once per tile.
- A bf16 dst cannot hold the unnormalized partial that a stream-k block writes into dst when it joins a tile midway
  and finishes it (`needs_fixup`), for the fixup to read back. With a bf16 dst that partial goes to an extra f32 slot per block after the fixup data, and both
  fixup kernels read it from there.
- The f16 K/V copies now come from the stream-ordered pool instead of space after dst. The output buffer is then
  exactly the output, so the copy that released the scratch is gone (`inference_fattn_alloc_size` removed).
- The wrapper passes f32 and bf16 Q straight through when the head dim is contiguous and the offset/strides meet
  the 16-byte vector loads; otherwise it makes a fresh offset-0 copy (bf16 stays bf16, f16 goes to f32).

Tests: the parity suite (13 tests) passes. New cases are `stream_k_fixups` (batch 1, few tiles) and a
misaligned-Q case. The old strided-Q case did not test a strided bf16 Q at all: candle's cast made it contiguous,
so views are now taken after the cast. An nsys run over the suite confirms it reaches every output path:
`flash_attn_ext_f16` 56 times, uniform fixup 40, general fixup 14, combine 12, vec 8, tile 4.

Bench after (us):

| shape | b x q x kv | fattn | FA2 | ratio | Run 3 ratio |
|---|---|---|---|---|---|
| Qwen3.5-0.8B | 1 x 512 x 512 | 100.3 | 53.4 | 1.88 | 2.45 |
| | 2 x 512 x 512 | 95.3 | 86.5 | 1.10 | 1.86 |
| | 1 x 2048 x 2048 | 373.5 | 407.7 | 0.92 | 1.27 |
| | 2 x 2048 x 2048 | 717.4 | 696.4 | 1.03 | 1.47 |
| | 1 x 8192 x 8192 | 3596.0 | 4551.1 | 0.79 | 0.91 |
| | decode kv 512 / 4096 / 16384 | 16.3 / 50.8 / 154.9 | 45.8 / 315.2 / 1216.0 | 0.36 / 0.16 / 0.13 | 0.49 / 0.18 / 0.15 |
| Llama-8B | 1 x 512 x 512 | 102.1 | 77.2 | 1.32 | 2.08 |
| | 2 x 512 x 512 | 129.1 | 95.6 | 1.35 | 2.74 |
| | 1 x 2048 x 2048 | 541.8 | 597.6 | 0.91 | 1.38 |
| | 2 x 2048 x 2048 | 1398.6 | 1162.0 | 1.20 | 1.71 |
| | 1 x 8192 x 8192 | 7217.7 | 8196.1 | 0.88 | 0.99 |
| | decode kv 512 / 4096 / 16384 | 36.1 / 87.8 / 303.7 | 48.5 / 331.5 / 1250.7 | 0.74 / 0.27 / 0.24 | 0.91 / 0.28 / 0.26 |

What remains is kernel-side, plus the K/V conversion to f16:
- Llama 2 x 2048: kernel 1334 vs FA2 1117 us, plus 2 x 31 us converting K/V.
- Llama 1 x 512: kernel 83 vs 66 us, plus 2 x 5.6 us converting K/V.
- Qwen 1 x 512: the general fixup above.

Review follow-up:
- The fallback used `contiguous()`, which keeps a contiguous view at an unaligned offset (and candle skips the stride
  check on size-1 dims), so such a Q could still reach the 16-byte loads. It now uses `force_contiguous()`. A Q passes
  through only if its byte strides fit the kernels' i32 `nb0x`.
- Every parity case now also asserts `supported()`. That caught a bug from Run 3: `supported()` described operands
  with null pointers, and the entry point treats a null mask as absent, so the GQA optimisation never applied there.
  At 512 and 576 it answered false for operands `flash_attn` runs. It now describes them at a non-null probe address.
- More coverage: a contiguous Q at an odd offset; strided and f32 Q on the vec (decode) and tile (hd 72) kernels.

Next: bf16 K/V read straight by the mma kernel (drops `convert_to_f16`), then paged K/V.

## Run 5 - 2026-10-04 19:34

Question: what does the bf16 to f16 K/V conversion (`convert_to_f16`, before every mma launch) cost, and can the
mma kernel read bf16 K/V itself?

Measurement (nsys, as in Run 4), decode rows after #273:

| row | convert_to_f16 | fattn kernel | total |
|---|---|---|---|
| Qwen3.5-0.8B q 1 kv 16384 | 2 x 59.4 us | 47.7 us | 175.6 us |
| Llama-8B q 1 kv 4096 | 2 x 30.5 us | 27.9 us | 96.7 us |
| Llama-8B q 1 kv 16384 | 2 x 117.3 us | 85.5 us | 360.7 us |

In decode the conversion is about two thirds of the call: the GQA-optimised mma path copies the whole cache to f16
on every step.

Change: the mma tile loader (`flash_attn_ext_f16_load_tile`) takes a runtime `KV_bf16` flag. With it, each 16-byte
chunk is loaded into registers, converted to f16 and stored to shared memory. cp.async cannot convert, so the chunk
goes the synchronous way. Sync stores land no later than the async ones they replace, and every reader already sits
behind `cp_async_wait_all(); __syncthreads()` (the mask still loads with cp.async), so the stage ordering holds. This adds no instances: the flag joins `Q_bf16` / `dst_bf16` on the kernel
signature (vec and tile ignore it; vec has bf16 instances and tile still converts).

First result, with the in-kernel conversion always on:
- Decode was 3-4x faster.
- Long prefill got slower: Llama 1 x 8192 went from 7218 to 8068 us, 2 x 2048 from 1399 to 1554 us. Each K/V tile is
  loaded once per Q tile, so with many Q tiles the lost cp.async pipelining costs more than the single conversion pass.

Crossover, with a temporary env switch between the two modes and new bench rows of q 16 to 1024 over a 4096 cache
(fattn us, converted pass / in kernel):

| row | Qwen3.5-0.8B | Llama-8B |
|---|---|---|
| q 16 kv 4096 | 97.7 / 64.0 | 94.5 / 45.7 |
| q 64 kv 4096 | 107.3 / 79.1 | 135.2 / 75.7 |
| q 256 kv 4096 | 185.3 / 157.1 | 304.3 / 278.8 |
| q 512 kv 4096 | 316.1 / 297.8 | 554.5 / 550.2 |
| q 1024 kv 4096 | 574.2 / 535.2 | 867.7 / 952.6 |
| 2 x 512 x 512 | 98.3 / 84.5 | 139.6 / 140.8 |
| 2 x 2048 x 2048 | 698.2 / 784.5 | 1409.5 / 1555.0 |
| 1 x 8192 x 8192 | 3673.2 / 4214.5 | 7298.9 / 8096.5 |

The in-kernel path wins or ties up to 512 Q rows per sequence and loses beyond. The mma launcher now uses it for
bf16 K/V when `Q->ne[1] <= FATTN_MMA_KV_BF16_MAX_Q` (512), and converts first otherwise.

Bench with the rule (us, fattn / FA2):

| row | Qwen3.5-0.8B | Llama-8B |
|---|---|---|
| q 1 kv 512 | 11.4 / 44.8 | 11.2 / 46.8 |
| q 1 kv 4096 | 23.1 / 317.1 | 31.9 / 333.4 |
| q 1 kv 16384 | 52.3 / 1212.9 (was 154.9) | 87.7 / 1260.9 (was 303.7) |
| q 16 kv 4096 | 68.0 / 345.0 | 45.4 / 163.3 |
| q 256 kv 4096 | 171.2 / 347.6 | 278.7 / 306.3 |
| 1 x 512 x 512 | 107.2 / 48.8 | 92.6 / 65.7 |
| 2 x 512 x 512 | 84.1 / 79.2 | 140.2 / 101.8 |
| 1 x 2048 x 2048 | 356.8 / 364.0 | 555.4 / 604.5 |
| 2 x 2048 x 2048 | 688.0 / 677.7 | 1426.1 / 1163.5 |
| 1 x 8192 x 8192 | 3694.3 / 4548.4 | 7267.8 / 8180.0 |

Tests: the 13 parity tests pass. Their bf16 cases cover both modes: decode, chunked prefill and MLA run in the
kernel, and `stream_k_fixups` at 1024 rows converts first.

Remaining gaps vs FA2: Qwen 1 x 512 (the general stream-k fixup, Run 4), Llama 512 prefill (1.4x), and Llama
2 x 2048 (1.23x). All three are kernel-side.

Next: paged K/V (block tables), which also needs bf16 read in place, so the paged cache never takes a conversion
pass.

Review follow-up:
- f16 K/V cost. The sync path is now compiled into every cp.async instance behind a runtime branch, and the bench
  only ran bf16, so it gained `FATTN_BENCH_F16`. f16, master kernels vs this branch, three runs each (us):

  | Qwen3.5-0.8B row | master | branch |
  |---|---|---|
  | q 1024 kv 4096 | 568.7 / 531.4 / 530.3 | 581.7 / 539.4 / 542.9 |
  | 2 x 512 x 512 | 146.3 / 138.5 / 139.1 | 147.2 / 140.3 / 142.5 |
  | 1 x 2048 x 2048 | 451.2 / 416.3 / 416.2 | 459.3 / 429.7 / 428.1 |
  | 2 x 2048 x 2048 | 842.9 / 851.4 / 849.7 | 866.5 / 872.1 / 873.4 |

  Register counts are unchanged: the 256/256 instance has 255 registers and 40 B of stack on both; 128/128 has 220
  on master and 213 here. Keeping the original direct 16-byte copy for f16 (the conversion goes through a register
  chunk only when bf16) did not remove it. That leaves about 2-3% on f16 head-dim-256 prefill, likely code size in
  an already huge instance. Llama (128) rows are flat. Accepted, since bf16 is the main path.
- The same f16 run shows f16 Q is slower than bf16 overall (Qwen 2 x 512: 139 us f16 vs 84 us bf16). f16 Q still
  goes through candle casts to f32 and back. Native f16 Q/out can be added the same way as bf16.
- K/V are read in place with 16-byte loads, and bf16 K/V used to get an aligned copy first. `validate` now rejects K/V
  whose offset or strides are not 16-byte aligned, rather than letting them fault.
- A new test, `long_prefill_converts_bf16_kv_first` (600 and 640 Q rows at head dims 128 and 256), covers the
  convert-first mode beyond `stream_k_fixups`.

## Run 6 - 2026-10-04 20:23

Question: can fattn read a paged K/V cache in place, and at what cost against dense K/V?

Cache layouts in use (survey of `inference-nn/src/paged_attention/`):
- Standard (vLLM): K is `[blocks, kv_heads, head_dim/x, block_size, x]` and V is `[blocks, kv_heads, head_dim,
  block_size]`. A token's row is not contiguous there, so fattn's row loads cannot read it.
- FlashInferHnd: `[blocks, kv_heads, block_size, head_dim]`. Rows are contiguous and each head's block is
  contiguous. FA2-paged already reads this layout, through a strided NHD view. This is the layout fattn targets.
- Block size defaults to 32 (`--pa-block-size`). FlashInfer, FA2-paged and FA3 need `% 32`.

Design:
- `fattn_paged_kv {block_table, seq_lens, max_blocks, block_size_log2, block_stride_K/V}` goes to every kernel by
  value. entry.cu passes it to the vendored dispatch through `op_params[6..8)`, so no new globals.
- Kernel selection sends paged calls to the mma kernel only; the vec and tile kernels walk K/V with fixed strides.
- The mma tile loader resolves row `i` of sequence `s` to `table[s][i >> log2] * block_stride + (i & mask) * row_stride`.
- Rows past `seq_lens[s]` (the tail of the last block, and the K/V length padded to 256) read the sequence's first
  row instead. Unused cache slots may hold anything, including NaN, and `0 * NaN` in P.V would poison the output.
  The mask (`(b, seq_q, paged_kv_len)`, now required) hides those rows.
- The K/V length is `max_blocks * block_size` rounded up to 256 (`FATTN_KQ_STRIDE`). That lets the GQA-batched kernel
  and mask-driven tile skipping apply.
- bf16 caches always convert in the tile loads; a paged cache cannot take an f16 copy of the whole pool.
- Rust: `flash_attn_paged(q, &PagedKv {k_cache, v_cache, block_table, seq_lens}, opts)`, plus `paged_kv_len` and
  `paged_causal_mask`. Tables and lengths are u32, and the block size must be a power of two.

Tests (`tests/integration/paged.rs`, 4 tests): f16 and bf16 against the dense reference per sequence. Block tables
are scattered (an LCG over 48 blocks), lengths are uneven (100/333/37, 50/129, 64/16/200), and every unused cache slot
holds NaN. Coverage: decode at head dims 64/128/256, a 17-row chunk over a cached prefix, block size 16, no GQA, and
GQA 2. Sanity check: with the first-row redirect removed, 3 of the 4 tests fail. One more test rejects a missing or
short mask and a non-power-of-two block size.

Bench (`bench::paged_vs_dense`): the same data, as a block-32 paged cache with interleaved tables and as dense K/V.
- First version: paged was 1.8-2.8x slower. nsys showed the same kernel and grid as dense, with the main kernel at
  222 us vs 91 us at Llama kv 16384, so the paged loads were latency bound. Hoisting the row lookup out of the
  16-byte chunk loop changed nothing.
- Reading the block table with `__ldg` (read-only cache, so the compiler may move the lookups above the tile's
  shared-memory stores) and using shift/mask for the block split fixed it:

| row | Qwen3.5-0.8B paged / dense (us) | Llama-8B paged / dense (us) |
|---|---|---|
| b 1 q 1 kv 4096 | 25.2 / 25.0 | 34.8 / 34.5 |
| b 1 q 1 kv 16384 | 58.6 / 58.1 | 93.9 / 93.8 |
| b 8 q 1 kv 4096 | 98.4 / 96.2 | 195.1 / 192.1 |
| b 8 q 1 kv 16384 | 348.2 / 347.2 | 752.1 / 734.2 |
| b 1 q 256 kv 4096 | 184.0 / 182.1 | 303.0 / 295.4 |
| b 1 q 1024 kv 4096 | 640.3 / 578.7 | 1035.2 / 974.6 |

The 1024 row is the only gap. There dense converts bf16 first and loads with cp.async; paged cannot.

Not wired in yet. Next: FP8 (e4m3) caches with per-layer scales, and varlen (packed) Q with per-sequence Q
lengths. Both are needed before the paged decode and prefix-prefill paths can move to fattn.

Review follow-up:
- Untrusted lengths: a `seq_len` beyond the table's span would have read past the sequence's table row. The kernel
  now clamps it to `max_blocks << log2`. A zero-width table would have divided by zero on the host (stream-k with 0
  KV tiles) and is now rejected. `seq_lens >= 1` and in-range table entries are documented requirements, since the
  device-side values cannot be checked on the host.
- Tile skipping: the mask scan that feeds `KV_max` only runs for batch > 1 or 1024+ Q rows. So a batch-1 call with an
  over-allocated table iterated the whole span (correct, as every row past the length rereads row 0, but wasted
  work). Forcing the scan for paged calls cost 3-8 us on exactly-sized batch-1 decode, so it was dropped. Instead the
  mma kernel ends each sequence at `ceil(len / nbatch_fa)` KV tiles: exact, with no extra kernel. Bench afterwards,
  paged / dense: 1.00-1.05 on every row.
- New tests: batch 1 with 20 spare table entries, so whole KV tiles lie past the sequence and stream-k blocks start
  beyond it; and a 600-row bf16 prefill over 700 rows, which converts in the loads where dense would convert first.
- Note on the bench's batch-1 decode rows: on Ada and newer, dense would choose the vec kernel there, while paged is
  always mma. On this Ampere card both take mma.

## Run 7 - 2026-10-04 21:36

Question: can fattn read fp8 (e4m3) K/V caches, which `--pa-cache-type f8e4m3` produces with per-layer scalar K/V
scales, in place?

Design:
- `fattn_paged_kv` became `fattn_kv_src`: where K/V live (paged or dense) plus how they are stored (`fp8`, `k_scale`,
  `v_scale`). fp8 tensors reach ggml as `GGML_TYPE_I8`, a one-byte type, so the strides and element-size asserts hold.
- Kernel selection sends fp8 to the mma kernel; launch never makes an f16 copy of fp8 K/V.
- The mma tile loader dequantizes `x * scale` into f16 shared memory. One-byte elements break the half2 pointer
  arithmetic the call sites used for column offsets (`K_h2 + k0_start`), so the loader now takes the column offset
  explicitly and addresses fp8 rows in bytes.
- Rust: `FattnOptions::kv_scales: Option<KvScales {k, v}>`, valid only with F8E4M3 K/V (an error otherwise). fp8 is
  rejected at head dim 576, where V is read out of K's tiles, which carry the K scale.

Tests: a new `fp8` module (dense decode and prefill at head dims 64/128/256, plus 600 Q rows) and an fp8 mode in
every paged case. Scales are k 0.05 and v 0.2, and the reference uses the exactly dequantized values, at tolerance
1e-2 with bf16 Q. Sanity check: swapping the K and V scales on the Rust side fails both fp8 suites. 22 tests pass.

Three performance findings, in the order they surfaced:

1. A runtime fp8 branch per 16-byte chunk inside the unrolled sync-load loop halved bf16 decode, even when not taken.
   Qwen q1 kv16384 went from 58 to 114 us, and Llama from 96 to 206 us. Register counts were unchanged, and moving
   the fp8 code out of line (`__noinline__`) did not help. Fix: fp8 is a separate instantiation of the loop body,
   chosen once per tile by a `std::integral_constant` tag (`ggml_cuda_unroll` forwards extra arguments).

2. That fixed decode, but bf16 prefill up to 512 Q rows was still about 1.7x slower than master (Llama q256 kv4096: 557
   vs 326 us). Bisecting by experiment:
   - with the fp8 instantiation removed: still slow;
   - with every fp8 reference removed from the loader: fast.
   The remaining culprit was the cp.async gate `if (!KV_bf16 && !rows.fp8)`. Many prefill instances gained 15-20
   registers with it (for example 128/128 8x4: 213 -> 233, 16x4: 229 -> 245). Fix: fp8 folds into the existing
   per-call flag, renamed `KV_convert` (the loads convert: bf16 or fp8), so the gate is one bool again. bf16 rows
   then matched master on every row (Llama q256 kv4096 330.8 us).

3. fp8 itself was 2.4-5.4x slower than bf16. On sm86, `__nv_cvt_fp8x2_to_halfraw2` has no hardware instruction (it
   arrives with sm89), so it is emulated. Replaced by exact bit placement: `__byte_perm` spreads 4 bytes into two
   half2 lanes, the sign goes to bit 15 and the exponent and mantissa shift up by 7, then a multiply by 2^8 covers
   the bias gap (15 - 7), exactly for normals and subnormals. The scale then multiplies in half2. That puts a 2^-11
   relative rounding on the scale, small next to fp8's own 2^-4 step. e4m3 NaN bytes are not preserved, which only
   matters for unwritten slots, and those are redirected or masked.

fp8 / bf16 after (us):

| row | Qwen3.5-0.8B | Llama-8B |
|---|---|---|
| b 1 q 1 kv 4096 | 21.7 / 25.4 | 42.1 / 35.4 |
| b 1 q 1 kv 16384 | 69.8 / 57.9 | 112.1 / 91.7 |
| b 8 q 1 kv 16384 | 449.7 / 346.3 | 1071.2 / 767.4 |
| b 1 q 256 kv 4096 | 208.3 / 174.3 | 382.6 / 296.3 |
| b 1 q 1024 kv 4096 | 795.6 / 633.7 | 1257.1 / 959.8 |

fp8 reads half the bytes and is still 1.2-1.4x slower. Llama q1 kv16384 is about 300 GB/s against bf16's ~700. Each
thread loads 8 bytes per 16-byte f16 chunk, which leaves the fp8 path latency-bound. Follow-up: load 16 fp8 bytes (two
chunks) per thread. For now, fp8 here buys memory capacity, not speed.

Next: varlen (packed) Q. Then wiring: CUDA paged caches move to the HND layout, decode and prefix-prefill move to
fattn behind parity tests, and FA2, FlashInfer and vLLM v1/v2 are deleted.

Review follow-up:
- Bug: `supported()` built its args without the K/V source, so fp8 K (described as `I8`) failed the K/V type check
  and every fp8 call answered false; `flash_attn` itself was fine. Both now share `dense_kv_src`, and every fp8 test
  asserts `supported()`.
- Coverage: for head dims 64-256 the tile loads take each row in one slice, so the new explicit column offset was
  always 0. Head dim 512 loads K and V in column slices, and a 512 case now covers the fp8 byte offset of a slice.
  112 adds the narrower load widths. Further new cases: f32 Q, softcap with sinks, and `every_code_decodes`. In that
  test every V row holds all 254 non-NaN e4m3 codes, so the output must equal the decoded row whatever the weights,
  which pins down +-0, the subnormals and +-448.
- The 2^8 bias gap now folds into the scale before it rounds to f16: one multiply per element instead of two, and a
  small scale keeps its precision (on its own it would round to an f16 subnormal below 6.1e-5).
- Scales must be finite and in (0, 146]: 448 * 146 is just under f16's 65504.
- C-side assert: no fp8 at head dim 576.

## Run 8 - 2026-10-04 22:36

Question: can fattn take varlen (packed) batches, the `[total_tokens, heads, dim]` + `cu_seqlens` form that packed
prefill hands FA2 today, without padding Q to the longest sequence?

Design:
- `fattn_kv_src` became `fattn_layout`, gaining `cu_q` (Q/dst starts per sequence) and `cu_kv` (dense K/V starts).
- Q and dst are described as `[d, max_q, h, n_seq]` with a batch stride of 0. The mma driver puts each sequence's Q and
  dst at `cu_q[s]`, bounds its tile loads and stores by that sequence's length (`process_tile` takes `q_len`), and
  skips Q tiles past it outright. Both stream-k fixup kernels take `cu_q` for their dst offsets and bounds.
- Dense packed K/V rows read from `cu_kv[s] + i`. Rows past the sequence's length read its first row, which keeps
  the last sequence from reading past the buffer, and the mask hides them. Each sequence ends at its own last KV
  tile. The K/V length is padded to 256 (`varlen_kv_len`), as for paged.
- Varlen calls take the mma kernel and always convert bf16 in the loads, because the f16 copy pass would size its
  copy from the padded descriptor.
- Rust: `Packed { cu_seqlens, max_len }`; `flash_attn_varlen` (packed Q over packed dense K/V);
  `flash_attn_paged_varlen` (packed Q over a paged cache); `varlen_kv_len`; `varlen_causal_mask`, which
  `paged_causal_mask` now delegates to. The mask is `(b, max_q, n_kv)` and is required. cu tensors are u32, and their
  device-side values are documented, not checked.

Tests (`tests/integration/varlen.rs`, 3 tests): f16 and bf16, each sequence against the dense reference. Q lengths are
5/37/1/64 over K/V lengths 5/100/40/64 (a chunk over a cached prefix included), at head dims 64 (GQA 4), 128 and
256 (GQA 2), plus one long sequence beside short ones (300/3/17 over 300/3/600) without GQA. Packed K/V are the head
of a buffer whose tail is NaN, so a read past the last sequence fails. The paged variant uses reversed blocks and
NaN padding. A rejection test covers a missing or wrongly sized mask, a batch mismatch and max_len past the rows.
28 tests in the crate pass.

Performance: what it took to stay level with master.
- First varlen build vs master. bf16 bench, interleaved master/branch runs, median of 3 (single runs drift 5-7% with
  GPU clocks: FA2's own times move that much): Qwen head-dim-256 prefill past 512 Q rows was 9-10% slower (1 x 8192:
  3878 vs 4235 us), and 128 was within 3%.
- The same A/B showed master itself 6-12% slower on long prefill than Run 5 (Qwen 1 x 8192 3924 vs 3694 us, Llama
  b2 x 2048 1530 vs 1426 us). So #275/#276 had cost bf16 too; Run 7's "matches master" had checked selected rows only.
- Cause, part 1: the per-sequence row structs (`fattn_kv_rows` for K and V, about 8 registers each) passed by value
  through process_tile and iter. 256/16/4 had 255 registers and 152 bytes of stack. They became a reference to the
  kernel's own layout parameter plus the sequence index, with `load_tile` resolving the rows itself. Stack fell to
  56 bytes, but Qwen long prefill was still slower.
- Cause, part 2: the general row lookup (paged table or packed base, length redirect) in loops that plain dense K/V
  run. Both load loops are now instantiated for plain dense rows (`i * stride`, master's code) and for general rows,
  chosen once per tile like fp8.
- Tooling: dev test binaries load the fattn kernels from an absolute-path shared library. The A/B swapped that file
  between runs at first, then used patchelf to point each binary at its own copy.

Final A/B, median of 5 interleaved runs, master / branch (us):

| row | Qwen3.5-0.8B | Llama-8B |
|---|---|---|
| q 16 kv 4096 | 72.3 / 69.6 | 52.6 / 51.1 |
| q 256 kv 4096 | 173.5 / 170.1 | 300.2 / 282.8 |
| q 512 kv 4096 | 320.8 / 313.1 | 602.1 / 565.3 |
| q 1024 kv 4096 | 541.5 / 572.0 | 950.5 / 882.5 |
| 2 x 512 x 512 | 94.7 / 93.5 | 151.4 / 145.6 |
| 1 x 2048 x 2048 | 387.5 / 389.0 | 610.1 / 573.2 |
| 2 x 2048 x 2048 | 755.7 / 755.9 | 1537.3 / 1439.1 |
| 1 x 8192 x 8192 | 3928.4 / 3945.7 | 7896.1 / 7365.1 |
| q 1 kv 512 | 12.5 / 11.2 | 12.4 / 11.7 |
| q 1 kv 4096 | 24.4 / 27.0 | 35.3 / 37.0 |
| q 1 kv 16384 | 56.3 / 57.3 | 94.2 / 92.0 |

Llama recovers the #275/#276 cost: 3-7% faster than master on every prefill row, and 1 x 8192 is back near Run 5's
7268 us. Qwen is level except q 1024 kv 4096 (+5.6%) and decode at kv 4096 (+10.7%, Llama +4.8%). Spreads are
tight, so these are real, but kv 512 is 10% faster and kv 16384 flat, which points at layout effects in specific grid
configurations rather than per-row cost. Recorded, not chased. Register and stack counts turned out to be a rough
proxy only: 256/16/4 ends at 96 bytes of stack and still benches level with master's 40.

Next: wiring. CUDA paged caches move to the HND layout, and the paged decode, prefix-prefill and packed-prefill paths
move to fattn behind parity tests. Then FA2, FlashInfer decode and vLLM paged v1/v2 are deleted.

Review follow-up:
- Inherited from upstream: the `KV_max` mask scan (`flash_attn_mask_to_KV_max`) read mask rows up to `ncols1 - 1`
  past the mask on the last Q tile when `max_q % ncols1 != 0`. That is harmless to the result (extra rows can only
  raise KV_max), but it is an out-of-bounds read, and varlen always runs the scan (batch > 1, K/V padded to 256).
  The scan now takes the mask's row count and stops there.
- A dense packed sequence with 0 K/V rows would redirect to row `cu_kv[s]`, one past the buffer for the last
  sequence. `Packed` now documents at least one K/V row per sequence. Packed totals above `i32::MAX` are rejected,
  since the kernels read `cu_seqlens` and row indices as i32.
- Tests: the last K/V length was 64, which fills a 64-row KV tile, so no tail row was read and the NaN-tail check
  only caught overreads in one case. It is now 70, leaving a partial tile. New cases cover fp8 varlen, softcap with
  sinks, batch 1, and a `max_len` 20 rows past the true maximum (with padded mask rows). The crate has 29 tests.
- Not done yet, needed for the wiring: a `supported()` probe for varlen and paged calls, so a caller can route by
  capability (pre-Turing, head dims 40/72, 192 without GQA) instead of by error.

## Run 9 - 2026-10-04 23:22

Question: how does the engine mask fattn calls during decode and chunked or packed prefill? Paged and varlen calls
required an f16 mask tensor, and building one per step on the host means a host-to-device copy per layer. CLAUDE.md
rules that out in hot loops, and a captured CUDA graph cannot contain it.

Design: an implicit mask, built by the kernel from the sequence lengths.
- `fattn_layout` gains `implicit_mask`, `causal` and `window_left`. Key `kp` is visible to the query at
  `qp = kv_len - q_len + row` iff `kp < kv_len`, and, when causal, `kp <= qp`, and, with a window,
  `qp - kp <= window_left`. That is the FA2/FlashInfer window convention.
- `flash_attn_ext_f16_make_mask` writes each KV tile's mask into shared memory where `load_mask` would have loaded
  one, at all three load sites. A placeholder mask descriptor (never read) keeps the GQA-batched kernels selectable,
  and the host skips the KV_max mask scan.
- Causal Q tiles end at their last visible KV tile, which replaces the scan's tile skipping and applies at every
  length, not only batch > 1 or q >= 1024.
- Implicit masks select the mma kernel. Found by the tests: a batched causal decode first chose the vec kernel,
  which read the placeholder pointer (`CUDA_ERROR_ILLEGAL_ADDRESS` in `decode_vec`).
- Rust: `FattnOptions::{causal, window_left}`. Paged and varlen calls without a mask use the implicit mask; a
  batched call uses it with `causal: true`. A mask together with `causal` is an error, a window needs `causal`, and
  causal needs `seq_q <= seq_kv`.

Tests: every causal case of the dense, paged and varlen suites also runs without the mask tensor, against the same
reference. That covers MLA, sinks, softcap, fp8, packed and paged layouts. `implicit_sliding_window` checks windows
of 40, 100 and 7 against a reference built over a window mask tensor. The crate has 31 tests.

Bench, interleaved A/B, median of 5. Master uses an explicit causal mask tensor; the branch uses `causal: true` (us):

| row | Qwen3.5-0.8B master / branch | Llama-8B master / branch |
|---|---|---|
| 1 x 512 x 512 | 114.7 / 110.3 | 95.3 / 76.0 (-20%) |
| 2 x 512 x 512 | 98.9 / 85.7 (-13%) | 148.9 / 122.2 (-18%) |
| 1 x 2048 x 2048 | 420.4 / 360.9 (-14%) | 602.5 / 559.3 (-7%) |
| 2 x 2048 x 2048 | 780.0 / 593.9 (-24%) | 1459.5 / 1075.6 (-26%) |
| 1 x 8192 x 8192 | 3988.0 / 3783.2 (-5%) | 7389.5 / 7290.2 |
| q 16-256 over kv 4096 | +1-5% | +2-3% |
| q 1024 kv 4096 | -3.5% | +1% |
| decode kv 512 / 4096 / 16384 | +2 / +5 / +2% | -4 / -3 / -1% |

- Where tile skipping applies (causal prefill), the gains are large.
- Where it does not (a short chunk over a long cache, decode), building the mask costs up to 5% against
  cp.async-loading a tensor.
- Against FA2, fattn now matches or beats it everywhere except Qwen 1 x 512 (2.24x, Run 4's general stream-k fixup)
  and Llama 512 prefill (1.15-1.21x).

Review follow-up:
- `causal: true` sent every dense call to mma, even a single query with no window, where causal masks nothing.
  That cost decode its vec kernel (the likely source of the +2-5% Qwen decode above), and the `decode_vec` test's
  implicit run no longer reached vec. Such calls now drop the mask entirely, except at the head dims fattn runs only
  GQA-batched (192/320/512/576), which need a mask, real or implicit, to be selected at all. The first version of
  this rule missed that exception, and the 512 and 576 tests caught it.
- Documented, not checked on the device: with `causal`, each paged or varlen sequence needs at least as many keys as
  queries, since a sequence with no visible key comes out NaN. Tiles wholly before a sliding window are still
  computed, which needs a lower clamp in stream-k.
- The two copies of the tile clamp are one helper, `fattn_kv_tiles_visible`. The redundant `min` is gone, and stale
  docs (the mask doc, `PROBE_PTR`, varlen/paged "the mask is") are updated.
- New tests: a sliding window over packed dense K/V, over a paged cache and with fp8, and a bidirectional packed
  encoder (no causality, mask from lengths only). The crate has 32 tests.

## Run 10 - 2026-10-05 02:11

Question: can fattn take over paged decode on the FlashInfer-layout (HND) cache, the first engine path to move, at
parity with FlashInfer decode in correctness and throughput, including under CUDA graphs?

Wiring:
- inference-nn depends on inference-fattn under `cuda`. `PagedAttention::try_run_fattn_decode` runs between the FA3
  attempt and FlashInfer in `run_flashinfer_decode`. It reads the padded block tables and context lens the layer
  already selects (`ctx.block_tables`, the windowed view for a sliding layer, the full one otherwise), with
  `causal: true`, `window_left = window - 1`, softcap and fp8 scales. Multi-token decode (MTP verify) has one table
  row per query row, so each row is its own sequence of one query.
- Decode metadata built only FlashInfer's CSR form for this layout; `conservative()` now always builds the padded
  tables too. Graph capture keeps them (they exist at capture) and drops the tile plan when nothing used it.
- `inference_fattn::supported_paged`: false for kernel limits (non-power-of-two block size, MLA head dim, fp8 scales
  outside (0, 146]), an error for malformed operands. The op's descriptor building is now `Fattn::describe`, shared
  by the launch and the probe, so the probe cannot drift from the call it predicts; `supported()` uses it too.
- Graph capture: fattn's pool allocations are `cudaMallocAsync` on the caller's stream, which relaxed-mode capture
  records as graph alloc nodes. Qwen3.5-0.8B captures and replays (`Captured 1 CUDA decode graphs`), and the bench
  ran with graphs on throughout.

Found by the new probe test: softcap at head dim 64 passed `supported()` but trapped (`NO_DEVICE_CODE` at
fattn-mma-f16.cuh:1946). Every kernel skips its softcap variants outside head dims 128/256/512 ("Skip unused kernel
variants"), and selection never checked. Selection now returns none there, for dense calls too.

Bench 1, engine, `inference bench -m Qwen3.5-0.8B --prompt-len 0 --gen-len 128`, CUDA graphs on (TPOT ms):

| depth | master (FlashInfer) | fattn, first cut |
|---|---|---|
| 4 | 2.54 | 2.54 |
| 4096 | 2.64 | 2.67 |
| 16384 | 2.77 | 2.98 |
| 16000 | 2.76 | 2.83 |

nsys at 16384: FlashInfer decode 46.6 us + merge 5.1 us per call (grid 256, 60 regs); fattn mma 84 us + uniform
fixup 4.7 us (grid 82, 235 regs). The graph's block table spans a power-of-two bucket (16512 tokens -> 32768), and
stream-k split the padded length, so about half the blocks got only tiles past the sequence. Depth 16000 fits the
16384 bucket and lost only 2.5%.

Fix: `fattn_iter_k`. With lengths on the device (paged or packed), the mma split covers only the longest sequence's
KV tiles (a warp max over `seq_lens` or `cu_kv`). Both fixups follow it:
- First the general fixup for every such call (it already handles blocks with no data): the main kernel fell to
  57-58 us, but the general fixup took 21.9 us (a serial walk back over ~40 blocks per tile).
- Then the uniform fixup with a `continue` for blocks without data: 11 us. The branch stopped the loop's loads from
  batching. Selects instead of the branch: 5.5 us.
- Then only the blocks that hold data. With fewer KV tiles than blocks, KV tile t falls in block
  `ceil((t + 1) * bpt / iter_k) - 1` (checked exhaustively against the kernel's split for bpt < 200, iter_k < 400),
  and a tile one block covered alone is left as written. The data-block choice sits outside the loop, so the dense
  loop is upstream's.
- The general fixup keeps upstream's fastdiv, with its divisors recomputed on the device (`fattn_fastdiv_values`)
  only when the lengths are there. A plain-division version had cost Qwen 1 x 512 x 512 21%.
- New test `graph_padded_tables_split_by_live_length` (120-250 spare table entries). It fails when the uniform fixup
  is used without the skip, and three varlen tests fail when the general fixup ignores the device split.

Dead end, the gap at long context: f16 K/V through cp.async is 15-17% faster than bf16 through the synchronous
converting loads at 16k decode (32.6 vs 38.1 us, isolated). Tried: bf16 tiles copied as stored by cp.async, with
fragments converted after each `ldmatrix` (a runtime `fattn_layout` flag, no new instances). Interleaved A/B
against master (patchelf'd binaries, min of 5, one test thread): batched paged decode -17% to -21% (b 8). But
single decode at 2-4k was +7-15%, prefill at or under 512 rows +5-28%, and rows over 512, which never take the path,
+4-13%. The runtime branch in the mma inner loop costs every instance, as in Run 7. A compile-time flag would double
the mma instances. Reverted. (The first two A/B runs here were void: one libtest process ran both benches on parallel
threads on one GPU. `--test-threads=1` from now on.)

Cost of the device split: the length-derived `iter_k` cannot be rematerialized from kernel parameters, so 190 of
1419 kernels gain 4-10 registers (some 255-register instances spill 16 bytes more). A/B against master over the
fattn benches, min of 5: prefill within +-4% except Llama q 16 (+6%); dense decode with a mask tensor (not an engine
path) +6-10% at 2-4k. Without the device split (a build where `fattn_iter_k` returns the host value), engine decode
at 16384 is +8.3% against master, and +3.3% with it, so it stays.

Bench 2, engine, final build, min of 3 interleaved rounds (TPOT ms):

| depth | master (FlashInfer) | fattn | fattn, host split |
|---|---|---|---|
| 4 | 2.51 | 2.55 (+1.6%) | 2.55 |
| 4096 | 2.64 | 2.63 (-0.4%) | 2.69 |
| 16000 | 2.73 | 2.83 (+3.7%) | 2.83 |
| 16384 | 2.76 | 2.85 (+3.3%) | 2.99 |

- fp8 e4m3 cache (`--pa-cache-type f8e4m3`): master 3.45 / 5.45 ms at d4 / d16000, fattn 2.55 / 2.86 (-26% / -48%).
  FlashInfer's fp8 decode is slow on sm86; fattn's bit-placement decode (Run 7) is not.
- Graphs off (eager, CPU-bound): fattn +2%. The eager metadata now also uploads the padded tables, 2 more pageable
  host-to-device copies per step at ~65 us each. FlashInfer's 8 CSR and plan tensors go when FlashInfer does.
- What is left at 16k is the kernel: the mma decode tile (sync bf16 loads, one 4-warp block per SM at head dim 256)
  streams about 590 GB/s against FlashInfer's ~720. A single tile also costs ~13 us of latency in the engine against
  FlashInfer's 8.7 at short contexts.

Correctness:
- `fattn_decode_tests` (inference-nn, GPU) runs the layer's decode on real decode metadata
  (`DecodePagedRows::build_materialized`) and compares with `flashinfer_decode` on the same plan. Head dims
  64/128/256/512 with bf16, f16 and fp8 caches, and windowed models with block sizes 16 and 32, for a sliding layer
  (with and without softcap) and a full one. It also checks that FlashInfer's plan went unused, and that softcap at
  64 falls back to FlashInfer. Max abs diff is 1-2 bf16 ulps (0.0039-0.0078); halving the window gives 0.705.
- Real checkpoint: master's and the branch's plain greedy traces (Qwen3.5-0.8B Q8_0 GGUF, two prompts, 25-40
  steps) agree on every id. Logprobs move by 0.002-0.036 on average, 0.1 at most. That is fattn's f16 P*V
  accumulators (`T_C_VKQ = tile<16, 4, half2>` on Ampere) against FlashInfer's f32; the engine's MTP and plain
  paths already differ by up to ~0.12.
- `gguf_builtin_mtp_accepts_drafts_and_keeps_greedy_output` failed: at step 8 of the nursery-rhyme prompt the MTP
  trace's top two tie exactly (bf16 logits step logprob gaps by 0.125) and it took the plain trace's runner-up.
  The test only accepted a tie on the plain side and assumed none before step 16. On master the same prompt has
  near ties (0.125 < `TIE_MARGIN` 0.25) at steps 5 and 8. The check is now symmetric, with `MIN_AGREED` 5.

Next: packed and chunked prefill onto fattn (`supported_varlen`, `supported_paged_varlen` over the same `probe`),
then delete FlashInfer decode, which takes the CSR and tile-plan metadata with it. A decode-shaped kernel path that
keeps cp.async for bf16 (without a branch in the shared mma loop) is the open lead for the 16k gap.

Review and CI follow-up:
- CI: 10 tiny vision-language tests failed with `fattn takes f16, bf16 or fp8 e4m3 K and V of one dtype, got F32
  and F32`. Those run f32 caches, which FlashInfer takes, and `supported_paged` errored instead of answering false.
  The review found the same independently. K/V dtype, row alignment and the device ordinal are now limits
  (`operand_limit`, shared with `validate_operands`), so the layer falls back. New engine test
  `f32_caches_fall_back_to_flashinfer`.
- The review found `fattn_iter_k`'s full-mask warp shuffle undefined in a fixup at head dim 80 or 112, whose last
  warp is half full (block = DV threads). Not reachable from the engine (FlashInfer layers are 64/128/256/512), but
  reachable through the API. It now reduces over 16-lane groups under `__activemask()`.
  `graph_padded_tables_split_by_live_length` covers 80 and 112.
- New engine test `multi_token_decode_matches_flashinfer` (3 query rows per sequence, one table row each, as MTP
  verify builds them).
- `MIN_AGREED` is per prompt: 16 for counting, which never parts before it, and 5 for the rhyme.
- Not covered: on SM90 the FA3 fp8 decode runs before fattn, and the decode test's "plan unused" check cannot tell
  the two apart. This machine is sm86.

## Run 11 - 2026-10-05 07:15

Question: can fattn be the CUDA flash backend for `Sdpa` (prompt prefill, packed prefill, the gather-SDPA prefix
path, encoders, embedding models) in every CUDA build, with FA2 only behind it?

Design:
- `using_flash_attn()` is true with `cuda`. Builds without `flash-attn` used to run CUDA prefill eagerly: explicit
  causal masks, no packed prefill.
- `flash_backend_supports*` are fattn's static capabilities joined with FA2/FA3's:
  - fattn takes head dims 64/80/96/112/128/256 in every layout, with softcap only at 128/256;
  - 192/320/512/576 run only GQA-batched, so they are not claimed.
- `backends::flash_attn` now returns `Option`. Order: FA3 for its head dims (when built), then fattn
  (`try_fattn`), then FA2 (when built).
- `try_fattn` mirrors FA2's split. Varlen (`supported_varlen`/`flash_attn_varlen` over the `FlashParams` cu_seqlens)
  when the batch, the lengths or packing call for it, else dense. It uses FA2's causal default (`seq_len > 1`) and
  falls back for bidirectional windows.
- When no kernel takes a call, `run_attention` goes eager with the same mask the unsupported-head-dim branch builds.
  A packed call has no eager form, so it is an error. `supported()` and the new `supported_varlen()` answer false
  for operand limits (f32 K/V, alignment, fp8 scales).

Prefill profile first, at the weak row from Run 9 (Qwen 1 x 512 x 512, fattn 107.6 us against FA2 54.2). The mma
kernel took 63.9 us and `flash_attn_stream_k_fixup_general` took 40.9 us, serial after it. The grid had 164 blocks
over 64 output tiles. Rounding down to 128 would have lost 22% of the blocks, over upstream's 5% limit, so the
general fixup ran. Three attempts at the fixup itself made it worse:
- One CUDA block per stream-k block, looping over the 64 columns: 105 us. The columns' dependent chains ran
  serially.
- The contributing blocks found first, then their partials loaded in batches of 8: 67 us.
- One thread finds the chain once into shared memory: 100 us.
The cost is the many short-lived blocks (164 x 16 x 4 = 10,496 of 256 threads), each a short dependent chain,
across ~16 waves. Fix at the host instead: whenever rounding leaves more than one block per tile, round down
(uniform fixup). Qwen 1 x 512 x 512 fell to 72-79 us (mma 58.7 us with 128 blocks, uniform fixup 17.4 us). An A/B
against master (min of 5, one test thread) moved only that row (-37%); everything else was within +-3%.

Found by the suite: the tiny Qwen2-VL test pins greedy ids recorded on the eager path. On master built with
`flash-attn`, FA2 gives exactly the ids fattn gives, and both differ from eager (near-tied random logits). The test
now expects the flash ids under `cuda`. Two gemma tests asserted that softcap at head dim 128 needs FA2; fattn has
it.

Engine prompt latency (`inference bench --prompt-len ... --gen-len 0`, min of 2 interleaved rounds of 3, ms):

| model, prompt | FA2 build: master / branch | cuda-only build: master (eager) / branch (fattn) |
|---|---|---|
| Qwen3.5-0.8B 128 | 9.20 / 9.35 (+1.6%) | 9.08 / 9.41 (+3.6%) |
| Qwen3.5-0.8B 512 | 22.45 / 22.92 (+2.1%) | 24.52 / 22.97 (-6.3%) |
| Qwen3.5-0.8B 2048 | 84.68 / 84.91 | 106.68 / 86.20 (-19%) |
| Qwen3.5-0.8B 8192 | 356.83 / 356.91 | 660.02 / 361.41 (-45%) |
| Qwen2.5-Coder-3B Q4_K_M 128 | 32.86 / 34.02 (+3.5%) | - / 30.44 |
| Qwen2.5-Coder-3B 512 | 49.19 / 50.12 (+1.9%) | - / 51.38 |
| Qwen2.5-Coder-3B 2048 | 166.64 / 165.70 | - / 171.62 |
| Qwen2.5-Coder-3B 8192 | 762.54 / 759.33 | out of memory / 794.26 |

- Notes on the table:
  - The cuda-only eager runs of the coder model were lost to a script error except at 8192, where eager attention
    runs out of memory (it materializes the scores).
  - The cuda-only branch is ~4% behind the FA2-built branch at 2-8k, where both run fattn. FA2 builds still take
    `PrefixPrefillPlan::FlashAttentionPaged` (FA2 over the paged cache) for chunked prefill, and cuda-only builds
    gather.
- Short prompts lose 1.6-3.6% (fattn's fixed per-launch cost, as in decode, Run 10).

Next: the paged prefix/chunked prefill plan onto `flash_attn_paged_varlen`, replacing `FlashAttentionPaged`
(image prefix ranges need a decision there). Then FA2 can go.

Review follow-up. Most of these already existed in FA2 builds, but making the flash path the CUDA default would
have spread them to every CUDA build:
- A device-mapped CUDA model shares one `CausalFlash` mask across its layers. A layer on the CPU then ran the CPU
  flash kernel with no mask, attending to future tokens. f32 on CUDA (no flash kernel) lost causality on the
  cuBLASLt-less routes and the sliding window everywhere. Both now build the eager mask whenever the mask is
  `CausalFlash` or a window is set. New tests `causal_flash_masks_a_cpu_mapped_layer` (max abs diff 4.25 without
  the fix) and `causal_flash_masks_f32_on_cuda`.
- Before Turing, fattn runs causal, packed and paged calls on no kernel (mma only), so the static claim would have
  enabled packed prefill and then failed. `inference_fattn::mma_available()` (every visible device >= 7.5, cached)
  now gates `using_flash_attn()` and fattn's capabilities, so such builds keep their eager masks.
- Decode over gathered K/V (standard-layout cache with a window) passes `causal: false`, which fattn refuses with
  a window. One query per sequence sees the same keys either way, so `try_fattn` treats `seq_len == 1` as causal.
- Left as is: with head dims fattn doesn't claim (32/40/72/192), the eager fallback rebuilds its mask on the host
  each layer. That is FA2 builds' behaviour for their unsupported dims too, now reached in cuda-only builds.

## Run 12 - 2026-10-05 (afternoon)

Question: can prefix-cache and chunked prefill read the paged cache in place through fattn in every CUDA build?
Until now FA2 builds had `PrefixPrefillPlan::FlashAttentionPaged` and the rest gathered K/V into a workspace.

Design:
- New plan `FattnPaged`, after FA3's fp8 plan and before FA2's. It takes f16/bf16 activations over f16, bf16 or
  fp8 caches (FA2 needed the cache dtype to match), power-of-two blocks, the FlashInfer layout, fattn's head dims,
  and a window only when causal.
- Image prefix ranges (bidirectional spans inside causal attention), sinks, alibi and custom masks stay on the
  existing paths.
- `try_run_fattn_paged_prefill` takes the sequence lengths from `cu_kv` on the device (two narrows and a subtract,
  no host copy). One sequence per batch row of exactly `s` queries goes to `flash_attn_paged`; anything else is
  packed into one row and goes to `flash_attn_paged_varlen` with cu_q. fattn gained `supported_paged_varlen`. If a
  probe says no, the call falls back to the gather.
- First test run: "fattn needs contiguous u32 block tables ... got [2, 6] and [2]". The layer's
  `query_layout_is_dense` means padding-free, which a packed row is too, so the batched path got a packed query.
  The runner now tests for one sequence per batch row itself.
- Workspace planning already counts only the output for non-gather plans. The prompt workspace of the planner's
  test model (2 x 128 queries over 1k/8k cached rows, 16 heads, head dim 256) fell from 740 MB (padded gather) or
  39 MB (packed) to 4 MB. Six plan tests now expect `FattnPaged` and the output-only workspace when fattn can run
  (`fattn_reads_cache`, which asks fattn's own capability check).

Coverage: making the runner panic failed only the three tiny Qwen3.5 tests; the other tiny models' head dims
gather. So there are new direct GPU tests: `prefix_prefill_over_a_cached_prefix` (bf16, f16 and fp8 caches,
batched and packed queries over cached prefixes, head dims 128/256) and
`prefix_prefill_with_a_window_and_without_causality`. Both run on shuffled block tables against a reference over
rows gathered by hand. Halving the window gives a max abs diff of 1.99. One f16 case measured 0.00216, so the f16
tolerance is now 4e-3 (two ulps at O(1)).

Bench (`inference bench --prompt-len ... --gen-len 0`, min of 2 interleaved rounds of 3, ms):

| model, prompt | FA2 build: #280 / branch | cuda-only: #280 / branch |
|---|---|---|
| Qwen3.5-0.8B 2048 | 87.3 / 87.4 | 88.4 / 87.8 |
| Qwen3.5-0.8B 8192 | 372.0 / 361.5 (-2.8%) | 372.4 / 364.7 (-2.1%) |
| Qwen3.5-0.8B 16384 | 798.6 / 787.2 (-1.4%) | 813.3 / 792.6 (-2.6%) |
| Qwen2.5-Coder-3B 2048 | 178.1 / 178.1 | 181.8 / 185.7 |
| Qwen2.5-Coder-3B 8192 | 803.0 / 800.3 | 830.1 / 830.1 |

The coder model takes no prefix path here, and its cuda-only build was 2-4% behind with fattn on both. nsys, per
2048-token prefill: 108 extra `ucopy_bf16` kernels (5.4 ms) in the cuda-only build. `post_rope_output` made the
rope output contiguous unless FA2/FA3 were built, but fattn takes it strided. It now keys off
`using_flash_attn()`. After that, cuda-only 177.7 / 783.8 ms against the FA2 build's 175.6 / 785.1: FA2 no longer
buys anything on these paths.

Next: delete FA2 (`inference-flash-attn`, the `flash-attn` feature and `FlashAttentionPaged`). The dflash drafter
and image prefix ranges still call FA2 directly, so they need fattn paths or the gather first.

Review follow-up:
- Padded multi-sequence batches would have produced wrong output silently. Sequences with cache hits or a chunk
  offset run padded, not packed, and the plan never checked `query_layout_is_dense`. Such calls took the packed
  branch with cu_q offsets built from the padded length, so each shorter sequence's queries landed at the wrong end
  of its keys. `fattn_paged_prefill_supported` now requires a padding-free layout, and the runner's comment says
  why its batched/packed split is then exact. New plan test `fattn_reads_the_cache_only_without_padded_rows`.
- The planner reserves no gather workspace for `FattnPaged` layers, so a runtime refusal followed by a gather
  would fail the preflight limit (0 bytes).
  - The common trigger was a prefix hit with one new token per sequence on a sliding-window model. The runtime
    called that non-causal (no sequence has two queries), which fattn refuses with a window. The planner assumed
    causal. One query sees the same keys either way, so both plan input and runner now count single-query batches
    as causal.
  - A refusal now re-chooses without fattn (`choose_without_fattn`), so FA2 builds keep FA2's paged kernel rather
    than gathering.
  - Remaining refusals the static check cannot see: fp8 scales outside (0, 146], more than 16 devices.
- f16 activations: fattn computes f16 queries in f32 (an f32 Q copy and f32 output beside the f16 result), so the
  planner reserves 5 output-sized units for `FattnPaged` at f16 instead of 2.

## Run 13 - 2026-10-05 (evening)

Question: with every prefill path on fattn (Runs 11-12) and FA2 buying nothing on the measured paths, can FA2
(`inference-flash-attn`, Dao-AILab's v2) and the `flash-attn` feature go?

Remaining direct callers, and what replaced them:
- The DFlash drafter's windowed paged attention (`flash_attn_varlen_paged_windowed`). It now calls
  `flash_attn_paged_varlen` over the pool's HND caches (u32 tables and offsets already), with seq lens from the
  kv offsets on the device. Its non-causal layers used FA2's symmetric window (`right = window - 1`). Each draft
  block sits at the end of its keys, so only the left side ever binds. fattn now takes a window without causal
  masking (left-only; the kernel's mask already computed it, only the Rust check refused it), and
  `implicit_sliding_window` gained a non-causal row. The whole windowed DFlash path (the windowed pool, its CUDA
  graphs, the speculative wiring) was gated on `flash-attn`. It now builds and is linted in cuda builds, which
  surfaced one clippy `collapsible_if` in code CI had never compiled.
- `PrefixPrefillPlan::FlashAttentionPaged` and `run_flash_attention_paged_prefill`. FA2 builds sent image prefix
  ranges (bidirectional image spans inside causal attention) there. They now gather like every other build: the
  gather applies the ranges through its masks.
- The FA2 arm of `backends::flash_attn`. What is left: FA3 (feature `flash-attn-v3`, Hopper only, kept; it cannot
  be tested on this sm86 machine) for its head dims, else fattn, else eager.
- `mixed_cached_prefix_tests` (engine-level mixed cached-prefix packed prefill against per-sequence runs and a
  reference) was gated on `flash-attn`. It now runs in cuda builds against `FattnPaged`.

Elsewhere:
- Feature `flash-attn` removed from all crates and the CLI.
- The installers only add `flash-attn-v3` on Hopper; the doctor checks fattn's 7.5 requirement instead of FA2's
  8.0. The CI cutile lane no longer excludes the crate.
- Docs and CLAUDE.md build commands drop the feature.
- fattn's bench loses its FA2 column (`bench-fa2`); the numbers it compared are in Runs 4-12.
- The dev kernel library `libflashattention.so` (23.9 MB at sm86) is gone from every CUDA build.

Review follow-up:
- Every cuda build now creates the DFlash windowed pool, with no capability check before the fattn call. A
  pre-Turing GPU or a drafter head dim fattn lacks would have failed every draft step, eager fallback included,
  where it used to run DFlash's eager masked path. `windowed_kv_fits_fattn` now gates both the pool and its memory
  reservation: fattn takes the head dim on this device, every layer has a window, and every non-causal window is
  at least a draft block wide. A left-only window equals FA2's symmetric one only when the block fits inside it:
  the block sits at the end of its keys, so the keys to a query's right number at most `block - 1`. New test
  `windowed_kv_needs_every_layer_windowed_and_non_causal_windows_past_a_block`.
- With left-only windows in fattn, `Sdpa` and `FattnPaged` no longer refuse non-causal windows. That gives back
  the flash path FA2 builds had for those calls (FA2's `window_size_right = None`). The paged prefill runner passes
  the window for non-causal chunks too, and the prefill test gained a non-causal windowed case.
- Cleanup:
  - a doubled cfg in plan.rs;
  - the dead NHD pool view (`paged_attention_layer_cache`, FA2's layout);
  - stale "requires CUDA FlashAttention" and "fa2 aligns" wording;
  - two doc lines still listing `flash-attn`;
  - dflash's per-layer `seq_lens` now computed once per forward;
  - a cfg that my helper had split from `windowed_kv_cache_size_in_bytes`.

## Run 14 - 2026-10-05 (night)

Question: can FlashInfer's GQA paged decode go, now that fattn serves HND-layout decode (Run 10) and the gather
path can take what fattn refuses?

What stays: FlashInfer's library, for the MLA decode (`flashinfer_mla_decode`, which fattn cannot serve from separate
caches). With it stay the CSR page lists (FA3's fp8 decode reads them) and the tile plan (MLA reads it). What goes:
the GQA decode kernel (`flashinfer_decode*.cu/.cuh`, its FFI and Rust wrapper, and its test), the graph's decode
scratch (`tmp_v`/`tmp_s`), and `decode_metadata`. HND decode is now FA3's fp8 decode where it applies, then fattn,
then the gather (`run_decode_gather_sdpa`). The gather takes softcap at head dims fattn lacks (64), f32 caches, and fp8
scales above 146. MLA reads its view through `decode_view`, which marks the tile plan used for graph replays. The
review found the missing mark was never live: MLA families keep `CUDA_DECODE_GRAPHS = false`.

Found by the link step: the HND cache write and gather kernels (`reshape_and_cache_flashinfer`,
`gather_kv_cache_flashinfer`) lived in the deleted `flashinfer_decode.cu`. They moved, unchanged, to
`hnd_cache_kernel.cu`.

Found by the review, and my claim was wrong: I had written that a gather during graph capture fails the capture
because it uploads host-built offsets. The engine enables candle's byte-keyed host-to-device cache around capture,
and the eager warmup before capture has just uploaded the same bytes, so the upload hits the cache and the capture
succeeds. The graph then replays the lengths it captured with: wrong output from the first step past them, no error.
At HEAD, FlashInfer served these cases with device-side metadata, so this would have been a regression. The
Standard layout with a sliding window also gathers, so graph-enabled models there likely had the same latent bug.
Fix: `run_decode_gather_sdpa` refuses to run while its stream captures (`device_is_capturing`). The capture fails and
the engine falls back to eager, which is what the capture-failure path is for. New test
`gather_decode_refuses_graph_capture` begins a relaxed capture and runs a softcap-64 decode. Without the guard the
capture succeeds and the test fails.

Tests: the decode tests compared against `flashinfer_decode`, and now compare against the hand reference the prefill
tests use, generalized to block sizes and softcap. Each case asserts which path ran: the gather keeps a query axis
(rank 4), fattn does not (rank 3). The gather cases get a bf16 tolerance of 4e-2, since its eager attention runs tanh
and the softmax in bf16 (one softcap case measured 0.0188).

Still open: eager decode builds the CSR page lists and the tile plan every step, though only MLA and FA3 read them.
Gating them on the model (MLA layout, FA3) would recover Run 10's 2% eager cost.

## Run 15 - 2026-10-05 (late)

Question: which CUDA layers still use the Standard layout and vLLM's paged v1/v2 decode, and can the HND layout (fattn
decode) take them?

Who is on Standard on CUDA:
- gpt-oss and Llama 4 opt out (`KvCacheLayout::StandardNoFlashInfer`). gpt-oss for its sinks (FlashInfer's
  decode had none). Llama 4 for its chunked attention: a query sees keys from `floor(qp / chunk) * chunk` on,
  through custom masks, which is not a sliding window.
- Every model whose layers the FlashInfer layout refused: head dims other than 64/128/256/512, and GQA groups
  outside 1-8 and 16 (FlashInfer's `DISPATCH_GQA_GROUP_SIZE`).
- Metal, which keeps Standard by design.
- Neither gpt-oss nor Llama 4 has a local checkpoint or a tiny test fixture. Each needs a test-time tiny checkpoint
  before it moves.

This PR: the HND layout admits what fattn decodes. Head dims 64/80/96/112/128/256/512 (512 only GQA-batched, so
only with more Q heads than KV heads), any GQA group, and only when fattn's mma runs on every device. Without it,
decode would gather, slower than the Standard kernels. The group-size check stays for FA3's prefill split, which
needs it. The decode test covers the new head dims (80/96/112).

Bench, TinyLlama 1.1B Q4_K_M (head dim 64, GQA 8, so already HND on master), Standard (vLLM v1/v2 decode,
`INFERENCE_RS_FLASHINFER_DECODE=0`) against HND (fattn), TPOT ms:

| depth | graphs: Standard / HND | eager: Standard / HND |
|---|---|---|
| 4 | 1.57 / 1.62 | 2.24 / 2.41 |
| 1024 | 1.85 / 1.70 (-8%) | 2.65 / 2.48 |
| 1900 | 1.91 / 1.76 (-8%) | 2.62 / 2.62 |

Found on the way: the Standard layout with CUDA graphs crashes (`CUDA_ERROR_ILLEGAL_ADDRESS`, "invalid CUDA top-1
output") when one process decodes at depth 4 and then at 1024/1900, while each depth alone runs. It reproduces at
#278, before any of this work, and not with graphs off. So it is a latent fault in the Standard layout's graph
replay across context buckets, and one more reason to move CUDA models off that layout. Not chased here.

Next: tiny gpt-oss and Llama 4 checkpoints, then fattn sinks for gpt-oss (moving it to HND and retiring the sinks
kernel), a chunked mode in fattn's implicit mask for Llama 4, and then the vLLM v1/v2 CUDA kernels. Head dims fattn
lacks would then gather.

Review follow-up:
- Gemma 4 chose its layout with its own copy of the old rule: head dims 64/128/256/512, any group, no Turing gate.
  On a pre-Turing GPU it would have taken HND caches and then gathered every decode step. It now asks the same
  `supports_layer`.
- The rule lives in one place, `inference_fattn::paged_shape_supported(head_dim, q_heads, kv_heads)`: mma on every
  device, a paged head dim (64/80/96/112/128/256/512), and 512 only GQA-batched. `supports_layer` and Gemma 4 use
  it.
- FA3's fp8 decode only ever saw groups 1-8 and 16 (the layout admitted no others). Its schedule key now asks for
  those groups again (`fa3_group_size_supported`, renamed from the FlashInfer decode check), so a group-12 fp8 layer
  on Hopper decodes on fattn, not on an untested FA3 shape.
- "Any group" is now tested at the layer: `decode_with_any_gqa_group` covers 6/2, 18/2 and 71/1 (MQA, as
  Falcon-7B) against the reference. The layout test checks the 512 rule both ways, an 80-dim head and a head dim
  fattn lacks (72).
- Not measured: MHA (group 1) at 80/96/112, which Phi-2 and Phi-3-mini now take to HND. No such checkpoint is local.
  Phi-3-mini's window gains: Standard with a window gathered, HND runs fattn's window.

## Run 16 - 2026-10-05 13:32

Question: before moving gpt-oss (sinks plus a sliding window) and Llama 4 (chunked attention) off the Standard
layout, do they have any coverage, and does their CUDA paged decode agree with CPU today?

Neither had a test or a local checkpoint. `crates/inference/tests/integration/local_attention_tiny.rs` builds tiny
random-weight checkpoints at test time (head dim 64, window and chunk 8, prompts several windows long, 24 decode
steps). CPU runs twice must match exactly; on CUDA, paged bf16 decode is compared with CPU bf16 over three prompts.

```
cargo nextest run -p inference [--features cuda] --test integration -E 'test(/^local_attention_tiny::/)'
```

Raw findings, in the order they surfaced:
- Llama 4 on CPU: "Invalid sampling probability at index 0: NaN". The prompt step was fine, decode was not, and only
  with a chunk smaller than the prompt. Cause: the single-query CPU kernel (`cpu/single_q.rs`) starts with
  `m = -inf`; a masked key ahead of the first live one scores `-inf`, and `exp(-inf - -inf)` is NaN. A chunked decode
  mask hides the keys before its chunk, so every Llama 4 CPU decode past the first chunk returned NaN. Causal and
  rotating-window caches never have leading masked keys, which is why nothing else hit it. Fixed by skipping `-inf`
  keys (the multi-query path already guarded this); `test_flash_attn_cpu_single_q_with_leading_masked_keys` is NaN
  before the fix.
- Llama 4 on CUDA: `DriverError(CUDA_ERROR_NOT_FOUND, "named symbol not found")` from `position_ids.to_dtype(I32)`.
  candle's CUDA cast kernels have no integer-to-i32 casts at all, so Llama 4 never ran on CUDA. The only consumer
  casts to f32, so the i32 cast is gone.
- gpt-oss on CUDA: "matmul is only supported for contiguous tensors" with a `[1, 2, 81, 64]` query at strides
  `[10368, 64, 128, 1]`. Its sliding layers get a custom mask once the prompt exceeds the window, which routes sinks
  attention to the unfused fallback, and that passed the head-transposed prompt query to a GPU matmul. Any gpt-oss
  prompt longer than its 128-token window on CUDA would have failed this way. The fallback now makes q contiguous.
- Comparison design (dead ends kept): exact token equality over 24 steps fails on random weights; bf16 flips a near
  tie at step 13 (logprobs -2.021 vs -2.020). Stopping at the first step where either side's margin is under 0.1
  left only 5 steps. f16 did not help (same 5 steps, and Llama 4 overflowed to NaN on the GPU). What holds: each
  trace must match token for token until it splits, the split must be at a CPU near tie (margin under 0.5), and the
  matched steps across three prompts must reach 24. Baseline: 35 (gpt-oss) and 49 (Llama 4). bf16 logprob drift
  reached 0.30 on matching tokens, hence the 0.5 tolerance.
- Random sinks (std 0.5) are invisible next to scores of std ~4: dropping the sinks from the paged decode kernel
  still passed. The fixture now sets every sink to logit 8, near the top scores.
- Mutation checks, GPU side only, each reverted: no window or mask in sinks prefill fails gpt-oss; no sinks in paged
  decode fails gpt-oss; no chunking in prefill or decode masks fails Llama 4 (divergence at step 1).

Today's paths: Llama 4 is right on CUDA (its decode builds the chunked mask from paged metadata, then gathers K/V for
SDPA), but slow; gpt-oss decodes on the vLLM paged kernel with sinks. Both are the ground truth for the fattn moves.

Next: gpt-oss onto fattn sinks over the HND layout, retiring `flash_attn_sinks.cu`.

Review follow-up:
- Same class of NaN in the multi-query tiled CPU kernel (`cpu/full.rs`): a general (non-binary) mask row whose first
  128-key tile is fully masked softmaxed against a `-inf` max. Causal, window and chunked masks are binary and take
  another path, so it needs a biased mask. Fully masked tiles are now skipped;
  `test_flash_attn_cpu_full_q_with_a_masked_leading_tile` is NaN before the fix. The single-query skip moved ahead of
  the dot product, as in the other paths.
- The unfused sinks path serves every device for a custom mask, so it is `sinks_attn_unfused` now.
- The comparison got stricter: where a trace splits, the GPU must take the CPU's runner-up at a near tie (not any
  token), runner-up logprobs are compared as well, and a step with fewer than two top logprobs is an error. A minimum
  per prompt was tried and dropped: Llama 4's second prompt meets a genuine tie at step 1 (margin 0.094, GPU took the
  runner-up), so per-prompt counts are 24/1/24 there and 5/6/24 for gpt-oss. The total (24) stays. Both decode
  mutations still fail.
- Full CI caught `causal_flash_masks_a_cpu_mapped_layer` at max diff 0.0102 against its 0.01 tolerance. It passes
  alone; `--stress-count 300` failed 1 of 93 before stopping. The inputs are unseeded `randn` with d = 64 and
  scale 1.0, so logits have std ~8. With queries scaled by 1/sqrt(d), as real models do, 2000 of 2000 runs pass.
  It predates this PR and is not touched by the kernel fixes (6 queries, binary mask rows).

## Run 17 - 2026-10-05 14:29

Question: can gpt-oss run on fattn's sinks (prefill, paged prefill, HND decode), so `flash_attn_sinks.cu` can go?

Changes: Sdpa tries fattn first for sinks on CUDA (FA3 never takes sinks), and builds the causal/window mask
explicitly when it falls back to the unfused path. The paged fattn decode and prefill pass the sinks; the HND
decode plan and fattn's paged-prefill plan no longer reject them; packed sinks prefill stays packed on CUDA (only
Metal's kernel wants it padded). `sinks_backend_supports` now means fattn on CUDA (f16/bf16, its head dims, Turing+)
and the sinks kernels on Metal. gpt-oss drops its `StandardNoFlashInfer` pin and its forced custom masks, which had
sent every unpacked CUDA prompt through the unfused path.

Layer checks first (all pass): `decode_with_sinks_matches_a_reference` (head dim 64, bf16/f16/fp8, no window, a
100-token window, a full layer of a windowed model, and an 8-token window shorter than a 32-row block) and
`prefix_prefill_with_sinks_matches_a_reference` against a hand reference with the sink as an extra softmax logit;
`sinks_prefill_on_cuda_matches_the_cpu` for the dense Sdpa path. At gpt-oss-like magnitudes (queries x4, sinks 8)
bf16 decode stays under 0.01 max abs and the dense prefill at ~1 ulp once the reference sees bf16-rounded inputs.

End to end the tiny gpt-oss test then failed: prompt 3 split at step 5 with the GPU's top token (logprob -1.51)
outside the CPU's top three (-1.81/-2.00/-2.78). Bisected with env toggles: graphs off fails, Standard decode
(`INFERENCE_RS_FLASHINFER_DECODE=0`) fails, skipping only the fattn prefill passes, so fattn's dense prefill was
the trigger. Inside the model, fattn against the unfused path on the same inputs differed by 0.05-0.125 (1-2 bf16
ulps at these output magnitudes), with layouts reproduced exactly in the unit test, which passes. So not a kernel
bug: the old GPU path was the same unfused bf16 algorithm as the CPU and rounded alike, fattn rounds differently,
and the fixture's top-1-of-2 MoE routing turns ulps into a different expert.

Fixture changes, and the numbers behind them:
- Every expert per token (both models), so routing is continuous: still 16 agreed steps of 72 for gpt-oss in bf16,
  every split now a genuine runner-up swap at margins 0.03-0.06.
- The extra routing made gpt-oss sample its end token, so the requests ignore EOS (each step still checks attention).
- gpt-oss in f16: 72 of 72 steps match, with fattn and with the fallback alike. Llama 4 stays bf16 (f16 overflowed
  to NaN on the GPU in Run 16) and still matches.
- Mutations, each reverted: no sinks in fattn decode fails; no sinks in fattn prefill fails.

Deleted: `flash_attn_sinks.cu` and its bindings (six FFI entry points). Metal keeps its sinks kernels; a CUDA call
fattn cannot take (pre-Turing, f32, head dims it lacks) takes the unfused path, and a varlen one fails loudly
rather than run the CPU loop that applies no mask.

Not measured: no gpt-oss checkpoint is local, so there are no real-model numbers here. Prefill had run the unfused
path for every unpacked prompt and sliding decode gathered, so both should only get faster.

Next: Llama 4's chunked attention as a fattn mask mode, then the vLLM v1/v2 CUDA kernels.

Review follow-up:
- The explicit causal/window mask was built for every CausalFlash call on any device. On Metal that sent a fused
  sinks call to the unfused path, and built an unused O(max_q x total_kv) host mask for packed prefill. It is now
  built only where no fused kernel follows. `causal_flash_with_sinks_masks_the_unfused_path` covers the CPU side, which
  had silently dropped causality when a CausalFlash mask reached a CPU-mapped layer of a CUDA model.
- `DecodePlanInput::has_sinks` was dead after the HND plan stopped reading it; removed. The Metal sinks shader no
  longer says it ports the deleted CUDA file.
- Left as follow-ups: sinks are cast to f32 on every fattn call (one 64-element kernel per layer per step; each
  backend wants a different dtype, so a load-time copy needs its own field), and f32 caches on Turing+ take HND and
  then gather every decode (all models, not just gpt-oss; f32 serving is rare).

## Run 18 - 2026-10-05 15:35

Question: can fattn's implicit mask express Llama 4's chunked attention (a query sees keys from the start of its
own chunk), so Llama 4 decodes on fattn over the HND layout instead of gathering K/V for a custom mask every step?

Kernel: `fattn_layout` gains `chunk` and `full_lens`; the mma tile's implicit mask adds "same chunk of absolute
positions", with row 0 of a sequence at `full_lens[s] - kv_len` (0 without full lengths). The per-sequence `qkv`
became an `int3` carrying that origin. Absolute positions matter because a chunked layer's paged tables are a
window's: they start at a block-aligned row, which is a chunk edge only when the chunk divides the block size.
Masking only, as the window already is: tiles before the chunk are still computed, since bounding the left of the
stream-k loop would rework its fixups (a later optimization for window and chunk alike).

Rust: `FattnOptions::chunk`, `PagedKv::full_lens`. `SdpaParams` gained `chunk` (73 literals, scripted; a grep
confirms none lacks it, Metal-only files included). Sdpa sends a chunked call without a mask to fattn or fails;
the paged decode and prefill pass the chunk and, over a window's tables, the device full lengths (failing if the
metadata lacks them). A chunked layer that would decode off HND, or prefill on the gather, fails rather than
silently attending by window.

Llama 4: a chunked layer sets `chunk` where fattn serves it (CUDA, f16/bf16, Turing+) and then takes the plain
causal mask; elsewhere it keeps its host-built chunk masks, which the model now builds only if some layer needs
them. Both layout pins (model and loader) are gone, so its caches take HND.

Tests:
- `parity::implicit_chunks` (dense: decode at and after a chunk edge, prefill over many chunks, a non-causal block)
  and `paged::chunks_count_from_full_lens` (tables starting mid-chunk, at an edge, and at 0); each fails with its
  chunk, or the full lengths, removed.
- `chunked_decode_matches_a_reference` (bf16 and fp8, 1 and 3 query rows, chunk 100 over 32-row blocks, through the
  real decode metadata) and `chunked_prefix_prefill_matches_a_reference`; the decode one fails without full lengths.
- The tiny Llama 4's chunk went from 8 to 12: 8 divides the 32-row blocks, so a wrong origin still landed on an
  edge. At 12, bf16 ties left 16 matched steps of 72 (a runner-up swap at margin 0.031, not a fault); f16 still
  overflows on the GPU (as in Run 16, before any of this), so the fixture now has six prompts: 64 matched steps for
  Llama 4, 144 of 144 for gpt-oss. Removing the chunk from fattn decode, from fattn prefill, or the full lengths from
  decode each fails it.

Not measured: no Llama 4 checkpoint is local (Scout is 109B). Decode moves from a per-step gather plus custom mask to
fattn over HND; Llama 4 still declines CUDA decode graphs, which can now be revisited.

Next: delete the vLLM v1/v2 CUDA kernels (Standard layout on CUDA gathers for head dims fattn lacks; Metal keeps it).

Review follow-up (three real faults, all fixed):
- Eager CUDA (no paged attention) read the wrong keys: Llama 4's config advertises its chunk as a sliding window, so
  the eager inputs describe K as the window's last rows, and the chunked layers' `sliding_window` sent fattn down the
  varlen path over the full, unrotated cache. Silent past one chunk. The eager call now drops the window (the chunk
  is the tighter bound). The fixture gained an eager GPU run, which fails with the fix reverted ("Eager").
- Padded later prompt chunks would have failed on the new "runs only on fattn" guard, since fattn's paged prefill
  takes no padding. Llama 4 now builds its chunk masks (and gathers) for such a step. Neither path reaches it today:
  Llama 4 is on the multimodal pipeline, whose prompts are atomic, and its loader does not opt into prefix caching.
  The fixture's repeat run (prompts again, concurrently, 16-token prefill chunks) exercises padded prefix prefill
  for gpt-oss only; Llama 4's chunked prefix prefill is covered at the layer.
- `INFERENCE_RS_FLASHINFER_DECODE=0` (or any shape HND does not admit) would have failed every chunked decode:
  `chunks_on_fattn` now also asks `FlashInferAttentionBackend::supports_layer`, as Gemma 4 does.
- Also: a missing device entry in the full lengths now fails instead of counting from 0, and the unused
  `KvCacheLayout::StandardNoFlashInfer` is gone.
