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
