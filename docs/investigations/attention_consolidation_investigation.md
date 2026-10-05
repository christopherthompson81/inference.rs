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
