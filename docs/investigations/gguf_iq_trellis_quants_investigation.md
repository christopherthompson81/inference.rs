# GGUF IQ and trellis quants investigation

Goal: load and run GGUF checkpoints quantized with llama.cpp's IQ types (IQ1_S through IQ4_XS) and ik_llama.cpp's
types, starting with the trellis quants (IQ1_KT through IQ4_KT) and the `IQ*_K` / `IQ*_KS` families.

## Run 1 - 2026-10-02 (time approximate)

Question: what blocks an IQ GGUF today, and what reference code exists to build and check support against?

Trigger: a local Qwen3.8-27B IQ4_XS GGUF fails at load, before any model code:
`GGUF tensor blk.0.attn_gate.weight uses dtype IQ4_XS (23) ... direct GGUF loading currently supports F32, F16, BF16,
Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, and Q2_K through Q8_K`.

Findings:
- Our GGUF archive parser (`inference-quant/src/gguf/archive.rs`) reads every type id itself. Direct loading then goes
  through candle's `QTensor` / `QMatMul` (`inference-quant/src/gguf/mod.rs`, `GgufMatMul`), so it is limited to candle's
  `GgmlDType` (F32, F16, BF16, Q4_0 through Q8_1, Q2K through Q8K, at the pinned candle rev e65eb1d). Candle has no IQ
  or trellis types.
- The CUDA GGUF matmuls are ours (`inference-quant/kernels/cuda/{mmvq_gguf,mmq_gguf}`, about 9k lines ported from
  ggml), dispatched per candle dtype.
- Precedent for a type outside candle: MXFP4 binds raw blocks (`GgufTensorBinding::Mxfp4Blocks`) and runs its own
  kernels (`inference-quant/src/mxfp4`, `kernels/cuda/mxfp4`).
- References:
  - mainline: llama.cpp (local, `/home/chris/Programming/llama.cpp`). Type ids IQ2_XXS 16, IQ2_XS 17, IQ3_XXS 18,
    IQ1_S 19, IQ4_NL 20, IQ3_S 21, IQ2_S 22, IQ4_XS 23, IQ1_M 29, TQ1_0 34, TQ2_0 35. Its `gguf-py/gguf/quants.py` has
    numpy dequantizers for all of these, usable as test goldens.
  - ik_llama.cpp (MIT, shallow clone at 5f89bfc). It extends the type enum with its own ids:
    - IQ2_K 137, IQ3_K 138, IQ4_K 139, IQ5_K 140, IQ6_K 141;
    - IQ4_KS 144, IQ2_KS 145, IQ4_KSS 146, IQ5_KS 152, IQ3_KS 156, IQ2_KL 157;
    - trellis IQ2_KT 153, IQ3_KT 154, IQ4_KT 155, IQ1_KT 158;
    - CPU-repacked `_R4` / `_R8` variants (337+). Those are a runtime layout and should not appear in distributed
      GGUFs. Not targeted.
    - It has no Python dequantizers, so goldens come from its C reference (`ggml-quants.c` / `iqk` sources) built in a
      scratch harness.

Implication: candle can't take the new types without a fork. Following the MXFP4 pattern, an inference-quant quant
method would hold raw blocks with its own dequant and matmul kernels. Proposed stages, each its own PR:
1. Correctness: block layouts and a CPU dequant for each type, checked bit for bit against gguf-py (mainline) and the
   ik C reference. CUDA runs a dequant-to-BF16 kernel plus a GEMM. Slow, but every type is exact.
2. Decode speed: per-type mmvq (dequant-on-the-fly dot products), from ggml-cuda's IQ vec_dot and ik's KT kernels.
3. Prefill speed: mmq tiles for the most used types.
Order: IQ4_XS and IQ4_NL first (they unlock the local Qwen3.8-27B), then the rest of the IQ family, then the trellis
types, then the `IQ*_K` family.

## Run 2 - 2026-10-02 (time approximate)

Decision (user): fast kernels type by type, not a dequant-everything first pass. IQ4_XS first.

Question: what kernel code already exists, and what does the target file need?

Findings:
- The local Qwen3.8-27B IQ4_XS GGUF types (gguf-py reader): IQ4_XS 288, Q5_K 113, Q4_K 7, Q6_K 1, Q8_0 1, F32 456. Only
  linear projections are IQ4_XS: attention, MLP, GDN in_proj/attn_gate. token_embd is Q4_K and output is Q6_K, so
  dense (non-linear) IQ loads are not needed for it.
- Our ggml mmq port (`kernels/cuda/mmq_gguf/`) already carries mainline IQ code that is never instantiated:
  `load_tiles_iq{1_s,2_xxs,2_xs,2_s,3_xxs,3_s,4_nl,4_xs}`, the matching `mmq_type_traits`, and
  `vec_dot_iq*_q8_1` in `mmq_vecdotq.cuh`. There are no `mmq_instance_iq*.cu` files.
- `kvalues_iq4nl` is real. The grids are zero stubs: `iq2xxs_grid[256] = {0}`, `iq3xxs_grid[256] = {0}`,
  `iq1s_grid_gpu[512] = {0}`. The IQ1/2/3 types need the real tables from ggml-common.h.
- `mmvq_gguf.cu` (decode) is a separate implementation with its own vec_dots and plain / fused_glu / fused_qkv
  launchers, k-quants only.
- Reference binaries: `/home/chris/Programming/llama.cpp/build*/bin/{llama-cli,llama-simple}` are built, for greedy
  parity on the real file.

Implication, the IQ4_XS work:
- a ggml type layout that is independent of candle (block and type size);
- a raw-block `QuantMethod` for linear weights;
- mmvq launchers built on the existing `vec_dot_iq4_xs_q8_1`;
- an `mmq_instance_iq4_xs.cu`;
- CUDA and CPU dequant;
- a gguf-py golden for the CPU dequant;
- a real-file parity run against llama.cpp.
IQ4_NL shares the codebook and costs little extra.

## Run 3 - 2026-10-02 (time approximate)

Question: do IQ4_NL / IQ4_XS run, fast and correct, on the CUDA paths, and does the Qwen3.8-27B IQ4_XS file load and
match llama.cpp?

Change:
- `GgufType` is our ggml type enum: Candle's types plus Iq4Nl and Iq4Xs. `KernelWeight` (type, shape, device, pointer
  plus guard) is implemented by `QTensor` and by the new `RawGgufTensor`.
- `fast_mmvq` / `fast_mmq` dispatch on `GgufType`, and `plain` is generic over `KernelWeight`.
- mmvq: the IQ4 vec_dots ported into `mmvq_gguf.cu` with plain launchers. mmq: `mmq_instance_iq4_{nl,xs}.cu`
  instantiates the dormant port.
- `GgufRawMatMul`: CUDA mmvq for batches up to 8, mmq above; CPU dequant plus matmul otherwise. ISQ goes through the
  dequantized weight.
- The weight source builds it for raw types, direct and through the packed (structural) bindings.

Tests:
- CPU dequant equals gguf-py exactly (`goldens.json` from `make_goldens.py`).
- CUDA mmvq/mmq vs the dequantized weight: cosine > 0.999 at batches 1/8/33, F32 and BF16.
- `inference-quant` GGUF tests: 58 pass on CUDA.

27B file, first try: the load succeeds, then "PagedAttention KV cache requires 256 MB but only 0 MB is safely
available".
- Load-path counters: 288 IQ4_XS raw layers (10.7 GB). Dense loads are only the small F32 GDN alpha/beta.
- nvidia-smi peaks at 21.5 GB against ~15 GB of weights.
- Cause: the GDN recurrent-state pool, "Recurrent state pool reserved capacity=33". 48 value heads x 128 x 128 x F32 is
  ~3 MB per layer per slot; x 48 GDN layers x 33 slots is ~5 GB before the KV planner runs.
- Not an IQ issue. The tests cap `max_num_seqs` at 2.
- Follow-up: size the recurrent pool to the memory left. The paged KV can already be capped (`with_gpu_memory`) and
  quantized (`F8E4M3`), but the recurrent pool ignores both.

Results:
- `qwen3_5_mtp::gguf_builtin_mtp_*` on the 27B: 40/40 and 35/35 greedy ids identical with MTP; 40 of 58 drafts
  accepted.
- `gguf_iq::iq4_xs_greedy_continuation_matches_llama_cpp`: our greedy completion of "The three primary colors of
  light are" equals `llama-simple -n 32 -ngl 99` (llama.cpp build_cuda) token for token, apart from the leading space
  llama-simple prints. Decode ~50 tok/s.

## Run 4 - 2026-10-02 (time approximate)

Question: do the review findings on the IQ4 change hold up, and does the fixed tree pass?

Review findings and fixes:
- Raw CUDA weights had no tail padding.
  - mmq steps K in whole tiles: 8 blocks for IQ4_NL. Rows whose K is not a multiple of 256 read past the buffer;
    llama.cpp writes IQ4_NL for exactly those tensors.
  - Fix: zeroed padding of 512 elements' worth of blocks, as Candle pads its QTensors.
  - New K=288 IQ4_NL case. With the padding set to 0 it gives `Iq4Nl batch 33 F32: cosine NaN`; with 512 it passes.
- Raw types passed validation, but dense materialization (an IQ4_XS token_embd, or a transform binding) still went
  through `load_qtensor`. They now dequantize through `dequantize_blocks`, with the little-endian check that the
  unsliced raw load also lacked.
- `GgufRawMatMul` falls back to the CPU path for an empty batch or a non-BF16/F16/F32 input.
- Selection now resolves `--quant iq4_xs` / `iq4_nl`. Other IQ names still get a clear error, and numeric widths never
  fall back to IQ.
- The real-file test gates on CUDA (Metal has no raw-block kernels). Plus style nits.

Command: `./scripts/local_ci.sh --lint --tests --cuda --slim --bindings --docs --sweep`.
Result: exit 0, 2401 + 2724 + 1 tests passed, the 27B parity test included (16 s, run alone).

Not covered: IQ4 MoE expert stacks (rank 3 is rejected), IQ4 on Metal, the other IQ types, and the trellis and `IQ*_K`
types.

## Run 5 - 2026-10-02 (time approximate)

Question: can the 27B load at the default `max_num_seqs` (32) instead of the tests capping it at 2?

Change: the GDN recurrent pool is sized to the memory left.
- Before the pool is reserved, `automatic_recurrent_capacity_budget` caps the slot count. It uses the formula the
  checkpoint-lane budget already used: free or utilization-target memory, counting the current layout as reusable, less
  the reservations still to come and the KV floor.
- `HybridCache::fitted_serving_capacity` records the fitted count, and `Engine::new` lowers the scheduler's
  `max_num_seqs` to it. Without that, the engine would grow the pool back.

First tries, from the logs and temporary prints of the budget against the KV planner's inputs:
1. Fitted 28; the KV planner then had 48 MB for a 256 MB floor. Missing: the GDN deferred-state storage reserved right
   after (183 MB, scaling with capacity). Added `SpeculativeTargetMixin::recurrent_decode_deferred_slot_bytes`
   (6.3 MB/slot here) and `HybridCache::gdn_deferred_state_slot_bytes`, both forwarded by the delegate macro.
2. Fitted 27, 176 MB for 256. Also added the GGUF affine repack budget the KV planner subtracts (0 in this build), and
   stopped counting deferred bytes into the reusable current layout.
3. Plain load OK (27 slots). The MTP load had 249 MB for a 272 MB floor: the estimate left 17 MB of slack, and
   allocator rounding plus small per-pool tensors are outside `snapshot_bytes`. A binding budget now keeps one slot
   spare.

Result: at defaults, the plain 27B fits 25 sequences and the MTP build 22, both with the full 4064-token KV. Both
real-file tests pass without the cap. A unit test pins the measured 27B budget: 27 slots fit, 26 kept.

Not changed:
- The prefix cache reserves recurrent snapshots up front: 2.68 GB of `future_reserved_bytes` here, for 16 prefix
  slots. That is a separate trade-off.
- The paged KV can be F8E4M3 or capped; the recurrent state stays F32.

Run 5 review follow-ups, all fixed:
- The allocation check counted the current layout as freeable at the peak, but the new layout is allocated beside it.
  The peak check is now `allocation_available / layout_bytes`; the current layout is counted only for the deferred
  storage allocated after it is freed.
- Deferred bytes were the whole model's on every device. `gdn_deferred_state_slot_bytes_by_device` is per device, and
  the trait now reports the deferred-state spec, not a byte count.
- Explicit (non-auto) checkpoint lanes fit capacity at that lane count. Auto lanes still fit sequences first, then
  depth.
- Only CUDA devices are capped, and the GGUF affine budget applies only on CUDA, as in the KV planner.
- The lane-adjust log prints the capped count.
Unit test: measured 27B numbers give 26 slots at one lane and 13 at a fixed two lanes. Full CI: exit 0 (2402 + 2725 + 1).
31 recurrent tests and the 4
real-checkpoint tests pass at default `max_num_seqs`.

## Run 6 - 2026-10-02 17:30

Question: do the IQ1/IQ2/IQ3 kernels match llama.cpp on real files?

Types added: IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S and IQ1_M.
- CPU dequantizers are ports of ggml-quants.c. The grid tables are generated from ggml-common.h by
  `scripts/make_iq_tables.py`.
- mmvq covers all 7 types; mmq covers all except IQ1_M, which runs chunked mmvq for prefill.

Checks:
- Goldens: CPU dequantization equals gguf-py exactly for all 9 IQ types.
- GPU mmvq, mmq and the chunked paths reach cosine >= 0.99996 against the dequantized weight.

Real files: Qwen3.5-0.8B, quantized with `llama-quantize` from an F16 conversion made with `--no-mtp`, plus an imatrix.
Without `--no-mtp`, quantizing failed with "Missing importance matrix for tensor blk.24", the MTP layer.

Test 1, greedy text against `llama-simple` (24 tokens):
- IQ2_S, IQ2_XXS and IQ3_S match exactly.
- IQ1_M, IQ1_S, IQ2_XS and IQ3_XXS diverge after several identical tokens.

Test 2, perplexity against `llama-perplexity -c 512 --chunks 1 -ngl 99`. We score the same 512 tokens with
`prompt_logits`, over tokens 257..511, which is what llama-perplexity scores.

Calibration text (llama.cpp README plus build docs):

| Type | Ours | llama.cpp |
|---|---|---|
| IQ1_M | 69.29 | 70.77 |
| IQ1_S | 330.29 | 331.74 |
| IQ2_S | 8.105 | 8.111 |
| IQ2_XS | 9.220 | 9.306 |
| IQ2_XXS | 16.545 | 16.622 |
| IQ3_S | 3.847 | 3.814 |
| IQ3_XXS | 4.369 | 4.325 |

This repo's README (the committed test's text):

| Type | Ours | llama.cpp |
|---|---|---|
| IQ1_M | 290.54 | 290.41 |
| IQ1_S | 829.07 | 838.66 |
| IQ2_S | 24.444 | 24.461 |
| IQ2_XS | 30.577 | 30.304 |
| IQ2_XXS | 61.410 | 61.741 |
| IQ3_S | 10.327 | 10.352 |
| IQ3_XXS | 13.124 | 13.134 |

Every type agrees within about 2%, so the greedy divergences are near-ties on noisy low-bit models, not kernel bugs.
The committed test is `every_iq_gguf_matches_llama_cpp_perplexity`: 3% tolerance, run against the GGUF directory
named by `INFERENCE_TEST_IQ_GGUF_DIR`. It replaces the greedy test. It passes in 25 s.

Run 6 review follow-ups (18:00):
- Model selection now resolves the IQ1-IQ3 labels by name, including llama-quantize's mixes IQ2_M, IQ3_XS and IQ3_M.
- The test now tokenizes with special tokens, so a model that adds a BOS scores the same window llama-perplexity does.
- The test requires one file per type, and the tolerance is tightened to 2% (largest drift seen: 1.15%, IQ1_S).
- The GPU kernel test also covers F16 activations.
- IQ1_M prefill ran ceil(batch / 8) mmvq launches per linear, since ggml has no IQ1_M mmq tile. Run 7 replaces it.
- Full CI: exit 0 (2403 + 2726 + 1 tests). One fixture had used IQ2_XXS as its unsupported type; it now uses TQ2_0.

## Run 7 - 2026-10-02 19:10

Question: how does llama.cpp run IQ1_M prefill, and does doing the same fix our slow fallback?

Upstream master on GitHub (`bed0a856`, newer than the local checkout):
- `ggml_cuda_should_use_mmq` in `mmq.cu` lists every IQ type except IQ1_M, and there is no `mmq-instance-iq1_m.cu`.
- For prompt-sized batches, ggml dequantizes the weight to F16 (`convert.cu`) and runs a cuBLAS GEMM.

Change: ported `dequantize_iq1_m` from `dequantize.cuh` (BF16, F16 and F32 outputs). IQ1_M batches above 8 now
dequantize the weight on the GPU and run a dense matmul.

Results:
- CUDA dequantization matches the CPU port within 1e-5 at each output dtype (a new unit test).
- Timing, a temporary microbenchmark not committed: 3584x1024 IQ1_M weight, 512 BF16 rows, 50 iterations after
  warmup, on this machine's GPU:
  - chunked mmvq: 1.227 ms
  - dequantize plus matmul: 0.099 ms, about 12x faster
- Perplexity:
  - IQ1_M with a BF16 dense weight: 294.47 vs llama.cpp's 290.41 (1.4%). It was 290.54 with mmvq.
  - Review pointed out that ggml dequantizes to F16, not BF16. With F16 weight and activations (F32 stays F32):
    293.61 (1.1%).
  - The remaining gap is unexplained; this path also keeps unquantized activations where mmvq used Q8_1.
  - The other six types are unchanged.
- Memory: each IQ1_M prefill matmul allocates its dense F16 weight for the call. It is freed after, and the
  memory planner does not reserve it. That is about 180 MB for a 5120x17408 projection.
- Full CI: exit 0 (2403 + 2727 + 1 tests).

## Run 8 - 2026-10-02 20:15

Question: can the trellis types (IQ1_KT through IQ4_KT) load and run, matching ik_llama.cpp?

Format, from ik_llama.cpp at `5f89bfc` (`ggml-common.h`, `ggml.c`, `iqk_quantize.cpp`, ggml-cuda):
- Type ids: IQ2_KT 153, IQ3_KT 154, IQ4_KT 155, IQ1_KT 158.
- 256-element blocks of 68, 100, 128 and 56 bytes. The blocks carry no scale: each row starts with an f32 row
  scale (`row_meta_size = 4`).
- IQ3_KT and IQ4_KT rows may also end in 32-element tail sub-blocks (`ggml_row_size`, padded to 4 bytes).
- Values come from an integer trellis. A 12-16 bit index plus 4096 seeds a state that is multiplied by
  0xCBAC1FED; each value is the sum of the state's four 6-bit bytes minus 126. Scales are IQ4_NL codebook nibbles
  (IQ1/IQ2), plain nibbles (IQ3, which stores signs separately), or a 7-bit signed scale (IQ4).
- Discrepancy: the CUDA and CPU-GEMM kernels scale IQ2_KT by 1.05, but the reference `dequantize_row_iq2_kt` does
  not. IQ3_KT's 1.01 appears in all of them. We follow the kernels, since that is what inference computes.

Change:
- Row-unit sizing: a trellis tensor's unit is the whole row (cols, row bytes). Slicing and concatenating along rows
  works; splitting a row is refused.
- A CPU dequantizer ported from ik's CUDA dequant kernels.
- CUDA:
  - mmvq: a new translation unit, `mmvq_kt.cu`, porting `iqk_mul_mat_vec_q_kernel` and the four KT vec_dots,
    with tails.
  - Prefill: dequantize to F16 plus a dense matmul, the IQ1_M path. ik has KT mmq tiles; we don't port them yet.

Checks:
- Goldens (`tests/fixtures/gguf_kt/make_goldens.py`): it calls ik's `dequantize_row_iq*_kt` through its
  `libggml.so` via ctypes, with IQ2_KT's row scale pre-multiplied by 1.05. Ours matches to the bit for all four
  types, including a tail on IQ3_KT and IQ4_KT.
- GPU dequant matches the CPU within one rounding step. mmvq and the dense prefill reach cosine > 0.999 against the
  dequantized weight, including tail rows.

Real files:
- Built ik_llama.cpp with CUDA (sm_86) in the scratchpad.
- ik's `llama-quantize` could not read mainline's GGUF-format imatrix ("failed reading number of values for entry
  1"). Regenerated it with ik's `llama-imatrix` on the same calibration text.
- ik's default KT mixes put some tensors in `IQ3_K` / `IQ4_K` / `IQ5_K` / `Q6_K`; we don't support the `IQ*_K`
  types yet. Requantized with `--pure --token-embedding-type q8_0`.

Perplexity on this repo's README, against ik's `llama-perplexity -c 512 --chunks 1 -ngl 99`:

| Type | Ours | ik_llama.cpp |
|---|---|---|
| IQ1_KT | 160.29 | 160.40 |
| IQ2_KT | 33.280 | 33.388 |
| IQ3_KT | 11.545 | 11.643 |
| IQ4_KT | 9.7366 | 9.7347 |

All within 0.85%.
- `every_trellis_gguf_matches_ik_llama_cpp_perplexity` reads `INFERENCE_TEST_KT_GGUF_DIR` and
  `INFERENCE_TEST_IK_LLAMA_PERPLEXITY`.
- ik prints its estimate to stdout in a different form, and the parser takes both.

Also fixed: `gguf-support.md` still said IQ types were unsupported, which has been wrong since #238.

Not done:
- KT mmq tiles.
- The `IQ*_K` family, which ik's default KT mixes need.
- KT MoE expert stacks.
- Metal.

Run 8 review follow-ups (20:50):
- `shard_alignment` returned 256 for trellis types. It now returns an alignment no rank split meets, so the MoE
  planner replicates rather than failing mid-load. Concatenating trellis rows end to end is refused explicitly.
- Goldens now hold two rows each, plus 480-column IQ3_KT / IQ4_KT rows (seven tails, both IQ3_KT scale nibbles).
  They still match ik to the bit.
- GPU tests use an odd row count and batch 3, and cover 480-column tails.
- New weight-source unit test: trellis tensors slice and concatenate by whole rows and refuse column splits.
- Not changed: our CUDA mmvq applies 1.05/1.01 as `scale * ls * f` like ik's CUDA, while the CPU port folds the
  factor into the row scale like ik's reference. They differ only in the last bit.
- First full CI run failed clippy (`chunks_exact_to_as_chunks` in `kt_dequant.rs`); fixed with `as_chunks_mut`.
- Full CI: exit 0 (2406 + 2730 + 1 tests). The parity test now runs llama-perplexity in a temp dir, because ik's build writes llama.log into its working directory.

## Run 9 - 2026-10-02 21:40

Question: can ik_llama.cpp's IQK types load and run, including in its default mixes (which also unblocks its
default trellis mixes)?

The types are IQ2_K 137, IQ3_K 138, IQ4_K 139, IQ5_K 140, IQ6_K 141, IQ4_KS 144, IQ2_KS 145, IQ4_KSS 146, IQ5_KS 152,
IQ3_KS 156 and IQ2_KL 157. All use 256-element blocks.

Row scales: IQ*_K blocks carry an f16 scale. The `_KS` / `_KSS` / `_KL` types put one scale before each row: f32 for
IQ4_KS, IQ4_KSS and IQ5_KS; f16 for IQ2_KS, IQ3_KS and IQ2_KL.

IQ6_K discrepancy: ik's CUDA and CPU-GEMM kernels read IQ6_K values from `iq6nl_values`, whose second half is the
first plus one. Its reference `dequantize_row_iq6_k` instead evaluates the cubic that table rounds. We follow the
kernels.

Change:
- Row-scaled sizing generalized from the trellis types to every type with row metadata (2 or 4 bytes).
- `mmvq_rows.cuh` now holds the shared byte-stride matvec kernel. `mmvq_kt.cu` and the new `mmvq_iqk.cu` instantiate
  it.
- `mmvq_iqk.cu` is assembled from ik's sources: its 11 vec_dots and 11 dequant kernels, the value tables and
  helpers, copied verbatim by a brace-matching script, plus our launchers.
- CPU dequantizers (`iqk_dequant.rs`) ported from `dequantize_row_iq*_k*`.
- Prefill runs dequant plus a dense matmul, as for the trellis types.

Goldens:
- The ik golden generator (now `tests/fixtures/gguf_ik/`) also covers the 11 IQK types.
- First run: every IQK value came out 0. `GGML_FP16_TO_FP32` reads a table that `ggml_init` fills, so the script
  now calls `ggml_init` first.
- Result: ours matches ik's reference to the bit for 10 types. IQ6_K is within 1% of the row peak, the table
  rounding above.
- GPU: ik's CUDA code agrees with our CPU port (dequant within one rounding step; mmvq cosine > 0.999).

Real files: Qwen3.5-0.8B quantized with ik's default rules, no `--pure`. The mixes add IQ3_K, IQ4_K, IQ5_K, IQ4_KS,
Q2_K and Q6_K tensors. A default IQ2_KT mix is included too: 153 IQ2_KT, 27 IQ3_K, 7 IQ4_K.

First run failed: "gguf-raw does not support `embedding_forward`". ik's mixes store the tied token embedding in an
IQK type.
- Added `RawGgufTensor::embedding`. A small CUDA kernel (`gather_rows.cu`) gathers the selected rows' bytes on the
  device.
- Those rows are then dequantized on the GPU where the type has a dequant kernel, or on the host otherwise (the
  mainline IQ types).
- Covered by a unit test on CPU and CUDA.

Perplexity against ik's `llama-perplexity -c 512 --chunks 1 -ngl 99` (README text):

| File | Ours | ik_llama.cpp |
|---|---|---|
| IQ2_K | 30.070 | 30.375 |
| IQ2_KL | 19.887 | 20.049 |
| IQ2_KS | 48.366 | 48.445 |
| IQ2_KT (default mix) | 24.230 | 24.432 |
| IQ3_K | 11.542 | 11.529 |
| IQ3_KS | 14.219 | 14.195 |
| IQ4_K | 9.2229 | 9.1275 |
| IQ4_KS | 9.7094 | 9.7813 |
| IQ4_KSS | 10.329 | 10.330 |
| IQ5_K | 8.9767 | 9.0040 |
| IQ5_KS | 9.1433 | 9.1197 |
| IQ6_K | 8.8717 | 8.9436 |

All within 1.05% (worst: IQ4_K). The new test, `every_iqk_gguf_matches_ik_llama_cpp_perplexity`, reads
`INFERENCE_TEST_IQK_GGUF_DIR`.

Not done:
- mmq tiles for the ik types: measure prefill against ik first.
- ik's bitnet and `_R4` / `_R8` repacked types.
- MoE expert stacks.
- Metal.

Run 9 review follow-ups (22:10):
- Embedding lookups bounds-check ids. On the CPU an out-of-range id is an error. On CUDA the gather writes a zeroed row
  instead of reading past the weight, so no host sync is needed for a check.
- Empty id lists return an empty tensor before any launch.
- The embedding test now runs on CPU builds too and covers all 24 raw types, empty ids and an out-of-range id.
  Previously it was CUDA-only and covered 4 types.
- Known cost: mainline IQ types (IQ1_S through IQ4_XS) have no CUDA dequantizer. Their embedding rows go to the host
  for dequantization, which adds a sync on each decode step when the token embedding is one of them. ik's mixes use
  IQK types there, which stay on the GPU.
- Renamed trellis-only wording and constants to cover all row-scaled types. The archive row-size test now covers
  every `_KS` / `_KSS` / `_KL` id.
- Full CI: exit 0 (2408 + 2732 + 1 tests).

## Run 10 - 2026-10-02 23:20

Questions:
1. How do prefill and decode compare with ik_llama.cpp on the ik types?
2. Is ik's mmq worth porting?
3. Is the host round trip in mainline-IQ embedding lookups slow?

Setup: our dev-profile CLI (`target/debug/inference bench --prompt-len 512,2048,4096 --gen-len 128 --iterations 5`)
against ik's `llama-bench -p 512,2048,4096 -n 128 -r 5 -ngl 99`. Qwen3.5-0.8B on the RTX 3090. Our numbers are TTFT
throughput, which includes request overhead.

Rates in tok/s, prompt length in tokens:

| File | ik 512 | ik 2048 | ik 4096 | ik decode | ours 512 | ours 2048 | ours 4096 | ours decode |
|---|---|---|---|---|---|---|---|---|
| IQ2_KT (pure) | 16109 | 18118 | 17937 | 515 | 14816 | 14640 | 12650 | 584 |
| IQ4_KT (pure) | 15832 | 17588 | 17334 | 467 | 14509 | 14446 | 12604 | 536 |
| IQ2_K (mix) | 16302 | 17803 | 17773 | 540 | 13732 | 14537 | 12728 | 602 |
| IQ4_K (mix) | 16099 | 17681 | 17360 | 493 | 14396 | 14561 | 12590 | 542 |
| IQ2_KS (mix) | 16187 | 18049 | 17774 | 511 | 15419 | 14602 | 12650 | 588 |
| IQ4_KS (mix) | 16277 | 17724 | 17433 | 438 | 14456 | 14424 | 12571 | 547 |
| IQ4_XS (pure, Q8_0 emb) | 16155 | 18266 | 18030 | - | 20148 | 17755 | 14932 | 537 |

ik's 512-token runs are noisy (about +-5000).

Findings:
- Decode: ours is 10-25% faster on every ik type.
- Prefill on the ik types: we use dequant plus a dense matmul, and it is flat at about 14.5k across types. On
  IQ4_XS, which has an mmq tile, we reach 20.1k / 17.8k / 14.9k. So the dense path costs about 28% / 18% / 15%
  against mmq on this model, and porting ik's mmq for its types would recover that.
- The drop at 4096 appears on IQ4_XS too (14.9k ours vs 18.0k ik), so it is not a quant kernel. It is an
  engine-level long-prompt cost (attention / GDN) and a separate question.

Embedding round trip: two `--pure` IQ4_XS files, one with an IQ4_XS token embedding, one with Q8_0.
- Decode: 212.4 tok/s with the IQ4_XS embedding vs 527.5 tok/s with Q8_0, 2.5x slower. My estimate of "a few
  percent" was wrong.
- Isolated timings at full vocab (248320 x 1024): the IQ4_XS embedding lookup takes 0.018 ms and the tied-head
  mmvq 0.37 ms. Neither explains a 2.8 ms gap.
- nsys (`-t cuda,osrt`, 128 decode tokens x 2) shows the cause:
  - The IQ4_XS-embedding run made 220k `cudaLaunchKernel` calls, 948 `cuMemcpyDtoHAsync` and no `cuGraphLaunch`.
  - The Q8_0 run made 381 `cuGraphLaunch`.
  - The host copy in the embedding lookup cannot be captured, so decode loses its CUDA graph and runs eager.
- Fix: GPU dequantizers for the mainline IQ types (IQ1_S, IQ2_XXS / XS / S, IQ3_XXS / S, IQ4_NL, IQ4_XS), copied from
  llama.cpp `dequantize.cuh` at 4617ccc1a. IQ1_M already had one. With them every raw type now dequantizes on the
  GPU, and a unit test pins that. ggml's IQ4_NL kernel fills whole 256-element super-blocks, so
  its output is rounded up and trimmed. The weight's zeroed padding covers the extra reads.
- After: IQ4_XS-embedding decode is 531.5 tok/s, against 537.3 with a Q8_0 embedding.
- Review follow-ups: odd IQ4_NL element counts (for example one 288-element row) no longer fall back to the host; the
  embedding test covers 6 x 288. The dequant launchers share one `dequantize_superblocks` kernel template.
- Full CI: exit 0 (2408 + 2733 + 1 tests).

## Run 11 - 2026-10-03 00:30

Question: do ik's mmq loaders close the trellis prefill gap from Run 10?

How ik does it:
- ik's mmq expands trellis values into Q8_0-style int8 tiles, then reuses `vec_dot_q8_0_q8_1_{mma,dp4a}` with the D4 layout.
- Its `template-instances/mmq-instance-iq*_kt_id.cu` already use the current mainline loader signature
  (`load_tiles<mmq_y, need_check>`, `mmq_get_nwarps_device()`), so they drop into our port. The one difference: ik
  addresses rows by byte stride (`x + i*stride`), while ggml uses block stride.

Change:
- `mmq_byte_rows(type)` marks ik's types. `mul_mat_q_process_tile` then advances `x` by a 64-bit byte offset, and
  host code passes the row size in bytes as `stride_row_x`.
- `mmq_kt.cuh` holds the four KT loaders, copied by script, with `INT8_MMA_AVAILABLE` mapped to our MMA guard.
- IQ3_KT / IQ4_KT rows with tails keep the dense path (`fast_mmq::supports_shape`).

Prefill, tok/s (dev CLI, 5 iterations; the before row is Run 10's dense path):

| File | 512 | 2048 | 4096 |
|---|---|---|---|
| IQ2_KT before | 14816 | 14640 | 12650 |
| IQ2_KT mmq | 20380 | 17833 | 14983 |
| IQ4_KT before | 14509 | 14446 | 12604 |
| IQ4_KT mmq | 19391 | 17095 | 14558 |
| ik IQ2_KT | 16109 | 18118 | 17937 |

That is +38% / +22% / +18% on IQ2_KT, matching our IQ4_XS mmq numbers. We are now ahead of ik at 512 tokens and level
at 2048. The 4096 gap is the engine-level cost from Run 10.

Parity: all three perplexity suites pass. The KT numbers moved because mmq quantizes activations to Q8_1, as ik
does:

| File | Before | Now | ik |
|---|---|---|---|
| IQ1_KT | 160.29 | 162.15 | 160.40 |
| IQ2_KT | 33.28 | 33.82 | 33.39 |
| IQ3_KT | 11.54 | 11.53 | 11.64 |
| IQ4_KT | 9.737 | 9.702 | 9.735 |
| IQ2_KT default mix | 24.23 | 24.50 | 24.43 |

The largest drift is 1.3% (IQ2_KT).

Next: the IQK loaders. Their `_id.cu` files carry `_r4` / `_q8` variants, 16-element tile layouts
(`MMQ_DP4A_TXS_Q8_0_16`, `MMQ_MMA_TILE_X_K_Q3_K`) and more helpers, so they get their own PR.
- Review follow-ups: `mmq_byte_rows` lists the four KT ids instead of an id range, `row_stride` covers only KT until the IQK loaders land, and the provenance table lists `mmq_kt.cuh`.
- Full CI: exit 0 (2408 + 2733 + 1 tests).

## Run 12 - 2026-10-03 01:40

Question: do ik's IQK mmq loaders close the IQK prefill gap, the way the KT loaders did in Run 11?

Change:
- `mmq_iqk.cuh` holds 11 loaders, copied by script from ik's `mmq-instance-iq*_k*_id.cu`. Left out: the `_r4`
  (CPU-repacked) loaders and the `_q8` requant variants.
- The `_q8` variants re-quantize block-16 tiles to Q8_0 at large `mmq_x`. They need several more ik-only helpers
  (`get_int_from_table_16_q8`, `requant_int_q8`).
- Tiles:
  - IQ*_K: block-16 (`vec_dot_q8_0_16`; IQ6_K uses `vec_dot_q6_K` for mma).
  - `_KS` / `_KSS` / `_KL`: Q8_0 tiles.
- Byte row strides cover the IQK types too.
- The shared tables and helpers moved to `mmq_ik_common.cuh`, which `mmq_kt.cuh` now includes as well.

Prefill, tok/s at 512 / 2048 / 4096 tokens (dev CLI, 5 iterations; "before" is Run 10's dense path):

| File | Before | mmq |
|---|---|---|
| IQ2_K | 13732 / 14537 / 12728 | 19159 / 16934 / 14354 |
| IQ4_K | 14396 / 14561 / 12590 | 18531 / 16557 / 14004 |
| IQ6_K | - | 17926 / 16114 / 13770 |
| IQ2_KS | 15419 / 14602 / 12650 | 20333 / 17777 / 14988 |
| IQ4_KS | 14456 / 14424 / 12571 | 20089 / 17609 / 14787 |
| IQ2_KL | - | 19972 / 17487 / 14832 |

- The `_KS` / `_KL` types now match the KT and IQ4_XS mmq numbers.
- The IQ*_K types land about 5% lower. That is the block-16 path; ik's `_q8` requant wins there at large `mmq_x`
  (its traits switch at `mmq_x >= 40-48`). That is a follow-up.

Parity: both ik suites pass, all within 1%. Most IQK files moved closer to ik:

| File | Ours | ik |
|---|---|---|
| IQ5_K | 9.0076 | 9.0040 |
| IQ5_KS | 9.1166 | 9.1197 |
| IQ4_KS | 9.7993 | 9.7813 |
| IQ2_KS | 48.902 | 48.445 |
| IQ2_KT default mix | 24.462 | 24.432 |
- Review follow-ups: the MoE grouped paths take a `QTensor` (Candle types only), so ik types cannot reach them, but they now use `row_stride` too so a future raw type cannot get block units. `int_from_table_4` is `static`.
- Full CI: exit 0 (2408 + 2733 + 1 tests). The first rerun failed to compile: a local `row_stride` binding shadowed the helper, which is now `mmq_row_stride`.

## Run 13 - 2026-10-03 06:30

Question: where does the remaining gap to ik at 4096 prompt tokens come from? It shows on IQ4_XS too (14.9k vs 18.0k
tok/s), so it is not a quant kernel.

Setup: `nsys profile -t cuda,osrt`, then `nsys export --type sqlite` and a query of `CUPTI_ACTIVITY_KIND_KERNEL`.
- Ours: `inference bench -f pure-IQ4_XS-q8emb.gguf --prompt-len 4096 --gen-len 0 --iterations 3`.
- ik: `llama-bench -p 4096 -n 0 -r 3 -ngl 99`, same file.
- Totals are over the whole run (warmup plus 3).

Two traps along the way:
- `local_ci.sh --sweep` deletes the dev CLI's shared kernel libraries, so rebuild the CLI after CI before profiling.
- `nsys stats` reads a stale SQLite export unless given `--force-export=true`.

Kernel time by category:

| Category | Ours | ik |
|---|---|---|
| mmq (`mul_mat_q` + stream-k fixup) | 340.9 ms | 199.5 ms |
| GDN recurrence | 245.6 ms | 252.2 ms |
| copies and casts (`ucopy_bf16`, `cast_f32_bf16`, `cast_bf16_f32`) | 127.7 + 74.8 ms | 9.7 ms |
| attention | 54.7 ms (`softmax_f32` 51.0) | 20.6 ms (`flash_attn_mma_ext_f16`) |
| cuBLAS bf16 GEMM | 33.6 ms | 0 |
| total | 921.9 ms | 551.0 ms |

Findings:
- GDN is not the gap: about equal time. We launch 3.1x more recurrence kernels for the same total.
- mmq: about the same cost per call (ours 43.5 us vs ik 38.8 us on the `(82,1,1)` stream-k grid), but 1.55x as many
  calls (5952 vs 3843). Both use 512-token prefill chunks (our `DEFAULT_MAX_PREFILL_CHUNK_TOKENS`, ik's ubatch), so
  chunking is not the cause. More likely we run separately what ik keeps fused (QKV, gate/up), or we route some
  projections differently. The bf16 cuBLAS GEMMs are projections ik runs through mmq.
- Copies and casts are 22% of our kernel time (2% for ik). The large `ucopy_bf16` grids (6144, 7168, 8192, 12288,
  14336, 16384 blocks) match activation widths in the GDN and MLP blocks, so they look like `contiguous()` after
  splits or transposes, plus f32/bf16 round trips around the f32 recurrence.
- Attention: chunked prefill uses matmul plus `softmax_f32` (198 calls, 51 ms) rather than flash attention for the 6
  full-attention layers.

Implication: the 4096-token gap is in the Qwen3.5 model path (copies, launch count, the prefill attention kernel),
not in quantization. Each is its own optimization. Copies and casts are the biggest: about 200 ms of 922.

## Run 14 - 2026-10-03 06:50

Question: which call sites produce Run 13's copies and casts?

Attribution: gdb with Python breakpoints on Candle's `CudaStorage::copy_strided_src` and `to_dtype` (quoted trait-impl
names; raw addresses fail under PIE). Each hit records the first two `inference_*` frames. nsys CUDA backtraces need
CPU sampling, which `perf_event_paranoid` disables here. Run: 2048-token prompt, 1 iteration, 1084 hits.

- `repeat_kv` in `Sdpa::run_attention_noflash`: 420 copies (GQA 2 -> 8 heads, K and V).
- `run_attention_noflash` casts and copies, plus the paged-prefix gather (`prefix_gather_causal_mask`,
  `prefix_attention_output_layout`): about 230.
- `finish_recurrence` f32 -> bf16 cast: 144.
- The rest is load-time (norm and GDN weight casts).

The attention path was the non-flash one because the CLI under test was built with `--features cuda` only. The
standard CUDA build (CLAUDE.md) is `cuda flash-attn cudnn`. That build mistake invalidates Run 10's and Run 13's
long-prompt conclusions: ik was measured with flash attention and we were not.

Re-measured with `cargo build -p inference-cli --features "cuda flash-attn cudnn"`. tok/s at 512 / 2048 / 4096
prompt tokens, then decode:

| File | Ours | ik (Runs 10-11) |
|---|---|---|
| IQ4_XS (Q8_0 embedding) | 19854 / 21771 / 22524, 534 | 16155 / 18266 / 18030 |
| IQ2_KT | 19016 / 21686 / 22453, 584 | 16109 / 18118 / 17937, 515 |
| IQ4_KS | 21101 / 22609 / 22418, 553 | 16277 / 17724 / 17433, 438 |

Corrected conclusion: with flash attention we are 20-28% faster than ik at 2048 and 4096 tokens on every ik type,
and faster at 512 too. There is no engine-level long-prompt gap. What Run 13 measured was the non-flash fallback:
`repeat_kv` copies, casts and a separate softmax.

Still true for builds without flash-attn: the fallback's `repeat_kv` expands K and V per prefill chunk. That only
affects non-flash builds, and the default CUDA build includes flash-attn, so it is not worth optimizing now.

Benchmark rule: build the CLI with the full CUDA feature set before comparing against llama.cpp or ik.
