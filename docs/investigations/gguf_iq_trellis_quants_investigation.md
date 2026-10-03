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
