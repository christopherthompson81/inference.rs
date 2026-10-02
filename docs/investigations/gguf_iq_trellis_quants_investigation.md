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
