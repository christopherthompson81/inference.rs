# Inkling 2 speed

Recipe B (`Inkling-2-Mini-NVFP4-B.gguf`, NVFP4 experts 0-19, Q6_K/Q8_0 experts 20-23, Q8_0 dense; see
`inkling2_official_weights_investigation.md`) on the 3090. Bench: `inference bench --pa-context-len 8192 --format gguf`,
pp512 / tg128. Profiles: nsys `--cuda-graph-trace=node`, kernel sums.

## Run 1 - 2026-11-22 00:30

Where decode goes (`--prompt-len 16 --gen-len 128`, ~510 decoded tokens in the profile, 12.9 ms a token):

```
kernel                                  ms a token   launches a token
mmvq_gguf_nvfp4_moe_cuda                2.88         132   (44 NVFP4 visits x gate, up, down)
mmvq_gguf_q8_0_bf16_plain               1.91         192
depth_mix_kernel (bf16 + f32 out)       1.94         122   (~16 us each, one block a row)
controller_in + gemv (loop controllers) ~1.0
decode_kernel (attention)               0.32
rms_norm, quantize_q8_1, copy2d         ~0.8
```

Prefill (pp512): NVFP4 mmq 41%, attention `prefill_kernel` 22%, `ucopy_f32` 6%, depth mix 4%, `bmul_f32` 3%.

The depth mix at decode is one block for its one row: a latency-bound pass over up to 13 states of 2304, 65 block
reductions in series, a second pass to write. Split for <= 16 rows: block (row, chunk of 128 values) writes its
partial sums of squares and query dots to a scratch (`inkling_depth_mix_scratch` sizes it), a second kernel per
(row, chunk) reduces the row's partials, takes the softmax and writes its chunk. Many rows keep one block a row.
Test: `cuda_depth_mix_matches_the_reference` (1 and 5 rows split, 40 one block a row, against the CPU).

```
                      pp512   tg128   mix kernels in the decode profile
before                1531    81.7    988 ms
split, chunk 256      1547    89.1    491 ms (partials 4.3 us, apply 3.7 us)
split, chunk 128              90.1    (90.3 / 89.9)
ggml-org IQ4_NL       1913    83.6    (before the split)
```

Next: the softcapped SwiGLU is no activation the fused kernels take, so every NVFP4 decode visit is three matvec
launches plus `softcap_silu_mul` and copies, and prefill takes the explicit path (`ucopy_f32`, `bmul_f32`).

## Run 2 - 2026-11-22 01:30

The softcapped SwiGLU fused: `MoeGlu` (an activation, or `SoftcapSilu { gate_cap, up_cap }`, kernel code 5 with its
caps beside it) through the expert matvec (`mmvq_gguf.cuh` and `mmvq_rows.cuh` MoE entries) and the grouped GLU
quantize (`quantize_mmq_q8_1_glu`, whose own copy of the activations went for `glu_activation.cuh`'s).

```
                          pp512   tg128   KLD
split mix (Run 1)         1548    90.1    0.00381
+ fused softcapped GLU    1647    89.4    0.00368 (inside the floor)
```

Prefill +6% (the explicit path's `ucopy_f32` / `bmul_f32` gone). Decode level: per NVFP4 visit the separate gate and
up (~18 us each) plus `softcap_silu_mul` (1.6 us) became one fused launch of 38.2 us, the down 29.2 us either way.
The down reads the gate's bytes in 1.6x its time: its rows are 18 NVFP4 blocks, which at `qi / vdr` = 2 threads a
block keep 36 of 128 threads busy (IQ4_NL's 72).

But the vec dot's per-16 `ggml_cuda_ue4m3_to_fp32` (branches and `ldexpf`) was the cost, not the idle threads. The f32
bits built directly (exponent `exp + 119`, mantissa `man << 20`; subnormals `man * 2^-10`; 0x7F zero), exact for
every byte (test `cuda_nvfp4_scales_decode_as_the_cpu_for_every_byte`, all 256 against the CPU bit for bit), in one
header (`ue4m3.cuh`) for the matvec and mmq:

```
                          pp512   tg128
matvec decode             1652    93.8
VDR_NVFP4_MMVQ 4 -> 2             93.9   (no gain once the decode is cheap: reverted)
+ mmq decode              1749    93.8
ggml-org IQ4_NL           1932    93.8   (same build: the split mix and the fused GLU help it too)
```

Recipe B since Run 1: pp512 1531 -> 1749 (+14%), tg128 81.7 -> 93.8 (+15%), KLD unchanged. NVFP4 decode is now level
with IQ4_NL; prefill trails by 10% (was 20%).

`--size`: the baseline (#998) was 1.16 MiB behind master already (master alone fails: ffi `.text` +0.86 MiB from the
merges since); this branch adds 0.04 MiB. Baseline rewritten.

## Run 3 - 2026-11-22 02:00

Plain against speculative after Runs 1-2, as the drafter's Run 5 measured them (`inference serve`, greedy, 256 tokens a
completion, wall time a request including its prompt; the official checkpoint's drafter, `--mtp-model`):

```
recipe B (tok/s)       prose   code    story   explainer   accepted / drafted
plain                  85.0    86.6    91.2    91.2
drafts 1               82.1    103.7   80.7    88.9        397 / 625   (0.64)
drafts 2               83.2    120.2   76.1    90.8        501 / 1042  (per position 314, 187)
drafts 3               76.7    125.6   70.9    85.4        557 / 1392  (per position 280, 169, 108)

IQ4_NL at #1041 (drafter Run 5, before Runs 1-2)
plain                  76.5    79.0    80.9    81.0
drafts 3               69.3    122.9   74.7    83.4
```

Plain is ~11% faster than at #1041; speculation pays only on code (+45% at drafts 3), and loses elsewhere at every
depth: the verify step's cost did not fall as much as the plain step's.

## Run 4 - 2026-11-22 02:45

The loop controllers fused for <= 16 rows (decode, verify; rank <= 128): ~6 launches a controller (`controller_in`,
two f32 gemvs, two narrow copies, `controller_out`), 96 controllers a token, ~1.6 ms. The unit norms are scalars, so
`down . [x / |x|, m / |m|]` is `ix (down_x . x) + im (down_m . m)`: a down kernel of block (rank output, hidden slice,
row) writes each slice's two dots and two sums of squares; an up kernel sums the slices into the bottleneck as it loads
it, then a warp an index takes the gain's and forget's rows (lanes over the rank, each row read whole) and writes
`(1 - tanh(forget)) x + tanh(gain) m` in x's dtype. Test `cuda_fused_controller_matches_the_reference` (1 row f32, 4
rows bf16).

```
                                               tg128   controllers (ms a token)
before (Run 2)                                 93.8    ~1.6
fused, up kernel a thread an index (9 blocks)  86.0    (slower: uncoalesced rows, 9 SMs)
fused, up kernel a warp an index               98.3    0.81 (down 0.48, up 0.33)
+ down split over 4 hidden slices              98.75

serve, greedy, 256 tokens (tok/s)   prose   code    story   explainer
plain                               95.8    95.7    95.4    95.6
drafts 3                            78.7    128.4   76.9    88.4     564 / 1374 accepted
```

Greedy text changes on some prompts (summation order against cuBLAS's gemv, the floor's kind of difference).

## Run 5 - 2026-11-22 03:15

The verify step (drafts 3, prose, one served request under nsys; the trace caught ~21 of the run's verify steps):
per NVFP4 visit the 4-row grouped matvecs take 112 + 100 us against a decode step's 32 + 25 (4 rows x 8 routes reach
~29 distinct of 128 experts, ~3.6x the expert bytes); the fused controllers at 4 rows 10.5 + 7.4 us against 3.4 +
4.9 at one, each row's blocks rereading the weights. Rows moved inside the blocks (each weight read once, the up
kernel's gain/forget rows kept in registers across the rows):

```
                    tg128   prose   code    story   explainer   (serve, drafts 3)
Run 4               98.75   78.7    128.4   76.9    88.4
rows in the blocks  99.0    78.8    126.7   77.0    88.2
```

No speculative gain: the verify is bound by its expert reads, which follow the routing, not the kernels. Drafting
pays on this MoE only where acceptance is high (code).

## Run 6 - 2026-11-22 05:00

The NVFP4 prefill matmul: per launch ours 1.031 ms, llama.cpp's 0.989 (official weights Run 6), at different tile
widths: our grouped mmq chose its width (`mmq_x`) from `ncols_max`, the token count (512: the 128-wide tile), where an
expert sees 512 x 8 / 128 = 32 routes on average; llama.cpp takes the average (J 32). Ours pads each expert to whole
tiles (llama.cpp packs experts' rows into shared tiles), so each extra tile rereads the expert's weights. The width
as a multiple of the average routes an expert sees (the grid still covers `ncols_max`):

```
pp512                     Qwen3.6-35B-A3B recipe-v2   Inkling 2 recipe B
ncols_max (before)        4486                        1749
1x average                3718                        1893
1.5x                      4449                        1827
2x                        4868                        1858
3x                        4639                        1801
4x                        4743                        1745
8x                        4486                        1746

pp2048                    2x 4762 (before 4400)       2x 1265 (before 1230)
```

Twice the average taken (`MMQ_MOE_TILE_AVERAGES`): Qwen pp512 +9%, Inkling 2 recipe B +7% (1868), its IQ4_NL +5%
(2023); KLD unchanged. NVFP4 trails IQ4_NL by 8% at prefill.

A test found on the way (`grouped_mmq_spans_experts_over_several_tiles`: 40 tokens over 2-8 experts, every route as
when its token runs alone): every type's sums are width-independent but ik_llama's IQ2_K, IQ3_K, IQ4_K and IQ5_K,
which requantize their weights to Q8_0 rows past a width (`mmq_requant`, 48/40/32/48): ~0.4% off the narrow tiles'
sums, both 0.6-0.7% from the dequantized weights. The offload test's staged path (fewer experts a pass, so wider
tiles) crossed IQ3_K's 40 against the resident path's 16; both tests bound those types by the requantization.

## Run 7 - 2026-11-22 07:30

Prefill after Run 6 (pp512, nsys): the NVFP4 mmq 0.785 ms a launch against IQ4_NL's 0.634 (1.24x, llama.cpp's 1.34x),
~80 ms of a 1107 ms prefill; the attention (`prefill_kernel`) 1.52 ms a call, 26% of prefill on either GGUF. ncu: 40%
of the SMs, 2% of DRAM, scalar f32 with 8 queries a block (a 32-key tile reused 8 times, a shuffle loop a key):
~3% of the tensor cores' rate for its ~3 GFLOP. llama.cpp's fattn (inference-fattn) cannot take the bias: its mask
is per sequence, and its mma kernel packs GQA heads into a tile that shares one mask.

`prefill_mma_kernel` (sm80+, prompts from 32 queries, half-precision caches): a block 64 queries of one head, a warp 16;
each 64-key tile's K and V^T in shared memory as f16 (a BF16 cache's values exactly); S = Q K^T and O += P V by
`mma.sync.m16n8k16` with f32 sums; the bias by distance, the log scale, the causal and window masks and the sink as
the scalar kernel; softmax online.

```
                                         pp512   pp2048   KLD
scalar (Run 6)                           1868    1265     0.00368
mma, q and P in f16                      2218             0.00348
mma, q and P each f16 + its remainder    2209    1898     0.00368
IQ4_NL                                   2432             (2023 before)
```

f16 q and P alone moved the tiny model's pinned greedy traces (a step 0.21 in logprob) and its long prompt against
the CPU (0.003): the scalar kernel with the same f16 rounding moves them the same way (-7.24594 against the mma's
-7.24590, CPU -7.24290), so precision, not a bug. Each as an f16 part and its f16 remainder (two mma), and an f32
cache left to the f32 kernel (its K and V are not exact in f16), and every test passes at its tolerance; the
attention is 0.576 ms a call (2.6x). Tests: tensor-core cases at head sizes 32, 64 and 128 with a past, the log
scale, a window and paged BF16 caches, against the reference at 1e-3.
