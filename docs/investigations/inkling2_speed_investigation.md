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

## Run 8 - 2026-11-22 09:00

Recipe C (B with head and embedding at Q6_K, the preferred recipe) on the tensor-core attention, against B in the same
session:

```
          pp512   pp2048   tg128   KLD
recipe B  2191    1894     98.5    0.00368
recipe C  2209    1899     98.9    0.00387 (same as before the kernel)
```

C's prefill is level with B (the head is one matmul a prompt); its smaller head reads give decode +0.4 tok/s.

## Run 9 - 2026-11-22 10:30

Closing the NVFP4 prefill matmul's gap to IQ4_NL (recipe C, pp512, nsys over 5 prefills,
`ink_pp_prof.sh`):

```
                      ms a prefill   us a launch
NVFP4 mmq (C)         104            786
IQ4_NL mmq            83             630
```

The prefill gap between the two GGUFs (21 ms) is all the expert mmq. Both are 4.5 bits a weight and both expand to
int8 in shared memory; NVFP4 took the 16-wide-scale path (`vec_dot_q8_0_16_q8_1_mma`, Q3_K's), so it ran half-K
mma steps and twice the scale products of IQ4_NL's per-32 path.

The fix folds each pair of 16-wide UE4M3 scales into one f32 per 32 in `load_tiles_nvfp4`, so NVFP4 takes the Q8_0
tile and dot product. The larger scale's 16 map e2m1's 12 (6.0) to 120, which is exact. The other 16 are code x 10 x
(small / big), rounded within int8, which isn't exact.

```
                                                          pp512   KLD
before (per-16 path)                                      2209    0.00387
fold, every value scaled in floats                        2071    0.00389
fold a no-op (wrong values, the Q8_0 path's floor)        2568    -
fold on the 16-entry table (exact table for the larger)   2348    0.00389
+ int4 smem stores, __fdividef                            2342    0.00411 (fast divide moves .5 ties)
thread a 16 (coalesced rows), a table each                2328
thread a 32, exact divide                                 2361    0.00389
IQ4_NL, same session                                      2453
```

ncu, one launch at base clocks (a thread a 64, before the remap, against IQ4_NL): 803 against 750 us, the same
instructions (165M against 162M) and the same occupancy (1 block, 168 registers, 48 KiB). But `lg_throttle` was
34.5K against 14.6K and global load sectors 68M against 26M, because each thread read a whole 36-byte block, so a
warp's load touched ~32 scattered sectors. A thread a 16 cut the sectors to 24M and lg_throttle to 1.1K, but building
a table per 16 added 23M instructions (819 us). A thread a 32 (one small-scale table per group, the constant table
for the larger) keeps most of both.

Result: recipe C pp512 2209 -> 2361 (+7%), pp2048 1899 -> 1993, tg128 unchanged (the matvec keeps its per-16
scales), KLD 0.00389 against 0.00387 (noise). Against IQ4_NL the gap is 3.8% (was 9%).

Test `nvfp4_mmq_folds_each_scale_pair_into_one` takes 33 rows through mmq against each row alone through the matvec
(per-16, unfolded), with F32 in and out (BF16 output rounding alone is 2e-3). Equal pairs give 2.2e-4, the same as the
unfolded kernel on either weight, so that's the two kernels' own difference. Unequal pairs give 3.3e-3 relative, the
fold's rounding.

## Run 10 - 2026-11-22 11:30

Recipe C on master after #1046-#1047, serving with Run 3's method (`inference serve`, greedy, 256 tokens a
completion, wall time a request including its prompt; the official drafter, `--mtp-model`):

```
recipe C (tok/s)       prose   code    story   explainer
plain                  96.7    96.7    96.3    96.3
drafts 1               87.1    110.0   88.6    95.1
drafts 2               85.1    127.1   80.9    93.5
drafts 3               76.0    127.8   74.1    85.6

recipe B, Run 5        (plain ~95.6)
drafts 3               78.8    126.7   77.0    88.2
```

Plain is up ~1 tok/s from B (the smaller head) and the prompts' prefill. Speculation is where Run 5 left it: it pays
only on code (+32% at drafts 2-3) and loses elsewhere at every depth, even at one draft (-10% on prose). The prefill
work doesn't reach the verify, which runs the 4-row grouped matvec (Run 5: ~3.6x a decode step's expert bytes).

## Run 11 - 2026-11-22 12:30

Where a speculative cycle goes (recipe C, drafts 3, the prose prompt, one served request of 256 tokens under nsys
`--cuda-graph-trace=node`; 131 cycles, 1.97 tokens a cycle, accepted per position 88 / 35 / 18 of 146):

```
a cycle                         ms      (a plain decode step: 10.1)
wall                            28.4
GPU busy                        24.1
  verify graph                  21.4
    NVFP4 expert matvec (4 rows)  8.55   (89 launches, 96 us; decode ~2.5)
    Q6_K / Q8_0 experts 20-23     1.14
    Q8_0 dense matvec (4 rows)    2.21
    loop controllers              1.56   (decode 0.81)
    cast_bf16_f32                 1.38   (307 launches)
    copies (copy2d, ucopy)        1.33   (~670 launches)
    attention (decode_kernel)     0.86
    depth mix                     0.84
    head (mmvq_tc_k)              0.52
  drafter, eager                2.75   (206 launches: head 0.52, argmax 3 x 134 us, cuBLAS gemv 3 x 122 us, gemms 0.68)
host gaps                       4.3    (3.3 before the verify graph's launch)
```

The experts (9.7 ms) are the floor: 4 rows x 8 routes reach ~29 distinct experts. The rest (~7-8 ms: the gaps, the
drafter's eager pass, the casts and copies, the controllers past one row) doesn't follow the bytes.

The published setup (vLLM `models/inkling2/nvidia/dspark.py`, `v1/worker/gpu/spec_decode/dspark/speculator.py`) is
a linear chain: one argmax a position, `num_speculative_tokens` forced to `block_size - 1` (7), every step, rejection
sampling in the target's vocabulary. No tree, no confidence stop (the generic DSpark speculator stores a confidence
head's scores, but this checkpoint has none: `mtp.safetensors` holds `fuse_weight`, `mask_embed`, `norm`, the layers
and the `markov_head`). It suits batched serving, where a step reads most experts anyway and draft rows come nearly
free. At one sequence each row brings its own experts.

## Run 12 - 2026-11-22 13:15

Run 11's host gap was mostly the profiler: node-level graph tracing inflates `cuGraphLaunch` (2.95 ms a launch).
Graph-level tracing (`--cuda-graph-trace=graph`) gives the real cycle: 25.7 ms (3.46 s for 131 cycles, the
unprofiled run's 3.37 s): verify graph 21.3, drafter 2.8, GPU idle 1.9 (1.08 of it from the drafts' readback to the
verify graph's start, a launch costing ~0.9 ms of host for ~3,200 nodes). Plain decode's launch (0.44 ms host) overlaps
the previous step: 0.07 ms idle a step. A verify can't launch until the host has the drafts, so its launch is exposed.

Per kernel, the verify graph (4 rows) against a plain decode graph (1 row), us a step (node-level, GPU times):

```
                                 plain    verify   ratio
NVFP4 expert matvec              2530     8551     3.4   (expert bytes)
Q6_K / Q8_0 experts 20-23         303     1136     3.7   (expert bytes)
cast_bf16_f32                     184     1380     7.5   (113 -> 307 launches)
copies (copy2d, ucopy f32/bf16)   260     1333     5.1   (new in verify: ucopy_bf16 185, ucopy_f32 98 a step)
loop controllers                  748     1556     2.1
attention (decode_kernel)         461      861     1.9
depth mix, conv, norms, router   ~2.5 ms  ~2.5 ms  1.0
graph total                     10524    21358
```

Not bytes, ~3.5 ms: the casts, the copies (likely the rollback's stash of each layer's conv inputs and the taps), the
controllers and attention past one row. The drafter's 2.8 ms: its three dense layers' BF16 weights (~0.64 GB a pass,
cuBLAS gemms 0.68 ms), the target head over the draft rows (0.52), the recurrent head's fold (`[201024, 256]` BF16,
103 MB, 122 us a draft) and an argmax over the one 201K row (134 us a draft where 0.8 MB should take a few us).

## Run 13 - 2026-11-22 14:30

The verify's conv stash and the conv states' round trip.

Step 1: the stash kept each conv input as its BF16 (no cast to F32), and the rollback kernel (`conv_rollback_kernel<I>`)
widens it, which is exact. Casts went from 1460 to 562 us a cycle (322 to 129 launches, plain's level). Drafts 3 prose
went 76.0 to 77.5, the others within noise. Test `cuda_conv_rollback_takes_either_stash_dtype`.

Step 2: per layer visit, decode and verify gathered the slot's states (`index_select`, with a clamp), ran four convs
on the copy, concatenated their four new states (4 `copy2d_f32`), and scattered them back (`indexed_row_copy`). A verify
also concatenated its stash (4 `copy2d_bf16`). These are ~1.4 us kernels, mostly launch cost. The CUDA pool kernel
(`short_conv_pool_kernel`, like Vulkan's in-place step) takes a thread per (row, channel): it reads the state from the
row's slot at the conv's channel offset into a register window, runs the rows, and writes the new state back into the
slot. In a verify it also writes the state as read and the inputs into the rollback's buffers (`ConvRollback`). A slot
past the pool (a padded row) reads zeros and writes nothing. Decode and verify now take the pool on CUDA. Test
`cuda_pool_conv_matches_the_gathered_rows` (outputs, new states, before, stash, the other channels and slots left; t 1
and 4, with and without taps, a padded row). The tiny model's decode-against-prefill and drafter tests pass.

```
recipe C                         tg128   served plain   drafts 3: prose   code    story   explainer
Run 10                           99.0    ~96.5                    76.0    127.8   74.1    85.6
step 1 (BF16 stash)                                               77.5    128.8   73.3    84.4
step 2 (pool kernel)             103.0   ~99.2                    75.7    136.0   76.2    89.2
```

Plain decode +4% (each visit loses its gather, concatenation and scatter). Speculation gains on code (+6%), story and
explainer; prose is level (its greedy text and drafts moved: accepted 280 / 156 / 101 against 286 / 163 / 102).

## Run 14 - 2026-11-22 15:30

The drafter's recurrent head ran, a draft, a cuBLAS gemv over the fold (`[201024, 256]` BF16, 103 MB, 156 us), a
generic one-block argmax over the 201K row (`fast_argmax_f32`, 137 us), and ~8 vocabulary-wide elementwise kernels
(the casts, the bias, the softcap). One fused pick (`inkling_markov_pick`) does it in two stages: a warp a vocabulary
row takes `logits[v] + fold_b[v] + fold_w[v] . latent` (the latent in shared memory, one 16-byte load a lane), each
block keeps its best, then one block a batch row reduces the partials. Ties go to the lower index, as the argmax.
The softcap keeps the order, so the pick skips it. The bias sums in f32 where the matmul rounded it to BF16, so a
draft may differ at a near tie. Test `cuda_markov_pick_takes_the_biased_argmax` (f32 and BF16 logits, a strided
step, padded rows above every real one).

A cycle, graph-level trace, the prose request at drafts 3:

```
                         cycle   verify graph   drafter   idle   cycles for 256 tokens
Run 12                   25.73   21.28          2.82      1.89   132
Runs 13-14               23.72   20.20          2.29      1.45   143
```

Prose tok/s hides the cycle's 8%: its greedy text moved (a verify's rows change with the drafts and flip near ties),
so it took 143 cycles. Served (tok/s):

```
                 prose   code    story   explainer
drafts 3         74.0    136.5   77.0    89.9     (281 / 155 / 100 accepted)
drafts 2         86.1    135.1   85.5    101.8    (317 / 177)
plain            ~99.2
```

At drafts 2 the explainer passes plain and code keeps drafts 3's speed.

The drafter's pass stays eager: the DFlash host graphs a drafter only over its bounded windowed KV pool
(`enable_windowed_kv`), which Inkling's doesn't take (layer 1 is global), so it runs the paged path with no graph. Its
~200 launches leave gaps, and the verify's launch (0.54 ms of host for ~2,650 nodes) waits on the drafts' readback.

## Run 15 - 2026-11-22 16:45

Loop controllers at 4 rows (1.64 ms a verify against decode's 0.75): a down kernel summing 4 rows a pass over its
weights (one sync pair a tile) and an up kernel of 8-lane groups a row (3 shuffles a reduction). Per launch, node
trace:

```
                      decode down / up (us)   verify down / up
before                3.46 / 4.43             7.86 / 8.96
row tiles + groups    3.90 / 4.64             6.92 / 8.55
```

A verify saves ~0.13 ms and decode loses ~0.06 ms a token, so I reverted it. The controllers are 192 small launches a
token, bound by launch latency more than by their loops. The test gained a 7-row case (a second tile, a part-filled
group), which the kernels as they are pass.

Long context. `inference bench` decodes at depth 4 whatever the prompt ("@ d4"), so served requests measure it
(`avg_compl_tok_per_sec`, 256 tokens, `--pa-context-len 16384`):

```
plain, context (tokens)      8       5689    9824
before                       109.6   82.2    75.6
key splits, a combine        100.3   89.1    84.9
adaptive splits              109.5   89.6    89.1
drafts 2 (adaptive)          91.5    93.8    116.3    (summaries copy the text: 161 / 182 accepted at 9824)
```

The decode attention ran a 1,024-thread block a (query, head): 24 blocks a decode step on 82 SMs whatever the context.
Splitting the keys over blocks (`decode_splits`: 2 blocks an SM's worth of rows, at most 8, from the shapes alone so a
captured graph keeps it) with a combine kernel helped long contexts but cost short ones ~9% (a launch a visit and
near-empty splits). Now a row takes a split per 512 of its keys (`DECODE_SPLIT_KEYS`), the grid's other blocks
exiting at once. A row of one split writes its output as before. Several splits write (max, sum, values) partials,
and the last to finish (a per-row count, zeroed once per device and reset by that block; `__threadfence` and
`__syncwarp` before the count, `__ldcg` reads) merges them. No combine launch. `__launch_bounds__` keeps the
1,024-thread kernel within its registers (without it, CUDA status 701). Test
`cuda_split_decode_over_long_pasts_matches_the_reference` (2,000 past keys, a 1,500-key window, the log scale, 4 rows
and 1, view and paged BF16).

At 9,824 tokens the attention is 47 us a call (2.2 ms a token): a global layer's BF16 K/V at that depth is ~40 MB a
token (9824 x 8 KV heads x 128 x 2 x 2 bytes), the 3090's bandwidth for 47 us. What's left is the read itself: a
quantized KV cache would halve it.

## Run 16 - 2026-11-22 17:45

The verify's launch waited on the drafts. A CUDA API backtrace on the readbacks (`nsys --cudabacktrace=memory:10`,
`CUDA_CALLCHAINS`) put 14 of a request's DtoH copies in `DFlashDraftModel::finish_proposals`. Its greedy branch with no
selector read the picks to the host (`to_vec2`), so the host waited out the drafter before staging the tokens and
launching the verify graph. The engine already packs device draft rows into a verify's input (`pack_completion_input`,
which the graphed DFlash path's `DeviceTokens` takes). That branch now returns `DeviceTokens` on CUDA, so the host
stages them and launches without waiting:

```
drafts 3, prose, graph-level   cycle   verify graph   drafter   idle
Run 14                         23.72   20.20          2.29      1.45
device drafts                  22.94   20.16          2.30      0.70
```

Served: drafts 2 87.0 / 136.4 / 86.2 / 102.8, drafts 3 77.4 / 139.7 / 78.3 / 91.4 (prose / code / story / explainer).

On the way: `cuda_fused_controller_matches_the_reference` failed once at 4 BF16 rows (0.027 against a 2e-2 bound) on
unseeded random inputs. The kernel's BF16 output lands within a step of the f32 reference, and a step at the draw's
tails (|v| ~5-7) is ~0.02-0.03. The bound is now two BF16 steps of each element (2^-8 of the value each) plus 1e-4.
It passed six runs in a row.
