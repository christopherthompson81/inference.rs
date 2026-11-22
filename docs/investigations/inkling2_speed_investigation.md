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
