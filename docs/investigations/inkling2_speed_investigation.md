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
