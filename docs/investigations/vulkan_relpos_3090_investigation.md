# Parakeet's relative-position attention on a discrete NVIDIA GPU (Vulkan)

Follow-up to #859 (Parakeet and Nemotron-3 diarization on Vulkan, tuned on a Radeon 780M iGPU): on the RTX 3090 the
fused `flash_attn_relpos_cm` kernel made ASR slower than turning matrix cores off, and the faster blocked path with
matrix cores changed words. Clip: 605 s of synthetic Kokoro speech (one long file, full attention over 7,564 frames).

## Run 1 - 2026-11-07 23:10

Question: where does the 3090's default (matrix cores, fused relpos) ASR time go on master (04da9fa15 + #860)?

`inference from-config` with Parakeet on `vulkan = true`, warm-up, then the clip with word timestamps; word diffs
against the CPU's f32 run (`difflib` over the words, plus the largest start shift among matched words):

```
default (matrix cores, fused relpos):   asr 7.03 / 7.02 s, diar 1.00 s, both 7.56 / 7.46 s; 0 word diffs, 80 ms max shift
INFERENCE_VULKAN_NO_COOPMAT=1:           asr 4.77 s, diar 1.57 s, both 5.75 s; 0 diffs, 0 ms
(#859 A/B) matrix-core matmuls, blocked attention: asr 2.75 s; 4 diffs (fifteen -> fifty x3, gnawed -> nod)
```

`INFERENCE_VULKAN_PROFILE=1` over a 5 s window: `flash_attn_relpos_cm` 2,482 ms of 2,836 ms GPU time (3.4 ms a
dispatch); the dense projections run at ~40 TFLOPS on `matmul_f32_cm`.

A benchmark at one layer's shape (`relative_position_attention_throughput`, 7,564 x 8 heads x 128, `#[ignore]`):

```
master kernel: 222.8 ms a call, 1.6 TFLOPS useful (content, position and value products)
```

## Run 2 - 2026-11-07 23:30

Question: is it the kernel's loads from global memory (every subgroup loading its own K/V/P fragments, V in the
layout NVIDIA's mma B operand does not take natively)?

Benchmark after each change (the parity test `relative_position_attention_matches_its_definition` passes on each):

```
K and V tiles staged in shared memory, once a workgroup, 16 bytes a thread's load:   147.7 ms
  ablation, no position products at all (wrong output):                                57.4 ms
P spans staged too, a half-workgroup at a time (64 rows, two phases; one 64-row region
  takes K, each half's P span and V in turn: 45.3 KB of the 3090's 48 KB):            127.6 ms
f16vec4 shared stores (two a thread's 8 halves):                                      125.0 ms
```

Ablations on the staged kernel (ms saved off 127.6, wrong outputs): PV products 11, QK 13, QP products 38, staging
P 37; removing the rescale branch made it slower (179 ms; exponentials overflow without it). The useful work is
~0.41 TFLOP a call, ~12 ms at the card's rate: the kernel is bound by its five workgroup barriers a 32-key tile with
global loads exposed behind each, at 2 workgroups (8 warps) an SM.

End to end with the staged kernel: asr 4.69 / 5.29 s, 0 word diffs, 80 ms shift (master 7.0 s).

Next: the blocked path's drift. It normalizes the softmax before rounding the probabilities to f16 for P.V; over
7,564 keys a typical probability is ~1/7500 = 1.3e-4, at f16's smallest normal (6.1e-5), so many land in subnormals.
The fused kernel rounds exp(s - max) (unnormalized, the row's largest 1) instead. Try the blocked path with
unnormalized probabilities and the row sum divided after.

## Run 3 - 2026-11-08 00:10

Question: is the blocked path's drift f16 rounding at all? Four clips (the original, -6 dB, 0.97x tempo, pink noise
mixed in), each against the CPU's f32 words:

```
                         clip10  -6 dB  tempo  noise    asr time
matrix cores off            0      0      0      0      4.6-4.8 s
fused relpos (staged)       0      0      0      0      4.7-5.8 s
blocked, matrix cores       3      5      4      4      2.5-2.6 s
```

Dead ends: unnormalized probabilities (exp(s - max), the row sum divided after P.V, as the fused kernel rounds them):
the same 3 swaps on the first clip. Each product alone in split f16 (hi + lo halves, three products, near f32): QK
3/5/5/4, QP 4/6/2/4, PV 4/5/8/4 - none fixes it, so it is not one product's rounding.

## Run 4 - 2026-11-08 00:40

Question: does `matmul_f32_cm` compute attention's shapes correctly? Against the CPU over f16-rounded inputs, batch 1
and 8, b along n and transposed (a temporary test):

```
m=300 n=300, 384, 512, 640; 256 x 1024; 1000 x 1000; 2048 x 2048; 1024 x 2048: max error <= 0.0013
m=300 n=599 and m=2049 n=2049 (either b layout, either batch): max error 67-101 on values of ~60-90
INFERENCE_VULKAN_NO_COOPMAT=1 (matmul_f32_tiled): every shape within 0.07
```

Every failing width is not a multiple of 4. A whole tile stores its accumulators straight to c with
`coopMatStore(..., c, row * n + col, n, RowMajor)`; NVIDIA's driver stores rows that start off 16-byte alignment wrong
(the 780M took them). The position scores' width is 2T - 1, always odd: the blocked path's QP was partly garbage,
which is the drift, not f16.

Fix: a tile stores straight to c only when n % 4 == 0, else through the staged edge path it already had. Regression
test `vulkan_matmul_matches_the_cpu_at_odd_widths` (fails on the old shader).

```
blocked, matrix cores, fixed:  0 / 0 / 0 / 0 word diffs; asr 2.58 / 2.63 / 2.89 / 2.66 s
```

## Run 5 - 2026-11-08 01:00

Routing: full attention takes the fused kernel only on an iGPU (`inference_tensor::vulkan::is_discrete`); a discrete
GPU runs the 512-row blocks' products on matrix cores. The staged kernel of Run 2 is reverted (the 780M measured
direct global loads faster; this card no longer runs the kernel), the benchmark kept.

```
3090, 10-minute clip: asr 2.59 / 2.60 s (master 7.0), diar 1.04 / 1.03 s, both 4.33 / 4.28 s (master 7.5)
words: 0 diffs; diarization: 150 segments, same speakers, boundaries within 0.01 s of the CPU's
GPU peak: 8,895 MiB (master's fused 7,554; 1,217 before load)
```

`scripts/local_ci.sh --vulkan`: 48 passed; `--lint` clean.

For the 780M (not measured here): the staged kernel (K, V and each half-workgroup's P span through shared memory,
f16vec4 stores) took the 3090 from 223 to 125 ms a layer; worth an A/B there with
`relative_position_attention_throughput`.

## Run 6 - 2026-11-08 04:05

Question (the Vulkan session's ask): does an LLM on Vulkan move across #859 (and #862)? Lightning's weights were
deleted after #565; the user chose a local stand-in. CLI at eac8c98ec (before #859) against master e034660ea,
`inference bench --vulkan --prompt-len 4096 --gen-len 128`.

Stand-ins that do not run on Vulkan, before #859 as after (not regressions):
- Qwen3.6-35B-A3B UD-IQ4_XS: "unsupported CPU RoPE dtype F32 F32 F32"; Qwen3.8-27B IQ4_XS: no IQ4_XS kernel.
- TinyLlama Q4_K_M: panicked "Got normal cache, expected hybrid cache": a failed paged forward released recurrent
  snapshots through `cache().hybrid()` on a plain cache (#837). With that path guarded, the error underneath is
  "RoPE positions must be CPU u32": `inference_quant::rotary` has CUDA and Metal branches and no Vulkan one, so a
  RoPE model reaches the CPU kernel with device tensors. Vulkan runs RoPE-free models (Nemotron-H, the SSMs).
- Jamba reasoning 3B Q4_K_M: "paged decode of 20 heads over 1 kv heads, head size 128" (no kernel for that ratio).

Mamba-Codestral 7B Q4_0 (Mamba-2, Lightning's scan path) runs; warm-up 3, 6 iterations, alternating builds:

```
                 prefill (4096)          decode (128)
before #859      2115.0 / 1629.0 tok/s   123.1 / 122.9 tok/s
master           2099.4 / 2111.1 tok/s   128.0 / 127.5 tok/s
```

(1629 is one run's outlier, +-700 across its iterations.) No regression.
