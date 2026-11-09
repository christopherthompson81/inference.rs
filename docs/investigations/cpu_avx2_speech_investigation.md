# Parakeet and Nemotron-3 diarization on an AVX2-only CPU

The user saw this desktop's CPU take 131 s for the 10-minute clip's ASR against ~24-31 s on the laptop's Ryzen 7
7840U (AVX-512 BF16, `nemo_investigation.md` Runs 44-48). Desktop: Intel i7-10700K, 8 cores / 16 threads, AVX2 +
FMA + F16C, no AVX-512 or AVX-VNNI. Clip: the 605 s synthetic speech file of `vulkan_relpos_3090_investigation.md`;
`from-config` with both models on `cpu = true`, a warm-up, then ASR with word timestamps, diarization, and both.

## Run 1 - 2026-11-08 15:00

Question: where does the 131 s go, and how much is the machine?

```
dev build (scripts/build_cli.sh, debug-assertions and overflow-checks on), 8f914d2ef / master 2b0b1f9da:
  asr 134.8 / 135.1 s, diar 27.0 / 27.1 s; words identical (#869 neutral here: AVX2 has none of its kernels)
perf, dev build: gemm_f32 fma x2x6 microkernel 70.7 %, RelShiftSoftmax 11.5 %, dot_f32 (decoder) 5.2 %
```

GEMM rate (a temporary `#[ignore]` benchmark, `Tensor::matmul` on the CPU, 5 calls):

```
                              dev build   no debug assertions   + RAYON_NUM_THREADS=16   (pinned 8: no change)
7564 x 4096 x 1024            179 GFLOPS        356                    468
7564 x 1024 x 4096            167               403                    514
7564 x 1024 x 1024            158               311                    519
8 x 512 x 7564 x 128 (scores) 146               222                    218
8 x 512 x 128 x 7564 (values) 198               297                    452
```

The laptop's numbers come from release builds; the dev profile's checks halve `gemm`'s rate (as in the
perf-measure note). Built with `--config profile.dev.debug-assertions=false --config profile.dev.overflow-checks=false`:

```
asr 86.5 / 85.9 s, diar 16.3 / 14.9 s; words identical
RAYON_NUM_THREADS=16: asr 63.3 s, diar 16.6 s; words identical
perf at 16 threads: gemm 42.6 %, BarrierPool workers 25.4 % (spinning beside gemm's threads), RelShiftSoftmax 9.4 %,
  dot_f32 8.0 %
```

The hyperthreads help `gemm` here (+30-70 %): its x2x6 microkernel leaves FMA slots a second thread fills. The
7840U's f32 gemm ran 569-647 GFLOPS (Run 46), its bf16 1055-1337: against this CPU's best f32, ~2.5x per product.

Candidates, by expected gain:
- `gemm` on every logical CPU for large products on an SMT CPU where it helps, the elementwise pool kept at the
  physical count and parked during them (its spin is 25 % at 16).
- The relative-position attention fused in f32 for AVX2 (`cpu_backend::relpos_attn` is AVX-512 BF16 only): the scores
  written and read again, and the softmax (~10 %); on the laptop the fused kernel took ASR 35.9 -> 27.9 s.
- The decoder's dot products (8 %).

## Run 2 - 2026-11-08 16:00

Change: `utils::smt_gemm_pool`, a rayon pool of one thread per logical CPU, built where the CPU has more logical CPUs
than physical cores and `RAYON_NUM_THREADS` is unset (`INFERENCE_CPU_GEMM_SMT=0` turns it off). f32 products of 2^26
multiply-adds and more run their gemm calls in it; everything else keeps the physical cores' threads (the elementwise
pool still parks during them).

```
benchmark (no debug assertions):  pool on          pool off
7564 x 4096 x 1024                456 GFLOPS       370
7564 x 1024 x 4096                529              397
8 x 512 x 128 x 7564 (values)     465              304
7564 x 1024 x 1024                315-319          313; 512-539 at RAYON_NUM_THREADS=16 (not explained)
clip, CPU, no debug assertions:   asr 58.8 / 65.9 s, diar 11.3 / 14.8 s   vs  pool off 86.0 s, 16.1 s
```

Words identical in every run. Test: `a_large_f32_product_on_the_smt_pool_matches_its_sums` (300 x 520 x 700 against
f64 sums).

## Run 3 - 2026-11-08 16:30

Change: `cpu_backend::relpos_attn_f32`, `relpos_attn`'s fused relative-position attention in f32 on AVX2 and FMA, on
the SMT pool's threads: 48-row tasks, 512-key chunks, a 6 x 16 FMA tile over f32 panels. The CPU takes it for
Parakeet's full attention where the AVX-512 BF16 kernel does not apply (no bf16 pairs, or outside the bf16 scope).
Every product keeps f32. Tests: `f32_relative_position_attention_matches_its_definition` (f64 definition, within
1e-5, heads 128 / 32 / 16, past a key chunk; a diagonal one row off fails it at 2.16); the bf16 test now checks that
outside its scope the dispatcher gives the f32 kernel's output.

```
a layer (8 heads, 7564 x 128), useful products:
first version                 965 ms   364 GFLOPS
P V in 128-key steps          1053 ms  dead end
panels outer, row groups inner (each panel in L1 for all 8 groups; a chunk's 67 panels had streamed past L2 a group):
  P V steps 64 / 128 / 256 / 512:  673 / 735-739 / 634-664 / 642 ms; kept 256
clip, CPU: asr 47.7 / 48.8 s (Run 2's 58.8-65.9), diar 15.0 s; 1698 words identical, 0 ms start shifts
perf before the loop swap: the tile 48.8 %, gemm 28.5 %, dot_f32 (decoder) 7.3 %, the softmax row 3.8 %
```

Checks: lint, CPU (3159), CUDA (3362), `--vulkan` (63) pass.

Where it stands (no debug assertions): asr 135 (dev build) / 86 -> ~48 s, diar 27 / 15-16 -> ~15 s. Left: the
decoder's dot products (~7 %), the diarizer (a plain transformer: gemm plus a softmax composite), and gemm's
7564 x 1024 x 1024 at the SMT pool.

## Run 4 - 2026-11-08 17:10

Question: the decoder's dot products (7 %): Parakeet's LSTM and joint stream ~34 MB of f32 weights a step past the
16 MB L3. #869 holds them as bf16 copies (`cpu::Bf16RowWeights`, plain Rust that widens to f32) inside the model's
bf16 scope, but gated on `bf16_enabled()`, which also needs AVX-512 BF16. A weight's bf16 rounding is within the
model's measured property (both operands rounded), and the copy needs no bf16 pairs, so the decoder now asks
`cpu::bf16_scope()` (the model's scope, `INFERENCE_CPU_NO_BF16` unset) instead. Test:
`the_scope_holds_whatever_the_cpu_multiplies`.

```
asr 42.8 / 44.6 s (Run 3 47.7 / 48.8), diar 14.7 / 14.5 s; 1698 words identical, 0 ms start shifts
```

Diarization, profiled (`perf_diar.sh`, 11.4 s under perf): gemm 66 %, libm `__expf_fma` 5.7 % (the softmax), the
elementwise pool's spin 6 %. The laptop's f32 diarization was 11.4 s (Run 46), so this CPU is level with it there.

## Run 5 - 2026-11-08 18:00

Question: the CPU f32 softmax spent 5.7 % of a diarization in libm's `expf`: does `fast_exp` (Cephes, ~1e-7) help?

Dead end first: passing the exp as a `fn` pointer made each element an indirect call. Alternating runs (three 10-minute
diarizations a server, a temporary libm toggle): fast 15.99-16.31 s against libm 14.71-15.11. As a generic closure
(inlined, vectorized): fast 14.05-14.44 s against libm 14.62-15.21, kept.

Every CPU f32 softmax now takes `fast_exp`; the CPU suite (3160) passes, its pinned outputs unchanged; the clip's
diarization is byte-identical to the CPU reference and its 1698 words unchanged. This machine drifted slower over the
session (Qwen3.8-27B's Vulkan prefill read 362-433 on master against 480-533 earlier), so only alternating runs count.

## Run 6 - 2026-11-08 18:20

Run 2's open question (7564 x 1024 x 1024 at 315 GFLOPS on the SMT pool against 512-539 at RAYON_NUM_THREADS=16) was
the benchmark's operand layout: it multiplied a row-major (k, n) weight, the model a transposed (n, k) one. As the
model multiplies it (a temporary benchmark, 20 calls): pool 541 / 501 GFLOPS, RAYON_NUM_THREADS=16 551 / 527, pool
off 409 / 394. Nothing to fix.

## Run 7 - 2026-11-08 19:00 (the laptop's Ryzen 7 7840U, AVX-512 BF16, measured by the other session)

#873 against its base 68b2014df, on the 10-minute meeting:

```
bf16 path (default): asr and both-models responses byte-identical; asr 22.67 vs 22.62 s
  standalone diarization: 65 of 46,836 speaker-frames moved (0.14 %), the same each run of a build; the
  both-models response's word speakers identical. Diar 8.06 / 14.25 / 13.22 s against 7.72 / 12.65 / 11.76 (rounds
  2-3 ran hot there)
INFERENCE_CPU_NO_BF16=1 (relpos_attn_f32, the SMT pool, the softmax): asr, diar and both byte-identical to the
  base's f32 path; asr 38.5 -> 36.7 s, diar 11.5 -> 10.9, both 47.7 -> 46.2 (one run each)
```

The moved frames come from Run 5's softmax exp (`fast_exp`, ~1e-7) tipping a few near-threshold frames on the bf16
path; this desktop's diarization (f32 path) stayed byte-identical. Within `scripts/bf16_check.sh`'s 0.5 % for a
diarizer.

## Run 8 - 2026-11-08 20:00

Question (the other session's suggestion): can the bf16-measured speech models keep bf16 weights on an AVX2 CPU too?
On the 10700K they load at ~3.1 GB RSS against the 7840U's ~1.7, `Linear::to_half` halving on a CPU with bf16 pairs
only.

Kernel: `cpu_backend::gemm_bf16w`, x (f32) times a bf16 `(n, k)` weight's transpose: 192-row by 64-column tasks on the
SMT pool, the weight widened to f32 into 16-column panels 256 deep as each task packs them, the 6 x 16 FMA tile (now
`cpu_backend::fma_tile`, shared with `relpos_attn_f32`). All f32 but the weight. Tests: against the f32 product over
the rounded weight (partial row groups, a ragged depth; a 0.1 % wrong widening fails it), and `Linear::to_half` on an
AVX2 CPU (1, 3 and 512 rows with a bias; a 100-row weight stays f32).

```
benchmark (no debug assertions), GFLOPS:     KC 256, MC 96 / 192 / 48       gemm f32
7564 x 4096 x 1024 (feed-forward up)         538 / 602 / 328                 437-475
7564 x 1024 x 4096 (feed-forward down)       419 / 457 / 337                 422-450
7564 x 1024 x 1024 (attention projections)   468 / 517 / 355                 311-317
684 x 2048 x 512, 684 x 512 x 2048 (diarizer) 471, 442 / 581, 528 / 365, 350  446-545, 406-473
KC 384, 512 (MC 96): lower on every shape. Kept MC 192, KC 256.
```

Measurement (`bf16_check.sh`'s criteria by hand: it needs AVX-512 BF16 and a release build). Both models served
once each way (`INFERENCE_CPU_NO_BF16=1` keeps f32 weights), seven clips: the 10-minute clip, its -6 dB, noise and
0.97x tempo variants, and 20 / 45 / 90 s slices:

```
loaded RSS 3100 -> 1711 MiB; asr on the long clips 46.5-48.8 -> 41.6-45.5 s, diar 13.9-14.5 -> 13.5-15.3 s
clip10     1698 words: 1 differing span (night. Log -> night, log), 2 starts moved | speaker-frames 0.000 % moved
var_gain   1698 words: 1 differing span (fifteen -> fifty), 2 starts moved         | 0.000 %
var_noise  1698 words: 0, 0 starts                                                 | 0.000 %
var_tempo  1698 words: 0, 3 starts                                                 | 0.000 %
short20 / short45 / short90: 43 / 99 / 250 words, 0 differing, 0 starts            | 0.000 %
```

The diarizer passes (no speaker-frame moved on any clip). Parakeet fails the ASR rule (no word may change): two of
four long clips change a word, the second the same near-tie word ("fifteen") the Vulkan blocked path once flipped.
Its bf16 property was measured on the laptop's meeting with bf16 inputs; these clips were not part of it.

## Run 9 - 2026-11-08 21:00

Question: is there accuracy to recover? Run 8 said Parakeet failed under bf16 weights; its baseline was wrong.

- f16 instead of bf16 for the AVX2 weights (the kernel widens them anyway; three more mantissa bits, the same size;
  `gemm_halfw` now takes either): the same two words changed, 0 starts moved instead of 2.
- The same two flips under 8x finer rounding pointed elsewhere. `INFERENCE_CPU_NO_BF16=1` also turns off the
  decoder's bf16 copies (#869, extended to AVX2 in #873), so Run 8 compared against a full-f32 decoder. Master's
  default against full f32: the same two words change (clip10 "night. Log -> night, log", var_gain "fifteen ->
  fifty"). The encoder's halved weights against master's default: 0 words on every clip (f16: 1 start; bf16: 6).
  Earlier word checks (`cmp.py`) lowercased and stripped punctuation, hiding the first.
- The decoder's copies in f16 (`cpu::HalfRowWeights`, F16C widening 8 lanes at a time; bf16 where a CPU lacks F16C)
  with the encoder's weights in f16, against full f32 on all seven clips: 0 differing words, 0 starts moved, 0.000 %
  of speaker-frames.
- The halved diarizer read ~8 % slower (15.5-15.7 s against 14.4-14.6 in f32, alternated): the elementwise pool spun
  beside the kernel's threads (20 % of samples against 6 %), matmul parking it for a long product and this path not.
  Parked the same way: 13.9-14.1 s against 14.0-14.5.

```
seven clips, no debug assertions:     master default             this branch
loaded RSS                            3126 MiB                   1712 MiB
request peak (10-minute clip's asr, diar and both, sampled every 100 ms): f32 weights 4058 MiB, this branch 2826 MiB
asr, 10-minute clips                  44.3-47.1 s                44.6-47.7 s
diar, 10-minute clips                 13.5-14.3 s                13.9-14.2 s
short20 / 45 / 90 asr                 1.16 / 2.40 / 4.81 s       1.01 / 1.98 / 4.26 s
words, starts, speaker-frames against full f32: master 2 words and 1 start changed; this branch none
```

Tests: `a_half_weight_product_matches_the_rounded_weights` (f16 and bf16), `an_f16_weight_linear_widens_its_weight_
without_bf16_pairs`, `matches_the_rounded_weights_product` (the decoder copies, both formats, a tail past whole 16-lane
steps). Lint, CPU (3165), CUDA (3368), `--vulkan` (65) pass.

The process's lifetime high (VmHWM) stays ~5.0 GB: loading reads the f32 weights before `to_half` copies them.
On a CPU with AVX-512 BF16 (the 7840U) the decoder's copies also become f16 (it has F16C); its encoder keeps bf16
and its bf16 matmul inputs, whose words were measured on the laptop's meeting only.
