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
