# CPU Q8_0 activations investigation (#655)

On x86 the repacked Q8_0 matmul (`repack_x86`) quantizes its activations to Q8_K, one scale per 256 values, where
ggml's `vec_dot_q8_0_q8_0` gives every 32-value activation block its own scale. Models whose activations carry
outliers (Nomic BERT's FFN inputs) lose accuracy on CPU against CUDA and llama.cpp.

## Run 1 - 2026-10-27 18:43

Question: where does the CPU Q8_0 path stand today, and what is the prefill throughput to hold?

Command: `inference serve --cpu --dtype f32` on nomic-embed-text-v1.5 as a llama.cpp Q8_0 GGUF and as ISQ Q8_0 on
the safetensors checkpoint, and on CUDA from the same GGUF, against `nomic_remote_ref.py` (Nomic's own modeling code
at F32; four texts, one of 402 tokens). Prefill: `decode_matvec_bandwidth` (`cargo nextest run -p inference-tensor
--run-ignored ignored-only`, dev profile), the Q8_0 512-row lines, three runs. The CPU is an i7-10700K: AVX2 only, so
of the three Q8_0 kernels (AVX2, AVX-VNNI, AVX-512 VNNI) only `q8_0_tile2_avx2` runs here, and only at m > 1
(`select` keeps one-row AVX2 matmuls on the compact gemv).

```
nomic v1.5 Q8_0  CPU GGUF  worst cos 0.992571    CPU ISQ  0.992668    CUDA GGUF  0.998178
prefill Q8_0 2048x8192 x512: 50.1 / 50.1 / 44.7 ms   (343 / 343 / 384 GFLOP/s)
prefill Q8_0 4096x2048 x512: 24.4 / 24.4 / 21.1 ms   (352 / 352 / 408 GFLOP/s)
```

Finding: the CPU gap is as #655 describes (0.9926 against CUDA's 0.9982; Run 13 of the encoder investigation
measured 0.9911 / 0.9904 before later changes). The prefill numbers vary run to run by about 12%.

Implication: give the Q8_0 tiles their own activation blocks (256 values as eight 32-value blocks, a scale and an
integer sum each, quantized as ggml's `quantize_row_q8_0`), and have the three Q8_0 kernels read the per-block scale
and sum. Check against the non-repacked Q8_0 path, which already pairs Q8_0 with Q8_0.

## Run 2 - 2026-10-27 18:50

Question: with the q8_0 tiles taking q8_0 activations (`ActQ8_0`: eight 32-value blocks per superblock, each with
its f16-rounded scale and integer sum, from `BlockQ8_0::from_float`), does the CPU reach the CUDA level, and does
prefill hold?

Command: as Run 1, after the change; plus `q8_0_matmuls_take_a_scale_per_32_activations`, which checks the
repacked matmul (m = 1, 4, 9; an outlier 60x every 64 activations) against ggml's pairing computed directly
(weights and activations each `BlockQ8_0`, `BlockQ8_0::vec_dot`).

```
nomic v1.5 Q8_0  CPU GGUF  worst cos 0.998260    CPU ISQ  0.998108    (CUDA GGUF 0.998178, Run 1)
prefill Q8_0 2048x8192 x512: 53.5 / 45.6 / 45.8 ms   (Run 1: 50.1 / 50.1 / 44.7)
prefill Q8_0 4096x2048 x512: 26.0 / 23.0 / 23.0 ms   (Run 1: 24.4 / 24.4 / 21.1)
test against ggml's q8_0 x q8_0: passes; with the Q8_K activations (HEAD's kernels) it fails at m > 1
```

Finding: the CPU now matches CUDA (0.9983 / 0.9981 against 0.9982, from 0.9926). Prefill is within Run 1's spread
(medians 45.8 / 23.0 ms after, 50.1 / 24.4 ms before): the kernels do the same integer work and only read a scale and
a sum per 32-value block instead of per superblock. The AVX-VNNI and AVX-512 q8_0 kernels take the same change but do
not run on this CPU (AVX2 only); they are compiled, not executed, here.

Implication: commit; the AVX-VNNI / AVX-512 paths want a run on hardware that has them.

## Run 3 - 2026-10-27 19:03

Question: the full CI run failed `gemma3_tiny::gemma3_re_isq_decodes_as_recorded` ("Q8_0 decode moved"); is that
the fix or a regression?

Command: `scripts/local_ci.sh --lint --tests --cuda --bindings --size --sweep`, then the test alone.

```
Q8_0 decode moved: [(11, -3.6783395), (11, -3.678008), (11, -3.688088), (11, -3.6991017), (11, -3.6988003), (11, -3.6956322)]
```

Finding: the fix. The test pinned three Q8_0 decodes by CPU (every row tiled on VNNI, only prompt rows on AVX2, none
elsewhere) because the tiles quantized activations per 256. This machine's decode now equals, to every recorded
digit, the pin of the compact path (`None`), which already paired Q8_0 with Q8_0: the tiled kernel reproduces it
exactly. The three pins collapse to that one. A VNNI machine tiles the one-row decode too; with the same quantized
values only its f32 summation order differs, which should stay within the 1e-5 pin, but that is unchecked here.

Implication: rerun CI with the single pin.
