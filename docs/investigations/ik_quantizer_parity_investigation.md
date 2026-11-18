# ik_llama.cpp's quantizer in `inference quantize gguf`

Goal: write the GGUF bytes ik_llama.cpp's `llama-quantize` writes, so a recipe built for its better scale search
needs no ik build. Code: `crates/inference-tensor/src/quantized/{k_quants,utils}.rs` (`Quantizer`),
`crates/inference-quant/src/gguf/iq_quantize.rs` (IQ4_NL / IQ4_XS). Reference: ik_llama.cpp c446f1e0's `build_cuda`
(gcc 13, `-O3 -march=native`, AVX2 + FMA host).

These entries were written up after the runs, from the session's commands and outputs; the times are approximate.

## Run 1 - 2026-11-18 15:05

Question: our R1 recipe (Qwen3.6-35B-A3B, recipe investigation Run 17) made by `inference quantize gguf` differed
from ik's in every Q5_K / Q6_K / IQ4_XS tensor. Which types differ, and how?

`scripts/iq_quantize_parity.py` against ik's `libggml.so`: thousands of bytes differing in every k-quant, even plain
Q4_K / Q5_K whose C source is the same as llama.cpp's. Against llama.cpp's own build: 0 everywhere.

Two causes, both in the test harness, found by extracting each function from both sources (preprocessed) and by
building ik's `ggml-quants.c` alone with and without `-ffp-contract=off`:

- ik reads fp16 through `ggml_table_f32_f16`, which `ggml_init` fills; the script never called it, so every scale
  read back as 0. With `ggml_init` the plain Q2_K-Q5_K paths match llama.cpp's. The script now initialises.
- ik's build is `-march=native` (FMA) with gcc's default contraction; llama.cpp's `libggml-base` targets plain
  x86-64, so it never fuses. The object built with `-ffp-contract=off` differs from ik's by a few hundred bytes a
  type, the default build by 0: the remaining small differences are fused multiply-adds.

The algorithmic differences that remain: `make_qx_quants` (the 0 step tried too, a second candidate from the positive
end of the range, then single values moved a step along the largest weighted gradient while the error falls),
`make_qkx3_quants` (sums in f64, error in closed form, a greedy refinement), `make_qp_quants` / `make_qkx2_quants`
(edge guards), Q2_K (a ternary check, sigma2 x 0.75, levels from the unrounded scales), IQ4_NL / IQ4_XS (its AVX2
path: per-lane sums, a table index that keeps the lower entry on a tie, both codebook ends, refinement).

## Run 2 - 2026-11-18 16:30

Method: port each search behind `Quantizer::IkLlama`; check it first with fusion off against ik's file built with
`-ffp-contract=off` (algorithm), then find where gcc fused against ik's own build. The fusion sites were found by
switching each candidate `a * b + c` by source line (`#[track_caller]` and an environment list, removed after) over
several seeds.

What gcc fused, as the code now has it:

- `nearest_int(a * b)` after inlining: the product into the rounding add, everywhere.
- `make_qx_quants`: the first pass's sums fused, the 19 candidate passes' not; the refinement's `sumlx +/- w*x` not,
  its `suml2` step fused.
- `make_qkx2_quants`: `sum_l += w*l` not (gcc shares `w*l` with the other two sums), `sum_l2` / `sum_xl` fused;
  `this_scale = sum_w*sum_xl - sum_x*sum_l` fuses its second product, `d` and `this_min` their first.
- `make_qkx3_quants` (f64): the same `this_scale` rule; everything else fused.
- Q3_K's `sum_x2` (a plain reduction) not fused, its weights' `sqrt(sigma2 + x*x)` fused.

Result: `the_ik_quantizer_writes_ik_llamas_bytes` pins Q2_K-Q6_K, Q4_0-Q5_1, Q8_0, IQ4_NL and IQ4_XS, plain and
imatrix, to hashes computed by ik's own `ggml_quantize_chunk` on the same layer (all 24 equal). The parity script
and a dozen seeds of the harness (`ours_vs.py`, a direct `quantize_<type>` call) show 0 bytes differing for each.
llama.cpp parity unchanged (0 for every type). Along the way `make_q3_quants` lost a `.round()` (ties away from
zero) where ggml uses `nearest_int` (ties to even), and its `w*(l*l)` became ggml's `(w*l)*l`.

Not ported: the 1-3 bit grid types. ik rewrote IQ2_XS (an exhaustive split search over the sorted pairs, three
refinement rounds), IQ1_S and IQ1_M; IQ2_XXS, IQ2_S, IQ3_XXS and IQ3_S differ by a few bytes (zeroed levels moved,
`is_on_grid` no longer reset, fusion). `--quantizer ik-llama` writes them with llama.cpp's search and warns.

Next: the grid types; KLD of the R0-R2 recipes rebuilt with `inference quantize gguf` to confirm the files match.
