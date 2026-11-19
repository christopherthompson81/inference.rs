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

Next: the grid types; the R1 recipe rebuilt with `inference quantize gguf` to confirm whole files match.

## Run 3 - 2026-11-18 18:20

Question: does a whole recipe come out as ik's `llama-quantize` writes it?

Command: the recipe investigation's R1 rules (IQ4_XS experts, Q5_K `attn_qkv`, Q6_K attention / shared experts /
`ssm_out` / `attn_gate` / head, Q8_0 the rest, the Unsloth imatrix) through `inference quantize gguf` with its
default `--quantizer ik-llama` (dev build, 22.7 min), compared tensor by tensor with ik's file (its blk.40 MTP graft
left out): 733 identical, 0 differing.

## Run 4 - 2026-11-18 21:30

Question: the 1-3 bit grid types (IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S).

The scratch harness now runs each build's own grid init (`iq2xs_init_impl` / `iq3xs_init_impl`, the tables are
static per object). Against ik's file built with `-ffp-contract=off`, our llama.cpp-search output already matched
IQ2_XXS, IQ2_S, IQ3_XXS, IQ3_S and IQ1_S: their source changes (zeroed levels moved, IQ3_XXS's `is_on_grid` not reset
between blocks, IQ2_XXS's `eff_max <= 0` exit gone) only show on degenerate blocks; they are carried over anyway.
With every candidate `a * b + c` fused (the levels' `nearest_int(0.5 * (id * x - 1))` twice) all five matched ik's
own build on 9 seeds.

IQ2_XS: ik's rewrite ported (`quantize_iq2_xs_ik`): the best split of the sorted magnitudes into 1 / 3 / 5 as the
start, 15 scales around it, up to three rounds of re-snapping to the grid at the fitted scale, a 1.05 fudge, and the
block scale refit over every value as stored (grid bytes x 0.125). All fused: 0 bytes on 11 seeds, first try.

IQ1_M: `iq1m_process_1block` ported (sigma2 per 32 values x 1.5, per-half running sums, best score from 0, all four
shift pairs refit after a snap off the grid, a 1.085 fudge). Unfused it matched ik's unfused build exactly; fused,
the imatrix path stayed 2-14 bytes off whatever single sites were toggled, and a greedy search over sites stalled.
The disassembly of `iq1m_process_1block` (16 FMAs against 52 multiplies and 30 adds) gave the rule: gcc fuses only a
product with one use. The split search's first term of each half is hoisted out of the inner loop (added), the
second half's `x` terms are shared between two shift choices (added), the `w * x * x` terms keep a unique last
multiply (fused); `weight[i] * xb[i]` in the running sums appears in both branches of the half test, so it is
computed once before them (added). With those: 0 bytes on 8 seeds.

Result: `the_ik_quantizer_writes_ik_llamas_bytes` pins the seven grid types too (11 hashes, from ik's
`ggml_quantize_chunk`); the parity script shows 0 differing against ik's build and against llama.cpp's for every
grid type. `--quantizer ik-llama` now covers every type `quantize gguf` writes, so the fallback warning is gone.

