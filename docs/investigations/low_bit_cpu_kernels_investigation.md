# Low-bit CPU kernels investigation

Question (#938 step 1): can the CPU expert path (`cpu_dot`, behind the MoE expert cache) run IQ1_S, IQ1_M, IQ2_XS,
IQ2_S and IQ3_XXS fast enough for a 200-320B MoE offloaded onto one 24 GB GPU? Before this, `cpu_dot::supports` covered
IQ2_XXS and IQ3_S of the 1-3 bit types, so a layer of any other type skipped the offload.

Machine: i7-10700K (8 cores, 16 threads, AVX2, no AVX-512), dual-channel DDR4. Numbers from
`cargo test -p inference-quant --lib --config profile.dev.debug-assertions=false --config
profile.dev.overflow-checks=false -- --ignored --nocapture --exact gguf::cpu_dot::bench::single_thread_rate`
(512 rows of 8 super-blocks, weight GB/s per thread; x4 is one row against four tokens).

## Run 1 - 2026-11-13 13:00

One shared AVX2 loop for all five types: a sub-block decoder returning 32 signed bytes and two 16-value scales (IQ1's
+-1/8 offset held as 8 grid +- 1 under an eighth of the scale), the type matched per sub-block.

```
type     1 token   x4
Iq2Xxs   1.90      4.89   (existing kernel, reference)
Iq3S     4.73      11.06  (existing)
Iq1S     0.37      1.07
Iq1M     0.36      1.08
Iq2Xs    0.49      1.50
Iq2S     0.55      1.69
Iq3Xxs   0.66      2.01
```

Correct (matches the dequantized rows) but 3-5x below IQ2_XXS per byte. At 16 threads a type needs ~2.5 GB/s a
thread to keep up with DRAM.

## Run 2 - 2026-11-13 13:10

Kernels made generic over a per-type trait (const block size, decode inlined into each type's loop, one dispatch per
row). Barely moved: Iq1S 0.51, Iq1M 0.44, Iq2Xs 0.40, Iq2S 0.65, Iq3Xxs 0.86.

`perf record` of the bench: the time sat in out-of-line calls, `cpu_dot::avx2::load_q8`, the per-type `decode`, `u16_at`,
and a `Zip` fold over the tokens, each passing 256-bit values through memory. A `#[target_feature]` function cannot be
`#[inline(always)]`, and these were not inlined.

## Run 3 - 2026-11-13 13:20

`decode` without a target feature of its own and `#[inline(always)]` (it inlines into the AVX2 loops), the Q8 loads
inline, the byte helpers `#[inline(always)]`, index loops over the tokens:

```
type     1 token   x4
Iq1S     1.94      5.20
Iq1M     1.23      3.43
Iq2Xs    1.89      5.59
Iq2S     2.28      7.00
Iq3Xxs   2.54      7.85
```

Level with or above IQ2_XXS's kernel per byte; IQ1_M trails (its two scales and four offsets per sub-block). Kept.

## Run 4 - 2026-11-13 13:35

The offload test (`offloaded_ungated_experts_match_resident_ones`, CPU misses against the GPU mmvq / mmq with every
expert resident) over all six low-bit types: IQ1_M failed its prompt, "staged experts cannot run the grouped kernels".
IQ1_M has no MMQ kernel (llama.cpp has none either), so the staged prompt path cannot run it and the indexed matvec
caps at 8 tokens. Fix: a prompt over a layer whose types lack grouped kernels goes through the split path 8 tokens at
a time (cached experts on the GPU, misses on the CPU). Slower prefill for those layers, but the layer runs.

The test also hit "3 Iq2Xs experts of 256x256 overflow the expert stack": the staging buffer is process-wide and
sized by the offloaded layers that exist at the first staged prompt, which in a model is every layer, but the test
built one type's layer after another's prompt. The test now runs the types largest block first.

Not done: an IQ1_M MMQ kernel (would bring its prompts back onto the staged path); the ik_llama types (step 2).

## Run 5 - 2026-11-13 14:30

#938 step 2: ik_llama's table types IQ2_K, IQ2_KS, IQ2_KL, IQ3_K, IQ3_KS in the same module. The per-type trait now
gives each type its own scalar and AVX2 decode (grid lookups for ggml's types; one byte shuffle over a 16-entry
codebook for the 2- and 3-bit table types, two shuffles and a blend over IQ2_KL's 32 pair codes), and the loops read
the f16 row scale the KS/KL types carry ahead of their blocks. Same bench:

```
type     1 token   x4
Iq2K     4.55      9.33
Iq2Ks    5.48      8.19
Iq2Kl    3.14      8.44
Iq3K     3.75      9.98
Iq3Ks    4.58      11.73
```

Above the grid types per byte (no table gathers). The grid types did not move (Iq1S 1.95, Iq2Xs 2.03, Iq3Xxs 2.63).
The offload test runs all eleven low-bit types against the resident GPU kernels (decode and prompt), sorted by row
size for the shared staging.

## Run 6 - 2026-11-13 15:00

The trellis types IQ1_KT, IQ2_KT, IQ3_KT (f32 row scale, one integer scale per 32 values). A group's eight values are
eight steps of `state *= 0xCBAC1FED`, each the sum of the state's four 6-bit bytes minus 126 (-126..=126, so a byte);
in AVX2 one `mullo_epi32` of the start against the step's powers gives all eight states, two multiply-adds sum the
bytes, and two packs plus a permute put four groups' values in order. IQ3_KT takes the magnitudes and its qh sign bits.

First bench: Iq1Kt 0.49, Iq2Kt 0.66, Iq3Kt 1.25 GB/s for one token. Not the kernels: the bench fills rows with byte 3,
which as an f32 row scale is 3.9e-37, and times the activation scale that is denormal, the FPU's slow path (the
results printed ~1e-25). With a 0.01 row scale in the bench:

```
type     1 token   x4
Iq1Kt    1.04      3.28
Iq2Kt    1.72      4.64
Iq3Kt    2.22      5.96
```

The offload test passes over all 14 low-bit types. Left in #938: step 3, the writers.

## Run 7 - 2026-11-13 16:00

#938 step 3, writers. `scripts/iq_quantize_parity.py ~/Programming/llama.cpp/build_cuda/bin iq2_xxs iq2_xs iq2_s`
(64 rows of 4096, skewed values and imatrix, ggml's `ggml_quantize_chunk` against ours, byte for byte; ggml requires
an imatrix for IQ2_XXS / IQ2_XS / IQ1_S, so those run only with one).

ggml's `iq2xs_init_impl` tables (`kgrid_2bit_256/512/1024`, `kgrid_1bit_2048`) are the dequant grids' levels: each
coordinate's rank among the grid's values (8 < 25 < 43, or -1 < 0 < 1). Derived from our `iq_tables` grids they equal
ggml's four tables exactly (checked by parsing ggml-quants.c), so the writers build the maps from those.

First run: IQ2_XS and IQ2_S 0 bytes differing, IQ2_XXS 3266 of 67584 (544 of 1024 blocks; grid indices and scales,
never signs). Cause: IQ2_XXS's even-parity flip leaves one magnitude negative, and it goes through `make_qp_quants`,
where ggml keeps a negative level as an int in the sums but wraps it to uint8_t in `L`; our port wrapped then clamped
(255 min 4 = 4 where ggml has -1). Q2_K never passes a negative value, which is why its parity held. With the C
types mirrored: IQ2_XXS, IQ2_XS, IQ2_S and Q2_K all 0 bytes differing.

## Run 8 - 2026-11-13 16:40

IQ3_XXS / IQ3_S (`iq3xs_init_impl`: 4 coordinates of 3-bit levels, a 4096-entry map; its `kgrid_256` / `kgrid_512`
also equal our grids' levels) and IQ1_S / IQ1_M (the exhaustive 3-level split over the sorted values, the fit-scored
neighbour search, IQ1_M's final refit of `d` over every sub-block, empty ones included): all 0 bytes differing on the
first run. The full parity set, `iq1_s iq1_m iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s q2_k iq4_nl iq4_xs`, plain and with
an imatrix: 0 bytes differing throughout.

Two dead ends on the way:
- the unit test hung at 0% CPU: the grid tables were built with rayon inside a `OnceLock`, and the first quantizing
  worker to need them builds them while the rest of the rayon pool waits on the lock, so the build's own work items
  had no thread to run on (the parity example got through by scheduling luck). The build now runs on scoped threads.
- "an imatrix lowers the weighted error" first failed for IQ1_S, then gave identical errors with and without the
  imatrix for most types: the test's skew was constant over each sub-block, which scales that sub-block's whole fit
  and changes nothing. With an every-other-column skew (10:1; at 10000:1 ggml's IQ2_XXS loses, as ours does, being
  the same bytes) every low-bit type lowers it.

## Run 9 - 2026-11-13 17:30

ik_llama's table types: IQ2_K, IQ2_KS, IQ2_KL, IQ3_K, IQ3_KS, against `~/Programming/ik_llama.cpp/build_cuda/ggml/src`
(native build, g++).

- g++ contracts a product into the add (or subtract) that alone uses it in the same block, so the port writes each
  sum as those FMAs: `s += w*q*x` is `fma(w*q, x, s)`; a chain `a0*v0 + a1*v1 + a2*v2 + a3*v3` is
  `fma(a3, v3, fma(a2, v2, fma(a0, v0, a1*v1)))`; two products in one statement fuse the first into the second; a
  product with another use (`id*x` feeding an index lookup too) stays unfused. With that rule IQ2_K matched on the
  first run.
- IQ2_KS: ik's quantize passes a `quantize_user_data`, whose default picks `quantize_row_iq2_ks_fast_impl` (AVX2;
  `--slow-iq2ks` picks the O(n^3) reference, also far too slow for a 27B model). The parity script passed NULL (the
  slow path); it now passes a zeroed struct (mainline's 7-argument `ggml_quantize_chunk` ignores it). The fast path's
  eight-lane sums are emulated lane by lane with `hsum_float_8`'s pairing.
- IQ3_K / IQ3_KS run `find_best_scale_avx2` on an AVX2 build (truncated-index lookup into `k_simple_index_iq3nl`,
  both signs per lane), ported the same way.

All five, and IQ4_KS, 0 bytes differing, plain and with an imatrix. Round trips on uniform values (rms 0.58): Iq2K
0.132, Iq2Ks 0.139, Iq2Kl 0.130, Iq3K 0.066, Iq3Ks 0.073. Left: the trellis writers (IQ1_KT, IQ2_KT, IQ3_KT).

## Run 10 - 2026-11-13 19:30

The trellis writers IQ1_KT, IQ2_KT, IQ3_KT. The worry was `QuantizerIQKT`'s k-means over the 65536 codes, seeded by
`std::mt19937(1234)` with libstdc++'s distributions; but all three construct it with 256 clusters of 8-value groups,
which `cluster_points` bins directly (each coordinate above or below its split: the range midpoint, or IQ3_KT's
median) without the random start. So byte parity was in reach after all.

- First the shared codebook (`trellis.rs`: multiplier, add, decode; ik's KT types one instance, QTIP 3INST and
  EXL3's mcg/mul1 the same shape, #943), with the KT dequantizer and the CPU dot kernels moved onto it.
- `find_best_match`'s cluster search emulates `hsum_float_8x8` with its permutes and unpacks literally.
- IQ2_KT: 0 bytes differing on the first run. IQ1_KT 1 / 3 and IQ3_KT 0 / 7 bytes (plain / imatrix), every one in a
  row's f32 scale, 1 ulp. Dropping the fusion in the final refit made it worse (22); one sum each, 17 / 15.
- objdump of ik's `quantize_row_iq3_kt_impl`: `set_weights` inlined, its super-block `sumx2 += x*x` over the runtime
  `n` vectorized as products in a ymm then eight in-order `vaddss` (unfused; only the scalar tail uses
  `vfmadd231ss`). IQ1_KT's inlined copy does the same; IQ2_KT's stays scalar and fused. With that sum unfused for
  IQ1_KT and IQ3_KT: all three 0 bytes differing, plain and with an imatrix.

Round trips on uniform values (rms 0.58): Iq1Kt 0.254, Iq2Kt 0.205, Iq3Kt 0.113 (above the grid types of their size
on uniform data; the trellis codebook is shaped for bell-curve weights). #938 step 3 done.
