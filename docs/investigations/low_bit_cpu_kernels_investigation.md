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
