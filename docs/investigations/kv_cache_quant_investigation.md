# KV cache quantization investigation

Quantized paged KV caches beyond FP8: any bit width from 2 to 8, chosen separately for K and V, per model.

## Run 1 - 2026-10-17 10:00

Question: how do llama.cpp, ik_llama.cpp and exllamav3 quantize the KV cache, and which shape fits our paged
cache and its fattn reads?

Sources: llama.cpp at `8bb3cf0d2` (`common/arg.cpp`, `src/llama-kv-cache.cpp`, commit `a09d8abf8` "Support rotated
kv cache quant (#26180)"); ik_llama.cpp `common/common.cpp`; exllamav3 `exllamav3/cache/quant.py` and
`exllamav3_ext/cache/q_cache_kernels.cuh`.

```
             types                                   rotation                                K/V separate
llama.cpp    f16 bf16 q8_0 q5_0 q5_1 q4_0 q4_1       Hadamard, automatic when quantized and   -ctk / -ctv
             iq4_nl (GGUF blocks of 32)              head dim % 64 == 0: Q and K by the same
                                                     matrix (up to the head dim), V by 64-wide
                                                     blocks, undone on the output
ik_llama     the same plus q6_0, q8_KV               opt-in: -khad / -vhad                    -ctk / -ctv, plus
                                                                                              -ctk-first/-last ranges
exllamav3    2..8 bits each                          H32 inside each group of 32, in the      k_bits / v_bits
             groups of 32, one f16 absmax scale,     quantize kernel; undone on read
             symmetric midpoint grid (~5% lower
             MSE than rounding), optional cubic
             companding
```

Reads: llama.cpp's decode kernels read quantized K/V directly and prefill converts to f16 first; exllamav3 either
dequantizes inside attention or into a temporary f16 buffer.

Ours today: the paged cache is the activation dtype or FP8 e4m3 with per-tensor scales (from the checkpoint when
it ships them). Paged attention runs fattn's mma kernel for prefill and decode, whose tile loader converts FP8 to
f16 per 16-byte chunk of 8 elements.

Size, Qwen3.8 27B (16 of 64 layers attend, 4 KV heads, head dim 256):

```
cache    per token   128K ctx   256K ctx
bf16     64 KiB      8.00 GiB   16.00 GiB
fp8      32 KiB      4.00 GiB    8.00 GiB
8-bit    34 KiB      4.25 GiB    8.50 GiB
6-bit    26 KiB      3.25 GiB    6.50 GiB
4-bit    18 KiB      2.25 GiB    4.50 GiB
3-bit    14 KiB      1.75 GiB    3.50 GiB
```

Decision (with the user): exllamav3's format, any width 2..8 for K and V separately; per-layer types later.

Design consequence: H32 mixes a whole group, so the tile loader (8 elements per chunk) cannot undo it. Instead,
as llama.cpp does with its wider rotation: store H*k and rotate Q by the same block-diagonal H32 (dot products
unchanged), and store H*v and apply H32 to the attention output. Both rotations sit in the attention wrapper, so
the loader only dequantizes and no model changes. Packing 8 consecutive values of b bits fills exactly b bytes, so
a chunk's data is b contiguous bytes; each row ends with its groups' f16 scales.

## Run 2 - 2026-10-17 11:40

Question: do the quantizing write, the dequantizing gather and the H32 rotation (`quant_cache_kernel.cu`,
`inference-paged-attn`) round-trip within the grid's error?

Command: `cargo nextest run -p inference-paged-attn --features cuda -E 'test(/quant_cache|kv_quant/)'`. The write
test sends 10 positions of 2 heads x 128 values (a smooth signal with a 12.0 outlier per row) through table blocks
2 then 0, plus a token with slot -1, and gathers them as f32. For each group it bounds the RMS error by half a grid
step of the rotated group, scale / 2^(bits-1) / 2, with 2% for f16 scales.

Result: all pass at (K, V) bits (8,8), (6,4), (5,3), (4,2), (7,5); rows past the sequences gather as zeros; the
rotation matches a CPU reference and is its own inverse. A first failure was the test's own slots: position 3 of a
sequence lives in table block 2 at offset 3 (slot 19), not slot 0.

Next: dequantization in fattn's mma tile loader, and Q / output rotation around fattn calls.

## Run 3 - 2026-10-17 13:10

Question: does attention over a quantized cache, through fattn's loads (Q and output rotated by H32) and through
the dequantizing gather, equal attention over the cache's dequantized values?

Command: `cargo nextest run -p inference-nn --features cuda -E 'test(/fattn_tests/)'`. The new cases fill caches
with random rows through the quantizing write, gather the whole cache back as the reference's K/V, and compare the
layer's output with the existing hand reference at its fp16/bf16 tolerances (4e-3 / 1.6e-2).

```
decode   head 128 q8/q8 f16; 128 q6/q4 bf16; 64 q5/q3 f16 window 100; 256 q4/q2 bf16; 512 q7/q5 f16;
         64 q8/q4 softcap 30 (no fattn kernel: served by the dequantizing gather)
prefill  128 q8/q8 bf16 dense; 256 q6/q4 f16 packed; 64 q5/q3 bf16 window 40; 512 q4/q2 f16 packed
```

Result: 16 of 16 pass (the 14 existing cases unchanged). Fixes on the way: the donor path took a quantized cache's
row bytes for its head size (now the query's, when the rows fit it), and fattn's device-pointer match lacked u8.

Next: per-model cache types through the API and the manager, then real-model quality.

## Run 4 - 2026-10-17 14:50

Question: what does each cache type cost in perplexity on a real model (Qwen3.5 0.8B, head dim 256, hybrid)?

Command: the perplexity example (`examples/rust/advanced/perplexity`) with new `--pa-cache-type`, `--prefill-chunk`
and `--cached-prefix` options over ~4,300 words of this repo's docs, 1,024-token chunks.

```
auto f8e4m3 q8 q6 q5 q4 q8/q4 q3 q2, --prefill-chunk 64    11.626606 +/- 7.542532 (all identical)
auto q4 q2, --cached-prefix 512                            11.178299 +/- 8.02385  (all identical)
```

The log confirms each cache type was allocated (`PagedAttention KV cache type is q4 (U8)`) and the scheduler's
64-token prefill chunks. The scores are identical because a prompt-scoring request runs its whole prompt in one
forward pass (`engine_logits.rs`: "The whole prompt runs in one forward pass") and does not reuse the prefix cache,
so its attention reads the K/V of the same pass and never the cache.

Implication: prompt scoring cannot measure a cache type. Decoding reads the cache every step, so the next measure is
greedy continuations against the bf16 cache's.

## Run 5 - 2026-10-17 15:30

Question: decoding reads the cache every step; how far do greedy continuations stay with the bf16 cache's?

Command: the test server (built from this branch), the model loaded through `/v1/manager/load` with each
`paged_cache_type` (exercising the per-model option), then `/v1/completions` at temperature 0, 160 tokens, over 20
passages of ~300 words from the repo's docs. Reported: continuations identical to bf16's, and the mean share of
tokens before the first difference (any difference sends the rest elsewhere, so this is a harsh measure).

```
                  Qwen3.5 0.8B (16 attention layers,    Qwen3 0.6B (28 layers,
                  4 KV heads, head 256, hybrid)          8 KV heads, head 128)
auto (reloaded)   20/20  100.0%                          20/20  100.0%
f8e4m3            12/20   68.8%                           3/20   25.5%
q8                15/20   84.6%                           6/20   51.7%
q6                 9/20   55.7%                           3/20   27.4%
q5                 5/20   44.4%                           2/20   17.2%
q4                 4/20   28.4%                           0/20    5.2%
q8/q4              6/20   36.8%                           2/20   23.1%
q3                 1/20   12.6%                           0/20    2.8%
q2                 1/20    6.4%                             -
```

A sample at 70 tokens (Qwen3.5): q8 reproduces bf16 exactly; q4, q3 and q2 stay fluent but drift into generic,
repeated table rows.

Reading: q8 beats fp8 on both models, but the dense Qwen3 is far more sensitive, and Qwen models carry a few very
large K channels. H32 spreads an outlier only across its own group of 32, whose scale it then dominates;
llama.cpp rotates K by a Hadamard as wide as the head (up to 256), and V by 64-wide blocks, which it found better.

Next: a rotation width per tensor (K across the head, V at 64) with scale groups still 32, measured the same way.

## Run 6 - 2026-10-17 17:20

Question: greedy agreement (Run 5) tells how chaotically a model reacts to any change, not how much quality a cache
type costs; fp8 alone dropped Qwen3 0.6B to 26%. What do the cache types cost in teacher-forced perplexity as
decoding sees the cache, and which rotation width is best?

Command: `ppl_decode.py` (scratch) against the test server: 10 passages of the repo's docs, each a 256-token prefix
then 96 scored positions. Position i is one `/v1/completions` request whose token prompt is the passage up to i
(`max_tokens` 1, top-20 logprobs): consecutive requests share their prefix, so prefix caching serves every full
block from the cache under test and only the tail is recomputed. A true token outside the top 20 is scored at the
20th logprob (it understates perplexity alike for every type). Rotation widths set through a temporary
`INFERENCE_KV_ROTATION=k,v` override, since removed.

Rotation widths, Qwen3 0.6B (bf16 cache: 7.3192):

```
K/V rotation    q8       q6       q4       q8/q4    q3
32/32           7.3301   7.4014   8.9311   7.3249   12.1908
64/64           7.3120   7.3210  10.0241   7.2832   51.9655
128/128         7.3359   7.3832   9.9937   7.2948   68.9585
K head, V 64    7.3481   7.3244  10.2403   7.3537   68.3272
```

At 6 bits and up the width is noise (about +-0.05); at 4 and 3 bits narrow wins by far. With a scale per group of
32, a wider rotation spreads an outlier channel's energy into every group's scale and coarsens them all; a 32-wide
one keeps it inside one group. This is the opposite of what Run 5 suggested; Run 5's measure could not tell.

Final policy, 32-wide rotation for K and V:

```
           Qwen3 0.6B (dense, 28 of 28 layers attend)   Qwen3.5 0.8B (hybrid, 16 of 64 layers attend)
           ppl       top-1    vs bf16                   ppl      top-1    vs bf16
auto       7.3192    56.7%                              5.1245   62.5%
f8e4m3     7.3628    56.9%    +0.6%                     5.1334   62.6%    +0.2%
q8         7.3301    56.9%    +0.1%                     5.1210   62.5%    -0.1%
q6         7.4014    57.2%    +1.1%                     5.1188   62.5%    -0.1%
q5         7.8450    55.7%    +7.2%                     5.1189   62.2%    -0.1%
q4         8.9311    53.2%    +22%                      5.1540   62.3%    +0.6%
q3        12.1908    46.6%    +67%                      5.2567   62.3%    +2.6%
q8/q4      7.3249    57.3%    +0.1%                     5.1357   62.6%    +0.2%
q6/q4      7.4929    57.0%    +2.4%                     5.1486   62.3%    +0.5%
q6/q3      7.3732    56.7%    +0.7%                     5.2193   62.3%    +1.9%
q5/q4      7.7785    56.0%    +6.3%                     5.1351   62.4%    +0.2%
```

Differences of about 0.1 on Qwen3 are noise (q6/q4 above q6/q3). K carries the sensitivity: V at 4 or 3 bits
costs little once K has 6-8. The hybrid model, whose few attention layers hold the cache, takes even q4 for both.

Implication: q8/q4 (0.81 bytes a value, 2.5x smaller than bf16) is lossless on both models and beats fp8 at a
similar size; q6/q4 and q6/q3 (0.69 and 0.63 bytes) cost ~1-2% on the dense model. Hybrid models such as
Qwen3.5/3.8 can go to q4 (0.56 bytes, 3.6x smaller).

## Run 7 - 2026-10-17 19:40

Question: what does a quantized cache cost in decode speed, and does it buy the context the 27B needed?

Command: `/v1/completions`, 256 tokens at temperature 0 after a ~6,250-token prompt (the second of two runs, so the
prompt is a prefix-cache hit and the prefill rate means nothing), on an RTX 3090; nsys with
`--cuda-graph-trace=node` for kernel times.

```
Qwen3 0.6B decode, tok/s              bf16    f8e4m3   q8      q8/q4   q4
first build                           333     327      187     190     200
GQA batching restored (below)         292     286      208     208     205
```

The profile showed the quantized decode on fattn's ncols1=8 instance while bf16 ran the GQA-batched
ncols1=4, ncols2=2 one: `use_gqa_opt` turns off when any operand's byte strides are not multiples of 16, and a
quantized row (136 bytes at q8, 72 at q4 for head 128) is not. Those checks are for 16-byte loads, which
quantized rows never get (they are read in 4-byte words), so they now skip quantized K/V. On the same instance the
kernel still takes 94 us a call against bf16's 48.5 us: the dequantizing loads cost more than they save in bytes.
Neither inlining the chunk loader nor reading it as aligned words instead of bytes moved the end-to-end rate;
the loader stays a call (inlined it added 2.85 MiB to the fattn library, as a call 0.70 MiB).

On the 27B the weights dominate:

```
Qwen3.8 27B IQ4_XS, ~6,500-token prompt   bf16 43.5 tok/s   q8/q4 41.1   q4 41.5   (-5.5%)

loads with ~5.2 GB left after the weights (RTX 3090, 24 GB):
bf16  128K   refused: needs 8192 MB          q8/q4  128K   loads
bf16  256K   refused: needs 16384 MB         q4     256K   loads (the model's full context)
```

Implication: ship. On large models the speed cost is a few percent and the cache gets 2.5-3.6x smaller; small
models whose decode is attention-bound lose up to ~30%. A faster loader (a group's 32 values per thread, one scale
load, fewer and wider loads) is the follow-up.

Size (`scripts/local_ci.sh --size-update`): the package grows 107.74 -> 108.58 MiB (+0.84), almost all of it the
fattn library's quantized loads (17.01 -> 17.68 MiB).

## Run 8 - 2026-10-18 10:30

Question: does sensitivity to K's bits vary enough by layer to make per-layer cache types worth building?

Command: a temporary per-layer override (branch `exp/kv-layer-sensitivity`: `INFERENCE_KV_LAYER_FILE` lines of
`layer type`, read at each cache allocation, so a reload through the manager takes a new map), Run 6's decode
perplexity on Qwen3 0.6B (28 layers, bf16 7.3192): every layer at q8 except one at q3 K (q3/q8).

```
all q8          7.3287          all q3/q8   12.1119
one layer at q3/q8, rest q8:
 0  8.6068   1 7.4596   2 7.3755   3 7.2855   4 7.4861   5 7.7921   6 7.3243
 7  7.4150   8 7.3596   9 7.3596  10 7.5254  11 7.3293  12 7.3329  13 7.3885
14  7.3034  15 7.3386  16 7.3544  17 7.2815  18 7.3014  19 7.3358  20 7.3061
21  7.3845  22 7.3776  23 7.3227  24 7.4091  25 7.3774  26 7.3786  27 7.3544
```

Layer 0 alone carries +1.28 of the +4.78 all layers cost; layer 5 +0.46, layer 10 +0.20, layers 1 and 4
~+0.15; the other 23 are within noise (~+-0.05). Protecting the few:

```
q3/q8, layer 0 at q8           9.4477
q3/q8, layers 0 and 5 at q8    8.5833
q4,    layer 0 at q8           7.7282   (q4 everywhere 8.9311)
q4,    layers 0 and 5 at q8    7.4566   (+1.9% vs bf16, at 0.60 bytes a value; q8/q4 is 0.81)
q3,    layers 0 and 5 at q8    8.5302   (q3 everywhere 12.1908)
```

Implication: per-layer types are worth building: two of 28 layers at q8 take q4 from +22% to +1.9% for 7%
more cache. Layer 0 (the attention sinks' layer) leads here and in the literature; the rest are model-specific,
so the feature wants explicit layer ranges and a way to find the sensitive layers (this sweep, as a tool).

## Run 9 - 2026-10-18 11:30

Question: does a cubic compander (exllamav3's idea) lower the cost of 3-5 bit widths? A grid level u in [-1, 1]
stands for `a*u^3 + (1 - a)*u` of the group's scale (levels crowd toward zero), encoded by inverting the cubic with
Newton steps, decoded in the gather and in fattn's loader; `a` set by a temporary `INFERENCE_KV_COMPAND` for widths
of 5 bits and below.

Command: Run 6's decode perplexity, Qwen3 0.6B.

```
            q5       q4       q3       q8/q4    q6/q3
uniform     7.8450   8.9311  12.1908   7.3249   7.3732   (Run 6)
a = 0.3     8.4922   8.2894  13.4450   7.3458   7.4947
a = 0.5     8.3429   9.1262  18.5622   7.3165   7.5112
```

Only q4 at a = 0.3 gains; the rest lose, 3-bit badly. After the 32-point rotation and absmax scaling a group's
values are not Gaussian enough for a fixed curve to place levels better than the uniform midpoint grid.

Implication: no compander. Protecting sensitive layers (Run 8: q4 with two layers at q8, 7.4566) beats any
grid change here.

## Run 10 - 2026-10-18 14:25

Question: are the sensitive layers specific to a model, to a family, or common? Run 8's sweep on three more
models: TinyLlama 1.1B (Llama, 22 layers, Q4_K_M GGUF), Qwen3.5 0.8B (hybrid, attention at layers 3..23 every 4th)
and Qwen2.5-Coder 3B (Qwen2, 36 layers, Q4_K_M GGUF); each layer alone at q3/q8, the rest q8. Tokenizers for the
GGUFs from their Hub repos' `tokenizer.json`.

```
                 bf16     all q8   all q3/q8   worst single layers at q3/q8
TinyLlama 1.1B   3.7170   3.7098   3.8814      8: 3.7354  15: 3.7276  19: 3.7274  (all others within 0.02)
Qwen3.5 0.8B     5.1245   5.1210   5.1235      11: 5.1382  19: 5.1302  (no layer matters)
Qwen2.5-C 3B     5.3613   5.3543  25.1143      0: 24.2777   then 2: 5.3884  12: 5.3886 (all others within 0.03)
Qwen3 0.6B       7.3192   7.3287  12.1119      0: 8.6068    5: 7.7921  10: 7.5254 (Run 8)
```

Layer 0 is the whole story in Qwen2.5-Coder (one layer at 3-bit K takes perplexity from 5.35 to 24.3, all but
3.5% of the full damage) and the largest part in Qwen3. TinyLlama spreads a small cost thinly over its later
layers with none standing out, and the hybrid Qwen3.5 does not care. No mid-stack layer repeats across models.

Implication: a general rule of keeping layer 0's K at 8 bits covers the catastrophic case seen and costs one
layer's share; mid-stack layers (Qwen3's 5) are model-specific and belong in profiles. The 27B sweeps (2- and
3-bit K per attention layer) are next.

## Run 11 - 2026-10-18 13:57

Question: per-layer sensitivity of the model this is for, Qwen3.8 27B (IQ4_XS GGUF, hybrid: attention at layers
3, 7, ..., 63), at 2- and 3-bit K.

Command: Run 8's sweep, each attention layer alone at q2/q8 and then q3/q8, the rest at q8.

```
bf16 2.6837   all q8 2.6848   all q2/q8 2.7221 (+1.4%)   all q3/q8 2.6903 (+0.2%)
one layer at q2/q8: 3 2.6894  7 2.6805  11 2.6856  15 2.6827  19 2.6853  23 2.6769  27 2.6853  31 2.6918
                    35 2.6800  39 2.6982  43 2.6932  47 2.6835  51 2.6879  55 2.6940  59 2.6891  63 2.6859
one layer at q3/q8: every layer within 2.6788..2.6913
```

No layer stands out: the worst single layer at 2-bit K (39) costs 0.5%, and all of them together 1.4%. Like the
0.8B, the 27B's few attention layers take very low K widths, and no layer 0 problem arises (its first attention
layer is 3).

Implication: for the 27B no profile is needed; a uniform low width is fine. Per-layer protection matters for dense
models (Qwen2/Qwen3: layer 0); the general rule keeps the first attention layer's K at 8 bits when a model's
type is below 6 bits, and profiles carry the model-specific rest.

## Run 12 - 2026-10-18 16:10

Question: does the per-layer feature (profiles, the general rule, `paged_cache_layers`) reproduce the sweeps'
gains through the real load path?

Command: the test server on this branch; loads through `/v1/manager/load` with `paged_cache_type` and
`paged_cache_layers`, scored as in Run 6.

Plans chosen at q4 (from the load log): Qwen3 0.6B (safetensors), Qwen3.8 27B (GGUF), TinyLlama 1.1B (GGUF) and
Qwen2.5-Coder 3B (GGUF) each matched their profile by fingerprint; SmolVLM 256M, without one, got
`layer 0 at q8 (the first cached layer)`.

```
Qwen3 0.6B        q4 (profile: 0 and 5, K floor only)   7.5791
                  q4, layers 0 and 5 at q8 (K and V)     7.4566   (= Run 8)
                  q4@uniform                             9.0072
Qwen2.5-Coder 3B  q3 (profile: layer 0)                  5.5416   (bf16 5.3613)
                  q3@uniform                            26.3844
```

Raising the protected layers' V as well as K is worth its few bytes (7.5791 -> 7.4566), so profiles and the rule
keep protected layers at q8 for both. `uniform` reads a little above Run 6's q4 (9.0072 vs 8.9311) since Run 7's
fattn change (GQA batching) altered rounding. `scripts/kv_layer_sweep.py`, the sweep through
`paged_cache_layers`, reproduces Run 10 on Qwen3.5 0.8B (baseline 5.1210, layer 3 at q3/q8 +0.10%) and prints
its profile entry.

## Run 13 - 2026-10-18 19:30

Question: why does a quantized cache's fattn take twice bf16's time (Run 7), and can the loads get it back?

Command: `cargo nextest run -p inference-fattn --features cuda --run-ignored only -E 'test(/bench::quant_vs_bf16/)'
--no-capture`, a new bench: paged K/V of random codes against bf16 at three model shapes; RTX 3090. ncu has no
counter access here, so the causes come from variants.

```
llama-8b shape (32/8 heads, head 128)     q8/q8 us   bf16 us
b1 decode kv 16384                        220.1      100.2    (q4/q4 215.0: bits do not matter)
b8 decode kv 16384                       2075.4      794.7
b1 prefill q 1024 kv 4096                2287.6     1126.2
same, chunk loader inlined                213.4 / 2005.3       (no change)
same, quantized loads replaced by zeros    24.6 /  161.3       (the loop and stores are not the cost)
dense fp8 (also synchronous loads)        121.6 / 1092.7
```

The width does not matter and prefill, which bf16 runs compute-bound, is as slow, so the loads are latency-bound:
each thread dequantized one 8-value chunk at a time (3 words plus its group's scale from another cache line), and
the next chunk's loads waited behind the previous chunk's shared-memory store. The new loader takes a 32-value
group per work item (its 4 chunks, one scale load) and issues a batch of groups' loads before any store; 4 groups
per thread helps decode kernels (<= 16 columns) and costs prefill ones registers, so wide kernels take 2.

```
ratio to bf16                       old              batch 2          batch 4          chosen (4 narrow, 2 wide)
qwen3-0.6b b1 decode kv 16384       2.22             1.08             0.87             0.87
qwen3-0.6b b8 decode kv 16384       2.41             1.14             0.89             0.82
qwen3.5-0.8b b1 decode kv 16384     2.01             1.02             0.90             0.90
llama-8b b1 decode kv 16384         2.20             1.03             0.83             0.83
llama-8b prefill q 1024             2.03             1.20             1.27 (abs +24%)  1.19
qwen3.5-0.8b prefill q 1024         2.02             1.21             1.27             1.24
```

End to end (Run 7's harness, ~6,250-token prompt, 256 tokens):

```
Qwen3 0.6B decode tok/s       bf16 334.2   q8 330.3   q8/q4 332.9   q4 332.9     (Run 7: bf16 292, q8 208)
Qwen3.8 27B IQ4_XS            bf16  50.0   q8/q4 49.7  q4   49.9                (Run 7: -5.5%)
```

Implication: a quantized cache now decodes at bf16's rate (the kernel alone is 10-25% faster); prefill over it
stays 1.0-1.24x bf16's, where bf16's cp.async pipelining has no quantized counterpart yet.
