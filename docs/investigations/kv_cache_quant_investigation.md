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
