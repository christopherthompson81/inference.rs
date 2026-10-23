# Qwen3.5 UQFF round trip investigation

A Qwen3.5 (and Qwen3.6, Qwen3.8-Flash-Next) model loaded with ISQ and `write_uqff` decoded differently from a reload of
the UQFF it wrote, with no error. Found while adding a UQFF round-trip test for Qwen3.8-Flash-Next.

## Run 1 - 2026-10-22 21:10

Question: is the drift Flash-Next's, or Qwen3.5's?

Tiny random checkpoints, CPU f32, `with_isq(Q8_0).write_uqff(..)`, then `UqffTextModelBuilder` over the written file;
greedy traces compared.

```
Flash-Next tiny: different tokens from step 0
  same, dropping the indexer, then PLE, then hyper-connections, then the sigmoid gate: still different
dense Qwen3.5 tiny: same tokens, logprobs off by up to 0.14
  isq    [(9, -0.5139), (193, -0.7315), (189, -0.1607), ...]
  reload [(9, -0.4354), (193, -0.8410), (189, -0.1788), ...]
llama tiny: identical logprobs
```

The UQFF report lists every layer (33 q8_0 for dense Qwen3.5), so nothing is missing from the file. Implication: a
Qwen3.5 problem in what the live model runs, not in what is written.

## Run 2 - 2026-10-22 21:20

Question: which layers does the live model run differently from the reload?

The writer (`pipeline/isq.rs` `write_uqff_type`) quantizes each tracked layer from its source and swaps it into the
live model's cell (`module.ct.replace`). Anything built from those layers at load keeps the old weights, so a merged
projection would keep running unquantized. Debug prints at each packing and merge site, per load:

```
sizing pass:            gdn qkv_z packed=true (x3), attention packed=true
live (write) load:      gdn qkv_z packed=false, gdn b/a merged unquantized (x3), attention packed=false
UQFF reload:            gdn qkv_z packed=false, gdn b/a apart (x3), attention packed=false
```

Dead ends on the way: an all-full-attention variant does not load ("Internal server error"), and an ISQ topology cannot
isolate one layer group, since a UQFF write needs a default ISQ type.

Cause: the packed loader skips packing under pending ISQ, but GDN's `in_proj_b`/`in_proj_a` fallback,
`MergedDenseProjection::merge_unquantized` (there for a GGUF's dense `ssm_beta`/`ssm_alpha`), has no such check. In the
write flow they load unquantized, merge into one f32 projection that the forward always prefers, and the writer's swap
of the two tracked layers never reaches it. The reload sees quantized layers, so the merge declines and it runs them
quantized.

## Run 3 - 2026-10-22 21:30

Fix: skip the merge when ISQ is pending (`get_immediate_isq()` set, or either projection matches an immediate-ISQ
predicate), the packed loader's own condition. GGUF loads, which have no pending ISQ, still merge.

```
dense Qwen3.5 tiny: isq == reload to every printed digit
cargo nextest run -p inference --features cuda -E 'test(/^qwen3_5_text_tiny::/)'   21 passed
negative control (guard disabled): an_isq_load_and_a_reload_of_its_uqff_decode_alike fails
Flash-Next tiny round trip (qwen4exp branch): passes
```
