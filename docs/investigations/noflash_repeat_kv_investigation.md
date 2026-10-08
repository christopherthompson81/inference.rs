# #252: does repeat_kv dominate the noflash attention fallback?

## Run 1 - 2026-10-08 08:25

Question: how much of `Sdpa::run_attention_noflash` (`crates/inference-nn/src/attention/mod.rs`, the CUDA cuBLASLt
path) is the GQA `repeat_kv` copy of K and V? Since #282 every CUDA build tries flash first, so this path now serves
custom masks, F32 activations and head dims flash does not take.

Command: a temporary CUDA test (not committed) timing, on the RTX 3090 (sm_86), 8 query heads over 2 KV heads, head
dim 128, a zero custom mask, mean of 5 runs after a warmup: the GQA call; the same call with K/V pre-expanded
(`n_kv_groups` 1); and `repeat_kv` of K and V alone.

Raw:
- BF16, 2048 tokens: GQA 5.02 ms, pre-expanded 4.98 ms, repeat_kv 0.05 ms
- F32, 2048 tokens: GQA 4.51 ms, pre-expanded 4.28 ms, repeat_kv 0.05 ms
- BF16, 4096 tokens: GQA 16.49 ms, pre-expanded 16.38 ms, repeat_kv 0.08 ms

Conclusion: the copy is 0.5-5% of the fallback; avoiding it (grouping the queries per KV head, which also changes the
score rows the causal-mask kernel and the q chunking assume) would save at most ~0.2 ms here. The ~200 ms of copies
and casts in the pre-#282 measurement (gguf_iq_trellis_quants_investigation Runs 13-14) is not repeat_kv on today's
code. The fallback's cost is being an unfused attention at all; the lever is routing more cases to flash (custom
masks), not this copy.
