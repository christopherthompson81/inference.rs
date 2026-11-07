# Decode graphs for recurrent-only models (#853)

Models with no attention layer (RWKV-6/7, ARWKV, QRWKV6, Mamba, Falcon-Mamba, Mamba-2) ran without paged attention,
and decode graphs only capture on the paged path, so every launch was paid per token. Background:
`state_space_models_investigation.md` Run 9 (ARWKV-R1 1.5B Q8_0: 166.7 tok/s decode against llama.cpp's 207.5, about
1,050 kernels a token, 4.2 ms GPU against 6.0 ms wall).

## Run 1 - 2026-11-07 15:45

Question: can these models run on the paged path with zero attention layers, so the existing capture applies (as it
does for Granite's and Qwen3-Next's recurrent layers)?

Read: `DecoderPipeline::try_cuda_decode_graph_forward` skips without paged metadata
(`CudaGraphDispatchReason::PagedAttentionUnavailable`); the hybrid pool's CUDA gather and scatter take device slot
indices (`scatter_*_with_host_indices` uses the device tensor on CUDA), which a graph replays; Granite opts in through
`model_config` (its `HybridPagedKvCacheConfig`) and `supports_cuda_decode_graphs`.

Changes:

- The planner returned "attempt to divide by zero" with no KV bytes per token (Mamba-2's original reason to opt out).
  With no attention layer it now returns blocks for the model's context without reading memory: the blocks hold no KV.
- The loaders no longer opt out of paged attention.
- `RwkvModel::model_config` reports every layer recurrent (no paged KV, the recurrent bytes per sequence), and
  `supports_cuda_decode_graphs` is true. Mamba and Mamba-2 run on Granite's model, which already reports both.

First try: the model's own `ModelConfigMetadata` (one KV head per layer) sized the cache to 4 blocks, "Sequence 2 with
97 tokens exceeds the total KV cache capacity of 96 tokens". With `model_config` overridden: 4 graphs captured in
121 ms, then "4296 tokens exceeds the total KV cache capacity of 4096": the zero-KV blocks took the default token
budget. They take the model's context now.

```
ARWKV-R1 1.5B Q8_0 on the 3090 (hybrid_bench: decode 400 x2, ~4.3K prefill x2, thinking off)
  before:  decode 166.5 / 166.9 tok/s, prefill 14,381 / 14,358 tok/s
  after:   decode 250.6 / 250.6 tok/s, prefill 11,193 / 11,631 tok/s    (llama.cpp tg128 207.45, pp4096 10,061)
  the chat answer is the same text
```

Prefill dropped by a fifth: the paged scheduler chunks prompts (max_prefill_chunk_tokens=512 in the log). Next: the
other recurrent models, then whether the chunk size should follow from having no KV.

## Run 2 - 2026-11-07 16:10

Question: do the other recurrent-only models capture, and what does the paged path cost their prefill?

First pass over RWKV-7 1.5B Q8_0, RWKV-6 1.6B BF16, QRWKV6 7B Q4_K_M, Mamba-Codestral 7B Q8_0 and Falcon-Mamba 7B
Q4_K_M. Dead ends on the way:

- Mamba-Codestral's GGUF has no chat template, so the chat bench got 400s; the bench takes a completion mode now.
- A Falcon-Mamba server left from a manual run held the port, so two "runs" measured it again. The bench now checks
  that its own server answers.
- Falcon-Mamba and Codestral did not capture ("model_unsupported" in `inference_cuda_graph_dispatch_total`): Granite's
  model opts in, but `HybridPagedKvCacheConfig` reported every layer attended, so the capture check ran the attention
  backend test over the Mamba layers. A layer without paged KV is unattended now (the check is the only reader).
- The 127.6 tok/s Codestral decode in the SSM log was not this bench; like for like (same build, `--paged-attn off`)
  it is 83.

The paged scheduler chunks prompts at 512 tokens, which cost RWKV-7 a fifth of its prefill (13,090 -> 10,900).
`--max-prefill-chunk-tokens 4096`: 13,350 / 13,469, decode unchanged (285). A model with no paged KV now plans its
prompt chunk at the default batch budget (4096) when the load planned none.

| on the 3090 | decode before | decode after | prefill before | prefill after | llama.cpp tg128 / pp4096 |
|---|---|---|---|---|---|
| ARWKV-R1 1.5B Q8_0 | 166.7 | 248.9 | 14,370 | 13,649 / 14,371 | 207.45 / 10,061 |
| RWKV-7 1.5B Q8_0 | 219.5 | 274.2 / 276.4 | 13,090 | 13,088 / 13,239 | 209.59 / 9,844 |
| RWKV-6 1.6B BF16 | 165 | 196.9 / 197.5 | 10,538 | 10,545 / 10,704 | 169.18 / 3,826 |
| QRWKV6 7B Q4_K_M | 101.1 | 122.1 / 122.6 | 4,625 | 4,923 / 4,919 | 107.89 / 1,601 |
| Mamba-Codestral 7B Q8_0, `--paged-attn off` as before | 83.2 / 82.8 | 88.7 / 88.9 | 3,737 / 3,794 | 3,495 / 3,598 | |
| Falcon-Mamba 7B Q4_K_M, `--paged-attn off` as before | 104.1 / 103.5 | 120.6 / 120.8 | 3,875 | 3,721 / 3,739 | 107.5 / 3,324 |

"Before" for the RWKV rows is the SSM log's numbers (non-paged, same bench prompts). Every model captures 4 graphs.
The Mamba models keep a small prefill cost on the paged path (4-6%); Codestral's ~5K-token prompt arrives in two
4096 chunks where the fixed scheduler ran it whole. The same answers as before for every chat check.
