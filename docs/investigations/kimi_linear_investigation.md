# Kimi-Linear (#553)

Moonshot's Kimi-Linear-48B-A3B: 27 layers, 20 Kimi Delta Attention (KDA) and 7 MLA (`full_attn_layers` 4, 8, ..., 27,
one-based), DeepSeek-V3-style MoE (256 experts, top 8, one shared, sigmoid router with a correction bias,
`routed_scaling_factor` 2.446, renormalized; layer 0 dense). Hidden 2304; KDA 32 heads of 128 with a short conv of 4;
MLA 32 heads, no q compression, `kv_lora_rank` 512, nope 128 + rope 64 but no RoPE applied (`mla_use_nope`),
softmax scale `192^-0.5`.

## Run 1 - 2026-11-07 17:10

Question: what does KDA need beyond what GDN, MLA and the DeepSeek MoE already give us, and where should it live?

Read: transformers `models/kimi_linear/modeling_kimi_linear.py`; llama.cpp 970b51ce7 `src/models/kimi-linear.cpp`,
`delta-net-base.cpp` (its delta net takes a `[head_dim, n_head]` gate for KDA beside GDN's `[1, n_head]`),
`conversion/kimi_linear.py`; moonshotai/Kimi-Linear-48B-A3B-Instruct's config and tensor index.

- KDA per token and head: q, k, v each through its own depthwise causal conv (kernel 4) and SiLU; q and k L2-normed
  (`x / sqrt(sum x^2 + 1e-6)`, fla's form, not a max); q scaled by `K^-0.5`; the decay per key channel
  `g = -exp(A_log[h]) * softplus(f_b(f_a(x)) + dt_bias)` (softplus linear past 20); `beta = sigmoid(b_proj(x))` per
  head. Then `S = diag(exp(g)) S`, `S += k (beta (v - k^T S))`, `o = q^T S`. Output: a per-head RMSNorm with weight
  `o_norm` times `sigmoid(g_b(g_a(x)))`, then `o_proj`.
- The checkpoint keeps `q_conv1d`, `k_conv1d`, `v_conv1d` apart (transformers' own class packs one `conv1d`), the
  experts as `block_sparse_moe.experts.N.w1/w2/w3` with `gate.weight`, `gate.e_score_correction_bias` and
  `shared_experts`.
- No MTP layers (`num_nextn_predict_layers` 0), so no speculative state paths.
- Our GDN layer (`inference-nn/src/gdn/`) has fused chunked, decode, speculative and transition kernels, all with one
  gate per head. Threading a per-channel gate through every one is the large part of reusing it.
- The shared decoder (`CausalLm`) holds a plain KV cache, so hybrids (Qwen3-Next, Granite, LFM2) are bespoke models.

Plan:

1. A KDA recurrence op in inference-nn (it is also #552's and #555's prerequisite): CPU composite and one fused CUDA
   kernel (a block per sequence and head, a thread per value column holding it; the norms, decay, beta, the update,
   the output and the gated RMSNorm), reading the hybrid pool's rows in place through the checkout's slots.
2. A KDA layer on GDN's conv (the three convs as one over packed q|k|v) and `GdnLayerCache` checkout and commit, its
   state `Opaque [heads, K, V]` (key major on every device).
3. `kimi_linear.rs`: a bespoke hybrid model, MLA from the DeepSeek family with RoPE optional, the family's MoE with the
   checkpoint's expert names.
4. Loader, GGUF binding (`kimi-linear`), tokenizer, tiny test; parity against llama.cpp on bartowski's Q3_K_M (21.1
   GiB, resident on the 3090 for both), then speed.

## Run 2 - 2026-11-07 17:20

Question: do the op, the layer and the model match transformers, and run on CUDA?

Built (as planned in Run 1): `ops::kda` (CPU composite and the CUDA kernel `kda.cu`, heads of 64 or 128, rows through
the checkout's slots), `gdn::KimiDeltaAttention` on GDN's conv and `GdnLayerCache`, `kimi_linear.rs` with MLA from the
DeepSeek family (RoPE optional now: `MlaAttention::load` takes none) and the family's MoE (its expert names a
`MoeSpec` field now, Mixtral's for Kimi), the loader, the `kimi-linear` GGUF binding and config.

```
gpu case cuda_kda_matches_the_composite (64 and 128, two sequences in rows 2 and 0 of a three-row pool):  within 1e-4
scripts/kimi_linear_tiny_parity.py (5 layers, MLA at 3 and 5; CPU f32; 51 tokens):
  mean KL 9.61e-08 (worst 4.14e-07), top-1 1.000, max logit diff 0.0023; the 12 greedy tokens equal
```

The script writes transformers' random model in Moonshot's layout and reloads it into transformers (its conversion
mapping packs the convs and merges the experts) to check the layout first. The first CUDA runs failed three ways, none
of them Kimi's own code:

- FlashInfer's MLA decode is compiled for latent and rope widths of 512 and 64 only (DeepSeek's and Kimi's); the tiny
  test and the script now use those widths.
- "flashinfer_mla_decode metadata tensor shapes are invalid" (now reporting the shapes: 1 row, 2 tiles). The decode
  tile plan splits a long context into tiles when the batch times the heads leaves SMs idle, and pads rows to a graph
  capacity; MLA's decode kernel reads one tile a row. The tiny model's 2 heads split at once; DeepSeek's 128 would
  split past a few thousand tokens of context. A standard-metadata decode (MLA's) now builds its plan unsplit.
- "`c` end offset must be 32": the eager cuBLASLt attention wrote its output into q's buffer, which assumes q's head
  width equals v's; MLA's values are narrower (16 against 80 here, 128 against 192 in DeepSeek and Kimi). It reuses
  the buffer only when the widths match.

`kimi_linear_tiny` (decode against one prefill, the prefix cache, CUDA against the CPU with the decode's graph replays
scored by a CPU prefill) passes on both.

The 21 GiB Q3_K_M downloads at 1 MB/s for its first 15 minutes, then about 5 MB/s.

## Run 3 - 2026-11-07 17:30

Question: the tokenizer. The checkpoint ships `tiktoken.model` with `tokenization_kimi.py`, no tokenizer.json; llama.cpp's
GGUF names the pre-tokenizer `kimi-k2`.

- `tokenization_kimi.py`: the ranks, then 256 + 2 specials named from `tokenizer_config.json`'s `added_tokens_decoder`
  (`<|reserved_token_N|>` otherwise), split by a pattern whose letter runs leave out Han (character class
  intersections, `[...&&[^\p{Han}]]`), Han runs a piece of their own.
- llama.cpp's converter rebuilds BPE merges from the ranks; its `kimi-k2` split is a hand-written copy of that pattern
  (`unicode_regex_split_custom_kimi_k2`). The regex crate behind our split takes the intersections as written.

Built: `tiktoken.model` as a native tokenizer (Qwen 1's tiktoken loader, its ranks read shared), the pattern as
`inference_gguf::KIMI_K2_REGEX` for both it and the GGUF's `kimi-k2`.

```
ours (the checkpoint's tiktoken.model and tokenizer_config.json) against tiktoken 0.14 with tokenization_kimi.py's
pat_str and specials, add_special_tokens off:  6 of 6 texts equal (English with contractions, Chinese with digits,
whitespace runs, accented Latin with Japanese and Korean, the chat specials, code with emoji)
```

## Run 4 - 2026-11-07 18:00

Question: does the real model (bartowski's Q3_K_M, 21.1 GiB) run, score as llama.cpp does, and how fast?

The first loads failed four ways before the model ran:

- "No known expert checkpoint layout under `model.layers.1.block_sparse_moe.experts`": stacked GGUF experts keep
  `gate_proj`/`up_proj`/`down_proj` whatever the checkpoint's own names (Mixtral's are the same); the binding moved
  them under `block_sparse_moe` and renamed them `w1/w2/w3`. It moves them only now.
- "a quantized KV cache needs equal K and V head dims in the HND cache layout; layer 3 uses Mla": the MoE offload
  planner, moving experts off the GPU, quantizes an `auto` KV cache to q3; an MLA latent cache cannot be. A model with
  an MLA layer keeps its cache dtype.
- "flashinfer_mla_decode expects matching q_nope/q_pe dtype, got F32 and BF16", then "dtype mismatch in add/cat": the
  split `k_b`/`v_b` projections (Q5_1/Q5_K here) dequantize to a dense f32 weight and returned f32; they return the
  input's dtype now. DeepSeek's GGUFs would meet the same on this path.

```
chat (thinking off): "What is the capital of France?" -> "The capital of France is Paris."
gguf_perplexity_parity.sh, README.md:                       ref 2.3110, ours 2.3063 (-0.20%)
  reference/openai-compatibility.md:                        ref 13.7525, ours 13.5254 (-1.65%)
```

Ours scores lower on both; the tiny checkpoint matched transformers (Run 2), and llama.cpp L2-normalizes q and k with
the RMSNorm eps (1e-5) where fla adds 1e-6, one difference on its side.

| on the 3090 | ours | llama.cpp `llama-bench -fa 1 -ngl 99` |
|---|---|---|
| decode, 400 tokens | 55.4 / 91.6 tok/s (the expert cache warming) | tg128 164.35 |
| prefill, 3,915 tokens | 1,038 / 1,032 tok/s | pp4096 2,621.85 |

The planner keeps 228 of 256 experts a layer on the GPU and the rest in the file mapping (pageable: registering it
with the device is not supported here), beside a 32K-token context, 1,536 MB for post-load components and 1,600 MB
for runtime and activations; llama.cpp holds every expert. `--moe-expert-cache 256` is refused (the map "exceeds total
capacity by 6286MB"); `--pa-context-len 4096` frees enough for 248 slots but drops the activation reserve to 64 MB and
the 3.9K-token prompt then runs out of memory. Fitting this model resident is the planner's reservations, not Kimi's
code; next for it, outside this step.
