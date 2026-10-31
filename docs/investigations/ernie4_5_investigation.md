# ERNIE 4.5 (#570)

Baidu's ERNIE 4.5 text models (`Ernie4_5ForCausalLM`, `Ernie4_5_MoeForCausalLM`; GGUF `ernie4_5`, `ernie4_5-moe`) as
specs on the shared decoder (`crates/inference-models-other/src/ernie4_5.rs`): 1D RoPE over adjacent pairs and F32 RMS
norms, as PaddleOCR-VL's language model has them without its M-RoPE. What the decoder's MoE layer lacked:
`MoeRouting::BiasedTopK` (an f32 router, the top k of softmax plus `moe_statics.e_score_correction_bias`, weighed by
the softmax renormalized over a sum clamped at `moe_norm_min`, the shared `moe_router_topk` op already doing all of
it), and a shared expert that is ungated and named `shared_experts` (`SharedExpertSpec`, where Qwen2-MoE's is gated).

## Run 1 - 2026-10-30 18:50

**Question:** do our dense and MoE models, tokenizer and chat prompt match transformers, from safetensors (both expert
layouts) and llama.cpp's GGUF?

**Command:** `scripts/ernie4_5_tiny_parity.py --llama-cpp ~/Programming/llama.cpp` (transformers 5.19.0, llama.cpp
970b51ce7), on CPU in f32 and on CUDA in bf16 and f32. Seeded 3-layer models over ERNIE's real 103,424-token
vocabulary (tokenizer files from baidu/ERNIE-4.5-0.3B-PT at b565cf6c): dense, and MoE with layer 0 dense, 8 experts
top 3, two shared experts and a random selection bias; the MoE also with its experts stacked; both converted to f32
GGUFs, whose tokenizer llama.cpp rebuilds from `tokenizer.model`.

**Finding:**
- CPU f32, all five loads: mean KL under 2e-7, max logit diff 0.0025-0.0026, the same 12 greedy tokens; the README
  tokenizes to transformers' 5,695 tokens and a system + one-turn chat to its 38, from the GGUFs' SentencePiece vocab
  too. The MoE's top-1 agreement is 0.947 (one of 19 positions) at a KL of 3.5e-8: a near tie.
- CUDA bf16: KL 4.8e-5 to 8.0e-5, max logit diff under 0.09; CUDA f32: KL under 3e-7, max logit diff 0.0000 for all
  five, so the bf16 differences are precision.
- The script's first chat reference read `apply_chat_template(tokenize=True)`'s mapping as a list (2 "tokens"); it
  renders the text and encodes it instead.
- `ernie4_5_moe_prefill` fails with the selection bias dropped from routing.

**Next:** published quantized GGUFs against llama.cpp's perplexity.

## Run 2 - 2026-10-30 19:14

**Question:** do published quantized ERNIE 4.5 GGUFs load and score as llama.cpp scores them?

**Command:** `scripts/gguf_perplexity_parity.sh ~/Programming/llama.cpp/build_cuda/bin/llama-perplexity <dir>`
(llama.cpp 970b51ce7, README.md, ctx 512, one chunk) on unsloth's ERNIE-4.5-0.3B-PT Q8_0 (0.36 GiB) and
ERNIE-4.5-21B-A3B-PT Q4_K_M (12.4 GiB); for the 0.3B also `llama-perplexity --kl-divergence-base` and the perplexity
example's `--kld-base` over llama.cpp's own tokens; then one sampled chat with the 21B.

**Finding:**

| file | llama.cpp | ours | change |
|---|---|---|---|
| ERNIE-4.5-0.3B-PT Q8_0 | 1.6346 | 1.6099 | -0.0151 |
| ERNIE-4.5-21B-A3B-PT Q4_K_M | 1.3165 | 1.3197 | +0.0024 |

- Ours scores the 0.3B lower than llama.cpp does. The two sides tokenize the README differently in one token:
  llama.cpp's SentencePiece path adds a leading space (`' <'` for `'<'`), as its converter writes `add_space_prefix` only
  when `tokenizer_config.json` sets `add_prefix_space`, which ERNIE's does not, and ERNIE's tokenizer adds none; ours
  follows transformers. Scored on llama.cpp's own tokens: 1.6400 against 1.6346, KLD mean 0.0011, top-1 99.61%.
- `gguf_perplexity_parity.sh` took the change's absolute value, so a lower perplexity counted against us like a higher
  one; it now fails only when ours is higher past the tolerance, and prints the change signed.
- "Name three primary colors, comma separated." (temperature 0.7) on the 21B, through the GGUF's chat template: "red,
  yellow, blue", finish `stop` after 6 tokens.

**Next:** none for the text models. The MTP layer (`mtp_block.*`, which llama.cpp's converter drops) as a draft model
is the issue's optional follow-up.

## Run 3 - 2026-10-30 19:22

**Question:** can a load cut ERNIE's 131,072-token context (`max_model_len`), which every loader but Qwen3.5's and
Gemma 4's refused?

**Command:** the 0.3B Q8_0 through the C ABI with `runtime.max_model_len` 4096 and 1,048,576, one sampled completion.

**Finding:** before, "max_model_len=4096 is not supported by this model loader". A blanket default would be wrong for
some loaders (Phi-3's LongRoPE scales attention by `max_position_embeddings`), so `cap_context_length` is opt-in: it
cuts the config's context key, refusing zero or a length past the trained one, and both ERNIE loaders use it (plain
RoPE, whose tables only shrink). 4096 loads and completes ("red, green, and blue."); 1,048,576 fails with
"max_model_len=1048576 is outside the model's context of 131072 tokens".

**Next:** the MTP layer as a draft model.
