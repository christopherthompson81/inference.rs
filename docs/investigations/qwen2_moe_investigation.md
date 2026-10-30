# Qwen2-MoE (#562)

Qwen1.5-MoE and Qwen2-57B-A14B (`Qwen2MoeForCausalLM`, GGUF `qwen2moe`) as a spec on the shared decoder
(`crates/inference-models-qwen/src/qwen2_moe.rs`), with what the decoder's MoE layer lacked: a shared expert every
token runs, scaled by `sigmoid(shared_expert_gate(x))` (`MoeSpec::shared_expert`).

## Run 1 - 2026-10-30 16:51

**Question:** does our Qwen2-MoE match transformers, from both expert layouts and from llama.cpp's GGUF?

**Command:** `scripts/qwen2_moe_tiny_parity.py --llama-cpp ~/Programming/llama.cpp` (transformers 5.19.0, llama.cpp
970b51ce7), on CPU in f32 and with `--device cuda:0` in bf16. A seeded 3-layer checkpoint (4 experts, top 2,
`norm_topk_prob: false`, shared expert 48 wide, layer 1 dense, q/k/v bias and norms perturbed), saved one expert per
tensor; a copy with the experts stacked (`gate_up_proj`, `down_proj`); an all-MoE copy converted to an f32 GGUF.

**Finding:**
- transformers 5.19 saves the experts one by one, though it holds them stacked, so the stacked copy is written by the
  script.
- llama.cpp's converter fails on a dense layer (`Can not map tensor 'model.layers.1.mlp.down_proj.weight'`), and its
  graph runs every layer as MoE, hence the all-MoE copy for the GGUF. Its converter also rejects the fixture
  tokenizer (Qwen2Tokenizer adds a special past the 266-token vocab; the pre-tokenizer hash is unknown), so the script
  writes a `PreTrainedTokenizerFast` config and names the pre-tokenizer `qwen2`.
- CPU f32: per-expert, stacked and GGUF all at mean KL < 1e-7, top-1 agreement 1.000, max logit diff 0.0013-0.0015,
  and the same 12 greedy tokens.
- CUDA bf16: mean KL 3.2e-5 (GGUF 2.8e-5), top-1 1.000, max logit diff 0.03. The greedy run parts from transformers'
  at step 6 (140 vs 67); at f32 on CUDA it matches token for token (max logit diff 0.0000), so the script accepts a
  bf16 continuation whose every token scores within 0.25 of transformers' best, as `qwen4exp_tiny_parity.py` does.
- The pinned prefill (`qwen2_moe_tests.rs`) fails with the shared expert off (`tensor names moved`).

**Next:** a published quantized GGUF against llama.cpp's perplexity.

## Run 2 - 2026-10-30 17:03

**Question:** does a published quantized Qwen2-MoE GGUF load and score as llama.cpp scores it?

**Command:** `scripts/gguf_perplexity_parity.sh ~/Programming/llama.cpp/build_cuda/bin/llama-perplexity <dir>`
(llama.cpp 970b51ce7, README.md, ctx 512, one chunk) on a community Q4_K_M of Qwen1.5-MoE-A2.7B-Chat (60 experts,
top 4, shared expert 5632 wide, 24 layers, 8.8 GiB).

**Finding:**

| file | llama.cpp | ours | drift |
|---|---|---|---|
| Qwen1.5-MoE-A2.7B-Chat Q4_K_M | 2.3678 | 2.3623 | 0.0023 |

**Next:** none for Qwen2-MoE; Qwen 1 (`QWenLMHeadModel`, `qwen.tiktoken`) is the issue's other half.
