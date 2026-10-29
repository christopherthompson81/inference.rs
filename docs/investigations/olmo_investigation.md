# OLMo family (#569)

OLMo 1, OLMo 2, Olmo 3 and OLMoE as specs on the shared decoder (`crates/inference-models-other/src/olmo.rs`), with what the decoder lacked: a weightless LayerNorm (`NormKind::LayerNoAffine`), `clip_qkv`, a q/k norm over the whole projection before the head split (`QkNorm::FullWidth`), and post-norm layers (`NormNames.input = None`: each sublayer's output is normed before its residual add).

## Run 1 - 2026-10-29 09:14

**Question:** do OLMo 1 and OLMo 2 match transformers?

**Command:** `olmo/ref.py` (transformers 5.19.0, f32 CPU) and `olmo/ours.py` (Python SDK, `Plain`, f32 CPU), last-token logits over the first 300 tokens of README.md, on `allenai/OLMo-1B-0724-hf` (`OlmoForCausalLM`, `clip_qkv` 8) and `allenai/OLMo-2-0425-1B-Instruct` (`Olmo2ForCausalLM`).

**Finding:**

| checkpoint | max abs | cosine | logit scale | top 5 |
|---|---|---|---|---|
| OLMo-1B-0724 | 4.2e-3 | 0.9999999 | 22.2 | identical |
| OLMo-2-0425-1B | 6.8e-3 | 0.9999998 | 19.5 | identical |

**Next:** OLMoE and Olmo 3 (sliding layers need a prompt past the 4096 window) once their weights are down.

## Run 2 - 2026-10-29 09:30

**Question:** do llama.cpp's OLMo 1 and OLMo 2 GGUFs load and score as llama.cpp does?

**Command:** `convert_hf_to_gguf.py --outtype q8_0` (llama.cpp 970b51ce7) on both checkpoints, then `scripts/gguf_perplexity_parity.sh` (README.md, ctx 512, one chunk).

**Finding:** OLMo 1's GGUF first failed: `GGUF BPE pre-tokenizer 'olmo' is not supported`. llama.cpp's `olmo` shares its GPT-2 case (same regex, no other flags), so it joins our `gpt-2` profile. llama.cpp permutes OLMo 1's q/k as it does Llama's (adjacent pairing) and leaves OLMo 2's and OLMoE's in Hugging Face's half-split order; OLMo 1's norms carry no tensors in the GGUF, and `attention.clamp_kqv` becomes `clip_qkv`. Then:

| file | llama.cpp | ours | drift |
|---|---|---|---|
| OLMo-1B-0724 Q8_0 | 1.4239 | 1.4264 | 0.0018 |
| OLMo-2-0425-1B Q8_0 | 4.6537 | 4.6315 | 0.0048 |

**Next:** OLMoE and Olmo 3 from safetensors and GGUF.

## Run 3 - 2026-10-29 09:45

**Question:** does OLMoE match transformers?

**Command:** as Run 1 on `allenai/OLMoE-1B-7B-0125-Instruct` (64 experts, top 8, `norm_topk_prob: false`), at 8, 40 and 300 prompt tokens; then transformers' router probabilities over the 300-token prompt (a hook on each layer's `mlp.gate`), measuring the gap between each token's 8th and 9th expert.

**Finding:**

| prompt tokens | max abs | cosine | top 5 |
|---|---|---|---|
| 8 | 6.5e-3 | 1.0 | identical |
| 40 | 6.7e-3 | 0.9999999 | identical |
| 300 | 8.2e-2 | 0.999988 | identical |

The 300-token gap is routing at the top-8 boundary: in transformers' own run the smallest 8th-to-9th probability gap is 3.6e-7 (layer 13, token 78), 30 cutoffs are closer than 1e-5 and 268 closer than 1e-4, where f32 rounding can choose another expert for that token.

**Next:** Olmo 3 past its 4096-token window.

## Run 4 - 2026-10-29 09:45

**Question:** does Olmo 3 (3 sliding layers of 4096 to 1 full, YaRN factor 8 on the full layers, plain RoPE on the sliding ones) match transformers past its window?

**Command:** as Run 1 on `allenai/Olmo-3-7B-Instruct`, 4600 prompt tokens (README.md repeated), f32 CPU; control: ours with every layer forced to `full_attention`.

**Finding:** max abs 1.1e-2, cosine 0.9999999, top 5 identical. The control gives max abs 12.2, cosine 0.38 and another argmax, so the prompt runs past the window and the sliding layers' masks and RoPE choice are exercised.

**Next:** GGUF for OLMoE and Olmo 3.

## Run 5 - 2026-10-29 10:00

**Question:** do llama.cpp's OLMoE and Olmo 3 GGUFs (Olmo 3 converts as `olmo2` with a sliding-window pattern, resolved here to the Olmo 3 row) score as llama.cpp does?

**Command:** `convert_hf_to_gguf.py --outtype q8_0` on both, then `scripts/gguf_perplexity_parity.sh` (README.md, ctx 512, one chunk).

**Finding:**

| file | llama.cpp | ours | drift |
|---|---|---|---|
| OLMoE-1B-7B-0125 Q8_0 | 1.6305 | 1.6367 | 0.0038 |
| Olmo-3-7B Q8_0 | 3.5829 | 3.5813 | 0.0004 |

A 512-token context stays inside Olmo 3's 4096 window and under YaRN's original 8192, so this checks the GGUF load and the full-width q/k norms and post norms; the window itself is checked from safetensors (Run 4).

**Next:** full local CI; `OLMoForCausalLM` (the pre-HF `model.transformer.blocks.*` layout) is left out, its `-hf` conversions load.
