# Granite families (#568)

Loading the Granite architectures beyond Granite 4.0-H on the existing Granite model (`crates/inference-models-other/src/granite.rs`).

## Run 1 - 2026-10-28 20:00

**Question:** with `granite`, `granitemoe` and `granitemoeshared` loader rows on `GraniteMoeHybridLoader` (dense `mlp.gate/up/down_proj` read through the standard `Mlp`), do real checkpoints match transformers?

**Command:** `granite_ref.py` (transformers 5.19.0, f32 CPU) and `granite_ours.py` (Python SDK over `target/debug/libinference_ffi.so`, `Plain`, f32 CPU), last-token logits of a 15-token prompt ("The capital of France is Paris, and the capital of Japan is"), on `ibm-granite/granite-3.3-2b-instruct` (dense) and `ibm-granite/granite-3.1-1b-a400m-instruct` (MoE, 32 experts top 8).

**Finding:**

| checkpoint | max abs | cosine | logit scale | top 5 |
|---|---|---|---|---|
| granite-3.3-2b (dense) | 4.6e-3 | 1.0 | 23.1 | identical |
| granite-3.1-1b-a400m (MoE) | 9.3e-3 | 0.9999999 | 27.7 | identical |

**Next:** the issue also asks whether a real `granite` GGUF loads (the GGUF registry already maps it, not verified).

## Run 2 - 2026-10-28 20:05

**Question:** does llama.cpp's conversion of the dense 2B load and score?

**Command:** `convert_hf_to_gguf.py dense-2b --outtype q8_0` (llama.cpp 970b51ce7), then `granite_ours_gguf.py` (`GGUF` spec, f32).

**Finding:** load failed: `GGUF BPE pre-tokenizer 'refact' is not supported for standalone conversion`. llama.cpp's converter tags Granite's tokenizer `refact` by its hash (the StarCoder-derived vocabulary has single-digit tokens only, so the digit split is invisible to HF's plain ByteLevel). In `llama-vocab.cpp`, `REFACT` shares the `STARCODER` case (`\p{N}` then the GPT-2 regex) and sets no other flags, so `refact` joins our `starcoder` profile. After that: loads, top 5 identical to transformers, max abs 0.35 against the f32 safetensors (cosine 0.99982), expected Q8_0 spread.

**Next:** `scripts/gguf_perplexity_parity.sh` on the same GGUF to compare against llama.cpp itself rather than against f32.

## Run 3 - 2026-10-28 20:15

**Question:** on the same Q8_0 GGUF, does our score match llama.cpp's own?

**Command:** `scripts/gguf_perplexity_parity.sh ~/Programming/llama.cpp/build_cuda/bin/llama-perplexity <dir with dense-2b-Q8_0.gguf>` (README.md, ctx 512, 1 chunk).

**Finding:** reference 1.4435, ours 1.4425, drift 0.0007 (tolerance 0.02). The 0.35 gap of Run 2 is quantization against f32, not a load difference.

**Next:** the dense / MoE rows and the `refact` profile are done; SWA, Vision and Speech remain on #568.

## Run 4 - 2026-10-28 20:40

**Question:** the GGUF registry wants one adapter per loader type; with `granite` / `granitemoe` resolving to the new rows (`granitemoe` to `granitemoeshared` when `ffn_gate_shexp` is present), does the real GGUF still score the same?

**Command:** `granite_ours_gguf.py` on `dense-2b-Q8_0.gguf` again, compared against Run 2's output.

**Finding:** max abs 0.0, same top 5. Full local CI (lint, CPU, CUDA, bindings, size) passes.

## Run 5 - 2026-10-28 20:53

**Question:** with Granite SWA on the Granite model (per-layer sliding window from `layer_types`, `self_attn.sinks` through the gpt-oss sink path, per-layer RoPE theta, transformers v5 expert names `experts.gate_up_proj` / `experts.down_proj` / `router.weight`), do the swash checkpoints match transformers past the 128-token window?

**Command:** `granite_ref.py` / `granite_ours.py` on the first 300 tokens of README.md, `ibm-granite/granite-swash-2b` (dense, 24 layers, 6 full) and `ibm-granite/granite-swash-3b-a600m` (48 experts top 4, shared MLP).

**Finding:**

| checkpoint | max abs | cosine | top 5 |
|---|---|---|---|
| swash-2b | 3.9e-3 | 0.9999999 | identical |
| swash-3b-a600m | 1.3e-2 | 0.9999995 | identical |

Control: swash-2b with every layer forced to `full_attention` gives max abs 16.9, cosine 0.51, a different argmax, so the 300-token prompt does exercise the window.

The sink in transformers is `softmax(QK)V * sigmoid(lse - sink)`, which equals a softmax over the scores plus one sink logit (`exp(x) / (sum + exp(s))`), the form our sinks path computes.

**Next:** `granite_swa` GGUF (llama.cpp writes `attn_sinks`, fused `ffn_gate_up_exps` and `ffn_up_shexp`, a sliding-window pattern array, optional `rope_pattern`).

## Run 6 - 2026-10-28 21:00

**Question:** do llama.cpp's `granite_swa` conversions (swash 2B and 3B-A600M, Q8_0) load and score as llama.cpp does?

**Command:** `convert_hf_to_gguf.py --outtype q8_0` on both, `granite_ours_gguf.py` against the transformers logits of Run 5, then `scripts/gguf_perplexity_parity.sh` over all Granite GGUFs (CUDA build).

**Finding:** the files carry `granite_swa.attention.sliding_window` 128, a boolean `sliding_window_pattern`, `attn_sinks`, fused `ffn_gate_up_exps` and `ffn_up_shexp` (3B), and `tokenizer.ggml.pre = granite-docling` (llama.cpp's GPT-2 case). With the `granite_swa` registry entry, the pattern synthesized into `layer_types` / `sliding_window`, and the fused shared expert bound directly, both load: top 5 identical to transformers (max abs 0.25 and 0.48, Q8_0 spread). Perplexity at first:

| file | llama.cpp | ours |
|---|---|---|
| dense-2b | 1.4435 | 1.4425 |
| swash-2b | 3.4414 | 3.4615 |
| swash-3b | 3.9014 | error: `shape mismatch in gather, lhs: [512, 48], rhs: [4]` |

and a Granite 3.1 MoE GGUF (converted for the check) failed too: `dtype mismatch in mul, lhs: F32, rhs: BF16`. Two bugs, both CUDA-only and older than this work:

- Granite's quantized expert path multiplied the f32 output of a quantized `gather_forward` by bf16 routing weights. Weighting in the output's dtype and casting after fixes it.
- `Tensor::topk` on CUDA ranked only the last row of its input (`final_logits_row`, from the upstream CUDA graphs change of 2026-06), so any router that falls back to it got one token's experts. The fused router takes power-of-two expert counts (and a short list), so 48 experts took the fallback. The DeepSeek-style grouped router (`moe/router.rs`) and AnyMoE call it on per-token rows too. It now ranks every row; `cuda_topk_ranks_every_row` checks it against the CPU result.

After both:

| file | llama.cpp | ours | drift |
|---|---|---|---|
| dense-2b | 1.4435 | 1.4425 | 0.0007 |
| moe-1b | 1.5056 | 1.5066 | 0.0007 |
| swash-2b | 3.4414 | 3.4615 | 0.0058 |
| swash-3b | 3.9014 | 3.8562 | 0.0116 |

**Tokenizer (upstream, for a llama.cpp report):** llama.cpp tags swash's tokenizer `granite-docling` (the hash of its check text), whose case in `llama-vocab.cpp` is the GPT-2 split. granite-docling-258M's own tokenizer is indeed plain ByteLevel, but swash's (like Granite 4.0 / 4.1 / Switch, same 100352 merges) splits with the Llama 3 regex (`\p{N}{1,3}`, `[^\r\n\p{L}\p{N}]?\p{L}+`, newline runs). llama-server's `/tokenize` on README.md gives 5609 tokens against 4742 from the HF tokenizer; on `x=3.14159` it gives `14`, `159` where HF gives `141`, `59`. Granite 4.1 converts as `dbrx` (the right split): its vocab differs from swash's in some entries, so the hash differs. We follow the GGUF's tag like llama.cpp, so a swash GGUF tokenizes as llama.cpp's does. (`llama-tokenize` segfaults on every model in this build; the server endpoint works.)

**Next:** Granite 4 Vision and Granite Speech.

## Run 7 - 2026-10-28 21:40

**Question:** does Granite 4 Vision (SigLIP tower, Window Q-Former projectors from four tower states plus four spatial ones, deepstack injection before text layers 0/3/6/9/12/15/18/21, LLaVA-Next tiling) match transformers on `ibm-granite/granite-vision-4.1-4b`?

**Command:** `g4v/ref.py` (transformers 5.19.0 `AutoModelForImageTextToText`, f32 CPU) and our server (`inference serve --cpu --dtype f32`, `/v1/chat/completions` with `logprobs`), one synthetic 700x420 image of shapes and a text label, "Describe the shapes and the text in this image.", 24 greedy tokens.

**Findings, in order:**

1. 500: `unknown method: string has no method named index (in chat_template:15)`. The template calls Python's `str.index`, which minijinja's pycompat lacks (it has `find` / `rfind`). Added `index` / `rindex` around the pycompat callback, byte offsets like its `find`, since the template compares the two.
2. 400: two `<image>` markers for one image. The loader's prompt prefixer also prepended one; the template already writes `<image>\n` per image part, so the prefixer leaves the prompt alone (as LFM2-VL's does).
3. 1169 prompt tokens against transformers' 535, first token right. Our `select_best_resolution` capped a candidate's effective resolution at its own area; transformers caps it at the original image's, so upscaling past the image earns nothing and it takes the smallest covering grid (768x768 here, we took 768x1536). Fixed in the shared LLaVA-Next tiling (now `inference_nn::vision::llava_tiling`, which LLaVA-Next uses too), with a test of seven sizes against transformers' choices.

After the fixes:

| | ours | transformers |
|---|---|---|
| prompt tokens | 535 | 535 (494 image) |
| first token top 5 | 791 -0.129, 644 -3.236, 2028 -3.512, 50384 -4.423, 27 -4.653 | 791 -0.129, 644 -3.222, 2028 -3.496, 50384 -4.434, 27 -4.668 |
| 24-token greedy text | identical | |

The 0.016 logprob spread is in line with our CatmullRom resize against PIL's bicubic in the crops.

**Next:** a tiny-checkpoint engine test; `granite-4.0-3b-vision` ships its text weights partly as a PEFT adapter (`adapter_model.safetensors`), not checked here; the GGUF mmproj (`granite4_vision`) and Granite Speech remain.
