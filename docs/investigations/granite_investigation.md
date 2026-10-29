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
