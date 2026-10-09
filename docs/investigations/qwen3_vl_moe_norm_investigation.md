# Qwen3-VL-MoE layer norms against transformers (BF16)

Qwen3-VL-MoE built its layer norms and final text norm as the fused `RmsNorm`, where dense Qwen3-VL uses
`F32RmsNorm`. transformers' `Qwen3VLMoeTextRMSNorm` normalises in F32, casts back, then scales by the weight, which is
`F32RmsNorm`'s order. The tiny random checkpoints show the two norms give different BF16 logits but cannot say which
matches a real one.

## Run 1 - 2026-10-08 21:29

Question: on the real checkpoint in BF16, which norm brings our logits closer to transformers'?

Setup: Qwen3-VL-30B-A3B-Instruct is ~62 GB in BF16, too much for the 24 GB GPU, so both sides load its first N text
layers: `scripts/qwen3_vl_moe_parity.py` links the checkpoint into a temporary directory under a config with
`num_hidden_layers = N` and an index of only the tensors those layers (plus embeddings, final norm, lm_head and the
vision tower) use, which live in 5 of the 13 shards. transformers 4.57.6 (torch 2.11, CPU) and our engine (CUDA,
through the Python bindings) score the same 117 token ids of a fixed English passage; a temporary environment switch
(not committed) picked the MoE norm on our side. Commands: `scripts/qwen3_vl_moe_parity.py <checkpoint> --layers 12
--reference-cache ref12.json --reference-only` once for transformers' logits, then the same with `--layers 12` and
`--layers 6` (and their caches) per norm, with `PYTHONPATH=bindings/python`. Metric: mean over positions of KL(transformers || ours), top-1
agreement, largest logit difference.

Raw:

| layers | norm | mean KL | top-1 | max logit diff |
|---|---|---|---|---|
| 12 | fused `RmsNorm` (before) | 0.00454 | 0.932 | 1.375 |
| 12 | `F32RmsNorm` | 0.00434 | 0.949 | 1.328 |
| 6 | fused `RmsNorm` (before) | 0.00255 | 0.983 | 1.055 |
| 6 | `F32RmsNorm` | 0.00235 | 0.983 | 0.938 |

`F32RmsNorm` is closer at both depths (KL 4% and 8% lower, the largest logit gap smaller), as the transformers code
says it should be. The rest of the gap is BF16 matmul and attention accumulation, which the norm choice does not touch.

Change: Qwen3-VL's text stack uses `F32RmsNorm` for dense and MoE alike; the MoE BF16 snapshot in
`qwen_vl_tests.rs` moves to the new logits. The script stays for re-checks (its transformers pass takes ~15 minutes on
the CPU at 12 layers; `--reference-cache` keeps it). The q/k norms stay fused `RmsNorm`: interleaved M-RoPE needs them
in the RoPE kernel, which takes `Rms`/`Gemma` kinds only.

## Run 2 - 2026-10-08 21:47

Question: what does the switch cost, and can the cost go?

Command: `inference bench -m <12-layer truncated checkpoint> --prompt-len 512 --gen-len 128`, master and this branch.

Raw: the composite `F32RmsNorm` (about nine kernels, no fused residual add) took decode from 192.9 to 171.9 tok/s
(-11%) and prefill from 7753 to 7417 tok/s (-4%), and dense Qwen3-VL, Qwen2-VL and PaddleOCR-VL have paid the same.

Change: the fused CUDA norm kernels (plain, residual add, residual add then norm) take a `round_normed` flag for
transformers' order (normalise in F32, cast to the activation dtype, scale by the weight, cast again; the residual sum
and the next norm's variance read the cast values), and `F32RmsNorm` runs them on CUDA; CPU and Metal keep the
composite. A CUDA test holds all three against the composite to two ulps (three for the second norm), at sizes that
take the scalar and the vec8 kernels.

Raw after: decode 190.9 tok/s, prefill 7734 tok/s, at master's speed. Parity with the fused order: 12 layers mean KL
0.00449, top-1 0.940, max logit diff 1.328; 6 layers 0.00251, 1.000, 0.906. Computing the inverse RMS as a correctly
rounded 1 / sqrt (torch's) in place of `rsqrtf` changed none of these figures; the rest of the move from the composite's
0.00434 is the variance's summation order flipping BF16 casts.

Reading: the three variants (fused scale-then-cast 0.00454, composite 0.00434, fused cast-then-scale 0.00449 at 12
layers) sit within 5% of each other, about the size of the effect Run 1 measured, so end to end the norm order is near
the noise of summation order. The change stands on transformers' definition, which the fused kernel now implements at
no cost, rather than on a margin this check can resolve.

