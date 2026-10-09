# Custom attention masks on the flash kernels (CUDA)

Every `AttentionMask::Custom` on CUDA ran `Sdpa::run_attention_noflash`: an unfused cuBLASLt attention that
materializes the scores and runs the softmax in F32. fattn takes an additive f16 mask `(batch | 1, seq_q, seq_kv)`, so
a mask that is the same for every head can run there instead. Background: `noflash_repeat_kv_investigation.md`.

## Run 1 - 2026-10-08 19:11

Question: does fattn with the custom mask match the unfused path, including padded batches, and how much faster is it?

Change: `run_attention` sends a custom mask on CUDA (non-F32, a head dim fattn takes) to `fattn_masked`, which turns
a `(seq_q, seq_kv)`, `(1, seq_q, seq_kv)` or `(batch | 1, 1, seq_q, seq_kv)` mask into fattn's f16 form with
`causal` off (the mask carries causality, windows and chunks). Per-head masks keep the unfused path.

Command: `custom_masks_on_cuda_run_fattn_and_match_the_cpu` (8 query heads over 2 KV heads, head dim 128, 300 tokens,
BF16) against the CPU's fused path in F32: a causal mask with a bidirectional span (as Gemma 3 gives an image), a
two-sequence batch whose second sequence is left-padded (`f32::MIN` fill, as `expand_mask` builds), and a per-head mask.

Raw:
- span and padded batch take fattn; per-head falls back. Kept rows match within 3e-2 (bf16).
- The padded sequence's first 50 queries see no key. fattn wrote non-finite values for them (the F32 CPU reference,
  where `f32::MIN` stays finite, gives a uniform average). A NaN row would spread: the next layer's keys at those
  positions are masked, but fattn's `P @ V` multiplies their NaN values by 0. Fix: rows whose mask is all `-inf` are
  zeroed (they attend to every key), as transformers' `_unmask_unattended` does; the test now asserts every output
  row is finite. `-inf` stays elsewhere, since fattn trims trailing all-`-inf` key tiles with it.
- Side finding: the unfused path without a cuBLASLt controller handed `naive_sdpa` non-contiguous q (the per-head case
  hit "matmul is only supported for contiguous tensors"); it now makes q/k/v contiguous as the rank-2 branch did.

Timing (temporary test, not committed; same shapes as the earlier fallback measurement, causal custom mask, BF16, mean
of 20 after 3 warm-up calls, mask conversion included):

| tokens | unfused | fattn |
|---|---|---|
| 2048 | 4.27 ms | 0.55 ms |
| 4096 | 15.06 ms | 1.80 ms |

Next: a model whose prefill takes a custom mask, timed before and after.

## Run 2 - 2026-10-08 19:25

Question: does a model see it end to end, and which mask shapes did Run 1 not cover?

Command: `inference serve` with SmolVLM-256M-Instruct (safetensors, downloaded to a scratch folder) on master and on
this branch, one large page image with `max_tokens` 1, ten requests after three warm-ups.

Raw: wall time 1394.7 ms (master) and 1389.6 ms (branch); the server's prompt time 0.102 s and 0.103 s. Its vision
tower (head dim 64, 1024 patches per 512-pixel tile) sends `expand_mask` custom masks, which now take fattn, but that
attention is a sliver of the request; the text stack ran causal flash already. Most of the wall time is image
preprocessing. Too small a model to show the change.

From the review: a key-padding mask `(batch, 1, 1, seq_kv)`, which the eager path broadcasts over queries, was left
on the unfused path; it is now expanded to one row per query. The test gained a prefix-cached suffix (100 queries over
300 keys), a key-padding mask in F16, and a mask broadcast over heads by a zero stride (a copy to the GPU materializes
the broadcast, so the test applies it after the copy, as a model does). All six cases pass; the per-head one falls
back. Callers that now take fattn, per the review: Gemma 3 text with images (head dim 256), Llama 4 chunk masks,
the paged prefix-gather custom path, the Phi-4 conformer without relative bias, Mllama's vision tower, CLIP text in
half precision. Still unfused: per-head masks (Mllama cross-attention, the conformer's relative bias), head dim 72
towers (SigLIP so400m in Idefics2/3 and LFM2-VL) and Gemma 4's 512 global layers.

Next: Gemma 3 4B with an image, before and after.

## Run 3 - 2026-10-08 20:29

Question: does Gemma 3 with an image, whose prefill builds a custom mask with a bidirectional span, get faster?

Command: `inference serve --prefix-cache-n 0 --paged-attn {auto,off} --format gguf` on gemma-3-4b-it Q4_K_M with its
F16 mmproj (weights quantization does not touch the attention path), master and this branch; one page image plus
12000 characters of text (6212 prompt tokens), `max_tokens` 1, ten requests after three warm-ups. With the prefix
cache on, every request after the first was served from it (6208 cached tokens), so the first attempt measured
nothing.

Raw (server prompt time, wall time per request):

| | master | branch |
|---|---|---|
| paged attention (default) | 0.722 s, 925.6 ms | 0.722 s, 928.1 ms |
| paged attention off | 2.113 s, 2320.7 ms | 1.250 s, 1443.2 ms |

The unpaged prefill takes the custom mask and runs 1.7x faster; the default paged prefill does not build one for this
request and is unchanged (and faster than either unpaged run).

Side findings, the same on master and this branch, so not from this change:
- Both mmproj files for this checkpoint fail to load: ggml-org's has no `general.type`, unsloth's has
  `general.type = clip-vision`; the loader accepts only `mmproj`, where llama.cpp loads both. The runs above used
  unsloth's file with that one key rewritten (tensors checked identical).
- Gemma 3 GGUF vision reads this page of Chinese text as "colorful, intricate patterns of overlapping shapes", paged
  or not; llama-server on the same files reads it as a text page. The long-prompt request decodes only blank lines.
