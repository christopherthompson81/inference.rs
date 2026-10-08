# Qwen2-VL / Qwen2.5-VL: paged attention changes image-prompt output on CUDA

Found while adding tiny-checkpoint media tests for #219 (`crates/inference/tests/integration/qwen_vl_tiny.rs`). The
existing `qwen2_vl_images_and_video` already pinned different CPU and CUDA ids, attributed in a comment to "CUDA's flash
path rounding near-tied random logits"; the new `qwen2_5_vl_images_and_video` showed the same split.

## Run 1 - 2026-10-08 07:16

Question: are the CPU and CUDA ids near ties? Command: print `trace(model, images(&IMAGE_SIDES[..1]))` (token,
logprob per greedy step) in `qwen2_5_vl_images_and_video`, CPU build and `--features cuda` (paged attention on).

Raw: CPU `[(17, -1.5839), (158, -1.6477), (154, -1.8175), (184, -1.8099), (90, -1.5139), (248, -1.7135)]`; CUDA
`[(178, -1.4748), (247, -1.8464), (165, -1.6607), (246, -1.1896), (90, -0.3833), (118, -0.7147)]`. Not ties: the
distributions differ from the first step (vocab 265, so a flat distribution would sit near -5.6).

## Run 2 - 2026-10-08 07:16

Question: text or vision? Command: the same, with a text-only prompt, Qwen2.5-VL and Qwen2-VL, CPU and CUDA.

Raw: Qwen2.5-VL text-only CPU `(178, -1.27802), (149, -2.17650), ...`, CUDA `(178, -1.27551), (149, -2.17458), ...`:
agree to ~3e-3. So the divergence is in the image path.

## Run 3 - 2026-10-08 07:16

Question: is it `inference_quant::MatMul` casting to F16 on CPU (used by the Qwen2-VL family's vision attention)?
Command: remove the cast (branch `cpu-matmul-dtype`), rerun the CPU trace.

Raw: CPU `(17, -1.58474), (158, -1.64778), ...`: the logprobs move by ~1e-3, the ids do not. Negative result: the cast
is a precision issue on its own (owner chose to fix it), but not this divergence. The comment committed with the
tests (b70f807c) that blames it is wrong and must be corrected.

## Run 4 - 2026-10-08 07:16

Question: does a primitive op differ between backends? Command: a scratch test comparing CPU and CUDA for conv2d
(14x14 and 16x16 kernels, stride = kernel), softmax over an f32::MIN block mask, LayerNorm, matmul against a
transposed operand.

Raw: max abs diff conv14 1.5e-5 (outputs ~74), conv16 2.3e-5 (~117), softmax 6e-8, layernorm 5e-7, matmul 2e-6. No
primitive differs.

## Run 5 - 2026-10-08 07:16

Question: is it paged attention? Command: CUDA build with the test builder's `with_paged_attn` removed.

Raw: CUDA without paged attention `(17, -1.58441), (158, -1.65027), (154, -1.81743), ...` = the CPU trace. So with
images, the paged path disagrees with eager on both backends' reference; text-only prompts agree. Qwen3-VL (interleaved
M-RoPE, no `eager_attention_f32`) agrees paged vs eager. Next: what the paged dispatch does differently for Qwen2-VL
image prompts (mask, M-RoPE positions/tables, prefill chunking).

## Run 6 - 2026-10-08 07:20

Question: what does the paged path feed the model? Command: temporary `eprintln!` of the prompt ids, M-RoPE position
ids and image-pad ranges in `QwenVlModel::forward` (`crates/inference-models-qwen/src/qwen2vl/mod.rs`).

Raw: eager runs the 33-token prompt in one forward, pad range `[(7, 11)]`, positions `[0..6, 7,7,7,7, 9..30]` (t row;
h/w rows carry the 2x2 grid). Paged splits the prompt at the media boundary into three forwards: text `[0..5]`, image
`[260, 262 x4, 261]` with chunk-relative pad range `[(1, 5)]` and positions `[6, 7,7,7,7, 9]` (h/w `7,7,8,8` /
`7,8,7,8`), then text `[10..30]`. Embedding splice and positions are the same per chunk as in the single forward;
`packed_layout` is None (packed prefill is for multi-sequence batches).

## Run 7 - 2026-10-08 07:20

Question: is the media attention policy different (a NonCausal image span would make paged attend bidirectionally)?
Raw: `qwen_vl_inputs.rs:1006,1034` mark Qwen-VL media `Causal`, shared with Qwen3-VL. Not it. No sliding windows either
(`sliding_window: null`).

## Run 8 - 2026-10-08 07:20

Question: does text-only chunked prefill also diverge? Command: `with_max_prefill_chunk_tokens(16)` on a 113-token text
prompt, CUDA paged. Raw: the prompt still prefilled in one forward; the cap does not split text for this model, so the
only multi-forward prefill these models see is at media boundaries. Not yet isolated.

State: eager (CPU, CUDA without paged) agree with each other; paged differs only when the prompt is split at an image;
the splice, positions and policy per chunk look right. Remaining suspects: the attention of a later chunk over the
earlier chunks' paged KV (prefill with past: mask/causal alignment, the noflash fallback for custom masks, #252) for
this text stack (`qkv_bias`, F32 RMS norms, non-interleaved M-RoPE), versus Qwen3-VL whose chunked prefill agrees.
No real Qwen2-VL / Qwen2.5-VL checkpoint is on disk to see which side real output favours.

## Run 9 - 2026-10-08 07:21

Question: does Qwen3-VL's image prompt take the same chunked paged path? Command: temporary `eprintln!` of the
forward length, seqlen offsets and pad ranges in `Qwen3VLModel::forward`, CUDA paged.

Raw: `len=6 offsets=[0]`, `len=6 offsets=[6] pad=[(1, 5)]`, `len=21 offsets=[12]`, then single-token decodes: the same
three-chunk split as Qwen2.5-VL, and its traces match eager. So the generic chunked paged prefill is right; what
differs is specific to the Qwen2-VL family (`QwenVlModel::forward` in qwen2vl/mod.rs and its text spec: non-interleaved
M-RoPE, `qkv_bias`, F32 RMS norms, `eager_attention_f32`), not visible in the per-chunk inputs logged in Run 6.

Paused here. Next step when resumed: dump per-layer hidden states for the image chunk and the trailing text chunk on
both paths and find the first layer that differs; then check which side matches transformers' Qwen2-VL on the same
tiny weights (needs a working torch/transformers install; the local one is broken).

## Run 10 - 2026-10-08 07:35

Question: does the paged/eager split show on a real checkpoint? Command: a temporary test (not committed) loading
Qwen/Qwen2-VL-2B-Instruct (downloaded for this run and deleted after) in BF16 on CUDA, once with and once without
paged attention, greedy, on `crates/inference/tests/fixtures/paddleocr_vl/ocr.png`.

Raw: "What text is written in this image?" gives "The quick brown fox jumps 42 times." on both. "Write a short poem
inspired by this image." gives the same 24 token ids on both (the logprobs reported after top-k=1 truncation are
not raw logits, so they compare nothing).

Conclusion: no visible effect on a real Qwen2-VL; the tiny random-weight checkpoints amplify a small numeric
difference between the chunked paged prefill and the single eager forward for this text stack, which Qwen3-VL's does
not show. Not pursued further now. The tiny tests pin both sides (`cfg!(feature = "cuda")`), so any change on either
path still shows up. A layer-by-layer hidden-state diff (Run 9's next step) would locate the difference if it matters.
