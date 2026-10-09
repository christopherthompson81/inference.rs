# PaddleOCR-VL from GGUF

PaddleOCR-VL loaded only from safetensors. PaddlePaddle publishes an official GGUF of v1.6
([PaddlePaddle/PaddleOCR-VL-1.6-GGUF](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6-GGUF): a BF16 language model,
a BF16/F32 mmproj and `chat_template.jinja`, packaged for llama.cpp's `llama-server`). Issue #388.

## Run 1 - 2026-10-09 07:49

Question: what does the release hold, and does it map onto our model?

Command: `gguf.GGUFReader` over both files, and the safetensors header of the v1.6 checkpoint.

Raw:
- Text: `general.architecture = paddleocr`, 18 layers, hidden 1024, 16 heads over 2 KV heads, head dim 128,
  `rope.dimension_sections = [16, 24, 24, 0]`, Llama tensor names (`blk.N.attn_q`, `ffn_gate`, ...), all BF16;
  tokenizer `llama` (SentencePiece) with scores, 103424 tokens, `add_space_prefix = false`.
- mmproj: `general.type = mmproj`, `clip.projector_type = paddleocr`, the SigLIP tower (27 blocks, hidden 1152,
  `v.position_embd` 729 x 1152) and the connector as `mm.input_norm`, `mm.1`, `mm.2`. llama.cpp's converter
  (`conversion/ernie.py`) drops `packing_position_embedding` and the pooling `head`, which our model never reads, and
  keeps q/k unpermuted (Ernie 4.5 converter), so the RoPE pairing is Hugging Face's half split.

Change: `paddleocr_vl_bindings.rs` binds the text through `bind_llama_text`, the tower through `bind_siglip_vision`
and `mlp_AR.{pre_norm,linear_1,linear_2}` to `mm.{input_norm,1,2}`; the native multimodal GGUF registry maps the
`paddleocr` text/projector pair to `MultimodalLoaderType::PaddleOcrVl` with half-split pairing. Config, processor and
chat template come from the base model's assets (`general.base_model`, PaddleOCR-VL-1.5) or `--tok-model-id`.

## Run 2 - 2026-10-09 07:49

Question: does it read a page?

Command: `inference serve --format gguf -f PaddleOCR-VL-1.6-GGUF.gguf --mmproj ...-mmproj.gguf`, one page image
(a printed form with codes and numbers), `OCR:`, greedy.

Raw: it loads and reads the page, but every digit is missing: codes and dates come out with gaps where their
numbers were. The base model's
tokenizer.json has 101316 entries against the GGUF's 103424, so the loader converts the GGUF vocabulary, and in it
the digits 0-9 (and the OTSL table cells) are `USER_DEFINED` pieces (type 4). The converter registered every
non-normal piece as a special token, and decoding skips specials. transformers and llama.cpp match user-defined
pieces whole but render them as text; only control pieces are special.

Change: user-defined pieces become non-special added tokens (not normalized), control and the rest stay special.
This also stops Gemma 3's HTML-tag pieces from vanishing on decode.

## Run 3 - 2026-10-09 07:49

Question: does the GGUF match the reference?

Command: the same request through our safetensors load (BF16), through llama-server on the same GGUF files, and
through transformers 5.19's own PaddleOCR-VL on the GPU (F32 and BF16), all greedy, 621 prompt tokens each.

Raw:
- our GGUF (BF16) = our safetensors (BF16) = transformers F32 = transformers BF16, byte for byte.
- llama-server differs: it reads one more line and a time our run and transformers both miss (same prompt length).
  transformers is the reference; llama.cpp's own preprocessing and kernels give a different decode here.
- The safetensors goldens of `crates/inference/tests/integration/paddleocr_vl.rs` (from transformers on synthetic
  fixtures) pass through the GGUF: `official_gguf_matches_transformers`, a deep test reading
  `INFERENCE_TEST_PADDLEOCR_VL_GGUF`. The local v1.5 GGUF and mmproj read the OCR fixture correctly too.

The checkpoint's bundled remote code no longer runs on transformers 5.19 (`ROPE_INIT_FUNCTIONS['default']`), and
4.57 rejects it (`create_causal_mask(inputs_embeds=...)`); transformers 5 implements PaddleOCR-VL natively.

## Run 4 - 2026-10-09 07:49

Question: our F32 run on CUDA read the page much worse (lines out of order and missing) than BF16. Why?

Commands and raw, in order:
- CPU F32 = transformers, byte for byte: the defect is CUDA's.
- Text-only prompt, CPU against CUDA F32: same tokens, logprobs within 0.01. Vision tower dumps (temporary, not
  committed), CPU against CUDA: cosine above 0.999998 at every stage. Bilinear position interpolation and the
  unmasked tower attention also agree. The divergence starts at the first generated token, so in the prefill.
- The fused F32 norm kernels (#382) hold to the composite on CUDA in F32 too (added to their test): not them.
- The paged scheduler splits the image prompt into chunks at the image's edges (4 + 610 + 7 tokens), where the CPU
  runs one pass; each chunk's embeddings, positions and RoPE tables equal the CPU's.
- `--paged-attn off` on CUDA F32 = transformers. So: a later prompt chunk over the cached earlier ones.

Cause: in `try_prefix_gather_prefill` a single causal sequence (`simple_full_causal`) drops its mask to `None` for
the flash branch, whose flash params carry causality. A dtype or head dim without the packed flash path (F32 here)
falls through to `Sdpa::run_attention` with no mask and no flash params, which attends non-causally: every query in
the chunk saw the rest of the chunk.

Change: that fallback rebuilds the causal prefix mask. `paddleocr_vl_tiny::image_decodes_are_pinned` had pinned
CUDA's chunked decode separately from the CPU's ("CUDA runs paged flash, which moves the first trace"); with the fix
CUDA gives the CPU's ids, so the test keeps one expectation and fails without the fix. CUDA F32 paged now matches
transformers byte for byte on the page above.

The same fallback explains the Qwen2-VL and Qwen2.5-VL tiny tests' separate CUDA pins (`qwen_vl_tiny.rs`, and
`qwen2vl_paged_media_investigation.md`, which had put it down to tiny-weight numerics): with the fix their CUDA traces
equal the CPU's, and both tests keep one expectation.
