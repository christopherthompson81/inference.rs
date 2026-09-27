# PaddleOCR-VL paged prefix caching and its test

Moving the PaddleOCR-VL engine-behavior tests from the real checkpoint to a tiny random-weight checkpoint (built at
test time from `crates/inference/tests/fixtures/paddleocr_vl/tiny`) raised whether the tests can fail at all.

## Run 1 - 2026-09-27 02:40

- Question: does `prefix_cache_does_not_serve_one_image_for_another` catch a prefix cache that serves one image's KV
  blocks for another?
- Mutations, each run against the tiny test (CPU and CUDA) and the real-weight test (CUDA):
  - every image's hash forced to 0 in `register_image_span` (paged block hashing),
  - `seq.mm_features()` / `seq.image_hashes()` dropped from the `search_for_matching_cache` call in `add_request`.
- Result: every variant still passes, tiny and real.

## Run 2 - 2026-09-27 02:50

- Question: why? A temporary `eprintln!` in `paged_scheduler.rs` after the prefix lookup.
- Result:
  - Real-weight test: every request schedules 14 tokens, 0 full blocks (block size 32); nothing can be cached.
  - Tiny test with the original "OCR:" prompt: 28 tokens, 0 full blocks.
  - Tiny test with a ~100-character prompt: 163 tokens, 5 full blocks, raw lookup hits 0 tokens even when the same
    image repeats.
  - Control: the same text prompt without an image, repeated: second request hits 288 of 299 tokens.
- Findings:
  - The scheduler sees image prompts with the image as one unexpanded placeholder token, and image prompts never
    get a paged prefix hit. Nothing is served wrongly; repeated or shared-prefix OCR requests redo the vision encoder
    and the prefill.
  - The real-weight test never exercised the prefix cache, so it could not catch the collision it is named for.
- Implication: an image prefix-cache test only means something once image prompts can hit the cache.
- Filed as #48; the image prefix-cache test returns on the tiny checkpoint once image prompts can hit.

## Run 3 - 2026-09-27 03:30

- Question: why do image prompts never hit (#48)?
- Finding: `PaddleOcrVlImageProcessor` expanded the placeholder only in `process_inputs`, i.e. after the scheduler had
  already looked up prefix blocks for the unexpanded prompt. Finished sequences register blocks under the expanded
  prompt's hashes, so a repeat, looked up unexpanded, can never match. Other VL processors expand in
  `prepare_for_paged_prompt_planning`, which `add_request` calls before the prefix lookup.
- Change: the preprocess-and-expand step becomes `expand_image_prompt`, called from `prepare_for_paged_prompt_planning`
  (and still from `process_inputs`, where it is then a no-op).
- Result (paged, tiny checkpoint, probe): the scheduler now sees 326 tokens; page_00 and page_01 miss, the repeat of
  page_00 hits 320 of 326 tokens.

## Run 4 - 2026-09-27 03:45

- The non-paged (CPU) path then failed: `299 image tokens in this pass do not line up with whole images after
  position 9`. A probe in `window_images` showed `input_ids_full` holding only the 317-token suffix after the cached
  prefix: after a non-paged prefix hit `get_toks()` is the suffix (`PrefillTokenView::SuffixOnly`). The processor now
  builds `input_ids_full` from `prompt_position_source_toks()`, which is the whole prompt in that view. Before Run 3's
  change this path was unreachable, since image prompts never matched.
- The restored tiny test `prefix_cache_does_not_serve_one_image_for_another` compares greedy ids exactly and logprobs
  within 1e-3 (a hit reuses KV from a different prefill, so logprobs match only to rounding). Mutation checks:
  - pixel bytes dropped from `SequenceImages` hashing (same-size pages collide): fails on CPU and CUDA.
  - image-span hash forced to 0 in `register_image_span`: fails on CUDA (paged block hashes); passes on CPU, where the
    non-paged cacher compares per-image hashes instead.
  - media keys dropped from `search_for_matching_cache`: passes on CPU because the cached entry then has more images
    than the request and is skipped, so no wrong hit is served.

## Run 5 - 2026-09-27 04:20

- Review: with non-paged hits now reachable, `keep_num_images` drops the cached images and clears
  `cached_img_thw`, but the model takes every image's patches for a vision row and skips cached ones itself
  (`window_images`, the patch offset in `forward`), and mrope needs every image's grid.
- New tiny test `partial_prefix_hit_matches_a_fresh_two_image_decode`: a fresh model decodes [page_00, page_01]; a
  second model first serves [page_00], then [page_00, page_01] with a prefix hit over page_00; the decodes must match.
- First result: CPU `narrow invalid args ... start: 1196, len: 1196` (patch offset past the kept image); CUDA panic in
  `rope_index.rs` (position count != sequence length). A probe in `get_rope_index` on CUDA showed a 310-token first
  prefill chunk holding one image's 299 placeholders but both images' grids: chunked paged prefill stops between the
  images, and the processor attached every grid as soon as any placeholder was present. That predates this fix; the
  real-weight two-image test only ran without paged attention.
- Changes: `PaddleOcrVlProcessor::retain_prefix_cached_images` returns true (as mllama does), so a non-paged hit keeps
  every image; each pass attaches one grid per placeholder run in the prompt so far, counted over
  `prompt_position_source_toks`.
- Result: all three tiny tests pass on CPU and CUDA (twice each). Mutation: retain set back to false fails the new
  test on CPU (`cat expects at least one tensor`).

## Run 6 - 2026-09-27 04:50

- Second review: with grids truncated to the images seen, a batch whose earlier image row stops before its last image
  read later rows' patches at the wrong offset (the model walks rows by their grids, the processor still pushed every
  image's patches). The processor now passes only the seen images' patches and skips rows that have seen none, which
  also covers rows whose chunk had not reached their first image (misaligned before this work too).
- Tests now also assert `usage.prompt_tokens_details.cached_tokens > 0` where a hit must happen. That exposed two
  cases where none can:
  - Non-paged: an exact repeat never hits; `search_for_matching_cache` returns no match when the whole prompt is
    cached, since the forward needs at least one new token. General to every model, not PaddleOCR-specific.
  - Paged: hits are whole blocks and never end inside an image, so a prefix that ends at an image's end token (one
    image vs the same image plus a second) can only be served by the token-granular non-paged cacher.
- The prefix test now shares a long prompt past the image and extends it for the repeat, and compares each cached
  decode with a fresh model instance; the two-image partial-hit test asserts the hit on CPU only.
- Result: all three tiny tests pass on CPU and CUDA; the pixel-bytes hash collision mutation fails on both.
