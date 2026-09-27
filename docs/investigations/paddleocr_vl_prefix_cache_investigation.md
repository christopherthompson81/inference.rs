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
