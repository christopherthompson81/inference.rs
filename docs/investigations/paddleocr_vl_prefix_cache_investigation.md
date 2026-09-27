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
