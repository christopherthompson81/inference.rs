# PaddleOCR-VL investigation

Goal: support PaddleOCR-VL (1.5 and 1.6) in inference.rs, either by bringing in the open upstream mistral.rs PR or by
porting from scratch.

## Run 1 - 2026-09-25 07:05

Question: what is new in PaddleOCR-VL-1.6, and does upstream already have support?

Finding:

- The HF collection `PaddlePaddle/paddleocr-vl-16` holds `PaddleOCR-VL-1.6` (safetensors), `PaddleOCR-VL-1.6-GGUF`,
  `PP-DocLayoutV3` and `PP-DocLayoutV3_safetensors` (the layout model inference-layout already ports), plus a demo
  space and two papers.
- 1.5 and 1.6 were compared file by file: `config.json`, `preprocessor_config.json`, `processor_config.json`,
  `chat_template.jinja`, `generation_config.json`, `tokenizer_config.json`, `added_tokens.json`, and the reference
  `modeling_/image_processing_/processing_/configuration_paddleocr_vl.py`. **All are byte-identical.** Only
  `inference.yml` differs, in `model_name`. 1.6 is a weights-only update.
- Upstream mistral.rs:
  - #2320 was closed.
  - #2356 is its rework ("Support PaddleOCR-VL vision model"): 42 commits, +3780/-34, 33 files, opened 2026-07-26,
    no maintainer review as of 2026-09-25. It claims token-for-token greedy parity with transformers 5.13 on CPU
    f32 and GPU bf16, ISQ (Q4K/Q8_0/Q5K/HQQ4, UQFF round-trip), paged attention with the prefix cacher,
    multi-image requests, and an encoder cache.
  - It depends on #2319 (open, one file): llguidance's toktrie marks every added-vocab token special, so
    non-special OTSL table tokens (`<fcel>`, `<nl>`) are dropped on detokenize.

Implication: 1.6 needs no architecture work beyond what 1.5 needs, so evaluate #2356 + #2319 before writing anything.

## Run 2 - 2026-09-25 07:20

Question: does #2356 apply to this fork?

Method: in a scratch worktree off master, `git merge upstream-pr-2356` (the PR's base, upstream ddc999e7, is in our
history), then `git merge upstream-pr-2319`.

Finding:

- Directory-rename detection did not carry the new files across the fork's rename. They landed under
  `mistralrs-core/...` and were moved by hand (`inference-core/src/vision_models/paddleocr_vl/`,
  `inference/examples/models/paddleocr_vl_recognize/`, `inference/tests/paddleocr_vl.rs`), with `mistralrs_quant::`,
  `mistralrs::`, `from mistralrs import` and `mistralrs serve` rewritten to the fork's names.
- There were 5 content conflicts, all "both sides added" (loader lists, the arch error string, test modules, and the
  docs tables). They were resolved as unions.
- Two modify/delete conflicts (`paged_attention/scheduler.rs`, `pipeline/multimodal.rs`) needed the PR's hunks
  re-applied to the renamed files.
  - multimodal.rs was another union.
  - scheduler.rs: the PR defers media-incompatible sequences, so they are not re-queued onto `self.running` while it
    drains. Our tree already sends them to `_preempt`, which pushes onto `waiting`, so the spin cannot happen here.
    I kept ours and dropped the PR's `deferred` queue.
- #2319 merged with no conflicts.
- `cargo check`, then `cargo clippy --workspace --tests --examples -D warnings`: one real issue. The test file's
  `PAGES_ENV`/`pages()` are only used in its cuda/metal-gated module, so they are dead code on a CPU build (upstream
  CI evidently never ran clippy on it). I moved both into the gated module; clippy is now clean.
- `cargo test -p inference-core --lib paddle`: 11/11 pass. `--lib llg`: 2/2 pass.

Implication: the PR ports mechanically. The rest of the work is verification and cleanup, not a rewrite.

## Run 3 - 2026-09-25 07:40

Question: does the ported model reproduce transformers on 1.6?

Method:

- Reference: `pocr_ref.py` (scratch), transformers 5.17 `PaddleOCRVLForConditionalGeneration`, greedy with
  `max_new_tokens=1024`, on CUDA in bf16 and in f32.
- Port: `examples/pocr_parity` (scratch driver). It uses greedy `topk=1`, reads ids from `logprobs`, and runs CUDA bf16
  and CPU f32.
- Inputs: three public PaddleOCR demo images, each with a task prompt:
  - a short general-OCR photo, `OCR:`
  - a small table crop, `Table Recognition:`
  - a full magazine-style page, `OCR:`
- Aside: one demo file is a JPEG with a `.png` extension. PIL sniffs content and `image::open` trusts the extension,
  so the port got a correctly named copy.

Finding (first differing generated token, or "equal"):

| image | hf bf16 vs rs bf16 | hf f32 vs rs cpu f32 |
|---|---|---|
| OCR photo (168 tok) | equal | equal |
| table crop (71 tok) | equal | equal |
| page (305-306 tok) | 193 | 38 |

Cross-checks on the page:

| pair | first diff |
|---|---|
| hf bf16 vs hf f32 | **38** |
| hf bf16 vs rs cpu f32 | 193 |
| rs cuda bf16 vs rs cpu f32 | 257 |

At 38 the page has an underlined span (`\underline{\text{fiddly}}` vs `\underline{\text{fiddly so we'll move on}}`).
At 193 it is a casing choice on a stylized heading ("Do IFancy Most" vs "DO IFANCY MOST"). transformers itself flips
at 38 between its own dtypes, so both points are near-ties. The port stays within the reference's own dtype noise.

Speed (greedy, one image per request, same GPU): the port generated in 1.45 s / 0.51 s / 3.02 s. transformers took
72.7 s / 14.9 s / 83.9 s in bf16, but it ran while a CUDA build oversubscribed the CPU, so that number only says
the port is not slow. It is not a benchmark.

Implication: bring in #2356 + #2319 rather than start from scratch. Before merging, the fork still needs:

- this doc's parity checks as tests on public images (the PR's goldens depend on a `tests/fixtures/ocr.png` that is
  not in the PR);
- 1.6 as the documented default;
- comment-style cleanup to the fork's conventions, and removal of references to files outside the PR;
- a check of paged attention and ISQ on this fork.

GGUF (`PaddleOCR-VL-1.6-GGUF`, the format llama.cpp-based OCR pipelines consume) is not covered by the PR.

## Run 4 - 2026-09-25 08:55

Question: can the port's tests run from committed data, and do the goldens hold on 1.6?

Method: `inference/tests/fixtures/paddleocr_vl/make_fixtures.py` renders synthetic images in Noto Sans (the text is
made up):

- a one-line OCR crop;
- two pages at the same 640x360 size (the prefix-cache test needs byte-identical prompts);
- a 3x3 bordered table.

The goldens are `pocr_ref.py` (transformers 5.17, greedy) runs on 1.6, in f32 and in bf16.

Finding:

- Every fixture gives **identical ids in f32 and bf16**, so none of them sits on a near-tie, and every transcription is
  exact.
- The table comes back as OTSL (`<fcel>Fruit<fcel>Colour<fcel>Count<nl>...`).
- The text-only golden (`Reply with the single word: ok` -> "soon to the 20th of June.") has the same 12 ids as the
  PR's 1.5 golden.
- `cargo test -p inference --test paddleocr_vl` (CPU f32): 4/4 pass. That covers greedy ids for the OCR crop, exact
  text for the OCR crop, both pages and the table (so #2319's detok fix is covered end to end), the text-only ids, and
  the two-image message.
- A first version of the two-image test expected text from both pages and failed. transformers, given the same
  two-image message, also transcribes only the first page (it is an OCR model trained on one crop), so the test now
  asserts that. The engine-side regression it guards is still the per-image grid row.

Other fixes found while regenerating the docs:

- The fork rename had rewritten `mistralrs` to the Python module name inside `.py` files, so these were broken or
  wrong: `scripts/build_wheels.py` and `docs/scripts/render_pyi.py` pointed at a nonexistent `inference_rs-pyo3`,
  `render_examples.py` walked `inference_rs/examples`, and docstrings said `inference_rs serve`.
- Regenerating `supported-models.md` also dropped a stale Voxtral row that the docs-table conflict resolution had
  carried in from the PR's older copy.

## Run 5 - 2026-09-25 09:15

Question: do paged attention, the prefix cacher and ISQ work on this fork?

Command: `INFERENCE_TEST_PADDLEOCR_VL_MODEL=... cargo test -p inference --features cuda --test paddleocr_vl`, which runs
the CPU tests plus the cuda-gated ones. A new test, `isq_q8_0_keeps_ocr_text`, loads bf16 with ISQ Q8_0 and checks
exact text on the three OCR fixtures.

Finding: **7/7 pass** in 461 s, including the first CUDA build after the cudaforge switch.

- `prefix_cache_does_not_serve_one_image_for_another`: with paged attention and prefix cache 16 in bf16, the two
  same-size pages give their own exact golden text, and page_00 again after page_01 is unchanged.
- `mixed_text_and_image_batch_makes_progress`: passes on this fork's scheduler. That is consistent with dropping the PR's
  scheduler change (Run 2), but `tokio::join!` does not guarantee the two requests share a batch. The real evidence is
  the code: `scheduler.rs` sends a media-incompatible sequence to `_preempt`, which queues it on `waiting`.
- `isq_q8_0_keeps_ocr_text`: exact text under Q8_0.

Implication: every item from Run 3 is done except GGUF, which stays out of scope for this PR.

## Run 6 - 2026-09-25 09:45

Question: what did the PR #7 review find, and are the fixes right?

The review found two bugs inherited from #2356 (the adaptation matched upstream code exactly apart from comments), one
panic, and several smaller issues.

- **Multi-image windows.** `Merger::forward` fills image slots from row 0 of the embeddings, and the processor passed
  every image's pixels and grids but window-scoped hashes. A chunked re-prefill, or a prefix hit ending between two
  images, therefore gave image 2's slots image 1's embeddings and cached image 1's under image 2's hash.
  - Fix: `window_images` counts the image tokens in this pass's `input_ids`, walks the full prompt's placeholder runs
    from the window offset (`seqlen_offsets`), and embeds only those images, at their own pixel offsets. It errors if
    the tokens don't line up with whole images. The processor now passes the full hash list.
  - A unit test covers the whole prompt, a window after image 0, a text-only window, and misalignment.
- **Device mapping.** The ERNIE decoder never called `mapper.map`, so any multi-GPU or offload split failed. It now
  keeps the mapper, maps `h` per layer, moves the rope tables and a custom mask only when the device changes, and
  returns to the model device for the final norm and lm_head. The rope `inv_freq` is built on the compute device
  (under ISQ, `vb` stages on CPU, which copied it every step).
- **Panic.** A user prompt containing the literal image placeholder indexed past `grids`. `expand_placeholders` now
  errors when the placeholder count differs from the image count, with a unit test.
- **Degenerate images.** `smart_resize` now errors on a zero dimension or an aspect ratio above 200, like
  `image_processing_paddleocr_vl.py`. Before, 1x10000 produced a zero-height grid. A test also checks that 80x520
  scales up to (140, 868), which matches the reference formula with min_pixels 112896.
- **Tests.**
  - The block-hash guard asserted a value equal to itself. It now asserts that the span changes the hashes.
  - The mixed-batch test has a 120 s timeout.
  - Fixture paths use `CARGO_MANIFEST_DIR`.
  - The GPU module is renamed `gpu`, since the ISQ test isn't paged.
- **Fixtures.** `make_fixtures.py` now resolves paths from its own location and takes a font override. It reproduces
  the committed PNGs byte for byte with Pillow 12.3. The reviewer's `table.png` difference came from Pillow 10.2 glyph
  rasterization, and the goldens belong to the committed PNGs. `make_goldens.py` (committed) regenerates every golden
  in the tests exactly.
- **Docs and examples.**
  - `render_pyi.py` source links now point at this repo.
  - Its install block no longer points at upstream's PyPI and release wheels; it builds from a checkout.
  - The Python examples use a task prompt and a committed fixture, sent as a data URL, instead of a free-form prompt
    and a third-party template image.
