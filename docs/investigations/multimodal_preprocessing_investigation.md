# Multimodal preprocessing investigation

Can the per-model multimodal input processors (`inference-core/src/vision_models/`) leave core, so that core's
cold compile and lib test (the end of the cold build since Run 26 of the build-time log) shrink?

## Run 1 - 2026-09-28 (night)

- Question: what do the processors depend on in core, how much of their code is model math versus engine
  orchestration, and what would moving them take?
- Commands:
  - `cargo llvm-lines -p inference-core --lib --features cuda` (build-time Run 40 output), with functions naming
    `inference_core::vision_models` grouped by trait method and by model.
  - `grep` counts of `seq.<method>(` and `crate::<path>` references under `vision_models/`.
- Result:
  - Size: 24.2k lines (2.8k of tests). vision_models IR is 372.9k lines, about 10.7% of core's 3.49M.
  - By method: `InputsProcessor::process_inputs` 24.3%, generic instantiations 23.4%,
    `ImagePreProcessor::preprocess` 6.9%, config deserialization 6.6%, `prepare_for_paged_prompt_planning` 5.8%,
    iterator folds ~10%.
  - The pure image, audio and video math (`preprocess`) is a small part; most of the IR is `process_inputs`
    orchestration over `Sequence`.
  - By model: gemma4 39k, phi4 35k, llava 29k, qwen2vl 29k, preprocessor_config 27k, qwen3_vl 24k, then 14-21k each.
  - Every `process_inputs` has the same skeleton:
    1. validate;
    2. model-specific media work over the sequences (preprocess or reuse cached pixels, rewrite the prompt with
       placeholders, record `mm_features`, `set_toks_and_reallocate`);
    3. core's `get_prompt_input` / `get_completion_input`;
    4. model-specific args, such as a packed layout computed from the text output;
    5. `ModelInputs { .., adapter_leases, recurrent_batch_kind }`.

    Steps 2 and 4 are per model.
  - Coupling to core:
    - About 35 `Sequence` methods, led by `get_toks` 100, `mm_features` 50, `image_hashes` 47, `has_images` 46,
      `set_toks_and_reallocate` 28, `set_mm_features` 28, `set_initial_prompt` 27, `is_chunked_prefill_view` 22.
    - The `seq.multimodal` state (`MultimodalData`: media, cached pixels and grids, `has_changed_prompt`).
    - The text input builders (`pipeline::text_models_inputs_processor`).
    - `vision_models::adapter_leases` (21 uses; core's `AdapterLease`) and `speculative::staging` (5).
    - `ModelInputs`.
    - Everything else they use is already in inference-nn: `PagedAttentionMeta`, `block_hash`, `gdn`, `DeviceMapper`,
      `FlashParams`, `MultiModalFeature`.
- Options:
  1. Invert the skeleton. Core owns steps 1, 3 and 5; each model implements a small media-stage trait for steps 2
     and 4. This is the cleanest, and it dedupes 23 copies of the skeleton, but it rewrites every processor.
     Real-checkpoint coverage for multimodal models on this machine is PaddleOCR-VL only (plus the tiny
     random-weight checkpoint), so a rewrite risks unobserved regressions in the other 22.
  2. Keep each processor's code and abstract what it touches:
     - a `MediaSequence` trait in inference-nn for the `Sequence` methods and `MultimodalData`;
     - a host trait for the text input builders, adapter leases and staging;
     - `ModelInputs` moved beside the models.

     The processors then compile in their family crates with the same bodies, so behaviour is unchanged by
     construction. It needs more plumbing, and it keeps the duplicated skeleton.
- Next: take option 2, starting with a pilot on one processor (Gemma 3) while it is still in core, to prove the
  traits cover it. Then port the rest and move them with their family crates, one family per PR.

## Run 2 - 2026-09-29

- Change: option 2's seam, with every processor still in core.
  - `inference_nn::media_inputs` holds:
    - the media state (`MultimodalData` and the per-kind media, moved from `sequence.rs`);
    - video input;
    - the image-preprocessor trait and the preprocessor/processor configs;
    - `processor.rs`, the interface itself.
  - The interface has three traits:
    - `MediaSequence`: the 42 `Sequence` methods the processors use.
    - `InputsHost`: the prompt and completion text builders, the text-only fallback as `TextOnlyInputs`, staging,
      and block-diffusion progress.
    - `MultimodalInputsProcessor`.

    Beside them sit a lease-free `ModelInputs` and `NoncausalMmContext`.
  - Core's `media_host.rs`:
    - implements `MediaSequence` for `Sequence`;
    - implements the host by recovering `Sequence`s via `as_any`;
    - adds `MediaInputsProcessor`, which runs a processor as an `InputsProcessor` and adds the adapter leases from
      `seq_indices`, as every processor did.
  - The 19 processor files were ported by script, then fixed from compiler diagnostics. Their bodies are unchanged
    apart from types and call forms.
  - The text builders, staging and progress emitters became generic over `Deref<Target = Sequence>`. That lets the
    host pass shared views; the processors build their token lists by borrowing the same sequences.
  - The image-generation fields left `MultimodalData` for a core `ImageGenerationSettings` on `Sequence`, since
    their response-format type is a protocol type inference-nn must not depend on.
- Command: `scripts/local_ci.sh --lint --tests --cuda --slim --bindings --docs`.
- Result: green.
  - The multimodal coverage: the tiny PaddleOCR-VL engine tests, the real-checkpoint PaddleOCR-VL GPU check, and
    every processor's unit tests.
  - The review compared every rewritten builder call with the original, argument by argument. It also checked that
    the recomputed leases equal the old ones (no processor permutes its sequences), and that every write still goes
    through `multimodal_mut()`.
- Next: move the processors into their family crates, one family per PR. Each keeps its `Processor` impl (the chat
  template actions) in core and exports its input processor.

## Run 3 - 2026-09-29

- Change: the first family move. The LFM2-VL and PaddleOCR-VL input processors (and PaddleOCR-VL's `preprocess.rs`)
  moved into inference-models-other as `<model>/inputs_processor.rs`, with no body changes.
  - Core keeps each `Processor` impl in a new `vision_models/<model>/processor.rs`.
  - PaddleOCR-VL's associated token consts became module consts in the family crate, and `Lfm2VlImageProcessor::new`
    replaced the core constructor.
  - inference-models-other gains `anyhow`, `image`, `itertools` and `tokenizers`. None of them adds a crate to the
    build, since inference-nn already depends on each.
  - Nothing left in the moved code needed core. Every `crate::` path resolved through inference-nn's
    `media_inputs`, `paged_attention`, `gdn` and `device_map`, which confirms the seam covers these two.
- Command: `scripts/local_ci.sh --lint --tests --cuda --slim --bindings --docs --sweep`.
- Result: green (2191 CPU and 2511 CUDA tests). The review found no behaviour change. Three nits were applied: unused
  `Clone` derives, a split impl block, and the `pub` on `preprocess`.
- Next: the phi family (phi3, phi4). Its processors additionally use inference-audio, inference-vision, `regex` and
  `rubato`. The core-size effect will be measured once the larger families have moved; this pair is about 1.6k lines.

## Run 4 - 2026-09-29

- Change: the phi family. The Phi-3V and Phi-4MM input processors moved into inference-models-phi as
  `phi3_vision/inputs_processor.rs` and `phi4/inputs_processor.rs`; core keeps their `Processor` impls in
  `vision_models/phi{3,4}/processor.rs`.
  - `Phi3InputsProcessor` gains `Default`, with the image-tag regex as a const. `Phi4MMInputsProcessor::new` takes the
    preprocessor config. `DYHD_BASE_RESOLUTION` is pub for core's Phi-4MM loader.
  - Again every `crate::` path resolved through inference-nn. The only new paths are `inference_audio::AudioInput`
    and `vision::multimodal_layout`.
  - inference-models-phi gains `image`, `inference-audio`, `inference-vision`, `itertools`, `regex`,
    `regex-automata`, `rubato` and `tokenizers`. Unlike Run 3, three of them are not inference-nn dependencies:
    `inference-vision`, `rubato` and `regex-automata`. So the phi crate now also waits on inference-vision, a small
    crate over candle-core that builds alongside inference-nn. No core dependency became unused; gemma3n and voxtral
    still use rubato.
- Command: `scripts/local_ci.sh --lint --tests --cuda --slim --bindings --docs --sweep`.
- Result: green (2191 CPU and 2511 CUDA tests). The review found no behaviour change. It had two nits, both applied: a
  test built the regex by hand in place of `default()`, and a narration comment. The lint and CPU tests were rerun
  after them.
- Next: the qwen family (qwen2vl, qwen3_vl, minicpmo, muse_glimmer). qwen2vl's shared helpers serve the other two
  Qwen processors and muse_glimmer, so all four move together.

## Run 5 - 2026-09-29

- Change: the qwen family. The Qwen2-VL, Qwen3-VL, MiniCPM-o and Muse-Glimmer input processors moved into
  inference-models-qwen, together with qwen2vl's helpers that the other three share. The Qwen3.5 and MoE variants
  already alias the Qwen3-VL processor, so they follow.
  - Core keeps the four `Processor` impls. Each now holds an `Arc` of its family image processor in place of rebuilding
    it per `inputs_processor()` call; the processors are stateless, so that is equivalent.
  - The associated token consts became module consts. Qwen3-VL reuses Qwen2-VL's five, which it had duplicated.
    Muse-Glimmer's settings lost the `Arc` that only served sharing with the core struct.
  - First test snag: `prefix_cache_invalidation_rebuilds_retained_image_pixels` built a real core `Sequence` with
    `Sequence::new_waiting`'s 30 arguments.
    - Outside a chunked-prefill view, every `Sequence` method on that path reduces to a `MultimodalData` call. The cache
      invalidation under test is `MultimodalData::keep_num_images` itself.
    - So a test-module `ImageSequence` over `MultimodalData` now implements `MediaSequence`. It keeps the assertion that
      found the original bug.
    - The fixture is about 130 lines, because the trait has no defaults. Defaults that forward the media accessors to
      `multimodal()` would shrink it to about 12 methods; that is worth doing if a second family's tests need a
      fixture. No other processor test builds a core `Sequence`.
  - New dependencies for inference-models-qwen: `anyhow` (moved up from dev), `image`, `inference-vision` and
    `tokenizers`, plus `inference-audio` as a dev-dependency for the fixture's trait signatures.
- Command: `scripts/local_ci.sh --lint --tests --cuda --slim --bindings --docs --sweep`.
- Result: green (2191 CPU and 2511 CUDA tests). The review found no behaviour change: the special-token order, template
  actions, constant strings and Muse-Glimmer's zero-edge check are all preserved. Its nits were applied (the
  duplicated consts, inline format args, the placement of `new`, a single-use helper). The lint and CPU tests were
  rerun after them.
- Next: the llama family (llava, mllama, llama4, idefics2/3, mistral3, voxtral), then gemma.

## Run 6 - 2026-09-29

- Change: the llama family moved into inference-models-llama:
  - LLaVA and LLaVA-Next, with their shared `utils.rs`;
  - Mllama and Llama 4;
  - Idefics 2 (`idefics2_input_processor.rs` became `idefics2/inputs_processor.rs`) and Idefics 3;
  - Mistral 3;
  - Voxtral, with `audio_processing.rs`.

  Details:
  - The split was scripted with explicit start and end markers. The removed blocks went into core's `processor.rs`
    files verbatim, except that struct literals of the family image processors became `new` or `Default` calls.
  - Unlike the Qwen move, per-call construction was kept wherever the original built a fresh image processor in
    `inputs_processor()`. That matters for Mllama, whose image processor holds `RwLock` state. Idefics 2 keeps its
    `process` override (the prompt expansion) in core.
  - Mllama's `retain_prefix_cached_images` test covers the core `Processor`, so it moved to core's `mllama/processor.rs`.
  - inference-models-llama gains `image`, `inference-audio`, `inference-vision`, `itertools`, `ordered-float`,
    `regex-automata`, `rubato` and `tokenizers`.
  - Core drops `ordered-float`, since Llama 4 was its only user. This is the first core dependency the move removes.
- Command: `scripts/local_ci.sh --lint --tests --cuda --slim --bindings --docs --sweep`.
- Result: green (2191 CPU and 2511 CUDA tests). The review found no behaviour change. It checked the special tokens and
  their order, template actions, `retain_prefix_cached_images`, Idefics 2's `process`, the defaults, and per-call versus
  shared construction. Its style nits were applied:
  - an orphaned `// Input processor` banner;
  - Idefics 2's consts hoisted;
  - split impl blocks in the two LLaVA files.

  The lint and CPU tests were rerun after them.
- Next: the gemma family (gemma3, gemma3n, gemma4 with their audio processors). Then measure core's cold compile and
  IR against build-time Run 40.

## Run 7 - 2026-09-29

- Change: the gemma family, the last one. The Gemma 3, 3n and 4 input processors moved into inference-models-gemma,
  with Gemma 3n's and 4's `audio_processing.rs` (now private modules).
  - Core keeps the three `Processor` impls, still building the image processor per call. `Gemma4ImageProcessor` has 13
    fields, so they became `pub` in place of a 13-argument constructor, as Llama 4's already were.
  - Gemma 4's `defaults_audio_seq_length_to_reference_cap` covers the core `Processor`, so it moved to core.
  - No input processor is left in core, so it sheds what only they used:
    - the `rand_distr`, `inference-vision` and `rubato` dependencies;
    - the `multimodal_layout` alias and the `InputsProcessorValidationError` re-export;
    - the glob re-export of `media_inputs::media`, narrowed to the two items core uses.

    The `--slim` run caught two items that only core's tests used. Those tests cover inference-nn's image hashing and
    placeholder ranges, so they moved to `media.rs`.
  - The banner comments in the moved files went too.
- Command: `scripts/local_ci.sh --lint --tests --cuda --slim --bindings --docs --sweep`, then build-time Run 47's cold
  build and `cargo llvm-lines`.
- Result:
  - CI: green (2191 CPU and 2511 CUDA tests).
  - Core IR: 3,486,102 to 3,025,589 lines, -460k (-13%). That is more than Run 1's 373k lines of `vision_models`
    functions, since core also stops instantiating its own generics (text builders, iterator folds) for them.
  - Cold build: core's lib test 112 s to 104.5 s, core's lib 73.5 s, wall 243 s to 237.6 s.
  - The family crates each take 22-29 s, in parallel with each other and ahead of core's lib test.
- Done: every multimodal input processor now lives beside its model. Core keeps the `Processor` impls (the chat
  template actions) and the `media_host` adapter. Core's lib test still ends the cold build.
