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
