# Loaders investigation

After the multimodal input processors left core (multimodal preprocessing log, Run 7), core's lib test still ends the
cold build (build-time Run 47). The loaders (`inference-core/src/pipeline/loaders/`) are the largest part left.
What in them costs core's compile, and what can leave?

## Run 1 - 2026-09-29

- Question: where does core's IR go now, and how much of it is the loaders?
- Command: `CARGO_TARGET_DIR=<scratch> cargo llvm-lines -p inference-core --lib --features cuda`, with the function
  names grouped by their first `inference_core::` module path. Functions that name no core path were grouped by their
  defining crate.
- Result: 3,025,589 lines in total.
  - Functions under a core path are 47%. The largest are `pipeline` 628k (of which `pipeline::loaders` is 150k),
    `engine` 177k, `gguf` 131k, `sequence` 87k and `selection` 58k.
  - The loaders' own 150k lines are 5% of core: multimodal 70k, normal 48k, embedding 10k, diffusion 5k,
    auto_device_map 5k. By method, the largest are `layer_sizes_in_bytes` 17k, `non_mapped_size_in_bytes` 16k and
    `load` 12k.
  - The other 53% is generic code from other crates, instantiated in core: `core` 255k, serde_json about 220k,
    tokenizers 87k, hashbrown 75k, serde 62k.

    About 210k of it is defined in the family crates: llama 69k, qwen 67k, phi 35k, other 28k, gemma 10k. Of that,
    about 148k is serde's `visit_map`/`visit_seq`/`visit_str` for family config types:
    - Phi4MMConfig 5.7k, ConformerEncoderConfig 5.0k, the Qwen 3.5 MoE TextConfig 4.8k, lfm2::Config 4.5k, and
      about 80 more types.
    - The cause: core calls `serde_json::from_str::<FamilyConfig>` (usually `let cfg: X = serde_json::from_str(config)?`)
      in the loaders, the processors' `new`, pipeline/speech.rs and elsewhere. So the deserializer is instantiated in
      core, once per deserializer kind (`StrRead`, `Value`, flattened).
    - The family crates already have `inference_nn::json_config!`, a non-generic `from_json` that compiles the
      deserializer where the type lives, but only the text configs use it, and core bypasses it at many sites.
  - The rest of the family-crate code in core: `create_anymoe_layers` about 16k (a trait default instantiated per
    model), the xlora GGUF `from_gguf::<File>` 10.6k, `Debug::fmt` of configs about 8k (the loaders'
    `get_config_repr`), and `model_config` about 3.4k.
- Implication: there are two levers, in increasing size.
  1. Config deserialization: register every family config that core parses with `json_config!` and call `from_json`
     everywhere. That moves about 148k lines (5% of core) into the family crates, which build in parallel ahead of core.
     It is mechanical and changes no behaviour.
  2. The per-model loader impls (150k own lines, plus their sizing arithmetic): they are model knowledge (config to
     layer sizes, ISQ regexes, model metadata). Moving them needs the loader traits (`NormalModelLoader`,
     `IsqModelLoader`, `DeviceMappedModelLoader`, the sizing helpers) in inference-nn, and the multimodal loaders
     would have to split, because `get_processor` returns core's `Processor`. That is a larger design step, to plan
     after lever 1.
- Next: lever 1.

## Run 2 - 2026-09-29

- Question (raised in review): how concurrent is core's own compile? If the loaders' IR goes to the parallel LLVM
  phase, removing it would only leave core's serial phases more exposed.
- Commands:
  - `RUSTC_BOOTSTRAP=1 cargo rustc -p inference-core --lib --features cuda [--profile test] -- -Z time-passes` in a
    scratch target, with every dependency already built so core runs alone. This is a measurement only; nothing is
    committed that needs `RUSTC_BOOTSTRAP`.
  - The Run 47 cold build again, keeping `cargo-timing.html` for its `CONCURRENCY_DATA` and `CPU_USAGE`.
- Result, core built alone. `codegen_crate` contains the lowering to LLVM IR, and the LLVM passes overlap it on worker
  threads, so the rows do not add up to the total.

  | Phase | lib | lib test | Serial? |
  |---|---|---|---|
  | Total | 48.2 s | 46.8 s | |
  | Front end | ~20 s | ~14 s | yes |
  | Metadata | 8.6 s | none | yes |
  | Monomorphization collection | 7.3 s | 8.6 s | yes |
  | Lowering to LLVM IR | 12.9 s | 15.1 s | yes |
  | LLVM passes | 19.9 s | 21.4 s | no, worker threads |

  The front end (expand, type check, borrow check, coherence) is independent of IR. The monomorphization collection and
  the lowering scale with it.
- Result, cold build (254 s this time; load 4 at the start):
  - Core's lib test takes 112 s, against 47 s alone. The build also runs core's lib (t=121-198 s), inference-api, the
    server and the test binaries alongside it.
  - CPU is at 98-99% of 16 cores for t=112-248 s, apart from dips at t=104-112 s (54%) and t=160-176 s (62-86%), even
    when only 4-5 units are active. The big rustc processes each fill the cores with LLVM workers.
  - So after t=112 s the build is throughput-bound, not serial-bound. Parallel LLVM work is not free, because it
    competes for the same cores as core's serial path. Each line of core IR is compiled twice, once in the lib and once
    in the lib test, and both runs fall in that window.
  - The integration-test binaries all compile in that saturated tail (t=198-254 s): inference 5 binaries 89 s,
    server-core 2 binaries 70 s, ffi 3 binaries 33 s. Each one re-monomorphizes and links the stack on its own.
- Implication, levers for the tail in order:
  1. Config deserialization into the family crates (Run 1, lever 1). It shrinks core's serial monomorphization and
     lowering, and its LLVM work, in both the lib and the lib test. It moves that work to t=104-160 s, which includes
     the idle dip at t=104-112 s.
  2. Merge each crate's integration tests into one binary (inference, server-core, ffi). That cuts about 190
     unit-seconds of tail CPU down to one monomorphization and link per crate. The size of the cut needs measuring.
  3. The loader move (Run 1, lever 2). It has the same kind of effect as lever 1, but the design step is larger.

## Run 3 - 2026-09-29

- Change: lever 1.
  - Every family config type that core parses is registered with `json_config!` in its family crate: the multimodal
    configs of the llama, qwen, phi and other families, Dia, and FLUX's model and autoencoder configs.
  - A regex rewrite turned core's call sites into `T::from_json(..)`: 168 of them, `let x: T = serde_json::from_str(..)`
    and `serde_json::from_str::<T>(..)`. The FLUX pair was done by hand.
  - The review checked every site for the same argument, `mut` and error handling. It also asked for the GGUF config
    tests to change, because they instantiated a `Value` deserializer per family config in core's lib test. They now
    take the type's `from_json`.
- Commands:
  - `cargo llvm-lines -p inference-core --lib --features cuda`, in a scratch target.
  - The Run 2 `-Z time-passes` lib-test build, run with core's incremental directory cleared, once on this change and
    once on master.
- Result:
  - Core IR fell from 3,025,589 to 2,778,718 lines (-246,871, -8.2%). No deserializer of a family type is left in core.
    The cut is larger than the 148k of visitors, because serde_json's support code instantiated for them went too.
  - The family crates grew accordingly: llama 716k to 783k, qwen 755k to 845k, phi 488k, other 539k. They build in
    parallel, ahead of core.
  - Core's lib test, alone:

    | | master | this change |
    |---|---|---|
    | Total | 47.0 s | 43.4 s |
    | LLVM passes | 21.9 s | 18.0 s |
    | finish_ongoing_codegen | 3.7 s | under 2 s |
    | Monomorphization collection | 8.5 s | 8.5 s |
    | Lowering to LLVM IR | 15.9 s | 15.7 s |

    The serial phases are unchanged: the visitors are many small functions, cheap to collect and lower but costly to
    optimize. So the gain is parallel LLVM work, which Run 2 showed is the scarce resource in the saturated tail.
- CI: `scripts/local_ci.sh --lint --tests --cuda --slim --bindings --docs --sweep` green (2191 CPU and 2511 CUDA
  tests). After the test-helper change, `--lint --tests --slim` was rerun and is green.
- Next: lever 2, one integration-test binary per crate for inference, server-core and ffi.

## Run 4 - 2026-09-29

- Change: lever 2. Each crate's integration tests became modules of one binary, `tests/integration/main.rs`:
  - inference: embedding_tiny, llama_tiny, paddleocr_vl, paddleocr_vl_tiny and qwen3_5_text_tiny;
  - server-core: chat_route and flux;
  - ffi: engine_abi, header and layout_abi.

  Details:
  - The nextest filters that named binaries now name module prefixes: `binary(paddleocr_vl)` became
    `package(inference) & test(/^paddleocr_vl::/)`, and `binary(flux)` became `test(/^flux::/)`. The exact names gained
    the prefix. `nextest list` confirms the same selections: the default profile excludes all eight real-checkpoint
    tests, `models` runs all eight, and `cuda` keeps the two PaddleOCR-VL checks and FLUX.
  - The fixtures each include the recording helpers, so server-core and ffi can include one fixture alone. So the
    inference binary allows clippy's `duplicate_mod`.
- Command: in the shared target, delete the binaries' incremental directories, touch their sources, then
  `cargo test --no-run --features cuda --workspace --lib --bins --tests --timings`. Only the test binaries rebuild,
  with nothing else running.
- Result:

  | | 10 binaries | 3 binaries |
  |---|---|---|
  | Unit-seconds | 125 | 40 (-68%) |
  | User CPU | 210 s | 134 s (-36%) |
  | Wall | 17.4 s | 13.8 s |

  The merged binaries take 11-13 s each; the largest single binary before was 17 s. The 76 CPU-seconds saved all fall
  in the cold build's saturated tail (Run 2), where they are worth about 5 s of wall time at 16 cores.
- CI: `scripts/local_ci.sh --lint --tests --cuda --slim --bindings --docs --sweep` green (2191 CPU and 2511 CUDA
  tests). The CPU run now lists the seven real-checkpoint PaddleOCR-VL tests as skipped (18, was 11): nextest used to
  drop their whole binary without listing it.
- Next: measure the cold build on these two changes (Run 47 method), then plan lever 3, the loader move.

## Run 5 - 2026-09-29

- Question: what is the cold-build effect of Runs 3 and 4 together?
- Command: the Run 47 cold build, at 029fa506.
- Result: inconclusive for wall time.
  - Wall was 264.8 s (Run 2's master run: 254 s). The machine was loaded by other work: load 8 over the previous 5
    minutes, and 15 by the end.
  - Every stage was 7-10% slower, including stages neither change touches: the candle-kernels build script 64 to
    70 s, candle-core 38 to 42 s, inference-nn 42 to 47 s.
  - Despite that, core's lib test fell from 111.8 s to 105.4 s. The integration binaries (inference 26 s, server-core
    30 s, ffi 24 s) now finish with the other tail units, where before there were ten of them.
- Implication: the isolated measurements in Runs 3 and 4 stand. The cold wall time needs a quiet machine to resolve a
  change of a few seconds, so re-run it idle before comparing with Run 47.

## Run 6 - 2026-09-29

- Question: what would moving the per-model loaders into the family crates take (lever 3)?
- Method: a dependency survey of the loader traits, their helpers and the 51 per-model files. This was code reading,
  with no build.
- Result:
  - Almost every type the loader traits touch is already in inference-nn or inference-quant; core re-exports them
    (`XLoraConfig`, `RopePairing`, `AutoDeviceMapParams`, `ModelConfigLike`, `MatformerSliceConfig`, ...).
  - The core-only pieces are pure functions and types with no core dependencies, so they can move to a new
    `inference_nn::loaders`:
    - `IsqModelLoader` and `isq_regexes`;
    - `AutoDeviceMapQuantization` and the pack-factor helpers, `LanguageModelEnds` and
      `standard_non_mapped_size_in_bytes`, `bias_if!`;
    - `layer_indexed_device`, `LAYER_INDEX_PATTERN`, `standard_layer_index`, `NonMappedSubModel`;
    - `qk_rope_layout_from_config` and `QK_ROPE_LAYOUT_CONFIG_KEY`;
    - `MultimodalPromptPrefixer`, `Modalities` and `SupportedModality`;
    - `get_clip_vit_num_elems`, which belongs in `vision::clip`.
  - Two things are tied to core:
    - `DeviceMappedModelLoader::get_device_layers`, a default that calls core's device mapping. Nothing overrides
      it, and its one method-call use (`pipeline/gguf.rs`) can call the free function instead.
    - `MultimodalModelLoader::get_processor`, which returns core's `Processor`. It becomes a core trait,
      `MultimodalProcessorFactory`, implemented per loader type, with dispatch generated by
      `multimodal_loader_types!`.
  - Size: 13,551 lines in 51 files. By family: llama 3.2k, qwen 3.7k, gemma 2.7k, phi 1.1k, other 2.8k.
  - The loader tests (2.5k lines) only compile with all five families on, so `--slim` never runs them. Core's lib test
    compiles all of them. If the re-exported names stay the same they compile unchanged during the move; they can
    split into the family crates afterwards.
  - Risks:
    - Removing `get_processor` from the trait, the only behaviour-bearing seam change.
    - Gemma 3n's matformer sizing and Gemma 4.
    - Qwen 3.5's `runtime_config`, which has no loader-level test.
    - Most models' `layer_sizes_in_bytes` have no unit test, so the bodies must move unchanged. The review
      checks that with `git diff -M`, as for the processors.
  - An IR caveat: a trait default for a concrete type is instantiated where its vtable is built. If core keeps
    `Box::new(Loader)`, defaults such as `mapped_max_act_size_elems` stay in core. So each family crate should
    construct its own boxed loaders. The first family PR must check this with the Run 1 grouping.
- Plan:
  1. A seam PR: `inference_nn::loaders`, the core `MultimodalProcessorFactory`, and dispatch.
  2. One PR per family, in the order phi, other, qwen, llama, gemma.
  3. A PR moving the per-model tests into the family crates.

## Run 7 - 2026-09-29

- Change: the seam, PR 1 of the Run 6 plan. No loader moved yet.
  - New in `inference_nn::loaders`, copied from core with only paths and visibility changed:
    - the loader traits: `IsqModelLoader`, `DeviceMappedModelLoader` without `get_device_layers`,
      `NormalModelLoader`, `MultimodalModelLoader` without `get_processor`, and `EmbeddingModelLoader`;
    - `MultimodalPromptPrefixer`, `Modalities` and `SupportedModality`;
    - the sizing helpers: `AutoDeviceMapQuantization` and the pack factors, `LanguageModelEnds`,
      `standard_non_mapped_size_in_bytes`, and `bias_if!`;
    - the placement helpers: `layer_indexed_device` and `standard_layer_index`;
    - the qk-rope marker;
    - `get_clip_vit_num_elems`, which went to `vision::clip`.

    Their pack-factor and layer-placement tests moved with them. Core re-exports everything under the old paths.
  - `get_processor` became the core trait `MultimodalProcessorFactory`. Each of the 24 loaders' method moved
    unchanged into its own impl block, which stays in core when that family's loader leaves.
    - `multimodal_loader_types!` generates the `MultimodalLoaderType::get_processor` dispatch.
    - `AutoMultimodalLoader::loader_type` is factored out of `get_loader`.
    - `MultimodalLoader` resolves its stored type, or the Auto type, before building the processor. The review
      checked that this is the same type `inner` was built from.
  - The GGUF device-map path calls the free `get_device_layers`.
- Command: `scripts/local_ci.sh --lint --tests --cuda --slim --bindings --docs --sweep`.
- Result: green (2191 CPU and 2511 CUDA tests).
  - The review found every moved item identical to the original apart from paths.
  - It also found no widening of core's public API; the newly `pub` items are public only through inference-nn.
  - Its cleanups were applied: a stale error message that named the removed method, a third copy of `bias_if!`, two
    redundant `allow`s, a test module name, and `pub(crate)` on the factory trait. The lint, CPU tests and slim were
    rerun after them.
- Next: the phi family's loaders (PR 2).
