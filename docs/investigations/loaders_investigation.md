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
