# Codebase size investigation

Motivation: `target/debug` settles at ~18.5 GB, and its floor is set by the size of inference-core. The workspace is
520k lines of first-party Rust plus 78k lines of first-party CUDA/C++/Metal. The goal is to find where that size is
duplication or vestigial features, not engine the fork needs.

## Run 1 - 2026-09-25 19:30

Question: where do the lines live?

Command: `wc -l` over `*.rs` per crate and per inference-core module, excluding `target/` and `third_party/`.

Finding:

| crate | lines |
|---|---|
| inference-core | 325,728 |
| inference-quant | 107,042 |
| inference-server-core | 25,126 |
| inference-cli | 17,051 |
| inference-paged-attn | 9,022 |
| inference (SDK) | 8,772 |
| inference-pyo3 | 6,293 |
| inference-layout | 4,540 |
| examples | 3,724 |
| everything else | ~12,000 |

In inference-core: `vision_models/` 84.4k (24 models, gemma4 alone 10.4k), `pipeline/` 58.0k (`loaders/` 21.9k),
`models/` 24.9k (26 text models), `paged_attention/` 17.2k, `cuda/` 16.4k, `gguf/` 14.3k, `speculative/` 9.6k,
`ops/` 8.6k, `xlora_models/` 8.4k. Core and quant together hold 2,106 unit tests.

The shape suggests parallel per-architecture trees: text models, X-LoRA copies, GGUF bindings, vision models with
their own text stacks, and per-model loaders. Four read-only surveys are quantifying the duplication (Run 2).

## Run 2 - 2026-09-25 20:00

Question: how much of the size is duplication, and which features are vestigial for this fork?

Method: four read-only surveys: model code, pipelines and loaders, inference-quant, and a feature inventory. Each
used `wc -l`, `diff -w` over matching regions, grep for call sites and `git log` per path.

### Duplication (no feature loss)

| area | removable | evidence | risk |
|---|---|---|---|
| per-model helpers: AnyMoE layer builders (17 copies), paged/SDPA dispatch (47), `NormalModel` impls (37), RoPE-per-device loop (45) | ~5k | `create_anymoe_layers` bodies are ~119 lines each; `qwen3.rs:226-270` dispatch repeated | low |
| per-arch loaders: `DeviceMappedModelLoader` size math, ISQ regexes (31 copies), config re-parsed 405 times, stub `load_xlora` (16) | ~6-7k | `normal_loaders/llama.rs` 210 lines could be ~15 declarative | low-med |
| Qwen-VL family (2 / 2.5 / 3 / 3-MoE / 3.5-MoE) processors, mod, text, vision | ~5.5k | qwen2_5_vl vs qwen2vl `text.rs` 585/600 lines common | med |
| shared dense GQA decoder for 12 text models (llama, mistral, qwen2/3, smollm3, glm4, hunyuan, starcoder2, phi2/3, gemma/2) | ~4k | pairwise 60-90% common (mistral vs qwen2 627/807) | high (parity per model) |
| vision models re-implementing text stacks: llava_llm, phi3-vision, phi4, gemma3 text | ~3.2k | `llava_llm/mod.rs` says "temporary solution"; base models already expose `forward_embeds` | med |
| inference-quant: ten `mmq_instance_*.cu` differing by one token, dead VectorFP8, duplicated `apply_isq` requantize arms, trait boilerplate | ~4.9k | `vector_fp8_linear_b` has no caller; "HQQ/AFQ/MXFP4 does not support imatrix" string appears 13 times | low |
| same load options in 4 places (CLI `ModelSelected`, TOML selector, pyo3 `Which`, SDK) | ~1.5-2k | pyo3 `Architecture` already drifted (missing `Qwen3_5`) | low |
| normal vs multimodal pipeline mixins, speculative ext, stateless pipelines | ~1.2-1.6k | mixins 31 of ~190 lines differ | med (CUDA graphs) |
| arch enum plumbing (11 touch points per new text arch) | ~0.8k | two identical dispatch matches `normal.rs:510`, `auto.rs:25` | low |
| duplicated SigLIP/CLIP encoders, deepseek2 vs 3, llava 1.5 vs next, misc | ~2.4k | `idefics3/vision.rs` 402/524 common with `siglip.rs` | low-med |

Total: roughly 35-40k lines (~7% of the Rust), all feature-preserving. No dead models: every model is reachable
from a loader.

### Feature weight (every one is upstream code the fork has only renamed or split)

| feature | lines | tests | note |
|---|---|---|---|
| Metal backend | ~30k Rust + 11k shaders | ~0 | only the macOS CI checks it |
| GGUF (core + quant + kernels) | ~42k | 162 | core format, keep |
| GDN / Qwen3-Next | ~24k | 81 | model-specific kernels |
| paged attention | ~26k Rust + 17k kernels | 281 | core |
| LoRA (legacy + dynamic multi-adapter) | ~15k | 109 | |
| cutile (Blackwell) | 14.4k | CI | sm_100+ only |
| tools / agents / skills | ~11k | 120+ | |
| speculative (MTP, DFlash) | 9.6k | 66 | draft-model mode already removed; stale TOMLs remain |
| X-LoRA (+ quantized, GGML variants) | ~10k | 0 | untested |
| distributed (NCCL, ring) | ~6.4k | 32 | |
| code exec + sandbox | ~5.6k | 31 | |
| UQFF | ~6k | 48 | |
| diffusion (FLUX) | ~3.7k | 0 | |
| speech (Dia) | ~3.5k | 5 | |
| embeddings | ~3.7k | 0 in core | |
| MCP client + server | ~3.6k | 0 | |
| HQQ, bitsandbytes | ~3.4k, ~1.4k | | overlap with GGUF/AFQ ISQ; bnb only for HF bnb checkpoints |
| AnyMoE | ~1.5k | 0 | SDK/pyo3 only, no CLI or server |
| GGML legacy | ~0.6k + variants | 0 | |
| web search | 0.8k | 1 | |

Vestigial files: `speculative.toml` and `toml-selectors/speculative-*.toml` (the draft-model mode is gone, and the
selector rejects them) and `/api/mistral` in server docs. `website/` (upstream landing page) and `releases/`
(upstream notes) are for the owner to decide. `ring_configs/` documents the ring backend, and
`scripts/production_soak.py` (14k lines) is the DFlash serving harness, so both stay.

Vendored code inside first-party crates, not labelled as such: FlashAttention-2 kernels, FlashInfer/vLLM paged
kernels, MLX-derived Metal kernels, exllamav2 GPTQ/Marlin, llama.cpp mmq/mmvq, vLLM cutlass MoE. That code is not
the fork's to shrink, but should be marked as vendored.

Implication: consolidation alone is worth ~35-40k lines. The larger lever is scope. The biggest removable blocks are
product decisions (Metal, cutile, X-LoRA, AnyMoE, diffusion/speech, MCP, code exec, HQQ/bnb, GGML), not engineering
ones.

## Run 3 - 2026-09-25 20:40

Question: what foldering changes would make the layout match the architecture?

Finding (from `ls` of the root and crate `src/` dirs, and the locations of `.cu`/`.metal` files):

- The root holds 17 crate dirs plus 14 data/config dirs. Runtime configs are scattered (`toml-selectors/`,
  `topologies/`, `orderings/`, `matformer_configs/`, `ring_configs/`, `calibration_data/`, configs in `examples/`).
- There are four kernel layouts: `kernels/` (flash-attn, quant), `src/cuda/` (core, paged-attn), `.metal` next to
  `.rs` (quant `src/metal_kernels`), and `src/metal/kernels` (paged-attn).
- Vendored kernels sit unlabelled in first-party dirs: FA2, FlashInfer/vLLM, exllamav2, llama.cpp mmq/mmvq, vLLM
  cutlass MoE and MLX Metal.
- inference-core has 25 loose root files, and parallel per-kind model trees (`models/`, `vision_models/`,
  `xlora_models/`, `speech_models/`, `diffusion_models/`, `embedding_models/`). Adapter code spans five dirs in two
  crates, and attention spans five dirs.

Plan: `crates/`, `configs/`, `docker/`; each crate's `third_party/` with provenance READMEs;
`kernels/{cuda,metal}/`; inference-core `request/`, `sampling/`, `layers/`, `selection/`,
`models/{text,vision,audio,diffusion,embedding,common}`, `adapters/`, `attention/{sdpa,paged,flashinfer,mla,gdn}`.
Do these as pure-rename PRs separate from logic changes, with each core regroup landing just before the
consolidation of that area.

## Run 4 - 2026-09-25 22:00

Change: top-level layout, as pure renames.

- The 17 crates move to `crates/`.
- `toml-selectors/`, `topologies/`, `orderings/`, `matformer_configs/` and `ring_configs/` move to `configs/`.
- The Dockerfiles move to `docker/` (the build context stays the root).
- `res/` moves to `docs/assets/`, and `sample_speech.wav` to `examples/assets/`.
- `scripts/` gets `bench/`, `release/`, `convert/` and `testgen/`. `local_ci.sh` and `sweep_target.py` stay at the top.
- `chat_templates/` and `calibration_data/` stay at the root. They are documented paths that users type and that
  Docker copies.

What had to change besides the moves:

- Root manifest members, default-members and path deps.
- Crate `readme = "../../README.md"`.
- Three `CARGO_MANIFEST_DIR` + `/../docs` paths (openapi, CLI reference, supported models) and the `include_str!`
  of the TOML selectors.
- CI path filters (docs, metal shaders), the Makefile clang-format `find`, `.gitignore`, `.gitattributes` and
  `.typos.toml`.
- `render_pyi.py` (a `Path` join the text rewrite could not see), `build_wheels.py` repo-root math, and the soak
  test's package import.
- Docs and examples that name moved paths.

The crate-path rewrite was token-based: `inference/cuda` is a feature and `distributed-inference/` is a docs slug,
so a blanket replace would have broken both. Manifests only had their `path =` values rewritten.

Follow-up (after the reorganization): the layout move forced a cold CUDA workspace build (~25 min of kernels). It
showed two things:
- `inference-flash-attn/build.rs` capped itself at `.thread_percentage(0.5)` (upstream). Under the jobserver that
  only bites when FA2 is the last crate compiling, which is the tail of every cold CUDA build. The cap is removed.
- The long TUs are FA2 `flash_fwd_*_hdim{192,256,512}` / `splitkv` and the FlashInfer FP8 decode TUs, 4-5 min of
  cicc each. Next: time every TU from a cold build and split the worst, as was done for gdn.cu and
  flashinfer_decode.cu.

## Run 5 - 2026-09-25 23:00

Change: kernel layout and provenance, as renames only.

- Every kernel source moves to `<crate>/kernels/{cuda,metal}/`: flash-attn `kernels/`, quant `kernels/<group>` and
  `src/metal_kernels/*.metal`, paged-attn and core `src/cuda` kernel files and `src/metal/kernels/*.metal`. That is
  276 renames, all 100% similar. The Rust bindings stay in `src/`.
- Build globs, watches, header hashes, FA3 include paths, the Metal `source_dir`s, core's two Metal `include_str!`
  and the Metal CI `xcrun` paths are updated.
- The quant and flash-attn watches narrow to `kernels/cuda`, which keeps Metal edits out of the CUDA build now that
  quant's shaders sit under `kernels/`.

Provenance: a survey of copyright headers, attribution comments, identifiers and first-add commits traced every
kernel directory to an upstream. The sources are FA2 via candle, FA3 via vLLM's fork, vLLM (paged attention, cache,
fp8, MoE, marlin, rotary), FlashInfer, exllamav2, IST-DASLab marlin, llama.cpp (mmq/mmvq/ssm, Metal flash-attn),
MLX Metal, HQQ, bitsandbytes, candle, attention.rs and DeepGEMM. The rest is original.

Deviation from the Run 3 plan: code that is adapted and maintained in-tree stays in `kernels/`, and only pristine
drops live wholly in `third_party/`. This follows core's existing convention of `third_party/<upstream>/` holding
the attribution for adapted in-tree files. Moving adapted code into `third_party/` would signal "do not edit", yet
this fork edits it (mmq templating, the FlashInfer decode split). Each kernel crate gets a `third_party/README.md`
index: upstream, in-tree paths, revision where recorded, license, and status. Guesses are marked unconfirmed:
vLLM MLA cache kernels, paged-attn Metal cache ports, exllamav2 `qdq_3.cuh`, the attention.rs license, and the
FlashInfer revision.

Gaps: most adapted code has no pinned upstream revision, and the MIT/BSD components have no verbatim license text in
the tree, only per-file headers.

## Run 6 - 2026-09-25 23:45

Question: which kernel TUs dominate a cold CUDA build? The Run 5 moves forced one.

Command: `scripts/local_ci.sh --lint --tests --cuda --sweep` (2133 s, green), with a `ps` sampler every 2 s logging
each `cicc`/`ptxas` process and its compile unit. Unit time is the sample span per PID.

Finding:

- The kernel phase was 31.8 min wall for 135 units and 320 unit-minutes of `cicc`. The sampler missed `ptxas`,
  whose arguments are quoted, so the totals are lower bounds. At 16-way parallelism, 320 unit-minutes needs 20 min.
- Sampled concurrency by 2-min bucket: 16 16 13 12 12 13 13 13 11 13 12 11 12 11 10 8, undercounted for the same
  reason.
- The slowest units are 5-6.4 min each: FA2 `flash_fwd_splitkv_hdim256_{fp16,bf16}` (382 s, 378 s),
  `flash_fwd_hdim{160,224,256,128}_*` (300-367 s), gdn `{f16,f32,bf16}_bk128_vmajor` (~320 s), and the FlashInfer FP8
  decode `*_fp8_hd{64,128,256,512}` (286-310 s).

Implication: no single TU dominates, and FA2 alone is 53 units of 4-6 min. Splitting is not meant to shrink the total.
It shortens the tail: wall time is about max(total work / threads, the chain of long units still running at the end).
With 5-6 min units starting late, concurrency tapers (8 in the last bucket) rather than dropping to zero at once. The
complement is longest-first scheduling. cudaforge compiles in source-glob order, so a 6-min unit can start last and
set the finish alone. Ordering jobs by expected cost (previous time or source size) packs the tail, and splitting the
heaviest units tightens it further. What would: fewer instantiations (head dims and dtypes nothing
uses), cheaper templates, or checking why concurrency sits at 10-13 instead of 16 after the first ~4 min. Before
trusting the concurrency numbers, fix the sampler's `ptxas` matching.

## Run 7 - 2026-09-25 23:59

Change: regroup inference-core root modules that are sub-parts of one concept.

- `layers.rs` becomes `layers/mod.rs`, with `layers_masker` and `layers_utils` as `layers::{masker,utils}`.
- `model_selected`, `toml_selector`, `model_loader` and `model_metadata` move to `selection/`.
- `prefix_cacher` moves to `kv_cache/`.

Not moved: `request`/`response`/`sequence` (`sequence` is scheduler state, not part of the request), `sampler` (a
group of one), and the small standalone files. The planned `request/` and `sampling/` groups would be churn without
clarity.

Method: a script rewrote `crate::old`, `$crate::old`, `inference_core::old`, root-level `super::old`, and old names
inside `use crate::{...}` groups (brace-aware). Moved files' own `super::` paths were meant to become `crate::`, but
every hit was a `mod tests { use super::*; }`, which must stay `super`, so those were reverted.

Fallout found by the compiler:
- the new `mod` lines were placed above an inner attribute;
- `include_str!` needed one more `../`;
- clippy flagged dead code because `model_metadata` (docs-check helpers, public before) had gone private. `selection`
  is `pub mod`, and its other submodules stay `pub(crate)`.

Result: green, with 2131 CPU and 2449 CUDA tests (unchanged counts).

## Run 8 - 2026-09-25 22:20

Change: first consolidation. `attention::AttentionDispatch` replaces the hand-written paged/SDPA dispatch in models
whose copy matches it exactly.

Survey: 42 model-side copies of the `match &self.paged_attn { ... dummy ... }` block, in 25 normalized shapes.
Token-diffing each against llama's copy sorted them into two groups.

Cosmetic differences, safe to share:
- a local `flash_params` instead of `ctx.flash_params()`;
- `mask` instead of `attention_mask`;
- `is_none()` instead of `matches!`;
- `_ =>` instead of `None =>`;
- `cache_k` names;
- `.contiguous()` on k/v in the paged branch (phi3, phi3-vision, phi4). Those callers keep passing it, so the SDPA
  append now gets contiguous k/v too. Values are unchanged. On CPU/CUDA the cost is the same, because
  `KvCache::append` already made them contiguous. On Metal these three now take the fused append kernel that every
  other model already uses.

Real differences, left bespoke:
- MLA pad and narrow (deepseek2/3, glm4_moe_lite);
- Gemma 3/3n/4 sliding-window and shared-KV paths;
- gemma2's separate SDPA mask;
- Qwen2-VL/2.5-VL f32 masks;
- PaddleOCR dtype casts;
- qwen3_5's optional cache;
- llava's copied LLMs;
- lfm2, which has no mask check;
- SDPA on a forced-contiguous KV cache (llama4, muse_glimmer, qwen3_vl, qwen3_vl_moe, mllama). Sharing that would
  copy the whole cache every step for the other models.

A script rewrote a site only if both paged calls agreed, the SDPA branch was exactly append-then-run with the same q,
mask, flash and sdpa arguments, and the dummy path checked the mask. 23 sites were rewritten; every other site was
reported and left alone. The one behavior change is that a missing mask on the metadata-less path now returns an
error instead of panicking; mllama and voxtral already did this.

Verification:
- `local_ci.sh --lint --tests --cuda` is green (2131 CPU, 2449 CUDA).
- The unit suite barely exercises these models' forward paths, so there was also an end-to-end check.
  Qwen2.5-Coder-3B Q4_K_M GGUF runs through `models/qwen2.rs`. The CUDA CLI was built from master and from the
  branch, served, and given two prompts at temperature 0 with 96 tokens, with `--paged-attn off` and `on`. All four
  outputs were byte-identical before and after. Paged on vs off differ from each other, as expected from different
  kernels.

Result: 24 files, +275 / -1102 lines.

## Run 9 - 2026-09-26 00:40

Change: `AnyMoeBaseModelMixin::create_anymoe_layers` becomes a shared default, replacing 15 per-model copies (~91-99
lines each).

The default builds experts read-only from `get_mlps()`, then swaps in `MoeMlp` layers through `get_mlps_mut()`.
Each model supplies two hooks:
- `amoe_lora_targets()`: the LoRA-targetable projections as `AnyMoeLoraTarget::{up,down}`, or a custom shape;
- `amoe_fine_tuned_expert()`: its original expert constructor (`Mlp::replicate`, `MLP::new(&Config{..})`, or
  phi3's `Mlp::new`).

Survey: 26 impls. Of those, 17 are dense builders, 7 are multimodal wrappers that forward to their text model, and
2 are stubs that bail. The dense builders differ only in field names, the expert constructor and the LoRA target
table. Converted: gemma, gemma2, glm4, hunyuan_v1_dense, llama, mistral, qwen2, qwen3, smollm3, phi2, starcoder2,
phi3, gemma3 text, muse_glimmer text and phi3 vision. Left bespoke: granite (custom `GraniteMlp` and `get_mlp`), and
llava's copied LLMs, which will be deleted when llava reuses the base models.

Two upstream bugs are fixed:
- **Explicit layer lists got the wrong experts, or none.** `experts` had one row per selected layer, but the loop
  tested `layers.contains(&row_index)`, and the final zip paired rows with layer ids.
  - With `layers = [5, 6]`, no row matched, and layers 5 and 6 became MoE layers holding only the base MLP.
  - With `[1, 2]`, row 1 built layer 1's experts and the zip gave them to layer 2.
  - `layers = []` ("all") was unaffected. muse_glimmer's copy already had the fix (`zip(&layers)`),
  and the shared default uses it.
- **phi3's `down_proj` LoRA shape.** phi3 (text and vision) loaded it as `(hidden, intermediate)`. PEFT builds
  `lora_A = nn.Linear(in_features, r)` (see huggingface/peft `tuners/lora/layer.py`), and `down_proj`'s input is the
  intermediate size. The old shape expected `lora_A` of `(rank, hidden)` and produced an `(intermediate, hidden)`
  delta for a `(hidden, intermediate)` weight, so phi3 LoRA-adapter AnyMoE could not load a real adapter. It now
  uses `down("down_proj")`, like every other model.

Tests: AnyMoE had none. Two unit tests with a fake model are added:
- an explicit subset `[1]` of 3 layers builds experts only for layer 1, one per extra VarBuilder, and only layer 1
  becomes MoE;
- LoRA targets are filtered by name, delivered in target order, and shaped as B*A.

Review follow-ups: repeated layer ids are deduplicated (`[0, 0]` would have nested an MoE layer inside another). phi3's
target table is a module constant with a test that pins its PEFT shapes, and phi3-vision reuses it.

Result: green, with 2133 CPU and 2451 CUDA tests (+2 each). Net about -830 lines.

## Run 10 - 2026-09-26 01:40

Change: `device_map::per_layer_device(mapper, num_layers, fallback, make)` builds one value per distinct device that
layers map to, with unmapped layers using `fallback`. It replaces the plain RoPE-per-device loops.

Survey: 51 `for layer in 0..n { let device = mapper.device_for(layer, false)... }` loops.
- 36 are the plain form: `ropes.insert(device.location(), Arc::new(ctor))`, rebuilding the RoPE for every layer and
  overwriting per device. These are rewritten.
- 7 already build once per device, with `contains_key`/`entry` skips around large model-specific bodies (llama,
  mistral, phi3, phi3_5_moe, smollm3, granite, gemma4 x2). They would save a few lines at more risk, so they are left.
- Also left: gpt_oss (picks a RoPE type in the loop), the two phi2 copies (inline comment, matcher declined), and
  four loops that collect per-layer device lists.

The rewrite skipped any site whose constructor used the layer index; there were none. `mapper.get_unique_devices()`
is not equivalent, because it ignores the `real_device` fallback for unmapped layers.

Verification:
- Green, with 2134 CPU and 2452 CUDA tests.
- Qwen2.5-Coder-3B GGUF (`models/qwen2.rs`, a converted site): greedy outputs are byte-identical to master with
  `--paged-attn off` and `on`.
- Server ready time is unchanged (1.378 s / 1.588 s before and after). Building a RoPE per layer instead of per device
  was not a measurable load cost at this size, so the change is line count and clarity only.

Result: 35 files, about -125 lines.

## Run 11 - 2026-09-26 03:00

Change: one table per model kind generates everything that names an architecture.

- `normal_loader_types!` rows: `Variant { cli, hf, model_type, loader }`.
- `multimodal_loader_types!` rows: `Variant { cli | aliases, hf | aliases, loader }`.

Generated from the rows: the enum with its serde renames, `causal_lm_name`, `model_type_name`, `from_causal_lm_name`,
`FromStr` (the error lists every CLI name), `Display`, and a new `loader()`. `loader()` replaces the two identical
dispatch matches per kind (`normal.rs`/`auto.rs`, `multimodal.rs`/`multimodal_loaders/auto.rs`), and
`model_metadata::config_arch` delegates to `causal_lm_name`.

Method: before emitting the tables, a script parsed all seven existing sources per kind and asserted they agreed.
All 26 text and 24 multimodal architectures did. Aliases (lfm2_vl, museglimmer, the Gemma3/Gemma4 HF classes) became
alias lists. Two new tests check the round trip for every variant (CLI name and HF class), name uniqueness, and the
aliases.

Adding a text architecture used to mean touching 11 places. It is now one row, plus the model, its loader and any
GGUF bindings. CLAUDE.md's "adding a model" steps say so.

Not in scope: the pyo3 duplicate enums (`Architecture` misses `Qwen3_5`; the `.pyi` misses HunYuan and misnames
gpt_oss). Python is moving onto the C ABI and pyo3 will be retired (issue #19), so they are left alone. The GGUF
registry is a separate schema table.

Result: green, with 2137 CPU and 2456 CUDA tests (+2). 12 code files, +300 / -569. The multimodal FromStr error now
lists names in table order (gemma4 and muse_glimmer moved after voxtral); nothing parses that message.

Still hand-written outside the tables, for later: the multimodal family subsets in `pipeline/auto.rs:685-703` and
`pipeline/gguf.rs:1665-1678`, which could become a table column, and the GGUF schema registry.

## Run 12 - 2026-09-26 03:40

Question: how do the model-loading front ends differ, and can they share one loading path?

Method: a read-only survey of `ModelSelected` and `model_loader.rs`, the TOML selector, the SDK builders, the server
builder, the CLI conversion, pyo3 and the FFI.

Inventory, by lines of loader-building code:

| front end | code | lines |
|---|---|---|
| core | `loader_from_model_selected` plus helpers | ~830 |
| TOML | `loader_from_selected` plus helpers | ~590 |
| SDK | 7 `build_*_pipeline` fns plus 5 wrappers | ~1300 |
| server | single-model and multi-model build (the latter twice over) | ~520 |
| CLI | `convert_to_model_selected` (serve, plus a quantize copy) | ~720 |
| pyo3 | `parse_which` (retiring, #19) | 473 |

Duplication: UQFF path splitting (13 copies), `Topology::from_option_path` (36), hand-built `NormalSpecificConfig`
(13) and `GGUFSpecificConfig` (15), ordering-file loading (15), and `load_model_from_hf` plus MTP attach (13).
`ModelSelected::Toml` and `::MultiModel` are never constructed. About 1,700 lines are removable, not counting pyo3.

Bugs found:
- **SDK reload does not rebuild what the first load built.** LoRA, X-LoRA, AnyMoE, GGUF-LoRA and GGUF-X-LoRA store
  `loader_config: None`, so they cannot reload at all. Embedding reload drops imatrix and calibration. Speech reload
  drops its `cfg`. An MTP draft head is ISQ'd on reload but not on first load. An inline `with_topology(Topology)`
  is lost on reload.
- **SDK first load:** `with_device` is ignored for text, multimodal, diffusion and speech (`resolve_device(force_cpu,
  None)`). Built-in MTP is never enabled on the LoRA, X-LoRA and AnyMoE paths. Multimodal drops its MCP and
  code-execution configs and hard-codes `no_kv_cache = false`.
- **core:** `max_model_len` is silently dropped for X-LoRA GGUF and legacy-LoRA GGUF (`GGUFSpecificConfig {
  topology, ..Default::default() }`), and `ModelSelected::Toml` drops `mtp`.
- **server:** `hf_revision` is always `None`, and additional models in multi-model mode skip MTP.
- **TOML:** matformer is not settable; Lora and X-LoRA lack organization, imatrix and calibration; Embedding ignores
  the top-level `tokenizer_json`.

Test coverage is thin. There are none for `model_loader.rs` or the server builder, the TOML tests only parse, and no
test checks that a reload equals the first load.

Design proposal:
- **Intermediate:** keep `ModelSelected` as the "which weights" enum. Drop `Toml` and `MultiModel`, and add an
  optional AnyMoE spec plus the missing fields.
- **Load spec:** promote `ModelLoaderConfig` to the single load spec, `{ model, overrides (inline topology /
  ordering / speech cfg), runtime }`.
- **Builders:** `build_loader(&ModelLoaderConfig)` and `load_pipeline(&ModelLoaderConfig)` in `selection/`. Reload
  becomes `load_pipeline(stored)`, so it matches the first load by construction.

Migration, one PR per step:
1. Core helpers, the `max_model_len` and TOML-MTP fixes, and `LoaderBuilder` tests (~-300).
2. TOML converts into `ModelSelected` (~-390).
3. `load_pipeline`, with reload and the server on it (~-300).
4. The SDK builders produce a `ModelLoaderConfig` and use `load_pipeline`. This fixes the device and reload bugs and
   adds a reload-equals-first-load test per builder (~-700).
5. The FFI engine surface builds on it (#19).

## Run 13 - 2026-09-26 04:30

Change: loading-path step 1, core helpers and fixes in `selection/model_loader.rs`.

- `uqff_paths`, `gguf_files` and `load_ordering` replace 8, 5 and 4 inline copies. A missing ordering file now
  returns an error naming the path instead of panicking.
- `SafetensorsOptions { .. }.normal() / .multimodal(max_edge) / .embedding()` builds the per-kind configs from one
  place. The `Run` and auto-`Lora` arms used to write the same eight fields out three times.
- Fixes:
  - `max_model_len` reached X-LoRA GGUF and legacy-LoRA GGUF validation but was dropped
    (`GGUFSpecificConfig { topology, ..Default::default() }`), in both the core and the TOML paths.
  - The TOML path never enabled MTP. `TomlLoaderArgs` gains `mtp`, and TOML Plain and Multimodal call
    `with_mtp`, as core does.
- First tests for `LoaderBuilder` (5): the multi-file splitting, the ordering-file error, shared config fields,
  a Plain build (`get_id`), and `max_model_len` validation. That covers 0 rejected, unsupported formats rejected, and
  X-LoRA GGUF now accepted and built.

Result: green, with 2142 CPU and 2461 CUDA tests (+5). 3 files, +361 / -222. The net line count rises here, because
the shared struct and the tests outweigh the copies removed. The TOML and SDK steps are where this pays off, since
they reuse these helpers instead of their own copies.
