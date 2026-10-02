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
  - **`--max-model-len` on text GGUF was silently ignored.** The native GGUF text path built its
    `NormalSpecificConfig` with `..Default::default()`, so the value never reached `runtime_config`. It now passes
    through. `max_model_len` is a per-architecture capability: only `qwen3_5_text` implements `runtime_config`, and
    every other text loader refuses it ("not supported by this model loader"). Text GGUF now behaves like safetensors
    text: honored where the architecture supports it, an explicit error otherwise. End-to-end, a Qwen2.5 GGUF served
    with `--max-model-len 256` now fails at startup with that error, where master ignored the flag.
  - **X-LoRA GGUF and legacy-LoRA GGUF can never honor it.** Their legacy adapter pipeline takes its length from the
    model and never reads the config. Validation now rejects `max_model_len` for them in the core and TOML paths,
    instead of accepting and dropping it. The first attempt passed the value into their `GGUFSpecificConfig`, which
    nothing reads on that path; review caught that it was a no-op.
  - **The TOML path never enabled MTP.** `TomlLoaderArgs` gains `mtp`, and TOML Plain and Multimodal call
    `with_mtp`, as core does.
- First tests for `LoaderBuilder` (4): the multi-file splitting, the ordering-file error, a Plain build (`get_id`),
  and `max_model_len` validation. That covers 0 rejected, Embedding rejected, and X-LoRA GGUF rejected; the last one
  fails on master, which accepted it.

Result: green, with 2142 CPU and 2461 CUDA tests (+5). 3 files, +361 / -222. The net line count rises here, because
the shared struct and the tests outweigh the copies removed. The TOML and SDK steps are where this pays off, since
they reuse these helpers instead of their own copies.

## Run 14 - 2026-09-26 05:30

Question: before converting the legacy TOML selector into `ModelSelected` (loading-path step 2), is it reachable?

Finding: no.
- Its only entry point is `ModelSelected::Toml { file }`. The CLI, server, SDK and pyo3 never construct it.
- The CLI parses its own `ModelType` subcommands, not `ModelSelected`.
- Nothing deserializes `ModelSelected` from user input; the server only builds `ModelSelected::Run` in code.
- The docs mention the selector's format only in migration notes pointing to `from-config`, a separate, live TOML
  format with its own parser in `inference-cli`.
- Upstream superseded it without deleting it.

Decision (owner): delete it rather than convert it.

Change:
- `ModelSelected::Toml` and its match arms (`model_loader.rs`, `tuning.rs`, the CLI `tune.rs`) are removed.
- `selection/toml_selector.rs` (1626 lines), its public re-exports (`get_toml_selected_model_dtype`,
  `get_toml_selected_model_device_map_params`) and `configs/toml-selectors/` are removed.
- inference-core no longer depends on the `toml` crate. `from-config` is untouched.

The Run 13 TOML-MTP fix went with it, since the path it fixed no longer exists.

Result: green, with 2129 CPU and 2448 CUDA tests, down 12 each from 2141 / 2460: the selector's own parsing tests.
15 code files, +4 / -1748.

## Run 15 - 2026-09-26 11:00

Change: loading-path step 3. `ModelLoaderConfig::build_loader(no_kv_cache)` and
`ModelLoaderConfig::load(&loader, mtp_runtime)` replace the four copies of the load sequence: server single-model,
server multi-model first model, server multi-model additional models, and `InferenceRs::reload`. The sequence is
build the loader, `load_model_from_hf`, then attach MTP with the draft-head ISQ.

The server now builds the `ModelLoaderConfig` first and loads from it. What reload rebuilds is the same value that
was loaded, instead of a second set of `*_for_config` clones assembled afterwards.

Survey corrections (Run 12 claimed two server bugs):
- **Additional multi-model models skip MTP.** This is by design. The server's MTP setting is global, an external MTP
  checkpoint belongs to one model, and the paged-KV plan reserves MTP memory for the first model only. It stays that
  way, with a comment.
- **`hf_revision` is always `None` in the server.** This is not a dropped value: `serve` has no revision option (the
  CLI's `--revision` exists only on the UQFF inspection commands). It is a missing feature, left as is.

Verification (Qwen2.5-Coder-3B GGUF on CUDA, `--paged-attn off`, greedy, 96 tokens):
- The branch's first load is byte-identical to master's output for the same prompt (from Run 10).
- After `POST /v1/models/unload` then `/v1/models/reload` (200 / 200, status `loaded`), the output is byte-identical
  to the first load.
- The first harness attempt used the `default` alias as the model id. Unload and reload returned 404, and the "same
  output" was a model that never left memory. The id must be the real one, not the alias. The harness now fails fast
  if the server dies.

A new unit test checks that a stored `ModelLoaderConfig` rebuilds its loader, and that its options reach the builder,
so reload validates like the first load.

Result: green, with 2130 CPU and 2449 CUDA tests (+1). 4 files, +182 / -202, including this entry.

## Run 16 - 2026-09-27 (evening)

Change: loading-path step 4 (first part). The seven SDK builders (text, multimodal, gguf, diffusion, speech,
embedding, auto) build their `ModelLoaderConfig` first and load through `build_loader` + `load`, the path the server
and reload already use. `ModelLoaderConfig` gains `overrides: LoadOverrides { topology, speech_cfg }` for values
`ModelSelected` cannot carry; `LoaderBuilder` prefers an inline topology over the path. LoRA, X-LoRA and AnyMoE
(`build_pipeline_from_{text,gguf}_loader`) are left for the next part: they need an ordering override and an AnyMoE
spec in the selection.

Fixes that come with it (Run 12 list): `with_device` is honored for text and multimodal; the MTP draft head is ISQ'd
on first load too; embedding reload keeps imatrix and calibration; speech reload keeps its cfg; an inline topology
survives reload. A review compared every config field the old direct construction set against the LoaderBuilder
arms: no unintended differences. `GgufModelBuilder::with_max_model_len(0)` now asserts like the other builders
(LoaderBuilder would reject it at load).

Tests:
- `inline_topology_is_used_over_the_path` (core): a missing topology path fails the build, an inline topology wins.
- `reload_decodes_like_the_first_load` (tiny PaddleOCR-VL), unload then reload then decode again:
  - First run failed on CPU and CUDA with "model `default` was not found". Suspected the Run 15 alias pitfall and
    switched the id to `list_models()`; it failed the same way, and `list_models()` itself held the one real id.
  - Cause: a real bug. `unload_model` resets `default_engine_id` to the next engine, which is `None` when the model was
    the only one, and `do_reload_model` never set it back. After unload/reload of a single model, every request that
    names no model (the SDK's `send_chat_request`, a server request without `model`) fails with that error.
  - Fix: `do_reload_model` restores the reloaded model as the default when there is none. The test now passes and
    would fail on master, so it pins this fix as well as the round trip.

## Run 17 - 2026-09-27 (night)

Change: loading-path step 4 (second part). The adapter builders (LoRA, X-LoRA, GGUF LoRA, GGUF X-LoRA, AnyMoE) load
through `ModelLoaderConfig` too, so all five can now reload (they stored `loader_config: None` before). The SDK text
and GGUF pipelines split into a selection plus `build_{text,gguf}_pipeline_as(builder, selection, overrides)`; the
loader-based helpers are gone.

- `LoadOverrides` gains `ordering` (inline adapter ordering over the `order` path) and `anymoe` (`build_loader` wraps
  the loader in `AnyMoeLoader`).
- `ModelSelected::XLora` gains `organization`; `XLoraGGUF` / `LoraGGUF` gain tokenizer_json, organization, write_uqff,
  imatrix, calibration_file, hf_cache_path and the matformer fields (serde defaults). Their LoaderBuilder arms used
  `GGUFSpecificConfig { topology, ..Default::default() }` and dropped everything else; the CLI dropped the same
  options for these paths.

Behavior changes a review traced (none regress a working setup):
- `LoraModelBuilder` without an arch now detects through `AutoLoaderBuilder` (as `serve --lora` does): text models
  load the same, multimodal ones load instead of failing "Unknown architecture".
- GGUF LoRA / X-LoRA with `max_model_len` now error; the legacy adapter path never read it (Run 13).
- CLI `--imatrix` / `--calibration-file` with legacy LoRA/X-LoRA GGUF now reach the loader, which refuses them (ISQ
  conversion is native-GGUF only) instead of silently dropping them.
- The AnyMoE text base carries the builder's imatrix, calibration, matformer and built-in MTP settings. A reload with
  `training: true` re-trains the gate.

Tests: `inline_ordering_is_used_over_the_order_path` (an empty order path fails, the override builds) and
`anymoe_override_wraps_the_stored_loader`.

## Run 18 - 2026-09-30 (time approximate)

Question: after the CLI-surface and multi-client work, where do build time, IR and organization stand, and what is the
next housekeeping order?

Commands: `cargo llvm-lines --lib -p <crate>` (CLI: `--bin inference`); `touch <file>` then `cargo build -p inference-cli
--bin inference`, timed; `cargo machete`; two read-only sweeps (duplication; layering, cruft, docs, tests, CI).

Raw findings:
- IR lines: inference-core 1,599,227; inference-cli (bin) 1,233,511; inference-api 1,137,108; inference-server-core
  707,573; inference-agent 358,655; inference-webui 326,731; inference-selection 249,896; inference-protocol 232,082;
  inference-ffi 105,304.
- Incremental rebuild of the `inference` binary: touching `inference-core/src/engine/mod.rs` 57.4 s; touching
  `inference-api/src/responses.rs` 3.8 s; touching `inference-server-core/src/auth.rs` 3.0 s. A core edit is the
  expensive iteration; above core the cycle is already short.
- machete: `either` unused in inference-server-core and inference-cli, `indexmap` in inference-cli, `anyhow` in
  third_party/cudaforge.
- Duplication (largest): X-LoRA model copies (~8k lines in 10 files; the top-level `forward` byte-identical in five);
  Normal vs Multimodal pipeline (~1.2k duplicated lines incl. the CUDA-graph driver and two `load_model_from_path`s of
  559 and 731 lines); `pipeline/macros.rs` (13 loader-macro expansions, `api_get_file!` ~40 times inside them);
  Qwen2-VL vs Qwen2.5-VL text (32 of 597 lines differ, a type rename of two identical rotary types); Qwen-VL input
  processors (~900); DeepSeek2/3/GLM4-MoE-lite MLA (~1.5k); per-loader sizing math (36 copies); local gated `Mlp`
  copies (~15) beside `inference_nn::layers::Mlp`; SigLIP copies (3); API stream collectors (4).
- Layering: the web UI holds `Arc<InferenceRs>` via `Engine::state()`; server-core's `tune_model` builds selections
  itself; ~80 raw `InferenceRs` uses in server-core handlers; the CLI names `inference_core::` 55 times, and the shared
  model conversion lives in `commands/serve.rs`, which `args/` imports.
- Tests: inference-quant has 8 integration binaries (paged-attn 2, sandbox 2) against the one-binary rule; tiny
  checkpoints are shared by cross-crate `#[path]`, and server-core now dev-depends on inference-webui.
- Cruft: `.github/workflows/tests.yml` is a stale manual `cargo test` list; `ci_cuda.yaml` runs plain `cargo test
  --features cuda` (CLAUDE.md: needs nextest); docs pin `"0.8"`, the embed-in-axum example lacks `ModelSelected`'s
  `quant`, cargo-features.md omits `all-models`/`models-*`; unreferenced `scripts/convert` and `testgen`; 45
  TODO/FIXME (several empty); stale blanket allows; style debt (246 banners, 184 non-ASCII comment lines, 480
  `too_many_arguments` allows).

Implication: core is where iteration time goes, so its duplication (pipelines, macros) is the first build-time item;
the model-family merges cut cold builds and IR; the Engine-only boundary is the next architecture item.

## Run 19 - 2026-09-30 (time approximate)

Change: Run 18's cruft list as one PR.
- Dependencies: drop unused `either` (server-core, CLI) and `indexmap` (CLI); the CLI's crossterm 0.28 -> 0.29 and the
  workspace's fancy-regex 0.14 -> 0.17 match what comfy-table and tokenizers pull, so `cargo tree -d` lists neither
  crate twice.
- CI: delete `tests.yml` (a manual `cargo test` over a stale crate list that `local_ci.sh --tests` replaces);
  `ci_cuda.yaml` runs nextest, as plain `cargo test` shares one CUDA context per binary; checkout v5.
- Docs: version pins 0.8 -> 0.9; the embed-in-axum guide rewritten on `Engine::load` + `with_engine` (its
  `ModelSelected::Plain` literal lacked `quant` and no longer compiled); cargo-features documents `all-models` and
  `models-*` and every crate's real defaults; the TOML reference gains the batching keys, `mtp` and the
  `[models.multimodal]` keys; environment variables gain `HF_ENDPOINT`, `INFERENCE_RS_CPU_KV_F32`,
  `INFERENCE_RS_FORCE_AVX2`, `INFERENCE_RS_CUDA_PHASE_TIMINGS` and `INFERENCE_RS_MAX_OUTPUT_BYTES`; README's "request a
  model" link points at this repo; `examples/cli-config.toml` uses `isq`; CLAUDE.md's stale test command points at
  `local_ci.sh --tests`.
- Code: drop the commented-out Dia forward and Llama4 unfold blocks (the Dia mask's Python reference becomes one line),
  `#![allow(dead_code, unused)]` on inference-nn's `utils/normal.rs` (now an allow under `accelerate` only, where the
  device probe compiles away; the import it hid is gated on `cuda`), empty `// TODO`
  markers, and the unused `configure_paged_attn_from_flags` (its `--no-paged-attn` flag is gone); the two bare
  `#[ignore]`s get reasons.

Sweep corrections: `scripts/convert` and `testgen` were reported unreferenced, but they are tooling for supported
features (AWQ to Marlin, GPTQ conversion, X-LoRA ordering, the NVFP4 fixture fetch a quant test names), so they stay;
`[models.format] direct_file_only` is `#[serde(skip)]`, not a TOML key. Left: `inference-sandbox/tests/linux.rs`
creates a per-process temp dir it never removes.

## Run 20 - 2026-09-30 (time approximate)

Question: Run 18's first build-time item, core's pipeline duplication. How much of it is the `pipeline/macros.rs`
macros, which expand fresh code at every call site?

Change: `macros.rs` (1,127 lines) is gone. Its path-lookup macros (`get_paths!`, `get_embedding_paths!`,
`get_paths_gguf!`, `get_uqff_paths!`) become functions in `pipeline/paths.rs` over a `RepoFiles` (one repository at one
revision: its listing and a fetch), with `PathsRequest`/`GgufPathsRequest` carrying the loader's fields; every branch
keeps its old order (tokenizer.json over tekken.json, params.json over config.json for safetensors and the reverse for
GGUF, the `.jinja` template fetching `tokenizer_config.json` beside it). The model-loader macros
(`normal_model_loader!`, `multimodal_normal_model_loader!`, the `_sharded` and embedding variants, `xlora_model_loader!`,
`lora_model_loader!`) become `loading::WeightFiles` (files, dtype, device, layer devices, UQFF reader, shared by every
branch of a load) with `load`/`load_xlora`, `loading::uqff_placeholders`, and per-pipeline `load_from_files`,
`load_with_dynamic_lora` and (normal) `load_xlora` methods; the existing `LoadMetadataParts::metadata` and
`finish_dynamic_lora_runtime` replace the macros' inline copies. `api_get_file!`/`api_dir_list!` become direct
`hf::get_file`/`hf::list_repo_files` calls. One behavior detail: the multimodal distributed path attached the UQFF
reader twice (once after the sharded mapper, again inside the macro); it now attaches it once.

Raw finding: `cargo llvm-lines --lib -p inference-core` 1,599,227 -> 1,571,814 (-27.4k, -1.7%).

Implication: the macro expansions were a small share of core; the larger duplication is the normal vs multimodal
pipeline bodies themselves (mixins, CUDA-graph driver, `load_model_from_path`), the next item.

Review: no regression on a real load path. Kept from it: X-LoRA's missing classifier or config fail before loading
again (the first version skipped a missing classifier and failed inside the model); the generation and processor
configs fetch in the macros' order again (`ProcessorConfigs`), so the first failing download is the same; the loaders'
`token_source`/`revision` locks lost their only reader (`get_uqff_paths!`) and are gone; the attention mode moves into
`LoadMetadataParts` so the new methods stay under six arguments. Trace wording changed slightly (the "(Mistral
tokenizer)" notes), and `hf` errors now keep their anyhow chain instead of a flattened candle message.

## Run 21 - 2026-10-01 (time approximate)

Question: before merging the normal and multimodal pipelines, what would it buy? Run 18 put a core edit at 57 s.

Commands: `cargo llvm-lines --lib -p inference-core` (top functions); `cargo build -p inference-cli --bin inference
--timings` after (a) `touch` of a core file, (b) a new `pub const` in core's `engine/mod.rs`, (c) a `trace!` added to
the body of core's `prompt_chunk_inputs`, (d) a `trace!` added to `RmsNorm::new` in inference-nn's `layers/mod.rs`;
each from a warm build of the same binary, each reverted after.

Raw findings:
- Core's IR is flat: the largest function is 0.5% (`prompt_chunk_inputs` 7.8k lines); the three safetensors
  `load_model_from_path`s are 7.5k (multimodal), 4.8k (normal) and 3.9k (embedding); the mixins are a few hundred lines
  each. Merging normal and multimodal would remove on the order of 10k of 1.57M lines (under 1%).
- Rebuild of the binary: (a) 5.8 s, (b) 6.9 s, (c) 7.1 s (core 2.2 s, then api, server-core, webui, link 1.9 s), (d)
  13.0 s (nn 4.6 s, then the five family crates in parallel, qwen longest at 7.8 s, then core 3.7 s).
- Negative result: Run 18's 57 s was an artifact. It ran right after `cargo llvm-lines`, whose build leaves the
  upstream crates built differently, so the timed build rebuilt inference-nn, protocol and every family crate (a first
  repeat measured 70 s the same way). A warm edit to core costs about 7 s.

Implication: the edit-compile loop is already short; the pipeline merge's value is maintenance (one copy of the
CUDA-graph driver and load path to keep right), not build time, and it carries the CUDA-graph risk. The family-crate
twins (X-LoRA, Qwen2/2.5-VL) are what an inference-nn edit waits on (7.8 s for qwen), and cold builds.

## Run 22 - 2026-10-01 (time approximate)

Change: normal vs multimodal mixins, kept for maintenance (Run 21). Fields stay on each pipeline (the CUDA forward
code reads them); the logic moves:
- `SpeculativePipelineExt` gains required `speculative_target`/`speculative_target_mut` (the model, upcast to
  `dyn SpeculativeTargetMixin`) and defaults for its eight proposer steps, so each pipeline's impl is the two
  accessors, its verify inputs and its CUDA workspace.
- `cache_manager::{clone_in,clone_out,set_none}_cache_by_kind` dispatch on the cache kind over `&dyn Pipeline`; GGUF
  and GGML call `FullCacheManager` through `dyn Pipeline` too. The managers, generic over the pipeline, now
  instantiate once instead of once per pipeline type.
- `isq_flow::requantize_tracked_modules` (runtime re-ISQ), `cuda_graph::clear_decode_graphs` and
  `cuda_graph::reclaim_decode_graphs` replace the copies in both pipelines.

Raw finding: `cargo llvm-lines --lib -p inference-core` 1,569,962 -> 1,559,236 (-10.7k), mostly the cache managers'
per-pipeline copies; source 179 insertions, 238 deletions (embedding's re-ISQ copy folded in too).

## Run 23 - 2026-10-01 (time approximate)

Question: how much of the normal and multimodal `load_model_from_path` (485 and 691 lines) is one load written twice?

Finding (diff of the two bodies): less than Run 18's 430 differing lines suggested. Most of the shared shape (device
mapping, the ISQ plan, weight loading, `finish_isq_load`, cache sizing) is already calls to shared functions with
per-pipeline arguments. The rest differs for real and is interleaved: the multimodal path adds the processor and
preprocessor configs, the encoder cache budget, a matformer-aware auto device map, block diffusion, the loader's default
chat template and bos/eos, layer devices for an MTP head, and Metal scratch release; the normal path adds X-LoRA and its
non-granular state. One function with hooks for these would read worse than two.

Change: the two blocks that were the same code move out. `RecurrentReservation::reserve` (recurrent state pools
reserved before the paged KV cache is sized, then a CUDA context sync if any grew) replaces both copies, and
`cache_layer_count` the cache-kind layer counts. `loading::generation_config` reads the generation config for both; the
multimodal path panicked on a malformed `generation_config.json` (and both on an unreadable one) where the normal path
warned and fell back to the model config; both now warn.

Tests: `a_malformed_generation_config_falls_back_to_the_model_config`.

## Run 24 - 2026-10-01 (time approximate)

Question: can the HTTP server and web UI run on `inference_api::Engine` alone (Run 18's layering item), and what does
`Engine` lack for that without changing what a client sees?

Finding (a map of every handler to its closest `Engine` method): every route had one or nearly one. Error bodies match
already: the server's `ModelError` responder sent `ApiError::model_error()` with no partial response, as `Engine::chat`
does, and validation and internal errors map through the same `ApiError::from_error`. The gaps were: no way to attach
the access-log tap (usage, TTFT/ITL) to an engine stream; no `count_tokens`, container-file or typed file/skill methods;
the `re_isq`/calibration request log lines lived in the handlers; metrics' model-label lookups and MCP's text-model and
permission checks read core directly; the web UI forked, exported and imported sessions and listed models and MCP tools
on `InferenceRs`. Dead ends found on the way: the router's agentic, skills-dir and LoRA setters had no callers (every
router is built `with_engine`), its skill-store fallback was unreachable (every loaded engine has one), and the public
`create_streamer(rx, state, ...)`/`match_responses` helpers were used only by the crate-docs example.

Change: the router's state is the `Engine`; `types::OwnedEngine` scopes it to the request's owner, and handlers call its
methods. `Engine` gains files and container files, `count_tokens`, `fork_session`, `mcp_tools`, `describe_models`,
`default_model_id`, `serves_model`, `agent_permission` and `chats_in_text`, and logs re-ISQ and calibration requests
itself; the stream types gain `with_tap`. `tune_model`'s parsing moves into `inference_api::system`. The responders
collapse to `Sse`/`Json`/`Error(ApiError)`, and `create_streamer` takes the engine's stream, keeping the documented
per-chunk and end hooks. Raw-state uses: server-core 67 -> 0 (the remaining matches are the `RouteKind::InferenceRs`
label), web UI 6 -> 0. Intended differences: an Anthropic prepare failure that is internal now logs, as the other
routes always did; re-ISQ and calibration requests log for C ABI callers too.

Review: no HTTP route changed status, envelope, framing, owner scoping or tap. It caught that the web UI's fork error
would collapse to the generic internal message (fork now maps the store's message to an invalid-request error, so the
UI shows it again), that a public `Engine::new` could hand the router an unprepared adapter root or no skill store (now
private: `Engine::load` is the only constructor, and it prepares both), and leftover re-exports (removed). Kept: the
UI's export and import logs print the `ApiError` message, and a save-chat on an unloaded default model is a 404 where
it was a 500.

Next: the CLI on inference-api re-exports, `quantize` through `Engine::shutdown`, then `Engine::state()` made
crate-private (server-core's integration tests still use it to seed files and drive `OpenResponsesStreamer`).

## Run 25 - 2026-10-01 (time approximate)

Question: what does the `inference` CLI still take from below inference-api, and can it go through inference-api alone?

Finding (every `inference_core::`, `inference_selection::`, `inference_quant::` and `inference_sandbox::` path in the
CLI's source): 71 names in five groups. The spec vocabulary (model selection, dtypes, loader types, ISQ, LoRA and agent
config, runtime defaults) is what `EngineSpec` already carries; tuning and the doctor; UQFF inspection and reports;
logging and the version; the reasoning controls and `File` a chat uses. The CLI called core with raw state in one
place, `quantize` shutting the engine down through `state()`.

Change: inference-api re-exports each group where it belongs (`engine`, `system`, a new `uqff`, the crate root,
`engine_chat`, `files`); the CLI imports from there, drops its inference-core, inference-selection, inference-sandbox and
inference-quant dependencies (quant stays a dev-dependency for the shared tiny-checkpoint support) and forwards its
features to inference-api. `quantize` uses `Engine::shutdown`, which fixes it: shutting down a clone of `state()` while
the engine still held the state always failed ("Cannot shutdown while InferenceRs is shared"), after the UQFF files were
written but before the README and upload hint.

Dead end: `Engine::state()` can't go crate-private yet. Its callers are server-core's engine-level tests (the Responses
streamer storing tool calls, session import keeping a stored file's body, container-file tagging), which seed and read
core state that no client operation exposes. They belong beside the code in inference-api, which needs the tiny
checkpoint support shared outside `#[path]`: the test-support crate item.

## Run 26 - 2026-10-01 (time approximate)

Question: the target is an integration package, where a second project in another language builds an equally capable
server on the bindings. Does our server use anything the C ABI doesn't export?

Finding (every `Engine` method server-core and the web UI call, against `include/inference.h`): owner scoping, chat,
completions, Anthropic messages, Responses, files, sessions, models, LoRA, re-ISQ, calibration, images, speech,
embeddings, approvals, skills, tokenize and the system reports are exported. Missing: Anthropic `count_tokens`,
container files (list, metadata, content), `fork_session`, the MCP tool list, the default model's id (metrics labels
and the UI's first model; nothing in the models list says which is the default), and model auto-tuning
(`/v1/models/tune`). Run 24's `describe_models` duplicated the models list, which already carries category,
modalities and generation defaults. What `state()` still serves (tagging a file into a container, storing a generated
image) happens inside code-execution and image-generation runs, so an integrator never calls it; it's a test shortcut,
not an integration gap. Left to the integrator by design: HTTP framing, keys and browser sign-in, the file-listing
policy, metrics.

Each gap was then held to whether an inference engine needs it; a server concern stays out of the engine. Engine
concerns: token counting (the chat template renders the request), container files (the store code-execution runs
fill), session fork (the agent session store), the MCP tool list (what the engine loaded), the default model (its
routing default) and model-name resolution (whether a `model`, adapter aliases included, routes; metrics label by it
and an integrator would otherwise re-derive the routing rules). Support tooling, needing no engine: auto-tuning, as the
doctor already is. Server concerns that Run 24 had put in `Engine`: `chats_in_text`, the MCP server's own policy, which
the server can read off the default model's card; `describe_models`, a duplicate. Rust-only by nature: the accessors
that echo the caller's own spec (`agent_permission`, `adapter_config`), stream taps (an ABI stream already hands the
caller every event) and the internals (`chat_engine`, `skill_store`, `state`).

Implication: export the engine concerns and tuning, mark the default on its model card, take `chats_in_text` and
`describe_models` out of `Engine`, and add a test that fails when an `Engine` method has neither an ABI entry nor a
listed reason, so the server can't get ahead of the ABI again.

Change (ABI 0.0.15): `inference_anthropic_count_tokens`, `inference_container_files_list`, `_file_get` and
`_file_content`, `inference_session_fork` (the engine names the new session, as the web UI did), `inference_mcp_tools_list`,
`inference_model_served` and `inference_model_tune`, each in C# and Python with tests; model cards carry `"default":
true` on the default model. The web UI builds its model list from the models list and forks and lists MCP tools through
the same calls; the MCP server reads text-in, text-out off the default model's card. `engine_coverage` (in the FFI's
integration binary) parses `impl Engine` and fails on a public method that no FFI entry calls, directly or as its
`_json` form, unless it is listed with a reason; its first run flagged nothing beyond the gaps above once it matched
methods handed on as paths (`Engine::delete_session_json`) as well as calls.

Review and the first CI run: `model_served("default")` was false (core's status lookup doesn't resolve the alias), which
the new tests in all three languages caught; the alias now counts when a default exists. Token counting returned
OpenAI-shaped errors where the HTTP route and the Messages entries use the Anthropic envelope; fixed. A reloading model
can be listed twice, so only its first card is marked default. Clean: every new signature agrees across the header, the
Rust externs, C# and ctypes; the web UI lists the same models in the same order; MCP's text check is equivalent.

## Run 27 - 2026-10-01 (time approximate)

Question: of the ~8.4k lines in the ten X-LoRA model files (Run 18's largest duplicate), how much is one thing written
ten times?

Finding (each method's body hashed per file, the model type name normalized): the attention, MLP and decoder layers are
model-specific, as in their plain twins; what repeats is the frame around them. The top-level `forward` (scaling pass,
then the scaled pass over the full sequence without a KV cache or the new tokens with one, then the head, ~70 lines) is
identical in seven files; Phi-3 differs only in passing its position ids, and the two quantized models only in calling
the head `output`. The `ScalingsMaker::forward` wrapper is identical in nine, the cache selection at the top of
`inner_forward` in all ten (up to the field name). No test ran any of it.

Change: `ScalingsMaker` asks each model for `classifier`, `dtype`, `get_cache`, `inner_forward(XLoraPass)` and
`lm_head`; `xlora_forward` and the scaling pass are free functions over `&dyn ScalingsMaker` in inference-nn, compiled
once rather than per model, and `pass_cache` is the cache selection. Every model's pass now receives the position ids;
only Phi-3 reads them, as before (the others ignored the context lens the old scaling pass handed them). The ten files go
from 7,922 to 6,939 lines. IR of inference-models-llama (four of the ten): 793,066 -> 786,043 lines, so the win is
the source and one tested copy of the control flow, not codegen.

Tests: `without_a_classifier_one_unscaled_pass_runs_on_the_new_tokens`,
`a_classifier_scores_a_scaling_pass_then_scales_the_new_tokens`,
`without_a_kv_cache_both_passes_run_over_the_whole_sequence` (a recording model with a real classifier).

Review: the conversion is faithful in all ten (branch for branch, the destructured names, each model's cache field and
head). It found a bug the duplication had hidden: X-LoRA Gemma2's pass already ended in `lm_head` and the final
softcap, and the frame applied `lm_head` again, so the classifier was fed logits and the head met vocab-sized input;
Gemma2 X-LoRA could not have run on a real checkpoint. The pass now ends at the final norm and `lm_head` applies the
head and the softcap, as the plain Gemma2 does. The tests also record offsets, position ids, no_kv_cache and which
flash params a pass got, and `pass_cache` has its own test.

## Run 28 - 2026-10-01 (time approximate)

Question: Run 18 listed Qwen2-VL vs Qwen2.5-VL text as 32 differing lines of 597. Is it one model?

Finding (`diff` of the two `text.rs` with the type names normalized): yes. The differences are a local renamed
`cos_sin_relocated`, an import moved, and the cache-layout test's values. The two configs' text fields, serde
defaults, `MRopeScaling`, `AttentionType` and the sliding-window resolution are identical too; only `VisionConfig`
differs (Qwen2.5-VL's windowed vision tower). The vision models and the model forward really differ and stay apart.

Change: `qwen2vl::config::QwenVlConfig<V>` holds the shared fields over the family member's vision config, each model
aliasing its own `Config`; Qwen2.5-VL builds `qwen2vl::text::Qwen2VLTextModel`, whose constructors take any
`QwenVlConfig<V>` (the forward isn't generic and compiles once; the constructors compile per vision config). `qwen2_5_vl/text.rs` (597 lines) goes, and the
config loses its copy of the shared half; its test cases join the Qwen2-VL ones
(`sliding_layers_around_a_full_one_keep_their_windows`, the 5-layer window resolution, sliding attention with no
window). `Qwen2_5VLRotaryEmbedding` in inference-nn, a copy of Qwen2-VL's that only this text model used, goes too. A type with a
default parameter (`Config<V = VisionConfig>`) was the first try: `json_config!`'s inherent `from_json` on two
instantiations made `Config::from_json` ambiguous, hence the named generic and one alias per model.

## Run 29 - 2026-10-01 (time approximate)

Question: does a test-support crate pay for itself, and can `Engine::state()` then go crate-private?

Finding: the shared support is 238 lines (the tiny PaddleOCR-VL, Llama and Qwen3-embedding checkpoints and the
recording backend), included by `#[path]` in five places; a crate would save a few seconds at most. `state()`'s callers
are tests that seed or read core state no request reaches on a random-weight model (a file tagged into a container, a
generated image). A first take stopped here, as if those tests set the boundary. The better question is what
`state()` gives a client. It hands out all of `InferenceRs`, about 100 methods: internals (the raw request channel,
loggers, request ids, builders, file-store bookkeeping), what `Engine` already wraps, and four capabilities nothing else
reaches: adding a model to a running engine, removing one, changing the default model and registering an alias. No
binding can call `state()`, so it adds nothing to the integration package; for Rust callers it is a hole beside the ABI
coverage test, since anything done through it never shows up as an `Engine` method.

Implication: export the four (Run 30), then make `state()` crate-private, moving the tests to where what they test
lives: store semantics into inference-api, HTTP framing kept in server-core.

## Run 30 - 2026-10-01 (time approximate)

Change (ABI 0.0.16): `Engine::add_model` (one entry of the spec's `models`, loaded with the runtime settings the engine
started with), `remove_model`, `set_default_model` and `add_model_alias`, as `inference_model_add`, `_remove`,
`_set_default` and `_alias`, in C# and Python, and over HTTP as `POST /v1/models/add`, `/remove`, `/default` and
`/alias`, served only with `--allow-model-management` (`with_model_management` on the router builder), since adding a
model reads any path or hub repo the server can reach. The settings live in `ModelLoadSettings`, which
`build_multi_model` now loads each model through too, so a runtime add and a startup load are one path.

Found on the way: a second copy of a checkpoint under another `model_id` failed at registration, at startup as well
as at runtime, because each copy claimed the pipeline's own name as an alias; a taken name is now left to the model
holding it.

Review: removing or defaulting to an unloaded model came back as a conflict "not found" (the existence check counted
unloaded models; core only looks at running ones), and every core string error read as a conflict; an unloaded model
now gets "reload it first". Core sent the removed engine's terminate while holding the engines lock, so a full request
channel would stall every lookup; it now sends after the locks drop. A named duplicate was refused only after the whole
model loaded; it is checked first now, and an alias or id can't take `default`, a model's or an adapter's name. An
engine whose first model is a non-granular X-LoRA kept 32 sequences in its saved settings though startup ran 1; an
added model of that kind now runs 1 without changing startup. A tensor-parallel engine refuses to add a model, since
its worker processes only load the models they started with.

Tests: `models_are_added_made_default_aliased_and_removed_at_runtime` (FFI), the C# and Python equivalents, and
`model_management_routes_are_served_only_when_enabled`.

## Run 31 - 2026-10-01 (time approximate)

Change: `Engine::state()` is crate-private, and `chat_engine()` exists only under `cfg(test)`. The engine-level tests
that read or seeded core state move beside it, into inference-api (`engine_tests`, sharing the tiny-checkpoint support
by `#[path]` as the other crates do): the Responses streamer storing tool calls, session import keeping a stored file's
body, a container listing and serving only the files its run cited, and a generated image's url resolving to its
bytes. server-core keeps the HTTP side: the container routes list nothing and answer 404 for a container that cited
nothing, and file content is already served in `files_upload_and_serve_their_content`. With `state` and `chat_engine`
gone from `engine_coverage`'s exceptions, everything a Rust client of inference-api can do is an `Engine` method the
ABI exports or one listed there with its reason. `inference_for_server_builder`, whose `build` returns the raw core
state too, goes crate-private (server-core stops re-exporting it; nothing outside used it), which surfaced 14 setters
nothing called, removed. The review then listed the operation layer under `Engine`: 57 public functions across
the modules (files, models, operations, responses, LoRA, dispatch, generation) took the raw state, so a Rust client
holding core's `InferenceRs` could still bypass `Engine` through inference-api. They are crate-private now, and
server-core's re-export of the raw-channel dispatch helpers (unused) goes. It also caught two coverage gaps from the
move, now covered again: a cited container file's metadata lookup succeeding, and a stored PNG served over HTTP with its
media type (`a_stored_image_is_served_as_its_media_type`, seeded by an upload so it runs on CPU).

## Run 32 - 2026-10-01 (time approximate)

Question: after Runs 19-31, what is left of Run 18's list, and what did a fresh pass find?

Commands: `cargo llvm-lines --lib -p <crate>`; `cargo machete`; three read-only sweeps (duplication; layering and API
surface; cruft, docs, CI, tests, style).

Raw findings:
- IR lines (Run 18 in brackets): inference-core 1,557,113 (1,599,227); inference-api 1,248,410 (1,137,108);
  inference-server-core 715,086 (707,573); inference-webui 326,222 (326,731); inference-ffi 114,111 (105,304). The handler logic that moved from the
  server into `Engine` methods (Runs 24, 26, 30) landed in inference-api without the server shrinking: its framing,
  extractors and per-route utoipa paths stay.
- machete: only `anyhow` in third_party/cudaforge, as in Run 18.
- Layering: server-core, webui and CLI reach core only for plain types: `ChatCompletionResponse` and the other
  response types, `CalibrationAction`/`CalibrationStatus`, `SerializedSession`, the core `Response` enum through
  `ResponseTap`, `FILE_PURPOSE_USER_DATA`, `REQUEST_QUEUE_DURATION_METRIC`, `sandbox_key`; the FFI's callbacks need the
  tool and search types `EngineCallbacks` exposes. All could come through inference-api; server-core and webui still
  carry core (and selection) dependencies only for them. The skills routes call `SkillStore` directly for their
  Anthropic shapes rather than an `Engine` method. The `inference` Rust SDK (7,712 lines, 22 files) builds on core
  directly (`Model` wraps `Arc<InferenceRs>`, `pub use inference_core::*`), a second public surface beside
  inference-api with its own builders, request types and agent loop; only `examples/rust` uses it. C# has no typed
  layer; Python's typed `Engine` matches its JSON one.
- Duplication (largest): DeepSeek2/3 and GLM4-MoE(-lite) ~2.5k lines (DS2 vs DS3 differ by the MoE gate; drift
  started: DS3 alone has `add_moe_gate_residual_tensors`); Qwen-VL `mod.rs` wrappers ~900 (qwen3_vl vs its MoE: 2 of
  349 forward lines differ; qwen3_5 vs its MoE: 8 of 318); Qwen-VL input processors ~950; the three
  `load_model_from_path` callers still ~600 (Run 23 shared only the identical blocks); per-loader sizing ~1.1k in 37
  files; SigLIP 3 copies ~630; Qwen3-Next vs Qwen3.5-MoE text ~430; Qwen2-VL vs 2.5 vision ~420; Gemma3n vs Gemma4
  audio ~470; local gated `Mlp` ~10 copies ~500, none with `inference_nn::layers::Mlp`'s merged gate-up fast path;
  LoRA vs QLoRA linear ~200. To verify: qwen2vl clears the MRoPE delta on every step, qwen3_vl only on a prompt.
- CI: GitHub runs check, clippy, rustfmt, typos and doc links, but no tests since `tests.yml` went in Run 19; the
  test suites run only through `local_ci.sh`. `ci_cuda.yaml` carries a PR guard on a dispatch-only workflow;
  `ci.yml` and `metal_shaders.yml` comments are stale.
- Cruft: CLAUDE.md's crate list misses five members and its CLI command list most commands, and it still describes
  the server over `InferenceRs`; docs reference a nonexistent `--multi-model-config`, the retired Python `Runner`, a
  missing `allowed_tools.mjs` and `banner.png`, a wrong examples path, and an architecture page that predates
  inference-api; 9 unreferenced scripts (a 4,651-line soak test never discovered); ~15 stale TODO/FIXMEs of 61;
  ~30 dead public items in core, nn and quant; 7 crates off the one-integration-binary layout (inference-quant 8).
- Style debt is unchanged: 229 banners, 187 non-ASCII comment lines, 458 `too_many_arguments` allows.

Implication: the API boundary work is done for the server, UI, CLI and bindings; the open architecture item is the
Rust SDK as a parallel surface. Code size is now model-family duplication; build time favors the core items.

## Run 33 - 2026-10-01 (time approximate)

Question: should the `inference` Rust SDK be rebuilt on inference-api, merged into it, or removed?

Finding (its public surface against inference-api's): nearly everything it does, `Engine` already does. Its model
builders all end in a `ModelSelected` (`plain_text_selection` and siblings), which is what `EngineSpec`/`ModelSpec`
carry; multi-model, AnyMoe, paged attention, MTP, MCP, code execution, search and tool callbacks all have spec or
`EngineCallbacks` fields. Its `RequestBuilder`/`TextMessages` build what `ChatCompletionRequest` holds (grammars,
response formats, tools, reasoning, files, sessions, adapters); `Model`'s 67 methods map onto `Engine`'s, and its
`BlockingModel` onto `BlockingEngine`. Its agent loop (`agent.rs`, 841 lines) runs tools client-side by re-sending chat
requests, a weaker twin of the engine's own loop that every other client uses. Only the SDK has: raw logits
(`send_raw_chat_request`, two examples), custom logits processors (`add_logits_processor`), typed `Device`/`Topology`
and `IsqBits` (the spec takes strings), typed structured output and `#[tool]`-derived schemas, image helpers over
`image::DynamicImage`, and per-request approval closures (the engine's form is the approval event plus
`resolve_approval`). About 50 of the 59 examples would port with little more than `use` changes; 9 need a missing
feature or redesign (perplexity, logits processor, device map, topology, the three agent examples, structured output,
the approval closure). The docs change (the Rust reference, two guides, 35 Rust tabs, 59 example pages) is about the
same whichever way it goes.

Implication: keep `inference` as the crate name Rust users depend on, but as a facade over inference-api, with the
builders producing `EngineSpec`/`EngineCallbacks` and requests producing `ChatCompletionRequest` plus media, and
`Model` wrapping `Engine`. The client-side agent loop goes. Raw logits and logits processors need a decision: an
explicit low-level escape hatch, engine support, or dropping them.

Decided: `inference` stays the Rust crate name as a thin layer over inference-api with no core dependency, holding the
Rust-only conveniences, so the crate the C ABI links takes no Rust-only dependencies. Raw logits and logits processors
become engine features on every surface, as do per-request tool callbacks and tool-result stream events; the SDK's
client-side agent loop then goes. Order: those engine features, the SDK rebuild on them, then a sweep of the core
public items the old SDK kept alive.

## Run 34 - 2026-10-01 (time approximate)

Change: Run 32's plain types come through inference-api: `response` (the protocol's chat, completion and image
responses and `Usage`), `operations::{CalibrationAction, CalibrationStatus, SerializedSession}`,
`engine_chat::Response` (what a `ResponseTap` sees), `files::FILE_PURPOSE_*`, the callback types `EngineCallbacks`
carries in `engine`, and `REQUEST_QUEUE_DURATION_METRIC` and `sandbox_key` at the root. server-core, the web UI and the
FFI drop their inference-core (and server-core its inference-selection) dependency, server-core keeping core only for
its tests, and their features forward through inference-api alone. Now every client crate builds on inference-api
the way an outside project would.

## Run 35 - 2026-10-01 (time approximate)

Change: GitHub CI gets a `Test` job: the CPU suite as `local_ci.sh --tests` runs it (`cargo nextest run --workspace
--lib --bins --tests`, the doctests, the examples smoke build). It runs on pushes to master, the weekly schedule and
dispatch, not per PR: the dev profile builds at opt-level 3 and a full test build of the workspace is the slowest thing
CI could do (local `target/debug` is 36 GB with the CUDA variants), so PRs keep the fast check, clippy, fmt and typos
jobs and a regression that local CI missed shows on master right after its merge. The job frees the runner's preinstalled
toolchains first to fit the build. The first run, dispatched on the branch, passed in 12 minutes.

## Run 36 - 2026-10-01 (time approximate)

Question: what are the SDK's raw logits, and what should the engine offer in their place?

Finding (the `return_raw_logits` path in core): one prefill pass over the prompt, every position kept, returning a
`[prompt_tokens, vocab]` tensor per chunk and the prompt's tokens; nothing is generated (the SDK's doc said "the first
token generated"). There is always one chunk, since raw-logits requests turn prompt chunking off. They run alone
(batch of one, `n` 1, no prefix cache, CUDA graph or speculation), and the whole prompt must fit one forward. The
perplexity example uses them only for each prompt token's log-probability, and it no longer runs: core now rejects its
`max_tokens: 0`, though a raw-logits request ends after the prefill whatever the limit.

Change (ABI 0.0.17): `Engine::prompt_logits` scores a prompt (text tokenized with the model's special tokens, or token
ids): each token's log-probability, and with `"output": "logits"` the row-major f32 logits. `inference_prompt_logits`
returns the scores as JSON and the logits as an `application/x-f32le` blob; C# returns them as a `float[]`, Python as an
`array('f')`. It refuses non-text models, one-token prompts and prompts longer than the model's context up front.
Planned next, in order: logits processors as named callbacks (a C callback editing the f32 logits in place, selected per
request), the engine's tool loop running every call of a round in parallel with each result tagged by its call id, and
tools registered after load. The design found the SDK's agent examples could already run on `Engine` by executing the
returned tool calls themselves, but the engine's loop runs only the first call of each round, which every client meets.

Review: the core path holds (no token is generated, a raw request always runs alone, no prefix cache). Fixed: a
row/token count mismatch or a token id past the vocabulary is an error rather than a short result or a panic, and a
non-finite score an error rather than a null that reads like the first token's; the log-probabilities come from tensor
ops, so the full logits are copied out only when asked for; an unloaded model gets "reload it first" and `default`
means the default model; the ABI nulls `*out_blob` on entry and refuses a missing one before the forward pass. Known
limits: logits past 2 GiB overflow C#'s byte arrays (about 3.5k tokens at a 152k vocabulary), and adapter aliases
don't route scoring as they route completions.

Tests: `a_prompt_is_scored_by_log_probabilities_and_its_logits_agree` (FFI: log-softmax of the logits blob equals the
scores), the C# and Python equivalents, `each_token_is_scored_by_the_row_before_it`,
`a_token_outside_the_vocabulary_is_an_error`.

## Run 37 - 2026-10-01 (time approximate)

Change (ABI 0.0.18): logits processors are an engine feature. A host registers one by name
(`Engine::register_logits_processor`, `inference_engine_register_logits_processor`, C# `RegisterLogitsProcessor`,
Python `register_logits_processor`) and a chat, completion, Responses or Anthropic messages request selects it with
`"logits_processors": ["name"]`. The registry lives beside the chat engine, shared by every owner's handle. The core
`CustomLogitsProcessor` trait is unchanged: `inference_api::logits_processors::in_place` adapts a
`Fn(&mut [f32], &[u32])`, and both sampler call sites hand processors the 1-D f32 CPU tensor `apply_penalties` builds,
so the adapter's conversions cost nothing. The C callback edits the floats in place and returns 0, or a status that
fails its request. A duplicate name is a conflict (INVALID_REQUEST with `logits_processor_conflict`); an unknown name in
a request is a 400 with param `logits_processors`.

First run of the FFI test: the forced-token half passed, and the failing-processor half's message check failed. A
host's failure surfaces as an internal error, whose detail the engine keeps out of error bodies, so the test checks
INFERENCE_ERR_RUNTIME instead. First full CI: everything passed except the Python coverage test, which had no mapping
for a callback-typed parameter; it now maps `inference_logits_processor_callback` to its CFUNCTYPE.

Review findings:
- A sampling error fails only its own sequence (`handle_seq_error_stateaware_ok!`), but the macro's `return Ok(())`
  left the loop in `sample_and_add_toks_inner`, so every later sequence in the batch lost that step's token while its
  KV cache advanced. Measured with a counting processor on a healthy request batched beside one whose processor fails
  at step 3: old code 7 samplings for 6 tokens in 2 of 3 runs (6/6 when the failing sequence sorted after it), fixed
  code 6/6 every run. The tiny model's greedy text matched either way, so the test asserts samplings == tokens. The
  loop now fails the one sequence and continues. This predates processors (any per-sequence sampling error hit it).
- Binding registrations released the engine when the handle that made them closed, so closing a scoped handle before
  the registration skipped the native unregister and left the name taken by a dead id. Both now hold an engine
  reference until closed, as streams do.
- Responses and Anthropic messages accepted the field and ignored it. Anthropic already ran through `ChatEngine::prepare`
  once the field was passed through; Responses builds its core request separately and now resolves the names too.
- Documented: a grammar request may run a processor twice in a step (the resample after masking), and the views the
  callback gets are valid only during the call.

Tests: `a_registered_logits_processor_steers_the_requests_that_name_it` (FFI),
`a_failing_processor_fails_its_request_and_spares_the_batch`, `responses_and_anthropic_requests_resolve_their_processors`,
the registry unit tests, and the C# and Python equivalents including a registration outliving its scoped handle.

## Run 38 - 2026-10-01 (time approximate)

Question: the engine's tool loop ran only the first call of each round (`agentic_loop.rs` logged "executing only the
first"). What does running every call take?

Finding: each tool's dispatcher appended its own assistant message and tool reply to the request, so a round could only
ever hold one call, and the streaming and non-streaming branches each carried a copy of the round. Progress events and
the non-streaming collector paired a call's two phases by (round, tool name), as did the Responses shell items and the
web UI, which two calls of one tool in a round would break.

Change:
- Dispatchers return a `ToolOutcome`; `run_round` (shared by both branches) resolves every call's dispatcher first, so
  a round with any client-side call goes back to the client whole, then sends each Calling event, runs approvals one at
  a time, runs the calls, and appends one assistant message with every call and a reply per call tagged
  `tool_call_id`. Host callbacks and the HTTP tool run on the blocking pool so they overlap.
- Calls into the session's sandbox (Python, reset, shell, surface outputs) run in the model's order; everything else
  runs alongside them. Found by review: concurrent `execute_python` calls race for the session lock, so `x=1` then
  `print(x)` could run backwards, and two shell calls in one work dir each claimed the other's new files.
- `tool_call_id` on progress events, `agentic_tool_calls` records, `FileSource` (serde default, so stored files load)
  and the Files API metadata; every pairing keys by it, and Responses shell items reuse the model's id.
- Semantic change: an unregistered (client) tool now goes to the client before approval, where it used to be approved
  or denied first.

Tests: the tiny model is scripted with a logits processor that forces a JSON array of calls token by token (the
tokenizer's byte fallback encodes any text but spaces). `every_call_of_a_round_runs_at_once_and_reports_its_own_id`
(two host tools sleeping 300 ms: most concurrent = 2), `a_streamed_round_pairs_each_calls_phases_by_its_id`, and
`calls_into_the_sandbox_keep_the_models_order` (`x=41` after a sleep, then `print(x+1)` must see 42). With the ordering
turned off that last test failed 1 run in 3: the session lock decides, so detection is weak but the race is real.

Known limits: approval prompts carry no `tool_call_id`, and several Ask-mode calls are prompted one after another (up
to the approval timeout each); the CLI prints every call's header before the results.

## Run 39 - 2026-10-01 (time approximate)

Change (ABI 0.0.19): host tools can be registered on a running engine and offered per request. `Engine::register_tool`
/ `unregister_tool`, `inference_engine_register_tool` (taking the existing `inference_host_tool`) /
`inference_engine_unregister_tool`, C# `RegisterTool` and Python `register_tool`. Tools given at load stay offered to
every chat request; a registered tool only to a chat, Responses or Anthropic messages request naming it in
`"host_tools"`. The names resolve in the API layer into a new core `NormalRequest.host_tools`, which also makes core
enter the agent loop; the loop takes them, refuses a name clashing with the request's declared, search, engine or
other host tools, offers their definitions every round and dispatches through the resolved callback, so
`execute_custom_tool` now takes the callback rather than looking the name up on the engine.

Logits processors and host tools share one `Registry<T: Registered>` (moved out of `logits_processors.rs`); the
bindings' registration objects became one generic `HostRegistration` each (Python's `LogitsProcessor` from Run 37
renamed, unreleased).

First full CI: the `inference` SDK and the perplexity example build `NormalRequest` literals I had not checked; both
fixed, and `cargo check --workspace --tests --examples` plus `-p inference-examples --examples` added to my pre-CI
checks.

Review: the header promised `host_tool_conflict` on a clashing request, but the loop's refusal is a plain validation
error with no code; header corrected (status only). Not fixed, documented: registering a name a tool given at load
already has succeeds, and every request naming it is then refused (the API layer does not see the engines' tool
tables); `tool_choice` sees only declared tools; a refused request leaves its input files in the store, as the existing
internal-tool clash check already did.

Tests: `a_tool_registered_after_load_answers_the_requests_that_name_it` in the engine (named: answered by the
callback; unnamed: never called; clashing with a declared tool: refused; unregistered: 400 on `host_tools`) and in the
ABI (registration through `inference_host_tool`, duplicate refused, chat answered, unregister twice NOT_FOUND), and the
C# and Python registration checks.

## Run 40 - 2026-10-01 (time approximate)

Question: what does moving the `inference` SDK onto inference-api take, and what has no route there yet?

Finding (a read-only mapping of every public SDK item onto inference-api): most of the SDK maps directly; `Model`
wraps `Engine`, the builders produce `EngineSpec` + `EngineCallbacks`, the message builders produce a
`ChatCompletionRequest` plus `MediaAttachments`, streaming reuses `ChatStream`/`ChatStreamEvent`, and agent.rs goes
(host tools, `max_tool_rounds`, the engine's parallel round). `examples/rust/Cargo.toml` has 59 examples.

Gaps with no inference-api route today:
- Engine (needs ABI and bindings): tokenizing chat messages (`TokenizeRequest` takes text only), a `model` on re-ISQ
  and calibration, `hf_revision`, inline topology and ordering (paths only), the speech generation config, a
  best-effort paged-cache size, stop token ids, a per-request `tool_dispatch_url`, sequential tool rounds
  (`parallel_tool_calls`), f32 speech samples (PCM16 today).
- Rust-only (no ABI form): decoded media attachments (`DynamicImage`, `AudioInput`, pre-decoded `VideoInput`),
  `Default` on the request types, a private field blocking `LoadLoraAdapterRequest` literals, custom `Pipeline`
  injection, `Model::inner()` access to core.
- Kept in the SDK as conveniences: an async tool adapter, `chat_with_approval` (stream, answer approvals through
  `resolve_approval`), `with_shell_skill(path)` (upload, then reference), `generate_structured`, `save_file`.

Behaviour changes the move brings: Ask permission needs streaming, the engine names forked sessions, sessions are
scoped by owner rather than model, the engine's throughput logging defaults on. `inference-macros` emits SDK agent
types, so it changes with `Model`. Planned phases: A engine additions, B builders to specs, C `Model` on `Engine`, D
remaining examples, E drop the core dependency, F docs.

## Run 41 - 2026-10-01 (time approximate)

Change (ABI 0.0.20), the engine gaps worth keeping from Run 40:
- `Engine::tokenize_chat` / `inference_tokenize_chat`: a chat request tokenized as its template renders it, with
  tools, reasoning controls and the generation prompt. The Anthropic count-tokens path now calls the same function.
- A `model` on `ReIsqRequest` and `CalibrationApplyRequest`, and `CalibrationTarget {model}` for calibration start
  and status; the ABI's start and status take a request (an ABI break, allowed at 0.0.x), HTTP takes `?model=`.
- `hf_revision` on `runtime` (the single `model`) and on each `ModelSpec`, through to `ModelLoaderConfig`, where both
  sites had hard-coded `None`; `runtime.hf_revision` with `models` is refused.
- `stop_token_ids` beside `stop`: core's `StopTokens` became `{seqs, ids}` so a request can carry both.
- `parallel_tool_calls: false` (chat, and Responses where it was accepted and ignored) runs a round's calls one at a
  time in the model's order, the only reading the engine can honour for calls it runs; the model may still make several.
- `ModelSelected::Speech { generation }`: Dia's sampling per load, each unset field keeping its default.
Dropped, as decided: inline topology and ordering, a best-effort paged cache size, a per-request dispatch URL, f32
speech samples.

Dead end: the scripted-token test helper tracked its position by context length and restarted when the engine sampled
the same context twice (output "aab" for "abcdefgh"); it now plays the longest start of its script the context ends
with, so it holds no state.

Review:
- `operations::send` had started checking the named model was loaded, which broke on-demand reload for tokenize and
  detokenize (409 instead of waking the model) and bought nothing, since `get_sender`'s `ModelNotFound` already maps
  to NOT_FOUND. Removed; the not-found tests still pass.
- Stop ids went through the prefix check meant for single-token stop strings, so a newline id was refused, and an
  id past the vocabulary got a message about an empty string. Ids now only have to be inside the vocabulary.
- Header: `tokenize_chat` cannot resolve `media://N` sources (the call takes no buffers).
Not fixed: the server has no `/tokenize` or `/detokenize` routes (it never did; count_tokens is Anthropic's).

Tests: `a_chat_tokenizes_as_its_template_renders_it_and_counts_the_same` (ABI: chat tokens wrap the text's, and match
count_tokens), calibration and re-ISQ of an unknown model are NOT_FOUND (ABI, C#, Python),
`a_stop_token_id_ends_the_generation_where_it_is_produced` (scripted tokens, stop on the third: 3 tokens, `stop`;
an out-of-vocabulary id refused), `a_request_that_turns_parallel_calls_off_runs_them_one_at_a_time` (most
concurrent = 1, model order kept), the hf_revision spec rules, `a_speech_generation_spec_overrides_only_what_it_sets`.

## Run 42 - 2026-10-01 (time approximate)

Change: the `inference` Rust SDK is rebuilt on inference-api. It depends on inference-api (and inference-macros) only:
no inference-core, inference-selection, inference-agent or candle outside its test fixtures.
- `Model` wraps `Engine` and derefs to it, so every engine operation is the same call the server and the C ABI make;
  `Model` adds Rust conveniences (builder-typed chat and streams, structured output, embeddings, generation,
  request-scoped logits processors registered and unregistered around the request, `upload_skill`, `re_isq_model`).
- `RequestBuilder` / `TextMessages` / `MultimodalMessages` build a `ChatCompletionRequest` (messages as JSON, since
  `MessageContent` has a private field) plus decoded `Media` attachments named `media://N`, a new Rust-only
  `inference_api::media_source::Media` the image, audio and video loaders take without decoding.
- Every builder produces an `EngineSpec` + `EngineCallbacks` (`into_spec()`), sharing `LoadOptions` and one
  `load_options_methods!` set; the old `model_builder_trait.rs` pipeline construction is gone.
- agent.rs (the client-side tool loop) is deleted; `#[tool]` emits a `ToolCallbackWithTool` for the engine's loop.
- Engine additions: `Engine::chat_with_approver` (Ask answered by an in-process handler, no stream needed),
  `inference_api::sdk` re-exports of the engine-internal types a Rust caller names, protocol constructors.
- 59 examples, the SDK's integration tests (all passing, the 7 real-checkpoint parity tests included) and the Rust
  docs are ported; the queue-drop test that reached core's request channel moved into inference-api's engine tests.
Net about 4.4k fewer lines.

Dead ends and catches:
- The SDK's old requests were greedy by default (`SamplingParams::deterministic()`, top-k 1); the first rebuild sent
  the engine's defaults, which silently changes outputs. `RequestBuilder::new` sets top-k 1 again.
- Review: with no `with_paged_attn` the engine turns paged attention on for CUDA and sizes it at 90% of memory, where
  the old SDK used a plain KV cache; `LoadOptions` now sets it off until asked. `with_device("cuda")` failed (the spec
  wants `cuda:0`); bare names now mean the first device. An approval callback on a stream was silently ignored; a
  stream with one is now refused. `blocking.rs` was never declared as a module, and the crate docs still showed
  `Response::Chunk`; both fixed (the doctests and rustdoc caught them in CI).
- Two example bugs the port found: the AnyMoE examples passed path, prefix and mlp in the wrong order; the streaming
  example's buffered stdout was never flushed.

Dropped, as decided or with a replacement: custom pipeline injection and `Model::inner()`; in-memory `Topology` and
`Ordering` (file paths); `DeviceMapSetting` (`with_device_layers`, `with_auto_map_sizing`); `MtpConfig`
(`with_mtp_model`, `with_builtin_mtp`, `with_mtp_draft_sampling`); f32 speech samples (WAV/PCM bytes); a per-request
dispatch URL; `with_shell_skill(path)` (`upload_skill(dir)` then the id). Changed, not dropped: per-model settings in
`MultiModelBuilder` are what `ModelSpec` carries (template, ISQ, device layers, revision, encoder cache); engine-wide
ones come from the multi-builder. Not reachable any more: the client loop's stop reason and per-tool OK/error status,
and its "calling N tools" / "round done" events; loads always log (the engine has no silent load).

## Run 43 - 2026-10-01 (time approximate)

Question: with the SDK off core (Run 42), what in core, nn and quant is now dead?

Method: an audit parsed every `inference_core::...` path outside core (crates/ and examples/) against core's re-exports,
and for nn and quant counted each public item's name outside its crate, cfg-gated code included. Candidates were then
cut and the CPU and CUDA builds (`--features inference-core/cuda`) checked after each step, since the dead-code lint
only sees what a narrowed visibility exposes.

Change (1288 lines removed):
- Dead functions: core's per-engine terminate flags (and their static) and `get_model_file`; 19 nn methods (KV cache
  views, masker helpers, `from_qparts`, topology helpers, a CUDA pool query, ...); 10 quant functions (the legacy ISQ
  rayon pool, `matmul_affine_div`, four Metal kernel wrappers, ...). Deleting them exposed a second layer: a private
  mask helper, the csv import, two LoRA runtime helpers.
- `Response::as_result` with `ResponseOk` / `ResponseErr` (about 150 lines): the old SDK's, now used only to log one
  error in distributed.rs, which matches the error variants directly.
- `InferenceRs` methods nothing calls, found iteratively (removing a wrapper left its delegate unused): the blocking
  and file-based LoRA variants, `with_agent_runner`, `with_no_prefix_cache`, `with_tool_callback_with_tool`,
  `list_unloaded_models`, `attach_file_to_session`.
- About 30 re-exports in core's lib.rs nobody reads through core (protocol reasoning and file helpers, nn logging
  filters, `layers`, code-exec approval types, `SpeculativeConfig` / `matformer` now private uses).
- `speculative` became `pub(crate)`, which surfaced dead trait methods (`begin`, `make_verify_input_metadata`,
  `build_speculative_verify_inputs` with their impls) and helpers; `speculative_prepare_propose` is CUDA-only and now
  says so. quant's `f8q8` and `gemv` became private: with `set_enabled` gone the GEMV controller could never be off,
  so it is removed, and the CPU stubs of `gemv` / `should_use_gemv` had no caller (every call site is CUDA-gated).

Dead end: `cargo fix` on the CPU build removed imports the CUDA build needed (`LazyLock` in gemv,
`SpeculativeProposePrepareCtx` in the driver); both restored behind `cfg(feature = "cuda")`. Metal-only deletions are
in `metal_kernels` files that import by glob, so the PR's metal check is the verification.

Not done: narrowing the remaining crate-only `pub` items (about 540 in nn, a few hundred in quant, the per-family
loaders in core) to `pub(crate)`; it deletes nothing by itself, though it would let the lint find more.

## Run 44 - 2026-10-01 (time approximate)

Question: how much of DeepSeek-V2, DeepSeek-V3, GLM4-MoE and GLM4-MoE-Lite (inference-models-other) is one model?

Finding (a read-only diff map):
- DS3 is DS2 with a different router: about 89 of 1117 lines differ, all in `MoeGate` (noaux_tc with an optional
  `e_score_correction_bias`, a different renormalisation rule).
- GLM4-MoE-Lite (GLM-4.7-Flash) is a DS3 clone with hard-coded choices (required q_lora and bias, no yarn mscale,
  replicated shared expert); plain GLM4-MoE shares the MoE half (router body, `Moe` skeleton, decoder forward).
- Four distinct renormalisation rules exist and must stay distinct: DS2 renormalises with `norm_topk_prob` and then
  skips the scale; DS3 renormalises only under sigmoid scoring and always scales (ignoring `norm_topk_prob`); Lite
  always renormalises; GLM renormalises on `norm_topk_prob` (default true) and scales.
- No test runs a forward pass of any of the four; coverage is loader predicates, residual names and MLA helpers.
- Two existing bugs: `GroupLimitedGreedy` masks with `masked_fill(&score_mask, ..)` where it should mask the scores,
  so expert choice within the allowed groups is arbitrary and every weight is 1.0 (full DeepSeek-V2/V2.5
  checkpoints); DS2's non-greedy renormalisation divides (n,k) by (n,1) without broadcasting, so it errors.
- Smaller drifts: loader memory sizing uses `intermediate_size * n_shared` for DS shared experts where the model uses
  `moe_intermediate_size`; loaders report a `Standard` KV layout while the models use MLA.

Plan: tests first (router goldens per variant; tiny random-weight checkpoints of each model with a CPU forward and
snapshotted logits; a check that every tensor a checkpoint provides is consumed), then a shared router in
inference-nn, a `deepseek_family` module the four delegate to, shared loader helpers, and the bug fixes last, each
on its own with its golden updated. Estimated saving about 2.4k lines after the tests.

## Run 45 - 2026-10-01 19:50

Question: lock today's behaviour of the four DeepSeek/GLM4-MoE models before the Run 44 refactor.

Command: `cargo nextest run -p inference-models-other -E 'test(/family_tests/)'` (26 tests; shared fixtures in
`src/deepseek_family_tests/mod.rs`, one `family_tests` file per model attached with `#[path]`). Router goldens use an
identity gate so the hidden states are the logits; the non-pinned goldens match a numpy reference to 1e-6. Each
forward test builds a 2-layer checkpoint (dense layer 0, MoE layer 1, 8 experts, hidden 32) through the model's
loader, asserts the provided and requested tensor name sets are equal, and snapshots 4 logits plus sum and L2.

Findings:
- Run 44 was wrong about the group-limited weights: they are 0.0, not 1.0. `1. - &score_mask.ne(0.)?` on a u8
  tensor does not invert (every element comes out 1), so `masked_fill` zeroes everything and `topk` lands on experts
  0 and 1 for every token with zero weight: the routed experts contribute nothing in DS2/DS3 group-limited models.
- CPU eager attention (`run_flash_attn_cpu`) sets `dv = d` from the query head, so any MLA model with
  `v_head_dim != qk_nope_head_dim + qk_rope_head_dim` (DeepSeek-V2/V3: 128 vs 192) produces
  rows of the wrong width and fails at `o_proj`. The tiny checkpoints use v_head_dim = 16 to get a forward at all;
  `forward_narrow_v_head_errors_on_cpu` pins the failure.
- The split `k_b_proj`/`v_b_proj` path only works with 3-D (GGUF-bound) weights. 2-D safetensors k_b/v_b load (the
  names are consumed) and then fail in `expanded_split_weights` with "unexpected rank"; pinned per MLA model. The
  absorbed path would also hit the CPU `dv = d` issue (q is kv_lora + rope wide, v is kv_lora wide).
- GLM4-MoE and Lite build the shared expert at `moe_intermediate_size` regardless of `n_shared_experts`; the tests
  use n_shared_experts = 2 so a fix to multiply would show up as a coverage failure.

Not covered: paged attention, the CUDA MLA decode/cache paths, yarn rope scaling, tied embeddings, and the GGUF
split-weight load; the forward tests are CPU F32 eager only.

## Run 46 - 2026-10-01 (time approximate)

Question: does the Run 44 consolidation (shared router, `deepseek_family` module, shared loader helpers) keep every
pinned behaviour of DeepSeek-V2/V3 and GLM4-MoE(-Lite)?

Commands: after each step `cargo nextest run -p inference-models-other`, `cargo check --workspace --tests`,
`cargo clippy -p inference-models-other -p inference-nn --tests -- -D warnings`, and
`cargo check -p inference-models-other --features cuda`; once at the end
`cargo nextest run -p inference-core -p inference-gguf -E '(package(inference-core) & test(/normal_loaders|loaders::/)) | package(inference-gguf)'`
(172 passed). For the loader step, a throwaway test dumped every loader output (promoted/ISQ/MoQE regex strings in
order, layer sizes at pack factors 1, 2 and 4, non-mapped size, model config) for 7 DeepSeek and 5 GLM config
variants before and after; the dumps were byte-identical. A smaller pin of the layer sizes was committed.

Findings:
- The family tests passed unchanged through all six steps; only the gate constructor's path moved
  (`deepseek_family::MoeGate::new(&cfg.family(), ..)`).
- Flake at the starting commit (8cb99e5e), before any change: `deepseek2::family_tests::forward_narrow_v_head_errors_on_cpu`
  failed twice (forward returned Ok instead of "shape mismatch in matmul"), both on the first run after a fresh
  build, then passed in about 20 runs since. Cause not found; worth a look before anyone leans on that pin.
- Differences that looked identical but are not, kept as switches: DS2's non-greedy renormalisation divides
  without broadcasting (DS3/GLM broadcast); DS2 lacks the `quantization` serde alias; Lite builds the paged MLA KV
  layout only on a CUDA device while DS2/DS3 build it whenever paged attention is on; the DS loaders size the q
  projections unpacked and write the dense up_proj ISQ pattern with bare dots; DS3's loader leaves the
  correction bias out of the layer size while Lite/GLM count it; GLM4-MoE ignores `moe_layer_freq`.
- Lines: the four models and their loaders went from 5401 to 2822, plus the 204-line router in inference-nn.

Next: the Run 44 bug fixes, each with its golden updated, and the flake above.

## Run 47 - 2026-10-01 (time approximate)

Fixes on top of the consolidation (Run 46), each against its own updated goldens:
- CPU attention read value rows at the query head width (`let dv = d` in the CPU flash kernels), so a model whose
  value heads are narrower than its query heads (DeepSeek-V2/V3: 128 vs 192) failed on CPU, or, depending on what
  lay past each row, returned a wrong answer. That was the flaky pin Run 46 saw (`forward_narrow_v_head_errors_on_cpu`
  passing about one run in ten after a fresh build). The kernels now use `v`'s width; a new kernel test against the
  naive reference fails without the fix, and the DS2 narrow-value forward is a snapshot now.
- Group-limited routing masked the 0/1 group mask instead of the scores, and DS2's `norm_topk_prob` division did not
  broadcast. The router now multiplies the scores by the mask (HF's `scores.masked_fill(~score_mask, 0.0)`) and
  broadcasts. The goldens are HF DeepSeek-V2 values computed independently, with `topk_group: 1` so the group limit
  changes the answer (with two groups kept on these logits it picks what plain greedy does).
- Shared experts: HF builds them `moe_intermediate_size * n_shared_experts` wide in all four models (checked in the
  GLM4-MoE and GLM4-MoE-Lite modeling files). GLM built one expert's width; the DeepSeek loaders sized them at
  `intermediate_size * n_shared` for the device map. The loader size changes were derived by hand (DeepSeek MoE layers
  -12288 bytes, GLM +3072 at the test's dims, F32, pack factor 2) and matched before the constants were updated.
Left as found, noted: Lite uses the MLA paged layout only on a CUDA device (DS2/DS3 whenever paged attention is on);
DS3's loader leaves the correction bias out of its size; the DS loaders do not divide q projection sizes by the pack
factor; the dense `up_proj` ISQ pattern has unescaped dots; loaders use `%` on `moe_layer_freq`, which panics at 0.

Review of the branch (subagent, against master and HF): no unintended change to weight paths, device mapping, ISQ
patterns, paged/MLA layout, rope, the GLM decode-graph flag or the GGUF-synthesised configs. Its findings, acted on:
- DS3 renormalised under sigmoid scoring only, ignoring `norm_topk_prob`; HF DeepSeek-V3 renormalises when
  `norm_topk_prob` (config default true) and always scales. DS3 now reads `norm_topk_prob` (default true) and shares
  GLM4-MoE's rule; the GGUF synthesis already writes the key from `expert_weights_norm`. Real V3/R1 configs (sigmoid,
  true) route as before; a softmax config now renormalises unless it says false. The new golden is the existing
  softmax one renormalised by hand (1.556148/0.943852, 1.61414/0.88586) and matched first time.
- The narrow-value kernel test only reached the tiled f32 path; two more cover mask, softcap (the full-qblock
  fallback) and bf16/f16 (`compute_full_row`). All three fail with the kernel fix reverted.
- Stale comments from before the fixes removed; `O_PROJ` named for the GLM GQA pattern; test-only items `pub(crate)`.
Also left as found: GLM4-MoE shards KV for tensor parallelism from `hidden_size / num_attention_heads` rather than
`head_dim`, which differs for GLM-4.5 dims; the loaders still report a `Standard` KV layout for MLA models (Run 44).
- CI `--cuda`: the three group-limited router tests panicked (`not implemented!`, `ops/topk.rs`). With the `cuda`
  feature built, `topk_unsorted` and the MoE gather backend's prefill sort always used the custom `ArgSort` op, whose
  CPU path panics, so a CPU-device DS2 group-limited model in a CUDA build crashed. `ArgSortOp` now sends non-CUDA
  tensors to candle's `arg_sort_last_dim`/`sort_last_dim`, and the callers' `cfg(feature = "cuda")` splits are gone.

Issues filed for what Run 47 left as found: #209 (MLA loaders report a Standard KV layout), #210 (GLM4-MoE KV shard
head dim), #211 (MLA paged layout device condition), #212 (`moe_layer_freq: 0`), #213 (loader sizing drifts), #214
(GLM4-MoE-Lite ignores `norm_topk_prob`). Writing #212 up showed the models use `is_multiple_of`, which treats only
layer 0 as MoE at frequency 0, so the loaders and models also disagree there.

## Run 48 - 2026-10-01 (time approximate)

Question: how much of the Qwen-VL family's wrapper and text code is one implementation in two copies?

Measured (`diff` line counts between the pairs):
- `qwen2vl/mod.rs` vs `qwen2_5_vl/mod.rs`: the wrappers differ only in the vision tower type; the text model and
  config were already generic over the vision config.
- `qwen3_vl/mod.rs` vs `qwen3_vl_moe/mod.rs`: the wrapper bodies differ only in the text model type.
- `qwen3_vl/text.rs` vs `qwen3_vl_moe/text.rs`: 196 lines. The MoE text config is a superset of the dense one, and
  with no experts the MoE layer selection builds every layer dense. The real differences: the dense model uses
  `F32RmsNorm` for the layer and final norms where the MoE one uses the fused `RmsNorm`, and the dense MLP casts its
  output back to the input dtype.
- `qwen3_5/mod.rs` vs `qwen3_5_moe/mod.rs`: 60 lines, but the dense model carries MTP speculative decoding and the
  DFlash drafter (text 3135 vs 1005 lines) and the two use different decode MRoPE position helpers. Not a type-level
  merge; left for its own investigation.

Pinning first: there were no model-level tests for any of these. `inference_nn::testing` now holds the tiny
random-weight fixtures the DeepSeek-family tests used (moved, not copied; those 27 tests pass unchanged), plus
`load_synthesized`, which makes up each tensor a loader asks for at the requested shape, seeded by name, with the
MLX-layout probes answered absent. Two snags: the quantised linear layers check `contains_tensor` before loading,
so presence is "everything but the listed prefixes", and MoE expert layout detection reads tensor shapes up front,
so the stacked expert shapes (HF's transposed `[E, H, 2I]` / `[E, I, H]`) are declared. Text-only prefill snapshots
for Qwen2-VL, Qwen2.5-VL, Qwen3-VL and Qwen3-VL-MoE; the Qwen2 pair give identical logits, as they should with the
vision tower idle. The Qwen2 snapshots also pass with the pre-refactor wrapper files checked out.

Changes:
- `QwenVlModel<V: QwenVlVision>`: Qwen2VLModel and Qwen2_5VLModel are aliases; `compute_rope_index` is a free function.
- One Qwen3-VL text config and text model; a model without experts keeps the F32 norms. Qwen3VLMoEModel is an alias.
- Lines: the four Qwen-VL modules went from 11122 to 9512; the branch is -2251/+958 including the moved fixtures
  and the new tests.

Next: the Qwen2-VL and Qwen3-VL input processors (1501 of ~1900 lines differ, so a spec-driven share, not a merge).

Review of the branch (subagent): no behaviour change for real checkpoints of the four models. Acted on:
- Defaulting the MoE fields turned a missing `num_experts_per_tok` into `top_k = 0` routing and a missing
  `num_experts` into a misleading dense-weights error, and let the dense loader take a MoE checkpoint (sized and
  ISQ'd as dense). `TextConfig::check_experts` now runs in both loaders: qwen3vl rejects experts, qwen3vlmoe
  requires nonzero experts, top-k, expert width and `decoder_sparse_step` (whose 0 already panicked the loader's `%`).
- The F32 snapshots could not see the norm choice (F32RmsNorm and the fused RmsNorm agree far inside 1e-4 in F32)
  or vision-side weight names (a text-only prefill never runs the tower). Added BF16 snapshots for dense and MoE,
  which fail when the norm choice is flipped, a pinned digest of every tensor name each load reads, and a check that
  every `residual_tensors()` name is one the load read (the names ISQ and UQFF serialise). With the pre-merge Qwen3-VL
  sources checked out, all of these pass unchanged; only the new validation test fails there, as it should.
- Narrating comments carried over from the MoE text file removed; the dense file's imatrix note restored as one
  line; unused `max_window_layers`/`use_sliding_window` dropped; `TextNorm::new` takes the config, not a flag.
Left: HF uses the F32-then-cast RMSNorm for Qwen3-VL-MoE too, so the fused norm there is a small BF16 rounding
difference from HF that predates the branch; a real-checkpoint parity run would say whether `TextNorm` can go (#215).
`inference_nn::testing` stays compiled into normal builds: a dev-dependency feature would build inference-nn and
everything above it twice (test and non-test feature sets).

## Run 49 - 2026-10-01 (time approximate)

Question: how much of the three `load_model_from_path` copies (normal 479, multimodal 664, embedding 339 lines) is
one load, and what actually differs?

Finding (`diff` of the function bodies): most single steps were already shared `super::loading` helpers; the
duplication was the orchestration around them. Two blocks repeat in all three:
- devices, weight sources, map setting, mapper, ISQ plan, attention choice, load metadata, the weight-mode log;
- the weight dispatch: tensor parallel or not, times plain, LoRA or X-LoRA, times prepared source or files.
The real differences, kept as explicit inputs:
- the config: normal sizes everything from the runtime config (after `max_model_len`); multimodal sizes devices,
  the device map, the ISQ plan and a tensor-parallel mapper from the source config but builds the model from the
  runtime config.
- matformer: both load the slice for the model, but only multimodal applies it to device-map sizing.
- `non_mapped_unpacked` (multimodal), the multimodal `auto_device_map_params` adjustment, embedding's fixed
  organization and no LoRA, prepared source or matformer.
- tensor-parallel X-LoRA uses the distributed mapper, and a tensor-parallel LoRA load with no prepared source reads
  the weight files (the sharded var builder is dropped). Both kept.
- multimodal's prepared-source LoRA path built a plain `LoraLayerRegistry` where everything else used
  `new_dynamic_lora_registry`; that only differs for the Qwen3-Next text architecture, so sharing it changes nothing.

Changes: `open_load_session` returns a `LoadSession` and the mapper; `load_model` (generic over a small
`BuildModel` trait implemented for the three loader trait objects) builds the model, with X-LoRA as a hook.
The per-pipeline `weights_vb`/`load_from_files`/`load_with_dynamic_lora` are gone. The three functions went from
1482 to 931 lines; core is -901/+611.

Checks: the tiny engine tests drive all three pipelines (plain, ISQ, UQFF write and reload, calibration, paged
multimodal) and pass. LoRA had no test, so `llama_tiny::a_lora_adapter_applies_only_when_a_request_selects_it`
pins it (no adapter decodes like the base model, the adapter moves the logits); it passes on master's loader too.
Still untested: X-LoRA, tensor parallel, prepared-source loads.

Drifts left as found: the normal pipeline ignores the matformer slice when sizing the device map; multimodal sizes
from the pre-`max_model_len` config; multimodal's LoRA qk-rope layout check ignores X-LoRA (it has no X-LoRA path).
Filed: #218 (the config drift, with the matformer note) and #219 (the untested load paths).

Review of the branch (subagent): every case matches master, across all 12 weight-dispatch combinations and each
step's config. Low items, accepted as they are: multimodal prepared-source LoRA now builds its registry with
`new_dynamic_lora_registry` (same result for every config that exists); kinds that hit `unreachable!()` now load as
plain, though no builder produces them; normal's matformer slice and the LoRA runtime `expect` now run before the
devices are set up, which only changes which error shows first; the config trace logs after the ISQ plan. Fixed: an
empty `impl`, a redundant destructure field, a doc on `open_load_session` that was wrong under tensor parallelism,
two comments restating field names. CI passed (CPU 2333, CUDA 2654) before these cosmetic fixes.

## Run 50 - 2026-10-01 (time approximate)

Question (#218): does the multimodal pipeline sizing from the pre-`max_model_len` config change any load?

Traced every consumer of the sizing config in `open_load_session` and the tensor-parallel mapper for the four
multimodal loaders that rewrite `max_position_embeddings` (Qwen3.5, Qwen3.5-MoE, Muse-Glimmer, Gemma 4):
- `get_device_layers` takes `max_seq_len` from the auto params; its paged-KV estimate hands `calculate_cache_config`
  an explicit `max_seq_len * max_batch_size`, so `model_config().max_seq_len()` (the fallback) is never read there.
- layer sizes, activation sizes (the params' `max_seq_len` again), the tensor-parallel decision (head counts) and
  the ISQ plan read nothing context-dependent.
Finding: latent, not live; no load changes today. The issue overstated it and got a correcting comment.

Change: the multimodal session, its auto device map adjustment and the tensor-parallel mapper now take the runtime
config, as normal's do; the UQFF artifact still keeps the source config. Every pipeline now sizes and builds from
one config, so `ModelLoadInputs::session_config` is gone. No test can see the difference, since nothing reads it.
