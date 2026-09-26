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
