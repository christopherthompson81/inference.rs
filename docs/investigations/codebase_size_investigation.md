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
