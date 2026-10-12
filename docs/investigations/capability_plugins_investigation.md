# Capability plugins investigation (#425 phase 5)

Can the host library split into capability plugins (text, audio, image, layout) for a non-text package, and where would the
lines fall, given that model features (KV cache kinds, sliding window, linear attention, MoE, MTP) are shared across
modalities?

## Run 1 - 2026-10-11 21:00

Question: how are the model features spread over the models and modalities, and what do they and the modality leaves cost
in lines and in compiled code?

Method:
- Lines: `.rs` files per crate and module, excluding tests and `third_party/`.
- Use: for each model module in the family, speech and diffusion crates, a regex over its source for each feature's types:
  - SWA: `RotatingCache`, `sliding_window`, `layer_types`;
  - hybrid: `HybridCache`;
  - GDN: `GatedDeltaNet`, `GdnConfig`;
  - conv/SSM: `ShortConv`, `causal_conv1d`, `ssm::`;
  - MoE: `MoEExperts`, `GroupedRouter`;
  - MLA: `mla::`;
  - MTP: `MtpConfig`, `DFlash`;
  - MatFormer: `MatformerConfig`;
  - vision and audio encoders;
  - KV/paged attention.

  A first pass on module paths was noise: every multimodal input processor names `gdn::RecurrentBatchKind`, and every
  model implements `SpeculativeTargetMixin`. Both are engine-wide contracts, not per-model choices.
- Bytes: `RUSTFLAGS= cargo build --profile bundle -p inference-ffi --features cuda --config profile.bundle.strip=false`,
  87.6 MB. Then `nm -S --size-sort -C --defined-only`, with text symbols (61,581) assigned to the first
  `inference_<crate>::<module>` in the demangled name, else the first other crate. 48.1 MiB in all. Kernels are separate
  libraries since phase 1 and are not counted here.

Lines per crate (top): quant 105.7k, nn 89.2k, core 66.7k, tensor 46.7k, qwen 30.8k, gemma 23.7k, llama 22.3k, api 20.1k,
protocol 17.2k, other 16.4k, gguf 14.9k, speech 12.9k, phi 8.9k, layout 5.3k, diffusion 4.2k.

Features by model (abridged):

| model | features |
|---|---|
| Qwen3.5 (dense, MoE) | SWA, hybrid cache, GDN, MoE, MTP (DFlash), vision |
| Qwen3-Next | SWA, hybrid, GDN, MoE |
| Gemma 4 | SWA, MoE, MTP, vision, audio encoder |
| Gemma 3n | SWA, MatFormer, vision, audio encoder |
| Phi-4 MM | SWA, vision, audio encoder (conformer) |
| Voxtral | SWA, audio encoder, KV |
| Granite (hybrid), LFM2 | SWA, hybrid, conv/SSM (LFM2 also MoE) |
| DeepSeek V2/V3 | MoE, MLA |
| Llama 4 | SWA, MoE, vision |
| Dia (TTS) | KV cache, sliding window |
| Parakeet (ASR) | sliding (local) attention band |
| HiDream-O1 (image) | a Qwen3-VL-style decoder of its own |

- Sliding window is nearly universal.
- The newest models compose the most features: Qwen3.5 has six.
- Audio reaches the engine two ways:
  - as audio-LLMs inside the text families (Voxtral in llama, the conformer in phi, Gemma 3n/4, MiniCPM-o and Muse-Glimmer in qwen);
  - as the speech crate's own models (Parakeet, Sortformer/diarization, Silero, Kokoro, Dia).
- Every pipeline (text, speech, diffusion, transcription, embedding) implements the one `Pipeline` trait with its ISQ and
  AnyMoE mixins, and runs on the shared engine (sequences, scheduler, prefix cacher types).

Compiled host code of each feature (lines in brackets):

| feature | code | lines |
|---|---|---|
| paged attention (nn) + cache manager/prefix cacher (core) | 469 KiB | 14.7k + 2.9k |
| GDN (`nn::gdn` + `nn::cuda::gdn`) | 388 KiB | 5.6k + cuda |
| speculative/MTP (nn + core + rejection + DFlash context) | 308 KiB | 1.8k + 3.3k |
| MoE + AnyMoE (nn + core) | 230 KiB | 3.9k + 0.7k + 0.9k |
| KV caches incl. hybrid | 137 KiB | 5.8k |
| vision building blocks | 126 KiB | 1.9k |
| CUDA graphs (core) | 104 KiB | 3.3k |
| MLA | 44 KiB | 1.4k |
| MatFormer | 21 KiB | 0.2k |

The features are thousands of lines each, but mostly control logic: together ~1.8 MiB of the 48 MiB.

Split candidates (text symbols):

| group | MiB |
|---|---|
| text families (llama 1.12, qwen 1.53, gemma 1.22, phi 0.49, other 0.83) | 5.18 |
| speech leaf (speech models 1.11, phonemizer 1.96, rustfft 0.58, audio IO 0.1) | 3.74 |
| agent/tools (agent, MCP, code exec, sandbox 0.88; html5ever/html2text 0.44) | 1.32 |
| text-only deps (llguidance + derivre/toktrie 0.66, inference-gguf 0.41) | 1.06 |
| layout | 0.41 |
| diffusion (image) | 0.28 |
| shared core: tensor 10.3 (CPU backend 4.1), nn 3.1, quant 2.6, core 2.6, api, protocol, selection, std/alloc/core, tokenizers, minijinja, serde, TLS/HTTP, image codecs | ~36 |

Implications:
- **Splitting by feature buys nothing and costs a lot.** Each feature is 20-470 KiB, and the newest models compose most of
  them. A Qwen3.5-class model would load six feature plugins, each a stable boundary to version and test. The features
  belong in the shared core.
- **Even a split by modality is modest.** A non-text package (speech plus image plus layout) could drop the text families,
  the text-only deps and the agent layer: ~7.6 MiB of 48 MiB of code, maybe ~10 MiB with tables. The audio-LLMs (Voxtral,
  Gemma 3n/4, Phi-4 MM) live in the text families, so "audio" for them means text.
- **The bytes are elsewhere:**
  - kernels (~47 MiB per arch), which phases 2-4 already split;
  - the tensor CPU backend (4.1 MiB of dtype-monomorphized kernels);
  - the phonemizer data (2 MiB, which only Kokoro needs);
  - the shared engine.
- **Next questions:**
  - Is ~10 MiB worth a Rust-dylib plugin boundary (same-build only, like the kernel ids)?
  - Or are plain Cargo features per package enough? Vernacula builds its own package, so a `--no-default-features` build
    with the speech crate and no families may already be the non-text package.
