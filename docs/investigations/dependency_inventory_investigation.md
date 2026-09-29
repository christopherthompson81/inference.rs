# Dependency inventory investigation

A pass over every external crate the workspace declares: what each one is for, how much of our code uses it, what
it costs to build, and whether keeping it is reasonable.

## Run 1 - 2026-09-28 (evening)

- Question: which external dependencies earn their build cost, which are redundant, and which pull in more than
  we use?
- Commands:
  - Timings: the cold `cargo test --no-run --features cuda --workspace --lib --bins --tests --timings` of build-time
    Run 39 (scratch target, 254 s, 2318 unit-seconds). Per-crate cost is the sum of that crate's units: lib, build
    script, and build-script run.
  - Graph: `cargo metadata --features cuda`, using the resolved graph.
  - Exclusive cost of a dependency: the unit-seconds of every crate reachable from it that becomes unreachable once
    the workspace's own edges to it are cut. In other words, what dropping it from the workspace would save.
  - The graph is feature-resolved, so a crate that arrives through a feature flag rather than a dependency edge
    (aws-lc via reqwest's `rustls` feature, AVIF via `image`'s default formats) is not charged to its cause. Those
    were traced by hand with `cargo tree -e features -i <crate>`.
  - Duplicates: `cargo tree -d`.
  - Usage: files and references matching `<crate>::`, `use <crate>` or `#[derive(<crate>...)]` in the declaring
    crates' sources.
- Scope: 27 workspace members declare 120 external crates, which resolve to about 830 compiled units. None of them
  is on the cold build's critical path (candle-kernels, then candle-core, then core's lib test): every external crate
  finishes before core's lib starts. So each finding below saves CPU, not idle-machine wall time.

### Findings

Ranked by what they would save. The CPU seconds are per cold build.

1. **rustfft's SIMD kernels in our own `inference-audio`: ~30 s CPU, 938k IR lines in a 265-line crate.**
   - The planner is instantiated for f32 and f64, each with SSE, AVX and scalar kernels:
     - SSE: 404k IR lines, only a fallback for x86 CPUs without AVX.
     - AVX: 323k.
     - Scalar: 167k.
   - rubato's default `fft_resampler` feature also turns rustfft's SIMD back on through realfft. We only use rubato's
     sinc resampler.
   - Fix: `rubato` with `default-features = false`, and `rustfft` with `default-features = false` (scalar only, or
     `avx` only). Frame FFTs of 256-512 points are microseconds either way.
   - The f64 planner (522k lines) is used only by Phi-4-multimodal. Its mel front end mirrors numpy's float64 FFT, so
     moving it to f32 is a parity change; leave it unless parity is re-checked.
2. **aws-lc-sys: ~30 s CPU** (a C build that needs cmake), compiled alongside `ring`, which rustls already builds.
   - reqwest 0.13's `rustls` feature selects the aws-lc provider. Everything else (hf-hub's ureq, the Prometheus
     exporter, reqwest 0.12) uses ring.
   - Fix: reqwest's `rustls-no-provider` feature with the ring provider supplied at client construction. There are 7
     construction sites, so this wants one shared builder.
3. **openai-harmony pulls in `image` with its default formats, AVIF included: ~55 s CPU.** The AVIF part is ravif
   37 s, rav1e 10 s, av1-grain 3 s and helpers. harmony never uses `image`.
   - harmony also keeps reqwest 0.12 (24.5 s) alive next to our 0.13. hf-hub 0.4 uses 0.12 too; hf-hub 1.0 moves to
     0.13.
   - Fix options:
     - an upstream PR to openai/harmony dropping the unused `image` dependency;
     - a vendored or patched copy (5k lines, and it downloads the tiktoken vocab through reqwest);
     - hf-hub 1.0 plus a harmony on reqwest 0.13, which leaves one reqwest.
   - This is the largest item, and the most work.
4. **variantly: 9 s CPU.** A proc macro used once (`#[derive(Variantly)]` on one enum in
   `pipeline/model_config.rs`). It is the only thing that pulls in `syn` 1, darling 0.11 and uuid 0.8. Replacing it
   with handwritten accessors drops all three.
5. **html5ever built twice: ~5 s CPU.** scraper 0.25 uses 0.36 and html2text 0.16 uses 0.37. scraper 0.26 and
   html2text 0.17 both move to 0.39, which unifies html5ever, markup5ever and tendril.
6. **Small redundancies (seconds or less, but free):**
   - `tokio-test` is declared by inference-mcp and never used.
   - `tqdm` shows progress in the X-LoRA loaders and nn, and pulls in crossterm 0.25, while `indicatif` is already
     there for the same job.
   - `strum` 0.27 alongside llguidance's 0.28.
   - `directories` (one `ProjectDirs` call in the CLI) alongside `dirs`.
   - `futures-util` alongside `futures`, which re-exports it.
   - `indicatif` 0.18 alongside hf-hub's 0.17 (goes away with hf-hub 1.0).
7. **Kept, and reasonable despite the cost:**
   - llguidance (39 s): constrained decoding.
   - rust-mcp-schema (36 s): MCP protocol types, already limited to the `latest` schema.
   - symphonia (21 s): audio decoding, with only the codecs we accept.
   - metrics-exporter-prometheus (19 s).
   - scraper (18 s): web-search result extraction.
   - hf-hub (18 s).
   - serde-saphyr (16 s): the device-topology YAML format users write.
   - toml (9 s): CLI config.
   - bm25 (9 s): search and tool reranking.
   - mimalloc, rustyline, utoipa-swagger-ui.
   - tokenizers, minijinja, image, candle, tokio, axum and serde are shared foundations; nothing cheaper does their
     job.

### Inventory

Exclusive cost is CPU seconds that dropping the crate would save, from the Run 39 cold build. Files counts the source
files referencing it, and "Used by" lists the declaring crates without the `inference-` prefix. A dash marks a crate
this Linux CUDA build doesn't compile (Metal, macOS or optional targets).

| Crate | Version | Used by | Files | Excl. s | Role and verdict |
|---|---|---|---|---|---|
| ahash | 0.8.12 | core | 6 | 0 | Fast hasher for hot maps. Keep. |
| akin | 0.4.0 | nn | 1 | 0.3 | Macro repetition for GGUF metadata getters. Keep (tiny). |
| anyhow | 1.0.100 | 19 crates | 324 | 0 | Error type. Keep. |
| as-any | 0.3.2 | core | 1 | 0.1 | `AsAny` for loader downcasts. Tiny; could be a local trait. |
| async-trait | 0.1.89 | core, mcp | 11 | 0.9 | Async trait methods (Pipeline). Keep. |
| axum | 0.8.8 | cli, server-core, webui | 26 | 0 | HTTP server. Keep. |
| base64 | 0.22.1 | api, code-exec, core, ffi, server-core | 14 | 0 | Media payloads. Keep. |
| bm25 | 2.3.2 | core | 2 | 8.8 | Search and tool reranking. Keep. |
| bytemuck | 1.24.0 | quant | 1 | 0 | Pod casts for weights. Keep. |
| byteorder | 1.5.0 | quant | 2 | 0 | GGUF and imatrix I/O. Keep. |
| candle-core | 0.11.0 | 19 crates | 534 | 0 | Tensor backend. Keep. |
| candle-flash-attn-v3 | - | nn | 1 | - | Optional SM90 attention. Keep. |
| candle-metal-kernels | - | core, nn, paged-attn, quant, metal-compile | 11 | - | Metal backend. Keep. |
| candle-nn | 0.11.0 | 15 crates | 145 | 4.9 | Layers. Keep. |
| chrono | 0.4.43 | api, core, webui | 4 | 0 | Timestamps. Keep. |
| clap | 4.5.54 | cli, examples, layout | 16 | 0 | CLI parsing. Keep. |
| clap_complete | 4.5.65 | cli | 2 | 4.2 | Shell completions. Keep. |
| comfy-table | 7.2.2 | cli | 3 | 5.6 | CLI tables. Keep. |
| crossterm | 0.28.1 | cli | 1 | 4.8 | Terminal control for the REPL. Keep. |
| csv | 1.4.0 | core, nn | 3 | 1.6 | AnyMoE and MatFormer inputs. Keep. |
| ctrlc | 3.5.1 | cli | 1 | 1.9 | Interrupt handling. Keep. |
| cudaforge | 0.1.6 | flash-attn, nn, paged-attn, quant | 4 | 0 | CUDA kernel builds (build dependency). Keep. |
| cutile | - | quant | 27 | - | cuTile kernels (optional). Keep. |
| darling | 0.23.0 | macros | 1 | 0 | Our proc macros. Keep. |
| data-url | 0.3.2 | api | 1 | 0.8 | `data:` media URLs. Keep. |
| derive-new | 0.7.0 | nn | 1 | 0.4 | Two `#[derive(new)]` in varbuilder utils. Tiny; could be constructors. |
| derive_more | 2.1.1 | core | 2 | 0 | Derives. Keep. |
| directories | 6.0.0 | cli | 1 | 0.4 | One `ProjectDirs` for REPL history. Redundant with `dirs`. |
| dirs | 6.0.0 | cli, core | 2 | 0 | Home and cache dirs. Keep. |
| dispatch2 | - | paged-attn, quant | 2 | - | macOS GCD (Metal). Keep. |
| either | 1.15.0 | 8 crates | 27 | 0 | Keep. |
| fancy-regex | 0.14.0 | quant | 1 | 4.3 | Lookaround in LoRA target patterns. Keep (0.13 also comes via harmony). |
| float8 | 0.7.0 | nn, paged-attn, quant | 40 | 0 | FP8 types. Keep. |
| futures | 0.3.31 | 6 crates | 22 | 0 | Keep. |
| futures-util | 0.3.31 | mcp | 1 | 0 | Redundant: `futures` re-exports it. |
| fuzzy-matcher | 0.3.7 | cli | 1 | 1.2 | UQFF name suggestions. Keep. |
| galil-seiferas | 0.1.5 | core | 1 | 0.3 | Stop-string search. Tiny; `memchr::memmem` (already built) does the same. |
| gemm | 0.19.0 | layout | 1 | 0 | CPU GEMM for layout detection. Keep (shared with candle). |
| half | 2.7.1 | flash-attn, nn, paged-attn, quant | 87 | 0 | f16/bf16. Keep. |
| hf-hub | 0.4.3 | core | 17 | 18.1 | Hub downloads. Keep; 1.0 would unify reqwest and indicatif. |
| hound | 3.5.1 | audio | 1 | 0.4 | WAV I/O. Keep. |
| html2text | 0.16.6 | core | 1 | 6.8 | Web page text for search. Keep; 0.17 unifies html5ever. |
| http, http-body, http-body-util | 1.x | mcp, server-core | 3 | 0 | Keep. |
| image | 0.25.9 | 10 crates | 63 | 0 | Image I/O. Keep (harmony enables its defaults, see finding 3). |
| include_dir | 0.7.4 | webui | 1 | 0.5 | Embeds the UI bundle. Keep. |
| indexmap | 2.13.0 | 7 crates | 23 | 0 | Keep. |
| indicatif | 0.18.3 | 8 crates | 10 | 5.1 | Progress bars. Keep. |
| interprocess | 2.2.3 | core | 2 | 1.1 | Distributed IPC. Keep. |
| itertools | 0.14.0 | api, core, nn | 15 | 0 | Keep. |
| itoa | 1.0.18 | sandbox | 1 | 0 | Allocation-free formatting after fork. Keep (reason is real). |
| landlock, seccompiler, nix | - | sandbox | 1-4 | 0.7-3.1 | Linux sandbox. Keep. |
| libc | 0.2.186 | code-exec, quant, sandbox | 8 | 0 | Keep. |
| llguidance, toktrie | 1.8.0 | core | 27 | 38.6 | Constrained decoding. Keep. |
| memmap2 | 0.9.9 | quant | 2 | 0 | Weight mmap. Keep. |
| metrics | 0.24.6 | core, models-qwen, nn, server-core | 18 | 0 | Keep. |
| metrics-exporter-prometheus | 0.18.3 | server-core | 1 | 18.6 | `/metrics`. Keep. |
| mimalloc | 0.1.52 | cli | 1 | 7.5 | Allocator for the binary. Keep. |
| mime_guess | 2.0.5 | core, webui | 2 | 0 | Keep. |
| minijinja, minijinja-contrib | 2.14.0 | core | 1 | 3.5 | Chat templates (pycompat). Keep. |
| num-traits | 0.2.19 | core, nn | 3 | 0 | Keep. |
| num_cpus | 1.17.0 | layout | 1 | 0 | `get_physical` (std has no physical count). Keep. |
| objc, objc2-metal | - | core, nn, paged-attn, quant, metal-compile | 7 | - | Metal. Keep. |
| openai-harmony | 0.0.8 | core | 1 | 13.6 | GPT-OSS Harmony format. Needed; see finding 3 for what it drags in. |
| ordered-float | 5.1.0 | core | 1 | 0.5 | `NotNan` in the Llama 4 processor. Tiny. |
| paste | 1.0.15 | quant | 4 | 0 | Macro identifiers. Keep. |
| proc-macro2, quote, syn | 1.x/2.x | macros | 1 | 0 | Our proc macros. Keep. |
| rand, rand_distr, rand_isaac | 0.9/0.5/0.4 | 9 crates | 39 | 0.3 | Sampling. Keep. |
| rayon | 1.11.0 | layout, nn, quant | 20 | 0 | Keep. |
| regex, regex-automata | 1.12/0.4 | cli, core, nn, quant | 28 | 0 | Keep. |
| reqwest | 0.13.1 | core, examples, mcp | 12 | 24.5 | HTTP client. Keep; see finding 2 (aws-lc) and 3 (0.12 twin). |
| rubato | 0.16.2 | core | 4 | 1.1 | Resampling (sinc only). Keep, without default features (finding 1). |
| rust-mcp-schema | 0.9.5 | mcp | 2 | 35.8 | MCP types, latest schema only. Keep. |
| rustfft | 6.4.1 | audio | 1 | 0 | FFT. Keep, without SIMD features (finding 1). |
| rustyline | 15.0.0 | cli | 1 | 8.3 | REPL line editing. Keep. |
| safetensors | 0.8.0 | core, models-gemma, models-other, quant | 24 | 0 | Keep. |
| schemars | 1.2.0 | examples, inference | 7 | 0 | SDK tool schemas. Keep. |
| scraper | 0.25.0 | core | 1 | 18.2 | Search result extraction. Keep; 0.26 unifies html5ever. |
| serde, serde_json | 1.0 | ~20 crates | 368 | 0 | Keep. |
| serde-big-array | 0.5.1 | core | 1 | 0.1 | One fixed array in distributed IPC. Tiny. |
| serde-saphyr | 0.0.16 | nn | 1 | 15.8 | Device topology YAML. Keep (user-facing format; a 0.0.x crate). |
| serde_plain | 1.0.2 | models-diffusion | 1 | 0.3 | T5 activation name parse. Tiny. |
| sha2 | 0.10.9 | core, nn | 2 | 0 | Adapter hashing. Keep. |
| strum | 0.27.2 | core, nn | 13 | 2.2 | Enum derives. Keep; bump to 0.28 to unify with llguidance. |
| symphonia | 0.5.5 | audio | 1 | 21.4 | Audio decoding, chosen codecs only. Keep. |
| sysinfo | 0.36.1 | core, nn, quant | 3 | 6.9 | Memory queries for mapping. Keep. |
| tempfile | 3.25.0 | 13 crates | 44 | 0 | Keep. |
| thiserror | 2.0.18 | 6 crates | 13 | 0 | Keep. |
| tokenizers | 0.23.2 | core, models-diffusion, nn | 45 | 0 | Keep. |
| tokio, tokio-rayon | 1.49/2.1 | 12 crates | 172 | 0.1 | Keep. |
| tokio-test | 0.4.5 | mcp | 0 | 1.4 | Unused. Remove. |
| tokio-tungstenite | 0.28.0 | mcp | 1 | 7.8 | MCP WebSocket transport. Keep. |
| toml | 0.9.11 | cli | 2 | 9.2 | CLI config. Keep. |
| tower, tower-http | 0.5/0.6 | server-core, webui | 4 | 0 | Keep. |
| tqdm | 0.8.0 | core, nn, 4 model crates | 11 | 2.8 | Progress in X-LoRA and nn loops; brings crossterm 0.25. Redundant with `indicatif`. |
| tracing, tracing-subscriber | 0.1/0.3 | ~18 crates | 151 | 0 | Keep. |
| url, urlencoding | 2.5/2.1 | api, core, server-core | 6 | 0.3 | `urlencoding` is one call that `url`'s form encoder covers. Tiny. |
| utoipa, utoipa-swagger-ui | 5.4/9.0 | 7 crates | 56 | 7.0 | OpenAPI and Swagger UI. Keep. |
| uuid | 1.19.0 | 7 crates | 22 | 0.6 | Keep. |
| variantly | 0.4.0 | core | 1 | 9.4 | One derive; sole source of syn 1, darling 0.11, uuid 0.8. Replace (finding 4). |
| walkdir | 2.5.0 | cli, core, examples | 4 | 0 | Keep. |
| yoke | 0.8.1 | quant | 1 | 0 | Borrowed GGUF views. Keep. |
| zip | 8.6.0 | api | 1 | 0 | File uploads. Keep. |

- Implication: nothing here moves the idle cold wall time, which stays core's lib test. The CPU items worth doing:
  - findings 1, 4, 5 and 6 are self-contained edits;
  - finding 2 is a small refactor, one shared reqwest client builder;
  - finding 3 needs either an upstream change (a remote action) or a vendored crate.

## Run 2 - 2026-09-28 (night)

- Done:
  - Findings 1 and 4, and from 5 and 6: tokio-test, tqdm, strum, futures-util, and the scraper and html2text bumps.
  - Result: build-time Run 42. inference-audio IR is -79%, and syn 1, darling 0.11, uuid 0.8, tqdm, crossterm 0.25
    and realfft are no longer built.
- Declined:
  - Finding 2 (reqwest on ring). reqwest 0.13's `rustls-no-provider` panics at client build unless a process-wide
    provider is installed. So every client path, including the examples' `reqwest::get` calls, would need an install
    step, for ~30 s of CPU that is off the critical path.
  - `directories` -> `dirs`: `ProjectDirs` puts the REPL history under `com.inference.rs` on macOS and under
    `...\config` on Windows, so switching would move existing history files.
- Open: finding 3 (openai-harmony's unused `image` default formats and its reqwest 0.12). It needs an upstream
  change or a vendored copy.
