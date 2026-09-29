# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ABSOLUTE RULE: nothing leaves this machine without explicit approval

Never push, publish, or send ANY code or content off this machine unless Eric explicitly approves that specific action in that specific instance. No `git push`, no PR creation, no PR merges, no releases, no package publishes, no remote API writes. EVER. Prior approval of a plan does not count as approval to push; ask immediately before the push itself. Commits to the local working tree are fine; everything remote requires a fresh, explicit yes.

## Project Overview

inference.rs is a blazing-fast LLM inference engine written in Rust. It supports text, multimodal, image generation, and speech models with Rust and Python SDKs, plus OpenAI HTTP and MCP APIs.

## Essential Commands

### Building
```bash
# Basic release build
cargo build --release

# With CUDA support (Linux)
cargo build --release --features "cuda flash-attn cudnn"

# With Metal support (macOS)
cargo build --release --features metal

# Install CLI binary
cargo install --path crates/inference-cli --features <features>
```

### Testing & Quality
```bash
# Run core tests
cargo test -p inference-core -p inference-protocol -p inference-nn -p inference-models-llama -p inference-models-qwen -p inference-models-gemma -p inference-models-phi -p inference-models-other -p inference-models-speech -p inference-models-diffusion -p inference-quant -p inference-vision --features inference-protocol/openai

# Format code (uses rustfmt, ruff, clang-format)
make fmt

# Check formatting
cargo fmt --all -- --check

# Run clippy
cargo clippy --workspace --tests --examples -- -D warnings

# Canonical local checks (default: --lint --tests). Use these rather than ad-hoc cargo invocations: each mode always
# builds the same package/feature set, so artifacts are reused instead of rebuilt per combination.
# --docs checks the docs of the crates that differ from master (rustdoc is never incremental); --docs-all checks all.
# Neither renders HTML (`cargo doc` does): rendering merges into target/doc under a lock that serialized the runs.
# --bindings builds libinference_ffi and runs the C# (needs the .NET SDK) and Python binding tests.
scripts/local_ci.sh [--lint] [--tests] [--cuda] [--models] [--slim] [--docs|--docs-all] [--bindings]

# Same, then delete target/debug artifacts the selected modes don't use (including on-request example builds).
scripts/local_ci.sh --lint --tests --cuda --sweep
```

### Running Models
```bash
# Run interactive mode (model type auto-detected)
inference run -m <model_id>

# Run with GGUF quantized model
inference run --format gguf -m <repo> -f <file>

# Run server
inference serve -p 1234 -m <model_id>

# Run server (built-in web UI is on by default at /ui; pass --no-ui to disable)
inference serve -m <model_id>

# Run benchmarks
inference bench -m <model_id>
```

## Models

When integrating a new model, make sure it respects all of the varbuilder `.pp` calls. In Candle, a VarBuilder maintains an internal path vector that acts like a “current working directory” for model weights; every call to pp("sub") (alias for push_prefix) clones the builder and appends sub, so successive calls accumulate a dotted prefix such as transformer.h.0 while leaving the original builder untouched . When you eventually call get(...), Candle joins that prefix with the tensor name (prefix + "." + name) and looks it up in the checkpoint backend, producing keys that exactly match the dot-separated names emitted by PyTorch’s state_dict/named_parameters, which means PyTorch-trained weights can be loaded without any renaming  ￼. This lets you recreate the PyTorch module tree in Rust by “walking” it: e.g. vb.pp("word_embeddings") grabs word_embeddings.*, while a chain like vb.pp("encoder").pp("layers").pp(i.to_string()) targets keys such as encoder.layers.0.*, exactly as shown in community tutorials porting Transformers models to Candle  ￼. As one maintainer put it, the prefix system lets you “cd” around the parameter hierarchy, giving a lightweight namespace mechanism that keeps Candle fully compatible with PyTorch naming conventions while remaining ergonomic to use.

You should also look for a model.safetensors.index.json file for the model at hand to verify correct structure.

## Architecture Overview

### Workspace Structure
- `crates/inference-core/` - Core inference engine, model implementations, pipelines
- `crates/inference-models-{llama,qwen,gemma,phi,other}/` - Model families (one crate per family, built on `inference-nn`): text models plus the vision models built on their text stacks, each behind an `inference-core` feature (`models-llama`, ...; all on by default). A multimodal model's input processor (a `MultimodalInputsProcessor` over `inference_nn::media_inputs`) lives beside it; its `Processor` (chat template actions) stays in core. `--slim` checks core with each family alone
- `crates/inference-models-{speech,diffusion}/` - Speech (Dia) and image generation (FLUX) models, always built; their loaders, `SpeechLoaderType`/`DiffusionLoaderType` and request processors stay in core
- `crates/inference-nn/` - Model-facing building blocks: layers, attention and its metadata, KV/paged caches, GDN, MoE, device mapping, the loader traits with their sizing and placement helpers (`loaders`), and the CUDA/Metal kernels behind them
- `crates/inference-cli/` - Unified CLI binary (commands: run, serve, bench, from-config)
- `crates/inference-api/` - The engine surface with no HTTP: OpenAI request/response types, request parsing and dispatch, chat as an engine operation, the server/engine builder. The HTTP server builds on it and the C ABI exposes it; add engine features here, not in the server
- `crates/inference-server-core/` - HTTP server routing, OpenAI API implementation
- `crates/inference-webui/` - The chat web UI `serve` mounts at `/ui`: its HTTP handlers over the engine, the Svelte source (`webui/`) and the built bundle it embeds (`static/`, rebuilt with `npm run build` in `webui/`)
- `crates/inference/` - Rust SDK (high-level crate)
- `crates/inference-vision/` - Image processing utilities
- `crates/inference-quant/` - Quantization implementations (ISQ, GGUF, GPTQ, etc.)
- `crates/inference-paged-attn/` - PagedAttention implementation
- `crates/inference-audio/` - Audio processing
- `crates/inference-mcp/` - Model Context Protocol client
- `crates/inference-protocol/` - The wire protocol with no candle dependency (so it compiles alongside the kernel builds): request options, response bodies, tool types with their call parsers and grammars, reasoning parsers, files. Core re-exports it; the engine-internal `Request`/`Response` channel types stay in core
- `crates/inference-layout/` - Document layout detection (PP-DocLayoutV3) with custom CPU/CUDA kernels
- `crates/inference-ffi/` - C ABI (`libinference_ffi`, header `include/inference.h`) for bindings in other languages
- `bindings/csharp/` - .NET bindings over the C ABI (`InferenceRs.slnx`); a new ABI entry point needs its binding, which the coverage test enforces
- `bindings/python/` - the Python SDK: a pure-Python ctypes package over the C ABI (`inference_rs`); its coverage test enforces the same, and `scripts/release/build_wheels.py` builds wheels that bundle the library. Its typed classes (`inference_rs/types.py`) are generated from `docs/openapi.json`: after regenerating that, run `python3 bindings/python/scripts/generate_types.py`
- Kernel sources live in `<crate>/kernels/{cuda,metal}/` (inference-layout compiles inline sources with NVRTC); each kernel crate's `third_party/README.md` records upstream provenance and license.

### Key Design Patterns

1. **Pipeline Architecture**: All models implement the `Pipeline` trait in `crates/inference-core/src/pipeline/mod.rs`. Different model types (Plain, GGUF, GGML, Multimodal) have their own pipeline implementations.

2. **Model Loading**: Models are loaded through `Loader` traits that handle different formats and quantizations. See `crates/inference-core/src/pipeline/loaders/mod.rs` (the `Loader` trait) and `pipeline/loading.rs`.

3. **Request Handling**: The server uses message passing with `InferenceRs` struct managing a background thread pool. Requests flow through `crates/inference-core/src/engine/mod.rs`.

4. **Device Management**: Automatic and manual device mapping for multi-GPU setups handled in `crates/inference-nn/src/device_map/`.

### Adding New Features

When adding new model architectures:
1. Implement the model in its family crate, `crates/inference-models-<family>/`, with its input processor (`<model>/inputs_processor.rs`); its `Processor` goes in core's `crates/inference-core/src/vision_models/<model>/processor.rs`, beside a module that re-exports the model
2. Add its loader beside the model in `crates/inference-models-<family>/src/loaders/` and list it in that crate's `inference_nn::boxed_loaders!`, so its vtable and trait defaults compile there. A multimodal loader's `MultimodalProcessorFactory` impl stays in core's `vision_models/<model>/processor.rs`.
3. Add one row to `normal_loader_types!` (or `multimodal_loader_types!`) in core's `pipeline/loaders/{normal,multimodal}_loaders/mod.rs`. The row gives
   the CLI name, the HF class, the `model_type` (text only) and the loader, and the enum variant, parsing, display, HF detection and loader
   dispatch are all generated from it. Embedding loaders are listed by hand in `EmbeddingLoaderType` (`embedding_loaders/mod.rs`).
4. Add the GGUF bindings in `crates/inference-core/src/gguf/` if the model loads from GGUF

When adding new quantization methods:
1. Implement in `crates/inference-quant/src/`
2. Add to quantization loading logic in pipelines
3. Update documentation in `docs/src/content/docs/reference/quantization-types.md`

### Important Files to Know

- `crates/inference-core/src/engine/mod.rs` - Main engine orchestration
- `crates/inference-core/src/pipeline/mod.rs` - Pipeline trait and common logic
- `crates/inference-server-core/src/route_registry.rs` - HTTP API endpoints (routes wired in `inference_server_router_builder.rs`)
- `bindings/python/inference_rs/_typed.py` - Python SDK entry point (`Engine`)
- `examples/rust/` - Rust SDK examples (`inference-examples`, not a default member: build with `-p inference-examples --example <name>`)

### Pull Requests

Never include a "Test plan" section in PR descriptions.

### Code Style (Extremely important & convention for this codebase)

**Comments.** Default to none. Only add when the *why* isn't obvious from the code: hidden constraints, invariants, surprising edge cases, references to a spec/HF source. Never paraphrase what the next line does, never restate the function name, never narrate steps.

- Multi-line comments are discouraged in code, and only really allowed in documentation or where they are the best way to communicate information.
- Code comments should be one line each, up to ~120 cols. No multi-paragraph `///` blocks, no bulleted lists in doc comments, no `// === Section ===` or `// ── Section ──` banners.
- Tone for inline code comments should be terse, casual, and never explaining what the code directly below does.
- Only include code comments if they add new information, and never just for the sake of it.

- Unless otherwise instructed, use ASCII only. No em-dashes (`—`), en-dashes (`–`), ellipses (`…`), smart quotes, or box-drawing characters. Do not use `--`. It's ok to use `...`, `"`, `'` when appropriate.
- Don't reference the current task / PR / fix / commit in comments — that belongs in the PR description and rots as the codebase evolves.
- Trailing inline annotations like `// already sent above` are fine when terse.

**Magic values.** Hoist durations, sizes, sentinels, and other constants to named `const`s at the top of the file. A sentinel value that crosses module boundaries (e.g. one place sets `Some(0)`, another checks for it) must be a `pub const`, not a literal both sides happen to share.

**Function shape.** When a function passes 6+ args, prefer wrapping the invariants in a small context struct (e.g. `DispatchCtx<'a>`). Don't add error handling, fallbacks, or validation for scenarios that can't actually occur — trust internal code and framework guarantees. Don't add backwards-compatibility shims unless explicitly asked.

### Testing Approach

You should *always* run `cargo check`/`cargo c` before returning to make sure code compiles. If code does not compile, only make edits.

Avoid returning TODOs.

- Unit tests are colocated with source files
- Integration tests in `tests/` directories
- `scripts/local_ci.sh --tests` (CPU) and `--cuda` (GPU) run the whole workspace suite. Narrow with a test-name filter only for quick iteration, and keep the same features.
- In dev builds on Linux the always-built CUDA kernel sets are shared libraries under `target/debug/cuda-kernels` (one copy for every variant and test binary, loaded by absolute SONAME), so a dev binary only runs from this checkout. Release builds link static archives.
- Put build env (CC/CXX/NVCC) and model paths (INFERENCE_TEST_*) in `~/.cargo/config.toml` `[env]`, not on the command line: build scripts track them, and changing one rebuilds the dependency tree.
- Tests run under cargo-nextest (one process per test; see `.config/nextest.toml` for the GPU group sized by VRAM). It is required for `--features cuda`: plain `cargo test` shares one CUDA context across a binary's tests, so the memory-pool tests interfere. Install: `curl -LsSf https://get.nexte.st/latest/linux | tar zxf - -C ~/.cargo/bin`.
- Real-checkpoint parity tests are integration tests: `--models` runs them (CPU) and `--cuda` keeps one GPU parity check. Engine behavior is tested on tiny random-weight checkpoints built at test time (see `crates/inference/tests/integration/paddleocr_vl_tiny.rs`).
- Each crate's integration tests are modules of one binary, `tests/integration/main.rs`, since a binary per file re-monomorphizes and links the whole stack. Add a test file there as a `mod`; nextest selects it by `package(<crate>) & test(/^<module>::/)`.
- GPU tests use `skip_without_cuda!()` instead of `#[ignore]`, so `--features cuda` runs them wherever a device exists. Keep `#[ignore]` for hardware this suite can't assume (SM90, SM121, cuTile), benchmarks, and tests that write files.
- A check worth running by hand is a test worth committing.
- Python tests run from the checkout against `target/debug`; `scripts/local_ci.sh --bindings` builds the library and runs them

### Common Pitfalls

1. **Feature Flags**: Many features are gated behind Cargo features. Always check what features are needed for your use case.
2. **Device Indices**: CUDA device selection uses 0-based indexing
3. **Chat Templates**: Models may need specific chat templates - check `chat_templates/` directory
4. **Quantization**: Different quantization methods have different hardware requirements
5. **Never use `Tensor::{from_vec,arange}` in hot loops**: `Tensor::{from_vec,arange}` with a GPU device causes a CPU-to-GPU sync. If you need a small tensor on GPU during forward, either precompute it at model init or start of forward pass.

### Vision/Audio Model Pitfalls

6. **Vision encoder attention must be bidirectional (non-causal)**:  `Sdpa.run_attention` with `flash_params: None` defaults to `causal = seq_len > 1` on the CUDA flash-attn path, which silently breaks vision/audio encoders. Always pass `FlashParams { causal: false, cumulative_seqlens_q: HashMap::new(), cumulative_seqlens_k: HashMap::new(), max_q: 0, max_k: 0 }` with `Some(&flash_params)` for any encoder that needs bidirectional attention. The empty `cumulative_seqlens` cause the flash backend to use the non-varlen kernel path, avoiding any tensor allocation in the forward pass.

7. **`torch.bucketize(right=True)` requires `Ok(i) => i + 1`**: Rust's `binary_search_by` returns `Ok(i)` at the found position (bisect_left semantics). For `right=True` (bisect_right), you must use `Ok(i) => i + 1` to insert after equal elements. `Err(i) => i` is correct for both.

8. **Mistral `consolidated.safetensors` stores Q/K weights with interleaved head dimensions**: When loading from Mistral-native `consolidated.safetensors` (as opposed to HF-converted `model.safetensors`), the Q and K projection weights use an interleaved layout within each head: `[x0, x_{d/2}, x1, x_{d/2+1}, ...]` instead of the sequential HF layout `[x0, x1, ..., x_{d/2-1}, x_{d/2}, ...]`. This means you must use `is_gptx=false` (GPT-J/adjacent-pair style) for `RotaryEmbedding`, NOT `is_gptx=true` (GPT-NeoX/half-split style). Using the wrong RoPE style produces completely wrong attention outputs (cosine similarity ~0.02 with reference). To diagnose: compare a Q or K weight tensor between `consolidated.safetensors` and `model.safetensors` — if they differ (cosine ~0.02), apply the un-interleave: `reshape(n_heads, head_dim/2, 2, dim).permute(0,2,1,3)` and verify cosine ~1.0.

9. **Causal Conv1d padding formula**: For causal convolution (left-pad only, no right-pad), the correct left padding is `effective_kernel_size - stride`, NOT `(kernel_size - 1) * dilation` (which is the total padding for non-causal). For example, with kernel_size=3, stride=2, dilation=1: left_pad = 3 - 2 = 1, not 2. Verify against the HF model's `VoxtralRealtimeCausalConv1d` or equivalent source.
