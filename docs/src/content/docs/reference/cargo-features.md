---
title: Cargo features
description: Feature flags for the inference workspace crates.
---

inference.rs uses Cargo features to gate platform-specific and optional functionality.

## Accelerator features

| Feature | Crates | Purpose |
|---|---|---|
| `cuda` | `inference-cli`, `inference`, `inference-core`, `inference-server-core`, `inference-api`, `inference-ffi`, `inference-selection` | NVIDIA CUDA acceleration, including flash attention (Turing and newer) and [paged attention](/guides/perf/paged-attention/). |
| `cudnn` | as above | Routes candle's `conv1d`/`conv2d` through cuDNN. Not recommended: candle's integration plans every call from scratch, which measured 1.7-22x slower than the default path in BF16 and F16, and cuDNN adds ~1.15 GB of runtime libraries. |
| `flash-attn-v3` | `inference-cli`, `inference-core`, `inference-server-core`, `inference-api`, `inference-ffi`, `inference-selection` | Flash attention v3 (Hopper, requires `cuda`). Not exposed by the top-level `inference` crate. |
| `cutile` | `inference-cli`, `inference`, `inference-core`, `inference-ffi`, `inference-api`, `inference-selection` | cuTile acceleration for quantized linears, MoE, and routed LoRA. Enables `cuda`. Requires CUDA >= 13.2 on Ampere/Ada and Blackwell+, CUDA >= 13.3 on Hopper, and a compatible `tileiras` installation. [NVFP4](/reference/quantization-types/#nvfp4) requires Blackwell and CUDA >= 13.3. See [cuTile setup](/developer/moe-backends/). |
| `metal` | as above | Apple Silicon GPU support via Metal. |
| `accelerate` | as above | Apple Accelerate framework for CPU math. |
| `mkl` | as above | Intel MKL for CPU math. |
| `nccl` | `inference-cli`, `inference`, `inference-core`, `inference-server-core`, `inference-api`, `inference-ffi` | NCCL single-machine CUDA multi-GPU support. Requires the NCCL runtime library at build and runtime. |

Typical combinations:

- NVIDIA Hopper: `cuda flash-attn-v3` (add `cutile` with CUDA >= 13.3)
- NVIDIA Ampere or Ada: `cuda` (add `cutile` with CUDA >= 13.2)
- NVIDIA Blackwell with CUDA >= 13.2 and a compatible `tileiras`: `cuda cutile`
- Apple Silicon: `metal`
- Intel CPU with MKL: `mkl`

For Linux CUDA multi-GPU, add `nccl` when NCCL is installed. The Linux installer and CUDA wheel builder add it automatically when they detect `libnccl`.

## Functional features

| Feature | Crates | Purpose |
|---|---|---|
| `code-execution` | `inference-cli`, `inference`, `inference-core`, `inference-server-core`, `inference-api`, `inference-ffi` | Python code execution tool. In the `inference-cli` and `inference-ffi` defaults. |
| `ring` | as above | Multi-machine ring distributed inference. |
| `swagger-ui` | `inference-cli`, `inference-server-core` | Mounts Swagger UI on the HTTP server. On by default in both. |

## Model families

Each model family is its own crate (`inference-models-{gemma,llama,other,phi,qwen}`), compiled only when its feature is on. Speech (Dia) and image generation (FLUX) models are always built.

| Feature | Crates | Purpose |
|---|---|---|
| `all-models` | `inference-cli`, `inference`, `inference-core`, `inference-server-core`, `inference-api`, `inference-ffi` | Every model family. On by default in each. |
| `models-gemma`, `models-llama`, `models-other`, `models-phi`, `models-qwen` | as above | One family. With `--no-default-features`, list the families to build; a smaller set builds faster and, for `inference-ffi`, ships a smaller library. |

## Enabling features

From the repository with `cargo install`:

```bash
cargo install --git https://github.com/christopherthompson81/inference.rs inference-cli --features "cuda nccl"
```

From a source checkout:

```bash
cargo install --path crates/inference-cli --features "cuda nccl"
```

In a consumer crate depending on `inference`:

```toml
[dependencies]
inference = { git = "https://github.com/christopherthompson81/inference.rs", features = ["cuda", "nccl"] }
```

## Default features

`inference-cli` defaults to `code-execution`, `swagger-ui` and `all-models`; `inference-server-core` to `swagger-ui` and `all-models`; `inference`, `inference-core` and `inference-api` to `all-models`; `inference-ffi` to `code-execution` and `all-models`. To exclude defaults, use `--no-default-features` (then name the model families you want).

No crate enables an accelerator feature by default. Opt in to the accelerator matching your hardware.

## Feature verification

`inference doctor` prints a `Build features:` line listing the compiled-in accelerator features (`cuda`, `metal`, `cudnn`, `flash-attn-v3`, `cutile`, `accelerate`, `mkl`). Other features such as `nccl`, `ring`, `code-execution`, and `swagger-ui` are not shown on that line.
