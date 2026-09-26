---
title: Cargo features
description: Feature flags for the inference workspace crates.
---

inference.rs uses Cargo features to gate platform-specific and optional functionality.

## Accelerator features

| Feature | Crates | Purpose |
|---|---|---|
| `cuda` | `inference-cli`, `inference`, `inference-core`, `inference-server-core` | NVIDIA CUDA acceleration, including [paged attention](/guides/perf/paged-attention/). |
| `cudnn` | as above | cuDNN-accelerated kernels. |
| `flash-attn` | as above | Flash attention v2 (Ampere+, requires `cuda`). |
| `flash-attn-v3` | `inference-cli`, `inference-core`, `inference-server-core` | Flash attention v3 (Hopper, requires `cuda`). Not exposed by the top-level `inference` crate. |
| `cutile` | `inference-cli`, `inference`, `inference-core`, `inference-pyo3` | cuTile acceleration for quantized linears, MoE, and routed LoRA. Enables `cuda`. Requires CUDA >= 13.2 on Ampere/Ada and Blackwell+, CUDA >= 13.3 on Hopper, and a compatible `tileiras` installation. [NVFP4](/reference/quantization-types/#nvfp4) requires Blackwell and CUDA >= 13.3. See [cuTile setup](/developer/moe-backends/). |
| `metal` | as above | Apple Silicon GPU support via Metal. |
| `accelerate` | as above | Apple Accelerate framework for CPU math. |
| `mkl` | as above | Intel MKL for CPU math. |
| `nccl` | `inference-cli`, `inference`, `inference-core`, `inference-server-core` | NCCL single-machine CUDA multi-GPU support. Requires the NCCL runtime library at build and runtime. |

Typical combinations:

- NVIDIA Hopper: `cuda flash-attn flash-attn-v3 cudnn` (add `cutile` with CUDA >= 13.3)
- NVIDIA Ampere or Ada: `cuda flash-attn cudnn` (add `cutile` with CUDA >= 13.2)
- NVIDIA Blackwell with CUDA >= 13.2 and a compatible `tileiras`: `cuda flash-attn cudnn cutile`
- NVIDIA older: `cuda cudnn`
- Apple Silicon: `metal`
- Intel CPU with MKL: `mkl`

For Linux CUDA multi-GPU, add `nccl` when NCCL is installed. The Linux installer and CUDA wheel builder add it automatically when they detect `libnccl`.

## Functional features

| Feature | Crates | Purpose |
|---|---|---|
| `code-execution` | `inference-cli`, `inference`, `inference-core`, `inference-server-core` | Python code execution tool. In `inference-cli` defaults. |
| `ring` | as above | Multi-machine ring distributed inference. |
| `swagger-ui` | `inference-server-core` | Mounts Swagger UI on the HTTP server. On by default in `inference-server-core`. |

## Enabling features

From `cargo install`:

```bash
cargo install inference-cli --features "cuda nccl flash-attn cudnn"
```

From a source checkout:

```bash
cargo install --path crates/inference-cli --features "cuda nccl flash-attn cudnn"
```

In a consumer crate depending on `inference`:

```toml
[dependencies]
inference = { version = "0.8", features = ["cuda", "nccl", "flash-attn", "cudnn"] }
```

## Default features

`inference-cli`'s default feature is `code-execution`. `inference-server-core`'s default feature is `swagger-ui`. To exclude defaults, use `--no-default-features`.

No crate enables an accelerator feature by default. Opt in to the accelerator matching your hardware.

## Feature verification

`inference doctor` prints a `Build features:` line listing the compiled-in accelerator features (`cuda`, `metal`, `cudnn`, `flash-attn`, `flash-attn-v3`, `cutile`, `accelerate`, `mkl`). Other features such as `nccl`, `ring`, `code-execution`, and `swagger-ui` are not shown on that line.
