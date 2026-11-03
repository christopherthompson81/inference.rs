# A Vulkan backend for UMA iGPUs

Target: the Radeon 780M (RDNA3, gfx1103, Mesa RADV) beside a Ryzen 7 7840U, sharing dual-channel DDR5-5600. Goal:
decode and prefill on the iGPU for models the CPU path runs today, starting with Nemotron 3.5 Lightning
(`nemotron_lightning_cpu_investigation.md`, Runs 13-14).

## Why

- The CPU's reads cap at 58-59 GB/s (its fabric link); the iGPU reads 73-79 GB/s from a model-sized buffer.
- llama.cpp's Vulkan backend runs Lightning recipe A at 22.2 tok/s decode and 411 tok/s prefill, against our CPU
  path's 17.8 plain / 20.4 with MTP and ~106 prefill.

## Decisions

- **Its own crate, `inference-vulkan`**, on `ash` (the loader opened at runtime, so a build needs no Vulkan SDK to
  link), with no dependency on `inference-tensor`, so the tensor crate can take it as its backend later. Everything
  is behind its `vulkan` feature; without it the crate is empty, so `--workspace` builds need neither `glslc` nor a
  driver.
- **Shaders compile ahead of time**: `build.rs` runs `glslc` over `shaders/*.comp` and embeds the SPIR-V.
- **Zero-copy weights**: the iGPU reads host memory at near carve-out speed, so weights the CPU path already holds are
  imported (`VK_EXT_external_memory_host`) rather than copied: 73 GB/s imported against 79 GB/s for GTT allocations,
  for no second copy of the model. GTT must be raised from the 4 GiB default (`amdgpu.gttsize`, `ttm.pages_limit`).
- **Bindings stay under 4 GiB** (a storage binding past it reads zeros), and one submission stays well under the
  GPU's job timeout (~2 s of work loses the device).

## Phases

1. Device, memory (import, GTT allocations), pipelines from embedded SPIR-V, and one kernel: a Q8_0 matrix-vector
   product, checked against a CPU reference and measured at full bandwidth on an `lm_head`-sized weight.
2. The rest of a decode step's kernels (Q5_0 / IQ4_NL matvec, routed experts, RMSNorm and elementwise ops, the
   Mamba conv and scan, attention over the KV cache, the router), each against the CPU path.
3. Whole layers recorded into one command buffer per decode step, so a token costs one submit; the model's forward
   on the iGPU, then MTP and prefill.

## Runs

### Run 1 - 2026-11-03 ~16:00

**Change:** `inference-vulkan`: a context (the integrated GPU first; push descriptors, host-memory import, 16-bit
storage), host-visible buffers (cached), zero-copy imports rounded out to the import alignment, pipelines from the
embedded SPIR-V, and `mul_mat_vec_q8_0` (two rows a 64-thread workgroup, the threads striding over a row's blocks,
a shared-memory reduction; f32 activations, u16 loads of the blocks).

**Tests:** `a_q8_0_matvec_matches_the_cpu` (333 x 2688, a weight starting mid-page, offset activations and
outputs, against an f64 dequantized reference); `scripts/local_ci.sh --vulkan`.

**Result:** an `lm_head`-sized weight (131072 x 2688 Q8_0, 374 MB) imported from host memory, twenty matvecs a
submit: 76.1 GB/s, 4.9 ms each (the raw-read test's imported ceiling was 73-79 GB/s).
