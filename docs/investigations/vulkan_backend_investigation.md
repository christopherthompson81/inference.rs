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

### Run 2 - 2026-11-03 ~20:50

**Change:** `inference-vulkan` records into a stream: dispatches append to an open command buffer, submitted without
waiting every 64 dispatches, each followed by a barrier, with the resources a batch uses kept alive until its fence
signals. `Device::Vulkan` / `Storage::Vulkan` / `QStorage::Vulkan` in `inference-tensor` (a stub without the
feature): storage in host-visible buffers; every op, custom op and in-place op runs the CPU backend's kernel on a
copy and uploads the result; quantized weights keep CPU blocks. Model code that branches by device sends Vulkan
down the CPU's branch (Mamba's scan / conv / pool step, the MoE experts from their blocks, the mapped-CPU expert
backend), and CPU kernels' outputs follow their input's device. `--vulkan` / `vulkan:0` select it; F32 activations.

**Measured:** a submit-and-wait costs ~57 us on the 780M and a batched dispatch ~10-15 us, so the design submits only
when the CPU reads.

**Tests:** the backend's ops against the CPU's (`vulkan_backend::tests`); `nemotron_h_tiny::
a_long_prompt_scores_on_vulkan_as_on_the_cpu` (plain and Q8_0 experts, with the CPU's KV cache in f32: its f16 default
was the only difference, which the quantized checkpoint amplified to 0.015).

**Result (Lightning recipe A, 64-token prompt, 32 decoded):** prefill 23.3 tok/s, decode 3.0 tok/s (332 ms a token):
~0.5 ms an op of fallback (a synchronize, two copies, a fresh Vulkan allocation), the baseline the kernels replace.

### Run 3 - 2026-11-03 ~21:10

**Change:** host buffers pool by capacity (powers of two from 256 B to 256 MiB, up to 2 GiB idle): a dropped buffer
returns to its class, which the stream's retention makes safe, and a new one reuses it rather than allocating.

**Result (as Run 2):** prefill 23.3 -> 61.2 tok/s, decode 3.0 -> 8.1 tok/s (124 ms a token).

### Run 4 - 2026-11-03 ~21:40

**Change:** `map_f32` (unary, binary with broadcasting, affine, powf, elu, fill over strided f32 operands of up to 6
dims, by op code) and `copy_b32` (copies and fills of any 4-byte element bit for bit); `VulkanStorage` runs its unary,
binary, affine, powf, elu, strided and 2-D copies, same-dtype `to_dtype`, `try_clone` and `const_set` on them.

**Count first:** a decode step had ~285 fallback calls a token: binary ops, strided copies (`copy2d`), unary ops,
gathers, affine, f32 matmuls (router, attention), reductions, then the custom ops and in-place pool updates.

**Result (as Run 2):** prefill 61.2 -> 80.9 tok/s; decode 8.1 -> 6.6 tok/s. A native op leaves work in flight that
the next fallback op waits for (~60 us a submit-and-wait), so decode pays for every boundary until a layer's ops are
all on the GPU.

### Run 5 - 2026-11-03 ~23:00

**Change:** `mul_mat_vec_q5_0` beside Q8_0's; `CustomOp1::vulkan_fwd` (an op's Vulkan kernel, or `None` for the CPU's
on a copy); `QStorage::Vulkan` weights run decode and verify rows (up to 8) on the GPU from a driver-allocated copy of
their blocks, made on first use.

**Finding: imports cost every submit.** Weights first ran from zero-copy imports: decode fell to 3.4 tok/s, with a
submit of one small dispatch taking 1.3 ms. A tiny dispatch's submit with imports alive: 60 us with none, 181 us with
150 MB, 1547 us with 1.8 GB (~0.8 us a MB: RADV validates userptr pages each submit). With driver allocations alive:
36 us at 1.8 GB and 35 us at 6 GB. Imports suit one buffer streamed in long submits, not a model's weights; the GPU
clock (forced `high`) and the CPU's spinning workers were ruled out first.

**Kernels at Nemotron's shapes (synced / batched):** Q8_0 8448x2688 410 / 343 us (70 GB/s), Q5_0 3712x2688 157 / 100
us (69 GB/s), Q8_0 2688x4096 222 / 156 us (75 GB/s).

**Result (as Run 2):** decode 6.6 -> 7.1 tok/s (a weight copy each: +1.9 GB for recipe A's dense layers). Prefill
fell to 45.6 tok/s with the Vulkan quantized storage alone (the CPU path's tiles are a likely cause, to look at with
GPU prefill).
