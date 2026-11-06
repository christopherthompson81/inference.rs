# Vulkan prefill on a discrete GPU: where long prompts go

Nemotron 3.5 Lightning (`Lightning-exps-q4_0.gguf`) on the RTX 3090 over Vulkan prefills ~3260 tok/s at 4K and ~2000
at 8K. The model is a hybrid (7 attention layers of 53: 32 query heads, 2 KV heads, head size 128; the rest Mamba-2
and MoE), so per-token cost should barely grow with length.

## Run 1 - 2026-11-05

**Question:** what takes the extra time at 8K?

**Command:** `INFERENCE_VULKAN_PROFILE=1 inference bench --vulkan --format gguf -m nemotron-lightning-gguf -f
Lightning-exps-q4_0.gguf --prompt-len {4096,8192} --gen-len 1 --iterations 6 --warmup 1`, the per-kernel blocks summed.

**Finding:**

| kernel | 4K share | 8K share | per dispatch 4K / 8K |
|---|---|---|---|
| `flash_attn` | 46.1 % | 63.2 % | 2.88 / 5.6 ms |
| `mul_mm_cm2_q8_0` | 11.4 % | 7.5 % | |
| `moe_mm_cm2_q4_0_n128` | 9.2 % | 7.0 % | |
| `ssm_conv_step` + `ssd_*` | 12.6 % | 10.1 % | |

- Prompt attention runs in `flash_attn` (`vulkan_backend::flash_attn`, f32, no matrix cores); no `paged_prefill` or
  `paged_prefill_cm` dispatch appears in the bench at either length.
- Causal attention FLOPs over the 7 layers: 2 N^2 d H = 0.96 TFLOP at 4K, 3.85 at 8K. Against the share of TTFT that
  is ~1.65 TFLOPS at 4K and ~1.5 at 8K, a few percent of the 3090's f16 tensor rate.
- The rest scales linearly (MoE / Mamba dispatch time per prefill about doubles from 4K to 8K, as tokens do).
- A profiled 4K run segfaulted at exit, after the results printed (8K did not); not yet looked at.

**Implication:** the 8K drop is attention's quadratic term showing through a slow kernel: already half the GPU time at
4K. Prompt attention on matrix cores (as `paged_prefill_cm` does over a paged cache), at even 10-15 TFLOPS, would cut
4K TTFT ~1.27 -> ~0.75 s and 8K ~4.0 -> ~1.7 s.

## Run 2 - 2026-11-05

**Question:** does prompt attention on matrix cores close the gap?

**Change:** `flash_attn_cm.comp`, `paged_prefill_cm_main.glsl`'s tiling (f16 16 x 16 x 16 cooperative matrices, 64 rows
a workgroup as positions x the kv head's query heads, 32-key tiles, lazy rescale) over `flash_attn`'s contiguous f32
Q/K/V, push constants and mask, writing normalized rows directly (no splits). `Recorder::flash_attn` takes it where
the device has cooperative matrices and heads are multiples of 16 up to 128 with groups up to 64; otherwise the f32
kernel. The fused-attention test adds a causal-without-mask case past a tile and head 32, at 4e-3 for f16 inputs
(worst seen 1.3e-3).

**Finding** (3 iterations + 1 warmup, decode 16):

| prompt | TTFT before | TTFT after | tok/s after |
|---|---|---|---|
| 2048 | 499 ms | 385 ms | 5315 |
| 4096 | 1256 ms | 789 ms | 5189 |
| 8192 | 3.99-4.25 s | 2.08 s | 3941 |
| 16384 | 34.7 s | 18-22 s | ~790-975 |

`flash_attn_cm` runs 1.5 ms a dispatch at 16K where `flash_attn` took 5.6 ms at 8K. At 16K the run is no longer
attention-bound: the device fills (`vulkan: device memory is full (1024 MiB asked, 22114 MiB held, 965 MiB free)`) and
`ssd_chunk_state`, the MoE matmul and `matmul_f32` run 10-20x slower per dispatch over host memory. 16K spilled
before the change too.

The profiling build segfaults at exit intermittently (4K; also on the #826 binary without this branch's changes, not
on master at 2K); not followed up.

**Implication:** attention is done. What is left at long prompts is VRAM: loaded the model holds ~19.6 GB of 24, and a
prompt's forward grows to the device's edge (peak +1.0 GB at 2K, +2.5 at 4K, +4.0 at 8K, full at 16K). By design the
CPU records a whole forward ahead of the GPU, so every temporary of it stays alive until its batch's fence, and idle
pooled device memory is uncapped (synchronized and freed only once the device is full).

## Run 3 - 2026-11-05

**Question, from the user:** VRAM moves during a run while the model rides near the 24 GB edge; can nothing need more
after load?

**Command:** `serve --vulkan` with a 16,055-token chat prompt, VRAM sampled every 0.2 s; then `bench --vulkan` at 2K-16K.

**Finding, non-paged (the Vulkan default until now):** ready at 19.5 GB; the 16K prompt peaks at 24.1 GB and spills
(`vulkan: device memory is full (1024 MiB asked, 22216 MiB held, 992 MiB free)`), 1431 prompt tok/s. A prompt is one
forward, so its whole working set lives at once.

`--paged-attn on` (fixed-size cache, 512-token prompt chunks), one problem after another:
1. `key must be a cuda tensor`: with `cuda` on, `inference-paged-attn` exports only its CUDA functions, and the Vulkan
   branches live in the CPU module; `inference-nn` also planned decode for CUDA's kernels. Any `cuda,vulkan` build
   failed paged attention on Vulkan (the 780M's Vulkan-only build did not).
2. `Vulkan's paged caches hold head size 128, not 2`: CUDA builds write 4-D `[batch, seq, heads, head]` rows directly;
   Vulkan's write takes them flattened.
3. Decode at 8 tok/s: the KV cache took 12,288 MB, the model's whole 2M-token context. `MemoryUsage` reads every Vulkan
   device as an iGPU (a share of system RAM), and the post-load fit of the cache ran on CUDA only.
4. With both fixed, still 12,288 MB: post-load free memory read 22,735 MiB, as a discrete GPU's quantized weights reached
   its memory only on first use; the cache took the room and the weights landed in host memory.
5. Weights copied at load: a 4,273 MB cache leaving 128 MB for activations (the predicted reserve), so prompts spilled
   (140 tok/s on 16K) and decode fell to 3-4 tok/s.

**Change:** Vulkan entry points in `inference-paged-attn`'s CUDA build (write, gather, copy, decode to the shared
module), a Vulkan cache classed Standard and decoded by its own kernel, 4-D writes direct on CUDA only; a discrete
Vulkan device reports its heap and free memory (`Context::device_memory`), its weights go to device memory at load
(`QStorage::from_data`), the post-load fit covers it, with 2 GiB left for activations; paged attention defaults on for
discrete Vulkan GPUs (iGPUs unchanged).

**Finding, after:**

| `serve`, 16K prompt | non-paged | paged, sized |
|---|---|---|
| VRAM at ready | 19.5 GB | 22.1-22.4 GB (KV 2.3-2.5 GB, ~410K tokens) |
| peak | 24.1 GB, spilled | 23.6 GB |
| prompt | 1431 tok/s | 3779 tok/s |
| decode (300 tokens) | ~178 | 190-200 |

| bench, paged default | TTFT | peak VRAM |
|---|---|---|
| 2K | 385 ms | 23.5 GB |
| 4K | 969 ms (non-paged 789) | 23.6 GB |
| 8K | 1.91 s (non-paged 2.08) | 23.6 GB |
| 16K | 4.23 s (non-paged 18-22, spilled) | 23.6 GB |

Peak no longer depends on the prompt's length. Decode at depth 4 is 156-175 tok/s. The exit segfaults (Run 2) are
in an NVIDIA driver thread while `exit()` runs: the Vulkan device lives in a static never destroyed.

## Run 4 - 2026-11-05

**Question:** what are the exit segfaults (2 of 4 paged bench runs at 4K/8K, most profiled runs)?

**Command:** `coredumpctl debug <pid>` on two cores, `thread apply all bt`.

**Finding:** the faulting thread is the NVIDIA driver's own (`libnvidia-glvkspirv` called from `libnvidia-glcore`); the
main thread is in `exit()` -> `_dl_fini` -> `mi_process_done` (mimalloc's purge). The Vulkan context lives in
`vulkan_backend::CONTEXT`, a static never dropped, so the device and instance are never destroyed and the libraries'
exit handlers run under the driver's live threads. Not new: the #826 binary crashed the same way under profiling.

**Change:** `Context::shut_down` waits for the GPU and destroys the pipelines, pool, stream, device and instance while
other references remain (after it a buffer's drop leaks, recording and allocating fail); the backend registers it
with `atexit` when it creates the context, so it runs before the libraries' teardown.

**Result:** 0 crashes in 8 bench runs (4K and 8K alternating) and 3 profiled runs.
