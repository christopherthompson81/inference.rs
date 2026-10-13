# CUDA memory after unload investigation

Loading a model, running a request and unloading it doesn't give all the device memory back.

## Run 1 - 2026-10-13 17:40

**Question:** how much memory stays after an unload, and does it grow with each cycle?

**Command:** empty-start `serve -p 1236 --model-folder /mnt/data/models`; per cycle, `POST /v1/manager/load` of
Qwen3.8 27B IQ4_XS (8192 context, MTP 3), one 250-token chat, `POST /v1/manager/unload`, 3 s, then `nvidia-smi`
used memory. Then three cycles of Kokoro 82M with one speech request each.

**Raw finding:**
- Idle 1274 MiB; empty server 1500 MiB.
- Qwen cycles, after unload: 2060, 2124, 2127, 2285, 2348, 2413 MiB (still 2123 MiB 15 s after cycle 2).
- Kokoro cycles after those: 2414, 2447, 2383 MiB.
- The pool gauges in `/metrics` don't change after unload: the engine loop of a loaded model publishes them, so with
  no model they are stale.

**Implication:** about 560 MB stays after the first cycle (plausibly one-time: the CUDA context, kernel libraries
and library handles initialised on first use), and about 64 MiB more stays with each Qwen cycle, a fixed-size
allocation per load. Next: find what each load allocates once per engine (per thread or per stream) and never frees.

## Run 2 - 2026-10-13 18:10

**Question:** what does each load allocate once and never free?

**Finding (code):** `gguf/fast_mmvq.rs` and `gguf/fast_mmq.rs` (two maps: the MMQ scratch and the stream-k fixup)
each kept a device workspace per (device, stream, thread, power-of-two size), `Box::leak`ed so CUDA graph replay
sees a stable address, and never freed. A model's engine runs on its own multi-thread runtime, so every load brings new
threads and a new set of workspaces. The MoE dispatch workspaces in `gguf/cuda.rs` key by device and stream only and
cuBLASLt is once per process, so neither grows per load.

**Change:** one `gguf/thread_workspace.rs` (`ThreadWorkspaces`) for both launchers; a thread-local registry frees the
thread's workspaces when the thread ends. Only the owning thread can lock its workspaces, and an engine's runtime
threads end when its model is unloaded.

**Raw finding (same cycles):** empty server 1493 MiB; after unload 2066, 2039, 2028, 2041, 2030, 2030 MiB.

**Implication:** the per-cycle growth is gone; about 540 MB stays from the first load on. Next: whether the CUDA async
pool, whose release threshold is `u64::MAX` (it never hands memory back by itself), holds part of it.

## Run 3 - 2026-10-13 18:25

**Change:** when a model's engine thread ends, its runtime is dropped first (so its threads free their workspaces),
then the context is synchronized and the async pool trimmed to 0 (`release_device_memory` in core
`inference_rs/mod.rs`). Trimming releases only unused cached memory, so other models on the device keep theirs.

**Raw finding:** empty server 1490 MiB; Qwen cycles after unload 1896, 1900, 1935, 1902, 1901, 1899, 1899 MiB; two
Parakeet (.nemo) cycles after them 1902, 1902 MiB; no `Failed to return` warnings.

**Implication:** about 140 MB of the floor was pool cache. What stays (about 410 MB over the empty server) is held once
per process from the first load: the kernel libraries' modules, the cuBLAS/cuBLASLt handles and the context's own
growth; it doesn't grow with more loads.
