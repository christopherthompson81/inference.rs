# Engine teardown in an embedding process

A C ABI caller (here the Python bindings) that loaded a second engine after closing a first hung at exit, and the
second engine ran slowly.

## Run 1 - 2026-10-30 20:30

**Question:** why does a process that closes one engine and loads another hang at exit, and why is the second slow?

**Command:** a probe script through the bindings: load a GGUF engine, one 16-token chat, `close`, `nvidia-smi` memory
used before and after; once per process and twice in a row, each run under `timeout 45`. The hung run under gdb
(`thread apply all bt 45`, interrupted after 25 s). Models: ERNIE-4.5-21B-A3B and 0.3B, Qwen2.5-Coder-3B, Qwen-1.8B,
Qwen1.5-MoE-A2.7B GGUFs.

**Finding:**
- The hang. Main thread in the exit handlers, unloading cudart's libraries (`cuLibraryUnload`), waiting on a driver
  lock; a `moe-expert-fills` thread in an expert cache copy (`cuMemcpyHtoDAsync`) waiting on it; and the driver's
  callback thread, inside our host function `fill_landed`, dropping the last `Arc` of an offloaded layer whose engine
  had closed, so freeing the layer's copy stream with `cuStreamDestroy`. A host function may not call CUDA; the driver
  deadlocked. `fill_landed` now hands the layer back to the fill thread, which drops it. Every two-engine order (plain
  and plain, MTP and plain, plain and MTP) exits.
- The slow second engine had offloaded its experts: the first engine's memory was still in use after `close`
  (ERNIE 21B: 21,716 MiB after close against 1,407 before; with `max_model_len` 8192, 14,932). The pool reported it
  used, not just reserved (13.8 GB), and trimming it after a device synchronize released nothing. Neither CUDA graphs
  (`INFERENCE_RS_CUDA_GRAPHS=0` kept it) nor the MoE path was it: the dense ERNIE 0.3B and Qwen2.5-Coder-3B kept
  theirs too, while Qwen-1.8B and Qwen1.5-MoE freed. The engine's drop ran, joined its threads and dropped the
  pipeline's last reference, and `CausalLm`'s drop ran.
- Binding the CUDA context in `CausalLm`'s drop freed ERNIE's memory. cudarc's `CudaSlice` drop calls
  `cuMemFreeAsync` without making its context current (its `CudaStream` drop does bind; upstream 0.19.10 is the same).
  The engine is dropped on the caller's thread, which never bound a context, so every free failed with
  `CUDA_ERROR_INVALID_CONTEXT`, which `record_err` stores silently; the stored error also made the next
  `bind_to_thread` on that context return it, the "invalid device context" seen earlier. A model freed only when
  something earlier in its drop happened to bind the context on that thread: the four that kept memory differ from
  the two that freed in that order, not in a model feature.
- With `CudaSlice`'s drop binding first: ERNIE 0.3B 1,765 MiB after close, ERNIE 21B 1,813, Qwen2.5-Coder 1,778,
  Qwen-1.8B 1,800 (the rest is the context and loaded kernels), and a second engine's chat takes 0.1 s, not 1.0 s.
  `a_tensor_dropped_on_a_thread_without_a_context_frees_cleanly` fails without the bind.

**Next:** `record_err` keeps every drop-time error without a trace; logging it once when recorded would have shown this
on the first run.
