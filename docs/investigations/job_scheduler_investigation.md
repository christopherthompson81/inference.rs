# Job scheduler and model residency investigation

Goal: every unit of work (a chat request, an image, a transcription, a world generation, a splat training) is a job,
and the server decides which models are resident to serve them, evicting and loading as needed. The motivating case:
world generation peaks at about 22 GB of a 24 GB GPU and splat training at about 15 GB, for 10 to 30 minutes, so
neither can run beside a served LLM on the same card, and today nothing coordinates them.

## Run 1 - 2026-10-25 06:10

**Question:** what does the engine already have for residency, memory accounting, queueing and long work, so the
scheduler builds on it instead of beside it?

**Finding (code survey):**
- Residency state machine exists in core `InferenceRs` (`inference-core/src/inference_rs/mod.rs:125`): resident
  `engines`, `unloaded_models` (declared but not in memory, with their `ModelLoaderConfig`), `reloading_models`;
  `get_sender` (mod.rs:902) already reloads an unloaded model on the request path. Unload frees memory down to a
  ~410 MB per-process floor (cuda_unload_memory_investigation Run 3). But a never-loaded model cannot be declared:
  `unloaded_models` is only filled by `unload_model`.
- The server `ModelManager` (`inference-api/src/model_manager.rs:264`) serialises loads with a `gate`, keeps saved load
  requests with `load_at_start`, and reports load progress; its unload is a full `remove_model`, not a core unload.
- No memory ledger: nothing records what a resident model holds. Placement (`auto_device_map.rs:61`) sizes against
  whatever is free at load time; `auto_tune` (`inference-selection/src/tuning.rs:467`) estimates fit before a load;
  `plan_paged_kv` splits the KV budget statically at startup; `MemoryUsage::query` gives live free/total.
- No global queue or priority: each model's engine thread has its own channel (capacity 10,000) and its own
  admission policy; an unknown model is a 404, a reloading one a 409; nothing waits for a model.
- Long work: `BackgroundTaskManager` (Responses background mode) is in-memory with states and cancellation, no
  progress or persistence; the web UI's `jobs.rs` only records finished jobs. Reconstruction, world generation and
  splat training are CLI-only and load their own models on CUDA device 0 with no coordination.

**Implication:** the pieces for "declared, unloaded, reload on demand" exist; missing are a ledger of resident memory,
eviction, waiting instead of failing, a job runner with progress and persistence, and device leases that the
self-loading pipelines take before they allocate.

## Design (proposed)

**Resource ledger (per device).** Records each resident model's measured footprint (free memory before and after its
load, plus its KV pool), the process floor, and active leases. Estimates for unloaded models come from `auto_tune` and
the inventory's file sizes, corrected by the measured footprint once a model has been loaded.

**Jobs.** A job has an owner, a kind, its requirements (models it needs resident, or an exclusive or sized lease on a
device), a priority, stages with progress, a cancellation token and a persisted record. Two classes:
- Requests (chat, embeddings, speech, images, transcription): when their model is resident they dispatch as today,
  with no added latency; the scheduler is only involved on a miss (load, or wait for a reload or eviction to finish,
  bounded by a timeout, instead of 404/409).
- Long jobs (world generation, splat training, reconstruction, video to splats, calibration, quantization): queued,
  run on a job worker, report stage progress and events, can be cancelled between steps, and keep their record and
  outputs in the owner's job folder. Kinds that can resume (worldgen reuses finished clips) resume after a restart.

**Scheduling.** Admission checks a job's requirements against free memory plus what eviction can reclaim. Eviction
takes idle (no in-flight sequences), unpinned models, least recently used first, and only as much as needed. An
exclusive job drains its device: new requests for models there queue, in-flight ones finish, the models unload, the job
runs under its lease, and afterwards the models that were resident are reloaded (or left unloaded: a policy choice).
Interactive requests outrank long jobs while they wait; a long job is never preempted mid-stage.

**Pipelines declare before they allocate.** World generation, reconstruction and splat training stop calling
`Device::cuda_if_available(0)` themselves: they take a lease sized from their stage plan (worldgen's per-stage peaks
are already measured in its meta.json) and load their stage models inside it.

**Surface.** In inference-api (the engine surface, per the ABI-first rule): submit, get, list, cancel, an event stream,
and the ledger (what is resident, what each holds, what is queued). Then HTTP (`/v1/jobs...`, `/v1/residency`), C ABI
entries, the regenerated OpenAPI and Python types, and the web UI (a jobs panel; a Worlds view that runs world
generation and splat training as one job and shows the result in a bundled web splat viewer). Configuration gains, per
declared model: lazy (declared, not loaded at start), pinned, idle eviction timeout.

**Phases.**
1. Ledger and measured footprints; never-loaded models declarable (lazy); ledger in `/v1/models/status`.
2. Eviction and load-on-miss for requests: LRU of idle unpinned models, wait-for-model with a timeout.
3. Job runner: one lifecycle for long work (absorbing `BackgroundTaskManager`'s states and the web UI's job records),
   progress events, persistence, device leases, exclusive drain and restore.
4. Reconstruction, world generation and splat training as job kinds behind leases.
5. Surface: HTTP, ABI, bindings, web UI with the splat viewer (compressed splat export first: the 1.9M-splat PLY is
   126 MB).

**Decisions (user, 2026-10-25).**
- After an exclusive job, evicted models are not reloaded: load-on-miss brings them back when asked for.
- Models loaded by hand in the manager are evictable by default (pinning is opt-in).
- Placement is a speed policy: a job may take the faster route (more devices, or evicting more) when the queue is
  quiet, but not when it is busy. Configurable; the default is fastest.
- Restart survival is not required in the first version; the persisted job records should make it possible later.

## Run 2 - 2026-10-25 05:54

**Question:** phase 1 (a measured footprint per model, and models declared without loading): does it hold on the
tiny checkpoints, CPU and CUDA?

**Change:**
- `ModelLoaderConfig::load` returns the pipeline with its `Footprint` (device, bytes): the device's used memory before
  and after the load, each read after a synchronize and a pool trim, so cached blocks are not counted. It covers
  weights, the KV pool and load-time workspaces; a concurrent load or request on the same device leaks into it (loads
  are serialised by the manager's gate; requests are not). Only the load's own device is measured, so a model split
  over two GPUs is undercounted.
- The footprint rides on the engine instance, survives an unload in `UnloadedModelState`, and is reported by
  `/v1/models/status` (`memory`) and `/v1/models` (`memory_bytes`, `memory_device`).
- `UnloadedModelState` now holds a `SchedulerPlan`: `Realized` (from an earlier load, as before) or `FirstLoad` (a
  declared model: the scheduler is built for the pipeline and the tool callbacks set up when it first loads, as
  `add_model` does for a model loaded up front). Its unused category and config fields are gone.
- `InferenceRs::declare_model`; `lazy` on `ModelSpec`, `POST /v1/models/add` and `[[models]]` in the serve TOML. A
  lazy model needs `model_id` (its pipeline name is only known after loading); its KV cache is planned alone, like a
  model added later, and sized against what is free when it loads. A tensor-parallel engine refuses lazy models.
- The async request path now loads an unloaded model with `reload_model().await`. Before, only `get_sender`'s
  `reload_model_blocking` did, which refuses on a current-thread runtime (the first lazy request failed with an
  opaque internal error in a `#[tokio::test]`) and blocks a runtime worker for the whole load.

**Raw finding:**
- `a_lazy_model_loads_on_its_first_request` (CPU, f32): declared, status unloaded, the first request loads it, status
  loaded, greedy ids equal to the same checkpoint loaded up front. Passes, with every existing reload test.
- `a_cuda_model_reports_its_memory` (BF16 on device 0): status reports device 0 and a positive byte count; after an
  unload the status reports the same bytes. Passes.

**Implication:** phase 2 can size evictions from measured footprints and load on a miss without blocking. Still to
check on a real checkpoint: how close the footprint is to nvidia-smi's view, and how much noise a concurrent request
adds.

## Run 3 - 2026-10-25 06:03

**Question:** on a real checkpoint, how close is the measured footprint to nvidia-smi's view of the process?

**Command:** `inference serve -p 1299 --format gguf -m /mnt/data/models -f Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf`
(default runtime settings), nvidia-smi before start and once `/v1/models` answered, then `/v1/models`.

**Raw finding:**
```
idle 1109 MiB, serving 21780 MiB, delta 20671 MiB
/v1/models: loaded, memory_bytes 21313355776 (20326 MiB), memory_device 0
```
The footprint is within 345 MiB of the process's whole growth, about the per-process floor (CUDA context, kernel
libraries, library handles) that no single model owns.

**Implication:** the measurement is good enough to plan evictions with. But a 1.9 GB 3B model holds 20 GB: with the
default paged-attention settings the KV pool takes most of the memory free at load. So under the current defaults one
model fills the card and there is nothing to share; eviction is all-or-nothing. The scheduler needs a KV sizing policy
of its own: size a model's pool for its context and sequences (`paged_context_len`, `max_seqs`) instead of the free
memory, at least for models that should share a device, and let a model that runs alone take the rest. That has to be
settled before phase 2's eviction can share a card between models.

## Run 4 - 2026-10-25 09:21

**Decision (user):** need-sized KV pools by default, so models can share a device; filling free memory stays available
when configured.

**Change:** at load, a paged cache left at its default (no `--pa-context-len`, `--pa-memory-mb` or
`--pa-memory-fraction`) becomes a best-effort budget of one sequence at the model's context (its window after any
`max_model_len`), fitted down to what is free when that is less (`need_sized_default_cache` in core
`pipeline/loading.rs`). A planned MoE expert offload still overrides it with its planned context, as before. The
docs that described the 90% default now describe this.

**Command:** as Run 3 (Qwen2.5-Coder 3B Q4_K_M, default runtime settings), plus one chat request.

**Raw finding:**
```
idle 1060 MiB, serving 4451 MiB, delta 3391 MiB
/v1/models: loaded, memory_bytes 3160408064 (3014 MiB), max_model_len 32768
Allocating 1152 MB for PagedAttention KV cache per GPU; 1024 GPU blocks: available context length is 32736 tokens
max_num_seqs=4; the chat answered
```
**Implication:** 20.3 GB down to 3.0 GB for the same model and the same full context; a 24 GB card now has room for
several such models, so eviction can be partial. Throughput-bound deployments that relied on the big pool for many
concurrent long sequences set `--pa-memory-fraction` (the throughput guide says so).

Seen once on this change's CUDA suite: `a_cuda_model_reports_its_memory` passed, then its process died with SIGSEGV
during exit (3.6 s after "finished"). It did not recur in 25 runs of that test alone or in a full rerun of the suite;
noted in case an exit-time crash after an unload shows up again.
## Run 5 - 2026-10-25 09:39

**Question:** phase 2 (load-on-miss with eviction): does a request for a model that does not fit evict an idle one
and load, and does the evicted model come back?

**Change (core `inference_rs/residency.rs`):**
- Each engine records when a request last reached it; a model is idle when its scheduler has nothing running or
  waiting, its queue is empty and no request reached it in the last second. Evictable: idle, unpinned, on the load's
  device. `evict_one` unloads the least recently used.
- Reload (on a request or by hand) goes through `load_with_room`: a model with a known footprint first evicts until
  that plus 512 MiB is free; then the load is tried with its default KV cache required in full (`need_floor`, the
  best-effort budget's minimum set to the need); on a memory failure one more model is evicted and the load retried;
  once nothing can go, the cache may take less than its need. Memory failures are told apart by message: an
  allocation's out of memory, a best-effort minimum missed, "safely available", the device-map planner's "of usable
  capacity", and the paged cache's "not enough memory" (loader errors are untyped strings).
- `Engine::add_model` (also the manager's load) retries an eager load the same way.
- Requests for a model another request is loading wait for it (`ensure_resident`) instead of a 409.
- Unloading the default model no longer moves the default to another resident model (it now loads on its next
  request); this also removes a lock taken out of the documented order.
- `pinned` on `ModelSpec`, the add body and `[[models]]`; models are evictable by default (user decision).

**Raw finding:** `a_full_device_evicts_the_idle_model` (CUDA): a resident tiny model with a 400 MB KV pool, a lazy
copy declared, the device filled to the loader's 512 MiB reserve with synchronized ballast chunks (one big allocation
failed only at a later sync, as stream-ordered allocations do). The lazy request's first strict load failed ("Num GPU
blocks is 0 ... not enough memory"), the resident model was evicted, the retry loaded: lazy loaded, resident
unloaded. Ballast dropped, a default request brought the resident model back beside it. The tiny-model setup needed
the large resident pool: two 32 MiB models leave too little margin for the ballast's ~20 MiB allocation granularity.

**Implication:** requests now drive residency. Next (phase 3): the job runner, leases for the self-loading pipelines,
and a residency view in the API.

## Run 6 - 2026-10-25 13:01

**Question:** the Run 4 exit-time SIGSEGV in `a_cuda_model_reports_its_memory` came back in a full `local_ci.sh --cuda`
run, and `a_full_device_evicts_the_idle_model` failed beside it ("Internal server error" from a load). What is it?

**Reproduce:** both tests alone, 20 stress iterations (`cargo nextest run --profile cuda --features cuda -p inference
-E <the two> --stress-count 20`): 20/20 pass. The suite runs the CUDA tests while the CPU lint and tests take every
core, so again with 16 busy-loop shells beside it: 1 SIGSEGV in 20. Under gdb in a loop with the same load, the crash
came on the 5th run:
```
Thread 32 (engine thread, inference_rs/mod.rs:457, the closure's end):
  cublasDestroy_v2 <- CudaBlas::drop <- CudaDevice drop <- the engine thread's last Device clone
Thread 1 (main): libcublasLt statics <- __run_exit_handlers <- exit
```
The test had returned and the process was exiting while the unloaded model's engine thread was still tearing down.

**Cause:** `unload_model` removes the engine from the map, sends Terminate with `try_send` and drops the instance, so the
thread runs on detached; `InferenceRs::drop` only joins engines still in the map. Eviction goes through the same
unload, and `make_room` reads free memory right after it returns, before the evicted engine has let go of anything.
With one candidate, a load that then runs out of memory has nothing left to evict, and the smaller-cache retry fails
too: the likely source of the eviction test's failure (that one did not reproduce under load: 20 stress runs before the
change, 80 after, so it is inferred, not seen).

**Change:** unload terminates through `terminate()` (also the out-of-band instruction, as a full queue hides a queued
Terminate), stores the unloaded state, joins the engine thread (within the 10 s drop deadline), drops the instance and
then trims the device's pool. The trim has to come after the drop: the instance's reboot state holds the pipeline's
last reference once the thread has ended, so the thread's own trim ran before the weights were freed.

**Test:** `a_cuda_model_reports_its_memory` now checks that the load's memory is back when `unload_model` returns. On
the tiny model's default load it could not: its whole 32 MiB footprint is per-process (cuBLAS workspace, kernel
modules), and a probe polling free memory for 2 s after an unload saw none of it come back with or without the change.
With a 256 MB KV pool:
```
master:  5/5 fail, "0 of 301989888 returned"
change:  5/5 pass
under 16 busy cores, the memory, eviction and lazy-model tests, 40 stress iterations: 40/40 pass (master: 1 SIGSEGV in 20)
```

**Implication:** an unload is synchronous now: when it returns, the engine thread is gone and its memory is on the
device again, which is what eviction measures against.

## Run 7 - 2026-10-25 14:27

**Question:** phases 3 and 4 (job runner, device leases, the world pipelines as jobs): does a job hold its device to
itself, evicting what is there and keeping loads off it, and do world generation and reconstruction run as jobs with
their stages, previews and outputs recorded?

**Change:**
- Core `lease_device` (`inference_rs/residency.rs`): records the holder (another lease or a pinned resident model
  refuses it), then unloads each resident model on the device as it goes quiet, waiting out loads in flight, within a
  drain timeout (120 s by default). While the lease lives, `load_with_room` and `Engine::add_model` refuse the device
  with `DeviceBusy` (HTTP 409, `device_busy`). Requests for models on other devices are untouched.
- inference-recon `progress::Progress`: stage, advance, preview and cancelled hooks with no-op defaults. World
  generation reports its stages (scene, trajectories, caption, clips, alignment depth, alignment, splat data,
  reconstruction), a preview per start view and per clip (its middle frame), and checks cancellation between stages
  and clips; splat training reports training (every 10 steps), export and mesh, and renders a preview every 1000 steps.
- inference-api `jobs`: one FIFO worker thread holding the engine only weakly (the exit race of Run 6); each job takes
  its device's lease, runs, and records its stages, progress, previews (paths in the job folder), outputs and report
  in `<dir>/<job id>/job.json`, written at each stage, preview and end (progress only goes to subscribers). Kinds: `world`
  (panorama to keyframes, reconstruction, splat data, then splats and meshes unless `splats: false`) and
  `reconstruct` (images or a video); `submit_task` runs caller code under the same lifecycle. Records left queued or
  running by a stopped server read back as failed ("interrupted"); no resume yet. Owners see only their own jobs.

**Tests:**
```
jobs::a_job_records_its_stages_previews_and_outputs                     CPU, submit_task
jobs::a_cancelled_job_stops_and_a_cancelled_queued_job_never_starts     CPU
jobs::a_device_job_evicts_the_idle_model_and_holds_the_device           CUDA: the idle model unloaded when the job ran,
    a reload during it refused "the device is busy: held by job ... (hold)", the model loaded again after; a pinned
    model made a device job fail ("model `pinned-copy` is pinned there")
worldmirror_job (deep, real WorldMirror 2.0, the sample repo's 3 park photos)       pass, 7.2 s
```

## Run 8 - 2026-10-25 15:01

**Question:** does a whole world job (deep test `world_job`, the sample repo's indoor panorama at 1280 x 640, splat
training at 1000 steps) run through the job runner?

**First attempt (14:27):** the machine powered off about 18 minutes in, during the clip loop (the last sample at 14:44:
GPU 100% busy, 16.4 GB in use). The previous boot's journal simply stops at 14:45:04: no thermal, NVRM Xid, MCE or
shutdown entries, so the cause is unknown; an abrupt loss of power (thermal or supply trip) fits, a software fault
would have left a trace. The same stages ran to completion many times before (worldstereo Runs 19-25). The job's
folder lived in /tmp and went with the reboot.

**Second attempt (15:01):** rerun with an nvidia-smi sampler (temperature, power, SM clock, throttle reasons, fan,
every 5 s) and the CPU package temperature (every 30 s) written to /mnt/data/build/job-runner-runs, so another power-off
leaves a record.

**Result:** the machine stayed up (GPU at 69-72 C, power-capped at ~345 W; CPU package up to 79 C of its 82 C "high"),
but the job failed after 1342 s: `DriverError(CUDA_ERROR_OUT_OF_MEMORY)`. The samples place it at the end of the
alignment-depth stage (WorldMirror 2.0 at 832 on all 99 memory frames): memory fell to 3.5 GB as the clip loop ended,
then climbed to 22.8 GB in steps and the job died there.

## Run 9 - 2026-10-25 16:31

**Question:** is the alignment-depth stage too big for the card, or is the job's process the difference?

The CLI on the same panorama (fresh folder, all stages): passes, alignment depth 18,104 MiB allocated at peak, and
nvidia-smi at 23,593 MiB of 24,576 during it. The stage fits only by ~1 GB in a bare process; the job's process
carries the engine's own CUDA state as well, and lost.

The nvidia-smi steps during the stage are ~1.5 GB each, four of them inside the trunk: the four head feature maps
`[99 frames, ~2000 tokens, 2048]` f32 (~1.6 GB each) the trunk keeps while its later layers run. Reruns on the same
folder (clips reused, so only the late stages run):
```
                                   alignment depth           reconstruction (518)    nvidia-smi peak
baseline                           107.3 s, 17,598 MiB       32.1 s,  8,734 MiB      23,569 MiB
pool trims after trunk and chunks  (no change: the peak is inside the trunk)          23,569 MiB
kept maps in host memory           179.4 s, 12,927 MiB      140.6 s,  8,347 MiB      20,487 MiB
  + chunk slices copied out first  110.0 s, 12,927 MiB       29.9 s,  6,915 MiB      18,823 MiB
```
The trunk now moves its kept maps to host memory when the four together pass 2 GiB (the 99-frame passes; a 32-frame
reconstruction keeps them on the card), and the DPT heads bring each chunk's slice over. The first try was slow
because `Tensor::to_device` transfers a view's whole storage, so every 2-frame chunk uploaded the whole 1.6 GB map;
copying the chunk out first fixed it. The pool trims stay (they cut the reconstruction's peak a little). Same values:
the maps are copied, not recomputed.

## Run 10 - 2026-10-25 17:01

**Question:** with the kept maps in host memory, does the world job finish through the job runner?

`scripts/deep_checks.sh -E 'test(/^world_job::/)'`: pass in 1708.7 s, with every stage from scene to mesh, a preview
for the start views, each clip and the training render, and the splats and mesh written. nvidia-smi peak 18,344 MiB
(the failed run reached 22,779 before dying); GPU at most 72 C, and the machine stayed up.

**Implication:** the engine-side job path is whole; next, the full local checks and the HTTP, ABI and binding surface.
