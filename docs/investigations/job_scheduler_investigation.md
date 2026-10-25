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

**Open questions for the user.**
- After an exclusive job, reload the models it evicted, or leave them for load-on-miss?
- Default eviction policy for models loaded by hand in the manager: evictable, or pinned unless marked otherwise?
- Multi-GPU: should a job be able to take a whole second card while the first keeps serving?
- Persistence and resume across server restarts: required for long jobs in the first version, or later?
