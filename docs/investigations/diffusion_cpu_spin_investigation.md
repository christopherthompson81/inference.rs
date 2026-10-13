# Diffusion CPU spin investigation

An image edit pins one CPU core for as long as it runs.

## Run 1 - 2026-10-13 14:05

**Question:** which thread is busy during a HiDream-O1 edit, and is it doing work or waiting?

**Command:** one `POST /v1/images/edits` (HiDream-O1-Image-Dev Q8_0 with its vision tower, a 2048x2048 reference,
"Make the water green") against the running server, while a sampler read every thread's CPU time from
`/proc/<pid>/task/*/stat` and the GPU's utilization from `nvidia-smi` every 0.5 s.

**Raw finding:** 76.4 s for the edit. From the moment the GPU reached 100%, one engine thread (`tokio-runtime-w`) read
102-106% in every sample through 60 s, with nothing else above 6%. No context in the workspace sets a scheduling flag
(`CU_CTX_SCHED_*`), so CUDA's default applies: a thread waiting on the GPU spins.

**Implication:** the GPU is the bottleneck and the thread is waiting, not working. The denoising loop queues steps far
ahead; once the launch queue fills, each launch, and the final copy back, waits by spinning. A context-wide blocking
wait would also slow LLM decode, which waits once per token, so the fix belongs in the diffusion loops.

## Run 2 - 2026-10-13 14:20

**Change:** after each denoising step (HiDream-O1 and FLUX), wait for the device with a blocking-sync event
(`CU_EVENT_BLOCKING_SYNC`), which sleeps the waiting thread.

**Raw finding:** no thread above 1% in any sample while the GPU ran at 100%; 79.5, 80.0 and 79.5 s per edit.

**Implication:** the spin is gone, but waiting at the end of every step leaves the GPU idle while the next step is
queued. Next: let the queue run one step ahead.

## Run 3 - 2026-10-13 14:40

**Change:** `Device::fence` records a blocking-sync event after a step; the loop waits on the previous step's fence
after queueing the current one, so the GPU always has a step queued and the thread sleeps.

**Raw finding (same server, image and seed 7 for each build):**

| build | edits (s) | hottest thread while the GPU runs |
|---|---|---|
| master | 81.3, 81.0, 81.0 | ~105% (Run 1) |
| one step ahead | 77.5, 78.1, 80.3 | under 1%; 52-64% only for the ~0.5 s decoding and encoding the images at the request's ends |

**Implication:** the pinned core is freed with no loss of speed. Kept.
