# A Code-tab turn failing with a CUDA OOM in the MoE prefill

Symptom: a web UI Code session (Inkling 2 chat on the 3090, 163840 ctx, q8 KV, vision) ended a turn with "The model
failed to process the request." The server log behind it:

```
step - Model failed with error: moe experts forward
moe experts fast grouped
DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")
```

`fast grouped` is the MoE path for prompt steps of 32 tokens or more, so a prompt step ran out of device memory while
allocating its expert workspace. The client saw only the generic model error.

## Run 1 - 2026-11-27 20:10

Question: is the failure reproducible from the session's requests?

Replayed the session turn by turn against the live server (`replay.py`: each cut of the history ending in a user or
tool message, the screenshot inlined as a data URL, max_tokens 200). All 20 turns succeeded, including the one that
failed and one prefilling ~3,000 new tokens; the context never passed ~11K tokens. Nothing else ran on the GPU at
the time (journal clean; the agent's commands were `xxd`, `grep` and `curl`), and the Code tab's label requests start
only once a run ends, after the failing step.

Device at rest: the server's pool holds 22.5 GB reserved (22.2 GB in use) of 24.6 GB; KWin, Chrome's GPU process
and VS Code hold ~0.6 GB; ~1.1 GB is free at the driver.

Pool metrics since the server started (~5 h):

```
reclaimed bytes   prompt_preflight 24.8 GB, graph_reclaimed 14.7 GB, prompt_boundary 8.9 GB, idle 1.7 GB
trim successes    prompt_preflight 141 (of 218), graph_reclaimed 121, prompt_boundary 24, idle 5
pressure events   reserve 337, transient 60, hard 28
chunk reductions  44
```

Implies: not deterministic. The pool hands memory back to the driver constantly, most of it right before prompt
steps, which then draw it back from the driver mid-forward, where anything else on the device may have taken it.

## Run 2 - 2026-11-27 21:40

Question: does the admission estimate undercount what a prompt step draws, or is the memory lost between admission and
allocation?

Logged each prompt step's peak draw from the pool (used high-water reset at preflight, read at the step boundary)
beside the preflight's transient estimate; dev build, same model and load, replay of the session (by then 57 turns:
the session went on after the failure).

```
steps 175; with an estimate 37, peak > estimate 0, max peak/estimate 0.72 (590 MiB drawn against 828 MiB)
without an estimate 138, max peak 138 MiB (under the 256 MiB warm cache); max peak 590 MiB
```

The preflight trim targets `used + warm_cache` (256 MiB here) even when it knows the step is about to draw ~600 MiB,
so every trimmed step regrows ~330 MiB from the driver. With the trim keeping `max(warm_cache, transient, measured
peak)` before a prompt step: preflight trims released nothing (7 calls, all unchanged, vs 141 successes live). The
graph-reclaim trim inside preflight still targeted the warm cache (8 trims, 4.2 GB), so it takes the same rule when
called for a step about to run.

Implies: the estimate is conservative; the exposure is the trim-then-regrow. Kept: a trim for a prompt step keeps what
the step draws; the boundary and idle trims still target the warm cache, since the engine keeps a driver-free floor
(holding step memory there would trip graph eviction after every prompt step). Beside it, a GPU out-of-memory model
failure now reaches clients as an overloaded, retryable error (`out_of_device_memory`) instead of the generic one, and
the coding agent sends a turn again once when a server-side failure arrives before any of its reply.
