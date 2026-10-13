# Display GPU memory investigation

A model loaded on the GPU that also drives the desktop fills its memory, and the browser tab showing the web UI renders
corrupted until the model is unloaded.

## Run 1 - 2026-10-13 16:10

**Question:** where does the memory go when Qwen3.8 27B IQ4_XS loads with MTP on a 24 GB RTX 3090 that also runs the
desktop, and is the corruption memory exhaustion?

**Setup:** the empty-start server; the model loaded from the web UI's chat slot (8192 context, MTP 3 drafts). The
desktop (KWin, Plasma, two browsers, an editor) holds about 1.1 to 1.3 GB before any load.

**Raw finding:**
- With the model loaded from the UI: 24016 of 24576 MiB used, 128 MiB free; the server held 22620 MiB.
- Reproduced through the manager: 24068 MiB after load, 23419 MiB after one chat; the engine's own log has
  `CUDA decode graph precapture failed ... CUDA_ERROR_OUT_OF_MEMORY`.
- The user confirmed that unloading the model clears the corruption, so it is VRAM exhaustion, not a UI bug.
- The weights file is 15.7 GB and the KV cache 544 MB at 8192 tokens (`256 GPU blocks`).
- `serve -m /mnt/data/models -f Qwen3.8-27B-IQ4_XS.gguf --pa-context-len 8192` with the scratchpad `memrun.sh`
  (VRAM after load, then after one 250-token chat):

| max seqs | MTP 3 | after load | after chat | engine OOM |
|---|---|---|---|---|
| 32 (default) | off | 22097 MiB | 22152 MiB | no |
| 32 (default) | on | 23183 MiB | 23450 MiB | yes (graph precapture) |
| 4 | off | 18130 MiB | 18194 MiB | no |
| 4 | on | 18669 MiB | 18841 MiB | no |

- At 32 sequences the hybrid cache's recurrent pool reserves `capacity=33`; about 4 GB of the load is state for 32
  concurrent sequences.
- `automatic_recurrent_capacity_budget` (core `pipeline/mod.rs`) already fits the recurrent pool to the free memory:
  the engine fills the device by design, as it should on a GPU it has to itself.
- `nvidia-smi -q`: `Display Active: Enabled`. `cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT)` is 1 on
  this device (NVML's `nvmlDeviceGetDisplayActive` agrees), so CUDA alone tells a display GPU apart.

**Implication:** nothing leaves room for the desktop on a display GPU: the compositor and browsers, which allocate
as they repaint, get what the engine left (128 MiB). Free memory is read in one place for load planning
(`MemoryUsage::query`) and one for runtime pool upkeep (`query_cuda_allocator`); a reserve subtracted there on a
display device leaves the desktop room without touching each sizing path. Next: reserve 1 GiB on a device whose kernel
exec timeout is set, and measure again.

## Run 2 - 2026-10-13 16:20

**Question:** does subtracting a 1 GiB display reserve from the free memory `MemoryUsage::query` and
`query_cuda_allocator` report leave the desktop room?

**Change:** `DISPLAY_DEVICE_RESERVE_BYTES` (1 GiB) subtracted on a device whose `KERNEL_EXEC_TIMEOUT` is set; the log
says `A CUDA device drives a display: leaving 1024 MiB of it free for the desktop.`

**Raw finding (same `memrun.sh`):**
- 32 seqs: 22177 MiB after load, 22174 MiB after chat; 32 seqs + MTP 3: 23186 / 23211 MiB, graph precapture still
  OOMs. Unchanged from Run 1.
- `/metrics` after the MTP load: async pool `used` 21.34 GB current (23.14 GB high), `reserved` 21.64 GB; graph pool
  0.84 GB; `inference_cuda_memory_pressure_total{level="hard"} 4`. Live tensors, not allocator cache.
- `INFERENCE_RS_GGUF_AFFINE_BACKEND=off`: 22127 MiB. The affine repack is not it.
- `--max-seqs 16`: recurrent `capacity=17`, 19612 MiB. `--max-seqs 8`: 18171 MiB. About 156 MB of recurrent state per
  slot, so the default 32 sequences hold 5.1 GB.

**Implication:** the reserve reaches the planner, but the planner is not budget-bound here: 33 slots fit the free memory
with or without it, since the slot count comes from `max_seqs`. What eats the last GB is growth after load that the
plan doesn't count (CUDA graphs 0.84 GB, MTP state, activations, against a 48 MB activation reservation), and CUDA
hands that out until the device is full. A reserve in the planner alone can't protect the desktop; the slot count
(a web UI serves a handful of sequences, not 32) and the post-load growth are what decide it.

## Run 3 - 2026-10-13 16:25

**Question:** with the default concurrent sequences lowered from 32 (16 for the server builder) to 4, settable per
model (`max_seqs` in a model spec, the manager's load request, a TOML `[[models]]` entry and the UI's load options),
does the same load leave the desktop room?

**Command:** empty-start `serve -p 1236 --model-folder /mnt/data/models`; `POST /v1/manager/load` with the same source,
8192 context and MTP 3; one 400-token chat; then the same load again with `"max_seqs": 12`.

**Raw finding:**
- Default: `max_num_seqs=4`; 19401 MiB after load, 19046 MiB after the chat (from 24068 MiB in Run 1), no graph
  precapture OOM, 58.1 tok/s.
- `max_seqs: 12`: `Recurrent state pool reserved capacity=13`, `max_num_seqs=12`; 24037 MiB right after the load,
  21860 MiB after the chat, and the graph precapture OOM is back.
- Replacing a model loads the new one before the old one's memory is back with the device, so the reload peaks above
  either alone.

**Implication:** the slot count was the lever: the default load now leaves about 5 GB free on the 24 GB card. A user
who raises `max_seqs` on a display GPU can still crowd the desktop, since the growth after load is not planned
(Run 2); the replacement peak is a separate follow-up.

## Run 4 - 2026-10-13 17:05

**Question:** does the change pass the full local suite (`scripts/local_ci.sh --lint --tests --cuda --bindings --docs`)?

**Raw finding:** CPU 2672 passed; CUDA 3053 passed, 16 failed, all loads of tiny models with
`Primary device cuda[0] cannot fit its fixed model components, activations, and post-load reservation within 0 MB of
usable capacity` or `PagedAttention KV cache requires 4 MB but only 0 MB is safely available`. The GPU tests run side
by side (nextest's GPU group), so a test can see under 1 GiB free, and the display reserve took all of it.

**Implication:** the reserve turns a tight but workable load into a refused one, and Run 2 showed it doesn't decide
the desktop's room here. Dropped it; the lower default sequence count stays the fix.
