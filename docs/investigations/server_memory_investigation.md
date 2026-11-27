# Server GPU memory investigation

A server whose chat load fills a 24 GB card (Inkling 2 Mini, recipe C, vision, 216K q8 context, the drafter at 7,
Logits loop guidance) showed three failures on the LAN box: changing control vectors live and decoding two sequences
failed requests with CUDA out-of-memory and "invalid argument" errors, and an idle server then held 1.2-2.2 GB, so the
next load's KV cache no longer fit (4080 MB asked, 3898 free).

## Run 1 - 2026-11-25 16:25

Scripts in `/mnt/data/models/mem-eval`, each on a fresh dev server (`target/debug/inference serve -p 18081`) with the
LAN load's request; GPU memory from `nvidia-smi` (whole card, the desktop's ~1.1 GB included) or the server process
alone. Question: which path leaves memory behind, and which runs past the load's budget?

| path | result |
|---|---|
| idle server | 1392 MiB card |
| 4 rounds load, request, unload | unloaded settles at ~1889 (a fixed ~500 MB after the first load, no growth) |
| 3 loads failing on the KV cache (400K context), then a good one | 1398 after each failure; the good load fits |
| a refine-mesh job (512, textured) | peak 7603, after 1482 (~90 MB); the chat load fits after |
| 6 in-place reloads, alternating with and without a vector | 23.0-23.3 GB loaded, 1921 after the unload |
| 8-pair vector at load, 2 sequences of 3000 tokens | card peak 23,799 of 24,576 MiB, no failure |
| vector attached live, then 2 sequences | card peak 23,873, no failure |
| server process alone, 1 and 2 sequences | loaded 22,400, peak 22,980 / 23,010, settles ~22,550 |

No clean path leaks, and the server stays inside its plan: decode grows ~600 MB against the 1036 MB the load reserves
for runtime components and activations. What is left for the rest of the card is ~1.5 GB, about 450 MB over the
desktop's idle use; the failures came while the web UI's mesh viewer (WebGL in the browser, on the same GPU) showed
40-50 MB GLBs. The lasting damage is the failure handling: one failed allocation inside a decode step or graph capture
failed every request in flight, logged "Failed to bind CUDA context while restoring graph memory pool" and a run of
"invalid argument" errors, and kept memory until a restart.

Next: make an out-of-memory step recoverable (free what it held, fail only what must fail, keep the model serving), and
decide whether a display GPU's load should leave headroom for other processes.

## Run 2 - 2026-11-25 17:52

Question: can an out-of-memory step be recovered instead of failing its requests? `oom.sh` loads the 216K chat config,
has a second process (`squeeze.py`) take 350-380 MB of the card, sends two 300-token requests, releases, sends one
more, then unloads.

Before any change: the two requests fail (500, "CUDA graph replay launch failed ... CUDA_ERROR_OUT_OF_MEMORY"), the
next one succeeds, and an unload returns the card to 1878 MiB, as a clean one does. The failure is contained: no
memory stays behind and the model keeps serving. A launch failure also blocks the eager retry by design, since the
prelaunch has already staged the step's metadata.

Tried: on `CUDA_ERROR_OUT_OF_MEMORY` from `cuGraphLaunch`, synchronize, return cached memory to the driver and launch
once more (a launch that fails runs nothing, so relaunching the same staged graph is the launch that should have
happened).

- Trimming the device pool and the graph memory (`cuMemPoolTrimTo` + `cuDeviceGraphMemTrim`) after a stream sync: the
  relaunch succeeded ("launched after returning cached memory"), then the process died with SIGSEGV in the driver's
  `cuMemFreeAsync`, in 3 of 5 runs. Frees still queued on other streams against graph memory are the likely cause.
- The pool alone after a context-wide sync: no crashes in 4 runs, but no help either; the relaunch failed again
  (the other process held the memory; the pool had little cached to give back).

Reverted. The remedy is upstream of the step: the load left ~450 MB for everything else on the card because its
context was typed in (216K, a 4.08 GB cache) past what sizing to free VRAM picks (105,344 tokens, 2.19 GB, ~2.1 GB
left at a two-sequence peak). The web UI now has a "Fit to VRAM" checkbox that leaves the length to that sizing.

## Run 3 - 2026-11-27 03:50

Question: why do four concurrent prompts of a few thousand tokens fail together on the 196K chat load (another client's
classification evals, 4 at once, a JSON schema each), when one at a time works? Dev server (`serve -p 18083`) with the
LAN's saved chat load (196,608 q8, vision, DFlash 2 at 7 auto, LoopCD-Logits 0.5): 23,104 MiB of the card after the
load. A script sends 4 concurrent requests with random-word prompts (temperature 0, the schema, thinking off).

| prompts | result |
|---|---|
| 4 x 1.2K | all ok |
| 4 x 3.2K, twice | all ok; between them a lazy graph capture's upload failed (`CUDA_ERROR_OUT_OF_MEMORY`), decode graphs were disabled, and releasing them logged "Failed to bind CUDA context while restoring graph memory pool" and 124 "invalid argument" errors |
| 4 x 6.3K | 1 ok, 3 fail (500, `DriverError(CUDA_ERROR_OUT_OF_MEMORY)` in the step) |
| 1 x 1.2K, then 4 x 1.2K | 1 ok; then 3 of 4 fail, `moe experts forward ... out of memory`: without graphs, a batch that ran before no longer fits |

The 124-error burst is the same one the LAN log shows twice on 2026-11-25 (reloads during the loop-guidance runs); a
clean unload logs none (checked below), so it comes from releasing graphs after a failed upload.

Peak card memory at a 96K load (KV 1,836 MB, 21,412 MiB loaded), 6K-token prompts: one at a time +320 MB, four at once
+1,408 MB, about 725 KB per prompt token of a step (512-token chunks, four sequences: 2,048 a step). At 196K the cache
is 1,836 MB larger: 23,248 + 1,408 = 24,656 MiB of a 24,576 MiB card.

The scheduler splits a prompt batch whose activations would not fit (`admit_cuda_prompt`: the attention workspace plus
`prompt_activation_bytes_per_token` x the step's prompt tokens, when more than one prompt runs), but only Qwen 3.5 and
Nemotron-H report a per-token figure. Inkling reported none, so its batches were admitted on the attention workspace
alone. The shared MoE estimate (`moe_prompt_activation_bytes_per_token`: top-k 8, hidden 2304, expert width 1152)
gives ~342 KB; Inkling's loaders now report it plus 170 bytes per hidden unit for the rest (~733 KB), from the 725 KB
measured above (with LoopCD-Logits, whose second coda pass is in it).

After: the same 196K load, 4 x 1.2K, 3.2K twice, 6.3K twice: 20 of 20 ok, no errors or warnings in the log,
`inference_cuda_prompt_batch_reductions_total` 38. The unload after logged no "invalid argument" errors.

Implies: admission covers Inkling now. Still open: a failed lazy graph upload disables graphs for good and leaves the
eager path short of memory (the 4 x 1.2K failures above), and its release logs the error burst; that is Run 2's
recovery question again, now with a cause that admission avoids.
