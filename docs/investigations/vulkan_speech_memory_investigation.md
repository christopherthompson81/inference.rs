# Parakeet and Nemotron-3 diarization: Vulkan's GPU memory on a discrete GPU

The user noticed Vulkan using more GPU memory than CUDA for the speech models on the RTX 3090. Clip: the 605 s
synthetic speech file of `vulkan_relpos_3090_investigation.md`; both models served from one `from-config`
(`vulkan = true`, or the default CUDA device), a short warm-up, then ASR with word timestamps, diarization, and both
in one request.

## Run 1 - 2026-11-08 10:40

Question: how far apart are the two backends, and where does the difference come from? `nvidia-smi` memory.used
after each phase (master d8194c27c; the desktop idles at 1,192 MiB), MiB above idle:

```
          loaded  warmed  after asr  after diar  after both   asr / diar / both
Vulkan     3008    4086     7755       8859        9037       2.61 / 0.96 / 3.09 s
CUDA       1895    1959     2375       2375        2471       2.33 / 0.67 / 2.47 s
```

Load: Vulkan selects F32 for Parakeet, CUDA BF16 (~1.1 GB of the 0.6B's weights). The rest (~5.6 GB) comes with
the first long ASR request and stays after it. Next: split Vulkan's held memory into live buffers and the pool's
idle ones.

## Run 2 - 2026-11-08 11:00

Question: what holds Vulkan's memory? Temporary counters (held device bytes, the pool's idle bytes and classes,
sampled at allocations; removed):

```
after load: held 2785 MiB (weights and context), idle 0
after the first long ASR: held 8708, idle ~5000: classes 128 MiB x9, 32 x30, 40 x19, 10 x30, 96 x3, 112 x2, 5 x48
```

Allocations of 128 MiB and up, by caller: the encoder's attention blocks (softmax, the adds, SiLU), not one large
buffer. During the ASR request held sits at ~7.5 GB with 3.4-4.2 GB of it idle in the pool: the request keeps ~1 GB
of temporaries live beyond the weights; the rest is pooled buffers of other classes. Each phase (subsampling,
encoder, decoder, the diarizer) takes its own classes and the pool keeps every one, so it approaches the sum of the
phases' working sets. CUDA's allocator splits and merges blocks.

## Run 3 - 2026-11-08 11:20

Changes:
- `release_idle` (the engine's idle hook) frees a discrete GPU's idle pooled buffers too, after synchronizing so the
  last request's temporaries are idle.
- A pooled device allocation that misses frees idle buffers none has taken for `STALE_IDLE` first (an earlier phase's
  classes), then makes room as before.

```
nvidia-smi peak MiB (idle 1192), asr / diar / both s, words against the CPU:
master                   10242   2.54 / 0.95 / 3.01
release_idle only         ~8900  (after a request back to ~4.9-5.2 GB; the peak unchanged)
+ stale after 500 ms      9093   2.57 / 0.98 / 3.14-4.28; held still rises to 7.4 GB at the subsampling-encoder hand-off
+ stale after 150 ms      8067, 8567   2.63 / 1.01-1.03 / 4.35-4.62; 0 word diffs
CUDA                      3631   2.31 / 0.69 / 2.54
```

Mid-request the subsampling's classes (40, 16, 10 MiB) were still idle when the encoder's working set allocated.
What is left against CUDA: Parakeet's weights in F32 (CUDA BF16, ~1.1 GB), and a request's working set of ~4 GB
against CUDA's ~0.5 GB: Vulkan keeps a batch's temporaries until its fence (up to 3 batches in flight), and buffers
come back to the pool in bulk as batches finish, so the pool also holds those returned and not yet reused. CUDA's
allocator reuses a freed block in stream order at once.

## Run 4 - 2026-11-08 11:50

Question: do the two pool changes cost the LLMs anything? The pool is every Vulkan model's.
`inference bench --vulkan --prompt-len 2048 --gen-len 32`, prefill / decode tok/s (Run 12 of
`vulkan_qwen_investigation.md`: 0.8B 9906 / 234, 35B 2969 / ~75, 27B ~495 / 30):

```
as in Run 3:            0.8B 9060 / 229-235   35B 2872-2877 / 78-80   27B 476-480 / 30
stale past 1 GiB idle:  0.8B 9094-9175 / 217-233
+ idle release keeping 1 GiB:  0.8B 9806, 9835 / 234
```

Dead end: freeing every idle buffer when the engine goes idle made each prompt after a pause reallocate its whole
working set (the bench pauses between repetitions: ~18 ms a 2048-token prompt on the 0.8B). Both rules now keep 1
GiB idle (`STALE_KEEP_BYTES`): a discrete GPU's idle release frees what is past it, largest first, and a missed
allocation frees stale buffers only past it.

```
speech, 10-minute clip: nvidia-smi peak 8404 / 8559 MiB (master 10242); asr 2.61 / 2.60 s, diar 1.01 / 1.03 s,
both 4.48 / 4.33 s; 0 word diffs. After each request: 3029 / 4128 / 3449 MiB above idle (master 7755 / 8859 / 9037)
```

The combined request read 3.0-4.3 s on master and 4.3-4.5 here: within master's spread, noted. Tests:
`an_idle_release_empties_the_pool` (all of an iGPU's, a discrete GPU's past 1 GiB),
`a_missed_allocation_frees_stale_idle_buffers` (a stale class kept under 1 GiB idle, freed past it, a fresh one kept).
