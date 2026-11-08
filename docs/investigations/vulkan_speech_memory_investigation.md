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

## Run 5 - 2026-11-08 12:30

Question: can the models #866 measured to hold under bf16 matmul inputs (`bf16_matmuls`: Parakeet's offline
encoder, the Nemotron-3 diarizer) keep their weights in f16 on a discrete GPU? `matmul_f32_cm` already rounds both
operands to f16 into its shared tiles, so an f16 weight there gives the same products.

- `matmul_f32_cm_bh`: the matrix-core matmul with b stored as f16. `Linear::to_half` keeps a weight in f16 on a
  discrete GPU with matrix cores, and `Linear::forward` sends f32 activations against it there, other shapes
  through a widened copy; the transcription pipeline halves the flagged models' encoder projections after load.
- First version: the kernel widened b's f16 to f32 in registers and back to f16 for the shared tile, ~40% slower
  (7564 x 4096 x 1024: 2.50 ms against 1.77). b now stays f16 from load to tile: 1.62-1.68 ms; the diarizer's shapes
  (684 x 512 x 2048 and the rest) level with f32. Dead end on the way: loading uvec2 words and unpacking, no change.
- The widened copy took a host round trip: Vulkan's `to_dtype` ran every dtype change on the CPU, and diarization
  read 1.4 s in about half the runs (1.0 with f32 weights). f16 to f32 now runs on the device (`widen_f16`), and f32
  to f16 too where the matrix-core kernels are (`convert_f16`).
- Tests: `a_half_weight_linear_matches_its_f32_layer` (300 rows on the f16 kernel, 3 through the widened copy, with
  a bias; a wrong b stride fails it), `vulkan_half_conversions_match_the_cpu` (odd count, sliced offset, both ways,
  bit for bit).

```
10-minute clip: load 3658 MiB (was 4167-4200); asr 2.61-2.86 s, diar 1.00-1.16 s, both 3.27-3.61 s (4.3-4.5 before);
peak 6413-6846 MiB (8404-8559 before); 0 word diffs; diarization 150 segments within 0.01 s of the CPU's
```

## Run 6 - 2026-11-08 13:00

Question: how much of the rest is temporaries that batches in flight hold? With one batch of 64 MB in flight
(temporary build): peak 5229 / 5402 MiB, asr 2.62 / 2.64 s, diar 1.01 / 1.02 s. A batch count of one would cost a
decode step's overlap, so the cap counts bytes instead: a submit waits for the oldest batch while the batches in
flight have allocated more than `MAX_IN_FLIGHT_BYTES` (and more than one is in flight).

```
256 MB: speech peak 5422-5747 MiB; 0.8B 9818 / 226; 35B 2913 / 78; 27B 342 / 27.2 (was ~480-500 / 30): its
  batches waited too
512 MB: speech peak 5816 / 6314 MiB, asr 2.67-2.73 s, diar 1.01-1.02 s; 27B 533 / 30.8, 498 / 29.2
```

Kept at 512 MB. Checks: lint, CPU (3153), CUDA (3356) and the Vulkan set pass.

Where it stands (MiB above the idle desktop): Vulkan ~4600-5100 peak, CUDA ~2440. Load 2466 against 1895, the rest
a request's working set against CUDA's ~550. Next: stream-ordered reuse of a dropped buffer by the work recorded
after it.

## Run 7 - 2026-11-08 13:40

Question: can a dropped buffer be reused by the work recorded after it, as CUDA's allocator reuses a block in stream
order, rather than once its batch's fence signals?

Checked first: every ordered dispatch ends with a compute barrier, copies and fills have barriers on both sides, and
a barrier's first scope is every command earlier in submission order (other command buffers included). The one
dispatch free to overlap the next (the MoE cm2 tile widths) runs inside one op ending in an ordered dispatch, so no
tensor drops between them. The CPU never maps a discrete GPU's device buffer (staging copies on the stream).

Change: a pooled allocation is an `Arc<Held>`, shared by its buffer, the pool while idle, and the batches whose
commands use it (`Recorder::retain` keeps a pooled buffer's `Held`, not the buffer), and freed when the last drops
it. A buffer goes back to the pool when its last user drops it; a device allocation is taken again at once, a mapped
one (host staging, an iGPU's) only once no batch holds it. Freeing room prefers allocations nothing else holds, and
synchronizes before the rest. Test: `a_dropped_buffer_is_reused_in_stream_order` (a buffer dropped with a copy from
it pending is the next of its size; that buffer's fill runs after the copy, which reads the old values).

```
10-minute clip: peak 4795 / 4794 / 4933 MiB (desktop 1232; Run 6 5816-6314); load 3233; asr 2.65-2.69 s, diar
1.00-1.02 s, both 3.23-3.33 s; 0 word diffs; diarization 150 segments, the CPU's speakers, within 0.01 s
greedy text (greedy.sh): 27B, 35B, 0.8B as before, word for word
0.8B, alternating with the old retention: stream 177.5 / 184.1 / 181.0 vs 183.6 / 180.0 / 173.6 tok/s;
  prefill 9688 / 9598 / 9584 vs 9578 / 9432 / 9399
27B 493 / 29.2, 35B 2890 / 75.9
```

Checks: lint, CPU (3153), CUDA (3356), `--vulkan` (63), and the Vulkan set three times (662) pass.

Where it stands, MiB above the idle desktop: after load 2001 (CUDA 1895), peak 3562-3701 (CUDA ~2440).

## Run 8 - 2026-11-08 (laptop: Radeon 780M iGPU, unified memory)

Question: Run 7's stream-ordered reuse skips an iGPU, whose buffers are all mapped host RAM: a dropped one waits for
its batches before anything takes it again. How far does the 780M's working set come down with it?

Baseline (master 2b0b1f9da, 10-minute meeting, GTT MB; idle ~240, loaded ~3020): asr peak 5409-5519, diar 4455-4505,
both models in one request 6573-6719. CUDA's request working set is ~0.5 GB; the 780M's was ~3.5 GB.

Change: a mapped pooled buffer is taken in stream order too, an idle one first. Where batches recorded before it was
taken may still hold it (`Buffer::early`), the CPU writes and fills it on the stream (a staging copy, a recorded fill)
rather than in place, and `mapped_ptr` gives none; once those batches finish it is written in place again. Staging
buffers and uploads (`Buffer::for_upload`: `with_bytes`, a tensor's bytes) take only idle ones, so an upload stays a
memcpy.

Bug on the way: a recorded fill or staging copy did not retain the buffer it wrote. A mapped buffer dropped before
such a command ran looked idle once the earlier batches finished, an upload took it and wrote it in place, and the
pending fill then overwrote the upload. Lightning's greedy text moved near the end of a 400-token reply, the same in
every run of one build, and a build whose `take` ran slower matched master (fewer buffers were taken early). Both
commands now retain the written buffer's allocation. Test: `a_pending_fill_keeps_its_buffer_from_an_upload` (fails
without the retain); `a_dropped_buffer_is_reused_in_stream_order` now runs on an iGPU too.

```
10-minute meeting, GTT MB peak (master -> stream order):
asr   5409-5519 -> 4256        diar 4455-4505 -> 3207-3227        both 6573-6719 -> 4400-4420
A/B, 3 rounds: asr 14.78 -> 14.78 s, diar 4.21 -> 4.02 s; transcript (1614 words) and diarization byte-identical
Lightning (Q5 experts, 2048 prompt): prefill 453 -> 469 tok/s, decode 24.9-25.1 -> 25.1-25.8 tok/s
Lightning greedy text, 5 prompts x 300 tokens: identical to master
```

Where it stands on the 780M, MB above idle: loaded ~2790, peak ~4180 with both models (a request's working set ~1.4
GB, was ~3.7).

## Run 9 - 2026-11-08 (laptop: Radeon 780M iGPU)

Question: Run 5 keeps the flagged models' encoder projections in f16 only on a discrete GPU. The 780M's f32 matmuls
run `matrix_cores`' kernels too (`matmul_f32` and `matmul_f32_half_b` take `context.coop` directly; only quantized
prompt matmuls keep off an iGPU's matrix cores, Run 38 of `vulkan_backend_investigation.md`), so they round to f16
already. Does the same hold there?

Check first: `INFERENCE_VULKAN_IGPU_COOPMAT=1` (the quantized paths on matrix cores too) gives the speech models'
outputs byte-identical, asr 14.20-14.26 -> 14.39-14.71 s: their matmuls were on the matrix cores either way.

Change: `vulkan::has_matrix_cores` (what `Linear::to_half` asks) holds for any Vulkan GPU with matrix cores.

```
10-minute meeting, GTT MB (Run 8 -> f16 weights), 2 alternating rounds:
loaded 3040 -> 1693; asr peak 4256 -> 2909; diar 3227 -> 1880; both 4404-4420 -> 3064-3069
asr 13.94-14.28 -> 14.00 s, diar 4.00-4.03 -> 3.85-3.88 s, both 17.82-18.17 -> 17.49-17.54 s
transcript and diarization byte-identical; six clips of 4-33 s (both models) byte-identical
```

Where it stands on the 780M, MB above idle (~250): loaded ~1440, peak ~2810 with both models (the 3090: CUDA 1895
and ~2440, Vulkan 2001 and 3562-3701).
