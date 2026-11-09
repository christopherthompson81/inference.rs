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

## Run 8 - 2026-11-08 17:30

Question: after #870-#875 the pair on the 3090's Vulkan still peaks highest (~3.6-3.7 GB above the desktop; the
780M 3.1, the laptop's CPU 2.7, CUDA 2.4). Load is 2.0 GB, level with CUDA's 1.9: what is the request's ~1.7 GB?

Temporary tracking (each live device buffer of 16 MiB and up with its caller, and the pool's idle classes, printed
at each new high of held memory; removed): at the peak, 3371 MiB held, 1327 of it idle in the pool, ~460 MiB of
large buffers live. Idle at the peak: 128 MiB x4, 112 x2, 96 x2, 2 x61 and smaller. The 112 and 96 MiB allocations
had been made beside idle 128s, the pool taking exact classes only.

Change: a device allocation that misses its class takes the smallest idle class up to 1.5x its size, which goes back
to its own class (`a_miss_takes_an_idle_buffer_a_little_larger`).
Dead end: lowering the idle keep (`STALE_KEEP_BYTES`) to 256 MiB raised the peak (4329 MiB) and cost the 0.8B's
prefill ~1.3 %; kept 1 GiB.

```
nvidia-smi peak (desktop 862 MiB), master c03913572 against this, alternated in one session:
master:    4564 / 4440 / 4564 MiB
best fit:  3906 / 4255 / 4045 MiB   (3.04-3.39 GB above the desktop, from 3.58-3.70)
asr 2.55-2.65 s, diar 0.94-0.98 s; outputs byte-identical to master's; Qwen greedy text identical;
0.8B 9789 / 236, 27B 434 / 30.5, 35B 2966 / 77 (as before)
```

Checks: lint, CPU (3162), CUDA (3365), `--vulkan` (65), the Vulkan set twice (673) pass. The peak varies ~0.35 GB
run to run with the phases' timing; the rest is the pool's 1 GiB idle keep and small classes (2-5 MiB).

## Run 10 - 2026-11-08 22:00

Question: the pair on the 3090's Vulkan still peaks above CUDA (`phase.sh`: nvidia-smi every 50 ms, the max per phase,
MiB above the idle desktop). What is left?

```
                     load    loaded  asr     diar    both    times (asr / diar / both)
CUDA (bf16 model)    1896    1960    2376    2376    2408    2.31 / 0.66 / 2.53 s
Vulkan master        2603    2303    2969-3008  2597-2655  3044   2.52-2.59 / 0.92-0.95 / 3.09-3.12 s
```

Temporary tracking (held device bytes at each new high, the pool's idle classes, a live census by class; removed):

- Load: the high is the halving pass (Run 5's `Linear::to_half`), not a request: every f32 projection uploaded,
  narrowed to f16, and its f32 source dropped into the pool: 3289 MiB held with 2030 idle (16 MiB x96, 4 MiB x120)
  at the end. The idle engine's release keeps 1 GiB of it, and ~470 MiB of 4 and 2 MiB classes stay idle into the
  requests, sizes nothing takes.
- Requests: at the ASR peak ~2.67-2.95 GB held: ~1.2 GB f16 weights, ~1.0 GB live activations (31 MB hidden-state
  tensors x9, the attention blocks' 128 and 64 MiB scores, 124 MB feed-forward), 0.5-0.9 GB idle (128 MiB x4 from
  the previous block shape while the next allocates 32/64s).
- CUDA's working set is ~0.5 GB because it runs the whole model in bf16 (`DType selected is BF16`): its activations
  are half the size. Vulkan selects f32 (`determine_auto_dtype_all`: its kernels take f32 activations).

Tried (each A/B'd in the same session):

```
free the f32 sources during halving (Linear::halve: convert, replace, Context::free_idle), 3 rounds vs master:
  loaded 2303 -> 1912, load spike 2603-3467 -> 2577-2603, but asr 2969-3008 -> 3097, both 3044 -> 3141
  (the requests had been reusing the leftover idle classes); end state equal (1789-1808)
+ STALE_KEEP_BYTES 256 MiB:            asr peak 2843 held, smi peak 3108-3174 (no clear change)
+ STALE_IDLE 30 ms:                    smi peak 3000; asr 2.73 s (noise or cost, not separated)
+ FIT_RATIO 2x (instead of 1.5x):      worse, 3075-3084
QUERY_BLOCK 256 (instead of 512):      peak 3018-3033, asr 2.85-2.89 s (+12 %)
```

None kept. The fused `flash_attn_relpos_cm` would drop the score tensors but is ~2x slower on the 3090
(`vulkan_relpos_3090_investigation.md`, Run 1).

Where it stands: Vulkan ~3.0-3.1 GB above the desktop at peak, CUDA 2.4. The gap is f32 activations (~0.5 GB) and
the pool's idle size classes (~0.5-0.7 GB). Closing it needs f16 activations on Vulkan (its elementwise, norm,
softmax and conv kernels) or a pool that splits large idle allocations for smaller requests, as CUDA's does.

## Run 11 - 2026-11-08 23:30

Question: Run 10 put the rest of the gap on f32 activations. Can Vulkan run the pair in bf16, as CUDA's auto does?

Survey (`dtype = "bf16"` on both models, a counter in the CPU fallbacks, a 20 s clip): every bf16 op but index_select
went to the CPU (f32 to bf16 conversions 1120, binary ops 491, strided copies 472, matmuls 190, unary 143, affine 55,
the convs); the ASR request failed outright (the CPU's matmul takes no bf16). The f32 run's own fallbacks: one conv1d
(the diarizer's upsampling, a call a step) and one reduction.

Change: bf16 storage computed in f32 (`bf16.glsl`: widening is a shift, narrowing rounds to nearest even). Each kernel's
body moved to a `*_main.glsl` with element macros, its f32 build byte-identical SPIR-V (checked for map, rows, conv2d,
matmul, matmul_dot, matmul_cm and matmul_cm_bh), and a bf16 build beside it: map_bf16, rows_bf16 (an arg reduction
writes its u32 as two halves), conv2d_bf16, matmul_bf16 / matmul_bf16_dot (the products below the matrix-core tiles,
a decode step's), and matmul_bf16_cm (four bf16s ride in a uvec2, as many registers as the f16 build's f16vec4,
rounded to f16 at the tile store as the f32 build's inputs are; c through the staging tile). Plus copy_b16 (any 2-byte
element, strided, and fills) and convert_bf16. On the way: conv1d on the device as one row of conv2d (its padding split
into h and w), and a reduction over a middle dim moved last and copied contiguous (both f32 and bf16; both had gone to
the CPU). `auto` on a discrete Vulkan GPU with matrix cores is bf16 for the speech loaders (`TryIntoDType::is_auto`).
Test: `vulkan_bf16_ops_match_the_cpu` (conversions and copies bit for bit, ties included; maps and rows within a bf16
rounding; products at tile and fallback sizes, conv2d, argmax, middle-dim sums); conv1d in `vulkan_conv2d_matches_the_cpu`.

First build: peak level with CUDA but asr 2.71-2.76 s (f32 2.52-2.59). Profile over two 10-minute requests: GPU time
3414 ms against f32's 3361, `convert_bf16` 443 ms in 53,544 dispatches: the decoder's small products and argmax had
taken the widen-to-f32 fallback. With matmul_bf16 / _dot and the bf16 argmax: 3219 ms (matmul_bf16_cm 1563, map 723,
dot 336, copy 290, rows 245; f32: matmul 1539, map 717, rows 457, copy 340, dot 270).

```
phase.sh, MiB above the desktop, three rounds alternating with master (f32)
                 load    loaded  asr     diar    both    times (asr / diar / both)
master f32       2603-3467  2303  2969   2626-2793  3182   2.56-2.59 / 0.96-0.99 / 4.29-4.36 s
bf16             1625    1756    2349    2357    2370-2378  2.28-2.32 / 0.83-0.87 / 3.22-3.48 s
CUDA (bf16)      1896    1960    2376    2376    2408   2.31 / 0.66 / 2.53 s (Run 10)
```

Words (punctuation counted) and diarization frames (10 ms), seven clips, against Vulkan f32 and CUDA:

```
             bf16 vs f32                                        CUDA vs f32                          bf16 vs CUDA
short20/45/90  0 diffs; 1 / 0 / 6 frames                         0 diffs; 10 / 0 / 5 frames           0 diffs
var_gain     fifteen -> fifty, storm bitten -> storm-bitten,     fifteen -> fifty, night, log ->      2 diffs, 8 frames
             night, log -> night. Log; 14 frames                 night. Log (twice); 12 frames
var_noise    fifty -> fifteen; 37 frames                         fifty -> fifteen; 48 frames          0 diffs, 27 frames
var_tempo    0 diffs; 6 frames                                   shone, wind -> shone. Wind; 7        1 diff, 3 frames
clip10       weather, wind -> weather. Wind; 11 frames           0 diffs; 12 frames                   1 diff, 11 frames
```

The fifteen/fifty near-ties are `vulkan_relpos_3090_investigation.md`'s; bf16 on Vulkan moves about what CUDA's bf16
moves. Qwen 0.8B's greedy text on Vulkan (f32 activations) is unchanged.

Where it stands: the pair's peak on the 3090's Vulkan is level with CUDA's (2.37 against 2.41 GB above the desktop,
from 3.18), and ASR and diarization run faster than in f32. Not done: the 780M (an iGPU stays f32; its fused relpos
attention is f32-only), bf16 flash/relpos kernels, and LLMs (their fused kernels take f32 activations).

## Run 12 - 2026-11-09 00:30

Question: the combined request (Parakeet with `diarization_model`, the two engines run at once) looked to gain nothing
on the 3090's Vulkan (3.2-3.5 s against asr 2.30 + diar 0.85), where CUDA's takes 2.53 against 2.97. Would a stream
(queue) per model give Vulkan CUDA's overlap?

Tried (branch `vulkan-streams`, local, not for merge): a stream per queue (up to 4: stream 0 for loads and unscoped
work, the transcription pipeline's forward on its own), timeline semaphores between them, and each allocation's
producing batch plus the streams that used it (`Usage`), so a reader on another stream waits for the writer and the
pool hands another stream's buffer out only once idle. First cut tracked the last use of any kind: two streams
reading one buffer waited on each other's open batches and deadlocked (`two_streams_record_at_once` hung); the
producer-only tracking passed it 15 times. The idle engine's release had synchronized every stream, the other model's
included (now skipped while another stream works). Queues of one family, and the compute-only family beside the
main one: no difference.

Within one combined request (forward start and end per model): diar 1.2 s (0.68 alone), asr 3.65 s (2.17 alone); the
kernel time summed over the profile rises ~30 % when both queues run. The GPU shares out rather than fills gaps.

Measured again with repeats, the premise was wrong: the first runs had timed a server's first combined request.

```
clip10, a server each, five requests of each kind (seconds)
                    asr                       diar          both
master (one stream) 2.24-2.49 (3.37 the 2nd)  0.83-0.85     2.66-2.85
streams             2.26-2.55 (3.1-3.3 2nd)   0.84-0.88     2.84-3.05
cold start, asr and diar once, then both five times:
Vulkan master       2.23-2.26                 0.83-0.85     3.46-3.80, then 2.78-2.91
CUDA                2.24-2.26                 0.65          2.47-2.51 (no first-request cost)
```

One stream already overlaps the two models: both 2.8 s against 3.1-3.2 s in turn, the ~0.4 s CUDA saves. The streams
cost ~0.15 s more and stay unmerged. What is left against CUDA: the diarizer's own time (0.84 against 0.65 s) and the
first combined request's extra 0.6-0.9 s, which is not allocation (that request allocates ~20 new device buffers,
~10 ms); the second asr request after a start is ~1 s slower on both builds too.

## Run 13 - 2026-11-09 02:40

Question: what is the first combined request's extra 0.6-0.9 s on the 3090's Vulkan (Run 12), and the second asr
request's ~1 s?

Both are one event, and it is not ours. Per-batch GPU timestamps (temporary print in the profiler) over eight
back-to-back clip10 asr requests: the encoder's FF matmul (bf16 matrix cores, 7564 x 4096 x 1024) runs 1.7 ms, then
from one dispatch to the next 44 ms, then 30, 16.7, 3.6 over ~0.5-0.8 s (every matrix-core dispatch alike, element-wise
ones unchanged at 0.08 ms), then 1.7 again. It lands in the third request about 1.6-1.8 s in on most starts; a 20 s idle
start moves nothing, a 60 s pause after two requests moved it to the fourth after the pause, and later requests never
repeat it (12 in a row: only the third slow). CUDA, 12 in a row: 2.29-2.36 s, none slow.

Ruled out, each by measurement:
- allocation: no device allocation or free during any request (logged every one); no spill to host memory
- buffers: the slow and fast dispatches bind the same three buffers, offsets and strides (vec4 load path either way)
- clocks/heat: nvidia-smi 1845-1860 MHz, memory 9751 MHz, P0 through it; recurring heat would repeat, it does not
- another context: nsys GPU context-switch trace shows only our context over the window; CUDA hidden from the
  process (CUDA_VISIBLE_DEVICES=) still slow
- data: all eight transcripts byte-identical; without matrix cores (INFERENCE_VULKAN_NO_COOPMAT) no slow window
  (but 5.0 s a request)

nsys GPU metrics (10 kHz) over the window: GPC clock 1844 MHz, SMs active ~90 %, but SM issue 38 -> 2-12 %, tensor
11 -> 1 %, DRAM writes 14 -> 55 % and reads 18 -> 29 % of peak, copy engines idle, PCIe idle. The SMs are resident and
stalled on memory while something writes at ~0.5 TB/s that is not our work.

Reproduced without the model: an ignored test looping the two FF matmuls on fixed tensors (bf16, pooled outputs, no
other kernels): 1.64 ms each for ~12 s, then one 38 / 16.7 / 3.6 ms stretch, then 1.26 ms for good (23 % faster than
before, and past what f32 accumulation on this card should give). 20000 tiny matrix-core dispatches first moved it to
9 s, so it is not a plain dispatch count. Driver 615.71.10. No public report of it found.

Implication: a driver-side change to the matrix-core pipeline after sustained use (a slow, write-heavy transition,
then a faster binary). Run 12's "first combined request" and "second asr" costs are this one-time event landing
there, and the combined request's steady state was already level with in-turn plus CUDA's overlap gain. Left: whether
a load-time warm-up can trigger it before serving (it costs ~0.5-1 s of GPU once either way), and whether the faster
binary changes the products (transcripts identical across it).
