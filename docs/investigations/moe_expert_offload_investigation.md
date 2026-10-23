# MoE expert offload: CPU experts plus a hot VRAM cache

Goal: run a MoE whose experts do not fit in VRAM by keeping the expert weights in host memory, caching the hot ones
on the GPU, and computing the rest on the CPU. Test model: Qwen3.6 35B A3B UD-IQ4_XS GGUF (16.95 GiB; 40 MoE layers
plus the nextn block, 256 experts, 8 routed + 1 shared, hidden 2048, expert FFN 512). RTX 3090 on PCIe gen3 x16,
i7-10700K (8 cores, AVX2, no AVX-512), dual-channel DDR4, 125 GiB RAM.

## Run 1 - 2026-10-20 14:25

Question: where do the bytes sit, and what does llama.cpp's CPU-expert mode reach on this machine (a ceiling for the
CPU side, same file, same hardware)?

Experts are 14.4 of the 16.95 GiB: gate and up are IQ3_S (39 layers; one IQ4_XS, one Q3_K), down is IQ4_XS (37; three
Q6_K, one Q4_K). One expert is about 1.4 MB, one layer's 256 experts about 360 MB, and a decode token's 8 x 40 picks
about 450 MB.

`llama-bench -m <file> -ngl 99 -ncmoe N -p 512 -n 128 -r 2 -t 8` (llama.cpp 8bb3cf0d2, build_cuda):

```
experts on CPU     pp512 tok/s    tg128 tok/s
none               3676           146.4
20 of 40 layers    744            53.7
all 40 layers      448            32.3
```

All-CPU decode spends about 26 ms a token beyond the GPU part, about 17 GB/s of expert reads. Its prefill streams the
expert weights to the GPU for batches (14.4 GiB over gen3 for 512 tokens is about 1.2 s, which is the 448).

## Run 2 - 2026-10-20 14:30

Question: how skewed and how local is routing, i.e. what hit rate does a VRAM cache of C experts per layer give?

Temporary trace in `MoEExperts::forward` (each forward's top-k ids per layer to a file, graphs off), on our server with
this file: the Code tab demo through `inference agent`, then two 1500-token chat replies (essay, Rust code). 536,560
rows, 252,080 of them decode rows (a layer forward of one token). `moe_cache_sim.py` replays the trace with the cache
updated after every token:

```
experts per layer     oracle static top-C    LRU      LFU (decay 0.98)   LFU (decay 0.999)
32  (1.8 GiB)         0.380                  0.600    0.589              0.447
64  (3.6 GiB)         0.585                  0.770    0.774              0.644
128 (7.2 GiB)         0.832                  0.917    0.921              0.862
192 (10.8 GiB)        0.956                  0.976    0.976              0.965
```

Locality is strong and short-lived: a cache that follows recent use beats the best static set by 15-20 points, and
slow-decay LFU loses most of that. Rough decode estimate with misses on the CPU at Run 1's 17 GB/s plus the 9.5 ms GPU
step: 64 per layer about 63 tok/s, 128 about 87 tok/s (llama.cpp's layer split at the same 7 GiB: 54).

Plan: decode misses on the CPU (RAM bandwidth, separate from PCIe), hits on the GPU from a slot cache filled over PCIe
on a side stream; prefill streams uncached experts to the GPU as llama.cpp does. First piece: CPU dot products for the
IQ types (today the raw GGUF CPU path dequantizes them), useful on its own for CPU inference of IQ files.

## Run 3 - 2026-10-20 15:05

Question: can our own CPU kernels match ggml-cpu's expert rate on this machine?

New `gguf/cpu_dot.rs` (IQ3_S and IQ4_XS against Q8_K activations, ported from ggml-cpu's generic and AVX2
`ggml_vec_dot_*_q8_K`; the k-quants reuse the tensor crate's) and `gguf/cpu_experts.rs` (gate and up, GLU, Q8_K
again, down, then the weighted sum, rayon over 32-row tasks). Both match dequantized references. Benchmark: the 35B
layer shape over 256 random experts (374 MB, so weights come from RAM), 8 picks a call, debug assertions off.

```
                                   one thread IQ3_S   one thread IQ4_XS   a layer (8 picks)   rate
first port                         0.68 GB/s          3.67 GB/s           1866 us             6.3 GB/s
IQ3_S grid lookups on fixed arrays 3.55 GB/s          3.51 GB/s           640 us (16 thr)     18.2 GB/s
                                                                          690 us (8 thr)      16.9 GB/s
```

The first IQ3_S port spent its time on bounds-checked slice indexing for the 8 grid lookups per 32 values; with the
qs / qh / signs bytes taken as fixed-size arrays and the 9-bit index masked, it matches IQ4_XS. 18 GB/s is llama.cpp's
rate (Run 1), so the CPU side of a miss costs about 80 us an expert.

## Run 4 - 2026-10-20 15:40

Question: wired into the MoE layer, does the CPU expert path hold up on the real checkpoint, and what does it give CPU
inference on its own?

`FastExpertsWeights::forward_cpu_blocks`: when the three projections expose their GGUF blocks in host memory (new
`QuantMethod::cpu_gguf_blocks`, for k-quant QTensors and raw IQ tensors) and every type has a CPU dot product, the
layer runs `CpuExperts` instead of the gather path, which dequantized each touched expert per call.
`inference bench --cpu --format gguf -f <file> --prompt-len 64 --gen-len 32` (35B A3B, all on the CPU):

```
                         prompt 64 tok/s    decode tok/s
gather (dequantize)      2.6                0.5
CPU expert blocks        13.6               5.1
llama.cpp -ngl 0 -t 8    60.2               9.35
```

Greedy chat reply ("capital of France ... river", thinking off) is identical on both paths. Experts are now about 26
ms of the 196 ms CPU token; the rest (attention, GDN, shared experts, lm head on the CPU) is where llama.cpp leads,
and offload keeps all of that on the GPU, so it is not pursued here.

## Run 5 - 2026-10-20 16:00

Question: Run 4 left a CPU token at about 196 ms against llama.cpp's 107. Where is the rest going?

Temporary timers (wall time per part, decode-dominated `bench --cpu --prompt-len 8 --gen-len 128`):

```
gdn 72 ms (project 38, recurrence 18, finish 11.7, conv 4) | experts 39 | router 14.6 | attn 13 | lm_head 14 | shared 6
```

Isolated matvec benchmarks put the dense Q8_0 / Q6_K / Q4_K decode shapes at 20-26 GB/s, so those layers were near
bandwidth; the outliers were the router (F32 2048 x 256 at 6 GB/s), the experts (0.95 ms a layer against 0.64 in
isolation) and the GDN recurrence (0.6 ms a layer for about 4 MB of state).

Findings, in the order they showed up:

- The CPU expert path ran on rayon while the quantized matmuls run on the tensor crate's barrier pool, whose workers
  spin for a while after each call; the two pools fought over the same 8 cores. Moving the expert rows onto the barrier
  pool: experts 39 -> 24 ms.
- The F32 router stayed slow, and the isolated benchmark swung between 75 and 330 us with nothing but whether
  `RAYON_NUM_THREADS` was set. With it unset, `get_num_threads()` falls back to `num_cpus::get_physical()`, which reads
  /proc and /sys on every call, and every CPU F32 matmul asks: about 200 us per matmul. Cached in a OnceLock: the router
  2048 x 256 went 277 -> 73 us.
- `gemm`'s single-row path then reached 16 GB/s; a threaded column-split F32 matvec (barrier pool, 16-lane dot) reaches
  28.6 GB/s, so it stays for single-row matmuls past 256K weights.

```
                                 decode tok/s   prompt 64 tok/s   router   experts   gdn
Run 4                            5.1            13.6              14.6     39        72
pool + thread count + F32 gemv   7.2            17.0              3.3      24        64
llama.cpp -ngl 0 -t 8            9.35           60.2
```

Left: the GDN recurrence (18 ms, about 3x its traffic), and prefill, where the expert path reads each expert's rows
once per routed token instead of once per expert.

## Run 6 - 2026-10-20 17:10

Question: what holds CPU prefill at 15.7 tok/s (512 tokens in 32.6 s) against llama.cpp's 60?

Per-forward timers on a 512-token prompt: GDN 15.4 s (projection 6.5, conv 3.2, recurrence 3.7, output 2.0), experts
13.2 s, attention 2.7 s, shared expert 1.2 s. In turn:

- Experts read each expert row once per routed token. Grouping the routes by expert and dotting each row against four
  tokens per pass (hand-written AVX2 x4 kernels; a const-generic version compiled to slower code than the one-token
  kernel, 3.0 vs 4.05 GB/s a token) gives IQ3_S 5.7 and IQ4_XS 7.8 GB/s a token on one thread: experts 13.2 -> 9.0 s.
- The GDN conv's fused CPU kernel was fine; around it the input was transposed to [b, conv, seq], concatenated with the
  state twice and handed on as a transposed view. A conv straight over [b, seq, conv] (channels innermost, barrier pool)
  that builds the next state directly: conv 3.3 -> 0.25 s, and the recurrence after it 3.7 -> 1.3 s, as it no longer
  copies a strided input.
- Q8_0 GEMM at 512 rows ran at 81-123 GFLOP/s against Q4_K / Q6_K's 500: the tiled x86 kernels skipped Q8_0 on pure
  AVX2 because its +128 weights overflow maddubs' i16 pairs. Moving the weight's sign onto the activation (|w| <= 128
  unsigned, Q8_K activations within +-127, as ggml's mul_add_epi8 does) keeps the packed layout and fits: 401-444
  GFLOP/s, so GDN projection 6.3 -> 1.1 s, output 2.0 -> 0.5, attention 2.7 -> 0.8, shared expert 1.2 -> 0.25. The
  tiled kernels had no test; one now compares Q4_K / Q6_K / Q8_0 at 1, 3, 9 and 33 rows with a dequantized matmul.

```
bench --cpu                      prompt 64 tok/s   prompt 512 tok/s   decode tok/s
Run 4                            13.6              -                  5.1
Run 5                            17.0              15.7               7.2
this run                         34.0              36.3               7.3
llama.cpp -ngl 0 -t 8            60.2              -                  9.35
```

A 512-token forward is now experts 9.05 s of about 14; the x4 kernels reach about 112 GOPS against tiled Q8_0's 400,
so the next step there is decoding an expert's rows once into a cache-sized int8 tile and running the group's tokens
over it.

## Run 7 - 2026-10-20 17:40

Question: the decode recurrence took 0.6 ms a layer for about 4 MB of state; why?

`decode_recurrence_cpu_from_convolved` ran on rayon and looped over the value columns with the key rows innermost, a
128-float stride through each head's state. Row-wise passes (decay and read k^T S, then the rank-1 update and q
readout, as the prefill scan does) on the barrier pool: recurrence 17.7 -> 5.5 ms a token, decode 7.3 -> 8.0 tok/s
(125.8 ms a token against llama.cpp's 107).

The remaining decode parts run near bandwidth (GDN projections 30.6 ms for about 780 MB, experts 25.5, LM head 14).
Full lint and CPU / CUDA suites pass. Two recorded-logit tests moved by design: the x86 Qwen3.5 MoE BF16 snapshot
(the conv now accumulates in F32, where the BF16 tensor path rounded every tap) and the Gemma3 Q8_0 recording, which
gains a plain-AVX2 case (prompt rows tiled with per-256 activation scales, decode rows on the per-32 gemv). The
aarch64 BF16 snapshot will move with the conv too and needs recording on a Mac.

## Run 8 - 2026-10-20 19:30

Question: does a first offloaded backend (experts in pinned host memory, an LRU slot cache per layer on the GPU, misses
on the CPU) hold quality, and what does it reach?

Prototype behind `INFERENCE_RS_MOE_OFFLOAD_SLOTS=<slots per layer>` (CUDA graphs off, as every offloaded layer syncs
with the host). Decode (up to 8 tokens): the top-k ids and weights come to the host, cached experts run through the
indexed matvec over the slot stack with remapped ids (misses get weight 0), the CPU runs the misses at the same time,
and each missed expert is copied into the least recently used free slot on a side stream after the work queued so
far; a slot serves once its copy's event completes. Larger batches stage the layer's whole expert stack into one
buffer the layers share and run the grouped kernels.

First perplexity check (`examples/perplexity --gguf <35B> --llama-cpp-ctx 512`, 14 KB of repo docs; the split path
forced for whole prompts so it is the one scored):

```
resident                      1.5454
staged prompt path            1.5454 (same kernels)
all experts on the CPU        1.5524
64 slots, split forced        1.8745   <- wrong
256 slots, split forced       1.6842   <- wrong
```

A temporary check ran each forward's cache hits on the CPU as well: relative differences up to 957. The misses' picks
pointed at slot 0 with weight 0, and slot 0 could be mid-copy for an admission, so a half-written block's scale could
be non-finite and 0 times it is not 0. Slot 0 now stays zeroed and is never a victim (one slot more is allocated):
hits agree with the CPU to a median 2.5% (the two activation quantizations), and both 64 and 256 slots score 1.5480
(+0.17% over resident).

`inference bench --prompt-len 512 --gen-len 128` (graphs off for the offloaded rows):

```
                                   decode tok/s   prompt 512 tok/s
resident, graphs on                158.3          4180
resident, graphs off               95.0           4157
offload, 256 slots                 78.1           394
offload, 128 slots (7.2 GiB)       70.0           395
offload, 64 slots (3.6 GiB)        60.1           394
offload, 0 slots (all CPU)         25.5           392
llama.cpp -ncmoe 20 (about 7 GiB)  53.7           744
llama.cpp -ncmoe 40                32.3           448
```

The prompt rate is the PCIe copy of every layer's full expert stack (about 14 GB a forward over gen3). Next: skip
copying experts a layer already caches and overlap the next layer's copy with the current layer's compute; trim the
per-layer host round trips (the 256-slot row is 2.3 ms a token behind resident-without-graphs); then the real flag and the
device-map planner.

## Run 9 - 2026-10-20 19:50

Question: graphs gave resident decode 6.3 ms a token against 10.5 without; the CPU split rules them out, as every
offloaded layer waits on the host. Would the GPU reading missed experts straight from host memory (no host sync, so
graphs stay) do better?

The host expert stacks are now mapped pinned memory (`alloc_mapped_host`, with a new `CudaSlice::mapped_host_ptr` in
the vendored cudarc so the CPU reads the same bytes), which is also a `KernelWeight`. A temporary switch ran every
expert of every layer through the indexed matvec straight from host memory:

```
all experts read over PCIe, graphs off   18.2 tok/s (54.9 ms)
all experts read over PCIe, graphs on    19.5 tok/s (51.3 ms)
all experts on the CPU, graphs off       25.5 tok/s (39.3 ms)
```

About 45 ms for a token's 450 MB of experts: 10 GB/s of zero-copy reads over gen3, about the CPU's rate. With a cache,
a miss read over the bus still has to be copied in to be cached, so PCIe carries every miss twice; the CPU split keeps
the bus for fills only. Projected at 128 slots (8% misses): about 100 tok/s zero-copy with graphs against 70 measured
for the split, at 64 slots about even, both before counting the fills' share of the bus.

The split keeps its place. Getting graphs back for it means capturing the host side too: async copies into pinned
buffers, the hit/miss mapping and CPU experts as a host function node (no CUDA calls inside), and cache fills issued
between steps with a host-function flag marking when they land. That is a later phase; first the flag, the planner
and the prompt-side copies.

## Run 10 - 2026-10-20 21:10

Question: can a decode step with offloaded experts stay in a CUDA graph, and does that recover the graphs' share?

#524 synced with the host in every offloaded layer, so it ruled graphs out. Now nothing waits on the host: the picks,
weights and hidden rows go to mapped pinned buffers with async copies, a host function (`cuLaunchHostFunc`, captured
like any stream op, no CUDA calls inside) runs the CPU side, and its result comes back with an async copy. Fills
moved to one worker thread (host functions may not call CUDA); a second host function queued on the copy stream after
a fill opens the slot.

```
serve, greedy 400-token replies     128 slots   64 slots
#524 (graphs off)                   72-78       52-58
host function in the step           84-86       51-58
GPU slot lookup, CPU on a side      98          52-63
resident                            180-185
```

With the host function inline the GPU idled through it: per layer the host function took 68 us and the gap between
two layers' host functions was 234 us, against about 137 us for a resident layer. The slot lookup moved to the GPU:
a table in mapped host memory maps each expert to its slot or 0, the zeroed slot, so the cached experts run on the
stream straight after routing with the original weights (a miss adds exactly 0), while a side stream copies the picks
out, runs the misses in a host function and copies their sum back; the stream waits for it only to add the two. The
host function now reads which picks the GPU served from the copied slot ids, so the two sides cannot both serve a
pick, and evictions clear the table before their copies start.

Greedy replies are coherent but differ between runs at the same seed (two graph runs part at character 99): whether a
pick is served by the cache or the CPU depends on when its fill lands, and the two quantize activations differently.

## Run 11 - 2026-10-20 22:00

Question: prompts copied every layer's whole expert stack (about 30 ms a layer over gen3) whatever their length. What
should a prompt batch do instead?

`prompt_ttft.py` (raw completions, `max_tokens` 1, no prefix cache, best of three; first with a sixteen-word synthetic
text, then with real guide text):

```
synthetic text, 128 slots        73 tok     288 tok    758 tok
whole-layer staging              1196 ms    1260 ms    2514 ms
cache + CPU split for prompts    178 ms     263 ms     506 ms

real text                        73 tok     288 tok    758 tok    2648 tok
whole-layer staging (synthetic)  1196 ms    1260 ms    2514 ms    7629 ms (2965 tok)
split, 128 slots                 852 ms     1780 ms    2974 ms    8632 ms
split, 64 slots                  800 ms     2415 ms    6392 ms    30277 ms
sparse staging, 128 or 64 slots  403 ms     747 ms     1477 ms    5364 ms
```

The synthetic text routed to so few experts that 128 slots held nearly all of them, which flattered the split; real
text spreads over the experts, and the CPU's cost per routed token (about 57 us) makes it lose once an expert has more
than one token, as copying the expert over the bus takes about 127 us. So the split stays at decode sizes (up to 8
tokens), and a larger batch now reads its routes on the host (which also waits out any queued host function, so no
slot can be evicted under it), gathers only the experts it routes to into the shared staging (cached ones copied from
their slots, the rest over the bus, consecutive uncached ids in one copy) and runs the grouped kernels on that compact
stack with remapped ids. These prompts ran with no decoding between them, so the cache was cold and 128 and 64 slots
tie; with decode in between the cached share comes over at device speed.

## Run 12 - 2026-10-20 23:30

Question: CPU prefill experts ran at about 112 GOPS (9.4 s of a 15.9 s 512-token forward, 32.1 tok/s against
llama.cpp's 60). Does decoding each expert row once into a tile and running its tokens over it close that?

New `prefill_layer_rate` benchmark (one 35B-A3B layer's 256 experts, 512 tokens x 8 picks; `SKEW` piles routes onto
a few experts) and a temporary single-thread split of decode vs dot. In order:

```
                                                         uniform    skewed     whole model (bench --cpu, 512)
x4 kernels (before)                                      -          -          experts 9.37 s, 32.1 tok/s
i16 tiles (sub-block scale folded in, madd 2 rows x 4)   80 ms      162 ms     experts 4.6 s, 56.6 tok/s
busy experts split into spans of <= 32 routes            89         104
BlockQ8K::d()/qs() #[inline]                             81         -
quantize, widen, final sum on the pool; scratch reused   65         80
                                                                               experts 2.6 s, 78.6 tok/s
```

- Tile size (4-16 rows a work item, min tokens 4-16) moved prefill by 1-2%. Weights resident in L3 (8 experts) gave
  only 10%, and 1 / 4 / 8 threads scale 63 / 221 / 410 GOPS, so it is not memory-bound.
- The skewed case lost to load imbalance: the pool's chunked loop handed one thread a whole busy expert. Spans plus a
  shared cursor taking 16 items at a time fixed it. Taking items one at a time instead cost 25% on uniform routing,
  as neighbouring row tiles on different cores wrote into the same output cache lines.
- `BlockQ8K::d()` was a call through the GOT in every block (no `#[inline]`, no cross-crate inlining in dev), which
  spilled all eight accumulators around it. Inlining it doubled the existing x4 kernels too (IQ3_S 5.7 -> 11.5 GB/s a
  token on one thread).
- Gathering the IQ3_S grid with vector-built indices in place of eight scalar lookups changed nothing.
- On one thread the dot is 70% (64 GMAC/s, the isolated kernel's rate) and the tile decode 30%. A maddubs form would
  save only the adds; both forms issue two port-0/1 ops per 32 MACs.

Decode moved 8.5 -> 9.1 tok/s (109.6 ms a token; llama.cpp 9.35) with the inlined accessors. The "GDN recurrence/conv
prefill regression" noted at the end of Run 11 was the warm-up forward: steady forwards give recurrence 1.05-1.1 s and
conv 0.2 s, as in Run 6.

Now (ms of a 512-token forward): GDN projection 1000, conv 205, recurrence 1055, finish 465, experts 2540, attention
740, shared 215.

## Run 13 - 2026-10-21 00:20

Question: the GDN recurrence took 1.07 s of a 512-token forward for about 63 GFLOP; where does it go?

Temporary timers inside it: 244 ms narrowing, head expansion (`repeat`), gating and `l2_norm` as tensor ops, 271 ms
transposing to [b, h, t, d] and copying to vectors, 521 ms the scan, 37 ms back. The decode path already fused all of
that for one token (reading q / k / v straight from the convolved rows, normalizing and gating per head), so it now
loops each head over the prompt's tokens and serves every CPU length: recurrence 1073 -> 530 ms, prompt 78.6 -> 86.6
tok/s. Its test now checks 1 and 5 tokens against the tensor path. The Qwen3.5 MoE snapshots moved (F32 by 1e-4 of the
sum; BF16 more, as q / k normalization and gating no longer round through BF16) and were re-recorded; the aarch64 BF16
one will move too.

Dead ends on the expert dot (one thread, 64 GMAC/s, about 9 cycles a 16-value step where the ports allow about 5):
aligning the tile to 64 bytes, and token quads outermost (both within noise).

Now (ms of a 512-token forward): experts 2470, GDN projection 1020, attention 740, recurrence 530, finish 495, shared
215, conv 205.

## Run 14 - 2026-10-21 00:50

Question: a CPU decode token took 111 ms of wall time with 99 ms inside the timed sections; where are the other 12?

More timers: checking out each GDN layer's state took 5.3 ms a token and committing it 3.2 ms. On the CPU a checkout
gathered the pool row (index_select), the conv and recurrence copied it to a Vec and built a new tensor, and the
commit scattered that back: four copies of about 2 MB a layer. Now a CPU checkout of one row hands out `narrow` views
of the pool and records their tensor ids; the conv update and recurrence write a view in place (an `InplaceOp1` over
the F32 row), and the commit copies back only a state that was replaced (the prompt conv still builds a new one).
In-place writes are limited to those recorded views: tests clone one state into two caches, which an unconditional
in-place write broke, and production tensors can share storage the same way.

The decode conv moved off rayon onto the barrier pool; run inline it went 2.6 -> 8.3 ms, as each channel's SiLU is an
exp.

```
decode, ms a token        checkout  commit  conv  recurrence  wall
before                    5.3       3.2     2.6   4.9         111
in place                  0.03      0.01    1.3   2.9         96.8
```

CPU decode 9.1 -> 10.1 tok/s (99 ms; llama.cpp 9.35). The remaining sections each read their weights at about 30 GB/s
(GDN projections 27 ms, experts 19, LM head 12.7, GDN output 11, attention 10.8).

## Run 15 - 2026-10-21 12:30

Question: with speculation at 64 slots, decode fell to about 49 tok/s (recipe doc Run 8) and `MoE expert cache fill
failed` errors printed after the bench. Does the fill queue back up?

A split admits every expert its misses route to, up to the layer's slots, so at 64 slots each of 40 layers can have
63 copies in flight (about 1.5 MB an IQ4_KSS expert: up to 3.8 GB, 0.3 s of PCIe gen3). An 8-token verify misses far
more experts than land before they are evicted again. The errors came at process exit, while the fill thread was
still issuing copies and the driver tore its contexts down; this session's runs showed none even uncapped.

Change: a layer stops admitting misses once `MAX_PENDING_FILLS` of its copies are in flight, and the experts most of
a step's tokens missed go first. A temporary `MOE_FILL_CAP` swept the cap. Qwen3.6 35B-A3B IQ4_KSS file, RTX 3090.

`inference bench` (synthetic ids, decode tok/s, +/- 13-65):

```
cap          64 slots DFlash 5   64 slots none   128 slots DFlash 5
unbounded    75.9                79.9            171.0
16           98.0                76.8            175.9
4            120.1               83.9            152.7
1            99.3                87.5            116.1
```

Real text (`mixbench.py`, wiki / code / zh, q3 KV for DFlash):

```
cap          64 slots DFlash 3      64 slots MTP 1 draft   128 slots DFlash 3     128 slots none
unbounded    61.3 / 70.5 / 68.7     72.5 / 74.0 / 79.3     139.5 / 153.5 / 150.4  95.9 / 106.3 / 104.8
8            73.3 / 84.7 / 86.4     74.4 / 73.1 / 75.8     141.7 / 160.7 / 148.3
4            84.9 / 102.1 / 101.8   78.8 / 80.8 / 86.1     143.6 / 170.2 / 148.3  96.4 / 108.8 / 101.2
2            88.9 / 104.0 / 103.8                          145.9 / 167.7 / 157.6
1            92.2 / 104.5 / 92.2                           149.7 / 167.1 / 146.4
```

Caps 1-4 are one plateau on real text: 64-slot DFlash +38-48%, 64-slot MTP +9%, 128-slot DFlash +0-10%, no
speculation unchanged. The synthetic ids (little routing locality) lose at 128 slots below 4, so the constant is 4.
With it DFlash is the best speculation at 64 slots too (85-102 against MTP's 79-86), which removes Run 9's "fewer
drafts as slots shrink" rule from the planner's inputs, at least down to 64 slots.

## Run 16 - 2026-10-21 21:30

Question: can the loader choose expert offload and a cache size itself when a GGUF MoE model does not fit, as the
queue's "automatic planner" item asked?

Before: an automatic map that could not fit every layer put the rest on the CPU and then turned paged attention off.

Change: `ExpertCachePlan` in `resolve_map_setting` (an automatic map on one CUDA GPU, no `--moe-expert-cache`,
experts the source reports as offloadable). It plans for 32K tokens or the model's window if shorter, or more when
`--max-seq-len` or `--pa-context-len` asks; if the weights fit resident beside that it does nothing, otherwise it takes
the most experts per layer (sixteenths of a layer's experts, then 0) that fit, and else plans for the requested map's
context. `layer_expert_bytes` now reports no offloadable experts when a stack's type has no CPU kernels (those load
resident, which the explicit flag's sizing also assumed wrongly).

Dead ends on the fit test:

- Running the automatic map per trial with a `ContextSize` cache: it fit every candidate, as the map shrinks a paged
  cache to what is left (17.1 GB of weights + 5.4 GB of KV on a 23 GB card "fit").
- Adding the GDN recurrent pool (now `recurrent_state_bytes_per_sequence` on the model config, about 63 MB a slot for
  35B-A3B) still left a 1.2 GB shortfall: the default `Utilization(0.9)` sizes the cache as `0.9 x cap - used`, so 10%
  of the device stays free that the trial had counted.

The fit is now computed directly: weights (with the cache and staging) + activations (for the requested map; prompts
run in chunks) + KV for the context + post-load reservations + recurrent pool + the utilization share `<=` the device
cap. Unit tests cover the search order (resident, largest fitting step, fallback context, nothing fits).

Qwen3.6 35B-A3B IQ4_KSS + MTP block on RTX 3090:

```
flags                                           planned slots   context
(none)                                          resident        103,424
--max-seq-len 131072                            224             199,488
--max-seq-len 262144 (BF16 cache)               176             344,800
--max-seq-len 262144 --pa-cache-type q3 +DFlash 224             567,872
--pa-context-len 262144                         224             262,112
```

Every planned context is at least its target; the estimate is about one step (16 slots, ~1 GB) conservative.

## Run 17 - 2026-10-22 23:37

Question: why did a large MoE GGUF (88 GB, 512 experts per layer, Qwen3.8-Flash-Next UD-IQ4_XS) refuse expert offload,
load ever more slowly, and then produce NaN once offloaded?

Refused offload: `layer_expert_bytes` returns None when any expert stack is a type the CPU expert path cannot run, and
that path only ran 256-value super-block types with columns a multiple of 256. This file's down experts are IQ4_NL (43
layers) and Q8_0 (5) over 640 columns, so the automatic map put layers 16-47 on the CPU, paged attention turned off,
and the CPU layers ran at about one core with the GPU idle. Added: activations as Q8 blocks of 32 and IQ4_NL / Q8_0
dots against them (`cpu_dot::vec_dot_32`), row sizes from the type's block size, tests against dense weights for an
IQ3_S/IQ4_NL and an IQ4_XS/Q8_0 mix at 640 (a broken IQ4_NL dot fails them). Resident CPU MoE keeps Candle's Q8_0
(`runs_resident`); only offloaded Q8_0 takes the new dot.

Load (67 s, then 89, 150, 294 s of weight loading): 56 GiB pinned expert stacks, 27 GiB of heap, 19 GiB of file pages
resident on a 125 GB machine, so each load re-read what the copies evicted; each expert stack was copied twice (into
an owned buffer, then into pinned memory). `QuantizedWeightSource::mapped_blocks` now hands out a GGUF tensor's blocks
in place in the mapping (`MappedBlocks`), and `HostExpertStack::from_mapped` registers the shard read-only and mapped
(one shared registration per shard) or copies once from the mapping. Here registration fails with
`CUDA_ERROR_NOT_SUPPORTED` (read-only file-backed memory), so it copies once: weight loading 43 s.

NaN: prompts from about 10 tokens gave `invalid CUDA top-1 output: max_logit=NaN`, deterministic, the same at 32 and 64
cached experts. One layer's routed output had NaN only in the down projection's last row. The prompt staging every
offloaded layer shares is viewed per type; the grouped kernels read whole tiles past the last staged expert into bytes
a larger fill of another type left there, which read as IQ4_NL can hold an Inf/NaN scale (NaN times zero activations
is NaN). `DeviceExpertStack::zero_padding_async` after each staging fixes it;
`offloaded_32_wide_experts_match_the_cpu_path` poisons the staging with 0xFF and fails without the fix.

Also: `maybe_log_error` logs every internal error (a client sees "Internal server error." only), the SDK gains
`with_moe_expert_cache`, and the perplexity example takes split GGUFs, `--moe-expert-cache`, and paged attention
sized to its window. With these the model scores within 0.4% of llama.cpp (the model's own doc, Run 10).

Open: experts without pinning, as llama.cpp maps them (pageable copies from the mapping) to spare the 57 GiB copy.

## Run 18 - 2026-10-23 09:01

Question: an explicit `--moe-expert-cache 160` on Qwen3.8-Flash-Next did not fit, the automatic map put layers 43-47
on the CPU, and the run sat at one CPU thread with the GPU at 0% (13 min load, no prefill progress). Why is a CPU
fallback that slow, and why would the map fall back at all?

Method, after a wasted stretch sampling the whole 88 GB model under gdb (each load took 4 to 13 minutes): start from
one layer and work up until the one-thread behaviour shows.

Rung 1, one layer's experts (`cpu_experts::tests::flash_next_q8_0_down_layer`, IQ3_S gate/up and Q8_0 down, 512
experts over 640 x 2560, as the file's layers 2, 4, 30, 46 and 47):

```
                                          1 token          32 tokens
block kernels, whole layer               2.36 ms 7.9 cores    75 ms 7.9 cores
fallback IQ3_S gate gather, BF16          77 ms  2.3 cores  2490 ms 2.2 cores
fallback Q8_0 down gather, BF16         4159 ms  1.0 cores  5134 ms 1.0 cores
```

A resident CPU layer took the block kernels only when all three projections were types `runs_resident` allowed, and
it excluded Q8_0 (Run 17 kept resident Q8_0 on Candle). Layers 46 and 47 have Q8_0 down experts, so they took Candle's
indexed fallback, which dequantizes every expert single-threaded on each call: seconds a token on one core. The
exclusion had a reason: lifting it broke `flash_next_isq_load_and_a_reload_of_its_uqff_decode_alike` (greedy decodes
diverged from the first token).

Rung 2, the kernels at the tiny model's sizes (a temporary sweep against dense math): Q8_0 and IQ4_NL were wrong at
hidden 64 x inter 32, 64 x 64 and 96 x 64 (relative error exactly 1: zero output), right from 16K weights an expert.
`rows_per_item(cols)` picks up to 256 rows an item with no regard to the projection's rows, so `inter / item_rows`
was 0 and no item ran; a row count no power of two up to the item size divides also dropped its last rows. Now
`rows_per_item(rows, cols)` is a power of two dividing `rows`; `small_experts_match_dense_experts` covers it.

Rung 3, block path against gather inside the tiny model (temporary print): after the fix they agree to 1.8 to 2.1%
(the block kernels quantize activations to Q8, Candle's fallback multiplies in f32), but only the reload ran blocks:
the ISQ load's experts are `PendingIsqLayer`s, which delegate everything except `cpu_gguf_blocks` (and, on CUDA,
`kernel_weight`), so it ran Candle's gather. `QuantMethod::resolved_layer` now hands out what a pending layer resolves
to, and the fast experts swap their pending projections for it before dispatch; both sides run the same kernels and
the round trip passes.

Rung 4, load: a CPU layer copied its stacks out of the mapping (into an owned buffer, then a QTensor), which on a
125 GB machine already holding 57 GiB of pinned experts evicted the very file pages it read: layers 46 and 47 took
119 s and 136 s each. `MappedCpuExperts` runs the block kernels over the GGUF mapping itself (no ISQ, LoRA or
sharding): the forced split (`-n "0:43"`, 64 cached) now loads in about 80 s.

Rung 5, the rest of a CPU layer, on a checkpoint with Flash-Next's widths but 4 layers and tiny experts (CPU, 512
tokens: prefill 5.05 s, decode 500 ms a token): the busy thread sat in the CPU backend's BF16 matmul, which widened
the whole storage of both operands to F32 on one thread every call (the lm_head's 1.27 GB a token). BF16 matmuls of
up to 4 rows now read the weight in place across the barrier pool; larger ones widen only the span the view reaches,
across threads (`bf16_matmul_matches_f32_for_few_and_many_rows`).

Planner: an explicit cache that does not fit now shrinks to the most slots that fit (in sixteenths of the experts,
warning), instead of the map moving layers to the CPU; only when no slots fit does a layer move.

```
inference bench --format gguf -m <Flash-Next UD-IQ4_XS> --moe-expert-cache 64 -n "0:43" --prompt-len 512
    before: load 4 to 13 min, then one thread and no prefill progress in minutes, GPU 0%
    after:  load about 40 s, TTFT 14.6 s (512 tokens), decode 104 ms/token (all on the GPU: 3.7 s, 70 ms)
inference bench ... --moe-expert-cache 160 --pa-context-len 8192 --prompt-len 3000 (automatic map)
    "the requested MoE expert cache does not fit ...; caching fewer experts per layer requested=160 slots=128"
    Layers 0-47: cuda[0]; TTFT 16.9 s, decode 67.2 ms/token (as an explicit 128)
```

Open: a CPU layer's attention and GDN still run Candle's generic CPU ops (the forced split's TTFT is 4x the
all-GPU one); the GGUF dense CPU layers were not profiled past the experts.
