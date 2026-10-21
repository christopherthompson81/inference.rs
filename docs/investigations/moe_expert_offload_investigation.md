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
