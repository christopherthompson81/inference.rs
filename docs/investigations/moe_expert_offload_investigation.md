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
