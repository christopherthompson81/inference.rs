# Inkling 2 Mini prefill speed

Question: what bounds Inkling 2 Mini's prefill? Batched prefill tops out near 1,950 tok/s on a 3090 however many
prompts share a step (`concurrent_prefill_investigation.md` Run 1), so concurrent throughput on prefill-heavy work
follows this ceiling.

Setup: the LAN chat load (recipe C, q8 KV, LoopCD-Logits 0.5, DFlash 2 drafter q8_0, vision) at 32K, a dev server
with `--max-prefill-chunk-tokens 2048` so a ~1.9K-token prompt is one step. Profiles (scratchpad `prof.sh`): the
server under an nsys session (`--trace=cuda,nvtx --cuda-graph-trace=node`), loaded and warmed with two prompts, then
three unique ~1.9K-token prompts (max_tokens 1) captured.

## Run 1 - 2026-11-27 15:20

1,776 tok/s under the profiler (1,864 without). GPU kernel time over the three prompts, 2,923 ms in 29,637 launches:

```
32.0%  936 ms   567x  prefill_mma_kernel<4, Q8>        attention, q8 KV
22.8%  666 ms   414x  mul_mat_q<241, 128>              routed experts
 5.8%  169 ms   795x  depth_mix_kernel (bf16 / f32)
 4.2%  124 ms  2304x  ucopy_f32
 4.0%  116 ms   162x  prefill_kernel<4, Q8>            attention, scalar path
 3.4%   98 ms   216x  mul_mat_q<8, 128, true>
 2.6%   76 ms    48x  mul_mat_q<243, 128>
 2.6%   75 ms   438x  mul_mat_q<8, 128>
 2.1%   63 ms  1662x  ucopy_bf16
 1.5%   44 ms    24x  mul_mat_q<242, 128>
 ...    ~1% each: sgemm (f32 controller and router), controller in/out, quantize_mmq, moe reduce, short conv
```

Attention is ~36% of a prefill, ~310 ms of a ~1 s prompt: a 1.9K-token prompt in 48 layer visits, 20 of 24 layers
windowed at 1,024 keys, is on the order of 1 TFLOP of attention, so the kernel runs near 3 TFLOP/s against tensor
cores that sustain an order more. The experts (mul_mat_q on the quantized expert format) are ~30%.

Implies: the attention prefill kernel first.

## Run 2 - 2026-11-27 15:50

Nsight Compute on `prefill_mma_kernel<4, Q8>` (three launches mid-prompt): 1.3 ms and 0.70 ms launches of 6 and 8 query
tiles (384 / 512 queries; the windowed layers run their prompt in pieces), compute 15-22% and memory 6-9% of peak,
achieved occupancy 14% (254 registers a thread: 2 blocks an SM), tensor pipe active 3.5-5%, the top stall
`long_scoreboard` (waiting on global loads). The K/V staging dequantized each value of a 64-key tile on its own
(`load_at`: a byte load and its group's scale load a value, 128 of each a thread a tile for K and V).

Change: a q8 tile stages 8 values a thread, from two 4-byte loads and one scale (`stage_q8_span`; rows are only
4-byte aligned: an 8-byte load faulted on the head-dim-64 test rows), K as one 16-byte shared store; the same for the
scalar `prefill_kernel`. Kernel time over the three prompts:

```
                       before    after
prefill_mma_kernel     936 ms    244 ms
prefill_kernel         116 ms     39 ms
all kernels           2923 ms   2192 ms
```

Without the profiler (32K, chunk 2048), prefill tok/s:

```
              N=1    N=2    N=4
before        1864   1956   1965
after         2389   2407   2407
beside 2 decoders: 2313 / 2302 / 2301
```

+22-28%. The biased attention CUDA cases and the Inkling tiny tests pass. Now the routed experts lead (`mul_mat_q`
on the expert format, 31%), then the MMA attention (11%), the depth mixes (6.5%) and copies (`ucopy_f32` /
`ucopy_bf16`, ~9%).

Implies: the copies and depth mixes are the next look; the attention kernel still runs at low occupancy (registers),
and the experts' MMQ is llama.cpp's.
