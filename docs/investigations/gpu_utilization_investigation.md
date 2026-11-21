# GPU utilization of the quantization tools

Every step of the quantize/collect/score loop (#1013's tooling) kept busy on the GPU: what each spends its time on,
and what moved it. Measured on Qwen3.6-35B-A3B (BF16 GGUF, 66 GB) on one RTX 3090; `nvidia-smi dmon` for busy
share, temporary per-phase log lines for the host's view (removed after each run), nsys over a window started after
loading (`--delay`/`--duration`: tracing from the start hangs this process). Earlier work: calibration investigation
Runs 4-5 (Hessian collection 578 s -> 133 s) and Run 7 (the quantize profile this starts from).

## Run 1 - 2026-11-20 23:30

`quantize gguf` of the v2 recipe (alloc_v2_rules.txt) with GPTQ from `hessians-4x-nobos`: 106 s, GPU 33% busy. The
loader thread (read, BF16 to f32, the Hessians) runs a tensor ahead of the quantizer.

```
step                                                              total   GPU busy  bytes
start                                                             106 s   33%
GPU Q6_K / Q8_0 encoders in GPTQ's block loop (attn_qkv, attn_gate,  96 s   50%       identical
  output: 21 s -> 4 s of encoding)
the same encoders for tensors without a Hessian (token_embd 7.1 ->  89 s
  1.7 s); the loader passes BF16 bytes through, widened on the GPU
the loader three tensors ahead (was one; an expert stack is ~0.8 GB) 64 s   51-54%    identical
experts' Hessians factored in one batched pass                      64 s   54%       differ (same KLD)
```

- The block encoders (`k_quantize_instance_ggufq.cu`) are ik_llama.cpp's Q6_K (`make_qx_quants` with an imatrix,
  its gcc build's fused multiply-adds) and ggml's Q8_0, each writing the CPU's bytes and the values the CPU decodes
  (GPTQ carries the decoded values on, so they must match to the bit: a Q8_0 level rounded to -0 decodes to -0 on the
  GPU unless taken through the integer). An all-zero Q6_K super-block reads the NaN of `-inf * 0`, whose bits differ:
  x86's indefinite (0xffc00000) is written explicitly. Tests: bytes and decoded values against `quantize_blocks` and
  `dequantize`; GPTQ's GPU loop against its CPU loop.
- With the quantizer faster the loader showed: 13 s of waits on it with one tensor in flight, 3 s with three.
- Factoring: per expert, cuSOLVER's Cholesky, a solve for the inverse and a Cholesky of that, each followed by a
  host read of its info word (three syncs, plus the identity and a mask rebuilt from host data): ~1500 syncs and
  uploads a down stack, 39% of the GPU's time in a window. Now `linalg::inverse_upper_factors`: with `J` the
  reversal, `J H J = L L^T` (one `potrfBatched` for all experts) and `U = J L^-1 J` (one `trsmBatched` against the
  identity), so `H^-1 = U^T U` without forming the inverse; info words read once per pass. 256 experts of 512: 41 ms
  against 260 ms for the single Cholesky calls alone. The factor is the same matrix by another route, so the bytes
  differ from the earlier quant; KLD (ours, streamed): 0.01560 vs 0.01536 on WikiText B, 0.01196 vs 0.01208 on the
  chat set, within noise. A first version copied the Hessians twice on one thread (down stacks 29 s against 18 s);
  they are now damped straight into the stack in parallel.
- Timing pitfall: 20 GB outputs kept in the scratch dir pushed the 66 GB source and 17 GB of Hessians out of the page
  cache (a 76 s run, loader 62 s); with them deleted, two runs in a row read 64 s and 65 s.

Where it stands: the quantizer is the bound (~50 s: experts' GPU loops 40 s), the loader ~37 s beside it. Left: the
per-block importance uploads and byte read-backs in `gpu_row_scaled` (thousands of small copies a stack).

## Run 2 - 2026-11-21 00:30

The error carry inverted each block's diagonal of `U` on the host (`upper_inverse`, f64 back substitution): per
super-block of a down stack, 256 experts' 256x256 blocks read back (64 MB) and inverted on the CPU, then uploaded.
Now one `cublasDtrsmBatched` against the identity on the device, still in f64 (`linalg::upper_block_inverses`).

```
v2 recipe quantize           total        GPU busy   KLD wiki B / chat (ours)
batched factors (Run 1)      64 / 65 s    54%        0.01560 / 0.01196
+ block inverses on the GPU  64 / 60 s    57%        0.01560 / 0.01196
```

## Run 3 - 2026-11-21 01:30

Built-in per-op timing, so each step stops needing temporary log lines: `INFERENCE_PROFILE=1` makes named spans
(`inference_tensor::profile::span`) sync their device as they end and sum time and calls per span and thread; the
quantize commands print the table when they finish. The syncs serialize what overlaps, so a profiled run is slower
than a plain one (85 s against 61 s); the spans give the cost of each op. Spans: the quantizer's `gptq.tensor`,
`.upload`, `.factor`, `.row_scales`, `.encode`, `.carry`, its loader's `gptq.load` and the main loop's
`gptq.wait`, `write`; the streamed loop's `stream.read_wait`, `.build`, `.forward`, `.hessians`, `.score`,
`.write_base`.

The first profiled quantize: `gptq.tensor` 88 s, of it the carry 25 s (640 calls, 39 ms each). The carry formed
`E U_bb^-1`, then `(E U_bb^-1) U_rest`, subtracted it from a copy of the columns after the block and re-packed them.
Now `M = U_bb^-1 U_rest` first and one cuBLAS GEMM (alpha -1, beta 1) writes `w[:, after] -= E M` into the matrix in
place (`carry_in_place`; a stack's per-expert groups as one strided batch). All three of GPTQ's loops share it
(`Columns`), so the GPU loop and the CPU loop still match byte for byte; GPTAQ's residual term keeps the narrowing
carry. A batch of 3 experts and each alone now differ in 3 of 13,248 bytes (cuBLAS picks kernels by batch size,
and a value at a rounding tie lands the other way), so that test allows 1 in 1,000.

Then the values' upload and widening (11 s) moved into the loader thread, which syncs its stream before handing the
device tensor over:

```
quantize (plain runs)                        total      GPU busy  KLD wiki B / chat (ours)
Run 2                                        60-64 s    57%       0.01560 / 0.01196
in-place carry                               61 s       55%       0.01534 / 0.01220
+ upload in the loader                       54-58 s    58-61%

profiled spans (s)   tensor  load  encode  upload  row_scales  factor  carry  wait  write
in-place carry       73.2    35.4  17.0    11.0    9.7         7.7     7.4    2.4   4.4
+ upload in loader   62.9    48.1  18.9    12.5*   10.0        8.3     7.5    8.5   4.5   (* on the loader)
```

106 s at the start of this investigation, 54-58 s now. The quantizer and the loader are close (the waits came
back), so the next gains are on both sides: the loader's reads and Hessian unpacking, and the encode calls (14,088,
mostly Q8_0's 32-column blocks, each with its carry).

## Run 4 - 2026-11-21 02:30

Hessian collection (266K chat tokens, 133 s, GPU 61%), profiled: `stream.forward` 97 s of it, and under finer spans
(`layer.gdn`, `layer.attention`, `moe.gather`, `hessian.dense`, `hessian.routed`) the tracked experts' gather path
(`moe.gather`, a pass per expert each call) 117 of 162 profiled seconds.

Tracked BF16 experts now run the grouped CUTLASS MoE (`cutlass_fused_moe_routed`), which also returns the down
projection's inputs as the kernels ordered them (each expert's GLU outputs in a run, with the counts), fed to the
down stack's statistics (`process_sorted_routed_stats`, queued with their experts for the batched flush). Layer 0's
down Hessians match the gather path's to 1.3e-3 (the GLU's bf16 rounding differs), deeper ones by the usual route
flips. The grouped call still read 100 ms per 8 chunks, and a benchmark of the MoE alone (8192 tokens, 256 experts,
2048 -> 512) put CUTLASS at 74-80 ms against 10.6 ms for a loop of one cuBLAS GEMM per expert. Replacing its GEMMs
with cuBLAS's grouped GEMM changed nothing (76-95 ms): the stages, timed inside, summed to 13 ms. The rest was the
CUDA memory pool: its release threshold is 0 outside a running engine (which retains it), so every sync handed the
~1 GB the call allocates back to the driver and the next call mapped it again. With the pool retained, CUTLASS reads
12.3 ms and the cuBLAS version 12.8 ms; CUTLASS stays. (The engine retains the pool while it runs, so serving never
paid this; a model's load, the streamed tools and `quantize gguf` did.)

`inference_core::retain_memory_pool` now holds the pool for the streamed run and for `requantize_gguf` (through
inference-api's wrapper).

```
                                        before     now
Hessian collection, 266K tokens         133 s      97 s (GPU 75%)
  profiled forward (64K tokens)         50.7 s     16.3 s
quantize gguf, v2 recipe                54-58 s    52-53 s (GPU 61%)
```

Collection now: ~1.1 s a layer, of it the layer's build ~0.58 s (1.6 GB of BF16 uploaded from the mapping), the
Hessians' save ~0.25 s, the forward the rest.

## Run 5 - 2026-11-21 03:30

Collection again (266K chat tokens, 97 s after Run 4):

- Each layer's Hessian files (~400 MB) are written on a background thread while the next layer runs, one layer's
  writes in flight: 97 -> 94 s (the writes mostly land in the page cache, so cheaper than the profiled 0.25 s a
  layer suggested).
- The grouped call concatenated gate and up into `[E, 2I, H]` every call (7.3 ms of 22); a one-entry cache keyed by
  the two tensors' ids makes it once a layer: -> 91 s.
- Two runs in a row then differed in the down experts' Hessians (relative 1e-7, the dense ones identical): the
  kernels' row sort places rows with atomics, so their order within an expert's run, and the Gram's f32 summation
  order, varied. The GLU outputs now go back to slot order (`index_select` by the weighted gather's own
  permutation) and through `process_routed_stats`, which orders rows on the host as the gather path did; the
  sorted-rows entry point is gone. Two runs: 0 of 10,571 files differ. -> 88 s.
- Uploading the layer's weights through the pinned staging ring: 17.9 -> 16.3 s of profiled build time (733
  uploads, ~4.3 GB/s from the mapping), 45 -> 43 s plain on 64K tokens. Too small to keep; the build reads the
  mapped file page by page, which the read-ahead's `read()` warms but does not map.

```
Hessian collection, 266K tokens     start 578 s   calibration Run 5 133 s   Run 4 97 s   now 88 s (GPU 74%)
```
