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
