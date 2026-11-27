# Concurrent prefill on Inkling 2 Mini

Question: why do concurrent requests on the LAN chat load (Inkling 2 Mini recipe C, q8 KV, LoopCD-Logits 0.5, DFlash 2
drafter, 160K context) finish no sooner in total than the same requests one at a time? A prefill-heavy workload (2K
shared system prompt, 1-4K user messages, short JSON answers) took 22.5 s for 12 requests serially and 25.3 s at 4
concurrent (`prefix_cache_investigation.md` Run 9).

Bench (scratchpad `pf.py`, `sweep.sh`): a dev server on 18083 per setting, the chat load, N unique ~1.9K-token prompts
at once (max_tokens 1), alone or beside 2 decoders looping 48-token replies; prefill tok/s is new prompt tokens over
the batch's wall time, decoder tok/s their tokens finished in that window.

## Run 1 - 2026-11-27 11:50

Prefill chunk (`--max-prefill-chunk-tokens`, default 512) against prompts at once, 160K and 32K context:

```
context chunk  decoders  N=1    N=2    N=4   (prefill tok/s)
160K    512    0         1500   1823   1657
160K    512    2         1092    888   1651
160K    1024   0         1717   1744   1783
160K    1024   2         1274   1182   1265
160K    2048   -         CUDA_ERROR_OUT_OF_MEMORY in `moe experts forward`, every case
32K     512    0         1543   1813   1836
32K     512    2          863   1035   1823
32K     2048   0         1864   1956   1965
32K     2048   2         1805   1824   1714
32K     4096   0         1929   1935   1937
32K     4096   2         1873   1825   1892
32K     8192   0         1836   1976   1862
```

The chunk is per prompt: four prompts at 512 are a 2,048-token step, which is why N=4 nears the ceiling even at 512.
Prefill tops out near 1,950 tok/s (about 8 GFLOP a token: 48 layer visits of ~83M active parameters), reached from
2,048 tokens a step. The loss is a lone prompt: 512-token steps give 1,500-1,550 alone and 863-1,092 beside decoders,
where the scheduler runs up to 8 decode steps between its chunks. At 160K a larger chunk does not fit.

Implies: cut the prompt activations so a larger chunk fits, and look at the interleave.

## Run 2 - 2026-11-27 12:20

Card memory over one 1,898-token prompt (nvidia-smi at 20 ms, after a warm request), 32K load: the earlier
measurement was about 625 KB a token one at a time (+320 MB over 512-token chunks, `server_memory_investigation.md`).

`BlockStacks` kept every block stack a forward built for the depth mixes: with n attention-residual blocks that is
n(n+1)/2 block states a token alive to the forward's end, though each layer reads only its own block's stack. Keeping
only the latest:

```
chunk 2048: +738 MiB, ~400 KB a token, 1,877 tok/s
chunk 512:  +163 MiB, ~326 KB a token, 1,549 tok/s
```

At 160K chunk 2048 now runs prompts alone (1,837 / 1,868 / 1,900 tok/s for N = 1, 2, 4), but beside decoders it still
runs out of memory: admission (`admit_cuda_prompt`) splits a batch of prompts by their activations, never one prompt's
chunk. What is left is about the MoE estimate (~342 KB), transient per layer. The Inkling tiny tests pass.

Decode steps between prefill steps (`--max-decode-steps-before-prefill`), 160K, chunk 512, 2 decoders:

```
steps  N=1 prefill  decoders   N=2 prefill  decoders
8      1036         26         980          49
2      1403          0         1316         17
1      1426          0         1409          0
```

Prefill priority gains ~35% prefill throughput and stalls the decoders for the prompt's duration. No step mixes
decode rows with a prompt chunk in this engine, so the two always trade.

Implies: options are a chunk limit admission can lower per step (raise the default, shrink one prompt's chunk when its
activations do not fit), the prompt activation estimate brought down to the new measurement, a prefill-first
interleave while prompts wait, and mixed decode + prefill steps (the larger change).

## Run 3 - 2026-11-27 12:55

Change: a lone prompt's step now counts its activations too, and under memory pressure its chunk halves
(block-aligned, down to 256 tokens; `inference_cuda_prompt_chunk_reductions_total`) where it used to be rejected or
run out of memory; one that cannot shrink further runs on its attention workspace as before. Inkling's estimate comes
down to the measurement (moe + 32 bytes a hidden unit, ~416 KB, from 170: ~733 KB).

160K load, prefill tok/s (decoders = 2 looping 48-token replies):

```
chunk  words  decoders  N=1    N=2    N=4
2048   1800   0         1805   1844   1839
2048   1800   2         1591   1550   1506
4096   1800   0         1862   1846   1853
4096   1800   2         1753   1511   1559
4096   6000   0         1758   1766
4096   6000   2         1435   1447
```

No `out of memory` in any log (chunk 2048 beside decoders failed every case in Run 2). The 6K-word run halved 12
chunks and split 2 batches. Against the 512 default beside decoders (1,092 / 888 / 1,651) a lone prompt prefills
~45% faster; the decoders stall for the longer steps (0-20 tok/s in these windows).

Implies: memory no longer pins the chunk at 512; what the chunk should be is now the latency trade, the adaptive
scheduling's job.
