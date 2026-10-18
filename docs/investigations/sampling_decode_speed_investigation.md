# Sampling decode speed investigation

Qwen3.8 27B (IQ4_XS GGUF, MTP with 3 drafts) decoded far slower in the web UI than in greedy benchmarks.

## Run 1 - 2026-10-19 00:30

Question: is the slowdown the quantized KV cache and long context, as it first looked, or something else?

Command: the test server alone on the RTX 3090 (the LAN server stopped, its ~2 GB context freed), a fresh
process; `/v1/chat/completions` with a short prompt, 256 tokens with `ignore_eos`, three runs per setting (the
first after a load includes lazy graph capture).

```
decode tok/s                                   q4 @ 32K        q4 @ 256K       bf16 @ 32K
greedy                                         78-79           75-83           74-79
greedy, repetition_penalty 1.1                 72              66-72
temp 0.6, top_p 0.95, top_k 20                 58-78           50-71
temp 0.6, top_p 0.95, top_k 20, rep 1.1        49-52           35-36
(the web UI's defaults are the last row)
q8/q4 @ 32K greedy 78-79; q4 @ 128K greedy 78; q8/q4 @ 128K greedy 74-76; q3 @ 256K greedy 72
```

The cache type costs nothing at 32K and 128K. At 256K the load leaves little VRAM and decode graph precapture
fails (`CUDA_ERROR_OUT_OF_MEMORY`), and graphs are captured lazily. Sampling lowers MTP acceptance (about 62% greedy
to 39-50%). The combination of sampling and the repetition penalty costs far more than either alone (78 -> 50 at
32K, 80 -> 36 at 256K), and grows with the context reserved though the sequences are short.

Implication: the cost is in the sampling path, not the cache; next, profile sampled decode with a penalty at 256K
against 32K (nsys) to find what scales with the reserved context. The web UI gains a greedy toggle meanwhile.

## Run 2 - 2026-10-19 01:20

Question: what does sampled decode with a penalty spend its time on, and does it scale with the reserved context?

Command: `nsys profile -t cuda,osrt --cuda-graph-trace=node --duration=150` around the test server, one load per
run, three 256-token requests at temp 0.6 / top_p 0.95 / top_k 20 / repetition_penalty 1.1; q4 cache at 256K, then
at 32K.

```
decode tok/s under nsys    256K 45.5 43.7 46.6     32K 49.1 45.7 47.1
32K, the 17.4 s stretch holding the three requests: GPU busy 64%, 1,798 idle gaps over 1 ms totalling 5.85 s
(median 2.95 ms); kernels: IQ4_XS/Q5_K/Q6_K mmvq 8.6 s, is_u32_f32 287 ms over 15,984 calls
```

Run 1's 256K-vs-32K gap (36 vs 50) does not reproduce: the reserved context does not matter, and sampled rows vary
run to run with MTP acceptance. The cost is host time between steps. `Sampler::cuda_batch_sampling_plan`
(`crates/inference-nn/src/sampler.rs`) returns `None` when any of the repetition, frequency or presence penalties is
set, so every row, each MTP verify and draft row included, leaves the GPU for the CPU sampler (the GPU paths of
gpu_sampling_coverage_investigation.md never see penalized requests).

Implication: penalties on the device (a per-sequence token-count buffer updated with each step's accepted tokens,
applied to the logits before the GPU plan) would put the web UI's default sampling back on the GPU.

## Run 3 - 2026-10-19 02:30

Question: with penalties applied on the device, does penalized sampled MTP decode stay on the GPU and recover?

Change: each sequence keeps dense device counts of its tokens (all of them, and those generated), updated with the
tokens added since the last step (`CudaPenaltyCounts`); a kernel penalizes a sequence's `[rows, vocab]` logits from
them, row `j` also counting the drafts before it. The sparse-rejection verifier penalizes its target rows and takes
penalized samplers (`cuda_speculative_sampling_plan_with_penalties`); the MTP drafter samples unpenalized drafts,
which rejection sampling allows for any draft distribution. `device_penalties_match_the_host_per_verify_row` holds
the kernel to the host penalties per verify row as tokens are added and when the history is rewritten.

Command: Run 1's harness, q4 cache @ 32K, the LAN server stopped.

```
decode tok/s                                    before (Run 1/2)    after
temp 0.6, top_p 0.95, top_k 20, rep 1.1         46-52               64.6 73.8 64.2
temp 0.6, top_p 0.95, top_k 20                  58-78               61.5 72.3 82.7
greedy                                          78-79               82.4 82.6 81.2
```

Implication: the web UI's default sampling decodes as fast as the same sampling without a penalty. Left on the
host: greedy MTP verify with a penalty (its batch takes no drafts) and multi-sequence plain decode with a penalty
(the single-row device path already penalizes, from a per-step count of the whole context).
