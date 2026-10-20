# Agent decode utilization investigation

Question: nvtop showed GPU utilization near 100% for greedy decode but 70-95% while the Code tab demo ran (sampled,
Qwen3.8-27B recipe A GGUF, DFlash 2 at 7 drafts, RTX 3090). Where does the GPU sit idle?

## Run 1 - 2026-10-20 12:30

Utilization sampled every 100 ms by nvidia-smi (it refreshes about every 0.5 s) while one 1500-token decode streams
(`util_run.py`, scratchpad), same server, wiki-summary prompt:

```
mode                                         stream rate   util mean   power
greedy (temperature 0)                       144/s         98-99%      357 W
sampled (0.6 / top_p 0.95 / top_k 20)        149/s         97-99%      357 W
web UI defaults (0.7 / 0.9 / 40, rep 1.1)    120-122/s     99%         356 W
sampled + repetition penalty 1.1             119/s         99%         357 W
sampled + 6 tool definitions                 135/s         99%         357 W
no sampling fields (model defaults)          119-123/s     99%         357 W
```

Plain decode keeps the GPU busy (and at its power cap) in every mode. Repetition penalty costs ~20% of the rate
without idling the GPU (likely acceptance: the drafter does not see the penalty).

The demo itself, driven over the web UI's own routes (`/ui/api/code/demo`, then `/ui/api/code/send` in auto mode;
`demo_util.py`): 45.8 s, 6 model turns, 15 tool calls taking 0.1 s in all. Utilization 84.8% while tokens stream
(28 s), 84.1% between a tool's end and the next token (17.3 s, 1.3-4.5 s a turn). The same demo through
`inference agent --url` (agent out of process) measured 83.0%, so the in-process agent is not the cause.

nsys (graph level) over 2 s of the demo's reasoning: the verify -> drafter gap is 4.25 ms a step (benchmarks: ~1.2),
filled by ~3.1 device-to-host copies of 993280 bytes a step, one full-vocab f32 logits row per verified position,
sampled on the host. Cause: a request with `tools` gets a `ToolCallState`, and its mere presence turned off
stochastic GPU verification, greedy device verification and CUDA token sampling (`verifier.rs`, `sampling.rs`). Each
verify row was then sampled on the CPU and drafts accepted only on an exact match, which also explains the demo's
24-30% draft acceptance. The gate exists because a tool grammar can activate mid-step; while it is inactive and no
call is owed, the state only reads the text.

Also found: the GGUF exposes no generation defaults (no `generation_config.json`), so the agent, which sends no
sampling fields, samples at temperature 1.0 with no top_k or top_p; the model's config asks for top_k 20, top_p 0.95.

Fix: `ToolCallState::constrains_sampling` (grammar active, or a required call still owed) replaces the presence test
in those gates. When an accepted draft opens a tool call on the device or stochastic path, the step stops there and
the next token comes from the following verify row, sampled under the now-active grammar.

```
tools + no sampling fields, 1500 tokens       stream rate   util mean
before (b3392b3bd)                            78.7 / 79.1   87.5 / 87.9%
after                                         113.4 / 113.0 99.6 / 99.5%
Code tab demo over HTTP, after                              93.9 / 92.8% (was 83.0%); draft acceptance while
                                                            reasoning 35-44% (was 24-30%)
```

Next: the 1.3-4.5 s wait for each turn's first token; then the delay between weights loading and the server serving.

## Run 2 - 2026-10-20 12:50

Question: after a restart the weights fill VRAM long before the server serves (about a minute). Where does that go?

The serve log of a LAN restart: target weights and KV cache in place 5.6 s after loading starts, then 47 s with
nothing logged until the DFlash drafter reports its paged KV, then graph capture (0.5 s) and ready. The drafter is a
3.85 GB BF16 safetensors file quantized to q4k at every load (`--mtp-draft-isq q4k`). Each weight is moved to the GPU,
copied back for `QTensor::quantize`, and `QuantizedType::from_float` walked the tensor's blocks on one thread.

Blocks quantize independently, so `from_float` now hands runs of 1024 blocks to rayon tasks (the imatrix variant
takes whole rows, since its weights index a row). The result is byte-identical; the new test quantizes a 300 x 2048
tensor whole and row by row for Q4_0, Q8_0, Q4_K, Q5_K and Q6_K, plus the k-quants' imatrix path.

```
server start to ready (same flags)   drafter phase (log gap)   total
before                               49.5 s                    59.2 s
after                                 9.0 s                    17.1 s
```

The remaining ~9 s is reading the drafter's safetensors and quantizing it on 16 threads; caching the quantized
drafter would take most of it, if restarts keep mattering.
