# DSpine drafter investigation

Tencent's DSpine (`tencent/Qwen3.8-27B-DSpine-b16`, Apache-2.0, reference code Tencent/DSpine at 0cd6f10): a five-layer
DFlash block drafter for Qwen3.8 27B with block 16 and an 8,192-token sliding window. After every layer a code bridge
writes each row's gated message from its predecessor into the residual stream (anchor row exempt); the drafts come
from a serial refinement of the last write over the target head's top-16 candidates per row, each candidate scored
given each candidate the previous row could have taken.

Target throughout: Qwen3.8 27B recipe A GGUF, q3 KV cache, 32K context, RTX 3090. Prompts: three synthetic ones
(a word problem, a Python log parser with tests, a hash-map explainer with C), 700 tokens out at most, through
`bench.py` (scratchpad) reading `/metrics`. Sampled runs use the model card's temperature 1.0, top-p 0.95, top-k 20;
greedy is used only to compare acceptance (never as a published speed).

## Run 1 - 2026-11-17 08:25

**Question:** does the drafter load and fit beside the target, and at what depth?

The checkpoint is 4.05 GB BF16: the DFlash backbone (fc 25600 -> 5120, five Qwen3 layers) plus the bridge, whose
token codebook alone is 248320 x 1024 (508 MB). `--mtp-draft-isq q4k` quantized the projections only; the size
estimate divided the whole checkpoint by the pack factor, so the bridge's bf16 bytes were missing from the
reservation and the load ran out of memory. The estimate now keeps the tensors the drafter keeps at the model dtype
(`DFlashConfig::stays_unquantized`).

At 15 drafts (the trained block) the load still runs out of memory, at 32K and at 8K with one sequence: the target's
GDN layers roll a rejected tail back through the fused transition log only up to `GDN_SPEC_FUSED_MAX_TOKENS` = 8
verify tokens (7 drafts); past it every draft position takes a full recurrent-state checkpoint lane (16 lanes of all
48 linear layers' state per slot). SGLang's model card runs ReplaySSM for the same reason. So this drafter runs at up
to 7 drafts here until the transition log covers 16 tokens.

## Run 2 - 2026-11-17 08:31

**Question:** at 7 drafts, how does DSpine compare with DFlash 2 (`incoai/Qwen3.8-27B-DFlash2`, also q4k, 7 drafts)?

```
                 sampled: accepted/step, tok/s      greedy: accepted/step, tok/s
                 math        code        review      math        code        review
DSpine           5.72 240    3.29 154    1.81 101    5.17 228    3.90 181    2.18 117
DFlash 2         5.14 225    2.57 133    2.06 114    5.19 230    3.99 186    1.89 107
```

One sampled run each, so noisy; loaded memory 22,658 MiB (DSpine) vs 22,856 MiB (DFlash 2). Greedy acceptance is
level, sampled leans to DSpine on math and code. The card's gains (7.20 vs 5.69 on GSM8K) are at DSpine's block 16
against DFlash 2's 8, the depth Run 1 rules out here.

## Run 3 - 2026-11-17 08:34

**Question:** do our bridge writes and greedy tail match the reference's arithmetic?

With CUDA graphs off, a throwaway dump saved one draft step's bridge inputs and outputs per layer and the tail's
inputs (rows, stash, candidates, their head rows, logits) and drafts. `ref_bridge.py` (scratchpad) transcribes the
reference bridge (`sglang_dspine/kernels/dspine.py`, the MLX `bridge.py`) and the greedy serial table and chain
(MLX `RefinementTail.greedy`, `_serial_table`, `_serial_chain`) in torch over the dumped inputs, the codebook BF16:

```
layer 0..4 write   cos 0.99963 / 0.99996 / 0.99997 / 0.99999 / 0.99997 (our codebook q4k, the reference's BF16)
stash raw / rms / query   cos 1.000000 / 1.000000 / 0.999998
candidates         6 of 7 rows the same set; row 2 differs by an exact tie at the 16th place (logit 13.375)
drafts             reference [1144, 4087, 6673, 3299, 3377, 13, 14235]
                   ours      [1144, 4087, 6673, 3299, 3377, 13, 14235]
                   argmax    [1144, 4087, 6673, 3299, 13, 14235, 13]
```

The tail reproduces our drafts exactly, and they differ from the plain argmax from row 4 on, so the refinement is
doing its work. Eager and graph greedy runs accept the same (5.17 / 3.90 / 2.18).

## Run 4 - 2026-11-17 08:37

**Question:** what does quantizing the codebook (the user's suggestion) cost?

The codebook is only gathered by row (anchors and the candidates' senders), so it takes the drafter's weight type
like a token embedding. Greedy, 7 drafts:

```
codebook BF16   5.06 / 3.93 / 2.19 accepted per step   23,011 MiB
codebook q4k    5.17 / 3.90 / 2.18                     22,658 MiB
```

No measurable loss for 353 MB, so it follows `--mtp-draft-isq`. The rest of the bridge (reads, writes, gates, about
110 MB) stays BF16.

## Run 5 - 2026-11-17 08:45

**Question:** can the target's transition log cover a 16-row verify, so DSpine runs at its trained depth?

The log is the fused verify recurrence's per-row keys, deltas and decays, staged and applied lazily for the accepted
rows. Its cap (`GDN_SPEC_FUSED_MAX_TOKENS`) was 8 because the fused verify kernel gave each row a warp (8 warps of
256 threads) and kept every row's query, key, value and output in static shared memory. At 16 rows that shared memory
is about 57 KB with an F32 state, past the 48 KB static limit. The kernel now takes its row count as a template
parameter, carves dynamic shared memory sized by it (opting past 48 KB for the wide variant) and loops each warp over
rows `warp, warp + 8`; verifies of up to 8 rows keep the short variant and its two blocks per SM. The staging, apply
and commit kernels already looped over rows; their static arrays at 16 rows are 16 KB.

Tests: the transition commit against a prefix replay at 12 and 16 rows beside 4 and 8 (every activation and state
dtype, tiled and grouped heads), staging at 16. With the wide variant's row loop cut at 8 the 16-row case fails
(`batched transition recurrent state: max_diff=0.14`).

## Run 6 - 2026-11-17 08:52

**Question:** at its trained depth, what does DSpine give on the 3090?

The load now fits at 15 drafts: 22,705 MiB at 32K with the q3 cache (16 rollback lanes no longer needed).

```
                 sampled: accepted/step, tok/s      greedy: accepted/step, tok/s
                 math        code        review      math        code        review
DSpine 15        6.19 181    4.56 144    1.70  70    7.54 225    3.49 118    2.06  81
DSpine 7         5.72 240    3.29 154    1.81 101    5.17 228    3.90 181    2.18 117
```

Acceptance rises with depth on the word problem (7.54 of 15 greedy) and the code prompt sampled, but the step grows
from about 27 ms to about 38 ms (greedy math: 6.17 tokens in 27.1 ms against 8.54 in 38.0 ms). The cause is the
target, not the drafter: GGUF matvec kernels cover at most 8 rows (`MMVQ_MAX_BATCH`), so a 16-row verify drops every
quantized projection to the batched MMQ path. On this card and target 7 drafts stays the faster depth; 15 pays off
where 9-16 row matmuls cost about what 8 do (a BF16 or larger-batch-friendly target), or once the IQ4_KS / Q-type
matvec kernels take 16 rows.

Tests: `dspine::tests::writes_and_the_greedy_tail_match_the_reference_transcription` (the bridge write and the greedy
serial table against a scalar transcription of the reference kernels; fails with the predecessor offset changed) and
`qwen3_5_text_tiny::hybrid_dspine_drafts_and_keeps_greedy_output` (a two-layer tiny DSpine inside the drafter's decode
graph, greedy output unchanged, a sampled request drafting).

## Run 7 - 2026-11-17 09:20

**Question:** does the widened verify kernel slow the 8-row case (DFlash 2 at 7 drafts)?

Greedy DFlash 2 after the change read 220 / 180 / 104 tok/s against Run 2's 225-230 / 186 / 107. A/B on the same warm
card (66-67 C), master's GDN kernels and cap swapped in against the branch's, two runs each:

```
branch   220.6 / 180.7 / 104.3   222.5 / 179.6 / 103.9
master   220.3 / 179.4 / 103.7   221.3 / 179.1 / 103.6
```

Identical outputs and acceptance; the gap to Run 2 is the card's thermal state, not the kernel.
