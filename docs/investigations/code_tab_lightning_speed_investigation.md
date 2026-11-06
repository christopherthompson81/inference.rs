# Code tab speed on Nemotron 3.5 Lightning

The Code tab's agent decodes at 60-85 tok/s on Lightning (Q4_0 GGUF, built-in MTP 3 with a q4_0 draft head, RTX
3090), against ~250-300 in plain greedy chat benchmarks. Goal: find what costs the speed and bring the agent close to
the plain rate. (Getting the tab to work at all needed the tool-call grammar fix on `tool-call-trailing-newline`.)

## Run 1 - 2026-11-05

**Question:** which difference between the agent's requests and the benchmark costs the speed: tools, sampling
(the model's defaults, temperature 1.0 and top-p 0.95), context length, or tool calls under the grammar?

**Command:** one server (`serve --max-seqs 1 ... --paged-attn on --pa-context-len 32768 --mtp --mtp-n-predict 3
--mtp-draft-isq q4_0`); `code/speed.py` (a 300-token story, thinking off) and `code/speed_call.py` (a reply that is
one ~1.2K-character `write` tool call), decode rate from the response's `avg_compl_tok_per_sec`.

| variant | decode tok/s |
|---|---|
| plain, greedy | 186.7 / 188.3 |
| plain, sampled | 138.2 / 134.3 |
| tools offered (not called), greedy / sampled | 185.1 / 143.2 |
| tools, sampled, ~10.8K-token prompt | 122.9 |
| plain, greedy, ~10.5K-token prompt | 183.5 |
| reply is a tool call, greedy | 185.7 |
| reply is a tool call, sampled | **39.7** |

**Finding:** sampling costs ~25 %, long context ~15 %, offering tools nothing. A tool call under the grammar is free
greedy but 4.7x slower sampled, and the agent's replies are mostly tool calls (edits carrying code), sampled.

**Implication:** the sampled path under a grammar is the main cost. Next: read how a constrained, sampled step runs
(mask per token, where sampling happens, what MTP verification does under a grammar).

## Run 2 - 2026-11-05

**Question:** what makes a sampled tool call 4.7x slower than a greedy one?

**Reading:** under a grammar (or an active tool call) `stochastic_verification_allowed` is false, so MTP verify rows are
not verified on the device: each is sampled on the host (the full 131K-vocabulary logits, top-p), checked against the
grammar, and a draft kept only on an exact match (acceptance 37-41 %, length 2.1-2.2 in the log).

**Command:** `perf record -F 999 -g -p <server>` over `code/speed_call.py` (greedy then sampled tool call).

**Finding:** 29.4 % `quicksort<(u32, f32)>` plus 14.0 % `swap_if_less` and a few smaller sort helpers, all
`sort_unstable_by(cmp_desc_by_prob)`: with no `top_k` (Lightning's defaults set temperature 1.0 and top-p 0.95 only) the
sampler took `k` = the vocabulary and fully sorted 131,072 probabilities per sample to find the top-p nucleus, which on
these replies is a few dozen tokens. The same pattern sat at three call sites (`sample_top_kp_min_p`,
`sample_speculative_top_kp_min_p`, `filter_top_kp_min_p`).

**Change:** `kp_candidates`: without top-k, rank a prefix of 256 candidates (select_nth, then sort the prefix), growing
8x until its mass reaches the cutoff, the cutoff summed over the whole distribution as before. A test pins the
nucleus against the full sort's: exact on a peaked distribution; on a long tail and a flat one, where the cut falls
among near-equal tokens and the f32 cutoff (summed in another order) moves it by a token or so (8 of 124,518 on the
flat one), the kept set is the most likely tokens and its mass matches within 1e-4.

| variant | before | after |
|---|---|---|
| reply is a tool call, greedy | 185.7 | 179.6 / 184.1 |
| reply is a tool call, sampled | 39.7 | **98.1 / 98.1** |
| plain, greedy / sampled | 187 / 136 | 188.3 / 136.5 |
| tools (not called), greedy / sampled | 185 / 143 | 189.8 / 129.4 |

**Implication:** a sampled tool call is 2.5x faster; it still trails greedy (98 against 185) through the host-sampled,
exact-match verify, and plain sampling trails greedy (136 against 188) through MTP's lower acceptance when sampled.

## Run 3 - 2026-11-05

**Question:** with the sort gone, where does a sampled tool call's time go, and does the Code tab's demo get faster?

**Command:** `perf record` over `code/speed_call.py` again; `code/accept.py` (one 400-word sampled reply as text and as a
`write` call, MTP acceptance read from the log between them); the demo (`code/demo.py`, 2-minute limit) three times
on #826's branch (the grammar fix) with and without this branch's commit.

**Finding:**
- CPU samples: 63 % in `libcuda` (waiting on the GPU), the sampler's `select_nth` partition 7 %, then
  `WeightedIndex::new` 3.4 % and `expf` 3.4 %: the draw still built its index over all 131,072 probabilities, nearly
  all zero after the nucleus cut.
- Acceptance under sampling is about the same with or without the grammar: plain text 38.7 / 32.3 % (length 2.16 /
  1.97), a tool call 36.6 / 56.0 % (2.10 / 2.68). With deterministic drafts, host exact-match verification accepts a
  draft with probability p(draft), as rejection sampling does, so the grammar path loses no acceptance; what it adds
  is host work per verify row.
- The demo: #826 alone 63.1 / 73.3 / 64.4 s; with the ranked-prefix nucleus 59.8 / 40.8 / 54.4 s (every run passing
  its 19 tests).

**Change:** `sample_candidates`: the multinomial draw over the nucleus's tokens only, in ascending token order, so
its cumulative weights, its one rng draw and the token it picks are those of a draw over the whole vocabulary. A test
draws 200 seeds both ways on a distribution whose ranks are scattered over the vocabulary (it fails if the
candidates are drawn in probability order).

| sampled tool call | tok/s |
|---|---|
| before Run 2 | 39.7 |
| ranked-prefix nucleus | 98.1 / 98.1 |
| + draw over the candidates | 105.1 / 106.2 (the same 237 / 284 tokens as before it: the same draws) |

**Implication:** the agent's sampled tool calls are 2.7x faster than at the start; greedy is ~185. What is left is
sampling itself (MTP accepts ~2.2 tokens a step sampled against ~3 greedy) and the host-side verify rows (logits to
the host, a CPU softmax, the grammar check). Moving constrained verification onto the device is the next, larger step.

## Run 4 - 2026-11-05

**Question, from the user:** the drop from greedy to sampled shouldn't be that dramatic; is work sent to the CPU?

**Command:** `code/plain.py` (a 400-word story, greedy or sampled, MTP acceptance from the log); `perf record` over a
sampled one; then an Nsight session (`--cuda-graph-trace=node`) over one greedy and one sampled story, split at the
pause between them.

**Finding:**
- Acceptance explains little: length 2.22 greedy against 1.98 sampled (1.12x), while the rate is 186.9 against
  135.8 tok/s (1.38x).
- Plain sampling does not run on the CPU: 81 % of the CPU samples are in `libcuda` waiting, no sampler code.
- The GPU: a sampled step costs 17.5 ms against greedy's 14.4 (272 / 263 steps). The difference is sampling
  kernels:

| per sampled step | GPU time |
|---|---|
| `sparse_rejection_topk_f32_kernel`, once | 1.5-1.9 ms, launched `<<<batch, 1>>>`: one thread |
| drafter, x3 drafts: `topk_large_stage1` 144 us + `stage2_f32_packed` 207 us + sample 17 us | ~1.1 ms |
| verify, 4 rows: `topk_large_stage1` (grid y 4) 227 us + stage2 207 us | ~0.43 ms |

- The rejection kernel ran every candidate's work serially on one thread out of local memory: an O(K^2) duplicate
  check over the 128 candidates at each of 4 positions, `expf` per candidate, then on rejection an O(K x q_width)
  residual and an O(K^2) insertion sort by token id.
- `--mtp-draft-sampling greedy` changes nothing on the built-in head (the same tokens to the count): it samples its
  drafts whenever the request samples.

**Change:** the top-K rejection kernel as a block of one thread a candidate (`prepare_topk_block`,
`sample_topk_block`): reading and validation, the weights, the duplicate check (each slot against the earlier ones),
the residual and the draw's ranking by id run per slot; the order-dependent sums (top-k, top-p, min-p, the residual,
the draw's walk) stay serial on thread 0 in slot order, so outcomes are unchanged. The serial helpers are gone. The
kernel's 11 cases pass, the four CPU-reference ones (ties, top-p) included.

| plain story | before | after |
|---|---|---|
| sampled | 133.4 / 136.4 | **145.3 / 148.6** (the same 539 / 524 tokens: the same outcomes) |
| greedy | 190.9 | 190.8 |

**Implication:** sampled is now 1.29x behind greedy, of which acceptance is 1.12x. The rest is the ranked top-K
(`topk_large`: 64 blocks for stage 1, one block merging 8,192 candidates in stage 2), ~1.5 ms a step for the drafter's
three draws and the verify rows. (Tool-call numbers on this branch are not comparable: it lacks #826's grammar fix,
so tool-call replies run to max_tokens.)

## Run 5 - 2026-11-05

**Question:** after the rejection kernel, the ranked top-K (`topk_large`) was ~1.5 ms a sampled step; can it keep its
order and get fast?

**Reading:** both stages selected by k = 128 rounds of argmax: stage 1 rescanned its 2,048-value chunk from global memory
with two barriers a round, in 64 blocks; stage 2 rescanned a row's 8,192 candidates the same way in one block.

**Change:** stage 1 sorts its chunk once in shared memory (bitonic, by value then lower index, NaN and -inf last and
written as (-inf, 0) as before) when the chunk is a power of two, launched with 1,024 threads (two values a thread);
stage 2 merges the chunks' sorted lists in one warp (each round compares the list heads, ties to the lower chunk,
the lower index range) when a row has at most 1,024 lists. Other shapes keep the argmax rounds. Stage 2's
block-wide softmax prologue is untouched, so the normalizers top-p reads are the same bits. The top-K, sort,
sampler and rejection suites pass (ties across chunks, special values, offsets, mixed temperatures).

Nsight over one sampled story (263 steps), per launch:

| kernel | before | sort + merge (256 threads) |
|---|---|---|
| `sparse_rejection_topk_f32_kernel` | 1,476-1,906 us | 99.7 us (Run 4's block kernel) |
| `topk_large_stage1` | 144 us | 111 us |
| `topk_large_stage2_f32_packed` | 207 us | 74.7 us |

| prose story, 600 tokens | Run 4 | sort + merge | stage 1 at 1,024 threads |
|---|---|---|---|
| sampled | 145.3 / 148.6 | 153.0 / 156.2 | **154.1 / 157.8** |
| greedy | 190.8 | 190.1 | 190.9 |

The sampled replies are the same 539 / 524 tokens throughout: the same outcomes.

**Check, from the user** (why greedy reads 190 when plain decode was ~250 and MTP ~300): the same build on Run 25's
benchmark (`mtp/run.sh`, 4 prompts x 256 tokens, 8K context): plain 251.0 / 251.0 / 249.4 / 250.7 (mean 250.5),
MTP 3 with the q4_0 head 331.2 / 251.2 / 228.2 / 397.5 (mean 302.0, length 3.18). This run's 600-token prose story with a 32K context is
MTP's weakest case (the benchmark's prose prompt alone is 228); no regression.

**Implication:** sampled now trails greedy by 1.21x on prose, acceptance accounting for 1.12x of it.
