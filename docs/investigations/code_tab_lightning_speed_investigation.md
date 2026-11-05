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
