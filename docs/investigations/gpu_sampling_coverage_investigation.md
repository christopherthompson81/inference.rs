# GPU sampling coverage investigation

Starting point: a dense Qwen3.x 27B IQ4_XS GGUF in the Avalonia demo (C ABI, RTX 3090, paged cache 8192) decodes
"sort of slow", the GPU looks underused, and built-in MTP (`runtime.mtp`) makes it slower. The demo sends
`temperature 0.7, top_p 0.95` with no `top_k`.

## Run 1 - 2026-10-12 10:05

Question: does MTP help this checkpoint? Python bindings over the same `libinference_ffi`, the demo's spec
(`GGUF` from the model's directory, `mmproj_selection: any`), chat request `temperature 0.7, top_p 0.95`, 400 max
tokens, two timed requests after a warmup.

| Config | tok/s | Drafts accepted |
|---|---|---|
| no MTP | 27.2, 27.1 | - |
| MTP default (3 drafts) | 18.2, 17.1 | 434 / 675, per position 182 / 142 / 110 |
| MTP n_predict 2 | 18.2, 18.7 | 376 / 552 |
| MTP n_predict 1 | 21.2, 20.9 | 285 / 345 |

Acceptance is fine (1.83 tokens per step at n_predict 1), yet MTP is slower, so each verify step costs ~2.4x a plain
decode step. First guess was the verify forward (GDN multi-token path, IQ4_XS mmq at n=2..4); Run 3 shows otherwise.

## Run 2 - 2026-10-12 10:20

Question: is something running on the CPU or streaming weights from system RAM? `nvidia-smi dmon -s puct -d 1`
alongside the decode, `top -H` on the process, 1000-token requests.

Without `top_p` (temperature 0.7 only): 46.7 tok/s, SM 99%, memory 98-99%, PCIe rx 60-90 MB/s. No weight streaming
(per-token weight reads from host would be GB/s); the 16.6 GB RSS is the mmapped GGUF.

With `top_p 0.95`: 27.2 tok/s, SM and memory ~55-64%, PCIe tx ~70 MB/s, one thread pinned at 100% CPU.

So the slowness is the sampler, not placement. `Sampler::cuda_batch_sampling_plan`
(`crates/inference-nn/src/sampler.rs`) returns `None` for top-p or min-p without top-k, which sends the row to the CPU
sampler: full-vocab logits (~248k) copied to the host and sorted per token. 1/27.2 - 1/46.7 = 15.4 ms per token.

## Run 3 - 2026-10-12 10:35

Question: which sampling combinations leave the GPU path, with and without MTP? 600-token requests, one engine per
mode, temperature 0.7 throughout.

| Sampling | no MTP | MTP |
|---|---|---|
| temperature only | 46.8 | 23.8 |
| top_p 0.95 | 27.2 | 16.5 |
| top_k 40 + top_p 0.95 | 36.0 | 45.4 |

Three separate gaps:
- top-p / min-p without top-k has no GPU plan at all (CPU sort per token, worse for MTP since every verified row
  pays it).
- MTP with plain temperature sampling (Categorical kind) is slower than no MTP, so the speculative plan does not take
  Categorical and its verify sampling also falls to the CPU path.
- Plain decode with top-k is 10 ms/token slower than Categorical (36.0 vs 46.8), so the non-speculative top-k path has
  its own host round trip or sync that Categorical avoids.

MTP on the GPU path (top_k + top_p) reaches 45.4, only level with plain temperature decode: this dense checkpoint
does not get the ~1.4x the 35B-A3B MoE got (`qwen35_moe_gguf_mtp_investigation.md`), at least with sampling on.

Next: exact GPU top-p/min-p (top-128 kernel output carries the full-vocab normalizer, so the nucleus is exact whenever
the top-128 mass reaches top_p, with a CPU fallback for the rare row that does not); Categorical in the speculative
plan; find the top-k decode round trip.

## Run 4 - 2026-10-12 11:10

Question: what does llama.cpp get from the same file, and which of its flags matter? Mainline `8bb3cf0d2` (pulled
today), `build_cuda` (Release, FA on, CUDA graphs on). Same RTX 3090, nothing else on the GPU.

`llama-bench -m <file> -ngl 99 -fa 1 -p 512 -n 128 -r 3`: pp512 1508.9 tok/s, tg128 44.1 tok/s (14.62 GiB, about
69% of the 3090's bandwidth). Our plain decode on the GPU sampling path (temperature only) is 46.8, so raw decode is
level or slightly ahead.

`llama-server -m <file> -ngl 99 -fa on -c 8192 -np 1 [flags]`, the Run 3 prompt, 600 max tokens, temperature 0.7,
two requests per row, `timings.predicted_per_second` (decode only, excludes prompt). Drafts are from the second
request.

| Server flags | greedy | temp only | top_p 0.95 | top_k 40 + top_p 0.95 | llama.cpp defaults |
|---|---|---|---|---|---|
| plain | 43.2, 43.1 | 38.5, 29.2 | 26.3, 30.4 | 27.2, 26.0 | 24.6, 27.4 |
| draft-mtp n-max 3 | 67.0, 52.5 (366/696) | 39.9, 43.1 | 41.2, 46.3 | 39.5, 34.1 | 38.4, 44.8 |
| draft-mtp n-max 2 | 69.1, 57.9 (361/474) | 48.1, 48.8 | 56.8, 48.1 | 49.8, 59.0 | 55.8, 57.3 |
| draft-mtp n-max 1 | 62.3, 55.7 (255/343) | 49.5, 54.5 | 50.7, 50.6 | 53.3, 54.0 | 52.9, 55.9 |
| `--backend-sampling` | 44.0, 43.8 | 32.7, 34.2 | 36.2, 34.5 | 36.4, 36.3 | 34.7, 36.9 |
| `--backend-sampling` + draft-mtp n-max 2 | 80.1, 79.6 (361/474) | 64.5, 59.0 | 61.3, 67.0 (348/502) | 50.9, 58.7 | 58.4, 60.4 |
| `--backend-sampling` + draft-mtp n-max 1 | 66.4, 55.9 | 48.4, 58.9 | 54.5, 57.4 | 47.9, 56.9 | 50.4, 52.6 |

llama.cpp's own CPU sampler costs it the same way (43 greedy vs 26-38 sampled plain). `--backend-sampling`
(experimental) recovers part of it plain and most of it under MTP. Best config: `--backend-sampling --spec-type
draft-mtp --spec-draft-n-max 2`, 80 tok/s greedy and 61-67 at the demo's top_p 0.95. So MTP on this dense
checkpoint is worth ~1.4-1.8x when sampling stays on the GPU; our 45.4 (MTP, top_k path) is well short of it.

`--backend-sampling` + draft-mtp n-max 3: greedy 70.7, 70.7 (366/696); temp only 56.4, 56.9; top_p 0.95 65.5, 52.3;
top_k 40 + top_p 0.95 49.1, 51.0; defaults 62.8, 57.5. Worse than n-max 2 except on noise.

## Run 5 - 2026-10-12 11:45

Question: our engine on the same matrix. Python bindings, the demo's spec, `runtime.mtp.n_predict` 1-3, same prompt
and request shape as Run 4, two requests per row, `usage.avg_compl_tok_per_sec`.

| Ours | greedy | temp only | top_p 0.95 | top_k 40 + top_p 0.95 | drafts (whole run) |
|---|---|---|---|---|---|
| plain | 42.4, 33.9 | 34.4, 36.8 | 27.8, 27.6 | 43.6, 34.8 | - |
| MTP 1 | 74.0, 74.7 | 26.5, 26.0 | 21.4, 20.0 | 53.6, 49.0 | 2099 / 2705 |
| MTP 2 | 84.1, 77.4 | 22.8, 24.0 | 17.8, 18.2 | 55.0, 50.7 | 2696 / 4216 |
| MTP 3 | 86.9, 88.6 | 24.5, 22.7 | 16.6, 16.1 | 52.7, 54.8 | 3029 / 5313 |

Greedy MTP beats llama.cpp's best (88.6 at n_predict 3 vs 80.1 at n-max 2), so the verify forward is fine and Run 1's
guess about it was wrong: the losses are all sampling. Where sampling leaves the GPU (temp only and top_p under MTP)
we drop to 16-26, against llama.cpp's 59-67 with `--backend-sampling`. top_k + top_p stays on the GPU and gets
49-55, still under llama.cpp's 51-59, so the sampled verify path has a cost of its own beyond the CPU fallback.

Caveat: the plain rows are noisier than Run 3 (temp only 34-37 here vs 46.8 there; Run 3 timed wall clock, this
uses the engine's own figure), so plain decode needs a cleaner repeat before reading anything into a few tok/s.

## Run 6 - 2026-10-12 12:40

Question: where do the CPU fallbacks move data? `nsys profile -t cuda,osrt` on one 1500-token MTP request (n_predict 2,
temperature 0.7) against the same request greedy, exported to sqlite, `CUPTI_ACTIVITY_KIND_MEMCPY` grouped by size.
(perf and eu-stack/gdb attach are blocked by perf_event_paranoid 4 and ptrace_scope 1; I lost time on them before
going back to the nsys method in `gguf_iq_trellis_quants_investigation.md`.)

| | temperature 0.7 | greedy |
|---|---|---|
| tok/s | 24.6 | 66.0 |
| D2H copies of 993280 bytes (one f32 248320-vocab row) | 3940 (3.9 GB) | 37 |
| all D2H | 4.5 GB | 0.67 GB |
| H2D | 16.3 GB (load) | 16.3 GB (load) |

Attribution with gdb as the launching parent (conditional breakpoint on `cuMemcpyDtoHAsync_v2`, `$rdx == 993280`, 200
token request): the full rows come from `Sampler::speculative_target_probs` (CPU verify), the MTP drafter
(`speculative_propose` -> `sample_draft_rows`, which runs `Sampler::sample` per draft row) and the plain sampling
fallback. The Qwen3.5 MTP drafter returns `SpeculativeProposal::with_logits`, which the GPU verifier never takes
(`SpeculativeProposalDistribution::Logits(_) => false` in `can_batch_device_verify`), so every sampled MTP request
verifies on the CPU. DFlash already proposes `DeviceSparse` and does not have this problem. `draft_sampling: greedy`
does not help (24.8 vs 24.9): the drafter is not the deciding factor, the proposal kind is.

## Run 7 - 2026-10-12 12:55

Question: is the top-k resident path itself slow (Run 3 had top_k + top_p at 36 against 46.8 temperature only)?
Interleaved rounds on one engine (each config once per round, 4 rounds, 400 tokens), clocks logged per round.

| | r0 | r1 | r2 | r3 |
|---|---|---|---|---|
| greedy | 48.3 | 48.1 | 48.4 | 49.8 |
| temperature only | 47.8 | 36.8 | 43.9 | 41.6 |
| top_k 40 | 48.1 | 35.0 | 40.0 | 39.2 |
| top_k 40 + top_p 0.95 | 47.8 | 39.4 | 40.3 | 39.2 |
| top_p 0.95 | 27.8 | 27.8 | 28.1 | 28.5 |

SM clock 1875-1920 MHz and 64-65 C throughout, so not thermal. Greedy is stable; the sampled GPU modes swing 35-48
between requests for a reason not yet found (output differs per request), so earlier single-request gaps between
temperature-only and top-k were noise. top_p alone is the solid outlier.

Plan (exact sampling throughout, no approximation of the target distribution):
- Draft side: the MTP drafter samples on the GPU from its ranked top-128 with the request's filters and proposes
  device sparse probabilities. Rejection sampling stays exact for any q the draft was actually drawn from, so
  truncating q is safe.
- Plain decode, top_p / min_p without top_k: a nucleus kernel on the full row (full-vocab normalizer, top-128 fast
  path, threshold bisection over the row for the rare nucleus wider than 128).
- Verify, same filters: the top-k rejection kernel treats top_k 0 as the full vocabulary using a per-row normalizer,
  and reports NEEDS_CPU (the existing per-row fallback) only when the nucleus is wider than the packed top-k.

## Run 8 - 2026-10-12 13:50

Change: `CudaBatchSamplingKind::Nucleus` for top-p / min-p without top-k. It rides the resident top-k path with
`top_k = 0`: `sample_ranked_topk` skips those rows and `sample_nucleus` (one 1024-thread block per row, `sort.cu`)
takes them. It sums the full-vocab normalizer, samples from the ranked top-128 whenever the nucleus fits inside it
(the CPU sampler's exact rule), and otherwise bisects a score threshold over the row and samples the kept set in
vocabulary order. The test `cuda_resident_nucleus_sampling_matches_full_vocab_reference` checks the fast rows token
for token against the CPU semantics and the three wide rows (nucleus of thousands) against the kept set and CDF.

First measurement after the kernel: top_p still 26 tok/s. The step only keeps the logits batched on the GPU when
`can_sample_batch_cuda` passes (`preserves_causal_generation`), and the batch path declines nucleus rows, so the
resident path never saw them. `preserves_causal_generation` now also accepts `can_submit_cuda_token_batch_seqs`.

Interleaved rounds after both changes (plain decode, 400 tokens):

| | r0 | r1 | r2 | r3 |
|---|---|---|---|---|
| greedy | 45.2 | 37.3 | 39.1 | 41.2 |
| temperature only | 44.8 | 34.8 | 39.2 | 41.3 |
| top_k 40 | 44.3 | 34.4 | 39.2 | 41.2 |
| top_k 40 + top_p | 44.4 | 35.6 | 39.1 | 41.0 |
| top_p 0.95 | 43.2 | 37.0 | 38.9 | 39.8 |

top_p went from 26-28 to level with the other modes. Rounds 1-3 are slower for every mode alike: SM clock fell to
1230-1350 MHz at ~267 W (from 1695 MHz at 360 W in round 0), so something outside the run lowered the clocks; only
within-round comparisons hold.

## Run 9 - 2026-10-12 15:10

Changes since Run 8:
- MTP drafting on the GPU: `CudaDraftSampler` (`speculative/proposer.rs`) uploads every draft step's sampling
  parameters once per proposal, and `sample_ranked_topk_distribution` (`sort.cu`) draws each step from the ranked
  top-k (128 for top-k 0) with the request's filters, writing the q it drew from. The Qwen3.5 drafter proposes
  `with_sparse_probs` instead of full logits, so the device verifier takes it. Greedy rows keep the old path.
- Verify, top-p / min-p without top-k: `CudaSpeculativeSamplingPlan::uses_ranked_target` puts those rows in the
  ranked kernel with `BoundedTopK { nucleus_rows: true }`. The verifier then sums each target row's full
  normalizer, and `prepare_topk_distribution` treats top-k 0 as the whole vocabulary: exact when the nucleus fits
  the packed 128, NEEDS_CPU (the existing per-row fallback) when not. Test:
  `nucleus_rows_match_full_width_top_k_and_fall_back_when_wide` (equal to top-k = vocab on a 5-token vocabulary;
  flat 300-token rows report NEEDS_CPU, a peaked row does not).

After the drafter change alone (verify not yet changed), interleaved, 400 tokens:

| MTP 3 | r0 | r1 | r2 |
|---|---|---|---|
| greedy | 78.1 | 77.0 | 76.7 |
| temperature only | 55.2 | 59.3 | 58.3 |
| top_k 40 | 67.8 | 60.5 | 59.3 |
| top_k 40 + top_p | 58.0 | 61.3 | 62.3 |
| top_p 0.95 | 31.0 | 30.6 | 30.8 |

With the verify change too:

| | MTP 3 r0 / r1 / r2 | MTP 2 r0 / r1 / r2 |
|---|---|---|
| greedy | 77.2 / 76.2 / 75.9 | 73.9 / 70.9 / 68.4 |
| temperature only | 55.0 / 58.9 / 57.0 | 59.9 / 53.0 / 50.3 |
| top_k 40 | 67.7 / 59.8 / 58.3 | 69.4 / 64.1 / 61.0 |
| top_k 40 + top_p | 57.6 / 60.4 / 61.4 | 62.2 / 68.2 / 65.4 |
| top_p 0.95 | 56.1 / 55.2 / 53.6 | 53.9 / 54.9 / 49.5 |

Before these changes (Run 5): MTP temperature only 23-25, top_p 16-18, top_k + top_p 51-55. The demo's setting
(top_p 0.95) is now ~55 with MTP 3 against llama.cpp's 61-67 (`--backend-sampling`, draft-mtp n-max 2), and 43-45
without MTP (was 26-28).

Greedy check, since Run 5 had greedy MTP 3 at 87-89: same script, 3 requests each, base = the projector-fix commit.

| | 400 tokens | 600 tokens |
|---|---|---|
| branch (first) | 76.8 76.4 75.9 | 80.9 79.5 79.1 |
| base | 78.4 78.4 78.1 | 84.2 85.1 83.4 |
| branch (again, after base) | 77.4 77.5 77.2 | 81.8 83.0 83.0 |

The first gap was mostly run order; the rerun is within 1-2% of base, inside run-to-run variation (the card sits at
its ~357 W cap throughout). Nothing in the change touches the greedy path, so I am not chasing it further now. All
of today's numbers sit below Run 5's 87-89, so the card itself is running slower than this morning.
