# Vulkan MTP decode: why speculative decoding loses to plain decode

Nemotron 3.5 Lightning (`Lightning-exps-q4_0.gguf`) on the RTX 3090 over Vulkan, `serve --vulkan` on master 1de725144.
The Code tab demo took 71-97 s here against 41-47 s on CUDA, which runs MTP 3 with a q4_0 draft head.

## Run 1 - 2026-11-05

**Question:** is there a tool-call grammar slowdown on Vulkan, as on CUDA before #827?

**Command:** `code/vk_call.py` (a 400-word story as prose and as a `write` call, greedy and sampled, twice) against
`serve --vulkan`, then with `--mtp --mtp-n-predict 3 --mtp-draft-isq q4_0`, then `--mtp --mtp-n-predict 3`; kernels
by `INFERENCE_VULKAN_PROFILE=1`.

**Finding:**

| decode tok/s | prose | tool call |
|---|---|---|
| no MTP, greedy / sampled | ~199 / ~199 | ~199 / ~199 |
| MTP 3, q4_0 draft head | ~30 | ~30 |
| MTP 3, draft head as stored (q8_0) | 100-108 | |

- No grammar slowdown: Vulkan already ranks a constrained row on the device.
- MTP acceptance is normal (len 2.1-2.3, CUDA's 2.04-2.13), so each verify step costs ~67 ms (q4_0 head) or ~20 ms.
- q4_0 head: `mat_vecs_q4_0 m=131072 n=1 k=2688` 17.9 ms a call, 80 % of GPU time. 200 MB at VRAM bandwidth is ~0.25
  ms: the head is in host memory. `set_draft_lm_head` re-quantizes the target's lm_head when MTP attaches, after the
  load sized the KV cache to leave 2 GiB; `Buffer::exact` wants a weight to leave 2 GiB too, so it goes to the host.
  Copying ISQ'd blocks at once (`QVulkanStorage::filled`) does not help: the attach is after the cache.
- q8_0 head (the stored lm_head, on the device: 0.42 ms a call): GPU busy 2.65 s of each 5 s window, 85K `copy_b32`
  dispatches a window. The host leaves the GPU idle about half of each step.

**Implication:** two fixes. (1) Make the draft head before the KV cache is sized, so the post-load fit accounts for it.
(2) Profile the host side of a Vulkan verify step and the copies it issues.

## Run 2 - 2026-11-05

**Question:** why does the q4_0 draft head land in host memory, and does reserving it fix that?

**Change, first:** `SpeculativeTargetMixin::mtp_draft_head_bytes` (Granite: the lm_head's weights at the ISQ's block
size), carried by `PagedAttentionConfig::draft_head_isq` and added to the KV fit as `late_weight_bytes` on top of the
activation reserve. The cache shrank 1555 -> 1383 MB; the head still ran 17.8 ms a call.

**Finding:** a temporary print in `Buffer::exact` at the head's copy: `len=189 MiB free=732 MiB idle=483 MiB
fits=false`. Nothing ran between the cache and the attach; `GgufMatMul::apply_isq` dequantized the q8_0 lm_head with
`q.dequantize(&q.device())`, which for Vulkan dequantizes on the host and uploads ~1.4 GB of f32 to the device, which
`generate_isq!` then copies back to the host to quantize. That temporary held the reserve when the q4_0 copy was placed.

**Change, second:** `QTensor::dequantize(&Device::Cpu)` of a Vulkan weight dequantizes its host-side blocks directly,
and `apply_isq` dequantizes a Vulkan weight to the host. With the reservation, the head is on the device.

| MTP 3, q4_0 draft head, greedy prose | before | after |
|---|---|---|
| decode | 28.7 tok/s | 112 tok/s |
| `mat_vecs_q4_0 m=131072` | 17.8 ms a call | off the top of the profile |
| GPU busy, 5 s window | 4.47 s | 1.80 s |

**Implication:** the step is now host-bound (GPU busy 36 %), with 55K `copy_b32` dispatches a window. Next: a CPU
profile of the server during MTP decode.

CI: `--lint --tests --cuda --vulkan` green, after two false starts. Vulkan tests failed while a test server still held
22 GB, and `llama_tiny::a_cuda_model_reports_its_memory` failed once in the full run, then passed alone 3/3 and in
a `--cuda` rerun (it measures device memory beside the other GPU tests).

## Run 3 - 2026-11-05

**Question:** where does a Vulkan verify step's host time go (GPU busy 36 %)?

**Command:** `perf record -F 999 -g` and `--call-graph dwarf` on the server during greedy MTP decode; dispatch counts
from `INFERENCE_VULKAN_PROFILE`.

**Finding:**
- One core busy the whole window; self time in the NVIDIA driver under `Stream::flush` and `Pipeline::record`
  (`pthread_rwlock_*` ~10 %), malloc, `clock_gettime`.
- ~156K dispatches a 5 s window, ~590 a verify step at ~53 steps/s: not more a second than plain decode at 199 tok/s.
- The device paths a CUDA step takes are CUDA-only: `greedy_device_drafts` (`cuda_top1_logits_f32_packed_batched`),
  greedy device verify (`submit_cuda_top1_batch_owned`), `CudaDraftSampler`, sparse rejection. On Vulkan each draft
  step and the verify read 131K-vocabulary logits to the host and decide there: 4 waits a step, the GPU idle through
  each, and the CPU's recording of the next stage never overlaps the GPU.

**Implication:** a Vulkan device path: top-1 on the GPU for greedy drafts and verify, drafts left on the device across
draft steps; then sampled drafts and verification.

## Run 4 - 2026-11-05

**Question:** how many times does a Vulkan verify step wait on the GPU, and where?

**Command:** a temporary counter in `Context::synchronize` printing every 97th call's source lines (`TMP_SYNC_LOG`),
over greedy MTP prose replies; then `INFERENCE_VULKAN_PROFILE`.

**Finding and changes, one sync site at a time** (MTP 3, q4_0 head, greedy, 481-token replies):

| step | syncs a reply | decode tok/s |
|---|---|---|
| Run 2 | 2716 (~12 a step) | ~112 |
| greedy drafts on the device (argmax, `RowTokens::Device`), no clamp of their ids | ~920 | ~133 |
| verify rows' argmax in one device call, read back at once (`vulkan_greedy_verify_batch`) | ~680 | ~143 |
| drafts read with the verify tokens; snapshot scatters by host index | ~194 (~1 a step) | 133-143 |

- `drafts.clamp(...)` on u32 ids synchronized each draft step on Vulkan; an argmax is always an id, so the Vulkan path
  drops it.
- `sampler.rs` greedy read each verify row's argmax with its own `to_scalar`.
- `copy_recurrent_row` (prefix snapshots) built device index tensors that the non-CUDA scatter read back.
- With one sync a step the GPU is busy ~65 % (3.24 s of 5 s): a verify step is ~12 ms of GPU work (plain decode ~5 ms a
  token): the 4-row verify forward, ~410 `copy_b32` (Mamba checkpoints) and ~130 `index_b32` a step, the draft steps.
  The rest is the CPU recording ~1,100 dispatches a step while the GPU waits.

**Implication:** MTP on Vulkan (~115-143 tok/s) still trails plain Vulkan decode (~199). What is left is GPU work a step
(the checkpoint copies, the verify forward's 4-row matvecs) and recording overlapping the GPU, not round trips.

## Run 5 - 2026-11-05

**Question:** with one sync a step, what holds the step: the host or the GPU?

**Command:** `perf record --call-graph lbr` on the server during greedy MTP 3 decode (DWARF unwinding lost the
stacks); `INFERENCE_VULKAN_PROFILE`; then an alternating A/B of plain decode and MTP 1/2/3 (6 replies each, the last 4).

**Finding, host:** 12.8 % of host time uploaded tiny tensors (`storage_from_cpu_storage` -> `Buffer::write` -> a recorded
copy): slot index tensors built in `MambaLayer::forward_full` (twice a Mamba layer, every step) and `Tensor::new(rows)`
in the commit replay's `pick` (four a layer). One core was busy for the whole ~18.5 ms step, ~16 us of host per
dispatch.

**Change:** `MambaLayer::identity_slots` keeps the slots for the last batch size; the replay builds each group's rows
once (none when the group is the whole batch). MTP 3: ~140 -> 163-169 tok/s.

**Finding, GPU:** busy ~84 % after it (under the profiler). Per step (~362 steps a window): expert matvecs 3.38 ms, q8
matvecs 2.56, `copy_b32` 1.19, q5 0.75, the draft head 0.61, row ops 0.57, router 0.39, scan 0.32; ~11.6 ms. Four tokens
route to ~22 distinct experts of 128 (6 each), so the experts' weights (~2.8 GB a step) are read at ~88 % of the 3090's
bandwidth: MTP 3's expert reads cost ~4x a plain token's, a ceiling of ~2.14 / 11.6 ms ~ 184 tok/s.

| greedy prose, mean of the last 4 | round 1 | round 2 |
|---|---|---|
| plain | 173.3 | 182.0 |
| MTP 1 | **194.5** | **188.4** |
| MTP 2 | 176.8 | 180.8 |
| MTP 3 | 158.4 | 156.8 |

(acceptance len: MTP 1 1.64, MTP 2 1.95, MTP 3 2.12)

**Implication:** on Vulkan the MoE expert reads, not host overhead, now bound the verify; one draft is the best
setting (+4-12 % over plain), three the worst. Further gains need fewer bytes a step (copies ~1.2 ms, the two
vocabulary projections ~1.2 ms), not fewer round trips.

## Run 6 - 2026-11-06

**Question, from the user:** sampled drafting and verifying on the Vulkan GPU.

**Command:** the sync counter over sampled MTP 1 replies (`code/vk_sampled.py`, `code/vk_tool_sampled.py`); then two
alternating rounds of `code/vk_call.py` against plain decode.

**Finding, before:** a sampled step synchronized ~3 times: the draft's ranked draw read back to draw on the host, the
verifier ranking the draft logits again (`new_ranked`) and the target rows. Under a tool-call grammar (no stochastic
verification) the draft drew on the host and each verify row ranked alone.

**Changes:**
1. The draft draw's own support is the proposal distribution (`sample_ranked_draft_rows`), so the verifier does not
   rank the draft logits again: 3 -> 2 syncs, speed unchanged (165-178).
2. `ranked_draw.comp`: a token drawn on the device from a ranked row under the sampler's filters (`ranked_support`'s
   rules; a nucleus past K cut at K), writing the support's ids and probabilities. The MTP chain ranks, draws and feeds
   each draft step there (`vulkan_sampled_drafts`), the support kept as the proposal's `SparseProbs`. Rejection stays
   exact: the verifier uses the support the draft was drawn from. `a_ranked_draw_matches_the_hosts_support` pins
   the support and the token at five uniforms against the host, over five filter settings.
3. A host-verified (grammar-constrained) Vulkan step ranks every verify row at once and draws each row's first token
   from its support (`sample_sequence_with_first`); the grammar check and masked resample run as before.
4. Constrained sequences draft on the device too: their verify reads only the tokens.

| Vulkan, MTP 1 vs plain, tok/s | greedy prose | greedy tool | sampled prose | sampled tool |
|---|---|---|---|---|
| plain, two rounds | 188-200 | 194-200 | 191-200 | 196-200 |
| MTP 1 before | 190-196 | 183-195 | 176-190 | 168-174 |
| MTP 1 after | 194-198 | 197-200 | 185-196 | 192-196 |

**Implication:** sampled MTP now runs as greedy MTP does, and MTP 1 matches plain decode on Vulkan in every case, within
the card's run-to-run spread (~5 %). It does not beat plain: the verify's expert reads bound it (Run 5).

## Run 7 - 2026-11-06 07:57

**Question:** the laptop's check of #830 on the 780M found three gaps: `local_ci.sh --vulkan` never ran the sampler's
Vulkan tests (its filter is `test(vulkan)` and their names lacked it), the draw had no multi-row device test, and the
engine log's decode T/s counted every scheduled draft position (MTP 3 logged ~42 when ~23 came out).

**780M results (laptop):** greedy MTP 1 and MTP 3 text equals plain decode on every prompt tried. Four prompts, 256
tokens, thinking off: plain master 25.6 / PR 25.5, MTP 3 master 31.7 / PR 33.3, PR MTP 1 30.6 tok/s. With thinking on
(700 tokens) MTP 3 falls below plain on both (22-24 vs 25), acceptance 33-55 %.

**Changes:**
1. The sampler's three Vulkan tests renamed to contain `vulkan` (`a_vulkan_ranked_draw_matches_the_hosts_support`,
   `vulkan_penalties_rank_as_the_cpu_does`, `vulkan_ranked_speculative_rows_match_the_cpu`); `nextest -p inference-nn
   --features vulkan -E 'test(vulkan)'` now runs 6 tests, all passing on the 3090.
2. `a_vulkan_ranked_draw_keeps_each_row_apart`: four rows with different filters and uniforms in one dispatch, each
   checked against the host's support for its row alone. Passes.
3. The paged step logs decode tokens as the growth of each row's generated length over the step, not its scheduled
   positions.

**Check (3090, Vulkan, MTP 3, q4_0 head):** one 3000-token essay, client 178.9 tok/s; logged decode in the full
windows 165.4 / 175.0 / 157.0 (each window also holds the step's host time and the request's tail). The four-prompt
bench: 245.4 / 218.6 / 194.3 / 303.3 tok/s, acceptance 72.6 % (len 3.18).

**Implication:** the log now tracks what clients see. The old binary used for comparison was gone, so the before
number here is the laptop's.
