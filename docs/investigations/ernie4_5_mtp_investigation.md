# ERNIE 4.5 MTP drafting through the shared decoder

ERNIE-4.5-21B-A3B ships one MTP layer (`mtp_emb_norm.0`, `mtp_hidden_norm.0`, `mtp_linear_proj.0`, `mtp_block.0`)
that llama.cpp's converter drops. The shared decoder's `MtpHead` drafts with it; these runs measure whether drafting
pays on a 3090.

Setup for every run: `ERNIE-4.5-21B-A3B-PT-Q4_K_M-mtp.gguf`, the Q4_K_M GGUF with the HF MTP tensors added by
`scripts/gguf_add_hf_tensors.py` (BF16, checkpoint revision 87db9548), through `inference serve --max-model-len 8192`
(dev profile, CUDA). The bench is one warm-up and three timed 256-token completions of a code prompt at temperature
0.7 (top_p 0.8) and then 0.0, read from the response's usage and the server's speculative counters.

## Run 1 - 2026-10-30 (time approximate)

**Question:** does an MTP draft beat plain decode at all, and where does the step's time go?

**Command:** the bench without `--mtp`, with `--mtp --mtp-n-predict 1` and with `--mtp-n-predict 2`, as the drafter
gained device sampling and CUDA graphs.

**Finding:**
- The first drafter, which sampled its drafts on the host, ran at 51 tok/s sampled. With the drafts sampled on the device
  (`CudaDraftSampler`) it ran at 178 tok/s.
- After the drafter's forward ran as a CUDA graph (`MtpGraphs`), against plain decode's ~196 sampled / ~211 greedy:

| Setup | Sampled | Greedy | Acceptance |
|---|---|---|---|
| 1 draft | 192-203 | 219-242 | ~90% |
| 2 drafts | ~185 | ~220 | 77-81% |

- With about 1.9 tokens a step at 90% acceptance, 220 tok/s is about 8.6 ms a step. That is far more than the GPU
  needs.

**Next:** profile one request to see whether the GPU or the host is the limit.

## Run 2 - 2026-10-30 (time approximate)

**Question:** where does an MTP step's wall time go?

**Command:** `nsys profile -t cuda,osrt --cuda-graph-trace=node` over serve with `--mtp --mtp-n-predict 1`, two greedy
requests, exported to sqlite. The second request was read for kernel spans, graph launches and API calls between
them. Then the same profile without `--mtp`.

**Finding:**
- Busy and idle time (1186 ms wall): the GPU was busy for 721 ms (graphs 647, eager 74) and idle for 465 ms.
- GPU work per step:
  - the verify graph: 800 kernels, 6.4 ms;
  - the drafter graph: 26 kernels, 0.37 ms;
  - then, outside the drafter graph: the lm_head (`mmvq_gguf_q6_k`, 0.28 ms), a cast and the top-1 kernels.
- Two host round trips per step:
  - the verify graph, then the host reads its tokens back, then the host launches the drafter graph;
  - the drafter's eager lm_head and sampling, then the host reads the drafts back, uploads the verify inputs and
    launches the verify graph.
- Each read-back's `cuEventSynchronize` returned about 330 us after its copy finished on the GPU.
- Launch times under node tracing:
  - The verify graph's `cuGraphLaunch` blocked the host for 0.6 ms at a request's start. It rose over about a dozen
    steps to 3.4-3.6 ms, with the GPU idle for the whole call.
  - Plain decode's 852-kernel graph shows the same climb (0.6 to 2.1-2.8 ms), yet unprofiled plain decode runs at
    4.74 ms a token against 4.6 ms of graph GPU time.
  - So the climb is the node tracer's per-node overhead, not a real cost. The default graph-level trace collected no
    CUDA events at all, under both nsys 2024.6 and 2026.3 with driver 13.4.
  - `CUDA_MODULE_LOADING=EAGER` changed nothing.
- The real gap: about 6.8 ms of GPU work a step against about 8.3 ms measured, so about 1.4 ms of host time a step,
  much of it in the late wake-ups.

**Next:** the late wake-ups. The token read-back ring (`ops/top1.rs`) and the sparse rejection workspace
(`cuda/speculative_rejection.rs`) record `CU_EVENT_BLOCKING_SYNC` events, so the engine thread sleeps on them and is
woken late. This came from upstream with no reason given.

## Run 3 - 2026-10-30 21:27

**Question:** does waiting on those events by spinning instead of sleeping close the gap?

**Command:** a temporary `INFERENCE_RS_SPIN_READBACK` switch that created the read-back events with
`CU_EVENT_DISABLE_TIMING` (a spinning wait), first in the token ring, the rejection workspace and the graph-copy
completions (`cuda/graph_capture.rs`), then without the graph-copy completions. Each run was the bench, sleeping
against spinning.

**Finding (tok/s, three runs each):**

| Setup | Sleeping, sampled | Spinning, sampled | Sleeping, greedy | Spinning, greedy |
|---|---|---|---|---|
| plain | 194-199 | 199-200 | 210-211 | 212-213 |
| 1 draft | 187-188 | 204-207 | 218-222 | 252-254 |
| 1 draft, graph copies sleeping | | 204-208 | | 254 |
| 2 drafts | 181-184 | 197-198 | 221-224 | 255-257 |

- Spinning on the token and rejection read-backs is what helps. The graph-copy completions make no difference and stay
  asleep.
- Acceptance is unchanged (362 of about 402 drafts at 1 draft).
- MTP now beats plain decode in both modes: +3% sampled and +19% greedy at 1 draft.
- The cost is one CPU core spinning while a decode step waits.

**Next:** the remaining host time per step (about 0.7 ms at 254 tok/s greedy):
- the drafter's lm_head and sampling run eagerly outside its graph;
- the host reads the drafts back before every verify, although the verify inputs could take them on the device.

## Run 4 - 2026-10-30 21:59

**Question:** does leaving the drafts on the device, so the verifier reads them back with its own outputs, remove the
second round trip?

**Command:**
- The change: the head's propose keeps each step's drafts as a `[batch]` U32 tensor and returns them as
  `SpeculativeTokens::Device`.
  - Sampled rows take them from `CudaDraftSampler::sample_on_device`.
  - Greedy rows take them from the packed top-1 kernel's id column.
  - A chained step feeds them to the graph's token buffer by a device copy (`MtpGraphTokens::Device`), clamped below
    the vocabulary so an invalid id never reaches the embedding.
- The bench at one and two drafts, then the node-traced profile at one draft.

**Finding (tok/s, three runs each):**

| Setup | Sampled | Greedy | Run 3, sampled | Run 3, greedy |
|---|---|---|---|---|
| 1 draft | 208-209 | 255-259 | 204-207 | 252-254 |
| 2 drafts | 202-205 | 258-260 | 197-198 | 255-257 |

- In the profile, the drafter's lm_head and sampling run straight into the verify graph, which starts about 35 us after
  them.
- The only host gap left is about 130 us after the verify graph, where the host reads the verify outputs and the
  drafts together and decides what to accept.
- A step is now about 6.8 ms of verify, 0.37 ms of drafter graph and 0.3 ms of drafter lm_head (`mmvq_gguf_q6_k` over
  the 103K vocabulary). At about 1.9 tokens a step, 257 tok/s is about 7.4 ms a step, so the GPU is the limit.
- Moving the drafter's lm_head and sampling into its graph would now only save launches that already overlap GPU work.
- Acceptance is unchanged at one draft (363 of 402 sampled). At two drafts it is 461 of 604 sampled positions.

**Next:** nothing on the host side. The rest is GPU time:
- the verify graph's two rows touch more experts than plain decode's one (6.4 ms against 4.6);
- the drafter repeats the full-vocabulary lm_head.
