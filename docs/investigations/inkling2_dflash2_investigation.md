# Inkling 2 Mini with the DFlash 2 drafter

`incoai/Inkling-2-Mini-DFlash2` (Apache-2.0): a 5-layer DFlash 2 drafter (hidden 2304, all layers sliding at 2048,
block 8, selector rank 256 / top-k 16, two-tap convolutions) tapping Inkling 2 Mini's layer visits `[5, 23, 41, 48]`
(48 = the normed output), with `final_logit_softcapping` 30 and `selector_softcap_mode: after_correction` in its
`dflash_config`. Its README ships an sglang patch for the pinned Inkling image: DFlash captures alias Inkling's DSpark
captures, the selector's candidates are cut to the real vocabulary (201,024 is padded storage), and the target's
softcap goes on the selector's corrected scores. Checkpoint in `/mnt/data/models/Inkling-2-Mini-DFlash2`; bench
harness (`run.sh`, `bench.py`, load JSONs) in `/mnt/data/models/dflash2-eval`.

Bench: a dev server per load (`serve -p 18083`), five prompts (prose, code, story, explainer, a GSM8K item), 512
tokens at most, thinking off, seed 1, one request at a time; tok/s is completion tokens over wall time, acceptance
from `/metrics` deltas. Loads are the LAN chat load (inkling2-mini-official, q8 KV, LoopCD-Logits 0.5, drafts 7
auto, q8_0 drafter) at a 32K context without vision, unless noted.

## Run 1 - 2026-11-25 20:50

Question: does the drafter load and draft as-is?

It loaded (the Inkling target already attaches external DFlash drafters) but accepted nothing: `accepted/draft 0.00`
on every prompt, 76 tok/s mean, auto depth soon stopping drafting. A temporary trace of the drafted ids on "Count
from 1 to 30" gave noise (`[14, 105350, 177505, 41205, 200196, ...]`). Without the q8_0 drafter: the same noise.
Tapping visit outputs instead of inputs (`run_visit(id + 1)`): nearly the same noise, position 0 unchanged, so the
taps were not what decided it. A trace of the drafter's inputs: taps mean |x| 2-3 at the four ids, the embedded
block 0.65 (Inkling's `embed()`, with `embed_norm`); the raw table gives 0.79 and the drafts did not change at all.

With `INFERENCE_RS_CUDA_GRAPHS=0` the drafts were right (`[11, 220, 17, 11, 220, 18, 11]` = ", 2, 3,") and every
row was kept. So: the eager path works, the captured one does not. A DFlash 2 drafter (selector) whose layers are
all windowed skips the paged graph path and drafts through `proposals_cuda_graph`, whose `cuda_graph_output` embedded
the block with the raw `token_embedding` table; Inkling embeds through `embed_tokens` then `embed_norm`. Eager with
the raw table (a temporary switch) reproduced the noise bit for bit. The DFlash 2 drafter was trained on the normed
embedding (as sglang's `get_input_embeddings` gives it for Inkling).

Fix: the windowed graph path (and its precapture) takes a `TargetEmbedding`: the table, which still keys the captured
graphs, and the target's embedding function, which the capture runs. The paged graph path already did this.

Implies: measure, then the vocabulary cut and the softcap.

## Run 2 - 2026-11-25 21:10

Question: how does it compare, and does sampling work?

```
greedy (mean of 5)                  tok/s   accepted a draft
DSpark (the checkpoint's), auto     98.3    1.33
DFlash 2, auto                      101.7   1.52
DFlash 2, drafts 7 fixed            99.6    2.10
DFlash 2, drafts 7, no LoopCD       108.5   1.99
plain                               84.3
```

(The first post-fix auto run read 0.00 accepted; reruns all drafted. Not reproduced.)

Sampled (temperature 1) failed: `sparse CUDA verification rejected invalid proposal probabilities`. The target's
logits are cut to its 200,058 real ids (`unpadded_vocab_size`); the drafter's candidates came from all 201,024 rows of
the head, so a padded id could be proposed. Fix: `DFlashLoadTarget::logit_vocab` (Inkling passes
`unpadded_vocab_size`, the others none); the selector's codebooks are cut to it at load and the draft logits before
the selector. Then the softcap: `dflash_config.final_logit_softcapping` with `selector_softcap_mode` (default
`before_correction`, on the logits; `after_correction`, on each candidate's corrected score). Greedy picks keep their
order under a cap after the correction, so only the sampling kernel takes it (`softcap` argument, 0 = none).

```
sampled, temperature 1              tok/s   accepted a draft
DSpark, auto                        83.7    0.74
DFlash 2, auto                      107.0   1.88
DFlash 2, drafts 7 fixed            99.1    2.19
plain                               82.0
DFlash 2 auto, cap before           97.5    1.24
DFlash 2 auto, no cap               101.9   1.67
```

The configured mode (after the correction) is the best of the three. Implies: DFlash 2 at auto depth is the better
drafter for this load, most of all sampled (Inkling's default temperature is 1): +28% over DSpark, +30% over plain.

## Run 3 - 2026-11-25 21:40

Question: does a test catch these?

The tiny-weight greedy-output tests cannot: verification keeps the output however bad the drafts. A deep check,
`inkling2_dflash2::dflash2_drafts_land_greedy_and_sampled_on_cuda` (`INFERENCE_TEST_INKLING2_MODEL`,
`INFERENCE_TEST_INKLING2_DFLASH2`), counts to 60 greedy and sampled and requires acceptance 0.5 / 0.3: it read 1.00
and 1.00, and 0.01 / 0.00 with the raw-table embedding put back. Without the vocabulary cut it still passed (0.84):
counting rarely reaches the padded ids, so unit tests pin the cut and the softcap modes
(`a_padded_head_is_cut_to_the_target_vocabulary`, `the_selector_softcap_goes_where_its_mode_says`) and the kernel
(`cuda_dflash_sample_selector_caps_the_corrected_scores`).
