# Control vectors (#1062)

## Run 1 - 2026-11-24 07:30

How llama.cpp defines them (`common/common.cpp` `common_control_vector_load_one`, `src/llama-adapter.cpp`,
`src/models/llama.cpp`): a file's `direction.N` (F32, `[n_embd]`, N from 1; layer 0 is never steered) is summed over
files at their scales and added to 0-based layer N's output after its residual (`build_cvec(cur, il)`, `l_out`),
within `--control-vector-layer-range` (inclusive). Ours matches: `ControlVectorSet::combined` sums the files, the
shared decoder adds a row after each layer before its `after_layer` hook (taps and deepstack see the steered state), and
Inkling 2 adds it after each layer visit (its looped core's every pass). llama.cpp's Inkling 2 applies no control
vectors, so there is no reference for that placement.

Rows live one per layer on its device at the activations' dtype; a new set is written into them in place
(`slice_set`), so graphs that read them keep their nodes; attaching where none was, or removing, clears the graphs.
Changes run between engine steps (`Request::ControlVectors`). Cached states are keyed on the active set
(`ExtraHashKey::ControlVectors` in block hashes, `StateKey` in the non-paged cache), and steered sequences stay out of
the disk tier, as adapter sequences do.

Tiny Llama tests (`control_vectors_tiny`, CPU f32 and CUDA bf16): a zero vector and a range excluding the only steered
layer score bitwise as unsteered; a direction at scale 2 scores bitwise as the doubled direction at 1; a set on a loaded
model scores as a load with it, and clearing restores the unsteered bits; on CUDA a new scale replays the captured
decode graphs (5 replays, 0 captures) and decodes the fresh load's tokens.

## Run 2 - 2026-11-24 08:40

`scripts/control_vector_parity.sh` on TinyLlama-1.1B-Chat Q4_K_M, both engines on the CPU: llama.cpp's
`cvector-generator --method mean` (happy against sad, TinyLlama's template), then a greedy raw completion of 32 tokens.

```
setting (scale, layers)   plain          steered
0.8, 10-21                same text      differ from token 3 ("I am thrilled" / "I am delighted"), both happy
0.8, 21-21                same           same (direction.21 is small: both unchanged)
2.0, 21-21                same           same
0.4, 10-21                same           differ from token 2 ("I'm excited" / "I am delighted"), both happy
0.8, 5-15                 same           same (degenerate "Book at the best at our end ...", token for token)
1.5, 15-21                same           differ late, both happy
2.0, single layer 12..17  same           differ; 18..20: same (both unchanged)
f32 KV on both (-ctk f32 -ctv f32, INFERENCE_RS_CPU_KV_F32=1): the same splits
our layer L against theirs L-1, L, L+1 (layer 16): no match, so no index offset
```

At the split (layer 16, scale 2, `llama-server` n_probs against our logprobs): llama.cpp " am" -1.071 / " do" -1.296,
ours "do" -1.146 / "am" -1.168. Unsteered, the same positions already differ by 0.1-0.5 nats (" not" -0.767 against
-0.487, "As" -2.609 against -2.774): the Q4_K CPU paths' own difference, which the plain greedy texts survive only by
their margins. The steering moves both the same way at the split (" do" +0.25 there, +0.08 here; " am" -0.09, -0.05).
So the steered splits are near-ties the engines' baseline difference breaks apart, not a different steering: the
saturated 5-15 run, where the vectors dominate, agrees token for token.

## Run 3 - 2026-11-24 09:10

Inkling 2 (recipe C, drafts 7 auto, q8 KV, 65,536) through the manager. llama.cpp's `cvector-generator` cannot make
an Inkling vector (it asserts one `l_out` per layer, and the looped core visits layers 6-17 three times), so a random
direction (`N(0, 0.5)`, layers 6-17) and a zero one, set on the loaded model:

```
zero vector, scale 3                      greedy text == unsteered
random, scale 1                           steered (a numbered list in place of the answer)
random, scale 2                           steered further; 0 graph captures across the scale change
cleared, auto depth                       text differs from the first unsteered run
cleared, no drafter / fixed depth 7       == unsteered (set, steer, clear on one loaded model)
unsteered repeated (auto depth)           identical each time; drafter against no drafter differ (as before vectors)
```

The cleared run's difference under auto depth is the tuner: it learned from the steered steps and drafts at other depths
after, and the target's verify numerics follow the depth (the drafter-against-none difference). With the depth fixed,
or no drafter, clearing restores the unsteered text exactly, so no state of the vectors outlives them.
