---
title: Control vectors
description: Steer a model with per-layer directions added to its residual stream, at load or on a loaded model.
---

A control vector is a direction per layer added to the model's hidden state after that layer, scaled. It shifts tone, mood or persona without a fine-tune. The format is llama.cpp's: a file of `direction.<layer>` tensors (`[hidden]`, F32, layer 1 up; layer 0 is never steered), as written by llama.cpp's `cvector-generator` and by repeng. A safetensors file with the same tensor names works too.

## At load

```bash
inference serve -m <model> --control-vector happy.gguf:0.8 --control-vector-layer-range 10 21
```

`--control-vector FILE[:SCALE]` repeats: the files' directions add up, each at its scale (1 when none is given). `--control-vector-layer-range START END` limits the steering to layers START through END (0-based, inclusive). Layer N's direction is added to layer N's output after its residual, as llama.cpp does. The same fields are `runtime.control_vectors` in an engine spec, `control_vectors` on a model spec, and `control_vectors` in the model manager's load request, which the manager saves with a slot so it reloads steered. A model loaded lazily takes its vectors once it has loaded.

## Derived from texts

A vector can also be derived on the loaded model from contrasting texts, with no separate tool:

```bash
inference serve -m <model> --control-vector-texts happy.txt:sad.txt:6
```

The two files hold one text a line (`\n` for a newline inside one), and line N of the first pairs with line N of the second. This is the format llama.cpp's `cvector-generator` reads. In a spec or a request, the same is a `derived` entry: `{"positive": [...], "negative": [...], "scale": 6}`. The engine runs each pair through the model, records every layer's output, and takes the mean difference per layer with shared prefixes left out, normalized to unit length, as llama.cpp's `mean` method does. The result is a llama.cpp control vector GGUF, kept in `~/.cache/inference-rs/control-vectors` (or `INFERENCE_RS_CONTROL_VECTOR_DIR`) under a key of the model, its config overrides (such as a looped model's `run_passes`) and the texts. Setting the same texts again reads that file, and a set or a load reports a derived vector as its cached file.

Deriving takes one prompt pass per text, well under a second for a handful of short pairs on a GPU. The derived directions are unit length, so the scale is absolute: on Inkling 2 about 4 nudges, 8 is strong and 16 overwhelms the model. A negative scale steers the other way.

## In the web UI

On the Models tab, the Chat model slot (the Chat and Code tabs share it) has a **Steering** section: the texts to steer toward and away from (one a line, paired by line), a scale and an optional layer range (`6-17`). A load takes them with the slot, and the server keeps them with its settings. On a loaded model, **Apply now** derives the vector and steers the model from its next request without a reload. With steering unchecked, it stops.

## On a loaded model

```bash
curl localhost:1234/control_vectors/set -H 'Content-Type: application/json' \
  -d '{"model": "chat", "vectors": {"files": [{"path": "/models/happy.gguf", "scale": 1.2}], "layer_range": [10, 21]}}'
curl 'localhost:1234/control_vectors?model=chat'
```

`POST /control_vectors/set` is an admin route, since it reads files on the server's host. It changes the vectors from the model's next step. Empty `files` stops steering. `GET /control_vectors` reports what a model is steered by. The C ABI has `inference_control_vectors_get` and `inference_control_vectors_set`, and the Python and C# engines have `control_vectors` and `set_control_vectors`.

Changing the files or their scales on a model that is already steered rewrites its vectors in place. CUDA graphs captured for it keep running, with no capture again. Attaching vectors to a model that has none, or removing them, captures its graphs again on the next steps.

## Loop guidance

A looped model (Inkling 2, whose middle 12 layers run three times) can also be steered by loop guidance, LoopCD-Hidden
from [Decoding Looped Transformers Better for (Almost) Free](https://arxiv.org/abs/2610.02185): the last pass's output
is pushed away from the first pass's before the layers after the loop, `h + w (h - h1)`. The first pass is a weaker
prediction of the same token, so the push sharpens choices where the passes disagree. It costs one tensor and no
extra layers. The paper found about 0.2 to 0.3 safe for generation (larger values compound over the tokens); measure
on your own tasks before relying on it.

```bash
inference serve -m thinkingmachines/Inkling-2-Mini --loop-guidance 0.25
curl localhost:1234/control_vectors/set -H 'Content-Type: application/json' -d '{"model": "chat", "vectors": {"loop_guidance": 0.25}}'
```

It travels with the control vectors: `loop_guidance` beside `files` (on its own or with them), `--loop-guidance` at
load, **Loop guidance** in the web UI's Steering section for an Inkling 2 slot, `with_loop_guidance` in the Rust SDK.
A new strength rewrites the value its CUDA graphs read; attaching or removing it captures them again. A model that runs
each layer once refuses it.

### LoopCD-Logits

The paper's other form contrasts the two passes' predictions instead: the layers after the loop also run on the first
pass's output, and the logits become `z + w (z - z1)`. The strength adapts per token, `w = w_max (1 - (p1 - p2))` from
the last pass's top two probabilities, so it pushes hardest where the choice is close. The model's authors recommend
this form for Inkling 2 with `w_max = 0.5`. The extra layers keep their own KV cache, so the mode is chosen at load:

```bash
inference serve -m thinkingmachines/Inkling-2-Mini --loop-guidance-logits --loop-guidance 0.5
```

`with_loop_guidance_logits` in the Rust SDK. While loaded, decode runs about 11% slower at strength 0, and about 16%
slower with guidance on (the second read of the output head); the strength still switches live. The control tokens
(message, block and tool-call markers) keep the last pass's logits. On MATH-500 levels 4 and 5 (integer answers,
thinking on, a 6144-token thinking budget), Inkling 2 Mini went from 150 to 153 of 156 with `w_max = 0.5`.

## Notes

- Cached prompt states depend on the vectors, so the prefix cache keys them on the active set, and steered requests are not written to the disk prefix cache.
- Speculative decoding drafts from the steered states and verifies against the steered model. Greedy output stays exact against the target. Changing the vectors resets the `--mtp-auto-depth` tuner, since what it measured described the model as it was steered before.
- Inkling 2's looped core adds a layer's direction on each of its passes, and a derived vector pools the passes' outputs. llama.cpp applies no control vectors to Inkling 2, and its `cvector-generator` cannot build one for it; derive one here instead.
- Text models on the shared decoder and Inkling 2 take control vectors. Other model families and multimodal models refuse them for now.
- `scripts/control_vector_parity.sh` compares a vector's effect against llama.cpp's on a GGUF Llama.
