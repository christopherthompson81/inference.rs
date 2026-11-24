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

The two files hold one text a line (`\n` for a newline inside one), and line N of the first pairs with line N of the second. This is the format llama.cpp's `cvector-generator` reads. In a spec or a request, the same is a `derived` entry: `{"positive": [...], "negative": [...], "scale": 6}`. The engine runs each pair through the model, records every layer's output, and takes the mean difference per layer with shared prefixes left out, normalized to unit length, as llama.cpp's `mean` method does. The result is a llama.cpp control vector GGUF, kept in `~/.cache/inference-rs/control-vectors` (or `INFERENCE_RS_CONTROL_VECTOR_DIR`) under a key of the model and the texts. Setting the same texts again reads that file, and a set or a load reports a derived vector as its cached file.

Deriving takes one prompt pass per text, well under a second for a handful of short pairs on a GPU. The derived directions are unit length, so the scale is absolute: on Inkling 2 about 4 nudges, 8 is strong and 16 overwhelms the model. A negative scale steers the other way.

## On a loaded model

```bash
curl localhost:1234/control_vectors/set -H 'Content-Type: application/json' \
  -d '{"model": "chat", "vectors": {"files": [{"path": "/models/happy.gguf", "scale": 1.2}], "layer_range": [10, 21]}}'
curl 'localhost:1234/control_vectors?model=chat'
```

`POST /control_vectors/set` is an admin route, since it reads files on the server's host. It changes the vectors from the model's next step. Empty `files` stops steering. `GET /control_vectors` reports what a model is steered by. The C ABI has `inference_control_vectors_get` and `inference_control_vectors_set`, and the Python and C# engines have `control_vectors` and `set_control_vectors`.

Changing the files or their scales on a model that is already steered rewrites its vectors in place. CUDA graphs captured for it keep running, with no capture again. Attaching vectors to a model that has none, or removing them, captures its graphs again on the next steps.

## Notes

- Cached prompt states depend on the vectors, so the prefix cache keys them on the active set, and steered requests are not written to the disk prefix cache.
- Speculative decoding drafts from the steered states and verifies against the steered model. Greedy output stays exact against the target. Changing the vectors resets the `--mtp-auto-depth` tuner, since what it measured described the model as it was steered before.
- Inkling 2's looped core adds a layer's direction on each of its passes, and a derived vector pools the passes' outputs. llama.cpp applies no control vectors to Inkling 2, and its `cvector-generator` cannot build one for it; derive one here instead.
- Text models on the shared decoder and Inkling 2 take control vectors. Other model families and multimodal models refuse them for now.
- `scripts/control_vector_parity.sh` compares a vector's effect against llama.cpp's on a GGUF Llama.
