# Inkling 2 load-time recipe

Goal: `inference run -m thinkingmachines/Inkling-2-Mini` (the official checkpoint) builds recipe C once into a cache
and loads it, so a user gets the setup measured in `inkling2_speed_investigation.md` with no llama.cpp converter, no
ggml-org GGUF and no manual merge. Today recipe C (`Inkling-2-Mini-NVFP4-C.gguf`) is made by `quantize gguf
--checkpoint` over a source GGUF from llama.cpp's converter (official-weights investigation, Runs 7-9).

## Run 1 - 2026-11-24 02:30

What recipe C is, from the file (gguf-py over `Inkling-2-Mini-NVFP4-C.gguf`, tensors grouped by role): 744 tensors,
60 metadata keys (`inkling2_loop.*` from `config.json`, a gpt2 tokenizer with `pre = inkling2`, the chat template).

```
every 2-D projection (attn q k v o gate rel sink_proj, k/v conv_proj)     Q8_0
output.weight, token_embd.weight                                         Q6_K
ffn_gate_up_exps / ffn_down_exps, layers 0-19                            NVFP4 as stored (+ .scale F32)
ffn_gate_up_exps layers 20-23 Q6_K, ffn_down_exps layers 20-23           Q8_0 (from MXFP8 values)
everything else (norms, convs, clips, rel_bias, res queries, router, exp_probs_b, loop ctrl / msg query)  F32
```

So the rules are by role, with nothing per-layer outside the experts. Plan: (1) `quantize gguf` reads its source
tensors from a provider rather than only a GGUF; an Inkling provider maps the checkpoint to the GGUF names (composing
`inkling2_official::bind` with the inverse of `inkling2_bindings`) and writes the metadata and tokenizer from
`config.json` / `tokenizer.json`; acceptance is byte equality with recipe C. (2) A load of an Inkling checkpoint with
no quantization named builds recipe C once into a cache and loads it as a GGUF. (3) KLD and speed against today's C.

## Run 2 - 2026-11-24 03:50

Recipe C from the checkpoint alone: `requantize_gguf` now reads a `RequantizeSource` (the GGUF file one of two); the
checkpoint's (`Inkling2CheckpointSource`, core) lists the model's tensors from `inkling2_official` (other than the
experts) under the GGUF names the inverse of `inkling2_bindings` gives (`inkling2_gguf_tensor`), at F32, and the
expert stacks for the plan's checkpoint (`official_checkpoint_experts`) to read; the metadata and tokenizer come from
`config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json` and `chat_template.jinja`
(`inkling2_checkpoint_metadata`). The expert rules apply only to stacks the checkpoint stores other than as NVFP4.

```
inference quantize gguf --checkpoint /mnt/data/models/inkling2-mini-official -o recipe-test.gguf
Wrote (15.58 GiB from 51.56 GiB, 226 of 744 tensors quantized): 40 NVFP4 (10935.0 MiB), 6 Q6_K (2850.9 MiB),
180 Q8_0 (2031.6 MiB), 478 F32 (140.7 MiB)        1:42.6 wall, 21.6 GB peak RSS (the checkpoint's mapped pages)
```

Against `Inkling-2-Mini-NVFP4-C.gguf` (gguf-py, every tensor's bytes, every key's type and value): 744 of 744 tensors
byte-identical, every shared key identical (tokens, types, 424,413 merges, chat template). Missing only llama.cpp's
model-card keys (`general.name`, `basename`, `finetune`, `size_label`, `license`, `license.link`, `tags`,
`file_type`), which nothing here reads. So the GGUF a user would build is the file every speed run measured.

## Run 3 - 2026-11-24 05:00

The load path. `Engine` resolution (`resolve_model_source`, startup and `add_model`) passes a `Run` / `Plain` load of
a checkpoint to `inference_selection::recipes::with_load_recipe` when nothing it asks for (ISQ, UQFF, calibration,
Hessians, EXL3, MatFormer) would be undone; an Inkling 2 checkpoint (a directory, or a hub repo whose files it fetches)
becomes a GGUF load of its recipe, built first if the cache lacks it (keyed on the config, the weight files' names and
sizes and a recipe version; written to `.partial` and renamed; a free-space check up front). The model keeps the name
it was asked for. Opt-outs: `runtime.load_recipes: false` / `INFERENCE_RS_NO_RECIPE`; `runtime.recipe_dir` /
`INFERENCE_RS_RECIPE_DIR`.

```
manager load of /mnt/data/models/inkling2-mini-official, drafts 7 auto, q8_0 drafter, q8 KV, 131,072
first load    1:36.5 (recipe written in 82 s, then the GGUF load)      300 tok 157.8 tok/s   greedy = recipe C's text
second load   8.6 s (cache hit)                                       157.4, 162.6
from-config [[models]] model_id = the directory: cache hit, served as `/mnt/data/models/inkling2-mini-official`
INFERENCE_RS_NO_RECIPE=1: loads the checkpoint as stored
```

Tiny checkpoint (CPU): recipe C of the tiny official checkpoint scores up to 1.65 off the checkpoint, while the same
mapping with the dense tensors left F32 scores within 0.00099. The difference is the Q8_0 projections' rounding, which
the tiny random weights amplify through three loop passes (the real model's C sits at KLD 0.0039 against the
checkpoint). So the tests split: `an_unquantized_recipe_scores_as_its_checkpoint` (the `Unquantized` recipe, the
mapping, 5e-3) and `an_official_checkpoint_loads_as_its_recipe_once_built` (the automatic load, one file built, a
second load reading it unchanged). The tiny fixture's vocabulary is no gpt2 byte-level one, so the GGUF tokenizer gives
it under two tokens for the chat; that test scores token ids. Tiny Q6_K rows (32-64 wide) fall back to Q8_0: each
rule takes the recipe's type where the rows fit, else Q8_0, else F32.
