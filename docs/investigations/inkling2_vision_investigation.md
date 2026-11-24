# Inkling 2 vision investigation

Inkling 2 Mini ships a vision tower (`model.visual.*`, 0.6 GiB bf16) beside the text model; our loads were text only.
Goal: image input through our engine, its rows matching an independent reference.

## Run 1 - 2026-11-24 09:00

Question: what exactly is the tower, and which reference can we trust?

Sources read: llama.cpp `tools/mtmd/models/inkling2v.cpp` + `conversion/inkling2.py` (`Inkling2VisionModel`, projector
type `inkling2v`), vLLM `vllm/models/inkling2/{processing.py,nvidia/vision.py,nvidia/ops/vision.py}`, basalt's
`src/models/inkling2/vision*.{hpp,cpp}`. Transformers 5.19's `inkling` module is a different model (no `inkling2`).

The tower (all three agree):
- Image read at its own size, placed top left on a canvas of whole 40 px cells; padding is a raw pixel value of -1,
  normalized like the rest (about -1.0078, not -1.0).
- Pixels: `2 * (bf16((v / 255 - clip_mean) / clip_std) * clip_std + clip_mean) - 1` (vLLM), which is `v / 127.5 - 1`
  up to the bf16 rounding.
- 20 px native patches, features ordered (py, px, c); linear patch embedding 1200 -> 1024, no bias.
- Learned 32 x 32 position grid resized bilinear, align_corners false, no antialias, clamped (vLLM's kernel; identity
  at 32 x 32). CLS row first = class_embedding + cls_position_embedding.
- ln_pre, then 23 pre-LN layers: `h += ls_1 * attn(ln_1(h))`, `h += ls_2 * mlp(ln_2(h))`, full non-causal attention,
  16 heads of 64, q/k/v/out with bias, exact GELU MLP 4096. No post-LN.
- 2D RoPE theta 10000 on adjacent pairs: pairs 0-15 rotate by x + 1, pairs 16-31 by y + 1, inv_freq 10000^(-j/16);
  CLS at 0 (identity).
- Drop CLS, merge 2 x 2 in (y0x0, y0x1, y1x0, y1x1) order, cells in raster order; proj_in 4096 -> 4096, exact GELU,
  proj_out 4096 -> 2304, no bias.
- One `<|unused_200054|>` pad per cell after `<|content_image|>`; the template writes one pad, the processor expands
  it. The rows replace the pads' embeddings as they are: no embedding RMS norm (llama.cpp skips `tok_norm` for raw
  embeddings). The text model has no rotary positions, so no grid positions reach it.
- RGBA: composited over an 8 px checkerboard of 204 / 153, rounded (vLLM, basalt).
- Size limits: vLLM none; llama.cpp 4096 tokens (resize past that); basalt 103 x 103 cells.

Setup (scratchpad `vis/`): llama.cpp mmproj from the official dir
(`convert_hf_to_gguf.py --mmproj --outtype bf16`, kept at `/mnt/data/models/inkling2-mini/ref/mmproj-inkling2-bf16.gguf`),
`llama-mtmd-cli` rebuilt at 44c20b1c0. Sanity: IQ4_NL text + this mmproj on a synthetic 500 x 330 picture (red circle,
blue rectangle, green triangle, "HELLO 42") names all four correctly. A small mtmd C API program (`vis/dump/dump.cpp`)
writes llama.cpp's rows; `vis/ref.py` is an f32 torch reference written from the checkpoint and the notes above.

Result, torch f32 against llama.cpp (rows x 2304):

| picture | cells | rel L2 | min row cos |
|---|---|---|---|
| synthetic shapes 500 x 330 | 117 | 1.05e-2 | 0.99945 |
| table fixture 470 x 152 | 48 | 5.6e-3 | 0.99988 |
| noise 1280 x 1280 (no resize) | 1024 | 2.7e-3 | 0.999992 |

Resize variants against llama.cpp (shapes): align_corners true 1.7e-1, antialias 2.1e-1, bicubic 1.4e-1, so the plain
bilinear is the shared one. A bf16-activation run of the torch reference sits 4.2e-2 from f32 (min cos 0.990), so the
1e-2 gap is precision, not math.

Implication: the parity target for our encoder is the f32 torch reference (gate around llama.cpp's 1.5e-2), with
llama.cpp's rows as the second check.

## Run 2 - 2026-11-24 09:35

Question: does our tower (`crates/inference-models-other/src/inkling2_vision.rs`) compute the same thing?

Command: `cargo build -p inference-models-other --example inkling2_vision_parity --features cuda`, then
`inkling2_vision_parity --checkpoint /mnt/data/models/inkling2-mini-official --rgb <pic>.rgb --width W --height H
--out <pic>.ours.f32 [--dtype f32]` on the 3090, compared with `vis/cmp.py`.

| picture | f32 tower vs torch f32 | bf16 tower vs torch f32 | bf16 tower vs llama.cpp | encode (bf16) |
|---|---|---|---|---|
| shapes, 117 cells | 1.8e-6 | 2.8e-2 (min cos 0.9969) | 3.0e-2 | 136 ms (dev build, first call) |
| table, 48 cells | 9.8e-7 | 1.5e-2 (0.9989) | 1.5e-2 | 137 ms |
| noise, 1024 cells | 5.0e-7 | 3.9e-3 (0.99999) | 4.3e-3 | 235 ms |

The f32 tower is exact against the reference, so preprocessing (vLLM's CLIP round trip and -1 padding), the position
resize, the RoPE pairs and the merge order all match. The bf16 tower keeps an f32 residual stream and f32 norms; its
gap sits between llama.cpp (1e-2) and an all-bf16 run (4.2e-2), as vLLM's own bf16 encoder would.

Next: the multimodal row wrapping the text model, its inputs processor, and an end-to-end answer about a picture.

## Run 3 - 2026-11-24 09:50

Question: does a served Inkling 2 read pictures end to end, with the prefix cache and the drafter?

Command: `target/debug/inference serve -p 18234 --format gguf -m /mnt/data/models/inkling2-mini -f
Inkling-2-Mini-NVFP4-C.gguf --mmproj .../ref/mmproj-inkling2-bf16.gguf --max-seq-len 32768` (then again with
`--mtp-model /mnt/data/models/inkling2-mini-official --mtp-n-predict 7 --mtp-auto-depth`), greedy chat requests with a
PNG data URL (`vis/ask.py`).

- Loads as the new `Inkling2Vl` multimodal row (vision kept unmapped on cuda:0).
- Synthetic shapes picture: 151 prompt tokens (117 cells), the same reasoning and answer as llama.cpp's mtmd-cli word
  for word (red circle, blue rectangle, green triangle, "HELLO 42"); decode 110.9 tok/s plain.
- Table fixture: "Fruit | Colour | Count / Apple | Red | 12 / Lemon | Yellow | 7", exact. The same request again:
  64 of 75 prompt tokens from the prefix cache, the same 107 tokens.
- With the drafter at 7 + auto depth: the same greedy outputs (154 and 107 tokens), 156.7 and 159.2 tok/s
  (94/151 and 63/96 drafts accepted).

So the wrapper's delegation holds for the paged cache, the prefix cache and the external drafter's verify steps.
Open: tiny-model tests, the parity script, the official checkpoint's recipe load (no projector written yet).
