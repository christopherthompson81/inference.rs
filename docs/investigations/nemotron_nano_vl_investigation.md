# Nemotron Nano 12B v2 VL: images on the dense Nemotron-H decoder

Part of #565. `NemotronH_Nano_VL_V2`: Nemotron Nano 2 12B (dense `nemotron_h`, which loads from GGUF here since #788)
with the C-RADIO tower Bolt and Nano Omni use, and InternVL's tiling instead of their patch budget. Target: images from
a published text GGUF beside the checkpoint's tower shard, on one RTX 3090.

## Run 1 - 2026-11-06 13:10

**Question:** how far is Nano VL from Nano Omni, and does the tower-shard route run it?

**Command:** reading `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16`'s `config.json`, `preprocessor_config.json`,
`image_processing.py`, `processing.py`, `modeling.py` and the index; Bolt's `modeling_radio.py` for RADIO's position
grid; then `inference serve --format gguf` on `Vastined/...-Q4_K_M.gguf` (7.5 GB) with shard 1 of 7 (3.8 GB, the whole
tower and `mlp1`) as `--mmproj` and the checkpoint directory as `--tok-model-id`.

**Finding:**
- Same tower and projector as Nano Omni (timm names, `mlp1` RMSNorm 1e-5 / ReLU^2), RADIO's training config (4
  teachers, `register_multiple` 16: 4 cls + 12 registers), no sound.
- Images: InternVL's `dynamic_preprocess`, not a patch budget. Up to `max_num_tiles` 12 squares of 512 px on the grid
  whose aspect is closest (a tie goes to the larger grid when the image covers more than half of it), a 512 px
  thumbnail after them when there is more than one tile; bicubic on the float tensor. Each tile is 32 x 32 patches,
  256 tokens after the 2x pixel shuffle; `<img>` + every tile's tokens + `</img>` per image.
- RADIO resizes its 128 x 128 position grid to the tile's 32 x 32 with `align_corners=False`, as our C-RADIO does.
  llama.cpp's converter pre-resizes it with `align_corners=True`; another of its small departures, beside ImageNet
  mean/std for the published mmproj and `nemotron_v2_vl`'s one fixed 512 image.
- The checkpoint's `config.json` holds `time_step_limit: [0.0, Infinity]` as Python's `json` writes it, which serde
  rejects (`expected value at line 80 column 7`). Configs are now read with bare non-finite literals rewritten to NeMo's
  `{"__float__": ...}`, which Nemotron-H's `time_step_limit` already decoded.
- Served (CUDA, defaults): the table fixture (470 x 152: 3 x 1 tiles + thumbnail, 1024 image tokens, 1054 prompt
  tokens) described and read correctly; a 640 x 360 page (2 x 1 + thumbnail, 794 prompt tokens) read line for line.
  Decode 100 tok/s.

**Implication:** Nano VL works on the shard route. Next for it: a writer family for `inference quantize mmproj`.

## Run 2 - 2026-11-06 14:20

**Question:** can Nano VL's tower go into one mmproj that both engines run, and can our loader take the published one?

**Changes:** the writer's RADIO spec takes two layouts: Bolt's `nemotron_omni` (split qkv, full position table, patch
budget) for Nano Omni, and llama.cpp's `nemotron_v2_vl` (fused `attn_qkv`, the table resized to the tile's 32 x 32 grid
with `align_corners=False`, `clip.vision.preproc_max_tiles`) for Nano VL. A grid-resize source does the table. Bolt's
vision binding takes a fused qkv as it is; the config synthesis gives a `nemotron_v2_vl` projector the tiled layout
(`force_image_size` from `clip.vision.image_size`, the tile count, a thumbnail). New family `NemotronV2Vl`
(`nemotron_v2_vl` + `nemotron_h`). The writer reads configs through the non-finite rewrite as well (it failed on Nano
VL's `Infinity` first).

**Command:** `inference quantize mmproj -m nemotron-nano-vl-12b-bf16` (shard 1 of 7 on disk); gguf-py diff against the
published `...-BF16-mmproj.gguf`; `llama-mtmd-cli --jinja` (its non-jinja path rejects this GGUF's template) and
`inference serve` with no `--tok-model-id` on each.

**Finding:**
- 390 tensors, as the published file; 389 byte-identical. `v.position_embd` differs as intended: max abs 0.109 against
  a mean magnitude of 0.0064, so `align_corners` is far from a rounding matter at the grid's edges. Keys: ours CLIP mean
  [0.4815, 0.4578, 0.4082] where the published file has ImageNet's [0.485, 0.456, 0.406]; ours names the type under
  `clip.vision.projector_type`, the published one under `clip.projector_type`; ours adds the tile count.
- llama.cpp on ours and on the published file: the page read line for line on both.
- Ours on ours and on the published file: the page (794 prompt tokens) and the table (1054) on both, tiled, the same
  answers; 94-100 tok/s decode.
- `a_tiled_tower_written_as_nemotron_v2_vl_binds_back`: the tiny tiled tower written and bound back, every tensor equal
  to the checkpoint's but the table, which equals the checkpoint's resized by the tower's own bilinear op.

**Implication:** Nano VL runs from a GGUF pair alone, ours or the published one, in both engines.
