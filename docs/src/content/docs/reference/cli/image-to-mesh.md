---
title: "inference image-to-mesh"
description: "A mesh with artist-like topology of the object in one image with LATO.2: about as many vertices as asked, joined by predicted edges, as model.glb and mesh.obj (untextured, y up), beside the crop the models see and a meta.json; an image without a transparent background has it removed with BiRefNet"
sidebar:
  order: 19
---

<!-- Generated from clap definitions by inference-cli docgen. Do not edit. -->

A mesh with artist-like topology of the object in one image with LATO.2: about as many vertices as asked, joined by predicted edges, as model.glb and mesh.obj (untextured, y up), beside the crop the models see and a meta.json; an image without a transparent background has it removed with BiRefNet

```
inference image-to-mesh [OPTIONS] --image <IMAGE> --out <OUT>
```

| Option | Default | Description |
|---|---|---|
| `--image <IMAGE>` | required | The image (png, jpg, webp) |
| `--out <OUT>` | required | The output folder, created if missing |
| `--vertices <VERTICES>` | `2000` | The vertex count to aim for, 200 to 5000 (the result lands near it, not on it) |
| `--seed <SEED>` | `0` |  |
| `--vertex-steps <VERTEX_STEPS>` | `24` | The vertex flow's sampling steps |
| `--vertex-guidance <VERTEX_GUIDANCE>` | `3` | The vertex flow's guidance strength (0: none) |
| `--topology-steps <TOPOLOGY_STEPS>` | `50` | The topology flow's sampling steps |
| `--structure-steps <STRUCTURE_STEPS>` | `25` | The scaffold flow's sampling steps |
| `--max-ring <MAX_RING>` | `12` | Fill open rings of up to this many edges (below 4: leave the holes) |
| `--close-loops <CLOSE_LOOPS>` | `64` | Then close the boundary loops (holes and cracks) of up to this many edges, adding no vertices (below 3: none) |
| `--lato2 <LATO2>` |  | The LATO.2 folder (its .pt checkpoints); fetched from 0x4c48/LATO.2 when unset |
| `--trellis-large <TRELLIS_LARGE>` |  | The TRELLIS-image-large folder (its structure flow and decoder); fetched when unset |
| `--dinov2 <DINOV2>` |  | The DINOv2 ViT-L/14-reg folder; fetched from facebook/dinov2-with-registers-large when unset |
| `--birefnet <BIREFNET>` |  | The BiRefNet folder; fetched from ZhengPeng7/BiRefNet when unset and the background needs removing |
| `--remove-background` | `false` | Remove the background even when the image has transparency of its own |
| `--cpu` | `false` | Run on the CPU (very slow) instead of the first CUDA device |

