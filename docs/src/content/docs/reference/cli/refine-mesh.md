---
title: "inference refine-mesh"
description: "A coarse mesh refined under an image with Hi3D v1.0 (Sparc3D): model.glb (untextured, or painted by TRELLIS.2's texturing with --texture), mesh.ply (y up, centred, radius 1), the conditioning image and a meta.json. Without --mesh, TRELLIS.2 makes the coarse mesh from the image first. Hi3D is under the Hi3D Open License Agreement v1.0 (its Acceptable Use Policy applies; a commercial product built on it shows \"Powered by Hi3D 1.0\")"
sidebar:
  order: 16
---

<!-- Generated from clap definitions by inference-cli docgen. Do not edit. -->

A coarse mesh refined under an image with Hi3D v1.0 (Sparc3D): model.glb (untextured, or painted by TRELLIS.2's texturing with --texture), mesh.ply (y up, centred, radius 1), the conditioning image and a meta.json. Without --mesh, TRELLIS.2 makes the coarse mesh from the image first. Hi3D is under the Hi3D Open License Agreement v1.0 (its Acceptable Use Policy applies; a commercial product built on it shows "Powered by Hi3D 1.0")

```
inference refine-mesh [OPTIONS] --image <IMAGE> --out <OUT>
```

| Option | Default | Description |
|---|---|---|
| `--image <IMAGE>` | required | The image the refinement follows (png, jpg, webp) |
| `--mesh <MESH>` |  | The coarse mesh (glb, obj, ply); made by TRELLIS.2 from the image when unset |
| `--out <OUT>` | required | The output folder, created if missing |
| `--resolution <RESOLUTION>` | `1024` | The surface resolution: 512, 1024 or 1536 (one, two or three passes; 1536 takes ~18 GB) |
| `--seed <SEED>` | `0` |  |
| `--steps <STEPS>` | `25` | Sampling steps each pass takes |
| `--texture` | `false` | Paint the refined mesh with TRELLIS.2's texturing (baked textures) |
| `--hi3d <HI3D>` |  | The Hi3D v1.0 folder; fetched from hi3d-official/Hi3Dv1.0 when unset |
| `--dinov2 <DINOV2>` |  | The DINOv2-large transformers folder; fetched from facebook/dinov2-large when unset |
| `--clip <CLIP>` |  | The CLIP ViT-L/14 transformers folder; fetched from openai/clip-vit-large-patch14 when unset |
| `--trellis2 <TRELLIS2>` |  | The TRELLIS.2-4B folder (coarse mesh, texturing); fetched from microsoft/TRELLIS.2-4B when unset |
| `--trellis-large <TRELLIS_LARGE>` |  | The TRELLIS-image-large folder (its structure decoder, for the coarse mesh); fetched when unset |
| `--dinov3 <DINOV3>` |  | The DINOv3 ViT-L/16 folder (coarse mesh, texturing); fetched from facebook/dinov3-vitl16-pretrain-lvd1689m (gated) when unset |
| `--birefnet <BIREFNET>` |  | The BiRefNet folder; fetched from ZhengPeng7/BiRefNet when unset and the background needs removing |
| `--remove-background` | `false` | Remove the background even when the image has transparency of its own |
| `--up <UP>` |  | The mesh's up axis, y (glTF's) or z (TRELLIS.2's PLYs); unset, a PLY's recorded one (`comment up z`), else y |
| `--cpu` | `false` | Run on the CPU (very slow) instead of the first CUDA device |

