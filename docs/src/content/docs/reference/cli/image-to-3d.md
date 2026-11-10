---
title: "inference image-to-3d"
description: "A textured 3D model of the object in one image with TRELLIS.2: model.glb (unwrapped mesh with baked PBR textures, or per-vertex color and material), mesh.ply, the cut-out image and a meta.json; an image without a transparent background has it removed with BiRefNet"
sidebar:
  order: 16
---

<!-- Generated from clap definitions by inference-cli docgen. Do not edit. -->

A textured 3D model of the object in one image with TRELLIS.2: model.glb (unwrapped mesh with baked PBR textures, or per-vertex color and material), mesh.ply, the cut-out image and a meta.json; an image without a transparent background has it removed with BiRefNet

```
inference image-to-3d [OPTIONS] --image <IMAGE> --out <OUT>
```

| Option | Default | Description |
|---|---|---|
| `--image <IMAGE>` | required | The image (png, jpg, webp) |
| `--out <OUT>` | required | The output folder, created if missing |
| `--variant <VARIANT>` | `1024-cascade` | `1024-cascade` (shape at 512, then 1024; texture at 1024) or `512` (faster, coarser) |
| `--seed <SEED>` | `0` |  |
| `--vertex-colors` | `false` | Per-vertex color and material instead of baked textures (no unwrapping or baking) |
| `--texture-size <TEXTURE_SIZE>` | `2048` | The baked textures' width and height |
| `--remesh` | `false` | Rebuild the surface as a closed shell (narrow-band dual contouring) before baking |
| `--trellis2 <TRELLIS2>` |  | The TRELLIS.2-4B folder; fetched from microsoft/TRELLIS.2-4B when unset |
| `--trellis-large <TRELLIS_LARGE>` |  | The TRELLIS-image-large folder (its structure decoder); fetched when unset |
| `--dinov3 <DINOV3>` |  | The DINOv3 ViT-L/16 folder; fetched from facebook/dinov3-vitl16-pretrain-lvd1689m (gated) when unset |
| `--birefnet <BIREFNET>` |  | The BiRefNet folder; fetched from ZhengPeng7/BiRefNet when unset and the background needs removing |
| `--remove-background` | `false` | Remove the background even when the image has transparency of its own |
| `--cpu` | `false` | Run on the CPU (very slow) instead of the first CUDA device |

