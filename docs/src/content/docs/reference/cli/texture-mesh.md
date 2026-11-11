---
title: "inference texture-mesh"
description: "A mesh painted from an image with TRELLIS.2's texturing pipeline: model.glb (the mesh fitted to the unit cube with baked base color and metallic-roughness textures, on its own texture coordinates or an unwrap), the cut-out image and a meta.json; an image without a transparent background has it removed with BiRefNet"
sidebar:
  order: 15
---

<!-- Generated from clap definitions by inference-cli docgen. Do not edit. -->

A mesh painted from an image with TRELLIS.2's texturing pipeline: model.glb (the mesh fitted to the unit cube with baked base color and metallic-roughness textures, on its own texture coordinates or an unwrap), the cut-out image and a meta.json; an image without a transparent background has it removed with BiRefNet

```
inference texture-mesh [OPTIONS] --mesh <MESH> --image <IMAGE> --out <OUT>
```

| Option | Default | Description |
|---|---|---|
| `--mesh <MESH>` | required | The mesh (glb, obj, ply) |
| `--image <IMAGE>` | required | The image to paint it from (png, jpg, webp) |
| `--out <OUT>` | required | The output folder, created if missing |
| `--resolution <RESOLUTION>` | `1024` | The grid the mesh is encoded at, 512 or 1024 |
| `--seed <SEED>` | `0` |  |
| `--texture-size <TEXTURE_SIZE>` | `2048` | The baked textures' width and height |
| `--unwrap` | `false` | Unwrap the mesh even when it has texture coordinates of its own |
| `--texture-steps <TEXTURE_STEPS>` |  | The texture flow's sampling steps (pipeline.json's when unset) |
| `--texture-guidance <TEXTURE_GUIDANCE>` |  | The texture flow's guidance strength (1: none) |
| `--texture-guidance-rescale <TEXTURE_GUIDANCE_RESCALE>` |  | The texture flow's guidance rescale, 0 to 1 |
| `--texture-rescale-t <TEXTURE_RESCALE_T>` |  | The texture flow's timestep shift (1: uniform) |
| `--trellis2 <TRELLIS2>` |  | The TRELLIS.2-4B folder; its texturing files are fetched from microsoft/TRELLIS.2-4B when unset |
| `--dinov3 <DINOV3>` |  | The DINOv3 ViT-L/16 folder; fetched from facebook/dinov3-vitl16-pretrain-lvd1689m (gated) when unset |
| `--birefnet <BIREFNET>` |  | The BiRefNet folder; fetched from ZhengPeng7/BiRefNet when unset and the background needs removing |
| `--remove-background` | `false` | Remove the background even when the image has transparency of its own |
| `--cpu` | `false` | Run on the CPU (very slow) instead of the first CUDA device |

