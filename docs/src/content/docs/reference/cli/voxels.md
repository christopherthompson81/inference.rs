---
title: "inference voxels"
description: "The 64^3 voxel structure of the object in one image with JoVox, written as a cube mesh (voxels.ply) beside the prepared image and a meta.json; an image without a transparent background has it removed with BiRefNet"
sidebar:
  order: 15
---

<!-- Generated from clap definitions by inference-cli docgen. Do not edit. -->

The 64^3 voxel structure of the object in one image with JoVox, written as a cube mesh (voxels.ply) beside the prepared image and a meta.json; an image without a transparent background has it removed with BiRefNet

```
inference voxels [OPTIONS] --image <IMAGE> --out <OUT>
```

| Option | Default | Description |
|---|---|---|
| `--image <IMAGE>` | required | The image (png, jpg, webp); an object on a transparent background is used as it is |
| `--out <OUT>` | required | The output folder, created if missing |
| `--jovox <JOVOX>` |  | The JoVox-64 folder; fetched from xiangjuntang/JoVox-64 when unset |
| `--dinov3 <DINOV3>` |  | The DINOv3 ViT-L/16 folder; fetched from facebook/dinov3-vitl16-pretrain-lvd1689m (gated) when unset |
| `--birefnet <BIREFNET>` |  | The BiRefNet folder; fetched from ZhengPeng7/BiRefNet when unset and the background needs removing |
| `--seed <SEED>` | `0` |  |
| `--steps <STEPS>` | `72` | Sampling steps, 36 to 108 |
| `--cfg <CFG>` | `1.5` | Classifier-free guidance, at least 1 |
| `--temperature <TEMPERATURE>` | `0` | 0 takes each slot's most likely value |
| `--fill <FILL>` | `0.74` | The share of the prepared image the object's longest side fills |
| `--remove-background` | `false` | Remove the background even when the image has transparency of its own |
| `--cpu` | `false` | Run on the CPU (slow) instead of the first CUDA device |

