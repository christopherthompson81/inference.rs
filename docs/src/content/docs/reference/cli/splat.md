---
title: "inference splat"
description: "Train 3D Gaussian splats on a HY-World 2.0 gs_data folder (images, depths, normals, cameras.json, points.ply, sky_points.ply): the 3DGS stage of world generation"
sidebar:
  order: 16
---

<!-- Generated from clap definitions by inference-cli docgen. Do not edit. -->

Train 3D Gaussian splats on a HY-World 2.0 gs_data folder (images, depths, normals, cameras.json, points.ply, sky_points.ply): the 3DGS stage of world generation

```
inference splat [OPTIONS] --data <DATA> --out <OUT> --lpips <LPIPS>
```

| Option | Default | Description |
|---|---|---|
| `--data <DATA>` | required | The gs_data folder |
| `--out <OUT>` | required | The output folder for splats.ply and splats.json, created if missing |
| `--steps <STEPS>` | `8000` | Optimization steps; the densification schedule scales with it |
| `--lpips <LPIPS>` | required | LPIPS-VGG weights as safetensors: torchvision vgg16's features.{i}.weight/bias and the LPIPS v0.1 heads lin{0..4}.model.1.weight |

