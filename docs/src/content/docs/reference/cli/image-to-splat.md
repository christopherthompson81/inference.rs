---
title: "inference image-to-splat"
description: "Gaussian splats of the object in one image with Idol: splat.ply (standard 3DGS), splat.spz, the image the model saw and a meta.json with the input's predicted camera; an image without a transparent background has it removed with the BiRefNet Idol ships"
sidebar:
  order: 17
---

<!-- Generated from clap definitions by inference-cli docgen. Do not edit. -->

Gaussian splats of the object in one image with Idol: splat.ply (standard 3DGS), splat.spz, the image the model saw and a meta.json with the input's predicted camera; an image without a transparent background has it removed with the BiRefNet Idol ships

```
inference image-to-splat [OPTIONS] --image <IMAGE> --out <OUT>
```

| Option | Default | Description |
|---|---|---|
| `--image <IMAGE>` | required | The image (png, jpg, webp) |
| `--out <OUT>` | required | The output folder, created if missing |
| `--mode <MODE>` | `synthetic` | `synthetic` (renders and generated images of one object: a tight crop) or `real` (photos: a looser crop) |
| `--seed <SEED>` | `42` |  |
| `--gaussians <GAUSSIANS>` | `262144` | Gaussians to make, 32,768 to 1,048,576, rounded down to a multiple of 32 |
| `--steps <STEPS>` | `50` | Sampling steps |
| `--guidance <GUIDANCE>` | `3` | Classifier-free guidance strength (0: none) |
| `--rescale-t <RESCALE_T>` | `3` | The time grid's shift toward high noise (1: uniform) |
| `--print-size <PRINT_SIZE>` | `100` | The printable STL's (print.stl) longest side, in millimetres |
| `--idol <IDOL>` |  | Idol's folder; fetched from speridlabs/idol when unset |
| `--dinov3 <DINOV3>` |  | DINOv3 ViT-H+/16's model.safetensors; fetched from facebook/dinov3-vith16plus-pretrain-lvd1689m (gated) when unset |
| `--vae <VAE>` |  | A FLUX.2 VAE's diffusion_pytorch_model.safetensors; fetched from black-forest-labs/FLUX.2-klein-4B when unset |
| `--cpu` | `false` | Run on the CPU (slow) |

