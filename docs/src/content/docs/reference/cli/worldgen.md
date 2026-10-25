---
title: "inference worldgen"
description: "Generate a 3D world from an equirectangular panorama with WorldStereo-2: keyframes along camera trajectories, then WorldMirror 2.0 on them for cameras, depth, points and 3D Gaussian splats"
sidebar:
  order: 15
---

<!-- Generated from clap definitions by inference-cli docgen. Do not edit. -->

Generate a 3D world from an equirectangular panorama with WorldStereo-2: keyframes along camera trajectories, then WorldMirror 2.0 on them for cameras, depth, points and 3D Gaussian splats

```
inference worldgen [OPTIONS] --panorama <PANORAMA> --out <OUT> --worldstereo <WORLDSTEREO> --wan <WAN> --moge <MOGE>
```

| Option | Default | Description |
|---|---|---|
| `--panorama <PANORAMA>` | required | The panorama (2:1 equirectangular image) |
| `--out <OUT>` | required | The output folder, created if missing |
| `--worldstereo <WORLDSTEREO>` | required | The WorldStereo checkpoint folder (worldstereo-memory-dmd: config.json and model.safetensors) |
| `--wan <WAN>` | required | The Wan2.1-I2V-14B-480P diffusers folder (vae, image_encoder, text_encoder, tokenizer, transformer config) |
| `--moge <MOGE>` | required | The MoGe-2 ViT-L normal folder (config.json and model.safetensors) |
| `--worldmirror <WORLDMIRROR>` |  | The WorldMirror 2.0 checkpoint folder; without it the clips are written and the reconstruction skipped |
| `--prompt <PROMPT>` | `A photorealistic, detailed scene with natural lighting.` | The caption every clip is generated with |
| `--scene <SCENE>` |  | Treat the scene as outdoor (far distances compressed) or indoor; decided by detected sky when unset. Possible values: `outdoor`, `indoor`. |
| `--seed <SEED>` | `1024` | Seed of the first clip's noise (each later clip adds one) |
| `--reconstruct-size <RECONSTRUCT_SIZE>` | `518` | The longest edge WorldMirror reconstructs at (floored to a multiple of 14); memory grows with its square |
| `--attention <ATTENTION>` | `int8` | WorldStereo's self-attention: int8 (int8 Q.K^T, ~1.5x faster) or bf16 (exact flash attention). Possible values: `int8`, `bf16`. |
| `--weights <WEIGHTS>` | `q8_0` | WorldStereo's main-block linear weights: q8_0 (quantized at load, ~1.3x faster matmuls) or bf16. Possible values: `q8_0`, `bf16`. |

