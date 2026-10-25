---
title: "inference reconstruct"
description: "Reconstruct a scene from photos with WorldMirror 2.0: cameras, depth, normals, a point cloud and standard 3D Gaussian splats, written to a folder with a meta.json of conventions and settings"
sidebar:
  order: 14
---

<!-- Generated from clap definitions by inference-cli docgen. Do not edit. -->

Reconstruct a scene from photos with WorldMirror 2.0: cameras, depth, normals, a point cloud and standard 3D Gaussian splats, written to a folder with a meta.json of conventions and settings

```
inference reconstruct [OPTIONS] --model <MODEL> --input <INPUT> --out <OUT>
```

| Option | Default | Description |
|---|---|---|
| `--model <MODEL>` | required | The WorldMirror checkpoint folder (config.json and model.safetensors) |
| `--input <INPUT>` | required | A folder of images (jpg, jpeg, png, webp, read in name order), one image, or a video (mp4, mov, avi, mkv, webm, gif; FFmpeg for all but GIF) |
| `--out <OUT>` | required | The output folder, created if missing |
| `--target-size <TARGET_SIZE>` | `952` | Longest edge to infer at: the first image's, capped at this and floored to a multiple of 14 |
| `--fps <FPS>` | `1` | Frames a second sampled from a video input |
| `--max-frames <MAX_FRAMES>` | `32` | The most frames sampled from a video input |
| `--no-edge-mask` | `false` | Keep pixels at depth and normal discontinuities |
| `--no-sky-mask` | `false` | Keep pixels the model marks as sky |
| `--prior-cameras <PRIOR_CAMERAS>` |  | Known cameras as JSON: `extrinsics` (4x4 camera-to-world, OpenCV) and `intrinsics` (3x3, original pixels), each a list of {camera_id, matrix} with camera_id an image stem or index; either list may be omitted |
| `--prior-depth <PRIOR_DEPTH>` |  | A folder of known z-depth maps named by image stem: .npy (float32), .png (16-bit millimetres) or .exr |
| `--cpu` | `false` | Run on the CPU (slow) instead of the first CUDA device |

