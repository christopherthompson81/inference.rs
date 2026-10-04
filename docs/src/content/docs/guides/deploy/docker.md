---
title: Run inference in Docker
description: Build the CUDA container image from source and run the unified CLI in it.
---

No images are published for this repository; build one from a checkout. `docker/Dockerfile.cuda-13.0-ubi9` compiles the
`inference` CLI from source on Red Hat UBI 9 with CUDA 13.0 and copies it into a runtime image:

```bash
docker build -t inference:cuda -f docker/Dockerfile.cuda-13.0-ubi9 \
  --build-arg CUDA_COMPUTE_CAP=89 \
  --build-arg WITH_FEATURES=cuda,flash-attn .
```

- `CUDA_COMPUTE_CAP` is the GPU's compute capability without the dot (`80` A100, `86` RTX 30, `89` RTX 40/L4, `90`
  H100, `100` B200, `120` RTX 50, `121` DGX Spark); the kernels are built for that one capability. See
  [hardware support](/reference/hardware-support/).
- `WITH_FEATURES` takes [cargo features](/reference/cargo-features/); the default is `cuda`.
- Building with `flash-attn` is slow the first time; later builds use the layer cache.
- For CPU-only use, run the `inference` binary natively ([quickstart](/quickstart/)); there is no CPU Dockerfile.

Run it with the NVIDIA Container Toolkit ([NVIDIA's install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)):

```bash
docker run --rm --gpus all -p 1234:1234 -v hf-cache:/data -e HF_TOKEN=<token> \
  inference:cuda serve -m Qwen/Qwen3-4B
```

To pin a specific GPU: `--gpus '"device=0"'`. Running the container with no arguments prints the CLI help.

Model ids float: `-m Qwen/Qwen3-4B` resolves to whatever revision is tagged `main` at download time. The CLI has no
revision flag; to pin a revision, use the Rust SDK's `with_hf_revision`.

## Image contract

- Entrypoint is the `inference` binary; pass a subcommand and its flags as the container command.
- `inference serve` listens on port 1234 by default (the image's `EXPOSE`d port). To change it, change the flag and the mapping together: `serve -p 8080` with `-p 8080:8080`. There is no `PORT` environment variable.
- `HF_HOME=/data` is set in the image: mount a volume at `/data` to persist downloaded weights (they land in `/data/hub`). HF authentication for gated models: `-e HF_TOKEN=<token>`.
- Chat templates ship at `/chat_templates` for models that need one: `--chat-template /chat_templates/<file>.json`.
- The image does not include Python or NVIDIA's `tileiras`, so cuTile stays off; install `tileiras` in a derived image as described in [cuTile setup](/developer/moe-backends/) to enable it.

## Production deployment notes

**Persist the cache.** Weights are large enough that re-downloading on every restart is wasteful. Mount a named volume or host path at `/data`.

**Health check.** `/health` returns 200 when the server is up. Add a Docker healthcheck:

```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s \
  CMD curl -fsS http://localhost:1234/health || exit 1
```

The generous `--start-period` matters: first-run model loading can take minutes.

**Resource limits.** Set `--memory` and `--gpus` on `docker run` to bound the container's resources.

**Video input.** Install FFmpeg inside the image when serving video-capable models. See [set up video input](/guides/models/video-setup/) for the Docker snippet and runtime check.

## Kubernetes

The pieces above translate directly:

- Use a Deployment with a readiness probe hitting `/health` (or a model-aware check; see the [production checklist](/guides/deploy/production-checklist/)).
- Mount a PersistentVolumeClaim at `/data` for the Hugging Face cache.
- Use the NVIDIA device plugin and a `nvidia.com/gpu` resource request for CUDA.
- Use an initContainer to pre-download weights for fast pod startup.

There is no official Helm chart. Contributions welcome.

## See also

- [Production checklist](/guides/deploy/production-checklist/): operational concerns regardless of container layer.
- [Serve flag reference](/reference/cli/serve/): all `inference serve` options.
