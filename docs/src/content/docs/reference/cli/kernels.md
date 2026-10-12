---
title: "inference kernels"
description: "Manage the CUDA kernel libraries: download them for offline use, prune old ones"
sidebar:
  order: 11
---

<!-- Generated from clap definitions by inference-cli docgen. Do not edit. -->

Manage the CUDA kernel libraries: download them for offline use, prune old ones

```
inference kernels [OPTIONS] <COMMAND>
```

## inference kernels fetch

Download this GPU's kernel libraries that this install lacks: the base set, plus GGUF types' kernels

```
inference kernels fetch [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--types <TYPES>` |  | GGUF quantization types whose kernels to fetch too, comma separated (`q4_k,q6_k`) |
| `--all` | `false` | Fetch every GGUF quantization type's kernels |

## inference kernels prune

Remove other versions' downloaded kernels, dead partial downloads, and kernel build dirs unused for 30 days

```
inference kernels prune [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--dry-run` | `false` | List what would be removed without removing it |

