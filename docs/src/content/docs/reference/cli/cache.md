---
title: "inference cache"
description: "Manage the Hugging Face model cache"
sidebar:
  order: 10
---

<!-- Generated from clap definitions by inference-cli docgen. Do not edit. -->

Manage the Hugging Face model cache

```
inference cache [OPTIONS] <COMMAND>
```

## inference cache list

List all cached models

```
inference cache list [OPTIONS]
```

## inference cache delete

Delete a specific model from cache

```
inference cache delete [OPTIONS] --model-id <MODEL_ID>
```

| Option | Default | Description |
|---|---|---|
| `-m, --model-id <MODEL_ID>` | required | Model ID (e.g., "Qwen/Qwen3-4B") |

