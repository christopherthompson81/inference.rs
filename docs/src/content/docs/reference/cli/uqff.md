---
title: "inference uqff"
description: "Inspect, report, or verify UQFF artifacts"
sidebar:
  order: 6
---

<!-- Generated from clap definitions by inference-cli docgen. Do not edit. -->

Inspect, report, or verify UQFF artifacts

```
inference uqff [OPTIONS] <COMMAND>
```

## inference uqff report

Print or write a UQFF report

```
inference uqff report [OPTIONS] --model-id <MODEL_ID>
```

| Option | Default | Description |
|---|---|---|
| `-m, --model-id <MODEL_ID>` | required | Hugging Face model ID or local path containing UQFF files |
| `--quant <QUANT>` |  | Quant group to inspect, such as 3, q3k, afq3, or all |
| `--revision <REVISION>` |  | Hugging Face revision to use |
| `--write` | `false` | Write uqff_report.json beside the artifacts |
| `--json` | `false` | Print JSON instead of human-readable text |
| `--base-model <BASE_MODEL>` |  | Base model ID to include in a written report |
| `--repo-id <REPO_ID>` |  | Hugging Face repo ID to include in a written report |

## inference uqff verify

Validate UQFF artifact structure

```
inference uqff verify [OPTIONS] --model-id <MODEL_ID>
```

| Option | Default | Description |
|---|---|---|
| `-m, --model-id <MODEL_ID>` | required | Hugging Face model ID or local path containing UQFF files |
| `--quant <QUANT>` |  | Quant group to inspect, such as 3, q3k, afq3, or all |
| `--revision <REVISION>` |  | Hugging Face revision to use |
| `--json` | `false` | Print JSON instead of human-readable text |
| `--strict` | `false` | Fail on missing report/producer metadata or fallback layers |
| `--allow-newer-minor` | `false` | Allow same-major UQFF files with a newer minor version |

## inference uqff inspect

Open a UQFF-aware tensor explorer

```
inference uqff inspect [OPTIONS] --model-id <MODEL_ID>
```

| Option | Default | Description |
|---|---|---|
| `-m, --model-id <MODEL_ID>` | required | Hugging Face model ID or local path containing UQFF files |
| `--quant <QUANT>` |  | Quant group to inspect, such as 3, q3k, afq3, or all |
| `--revision <REVISION>` |  | Hugging Face revision to use |

