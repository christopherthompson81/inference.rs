---
title: "CLI reference"
description: "Subcommands and flags of the inference binary."
sidebar:
  order: 1
---

<!-- Generated from clap definitions by inference-cli docgen. Do not edit. -->

## Subcommands

| Subcommand | Purpose |
|---|---|
| [`inference serve`](/reference/cli/serve/) | Start HTTP/MCP server and (optionally) the UI at /ui |
| [`inference run`](/reference/cli/run/) | Run model in interactive mode, or one-shot mode with `-i` |
| [`inference launch`](/reference/cli/launch/) | Run Claude Code, opencode or Codex against an inference server, configured through their environment (no agent config file is written): a running server (`--url`, or localhost:1234), or one started from `--config` and stopped after |
| [`inference agent`](/reference/cli/agent/) | Coding agent: a tool loop (read, write, edit, bash, grep, glob) over the project directory, with the model in this process or on a server (`--url`) |
| [`inference completions`](/reference/cli/completions/) | Generate shell completions |
| [`inference quantize`](/reference/cli/quantize/) | Generate a UQFF quantized model file, or an `asr` GGUF of a speech recogniser |
| [`inference uqff`](/reference/cli/uqff/) | Inspect, report, or verify UQFF artifacts |
| [`inference doctor`](/reference/cli/doctor/) | Run system diagnostics and environment checks |
| [`inference tune`](/reference/cli/tune/) | Recommend quantization + device mapping for a model. Rejects `--quant auto`; pass `--quant <level>` or `--isq <level>` to bias the recommendation toward a specific quantization target. Adapter options are rejected because adapter memory is not included in the estimate |
| [`inference login`](/reference/cli/login/) | Authenticate with Hugging Face Hub |
| [`inference cache`](/reference/cli/cache/) | Manage the Hugging Face model cache |
| [`inference kernels`](/reference/cli/kernels/) | Manage the CUDA kernel libraries: download them for offline use, prune old ones |
| [`inference bench`](/reference/cli/bench/) | Run performance benchmarks for base or LoRA model generation |
| [`inference from-config`](/reference/cli/from-config/) | Run from a full TOML configuration file |
| [`inference update`](/reference/cli/update/) | Update or migrate an install using the installer |
| [`inference uninstall`](/reference/cli/uninstall/) | Remove an installer-managed install |

## Global options

| Option | Default | Description |
|---|---|---|
| `--seed <SEED>` |  | Random seed for reproducibility |
| `-l, --log <LOG>` |  | Log all requests and responses to this file |
| `--token-source <TOKEN_SOURCE>` | `cache` | Token source for Hugging Face authentication. Formats: `literal:<token>`, `env:<var>`, `path:<file>`, `cache`, `none` |
| `-v, --verbose` | `0` | Increase logging verbosity. Use -v for debug and -vv for trace-level internals |

