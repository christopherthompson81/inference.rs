---
title: Python API
description: "The inference_rs Python package."
sidebar:
  order: 6
---

The `inference_rs` Python package exposes the same engine that powers the `inference` CLI.

## Install

inference.rs does not publish wheels yet; build the package from a checkout (add `--features cuda` or `metal` through `MATURIN_PEP517_ARGS`). See [Python SDK getting started](/guides/python/getting-started/#installing) and [hardware support](/reference/hardware-support/).

```bash
pip install ./inference-pyo3                              # CPU
MATURIN_PEP517_ARGS="--features cuda" pip install ./inference-pyo3
```

## Pages

| Page | Covers |
| --- | --- |
| [Runner](/reference/python/runner/) | The main entry point. Load a model and send requests. |
| [Which](/reference/python/which/) | Variants that select which kind of model to load. |
| [Requests](/reference/python/requests/) | Request dataclasses passed to Runner methods. |
| [Responses](/reference/python/responses/) | Response and streaming types returned by the engine. |
| [Enums](/reference/python/enums/) | Architecture, dtype, and option enums. |
| [Search](/reference/python/search/) | Types for web-search tool configuration. |
| [AnyMoE](/reference/python/anymoe/) | AnyMoE expert and config types. |
| [Code and shell execution](/reference/python/code-execution/) | Configuration for the built-in Python and shell executors. |
| [Agent approvals](/reference/python/agent-approvals/) | Request and decision types for agent action approval callbacks. |
| [Files](/reference/python/files/) | Input files and first-class output files surfaced from agentic runs. |
| [MCP](/reference/python/mcp/) | MCP client configuration types. |
| [Auto-mapping](/reference/python/automap/) | Hints for automatic device mapping. |

See [Python getting started](/guides/python/getting-started/) for a walkthrough and the [Python guides](/guides/python/) for task-oriented recipes.

---

<small>Generated from [`inference-pyo3/inference_rs.pyi`](https://github.com/christopherthompson81/inference.rs/blob/master/inference-pyo3/inference_rs.pyi).</small>
