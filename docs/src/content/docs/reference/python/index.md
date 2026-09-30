---
title: Python API
description: "The inference_rs Python package."
sidebar:
  order: 6
---

The `inference_rs` Python package runs the same engine as the `inference` CLI, through its C ABI (`libinference_ffi`). Specs, requests and responses are dataclasses generated from the server's OpenAPI document, so they match what the engine accepts.

## Install

inference.rs does not publish wheels yet; build the library from a checkout (add `--features cuda` or `metal`) and install the package. See [Python SDK getting started](/guides/python/getting-started/#installing) and [hardware support](/reference/hardware-support/).

```bash
cargo build --release -p inference-ffi
pip install -e bindings/python
```

## Pages

| Page | Covers |
| --- | --- |
| [Engine](/reference/python/engine/) | Load a model and serve requests; streams, results, errors and host callbacks. |
| [Engine spec](/reference/python/spec/) | What to load and how to run it: EngineSpec, the ModelSelected variants and their options. |
| [Chat and completions](/reference/python/chat/) | Chat completion, completion and embedding requests and responses, tools and output formats. |
| [Responses](/reference/python/responses/) | OpenResponses requests, resources and stream events. |
| [Anthropic](/reference/python/anthropic/) | Anthropic Messages requests, responses and skill listings. |
| [Models, adapters, files and skills](/reference/python/management/) | Model status and cache counters, LoRA adapters, files, skills, approvals, sessions, calibration, tokenization and the media generation calls. |
| [Layout](/reference/python/layout/) | PP-DocLayoutV3 document layout detection. |

See [Python getting started](/guides/python/getting-started/) for a walkthrough and the [Python guides](/guides/python/) for task-oriented recipes.

---

<small>Generated from [`bindings/python/inference_rs`](https://github.com/christopherthompson81/inference.rs/blob/master/bindings/python/inference_rs).</small>
