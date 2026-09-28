---
title: Python SDK
description: Run the inference.rs engine in-process from Python.
---

The `inference_rs` Python package runs the engine in-process, as a pure-Python package over the engine's C ABI (`libinference_ffi`). Build the library with `cargo build --release -p inference-ffi` and install the package with `pip install -e bindings/python`. Start with [getting started](/guides/python/getting-started/), then [streaming](/guides/python/streaming/); the full API surface is in the [Python reference](/reference/python/), with [`Engine`](/reference/python/engine/) and the [engine spec](/reference/python/spec/) as the entry points.
