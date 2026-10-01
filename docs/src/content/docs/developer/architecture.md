---
title: Architecture
description: How inference is organized. Request flow, threading, and how pieces interact.
---

## The layers

From the outside in:

**Clients.** The HTTP server (`inference-server-core`: routes, API keys, SSE, metrics, the MCP server), the web UI, the `inference` CLI and the C ABI (`inference-ffi`, with the C# and Python bindings over it). Each is a client of the engine API and knows nothing about model internals.

**Engine API.** `inference-api`'s `Engine`: loading from an `EngineSpec`, the OpenAI, Anthropic and Responses operations, files, sessions, models, LoRA adapters and approvals. Every client calls the same methods, so a capability added here reaches the server, the CLI and every binding; a test in `inference-ffi` fails when an `Engine` method has no ABI entry and no listed reason for going without one.

**Engine.** `inference-core`: the request queue, scheduler and session store, one engine loop per loaded model, with `AgentRunner` as the seam to `inference-agent`'s tool loop. It drives pipelines without knowing about specific architectures.

**Pipelines and models.** Model implementations (the `inference-models-*` family crates), tokenization, quantization and attention kernels, one pipeline per model type behind a shared trait.

Requests enter through a client and flow down. New model architectures touch the model and pipeline layers; new operations go in the engine API, and a client only adds its framing.

## Engine threads

Startup spawns one engine thread per loaded model. Requests enter through a channel; the thread drives the loop that turns requests into token generation.

A single engine thread is the unit of concurrency. Concurrent requests to the same model share a queue and are batched by the scheduler. Requests to different models go to different threads and run independently. Multi-model routing and rehydration (reloading) of unloaded models are covered in [running multiple models](/guides/serve/multiple-models/).

## Scheduling

Each engine thread has a scheduler that decides which sequences to generate tokens for on each pass. Default scheduling is continuous batching: every active sequence with an available slot is included on every decoding step. With [paged attention](/guides/perf/paged-attention/), a slot is a KV cache block rather than a full sequence's cache, so many more sequences can coexist.

[Speculative decoding](/guides/perf/speculative-decoding/) alternates drafting and verification passes. MCP tool calls pause a sequence during execution and resume it when the result arrives.

On CUDA, supported paged-attention decode steps can be replayed through [CUDA graphs](/guides/perf/paged-attention/#cuda-graphs) by default. Graph capture lives in the pipeline layer; the scheduler still selects the active sequences normally.

## Tool loop and sessions

Server-side tool calling runs in the outer engine loop, inside one HTTP request:

1. Run inference until a tool call.
2. Execute the tool.
3. Append the result to history.
4. Resume.

Repeat until the model returns a non-tool-call response or the round cap is hit. Loop semantics, entry conditions, and configuration live in [agentic runtime for apps](/guides/agents/agentic-runtime/).

Agentic requests are stateful; state lives in an in-memory session store. Matching, splicing, and eviction are documented in [session memory](/developer/session-memory/), the user-facing workflow in [persist sessions](/guides/agents/persist-sessions/).

## See also

- [Session memory](/developer/session-memory/).
- [cuTile setup](/developer/moe-backends/).
- [The multimodal pipeline](/developer/multimodal-pipeline/).
