---
title: Stream tokens from Python
description: Async iteration, FastAPI integration, and mid-stream error handling for Python streaming responses.
---

This guide covers consuming a streaming response from async code, from web frameworks, and handling failures mid-stream. The basics (setting `stream=True` and iterating `engine.chat_stream(...)`) are in [getting started](/guides/python/getting-started/#streaming-tokens).

## Async streaming

The SDK does not expose a native async iterator. `Stream.next()` blocks until the next event and returns `None` at the end, and it releases the GIL, so run it in an executor:

```python
import asyncio

import inference_rs as ir
from inference_rs import types as t

engine = ir.Engine(t.EngineSpec(model=t.ModelSelectedPlain(model_id="Qwen/Qwen3-4B")))


async def stream_response(prompt: str):
    request = t.ChatCompletionRequest(
        model="default",
        messages=[t.Message(role="user", content=prompt)],
        stream=True,
    )
    loop = asyncio.get_running_loop()
    with engine.chat_stream(request) as stream:
        while (event := await loop.run_in_executor(None, stream.next)) is not None:
            if event.name == "chunk":
                delta = event.data.choices[0].delta.content
                if delta:
                    yield delta
```

Consume with `async for`:

```python
async def main():
    async for delta in stream_response("Write a haiku."):
        print(delta, end="", flush=True)
```

`stream.next(timeout)` takes a timeout in seconds and returns `None` when it expires; `stream.done` tells a timeout apart from the end of the stream. Closing the stream (or leaving the `with` block) abandons the request. To stop it early and still get its final event, call `stream.cancel()` (from any thread, including while another waits in `next`) and keep reading: chat ends with a chunk whose `finish_reason` is `canceled`, Anthropic with `message_stop` (its `stop_reason` is `end_turn`, since Anthropic has no cancelled reason), Responses with `response.cancelled`, each with usage.

## Streaming into a web framework

For FastAPI, the same pattern works as a response generator:

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

import inference_rs as ir
from inference_rs import types as t

app = FastAPI()
engine = ir.Engine(t.EngineSpec(model=t.ModelSelectedPlain(model_id="Qwen/Qwen3-4B")))


@app.get("/stream")
async def stream(prompt: str):
    def iter():
        request = t.ChatCompletionRequest(
            model="default",
            messages=[t.Message(role="user", content=prompt)],
            stream=True,
        )
        with engine.chat_stream(request) as s:
            for event in s:
                if event.name == "chunk":
                    delta = event.data.choices[0].delta.content
                    if delta:
                        yield delta

    return StreamingResponse(iter(), media_type="text/plain")
```

For production, run inference as an HTTP server and call it with the OpenAI Python client rather than loading the model in the web app process. The HTTP server's streaming is more robust under load; see the [OpenAI-compatible API guide](/guides/serve/openai-compatible-apis/).

## Catching errors during streaming

A request the engine rejects up front raises `ir.InferenceError` from `chat_stream`. Streaming can also fail mid-response: out of memory, generation failure, validation errors. Then the stream ends with an `error` event whose data is the OpenAI error envelope, as parsed JSON. Chunks already received are unaffected, so partial output survives:

```python
import sys

try:
    with engine.chat_stream(request) as stream:
        for event in stream:
            if event.name == "chunk":
                print(event.data.choices[0].delta.content or "", end="", flush=True)
            elif event.name == "error":
                print(f"\n\nStream ended: {event.data['error']['message']}", file=sys.stderr)
except ir.InferenceError as e:
    print(f"Request failed: {e.code}: {e}", file=sys.stderr)
```

The stream ends after the chunk whose choices all carry a `finish_reason`.

When server-side tools run during generation (web search, code execution, shell, MCP tools), the stream also carries their events alongside the chunks: `agentic_tool_call_progress` for tool progress, `agentic_tool_approval_required` for a call waiting on `engine.resolve_approval(...)`, and `file_produced` for files a tool wrote. Their data is parsed JSON. Filtering on `event.name == "chunk"`, as above, skips them. See the [agentic runtime](/guides/agents/agentic-runtime/) for what they carry.
