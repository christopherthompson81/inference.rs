---
title: Stream chat responses from Rust
description: Handle stream events, errors, tool progress, cancellation, and task spawning with stream_chat_request.
---

`stream_chat_request` returns a `ChatEventStream`, a `futures::Stream<Item = ChatStreamEvent>`. The minimal loop is in [getting started](/guides/rust/getting-started/#streaming); this guide covers the events and production patterns.

## Handling every event

```rust
use inference::{ChatCompletionChunkResponse, ChatStreamEvent};
use std::io::Write;

let mut stream = model.stream_chat_request(messages).await?;
let mut out = std::io::BufWriter::new(std::io::stdout());

while let Some(event) = stream.next().await {
    match event {
        ChatStreamEvent::Chunk(ChatCompletionChunkResponse { choices, .. }) => {
            if let Some(text) = choices.first().and_then(|choice| choice.delta.content.as_ref()) {
                out.write_all(text.as_bytes())?;
                out.flush()?;
            }
        }
        ChatStreamEvent::Error(error) => {
            eprintln!("stream error: {error}");
            break;
        }
        _ => {}
    }
}
```

The events a chat stream carries:

- `Chunk`: the common case. Carries incremental text in `choices[0].delta.content`; the last chunk has a `finish_reason` and the usage.
- `Error`: the request failed. Nothing follows it; `error.kind` says whether it was the request's fault or the engine's.
- `AgenticToolCallProgress`, `AgenticToolApprovalRequired`, `FileProduced`: emitted when engine-run tools work mid-stream (next section).
- `BlockDenoisingProgress`: a block-diffusion model's denoising steps, on streams that ask for them.

The stream ends after its last chunk. The example uses `_ => {}` for brevity; production code should match the agentic events explicitly. Full example: [streaming](/examples/rust/getting-started/streaming/), [error-handling](/examples/rust/advanced/error-handling/).

## Streaming with tool calls

When the engine's [tool loop](/guides/agents/build-an-agent/) runs a tool mid-stream (web search, code execution, shell, [MCP (Model Context Protocol)](/guides/agents/connect-mcp-server/) tools, host tools), the stream interleaves progress events with content chunks, in stream order. Each call's two phases share its `tool_call_id`:

```rust
use inference::{AgenticToolCallPhase, ChatStreamEvent};

ChatStreamEvent::AgenticToolCallProgress(progress) => match progress.phase {
    AgenticToolCallPhase::Calling(_) => println!("[round {}: calling {}]", progress.round, progress.tool_name),
    AgenticToolCallPhase::Complete(_) => println!("[round {}: completed {}]", progress.round, progress.tool_name),
},
```

The non-streaming `send_chat_request` returns the tool calls in the response's `agentic_tool_calls` instead.

## Spawning, backpressure, and cancellation

`Model` is a cheap handle (`Clone` shares the loaded engine), and a stream owns what it needs, so either can move into a spawned task:

```rust
let handle = tokio::spawn({
    let model = model.clone();
    async move {
        let mut stream = model.stream_chat_request(messages).await?;
        while let Some(event) = stream.next().await {
            // forward chunks to a channel, websocket, etc.
        }
        anyhow::Ok(())
    }
});
```

The response channel behind the stream is bounded, so a consumer that stops polling applies backpressure to the engine. To cancel early, drop the stream: the engine stops generating for that request.

## Collecting the full response

To stream for early feedback while also assembling the final text:

```rust
let mut full_response = String::new();

while let Some(event) = stream.next().await {
    if let ChatStreamEvent::Chunk(chunk) = event
        && let Some(text) = chunk.choices.first().and_then(|choice| choice.delta.content.as_ref())
    {
        full_response.push_str(text);
        out.write_all(text.as_bytes())?;
        out.flush()?;
    }
}
```

`full_response` holds the complete assistant output once the stream ends; use it to log or persist the final text.
