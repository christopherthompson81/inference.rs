---
title: Embed inference inside an Axum application
description: Mount the HTTP API inside an existing Axum router.
---

To add inference.rs to an existing Axum app, load an engine and mount its router under a sub-path:

- `inference_api::Engine` loads the models from an `EngineSpec` (the JSON the C ABI and the Python and C# bindings take too) and serves requests on them.
- `InferenceRsServerRouterBuilder` from `inference-server-core` produces an Axum `Router` over that engine.

## Dependencies

```toml
[dependencies]
anyhow = "1"
inference-api = { git = "https://github.com/christopherthompson81/inference.rs" }
inference-server-core = { git = "https://github.com/christopherthompson81/inference.rs" }
axum = "0.8"
serde_json = "1"
tokio = { version = "1", features = ["full"] }
```

## Mount under a sub-path

```rust
use axum::{Router, routing::get};
use inference_api::Engine;
use inference_server_core::inference_server_router_builder::InferenceRsServerRouterBuilder;
use serde_json::json;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let spec = serde_json::from_value(json!({
        "model": {"Plain": {"model_id": "Qwen/Qwen3-4B", "quant": "4"}},
    }))?;
    let engine = Engine::load(spec).await?;

    let inference_router = InferenceRsServerRouterBuilder::new()
        .with_engine(&engine)
        .build()
        .await?;

    let app = Router::new()
        .route("/", get(|| async { "My app" }))
        .nest("/ai", inference_router);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    axum::serve(listener, app).await?;
    Ok(())
}
```

`POST /ai/v1/chat/completions` then behaves identically to the standalone server, as do the other routes.

`"quant": "4"` picks a 4-bit build the repository publishes, or [quantizes in place](/reference/quantization-types/); omit it to run the model unquantized. The spec's full shape is the `EngineSpec` schema in `docs/openapi.json`: `runtime` (device, batching, paged attention), `agentic` (tools and permissions), several `models`, and so on.

## Builder options

`InferenceRsServerRouterBuilder` exposes:

- `with_include_swagger_routes(bool)`
- `with_base_path(&str)`
- `with_allowed_origins(Vec<String>)`
- `with_max_body_limit(usize)`
- `with_max_tool_rounds(usize)`
- `with_tool_dispatch_url(String)`
- `with_agent_permission(AgentPermission)` and `with_code_execution_permission(CodeExecutionPermission)`
- `with_api_keys(ApiKeys)`, to require [API keys](/reference/http-api/#authentication)
- `with_file_listing(bool)` and `with_observability_config(ObservabilityConfig)`

## Calling the model directly from a handler

For custom request shapes, share the `Engine` (it is cheap to clone) with your handlers and call it: `engine.chat(request, media)` and `engine.chat_stream(...)` apply the same agent policy (permissions, tool-round limits, approvals) that `/v1/chat/completions` applies, and `engine.responses(...)`, `engine.completion(...)`, `engine.embeddings(...)` and the file and session methods mirror their routes. `engine.for_owner(name)` scopes a clone to one owner, as a [keyed server](/reference/http-api/#authentication) does per API key.
