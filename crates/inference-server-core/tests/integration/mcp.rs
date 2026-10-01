//! The MCP chat tool on a tiny random-weight PaddleOCR-VL: it runs through the engine's chat operation and its policy.

use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use inference_server_core::mcp_server::{MCP_ROUTE, create_mcp_router};
use serde_json::{Value, json};
use tower::ServiceExt;

use crate::support;

// JSON-RPC's invalid-params code, which the MCP server uses for requests the engine rejects.
const INVALID_PARAMS: i64 = -32602;
const MAX_TOKENS: usize = 4;
const PROMPT: &str = "Reply with the single word: ok";

async fn mcp(agentic: Value, method: &str, params: Value) -> anyhow::Result<Value> {
    let dir = support::tiny_checkpoint()?;
    let spec = serde_json::from_value(json!({
        "model": {"MultimodalPlain": {"model_id": dir.path().to_string_lossy(), "dtype": "f32"}},
        "runtime": {"device": "cpu"},
        "agentic": agentic,
    }))?;
    let engine = inference_api::Engine::load(spec).await?;
    let body = json!({"jsonrpc": "2.0", "id": 1, "method": method, "params": params});
    let response = create_mcp_router(&engine, None)
        .oneshot(
            Request::post(MCP_ROUTE)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))?,
        )
        .await?;
    assert_eq!(response.status(), StatusCode::OK);
    Ok(serde_json::from_slice(
        &to_bytes(response.into_body(), usize::MAX).await?,
    )?)
}

async fn call_chat(agentic: Value) -> anyhow::Result<Value> {
    let arguments = json!({
        "model": "default",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
    });
    mcp(
        agentic,
        "tools/call",
        json!({"name": "chat", "arguments": arguments}),
    )
    .await
}

#[tokio::test(flavor = "multi_thread")]
async fn the_chat_tool_answers_with_text() -> anyhow::Result<()> {
    let body = call_chat(json!({})).await?;
    assert!(body["error"].is_null(), "{body}");
    assert_eq!(body["result"]["content"][0]["type"], "text", "{body}");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn an_ask_server_offers_no_chat_tool() -> anyhow::Result<()> {
    // `ask` needs a stream to carry approvals, so the blocking tool is neither listed nor callable.
    let ask = json!({"agent_permission": "ask"});
    let listed = mcp(ask.clone(), "tools/list", json!({})).await?;
    assert_eq!(listed["result"]["tools"], json!([]), "{listed}");
    let called = call_chat(ask).await?;
    assert_eq!(called["error"]["code"], INVALID_PARAMS, "{called}");
    assert!(called["result"].is_null(), "{called}");
    Ok(())
}
