//! /v1/chat/completions end to end on a tiny random-weight PaddleOCR-VL, as JSON and as SSE.

use axum::{
    body::{to_bytes, Body},
    http::{Request, StatusCode},
};
use inference_server_core::inference_server_router_builder::InferenceRsServerRouterBuilder;
use serde_json::{json, Value};
use tower::ServiceExt;

#[path = "../../../inference/tests/support/paddleocr_vl_tiny.rs"]
mod support;

const MAX_TOKENS: usize = 6;
// Streamed and non-streamed decodes share the model and greedy sampling, so they must agree exactly.
const PROMPT: &str = "Reply with the single word: ok";

// Loaded as `inference serve` loads: an EngineSpec through the engine API, served with its policies.
async fn router(dir: &std::path::Path) -> anyhow::Result<axum::Router> {
    let spec = serde_json::from_value(json!({
        "model": {"MultimodalPlain": {"model_id": dir.to_string_lossy(), "dtype": "f32"}},
        "runtime": {"device": "cpu"},
    }))?;
    let engine = inference_api::Engine::load(spec).await?;
    InferenceRsServerRouterBuilder::new()
        .with_engine(&engine)
        .build()
        .await
}

fn chat(stream: bool) -> Request<Body> {
    let body = json!({
        "model": "default",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "top_k": 1,
        "stream": stream,
    });
    Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

async fn body_text(response: axum::response::Response) -> anyhow::Result<String> {
    Ok(String::from_utf8(
        to_bytes(response.into_body(), usize::MAX).await?.to_vec(),
    )?)
}

#[tokio::test(flavor = "multi_thread")]
async fn chat_completions_stream_and_json_agree() -> anyhow::Result<()> {
    let dir = support::tiny_checkpoint()?;
    let app = router(dir.path()).await?;

    let response = app.clone().oneshot(chat(false)).await?;
    assert_eq!(response.status(), StatusCode::OK);
    let full: Value = serde_json::from_str(&body_text(response).await?)?;
    let text = full["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    assert!(full["usage"]["completion_tokens"].as_u64().unwrap() > 0);

    let response = app.oneshot(chat(true)).await?;
    assert_eq!(response.status(), StatusCode::OK);
    let sse = body_text(response).await?;
    let data: Vec<&str> = sse
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .collect();
    assert_eq!(data.last(), Some(&"[DONE]"), "{sse}");
    let mut streamed = String::new();
    for chunk in &data[..data.len() - 1] {
        let chunk: Value = serde_json::from_str(chunk)?;
        assert!(chunk.get("error").is_none(), "{chunk}");
        if let Some(delta) = chunk["choices"][0]["delta"]["content"].as_str() {
            streamed.push_str(delta);
        }
    }
    assert_eq!(streamed, text);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn ask_permission_without_streaming_is_a_validation_error() -> anyhow::Result<()> {
    let dir = support::tiny_checkpoint()?;
    let app = router(dir.path()).await?;
    let body = json!({
        "model": "default",
        "messages": [{"role": "user", "content": PROMPT}],
        "agent_permission": "ask",
    });
    let response = app
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))?,
        )
        .await?;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value = serde_json::from_str(&body_text(response).await?)?;
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("requires stream=true")),
        "{body}"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn missing_models_and_responses_are_typed_not_found() -> anyhow::Result<()> {
    let dir = support::tiny_checkpoint()?;
    let app = router(dir.path()).await?;
    let body = json!({"model": "no-such-model", "messages": [{"role": "user", "content": PROMPT}]});
    let response = app
        .clone()
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))?,
        )
        .await?;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body: Value = serde_json::from_str(&body_text(response).await?)?;
    assert_eq!(body["error"]["code"], "model_not_found", "{body}");

    let response = app
        .clone()
        .oneshot(Request::get("/v1/responses/resp_missing").body(Body::empty())?)
        .await?;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body: Value = serde_json::from_str(&body_text(response).await?)?;
    assert_eq!(body["error"]["code"], "response_not_found", "{body}");

    let body = json!({"input": PROMPT, "previous_response_id": "resp_missing"});
    let response = app
        .oneshot(
            Request::post("/v1/responses")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))?,
        )
        .await?;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body: Value = serde_json::from_str(&body_text(response).await?)?;
    assert_eq!(
        body["error"]["code"], "previous_response_not_found",
        "{body}"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn responses_stream_names_its_events_and_ends_with_done() -> anyhow::Result<()> {
    let dir = support::tiny_checkpoint()?;
    let app = router(dir.path()).await?;
    let body = json!({
        "input": PROMPT,
        "max_output_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "top_k": 1,
        "stream": true,
    });
    let response = app
        .oneshot(
            Request::post("/v1/responses")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))?,
        )
        .await?;
    assert_eq!(response.status(), StatusCode::OK);
    let sse = body_text(response).await?;
    let names: Vec<&str> = sse
        .lines()
        .filter_map(|line| line.strip_prefix("event: "))
        .collect();
    assert_eq!(names.first(), Some(&"response.created"), "{sse}");
    assert_eq!(names.last(), Some(&"response.completed"), "{sse}");
    let data: Vec<&str> = sse
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .collect();
    assert_eq!(data.last(), Some(&"[DONE]"), "{sse}");
    for (name, data) in names.iter().zip(&data) {
        let event: Value = serde_json::from_str(data)?;
        assert_eq!(event["type"], *name, "{event}");
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn adapter_routes_use_the_openai_error_envelope() -> anyhow::Result<()> {
    let dir = support::tiny_checkpoint()?;
    let app = router(dir.path()).await?;
    let response = app
        .clone()
        .oneshot(Request::get("/v1/lora_adapters?model=no-such-model").body(Body::empty())?)
        .await?;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body: Value = serde_json::from_str(&body_text(response).await?)?;
    assert_eq!(body["error"]["code"], "model_not_found", "{body}");
    assert_eq!(body["error"]["type"], "invalid_request_error", "{body}");

    let response = app
        .oneshot(Request::get("/v1/lora_adapters?model=a&model=b").body(Body::empty())?)
        .await?;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value = serde_json::from_str(&body_text(response).await?)?;
    assert_eq!(body["error"]["code"], "invalid_query", "{body}");
    Ok(())
}

fn multipart_upload(fields: &[(&str, Option<&str>, &str)]) -> Request<Body> {
    const BOUNDARY: &str = "inference-test-boundary";
    let mut body = String::new();
    for (name, filename, value) in fields {
        body.push_str(&format!(
            "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\""
        ));
        if let Some(filename) = filename {
            body.push_str(&format!(
                "; filename=\"{filename}\"\r\nContent-Type: text/csv"
            ));
        }
        body.push_str(&format!("\r\n\r\n{value}\r\n"));
    }
    body.push_str(&format!("--{BOUNDARY}--\r\n"));
    Request::post("/v1/files")
        .header(
            "content-type",
            format!("multipart/form-data; boundary={BOUNDARY}"),
        )
        .body(Body::from(body))
        .unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn files_upload_and_serve_their_content() -> anyhow::Result<()> {
    let dir = support::tiny_checkpoint()?;
    let app = router(dir.path()).await?;
    let response = app
        .clone()
        .oneshot(multipart_upload(&[
            ("purpose", None, "user_data"),
            ("file", Some("table.csv"), "a,b\n1,2\n"),
        ]))
        .await?;
    assert_eq!(response.status(), StatusCode::OK);
    let uploaded: Value = serde_json::from_str(&body_text(response).await?)?;
    let id = uploaded["id"].as_str().unwrap();

    let response = app
        .clone()
        .oneshot(Request::get(format!("/v1/files/{id}/content")).body(Body::empty())?)
        .await?;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["content-type"], "text/csv");
    assert!(response.headers()["content-disposition"]
        .to_str()?
        .contains("filename=\"table.csv\""));
    assert_eq!(body_text(response).await?, "a,b\n1,2\n");

    let response = app
        .oneshot(multipart_upload(&[("file", Some("table.csv"), "a,b\n")]))
        .await?;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value = serde_json::from_str(&body_text(response).await?)?;
    assert_eq!(body["error"]["param"], "purpose", "{body}");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn an_engine_shuts_down_from_its_last_clone() -> anyhow::Result<()> {
    let dir = support::tiny_checkpoint()?;
    let spec = serde_json::from_value(json!({
        "model": {"MultimodalPlain": {"model_id": dir.path().to_string_lossy(), "dtype": "f32"}},
        "runtime": {"device": "cpu"},
    }))?;
    let engine = inference_api::Engine::load(spec).await?;
    let other = engine.clone();
    assert!(engine.shutdown().await.is_err());
    other.shutdown().await.map_err(anyhow::Error::msg)?;
    Ok(())
}
