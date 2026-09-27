//! /v1/chat/completions end to end on a tiny random-weight PaddleOCR-VL, as JSON and as SSE.

use axum::{
    body::{to_bytes, Body},
    http::{Request, StatusCode},
};
use inference_core::{AutoDeviceMapParams, ModelDType, ModelSelected};
use inference_server_core::{
    inference_for_server_builder::InferenceRsForServerBuilder,
    inference_server_router_builder::InferenceRsServerRouterBuilder,
};
use serde_json::{json, Value};
use tower::ServiceExt;

#[path = "../../inference/tests/support/paddleocr_vl_tiny.rs"]
mod support;

const MAX_TOKENS: usize = 6;
// Streamed and non-streamed decodes share the model and greedy sampling, so they must agree exactly.
const PROMPT: &str = "Reply with the single word: ok";

async fn router(dir: &std::path::Path) -> anyhow::Result<axum::Router> {
    let state = InferenceRsForServerBuilder::new()
        .with_model(ModelSelected::MultimodalPlain {
            model_id: dir.to_string_lossy().into_owned(),
            tokenizer_json: None,
            arch: None,
            dtype: ModelDType::F32,
            topology: None,
            write_uqff: None,
            from_uqff: None,
            max_edge: None,
            calibration_file: None,
            imatrix: None,
            max_seq_len: AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN,
            max_batch_size: AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE,
            max_num_images: AutoDeviceMapParams::DEFAULT_MAX_NUM_IMAGES,
            max_image_length: AutoDeviceMapParams::DEFAULT_MAX_IMAGE_LENGTH,
            hf_cache_path: None,
            matformer_config_path: None,
            matformer_slice_name: None,
            organization: None,
        })
        .with_cpu(true)
        .build()
        .await?;
    InferenceRsServerRouterBuilder::new()
        .with_inference(state)
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
