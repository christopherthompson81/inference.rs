//! /v1/chat/completions end to end on a tiny random-weight PaddleOCR-VL, as JSON and as SSE.

use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use inference_server_core::inference_server_router_builder::InferenceRsServerRouterBuilder;
use serde_json::{Value, json};
use tower::ServiceExt;

use crate::support;

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

pub(crate) async fn body_text(response: axum::response::Response) -> anyhow::Result<String> {
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
async fn cache_stats_list_each_loaded_models_counters() -> anyhow::Result<()> {
    let dir = support::tiny_checkpoint()?;
    let response = router(dir.path())
        .await?
        .oneshot(Request::get("/v1/models/cache_stats").body(Body::empty())?)
        .await?;
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = serde_json::from_str(&body_text(response).await?)?;
    assert_eq!(body["object"], "list", "{body}");
    let model = &body["data"][0];
    assert_eq!(model["prefix_cache_sequences"], 0, "{body}");
    // The tiny PaddleOCR-VL keeps an encoder cache; nothing has been encoded yet.
    assert_eq!(
        model["encoder_cache"],
        json!({"hits": 0, "misses": 0}),
        "{body}"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_capped_response_is_incomplete_and_can_be_continued() -> anyhow::Result<()> {
    let dir = support::tiny_checkpoint()?;
    let spec = serde_json::from_value(json!({
        "model": {"MultimodalPlain": {"model_id": dir.path().to_string_lossy(), "dtype": "f32"}},
        "runtime": {"device": "cpu"},
    }))?;
    let engine = inference_api::Engine::load(spec).await?;
    let request = json!({"model": "default", "input": PROMPT, "max_output_tokens": MAX_TOKENS});
    let response = engine
        .responses(serde_json::from_value(request)?)
        .await
        .map_err(anyhow::Error::msg)?;
    let body = serde_json::to_value(&response)?;
    assert_eq!(body["status"], "incomplete", "{body}");
    assert_eq!(
        body["incomplete_details"]["reason"], "max_output_tokens",
        "{body}"
    );
    assert_eq!(body["output"][0]["status"], "incomplete", "{body}");

    // Unlike a cancelled reply, one cut off by its cap is a conversation to continue.
    let follow_up = json!({"model": "default", "input": PROMPT, "previous_response_id": response.id,
        "max_output_tokens": MAX_TOKENS});
    engine
        .responses(serde_json::from_value(follow_up)?)
        .await
        .map_err(anyhow::Error::msg)?;
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
    // The random weights never stop on their own, so the token cap ends the run.
    assert_eq!(names.last(), Some(&"response.incomplete"), "{sse}");
    let terminal: Value = serde_json::from_str(
        sse.lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .rev()
            .nth(1)
            .expect("the terminal event has data"),
    )?;
    assert_eq!(
        terminal["response"]["incomplete_details"]["reason"], "max_output_tokens",
        "{terminal}"
    );
    assert_eq!(
        terminal["response"]["output"][0]["status"], "incomplete",
        "{terminal}"
    );
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

pub(crate) fn multipart_upload(fields: &[(&str, Option<&str>, &str)]) -> Request<Body> {
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
async fn only_an_opted_in_server_lists_files_and_a_container_lists_its_own() -> anyhow::Result<()> {
    let dir = support::tiny_checkpoint()?;
    let spec = serde_json::from_value(json!({
        "model": {"MultimodalPlain": {"model_id": dir.path().to_string_lossy(), "dtype": "f32"}},
        "runtime": {"device": "cpu"},
    }))?;
    let engine = inference_api::Engine::load(spec).await?;
    let app = |listing| {
        InferenceRsServerRouterBuilder::new()
            .with_engine(&engine)
            .with_file_listing(listing)
            .build()
    };
    let get = |path: &str| Request::get(path).body(Body::empty());
    let shared = app(false).await?;
    let response = shared
        .clone()
        .oneshot(multipart_upload(&[
            ("purpose", None, "user_data"),
            ("file", Some("table.csv"), "a,b\n1,2\n"),
        ]))
        .await?;
    let uploaded: Value = serde_json::from_str(&body_text(response).await?)?;
    let id = uploaded["id"].as_str().unwrap().to_string();

    let refused = shared.clone().oneshot(get("/v1/files")?).await?;
    assert_eq!(refused.status(), StatusCode::FORBIDDEN);
    assert!(body_text(refused).await?.contains("--allow-file-listing"));
    let listed = app(true).await?.oneshot(get("/v1/files")?).await?;
    assert_eq!(listed.status(), StatusCode::OK);
    assert!(body_text(listed).await?.contains(&id));

    let container_ids = |body: Value| -> Vec<String> {
        body["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|file| file["id"].as_str().unwrap().to_string())
            .collect()
    };
    let container = |name: &str| {
        let shared = shared.clone();
        let path = format!("/v1/containers/{name}/files");
        async move {
            let response = shared.oneshot(get(&path)?).await?;
            anyhow::Ok(serde_json::from_str::<Value>(&body_text(response).await?)?)
        }
    };
    assert!(container_ids(container("cntr_mine").await?).is_empty());
    // A Responses run tags the files it cites with its container id.
    assert!(engine.state().try_tag_file(&id, "cntr_mine", None)?);
    assert_eq!(
        container_ids(container("cntr_mine").await?),
        vec![id.clone()]
    );
    assert!(container_ids(container("cntr_other").await?).is_empty());
    for (name, status) in [
        ("cntr_mine", StatusCode::OK),
        ("cntr_other", StatusCode::NOT_FOUND),
    ] {
        for path in [
            format!("/v1/containers/{name}/files/{id}"),
            format!("/v1/containers/{name}/files/{id}/content"),
        ] {
            let response = shared.clone().oneshot(get(&path)?).await?;
            assert_eq!(response.status(), status, "{path}");
        }
    }
    Ok(())
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
    assert!(
        response.headers()["content-disposition"]
            .to_str()?
            .contains("filename=\"table.csv\"")
    );
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

#[tokio::test(flavor = "multi_thread")]
async fn generated_images_are_served_from_the_file_store() -> anyhow::Result<()> {
    let dir = support::tiny_checkpoint()?;
    let spec = serde_json::from_value(json!({
        "model": {"MultimodalPlain": {"model_id": dir.path().to_string_lossy(), "dtype": "f32"}},
        "runtime": {"device": "cpu"},
    }))?;
    let engine = inference_api::Engine::load(spec).await?;
    let png = inference_core::images::encode_png(&image::DynamicImage::new_rgb8(5, 3))?;
    let url = inference_api::files::store_generated_image(engine.state(), None, png.clone(), None)
        .map_err(anyhow::Error::msg)?;
    let app = InferenceRsServerRouterBuilder::new()
        .with_engine(&engine)
        .build()
        .await?;
    let response = app.oneshot(Request::get(&url).body(Body::empty())?).await?;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["content-type"], "image/png");
    assert_eq!(to_bytes(response.into_body(), usize::MAX).await?, png);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn responses_apply_the_servers_ask_permission() -> anyhow::Result<()> {
    use futures::StreamExt;

    let dir = support::tiny_checkpoint()?;
    let spec = serde_json::from_value(json!({
        "model": {"MultimodalPlain": {"model_id": dir.path().to_string_lossy(), "dtype": "f32"}},
        "runtime": {"device": "cpu"},
        "agentic": {"agent_permission": "ask"},
    }))?;
    let engine = inference_api::Engine::load(spec).await?;
    let request = json!({"model": "default", "input": PROMPT, "max_output_tokens": MAX_TOKENS});

    // Approvals travel as stream events, so an `ask` server refuses a blocking request, as chat does.
    let refused = engine
        .responses(serde_json::from_value(request.clone())?)
        .await
        .expect_err("an ask server refuses a blocking Responses request");
    assert_eq!(
        refused.param.as_deref(),
        Some("agent_permission"),
        "{refused:?}"
    );

    let mut stream = engine
        .responses_stream(serde_json::from_value(request.clone())?)
        .await
        .map_err(anyhow::Error::msg)?;
    let mut names = Vec::new();
    while let Some(item) = stream.next().await {
        names.push(item.name());
    }
    assert_eq!(names.last(), Some(&"response.incomplete"), "{names:?}");

    let app = InferenceRsServerRouterBuilder::new()
        .with_engine(&engine)
        .build()
        .await?;
    let response = app
        .oneshot(
            Request::post("/v1/responses")
                .header("content-type", "application/json")
                .body(Body::from(request.to_string()))?,
        )
        .await?;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value = serde_json::from_str(&body_text(response).await?)?;
    assert_eq!(body["error"]["param"], "agent_permission", "{body}");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn models_describe_what_a_loaded_model_serves() -> anyhow::Result<()> {
    let dir = support::tiny_checkpoint()?;
    let spec = serde_json::from_value(json!({
        "model": {"MultimodalPlain": {"model_id": dir.path().to_string_lossy(), "dtype": "f32"}},
        "runtime": {"device": "cpu"},
    }))?;
    let engine = inference_api::Engine::load(spec).await?;
    let models = serde_json::to_value(engine.models().map_err(anyhow::Error::msg)?)?;
    // The `default` alias describes the model it stands for, so a client can pick its mode without a lookup.
    for model in models["data"].as_array().unwrap() {
        assert_eq!(model["category"], "multimodal", "{model}");
        assert!(
            model["modalities"]["input"]
                .as_array()
                .unwrap()
                .contains(&json!("vision")),
            "{model}"
        );
        assert!(model["max_model_len"].as_u64().is_some(), "{model}");
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn completions_take_token_id_prompts() -> anyhow::Result<()> {
    let dir = support::tiny_checkpoint()?;
    let spec = serde_json::from_value(json!({
        "model": {"MultimodalPlain": {"model_id": dir.path().to_string_lossy(), "dtype": "f32"}},
        "runtime": {"device": "cpu"},
    }))?;
    let engine = inference_api::Engine::load(spec).await?;
    let request = json!({"model": "default", "prompt": [1, 2, 3, 4], "max_tokens": MAX_TOKENS});
    let response = engine
        .completion(serde_json::from_value(request.clone())?)
        .await
        .map_err(anyhow::Error::msg)?;
    assert_eq!(response.usage.prompt_tokens, 4);

    let mut echoed = request;
    echoed["echo"] = json!(true);
    let refused = engine
        .completion(serde_json::from_value(echoed)?)
        .await
        .expect_err("echo needs a text prompt");
    assert!(refused.message.contains("text prompt"), "{refused:?}");
    Ok(())
}

// Enough to outlast the few steps a cancel takes to land, on a model whose random weights never stop on their own.
const LONG_COMPLETION: usize = 512;

#[tokio::test(flavor = "multi_thread")]
async fn a_cancelled_chat_stream_ends_with_its_usage() -> anyhow::Result<()> {
    use futures::StreamExt;
    use inference_api::engine_chat::ChatStreamEvent;

    let dir = support::tiny_checkpoint()?;
    let spec = serde_json::from_value(json!({
        "model": {"MultimodalPlain": {"model_id": dir.path().to_string_lossy(), "dtype": "f32"}},
        "runtime": {"device": "cpu"},
    }))?;
    let engine = inference_api::Engine::load(spec).await?;
    let request = json!({
        "model": "default",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": LONG_COMPLETION,
        "ignore_eos": true,
    });
    let mut stream = engine
        .chat_stream(serde_json::from_value(request)?, Default::default())
        .await
        .map_err(anyhow::Error::msg)?;
    let mut last = None;
    while let Some(event) = stream.next().await {
        match event {
            ChatStreamEvent::Chunk(chunk) => {
                stream.cancel();
                last = Some(chunk);
            }
            ChatStreamEvent::Error(error) => anyhow::bail!("{error:?}"),
            _ => {}
        }
    }
    let last = last.expect("the stream sent chunks");
    assert_eq!(
        last.choices[0].finish_reason.as_deref(),
        Some("canceled"),
        "{last:?}"
    );
    let usage = last.usage.expect("the final chunk carries usage");
    assert!(usage.completion_tokens < LONG_COMPLETION, "{usage:?}");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn anthropic_count_tokens_counts_the_rendered_prompt() -> anyhow::Result<()> {
    let dir = support::tiny_checkpoint()?;
    let app = router(dir.path()).await?;
    let count = |body: Value| {
        Request::post("/v1/messages/count_tokens")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
    };
    let response = app
        .clone()
        .oneshot(count(json!({
            "model": "default",
            "messages": [{"role": "user", "content": PROMPT}],
        }))?)
        .await?;
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = serde_json::from_str(&body_text(response).await?)?;
    assert!(body["input_tokens"].as_u64().unwrap() > 0, "{body}");

    let response = app
        .oneshot(count(json!({"model": "default", "messages": []}))?)
        .await?;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value = serde_json::from_str(&body_text(response).await?)?;
    assert_eq!(body["type"], "error", "{body}");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn model_management_routes_are_served_only_when_enabled() -> anyhow::Result<()> {
    let (_dir, engine) = crate::cancel::tiny_engine().await?;
    let model_id = engine
        .default_model_id()
        .expect("a loaded engine has a default");
    let set_default = || {
        Request::post("/v1/models/default")
            .header("content-type", "application/json")
            .body(Body::from(json!({ "model_id": model_id }).to_string()))
    };
    for (enabled, status) in [(false, StatusCode::NOT_FOUND), (true, StatusCode::OK)] {
        let app = InferenceRsServerRouterBuilder::new()
            .with_engine(&engine)
            .with_model_management(enabled)
            .build()
            .await?;
        let response = app.oneshot(set_default()?).await?;
        assert_eq!(response.status(), status, "model management {enabled}");
    }
    Ok(())
}
