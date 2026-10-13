//! The model manager's routes: an empty server loads models from its folders, serves them under the ids it was given,
//! and unloads them, all an admin's to do.

use std::net::SocketAddr;

use axum::{
    Router,
    body::Body,
    extract::connect_info::MockConnectInfo,
    http::{Request, StatusCode},
};
use inference_api::{
    Engine,
    model_manager::{ModelManager, ModelManagerConfig},
};
use inference_server_core::inference_server_router_builder::InferenceRsServerRouterBuilder;
use serde_json::{Value, json};
use tower::ServiceExt;

use crate::chat_route::body_text;

const LOOPBACK: [u8; 4] = [127, 0, 0, 1];
const LAN: [u8; 4] = [192, 168, 1, 20];
const PORT: u16 = 40000;

async fn router(folders: Vec<std::path::PathBuf>, peer: [u8; 4]) -> anyhow::Result<Router> {
    let engine = Engine::load(serde_json::from_value(
        json!({"runtime": {"device": "cpu"}}),
    )?)
    .await?;
    let manager = ModelManager::new(
        engine.clone(),
        ModelManagerConfig {
            folders,
            ..Default::default()
        },
    );
    Ok(InferenceRsServerRouterBuilder::new()
        .with_engine(&engine)
        .with_model_manager(manager)
        .build()
        .await?
        .layer(MockConnectInfo(SocketAddr::from((peer, PORT)))))
}

fn post(path: &str, body: Value) -> Request<Body> {
    Request::post(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

// the SSE stream's events, in order, as `(event, data)`
async fn events(app: &Router, request: Request<Body>) -> anyhow::Result<Vec<(String, Value)>> {
    let response = app.clone().oneshot(request).await?;
    assert_eq!(response.status(), StatusCode::OK);
    let text = body_text(response).await?;
    let mut events = Vec::new();
    for block in text.split("\n\n").filter(|block| !block.trim().is_empty()) {
        let field = |name: &str| {
            block
                .lines()
                .find_map(|line| line.strip_prefix(name))
                .map(str::trim)
                .unwrap_or_default()
                .to_string()
        };
        let data = field("data:");
        events.push((
            field("event:"),
            serde_json::from_str(&data).unwrap_or(Value::Null),
        ));
    }
    Ok(events)
}

async fn json_body(app: &Router, request: Request<Body>) -> anyhow::Result<(StatusCode, Value)> {
    let response = app.clone().oneshot(request).await?;
    let status = response.status();
    let text = body_text(response).await?;
    Ok((status, serde_json::from_str(&text).unwrap_or(Value::Null)))
}

#[tokio::test(flavor = "multi_thread")]
async fn an_empty_server_loads_serves_and_unloads_models_from_its_folders() -> anyhow::Result<()> {
    let chat = crate::support::tiny_checkpoint()?;
    let (asr_dir, nemo) =
        crate::parakeet_support::tiny_parakeet_nemo(crate::parakeet_support::HEADS[0])?;
    let app = router(
        vec![chat.path().to_path_buf(), asr_dir.path().to_path_buf()],
        LOOPBACK,
    )
    .await?;

    let (status, scan) = json_body(
        &app,
        Request::get("/v1/manager/inventory").body(Body::empty())?,
    )
    .await?;
    assert_eq!(status, StatusCode::OK, "{scan}");
    let listed: Vec<&str> = scan["models"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|model| model["path"].as_str())
        .collect();
    assert!(listed.contains(&chat.path().to_str().unwrap()), "{scan}");
    assert!(listed.contains(&nemo.to_str().unwrap()), "{scan}");

    let chat_path = chat.path().to_string_lossy();
    let loaded = events(
        &app,
        post(
            "/v1/manager/load",
            json!({"source": chat_path, "model_id": "chat"}),
        ),
    )
    .await?;
    let (last, model) = loaded.last().unwrap();
    assert_eq!(last, "loaded", "{loaded:?}");
    assert_eq!(
        (model["model_id"].clone(), model["kind"].clone()),
        (json!("chat"), json!("multimodal"))
    );
    assert!(
        loaded.iter().any(|(name, _)| name == "progress"),
        "{loaded:?}"
    );
    let asr = events(
        &app,
        post(
            "/v1/manager/load",
            json!({"source": nemo, "model_id": "asr"}),
        ),
    )
    .await?;
    assert_eq!(
        asr.last().unwrap().1["kind"],
        json!("transcription"),
        "{asr:?}"
    );

    let request =
        json!({"model": "chat", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 2});
    let (status, reply) = json_body(&app, post("/v1/chat/completions", request)).await?;
    assert_eq!(status, StatusCode::OK, "{reply}");
    let (_, models) = json_body(
        &app,
        Request::get("/v1/manager/models").body(Body::empty())?,
    )
    .await?;
    let ids: Vec<&str> = models["data"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|m| m["model_id"].as_str())
        .collect();
    assert_eq!(ids, ["asr", "chat"], "{models}");
    assert_eq!(models["data"][1]["source"], json!(chat_path), "{models}");

    // a second load under an id replaces the first
    let again = events(
        &app,
        post(
            "/v1/manager/load",
            json!({"source": chat_path, "model_id": "chat"}),
        ),
    )
    .await?;
    assert_eq!(again.last().unwrap().0, "loaded", "{again:?}");
    let (status, _) = json_body(
        &app,
        post("/v1/manager/unload", json!({"model_id": "chat"})),
    )
    .await?;
    assert_eq!(status, StatusCode::NO_CONTENT);
    let (status, _) = json_body(
        &app,
        post("/v1/manager/unload", json!({"model_id": "chat"})),
    )
    .await?;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (_, models) = json_body(
        &app,
        Request::get("/v1/manager/models").body(Body::empty())?,
    )
    .await?;
    assert_eq!(models["data"].as_array().unwrap().len(), 1, "{models}");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn loads_reach_only_the_folders_and_only_for_admins() -> anyhow::Result<()> {
    let chat = crate::support::tiny_checkpoint()?;
    let elsewhere = crate::support::tiny_checkpoint()?;
    let app = router(vec![chat.path().to_path_buf()], LOOPBACK).await?;
    for source in [
        elsewhere.path().to_string_lossy().into_owned(),
        format!(
            "{}/../{}",
            chat.path().display(),
            elsewhere.path().file_name().unwrap().to_string_lossy()
        ),
        "relative/path/model".to_string(),
    ] {
        let refused = events(
            &app,
            post(
                "/v1/manager/load",
                json!({"source": source, "model_id": "x"}),
            ),
        )
        .await?;
        assert_eq!(refused.last().unwrap().0, "error", "{source}: {refused:?}");
    }

    let lan = router(vec![chat.path().to_path_buf()], LAN).await?;
    let source = chat.path().to_string_lossy();
    let (status, body) = json_body(
        &lan,
        post(
            "/v1/manager/load",
            json!({"source": source, "model_id": "x"}),
        ),
    )
    .await?;
    assert_eq!(
        (status, body["error"]["code"].clone()),
        (StatusCode::FORBIDDEN, json!("admin_required"))
    );
    let (status, _) = json_body(
        &lan,
        Request::get("/v1/manager/inventory").body(Body::empty())?,
    )
    .await?;
    assert_eq!(status, StatusCode::FORBIDDEN);
    // what is served is anyone's to see
    let (status, _) = json_body(
        &lan,
        Request::get("/v1/manager/models").body(Body::empty())?,
    )
    .await?;
    assert_eq!(status, StatusCode::OK);
    Ok(())
}
