//! A keyed server: each key's owner reaches only what it stored, and a request without a known key is refused.

use axum::{
    Router,
    body::Body,
    http::{Request, StatusCode, header::AUTHORIZATION},
};
use inference_server_core::{
    auth::ApiKeys, inference_server_router_builder::InferenceRsServerRouterBuilder,
};
use serde_json::{Value, json};
use tower::ServiceExt;

use crate::{
    cancel::tiny_engine,
    chat_route::{body_text, multipart_upload},
};

const KEYS: &str = "team-a = key-alpha\nteam-b = key-beta\n";
const ALPHA: &str = "key-alpha";
const BETA: &str = "key-beta";
const MAX_TOKENS: usize = 4;

async fn keyed_router(engine: &inference_api::Engine) -> anyhow::Result<Router> {
    InferenceRsServerRouterBuilder::new()
        .with_engine(engine)
        .with_api_keys(ApiKeys::parse(KEYS)?)
        .build()
        .await
}

fn with_key(mut request: Request<Body>, key: &str) -> Request<Body> {
    let value = format!("Bearer {key}").parse().expect("header value");
    request.headers_mut().insert(AUTHORIZATION, value);
    request
}

fn get(path: &str, key: &str) -> Request<Body> {
    with_key(Request::get(path).body(Body::empty()).unwrap(), key)
}

fn send_json(method: &str, path: &str, key: &str, body: Value) -> Request<Body> {
    let request = Request::builder()
        .method(method)
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    with_key(request, key)
}

async fn status_and_body(
    app: &Router,
    request: Request<Body>,
) -> anyhow::Result<(StatusCode, String)> {
    let response = app.clone().oneshot(request).await?;
    let status = response.status();
    Ok((status, body_text(response).await?))
}

#[tokio::test(flavor = "multi_thread")]
async fn a_request_without_a_known_key_is_refused_but_health_is_open() -> anyhow::Result<()> {
    let (_dir, engine) = tiny_engine().await?;
    let app = keyed_router(&engine).await?;
    let unkeyed = Request::get("/v1/models").body(Body::empty())?;
    let (status, body) = status_and_body(&app, unkeyed).await?;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert!(body.contains("invalid_api_key"), "{body}");
    let (status, _) = status_and_body(&app, get("/v1/models", "key-gamma")).await?;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    let with_header = Request::get("/v1/models")
        .header("x-api-key", ALPHA)
        .body(Body::empty())?;
    assert_eq!(status_and_body(&app, with_header).await?.0, StatusCode::OK);
    let lowercase = Request::get("/v1/models")
        .header(AUTHORIZATION, format!("bearer {ALPHA}"))
        .body(Body::empty())?;
    assert_eq!(status_and_body(&app, lowercase).await?.0, StatusCode::OK);
    for probe in ["/health", "/"] {
        let request = Request::get(probe).body(Body::empty())?;
        assert_eq!(
            status_and_body(&app, request).await?.0,
            StatusCode::OK,
            "{probe}"
        );
    }
    // an id that happens to be `health` is no probe
    for path in [
        "/v1/sessions/health",
        "/v1/files/health",
        "/v1/responses/health",
    ] {
        let request = Request::get(path).body(Body::empty())?;
        assert_eq!(
            status_and_body(&app, request).await?.0,
            StatusCode::UNAUTHORIZED,
            "{path}"
        );
    }
    let anthropic = Request::post("/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from("{}"))?;
    let (status, body) = status_and_body(&app, anthropic).await?;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    let body: Value = serde_json::from_str(&body)?;
    assert_eq!(body["type"], "error", "{body}");
    assert_eq!(body["error"]["type"], "authentication_error", "{body}");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_file_is_listed_read_and_deleted_only_by_its_owner() -> anyhow::Result<()> {
    let (_dir, engine) = tiny_engine().await?;
    let app = keyed_router(&engine).await?;
    let upload = multipart_upload(&[
        ("purpose", None, "user_data"),
        ("file", Some("table.csv"), "a,b\n1,2\n"),
    ]);
    let (status, body) = status_and_body(&app, with_key(upload, ALPHA)).await?;
    assert_eq!(status, StatusCode::OK, "{body}");
    let id = serde_json::from_str::<Value>(&body)?["id"]
        .as_str()
        .unwrap()
        .to_string();

    let (status, listed) = status_and_body(&app, get("/v1/files", ALPHA)).await?;
    assert_eq!(
        status,
        StatusCode::OK,
        "a keyed server lists each owner's own files"
    );
    assert!(listed.contains(&id));
    let (_, listed) = status_and_body(&app, get("/v1/files", BETA)).await?;
    assert!(!listed.contains(&id), "{listed}");
    for path in [format!("/v1/files/{id}"), format!("/v1/files/{id}/content")] {
        assert_eq!(
            status_and_body(&app, get(&path, BETA)).await?.0,
            StatusCode::NOT_FOUND,
            "{path}"
        );
        assert_eq!(
            status_and_body(&app, get(&path, ALPHA)).await?.0,
            StatusCode::OK,
            "{path}"
        );
    }
    let delete = |key| {
        with_key(
            Request::delete(format!("/v1/files/{id}"))
                .body(Body::empty())
                .unwrap(),
            key,
        )
    };
    assert_eq!(
        status_and_body(&app, delete(BETA)).await?.0,
        StatusCode::NOT_FOUND
    );
    assert_eq!(
        status_and_body(&app, delete(ALPHA)).await?.0,
        StatusCode::OK
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_session_id_one_owner_holds_is_closed_to_another() -> anyhow::Result<()> {
    let (_dir, engine) = tiny_engine().await?;
    let app = keyed_router(&engine).await?;
    let session = json!({"messages": [{"role": {"Left": "user"}, "content": {"Left": "hello"}}]});
    let path = "/v1/sessions/session_named";
    let (status, body) =
        status_and_body(&app, send_json("PUT", path, ALPHA, session.clone())).await?;
    assert_eq!(status, StatusCode::OK, "{body}");

    assert_eq!(
        status_and_body(&app, get(path, BETA)).await?.0,
        StatusCode::NOT_FOUND
    );
    let (status, body) = status_and_body(&app, send_json("PUT", path, BETA, session)).await?;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body.contains("in use"), "{body}");
    let delete = with_key(Request::delete(path).body(Body::empty())?, BETA);
    status_and_body(&app, delete).await?;
    assert_eq!(
        status_and_body(&app, get(path, ALPHA)).await?.0,
        StatusCode::OK
    );

    // an agentic chat naming the id is refused too, rather than run in, or replace, the other owner's session
    let chat = json!({
        "model": "default",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": MAX_TOKENS,
        "session_id": "session_named",
        "max_tool_rounds": 1,
    });
    let (status, body) =
        status_and_body(&app, send_json("POST", "/v1/chat/completions", BETA, chat)).await?;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body.contains("in use"), "{body}");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_stored_response_is_read_and_continued_only_by_its_owner() -> anyhow::Result<()> {
    let (_dir, engine) = tiny_engine().await?;
    let app = keyed_router(&engine).await?;
    let create = json!({"model": "default", "input": "hello", "max_output_tokens": MAX_TOKENS});
    let (status, body) =
        status_and_body(&app, send_json("POST", "/v1/responses", ALPHA, create)).await?;
    assert_eq!(status, StatusCode::OK, "{body}");
    let id = serde_json::from_str::<Value>(&body)?["id"]
        .as_str()
        .unwrap()
        .to_string();

    let path = format!("/v1/responses/{id}");
    assert_eq!(
        status_and_body(&app, get(&path, BETA)).await?.0,
        StatusCode::NOT_FOUND
    );
    assert_eq!(
        status_and_body(&app, get(&path, ALPHA)).await?.0,
        StatusCode::OK
    );
    let follow_up = |key| {
        let body = json!({"model": "default", "input": "again", "previous_response_id": id, "max_output_tokens": MAX_TOKENS});
        send_json("POST", "/v1/responses", key, body)
    };
    assert_eq!(
        status_and_body(&app, follow_up(BETA)).await?.0,
        StatusCode::NOT_FOUND
    );
    assert_eq!(
        status_and_body(&app, follow_up(ALPHA)).await?.0,
        StatusCode::OK
    );
    Ok(())
}
