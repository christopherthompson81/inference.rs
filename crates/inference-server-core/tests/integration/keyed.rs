//! A keyed server: each key's owner reaches only what it stored, and a request without a known key is refused.

use axum::{
    Router,
    body::Body,
    http::{Request, StatusCode, header::AUTHORIZATION},
};
use inference_server_core::{
    auth::{ApiKeys, Auth, SESSION_COOKIE},
    inference_server_router_builder::InferenceRsServerRouterBuilder,
    mcp_server::{MCP_ROUTE, create_mcp_router},
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

fn with_cookie(mut request: Request<Body>, cookie: &str) -> Request<Body> {
    request
        .headers_mut()
        .insert("cookie", cookie.parse().expect("cookie"));
    request
}

/// Signs a browser in with `key` and returns the cookie it then carries.
async fn sign_in(app: &Router, key: &str) -> anyhow::Result<String> {
    let request = Request::post("/auth/session")
        .header("content-type", "application/json")
        .body(Body::from(json!({"key": key}).to_string()))?;
    let response = app.clone().oneshot(request).await?;
    assert_eq!(response.status(), StatusCode::NO_CONTENT);
    let set_cookie = response.headers()["set-cookie"].to_str()?.to_string();
    assert!(
        set_cookie.contains("HttpOnly") && set_cookie.contains("SameSite=Strict"),
        "{set_cookie}"
    );
    let pair = set_cookie.split(';').next().unwrap().to_string();
    assert!(pair.starts_with(&format!("{SESSION_COOKIE}=")), "{pair}");
    assert!(
        !pair.contains(key),
        "the cookie names a session, never the key"
    );
    Ok(pair)
}

#[tokio::test(flavor = "multi_thread")]
async fn a_browser_signs_in_with_a_key_and_out_again() -> anyhow::Result<()> {
    let (_dir, engine) = tiny_engine().await?;
    let app = keyed_router(&engine).await?;
    let wrong = Request::post("/auth/session")
        .header("content-type", "application/json")
        .body(Body::from(json!({"key": "key-gamma"}).to_string()))?;
    assert_eq!(
        status_and_body(&app, wrong).await?.0,
        StatusCode::UNAUTHORIZED
    );

    let cookie = sign_in(&app, ALPHA).await?;
    let models = || Request::get("/v1/models").body(Body::empty()).unwrap();
    assert_eq!(
        status_and_body(&app, with_cookie(models(), &cookie))
            .await?
            .0,
        StatusCode::OK
    );
    let forged = format!("{SESSION_COOKIE}=0123456789abcdef");
    assert_eq!(
        status_and_body(&app, with_cookie(models(), &forged))
            .await?
            .0,
        StatusCode::UNAUTHORIZED
    );

    // another port or subdomain of this host is the same site, so only the origin tells its forms apart
    let upload = || {
        let request = multipart_upload(&[
            ("purpose", None, "user_data"),
            ("file", Some("table.csv"), "a,b\n"),
        ]);
        with_cookie(request, &cookie)
    };
    let mut same_site = upload();
    same_site
        .headers_mut()
        .insert("sec-fetch-site", "same-site".parse()?);
    assert_eq!(
        status_and_body(&app, same_site).await?.0,
        StatusCode::UNAUTHORIZED
    );
    let mut other_port = upload();
    other_port
        .headers_mut()
        .insert("origin", "http://localhost:8080".parse()?);
    other_port
        .headers_mut()
        .insert("host", "localhost:1234".parse()?);
    assert_eq!(
        status_and_body(&app, other_port).await?.0,
        StatusCode::UNAUTHORIZED
    );
    let mut same_origin = upload();
    same_origin
        .headers_mut()
        .insert("sec-fetch-site", "same-origin".parse()?);
    assert_eq!(status_and_body(&app, same_origin).await?.0, StatusCode::OK);
    let planted = format!("{cookie}; {SESSION_COOKIE}=0123456789abcdef");
    assert_eq!(
        status_and_body(&app, with_cookie(models(), &planted))
            .await?
            .0,
        StatusCode::UNAUTHORIZED
    );

    let sign_out = with_cookie(
        Request::delete("/auth/session").body(Body::empty())?,
        &cookie,
    );
    assert_eq!(
        status_and_body(&app, sign_out).await?.0,
        StatusCode::NO_CONTENT
    );
    assert_eq!(
        status_and_body(&app, with_cookie(models(), &cookie))
            .await?
            .0,
        StatusCode::UNAUTHORIZED
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn mcp_requires_a_key_on_a_keyed_server() -> anyhow::Result<()> {
    let (_dir, engine) = tiny_engine().await?;
    let auth = Auth::new(ApiKeys::parse(KEYS)?);
    let app = create_mcp_router(&engine, auth.clone());
    let initialize = || {
        Request::post(MCP_ROUTE)
            .header("content-type", "application/json")
            .body(Body::from(
                json!({"jsonrpc": "2.0", "id": 1, "method": "initialize"}).to_string(),
            ))
            .unwrap()
    };
    assert_eq!(
        status_and_body(&app, initialize()).await?.0,
        StatusCode::UNAUTHORIZED
    );
    let (status, body) = status_and_body(&app, with_key(initialize(), BETA)).await?;
    assert_eq!(status, StatusCode::OK, "{body}");
    // no browser talks MCP, so a signed-in cookie is no key there
    let token = auth.as_ref().unwrap().sign_in(ALPHA).unwrap();
    let cookie = format!("{SESSION_COOKIE}={token}");
    assert_eq!(
        status_and_body(&app, with_cookie(initialize(), &cookie))
            .await?
            .0,
        StatusCode::UNAUTHORIZED
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn each_owner_keeps_its_own_web_ui_chats() -> anyhow::Result<()> {
    let cache = tempfile::tempdir()?;
    // the UI keeps its chats under the cache dir; nextest runs each test in its own process
    unsafe { std::env::set_var("XDG_CACHE_HOME", cache.path()) };
    let (_dir, engine) = tiny_engine().await?;
    let auth = Auth::new(ApiKeys::parse(KEYS)?);
    let api = InferenceRsServerRouterBuilder::new()
        .with_engine(&engine)
        .with_auth(auth.clone())
        .build()
        .await?;
    let options = inference_webui::UiOptions {
        auth,
        ..Default::default()
    };
    let app = inference_webui::mount(api, &engine, options, Default::default()).await?;

    for page in ["/ui", "/ui/index.html"] {
        let request = Request::get(page).body(Body::empty())?;
        let status = status_and_body(&app, request).await?.0;
        assert_eq!(
            status,
            StatusCode::OK,
            "{page} loads to show the sign-in prompt"
        );
    }
    let slash = Request::get("/ui/").body(Body::empty())?;
    assert_eq!(
        status_and_body(&app, slash).await?.0,
        StatusCode::PERMANENT_REDIRECT
    );
    let unkeyed = Request::get("/ui/api/list_chats").body(Body::empty())?;
    assert_eq!(
        status_and_body(&app, unkeyed).await?.0,
        StatusCode::UNAUTHORIZED
    );

    let (alpha, beta) = (sign_in(&app, ALPHA).await?, sign_in(&app, BETA).await?);
    let models = with_cookie(
        Request::get("/ui/api/list_models").body(Body::empty())?,
        &alpha,
    );
    let (_, models) = status_and_body(&app, models).await?;
    let model = serde_json::from_str::<Value>(&models)?["models"][0]["name"]
        .as_str()
        .unwrap()
        .to_string();
    let new_chat = with_cookie(
        send_json("POST", "/ui/api/new_chat", "", json!({"model": model})),
        &alpha,
    );
    let (status, body) = status_and_body(&app, new_chat).await?;
    assert_eq!(status, StatusCode::OK, "{body}");
    let chat_id = serde_json::from_str::<Value>(&body)?["id"]
        .as_str()
        .unwrap()
        .to_string();

    let list = |cookie: &str| {
        with_cookie(
            Request::get("/ui/api/list_chats")
                .body(Body::empty())
                .unwrap(),
            cookie,
        )
    };
    assert!(
        status_and_body(&app, list(&alpha))
            .await?
            .1
            .contains(&chat_id)
    );
    assert!(
        !status_and_body(&app, list(&beta))
            .await?
            .1
            .contains(&chat_id)
    );
    let load = with_cookie(
        send_json("POST", "/ui/api/load_chat", "", json!({"id": chat_id})),
        &beta,
    );
    assert_eq!(status_and_body(&app, load).await?.0, StatusCode::NOT_FOUND);
    Ok(())
}
