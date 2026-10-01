use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use axum::body::Body;
use axum::extract::{DefaultBodyLimit, Request, State};
use axum::http::{Method, Response, StatusCode};
use axum::middleware::{self, Next};
use axum::response::Redirect;
use axum::routing::{get, get_service, post};
use axum::{Extension, Router};
use include_dir::{Dir, include_dir};
use indexmap::IndexMap;
use inference_api::engine::SearchEmbeddingModel;
use inference_api::lora_routing::DEFAULT_MODEL_ID;
use inference_api::openai::{Modality, ModelCategory};
use inference_api::{Engine, engine::AgenticSpec};
use inference_server_core::{
    auth::{Auth, Guard, Owner, require},
    inference_server_router_builder::DEFAULT_MAX_BODY_LIMIT,
    metrics::{ObservabilityConfig, ObservabilityState, observe_http},
    route_registry::{RouteInfo, RouteKind},
};
use tokio::fs;
use tower_http::services::ServeDir;

use crate::handlers::api::*;
use crate::types::{AppState, GenerationParams, UiModelInfo};
use crate::utils::get_cache_dir;

mod chat;
mod handlers;
mod types;
mod utils;

static STATIC_DIR: Dir = include_dir!("$CARGO_MANIFEST_DIR/static");
const UI_API_PREFIX: &str = "/api/";

pub(crate) const UI_UPLOAD_IMAGE_ROUTE: RouteInfo =
    RouteInfo::new("/api/upload_image", "POST", RouteKind::Ui);
pub(crate) const UI_UPLOAD_VIDEO_ROUTE: RouteInfo =
    RouteInfo::new("/api/upload_video", "POST", RouteKind::Ui);
pub(crate) const UI_UPLOAD_TEXT_ROUTE: RouteInfo =
    RouteInfo::new("/api/upload_text", "POST", RouteKind::Ui);
pub(crate) const UI_UPLOAD_AUDIO_ROUTE: RouteInfo =
    RouteInfo::new("/api/upload_audio", "POST", RouteKind::Ui);
pub(crate) const UI_LIST_MODELS_ROUTE: RouteInfo =
    RouteInfo::new("/api/list_models", "GET", RouteKind::Ui);
pub(crate) const UI_SELECT_MODEL_ROUTE: RouteInfo =
    RouteInfo::new("/api/select_model", "POST", RouteKind::Ui);
pub(crate) const UI_LIST_CHATS_ROUTE: RouteInfo =
    RouteInfo::new("/api/list_chats", "GET", RouteKind::Ui);
pub(crate) const UI_NEW_CHAT_ROUTE: RouteInfo =
    RouteInfo::new("/api/new_chat", "POST", RouteKind::Ui);
pub(crate) const UI_DELETE_CHAT_ROUTE: RouteInfo =
    RouteInfo::new("/api/delete_chat", "POST", RouteKind::Ui);
pub(crate) const UI_LOAD_CHAT_ROUTE: RouteInfo =
    RouteInfo::new("/api/load_chat", "POST", RouteKind::Ui);
pub(crate) const UI_RENAME_CHAT_ROUTE: RouteInfo =
    RouteInfo::new("/api/rename_chat", "POST", RouteKind::Ui);
pub(crate) const UI_APPEND_MESSAGE_ROUTE: RouteInfo =
    RouteInfo::new("/api/append_message", "POST", RouteKind::Ui);
pub(crate) const UI_EDIT_MESSAGE_ROUTE: RouteInfo =
    RouteInfo::new("/api/edit_message", "POST", RouteKind::Ui);
pub(crate) const UI_SET_TAIL_ROUTE: RouteInfo =
    RouteInfo::new("/api/set_tail", "POST", RouteKind::Ui);
pub(crate) const UI_FORK_SESSION_ROUTE: RouteInfo =
    RouteInfo::new("/api/fork_session", "POST", RouteKind::Ui);
pub(crate) const UI_SAVE_CHAT_SESSION_ROUTE: RouteInfo =
    RouteInfo::new("/api/save_chat_session", "POST", RouteKind::Ui);
pub(crate) const UI_RESTORE_CHAT_SESSION_ROUTE: RouteInfo =
    RouteInfo::new("/api/restore_chat_session", "POST", RouteKind::Ui);
pub(crate) const UI_SETTINGS_ROUTE: RouteInfo =
    RouteInfo::new("/api/settings", "GET", RouteKind::Ui);
pub(crate) const UI_CAPABILITIES_ROUTE: RouteInfo =
    RouteInfo::new("/api/capabilities", "GET", RouteKind::Ui);
pub(crate) const UI_MCP_TOOLS_ROUTE: RouteInfo =
    RouteInfo::new("/api/mcp_tools", "GET", RouteKind::Ui);
pub(crate) const UI_GENERATE_SPEECH_ROUTE: RouteInfo =
    RouteInfo::new("/api/generate_speech", "POST", RouteKind::Ui);
pub(crate) const UI_SPEECH_ROUTE: RouteInfo = RouteInfo::new("/speech", "GET", RouteKind::Ui);
pub(crate) const UI_UPLOADS_ROUTE: RouteInfo = RouteInfo::new("/uploads", "GET", RouteKind::Ui);
pub(crate) const UI_ROOT_ROUTE: RouteInfo = RouteInfo::new("/", "GET", RouteKind::Ui);
pub(crate) const UI_STATIC_ROUTE: RouteInfo = RouteInfo::new("/{*path}", "GET", RouteKind::Ui);

async fn static_handler(uri: axum::http::Uri) -> Response<Body> {
    let path = uri.path().trim_start_matches('/');
    let path = if path.is_empty() { "index.html" } else { path };
    if let Some(file) = STATIC_DIR.get_file(path) {
        let mime = mime_guess::from_path(path).first_or_octet_stream();
        Response::builder()
            .status(StatusCode::OK)
            .header(axum::http::header::CONTENT_TYPE, mime.as_ref())
            .body(Body::from(file.contents()))
            .unwrap()
    } else {
        // SPA fallback: serve index.html for unrecognized paths
        if let Some(file) = STATIC_DIR.get_file("index.html") {
            let mime = mime_guess::from_path("index.html").first_or_octet_stream();
            Response::builder()
                .status(StatusCode::OK)
                .header(axum::http::header::CONTENT_TYPE, mime.as_ref())
                .body(Body::from(file.contents()))
                .unwrap()
        } else {
            Response::builder()
                .status(StatusCode::NOT_FOUND)
                .body(Body::from("Not Found"))
                .unwrap()
        }
    }
}

// The models a chat can pick: loaded text, multimodal and speech models, without the `default` alias or adapter cards.
fn build_model_list(engine: &Engine) -> IndexMap<String, UiModelInfo> {
    let mut models = IndexMap::new();
    let Ok(cards) = engine.models() else {
        return models;
    };
    for card in cards.data {
        if card.id == DEFAULT_MODEL_ID || card.parent.is_some() {
            continue;
        }
        let kind = match card.category {
            Some(ModelCategory::Text) => "text",
            Some(ModelCategory::Multimodal) => "multimodal",
            Some(ModelCategory::Speech) => "speech",
            _ => continue,
        };
        let labels = |modalities: &[Modality]| modalities.iter().map(modality_label).collect();
        let (input_modalities, output_modalities) = card
            .modalities
            .map(|modalities| (labels(&modalities.input), labels(&modalities.output)))
            .unwrap_or_default();
        models.insert(
            card.id.clone(),
            UiModelInfo {
                name: card.id,
                kind: kind.to_string(),
                input_modalities,
                output_modalities,
                generation_defaults: GenerationParams::from_model_defaults(
                    card.generation_defaults.as_ref(),
                ),
            },
        );
    }
    models
}

fn modality_label(modality: &Modality) -> String {
    match modality {
        Modality::Text => "text",
        Modality::Audio => "audio",
        Modality::Vision => "vision",
        Modality::Video => "video",
        Modality::Embedding => "embedding",
    }
    .to_string()
}

pub use inference_server_core::route_registry::UI_ROUTE;

/// The tools the UI offers and reports; `from_agentic` reads them from the spec the engine was loaded with.
#[derive(Debug, Clone, Default)]
pub struct UiOptions {
    /// Web search and the model reranking its results; `None` when search is off.
    pub search: Option<SearchEmbeddingModel>,
    pub code_execution: bool,
    pub shell: bool,
    pub tool_dispatch_url: Option<String>,
    /// The keys the UI's data needs, shared with the API router so a signed-in browser reaches both.
    pub auth: Option<Arc<Auth>>,
}

impl UiOptions {
    pub fn from_agentic(agentic: &AgenticSpec) -> Self {
        Self {
            search: agentic.search.as_ref().map(|search| search.embedding_model),
            code_execution: agentic.code_execution.is_some(),
            shell: agentic.shell.is_some(),
            tool_dispatch_url: agentic.tool_dispatch_url.clone(),
            auth: None,
        }
    }
}

// The page and its assets load without a key, so a browser can show the sign-in prompt; data and media need one.
fn ui_public_requests(method: &Method, path: &str) -> bool {
    method == Method::GET
        && ![UI_API_PREFIX, UI_UPLOADS_ROUTE.path, UI_SPEECH_ROUTE.path]
            .iter()
            .any(|prefix| path.starts_with(prefix))
}

/// The UI state for each owner, made on first use from the open one.
struct OwnerStates {
    open: Arc<AppState>,
    by_owner: Mutex<HashMap<String, Arc<AppState>>>,
}

async fn owner_state(
    State(states): State<Arc<OwnerStates>>,
    Extension(owner): Extension<Owner>,
    mut request: Request,
    next: Next,
) -> Result<axum::response::Response, StatusCode> {
    let state = match owner.as_deref() {
        None => states.open.clone(),
        Some(owner) => {
            let mut by_owner = states.by_owner.lock().unwrap();
            match by_owner.get(owner) {
                Some(state) => state.clone(),
                None => {
                    let state = Arc::new(states.open.for_owner(owner).map_err(|error| {
                        tracing::error!("UI state for an owner: {error}");
                        StatusCode::INTERNAL_SERVER_ERROR
                    })?);
                    by_owner.insert(owner.to_string(), state.clone());
                    state
                }
            }
        }
    };
    request.extensions_mut().insert(state);
    Ok(next.run(request).await)
}

/// Nests the UI at `UI_ROUTE` in `app`, with the same request logging and metrics as the API routes.
pub async fn mount(
    app: Router,
    engine: &Engine,
    options: UiOptions,
    observability: ObservabilityConfig,
) -> Result<Router> {
    let observability = ObservabilityState::with_max_body_bytes(
        observability,
        engine.clone(),
        DEFAULT_MAX_BODY_LIMIT,
    );
    let auth = options.auth.clone();
    let ui = build_ui_router(engine, options)
        .await?
        .layer(middleware::from_fn_with_state(observability, observe_http));
    let guard = Guard {
        public: ui_public_requests,
        cookies: true,
    };
    // the page's base is `/ui/`, which the nest itself doesn't route
    let slash = format!("{UI_ROUTE}/");
    Ok(app
        .route(&slash, get(|| async { Redirect::permanent(UI_ROUTE) }))
        .nest(UI_ROUTE, require(ui, auth, guard)))
}

async fn build_ui_router(engine: &Engine, options: UiOptions) -> Result<Router> {
    let models = build_model_list(engine);

    let base_cache = get_cache_dir();
    let chats_dir = base_cache.join("chats");
    fs::create_dir_all(&chats_dir).await?;
    let speech_dir = base_cache.join("speech");
    fs::create_dir_all(&speech_dir).await?;
    let uploads_dir = base_cache.join("uploads");
    fs::create_dir_all(&uploads_dir).await?;
    inference_server_core::configure_ui_upload_dir(&uploads_dir).await?;

    let default_model = engine
        .default_model_id()
        .or_else(|| models.keys().next().cloned());

    let app_state = Arc::new(AppState {
        engine: engine.clone(),
        models,
        default_model: default_model.clone(),
        current: tokio::sync::RwLock::new(default_model),
        owner: None,
        chats_dir: chats_dir.to_string_lossy().to_string(),
        speech_dir: speech_dir.to_string_lossy().to_string(),
        current_chat: tokio::sync::RwLock::new(None),
        default_params: GenerationParams::default(),
        search_enabled: options.search.is_some(),
        search_embedding_model: options.search,
        code_execution_enabled: options.code_execution,
        shell_enabled: options.shell,
        tool_dispatch_url: options.tool_dispatch_url,
    });

    let router = Router::new()
        .route(UI_UPLOAD_IMAGE_ROUTE.path, post(upload_image))
        .route(UI_UPLOAD_VIDEO_ROUTE.path, post(upload_video))
        .route(UI_UPLOAD_TEXT_ROUTE.path, post(upload_text))
        .route(UI_UPLOAD_AUDIO_ROUTE.path, post(upload_audio))
        .route(UI_LIST_MODELS_ROUTE.path, get(list_models))
        .route(UI_SELECT_MODEL_ROUTE.path, post(select_model))
        .route(UI_LIST_CHATS_ROUTE.path, get(list_chats))
        .route(UI_NEW_CHAT_ROUTE.path, post(new_chat))
        .route(UI_DELETE_CHAT_ROUTE.path, post(delete_chat))
        .route(UI_LOAD_CHAT_ROUTE.path, post(load_chat))
        .route(UI_RENAME_CHAT_ROUTE.path, post(rename_chat))
        .route(UI_APPEND_MESSAGE_ROUTE.path, post(append_message))
        .route(UI_EDIT_MESSAGE_ROUTE.path, post(edit_message))
        .route(UI_SET_TAIL_ROUTE.path, post(set_tail))
        .route(UI_FORK_SESSION_ROUTE.path, post(fork_session))
        .route(UI_SAVE_CHAT_SESSION_ROUTE.path, post(save_chat_session))
        .route(
            UI_RESTORE_CHAT_SESSION_ROUTE.path,
            post(restore_chat_session),
        )
        .route(UI_SETTINGS_ROUTE.path, get(get_settings))
        .route(UI_CAPABILITIES_ROUTE.path, get(get_capabilities))
        .route(UI_MCP_TOOLS_ROUTE.path, get(list_mcp_tools))
        .route(UI_GENERATE_SPEECH_ROUTE.path, post(generate_speech))
        .nest_service(
            UI_SPEECH_ROUTE.path,
            get_service(ServeDir::new(speech_dir.clone())),
        )
        .nest_service(
            UI_UPLOADS_ROUTE.path,
            get_service(ServeDir::new(uploads_dir.clone())),
        )
        .route(UI_ROOT_ROUTE.path, get(static_handler))
        .route(UI_STATIC_ROUTE.path, get(static_handler))
        .layer(DefaultBodyLimit::max(50 * 1024 * 1024))
        .layer(middleware::from_fn_with_state(
            Arc::new(OwnerStates {
                open: app_state,
                by_owner: Mutex::new(HashMap::new()),
            }),
            owner_state,
        ));

    Ok(router)
}
