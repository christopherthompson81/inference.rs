//! ## inference.rs server router builder.

use anyhow::Result;
use axum::{
    Extension, Router,
    extract::DefaultBodyLimit,
    http::{self, HeaderMap, Method, StatusCode, Uri, header::HeaderName},
    middleware,
    response::IntoResponse,
    routing::{get, post},
};
use tower_http::cors::{AllowOrigin, CorsLayer};
#[cfg(feature = "swagger-ui")]
use utoipa_swagger_ui::SwaggerUi;

#[cfg(feature = "swagger-ui")]
use crate::openapi_doc::get_openapi_doc;
use crate::{
    anthropic::{anthropic_count_tokens, anthropic_error_response, anthropic_messages},
    approvals::resolve_agent_approval,
    audio_analysis::{diarization, transcription, voice_activity},
    chat_completion::chatcompletions,
    completions::completions,
    embeddings::embeddings,
    files::{
        delete_file, get_container_file, get_container_file_content, get_file, get_file_content,
        list_container_files, list_files, upload_file,
    },
    handler_core::{ApiError, ApiErrorKind, openai_error_response},
    handlers::{
        add_model, add_model_alias, calibration_apply, calibration_start, calibration_status,
        delete_session, get_model_status, get_session, health, model_cache_stats,
        model_speculative_stats, models, put_session, re_isq, reload_model, remove_model,
        set_default_model, system_doctor, system_info, tune_model, unload_model,
    },
    image_generation::image_generation,
    lora_adapters::{list_lora_adapters, load_lora_adapter, unload_lora_adapter},
    metrics::{ObservabilityConfig, ObservabilityState, metrics, metrics_disabled, observe_http},
    responses::{cancel_response, create_response, delete_response, get_response},
    route_registry::{
        ADD_MODEL_ROUTE, AGENT_APPROVAL_ROUTE, ANTHROPIC_COUNT_TOKENS_ROUTE,
        ANTHROPIC_MESSAGES_ROUTE, AUTH_SESSION_ROUTE, CALIBRATION_APPLY_ROUTE,
        CALIBRATION_START_ROUTE, CALIBRATION_STATUS_ROUTE, CANCEL_RESPONSE_ROUTE,
        CHAT_COMPLETIONS_ROUTE, COMPLETIONS_ROUTE, CONTAINER_FILE_CONTENT_ROUTE,
        CONTAINER_FILE_ROUTE, CONTAINER_FILES_ROUTE, DEFAULT_MODEL_ROUTE, DIARIZATION_ROUTE,
        EMBEDDINGS_ROUTE, FILE_CONTENT_ROUTE, FILE_ROUTE, FILES_ROUTE, HEALTH_ROUTE,
        IMAGE_GENERATION_ROUTE, LIST_LORA_ADAPTERS_ROUTE, LOAD_LORA_ADAPTER_ROUTE,
        MODEL_ALIAS_ROUTE, MODEL_CACHE_STATS_ROUTE, MODEL_SPECULATIVE_STATS_ROUTE,
        MODEL_STATUS_ROUTE, MODELS_ROUTE, RE_ISQ_ROUTE, RELOAD_MODEL_ROUTE, REMOVE_MODEL_ROUTE,
        RESPONSE_ROUTE, RESPONSES_ROUTE, ROOT_ROUTE, SESSION_ROUTE, SKILL_VERSIONS_ROUTE,
        SKILLS_ROUTE, SPEECH_GENERATION_ROUTE, SYSTEM_DOCTOR_ROUTE, SYSTEM_INFO_ROUTE,
        TRANSCRIPTION_ROUTE, TUNE_MODEL_ROUTE, UNLOAD_LORA_ADAPTER_ROUTE, UNLOAD_MODEL_ROUTE,
        VOICE_ACTIVITY_ROUTE,
    },
    skills::{list_skill_versions, list_skills, upload_skill, upload_skill_version},
    speech_generation::speech_generation,
};
use inference_api::Engine;

// NOTE(EricLBuehler): Accept up to 50mb input
const N_INPUT_SIZE: usize = 50;
const MB_TO_B: usize = 1024 * 1024; // 1024 kb in a mb
const ROUTE_NOT_FOUND_MESSAGE: &str = "The requested API route was not found.";
const METHOD_NOT_ALLOWED_MESSAGE: &str = "The requested HTTP method is not allowed for this route.";

/// This is the axum default request body limit for the router. Accept up to 50mb input.
pub const DEFAULT_MAX_BODY_LIMIT: usize = N_INPUT_SIZE * MB_TO_B;

/// A builder for creating a inference.rs server router with configurable options.
///
/// ### Examples
///
/// Basic usage:
/// ```ignore
/// use inference_server_core::inference_server_router_builder::InferenceRsServerRouterBuilder;
///
/// let router = InferenceRsServerRouterBuilder::new()
///     .with_engine(&engine)
///     .build()
///     .await?;
/// ```
///
/// With custom configuration:
/// ```ignore
/// use inference_server_core::inference_server_router_builder::InferenceRsServerRouterBuilder;
///
/// let router = InferenceRsServerRouterBuilder::new()
///     .with_engine(&engine)
///     .with_include_swagger_routes(false)
///     .with_base_path("/api/inference")
///     .build()
///     .await?;
/// ```
pub struct InferenceRsServerRouterBuilder {
    /// The engine the routes serve; its agent policy, skill store and adapter policy apply to every request.
    engine: Option<Engine>,
    /// Whether to include Swagger/OpenAPI documentation routes.
    /// Only available when the `swagger-ui` feature is enabled.
    #[cfg(feature = "swagger-ui")]
    include_swagger_routes: bool,
    /// Optional base path prefix for Swagger UI routes.
    /// Only available when the `swagger-ui` feature is enabled.
    #[cfg(feature = "swagger-ui")]
    base_path: Option<String>,
    /// Optional CORS allowed origins
    allowed_origins: Option<Vec<String>>,
    /// Optional axum default request body limit
    max_body_limit: Option<usize>,
    observability: ObservabilityConfig,
    file_listing: bool,
    model_management: bool,
    auth: Option<std::sync::Arc<crate::auth::Auth>>,
}

impl Default for InferenceRsServerRouterBuilder {
    /// Creates a new builder with default configuration.
    fn default() -> Self {
        Self {
            engine: None,
            #[cfg(feature = "swagger-ui")]
            include_swagger_routes: true,
            #[cfg(feature = "swagger-ui")]
            base_path: None,
            allowed_origins: None,
            max_body_limit: None,
            observability: ObservabilityConfig::default(),
            file_listing: false,
            model_management: false,
            auth: None,
        }
    }
}

impl InferenceRsServerRouterBuilder {
    /// Creates a new `InferenceRsServerRouterBuilder` with default settings.
    ///
    /// This is equivalent to calling `Default::default()`.
    pub fn new() -> Self {
        Default::default()
    }

    /// Serves a loaded engine; each request acts through it for the request's owner.
    pub fn with_engine(mut self, engine: &Engine) -> Self {
        self.engine = Some(engine.clone());
        self
    }

    /// Configures whether to include OpenAPI doc routes.
    ///
    /// When enabled (default), the router will include routes for Swagger UI
    /// at `/docs` and the OpenAPI specification at `/api-doc/openapi.json`.
    /// These routes respect the configured base path if one is set.
    ///
    /// Only available when the `swagger-ui` feature is enabled.
    #[cfg(feature = "swagger-ui")]
    pub fn with_include_swagger_routes(mut self, include_swagger_routes: bool) -> Self {
        self.include_swagger_routes = include_swagger_routes;
        self
    }

    /// Sets a base path prefix for Swagger UI routes.
    ///
    /// When set, Swagger UI routes will be prefixed with the given path. This is
    /// useful when including the inference.rs server instance in another axum project.
    ///
    /// Only available when the `swagger-ui` feature is enabled.
    #[cfg(feature = "swagger-ui")]
    pub fn with_base_path(mut self, base_path: &str) -> Self {
        self.base_path = Some(base_path.to_owned());
        self
    }

    /// Sets the CORS allowed origins.
    pub fn with_allowed_origins(mut self, origins: Vec<String>) -> Self {
        self.allowed_origins = Some(origins);
        self
    }

    /// Sets the axum default request body limit.
    pub fn with_max_body_limit(mut self, max_body_limit: usize) -> Self {
        self.max_body_limit = Some(max_body_limit);
        self
    }

    /// Lets `GET /v1/files` list every stored file. Off by default: the store is shared by every client, and file ids
    /// otherwise reach only the clients they were handed to.
    pub fn with_file_listing(mut self, file_listing: bool) -> Self {
        self.file_listing = file_listing;
        self
    }

    /// Serves the routes that change the served models; off by default, as adding one reads any path or hub repo.
    pub fn with_model_management(mut self, model_management: bool) -> Self {
        self.model_management = model_management;
        self
    }

    /// Requires a key on all but health probes and sign in, scoping each request to its owner; later routes aren't.
    pub fn with_api_keys(self, keys: crate::auth::ApiKeys) -> Self {
        self.with_auth(crate::auth::Auth::new(keys))
    }

    /// As [`Self::with_api_keys`], sharing `auth` (and its signed-in browsers) with routers built beside this one.
    pub fn with_auth(mut self, auth: Option<std::sync::Arc<crate::auth::Auth>>) -> Self {
        self.auth = auth;
        self
    }

    /// Sets server observability options.
    pub fn with_observability_config(mut self, observability: ObservabilityConfig) -> Self {
        self.observability = observability;
        self
    }

    /// Builds the configured axum router.
    pub async fn build(self) -> Result<Router> {
        if self.observability.metrics {
            crate::metrics::install_prometheus_recorder();
        }
        let engine = self
            .engine
            .ok_or_else(|| anyhow::anyhow!("an engine must be set; use `with_engine`"))?;
        let router_max_body_limit = self.max_body_limit.unwrap_or(DEFAULT_MAX_BODY_LIMIT);
        let observability = ObservabilityState::with_max_body_bytes(
            self.observability.clone(),
            engine.clone(),
            router_max_body_limit,
        );

        let mut router = init_router(
            engine,
            self.allowed_origins,
            router_max_body_limit,
            &self.observability,
            self.model_management,
        )?;

        #[cfg(feature = "swagger-ui")]
        if self.include_swagger_routes {
            let prefix = self.base_path.as_deref().unwrap_or("");
            let doc = get_openapi_doc(None);
            router = router.merge(
                SwaggerUi::new(format!("{prefix}/docs"))
                    .external_url_unchecked(format!("{prefix}/api-doc/openapi.json"), doc),
            );
        }

        router = router
            .layer(Extension(crate::files::FileListing(self.file_listing)))
            .layer(middleware::from_fn_with_state(observability, observe_http));
        if let Some(auth) = &self.auth {
            router = router.layer(Extension(auth.clone()));
        }
        Ok(crate::auth::require(
            router,
            self.auth,
            crate::auth::API_GUARD,
        ))
    }
}

/// Initializes and configures the underlying axum router with InferenceRs API endpoints.
///
/// This function creates a router with all the necessary API endpoints,
/// CORS configuration, and body size limits.
fn init_router(
    engine: Engine,
    allowed_origins: Option<Vec<String>>,
    router_max_body_limit: usize,
    observability: &ObservabilityConfig,
    model_management: bool,
) -> Result<Router> {
    let allow_origin = if let Some(origins) = allowed_origins {
        let parsed_origins: Result<Vec<_>, _> = origins.into_iter().map(|o| o.parse()).collect();

        match parsed_origins {
            Ok(origins) => AllowOrigin::list(origins),
            Err(_) => anyhow::bail!("Invalid origin format"),
        }
    } else {
        AllowOrigin::any()
    };

    let metrics_route = if observability.metrics {
        get(metrics)
    } else {
        get(metrics_disabled)
    };

    let cors_layer = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST, Method::PUT, Method::DELETE])
        .allow_headers([
            http::header::CONTENT_TYPE,
            http::header::AUTHORIZATION,
            HeaderName::from_static(crate::auth::API_KEY_HEADER),
            HeaderName::from_static("anthropic-version"),
            HeaderName::from_static("anthropic-beta"),
            HeaderName::from_static("x-request-id"),
            HeaderName::from_static("request-id"),
        ])
        .expose_headers([
            HeaderName::from_static("x-request-id"),
            HeaderName::from_static("request-id"),
        ])
        .allow_origin(allow_origin);

    let mut router = Router::new()
        .route(CHAT_COMPLETIONS_ROUTE.path, post(chatcompletions))
        .route(ANTHROPIC_MESSAGES_ROUTE.path, post(anthropic_messages))
        .route(
            ANTHROPIC_COUNT_TOKENS_ROUTE.path,
            post(anthropic_count_tokens),
        )
        .route(COMPLETIONS_ROUTE.path, post(completions))
        .route(EMBEDDINGS_ROUTE.path, post(embeddings))
        .route(MODELS_ROUTE.path, get(models))
        .route(LIST_LORA_ADAPTERS_ROUTE.path, get(list_lora_adapters))
        .route(UNLOAD_MODEL_ROUTE.path, post(unload_model))
        .route(RELOAD_MODEL_ROUTE.path, post(reload_model))
        .route(MODEL_STATUS_ROUTE.path, post(get_model_status))
        .route(MODEL_CACHE_STATS_ROUTE.path, get(model_cache_stats))
        .route(
            MODEL_SPECULATIVE_STATS_ROUTE.path,
            get(model_speculative_stats),
        )
        .route(TUNE_MODEL_ROUTE.path, post(tune_model))
        .route(SYSTEM_INFO_ROUTE.path, get(system_info))
        .route(SYSTEM_DOCTOR_ROUTE.path, post(system_doctor))
        .route(HEALTH_ROUTE.path, get(health))
        .route(
            AUTH_SESSION_ROUTE.path,
            post(crate::auth::sign_in).delete(crate::auth::sign_out),
        )
        .route("/metrics", metrics_route)
        .route(ROOT_ROUTE.path, get(health))
        .route(RE_ISQ_ROUTE.path, post(re_isq))
        .route(CALIBRATION_START_ROUTE.path, post(calibration_start))
        .route(
            CALIBRATION_STATUS_ROUTE.path,
            axum::routing::get(calibration_status),
        )
        .route(CALIBRATION_APPLY_ROUTE.path, post(calibration_apply))
        .route(IMAGE_GENERATION_ROUTE.path, post(image_generation))
        .route(FILES_ROUTE.path, get(list_files).post(upload_file))
        .route(FILE_ROUTE.path, get(get_file).delete(delete_file))
        .route(FILE_CONTENT_ROUTE.path, get(get_file_content))
        .route(CONTAINER_FILES_ROUTE.path, get(list_container_files))
        .route(CONTAINER_FILE_ROUTE.path, get(get_container_file))
        .route(
            CONTAINER_FILE_CONTENT_ROUTE.path,
            get(get_container_file_content),
        )
        .route(SPEECH_GENERATION_ROUTE.path, post(speech_generation))
        .route(TRANSCRIPTION_ROUTE.path, post(transcription))
        .route(VOICE_ACTIVITY_ROUTE.path, post(voice_activity))
        .route(DIARIZATION_ROUTE.path, post(diarization))
        .route(AGENT_APPROVAL_ROUTE.path, post(resolve_agent_approval))
        .route(RESPONSES_ROUTE.path, post(create_response))
        .route(SKILLS_ROUTE.path, get(list_skills).post(upload_skill))
        .route(
            SKILL_VERSIONS_ROUTE.path,
            get(list_skill_versions).post(upload_skill_version),
        )
        .route(
            RESPONSE_ROUTE.path,
            get(get_response).delete(delete_response),
        )
        .route(CANCEL_RESPONSE_ROUTE.path, post(cancel_response))
        .route(
            SESSION_ROUTE.path,
            get(get_session).put(put_session).delete(delete_session),
        );

    if model_management {
        tracing::warn!(
            "model management is enabled; authorized clients can load models from any path or hub repo the server can reach"
        );
        router = router
            .route(ADD_MODEL_ROUTE.path, post(add_model))
            .route(REMOVE_MODEL_ROUTE.path, post(remove_model))
            .route(DEFAULT_MODEL_ROUTE.path, post(set_default_model))
            .route(MODEL_ALIAS_ROUTE.path, post(add_model_alias));
    }

    let lora_adapter_api = engine.adapter_config();
    if lora_adapter_api.enabled() {
        if let Some(root) = lora_adapter_api.allowed_root() {
            tracing::warn!(
                adapter_root = %root.display(),
                "runtime LoRA adapter management is enabled; authorized clients can load adapter files from this root"
            );
        } else {
            tracing::warn!(
                "runtime LoRA adapter management is enabled without an adapter root; authorized clients can load any adapter path readable by the server"
            );
        }
        router = router
            .route(LOAD_LORA_ADAPTER_ROUTE.path, post(load_lora_adapter))
            .route(UNLOAD_LORA_ADAPTER_ROUTE.path, post(unload_lora_adapter));
    }

    let router = router
        .fallback(api_route_not_found)
        .method_not_allowed_fallback(api_method_not_allowed)
        .layer(cors_layer)
        .layer(DefaultBodyLimit::max(router_max_body_limit))
        .with_state(engine);

    Ok(router)
}

fn path_is_in_namespace(path: &str, namespace: &str) -> bool {
    path == namespace
        || path
            .strip_prefix(namespace)
            .is_some_and(|suffix| suffix.starts_with('/'))
}

fn is_anthropic_api_request(uri: &Uri, headers: &HeaderMap) -> bool {
    if path_is_in_namespace(uri.path(), ANTHROPIC_MESSAGES_ROUTE.path) {
        return true;
    }
    if !path_is_in_namespace(uri.path(), SKILLS_ROUTE.path) {
        return false;
    }
    headers.contains_key("anthropic-version")
        || headers.contains_key("anthropic-beta")
        || uri.query().is_some_and(|query| {
            url::form_urlencoded::parse(query.as_bytes()).any(|(key, _)| key == "source")
        })
}

fn is_versioned_api_path(path: &str) -> bool {
    path == "/v1" || path.starts_with("/v1/")
}

fn api_protocol_error(
    uri: &Uri,
    headers: &HeaderMap,
    status: StatusCode,
    error: ApiError,
) -> axum::response::Response {
    let mut response = if is_anthropic_api_request(uri, headers) {
        anthropic_error_response(error)
    } else {
        openai_error_response(error)
    };
    *response.status_mut() = status;
    response
}

async fn api_route_not_found(uri: Uri, headers: HeaderMap) -> axum::response::Response {
    if !is_versioned_api_path(uri.path()) {
        return StatusCode::NOT_FOUND.into_response();
    }

    api_protocol_error(
        &uri,
        &headers,
        StatusCode::NOT_FOUND,
        ApiError::new(
            ApiErrorKind::NotFound,
            ROUTE_NOT_FOUND_MESSAGE,
            Some("not_found"),
            None,
        ),
    )
}

async fn api_method_not_allowed(uri: Uri, headers: HeaderMap) -> axum::response::Response {
    if !is_versioned_api_path(uri.path()) {
        return StatusCode::METHOD_NOT_ALLOWED.into_response();
    }

    api_protocol_error(
        &uri,
        &headers,
        StatusCode::METHOD_NOT_ALLOWED,
        ApiError::new(
            ApiErrorKind::InvalidRequest,
            METHOD_NOT_ALLOWED_MESSAGE,
            Some("method_not_allowed"),
            None,
        ),
    )
}

#[cfg(test)]
mod tests {
    use axum::body::to_bytes;

    use super::*;

    async fn response_body(response: axum::response::Response) -> serde_json::Value {
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    #[tokio::test]
    async fn unknown_openai_routes_use_openai_errors() {
        let response = api_route_not_found(Uri::from_static("/v1/unknown"), HeaderMap::new()).await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let body = response_body(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["code"], "not_found");
    }

    #[tokio::test]
    async fn unknown_anthropic_message_routes_use_anthropic_errors() {
        let response =
            api_route_not_found(Uri::from_static("/v1/messages/unknown"), HeaderMap::new()).await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let body = response_body(response).await;
        assert_eq!(body["type"], "error");
        assert_eq!(body["error"]["type"], "not_found_error");
    }

    #[tokio::test]
    async fn wrong_openai_methods_use_openai_errors() {
        let response =
            api_method_not_allowed(Uri::from_static("/v1/chat/completions"), HeaderMap::new())
                .await;
        assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
        let body = response_body(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["code"], "method_not_allowed");
    }

    #[tokio::test]
    async fn wrong_anthropic_message_methods_use_anthropic_errors() {
        let response =
            api_method_not_allowed(Uri::from_static("/v1/messages"), HeaderMap::new()).await;
        assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
        let body = response_body(response).await;
        assert_eq!(body["type"], "error");
        assert_eq!(body["error"]["type"], "invalid_request_error");
    }

    #[tokio::test]
    async fn anthropic_skill_fallbacks_use_anthropic_errors() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "anthropic-version",
            "2023-06-01".parse().expect("valid header value"),
        );
        let response = api_method_not_allowed(Uri::from_static("/v1/skills"), headers).await;
        assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
        let body = response_body(response).await;
        assert_eq!(body["type"], "error");
        assert_eq!(body["error"]["type"], "invalid_request_error");

        let response = api_route_not_found(
            Uri::from_static("/v1/skills/unknown?source=custom"),
            HeaderMap::new(),
        )
        .await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let body = response_body(response).await;
        assert_eq!(body["type"], "error");
        assert_eq!(body["error"]["type"], "not_found_error");
    }
}
