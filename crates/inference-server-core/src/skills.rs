use std::sync::Arc;

use anyhow::Result;
use axum::{
    extract::{
        multipart::{MultipartError, MultipartRejection},
        rejection::QueryRejection,
        Multipart, Path as AxumPath, Query, RawQuery,
    },
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    Extension, Json,
};
use serde::Serialize;

use crate::handler_core::{
    openai_error_response, ApiError, ApiErrorKind, ResponseErrorMessage,
    SERVICE_UNAVAILABLE_MESSAGE,
};

const ANTHROPIC_OVERLOADED_STATUS: u16 = 529;

use crate::skill_store::{
    invalid_skill_upload, skill_upload_too_large, ANTHROPIC_SKILL_SOURCE, CUSTOM_SKILL_SOURCE,
};
pub use crate::skill_store::{
    AnthropicSkillListObject, AnthropicSkillObject, AnthropicSkillVersionListObject,
    AnthropicSkillVersionObject, SkillFiles, SkillListObject, SkillListQuery, SkillObject,
    SkillStore, SkillVersionObject,
};

async fn read_skill_files(mut multipart: Multipart) -> Result<SkillFiles> {
    let mut files = SkillFiles::default();
    while let Some(field) = multipart.next_field().await.map_err(multipart_error)? {
        let field_name = field.name().unwrap_or_default().to_string();
        if field_name != "files" && field_name != "file" && field_name != "files[]" {
            continue;
        }
        let file_name = field
            .file_name()
            .ok_or_else(|| invalid_skill_upload("Uploaded skill file is missing a filename."))?
            .to_string();
        let bytes = field.bytes().await.map_err(multipart_error)?;
        files.push(file_name, bytes.to_vec())?;
    }
    Ok(files)
}

fn multipart_error(error: MultipartError) -> anyhow::Error {
    match error.status() {
        StatusCode::PAYLOAD_TOO_LARGE => skill_upload_too_large(error.body_text()),
        status if status.is_client_error() => invalid_skill_upload(error.body_text()),
        _ => error.into(),
    }
}

#[derive(Serialize)]
struct AnthropicSkillErrorBody {
    #[serde(rename = "type")]
    tp: &'static str,
    message: String,
}

#[derive(Serialize)]
struct AnthropicSkillError {
    #[serde(rename = "type")]
    tp: &'static str,
    error: AnthropicSkillErrorBody,
}

fn anthropic_skill_error_response(error: ApiError) -> axum::response::Response {
    let status = match error.kind {
        ApiErrorKind::InvalidRequest | ApiErrorKind::UnsupportedMediaType => {
            StatusCode::BAD_REQUEST
        }
        ApiErrorKind::NotFound => StatusCode::NOT_FOUND,
        ApiErrorKind::Conflict => StatusCode::CONFLICT,
        ApiErrorKind::PayloadTooLarge => StatusCode::PAYLOAD_TOO_LARGE,
        ApiErrorKind::RateLimited => StatusCode::TOO_MANY_REQUESTS,
        ApiErrorKind::Unavailable | ApiErrorKind::Internal => StatusCode::INTERNAL_SERVER_ERROR,
        ApiErrorKind::Overloaded => StatusCode::from_u16(ANTHROPIC_OVERLOADED_STATUS)
            .expect("Anthropic overloaded status must be valid"),
    };
    let error_type = match error.kind {
        ApiErrorKind::InvalidRequest | ApiErrorKind::UnsupportedMediaType => {
            "invalid_request_error"
        }
        ApiErrorKind::NotFound => "not_found_error",
        ApiErrorKind::Conflict => "conflict_error",
        ApiErrorKind::PayloadTooLarge => "request_too_large",
        ApiErrorKind::RateLimited => "rate_limit_error",
        ApiErrorKind::Unavailable | ApiErrorKind::Internal => "api_error",
        ApiErrorKind::Overloaded => "overloaded_error",
    };
    let message = if error.kind == ApiErrorKind::Overloaded {
        SERVICE_UNAVAILABLE_MESSAGE.to_string()
    } else {
        error.message
    };
    let mut response = (
        status,
        Json(AnthropicSkillError {
            tp: "error",
            error: AnthropicSkillErrorBody {
                tp: error_type,
                message: message.clone(),
            },
        }),
    )
        .into_response();
    response
        .extensions_mut()
        .insert(ResponseErrorMessage(message));
    response
}

fn protocol_error_response(error: ApiError, anthropic: bool) -> axum::response::Response {
    if anthropic {
        anthropic_skill_error_response(error)
    } else {
        openai_error_response(error)
    }
}

fn skill_error(error: anyhow::Error, anthropic: bool) -> axum::response::Response {
    let api_error = ApiError::from_error(error.as_ref(), ApiErrorKind::Internal);
    if matches!(
        api_error.kind,
        ApiErrorKind::Internal | ApiErrorKind::Unavailable | ApiErrorKind::Overloaded
    ) {
        tracing::error!(%error, "skill request failed");
    }
    protocol_error_response(api_error, anthropic)
}

fn multipart_rejection_error(error: MultipartRejection) -> ApiError {
    let kind = if error.status() == StatusCode::PAYLOAD_TOO_LARGE {
        ApiErrorKind::PayloadTooLarge
    } else {
        ApiErrorKind::InvalidRequest
    };
    let code = if kind == ApiErrorKind::PayloadTooLarge {
        "request_body_too_large"
    } else {
        "invalid_skill_upload"
    };
    ApiError::new(kind, error.body_text(), Some(code), Some("files"))
}

fn query_rejection_error(error: QueryRejection) -> ApiError {
    ApiError::new(
        ApiErrorKind::InvalidRequest,
        error.body_text(),
        Some("invalid_query"),
        None,
    )
}

fn prefers_anthropic_shape(
    headers: &HeaderMap,
    query: Option<&SkillListQuery>,
    raw_query: Option<&str>,
) -> bool {
    headers.contains_key("anthropic-version")
        || headers.contains_key("anthropic-beta")
        || query.and_then(|query| query.source.as_deref()).is_some()
        || raw_query.is_some_and(|raw_query| {
            url::form_urlencoded::parse(raw_query.as_bytes()).any(|(key, _)| key == "source")
        })
}

fn anthropic_list_response(
    skills: Vec<SkillObject>,
    query: Option<&SkillListQuery>,
) -> AnthropicSkillListObject {
    let mut data = match query.and_then(|query| query.source.as_deref()) {
        Some(ANTHROPIC_SKILL_SOURCE) => Vec::new(),
        Some(CUSTOM_SKILL_SOURCE) | None => skills
            .iter()
            .map(AnthropicSkillObject::from)
            .collect::<Vec<_>>(),
        Some(_) => Vec::new(),
    };

    let limit = query.and_then(|query| query.limit).unwrap_or(data.len());
    if limit < data.len() {
        data.truncate(limit);
    }
    let _ = query.and_then(|query| query.page.as_deref());

    AnthropicSkillListObject {
        data,
        has_more: false,
        next_page: None,
    }
}

#[utoipa::path(
    post,
    tag = "Mistral.rs",
    path = "/v1/skills",
    responses((status = 200, description = "Skill uploaded", body = SkillObject))
)]
pub async fn upload_skill(
    headers: HeaderMap,
    RawQuery(raw_query): RawQuery,
    Extension(store): Extension<Arc<SkillStore>>,
    payload: std::result::Result<Multipart, MultipartRejection>,
) -> axum::response::Response {
    let anthropic = prefers_anthropic_shape(&headers, None, raw_query.as_deref());
    let multipart = match payload {
        Ok(multipart) => multipart,
        Err(error) => return protocol_error_response(multipart_rejection_error(error), anthropic),
    };
    match async { store.create_skill(read_skill_files(multipart).await?) }.await {
        Ok(skill) if anthropic => Json(AnthropicSkillObject::from(&skill)).into_response(),
        Ok(skill) => Json(skill).into_response(),
        Err(error) => skill_error(error, anthropic),
    }
}

#[utoipa::path(
    get,
    tag = "Mistral.rs",
    path = "/v1/skills",
    responses((status = 200, description = "Uploaded skills", body = SkillListObject))
)]
pub async fn list_skills(
    headers: HeaderMap,
    RawQuery(raw_query): RawQuery,
    payload: std::result::Result<Query<SkillListQuery>, QueryRejection>,
    Extension(store): Extension<Arc<SkillStore>>,
) -> axum::response::Response {
    let query = match payload {
        Ok(Query(query)) => query,
        Err(error) => {
            let anthropic = prefers_anthropic_shape(&headers, None, raw_query.as_deref());
            return protocol_error_response(query_rejection_error(error), anthropic);
        }
    };
    let anthropic = prefers_anthropic_shape(&headers, Some(&query), raw_query.as_deref());
    match query.source.as_deref() {
        Some(source) if source != CUSTOM_SKILL_SOURCE && source != ANTHROPIC_SKILL_SOURCE => {
            return protocol_error_response(
                ApiError::new(
                    ApiErrorKind::InvalidRequest,
                    format!("Unsupported skill source `{source}`."),
                    Some("invalid_query"),
                    Some("source"),
                ),
                anthropic,
            );
        }
        _ => {}
    }
    match store.list() {
        Ok(data) if anthropic => Json(anthropic_list_response(data, Some(&query))).into_response(),
        Ok(data) => Json(SkillListObject {
            object: "list",
            data,
        })
        .into_response(),
        Err(error) => skill_error(error, anthropic),
    }
}

#[utoipa::path(
    post,
    tag = "Mistral.rs",
    path = "/v1/skills/{skill_id}/versions",
    responses((status = 200, description = "Skill version uploaded", body = SkillVersionObject))
)]
pub async fn upload_skill_version(
    AxumPath(skill_id): AxumPath<String>,
    headers: HeaderMap,
    RawQuery(raw_query): RawQuery,
    Extension(store): Extension<Arc<SkillStore>>,
    payload: std::result::Result<Multipart, MultipartRejection>,
) -> axum::response::Response {
    let anthropic = prefers_anthropic_shape(&headers, None, raw_query.as_deref());
    let multipart = match payload {
        Ok(multipart) => multipart,
        Err(error) => return protocol_error_response(multipart_rejection_error(error), anthropic),
    };
    match async { store.create_version(&skill_id, read_skill_files(multipart).await?) }.await {
        Ok(version) if anthropic => {
            Json(AnthropicSkillVersionObject::from(&version)).into_response()
        }
        Ok(version) => Json(version).into_response(),
        Err(error) => skill_error(error, anthropic),
    }
}

#[utoipa::path(
    get,
    tag = "Mistral.rs",
    path = "/v1/skills/{skill_id}/versions",
    responses((status = 200, description = "Skill versions", body = AnthropicSkillVersionListObject))
)]
pub async fn list_skill_versions(
    AxumPath(skill_id): AxumPath<String>,
    headers: HeaderMap,
    RawQuery(raw_query): RawQuery,
    Extension(store): Extension<Arc<SkillStore>>,
) -> axum::response::Response {
    let anthropic = prefers_anthropic_shape(&headers, None, raw_query.as_deref());
    match store.list_versions(&skill_id) {
        Ok(versions) => Json(AnthropicSkillVersionListObject {
            data: versions
                .iter()
                .map(AnthropicSkillVersionObject::from)
                .collect(),
            has_more: false,
            next_page: None,
        })
        .into_response(),
        Err(error) => skill_error(error, anthropic),
    }
}

#[cfg(test)]
mod tests {
    use crate::skill_store::SKILL_OBJECT;
    use axum::{
        body::Body,
        extract::FromRequest,
        http::{header, Request, Uri},
    };
    use http_body_util::BodyExt;
    use serde_json::Value;

    use super::*;

    fn test_store() -> (tempfile::TempDir, Arc<SkillStore>) {
        let root = tempfile::tempdir().unwrap();
        let store = Arc::new(SkillStore::new(root.path().to_path_buf()).unwrap());
        (root, store)
    }

    fn anthropic_headers() -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert("anthropic-version", "2023-06-01".parse().unwrap());
        headers
    }

    async fn response_json(response: axum::response::Response) -> Value {
        let body = response.into_body().collect().await.unwrap().to_bytes();
        serde_json::from_slice(&body).unwrap()
    }

    async fn invalid_boundary_multipart() -> std::result::Result<Multipart, MultipartRejection> {
        let request = Request::builder()
            .header(header::CONTENT_TYPE, "multipart/form-data")
            .body(Body::empty())
            .unwrap();
        Multipart::from_request(request, &()).await
    }

    async fn multipart_with_body(
        body: &'static str,
    ) -> std::result::Result<Multipart, MultipartRejection> {
        let request = Request::builder()
            .header(
                header::CONTENT_TYPE,
                "multipart/form-data; boundary=skill-test",
            )
            .body(Body::from(body))
            .unwrap();
        Multipart::from_request(request, &()).await
    }

    fn test_skill() -> SkillObject {
        SkillObject {
            id: "skill_abc".to_string(),
            object: SKILL_OBJECT,
            created_at: 1_700_000_000,
            name: "invoice-auditor".to_string(),
            description: "Checks invoices.".to_string(),
            latest_version: 2,
        }
    }

    #[test]
    fn anthropic_list_shape_filters_custom_skills() {
        let query = SkillListQuery {
            source: Some(CUSTOM_SKILL_SOURCE.to_string()),
            limit: None,
            page: None,
        };
        let response = anthropic_list_response(vec![test_skill()], Some(&query));

        assert!(!response.has_more);
        assert!(response.next_page.is_none());
        assert_eq!(response.data.len(), 1);
        assert_eq!(response.data[0].id, "skill_abc");
        assert_eq!(response.data[0].tp, SKILL_OBJECT);
        assert_eq!(response.data[0].display_title, "invoice-auditor");
        assert_eq!(response.data[0].latest_version, "2");
        assert_eq!(response.data[0].source, CUSTOM_SKILL_SOURCE);
    }

    #[test]
    fn anthropic_list_shape_returns_empty_anthropic_source() {
        let query = SkillListQuery {
            source: Some(ANTHROPIC_SKILL_SOURCE.to_string()),
            limit: None,
            page: None,
        };
        let response = anthropic_list_response(vec![test_skill()], Some(&query));

        assert!(response.data.is_empty());
    }

    #[tokio::test]
    async fn multipart_rejections_use_protocol_error_envelopes() {
        let (_root, store) = test_store();
        let response = upload_skill(
            HeaderMap::new(),
            RawQuery(None),
            Extension(store.clone()),
            invalid_boundary_multipart().await,
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = response_json(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["code"], "invalid_skill_upload");

        let response = upload_skill(
            anthropic_headers(),
            RawQuery(None),
            Extension(store),
            invalid_boundary_multipart().await,
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = response_json(response).await;
        assert_eq!(body["type"], "error");
        assert_eq!(body["error"]["type"], "invalid_request_error");
    }

    #[tokio::test]
    async fn upload_validation_is_a_bad_request() {
        const BODY: &str = "--skill-test\r\nContent-Disposition: form-data; name=\"ignored\"\r\n\r\nvalue\r\n--skill-test--\r\n";

        let (_root, store) = test_store();
        let response = upload_skill(
            HeaderMap::new(),
            RawQuery(None),
            Extension(store),
            multipart_with_body(BODY).await,
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = response_json(response).await;
        assert_eq!(body["error"]["code"], "invalid_skill_upload");
    }

    #[tokio::test]
    async fn query_rejection_uses_raw_source_to_select_anthropic_shape() {
        let (_root, store) = test_store();
        let raw_query = "source=custom&limit=invalid";
        let uri: Uri = format!("/v1/skills?{raw_query}").parse().unwrap();
        let payload = Query::<SkillListQuery>::try_from_uri(&uri);
        assert!(payload.is_err());

        let response = list_skills(
            HeaderMap::new(),
            RawQuery(Some(raw_query.to_string())),
            payload,
            Extension(store),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = response_json(response).await;
        assert_eq!(body["type"], "error");
        assert_eq!(body["error"]["type"], "invalid_request_error");
    }

    #[tokio::test]
    async fn unsupported_source_is_an_invalid_anthropic_query() {
        let (_root, store) = test_store();
        let raw_query = "source=unknown";
        let uri: Uri = format!("/v1/skills?{raw_query}").parse().unwrap();
        let payload = Query::<SkillListQuery>::try_from_uri(&uri);

        let response = list_skills(
            HeaderMap::new(),
            RawQuery(Some(raw_query.to_string())),
            payload,
            Extension(store),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = response_json(response).await;
        assert_eq!(body["type"], "error");
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("unknown"));
    }

    #[tokio::test]
    async fn anthropic_conflicts_preserve_the_conflict() {
        let response = anthropic_skill_error_response(ApiError::new(
            ApiErrorKind::Conflict,
            "private conflict detail",
            None,
            None,
        ));
        assert_eq!(response.status(), StatusCode::CONFLICT);
        let body = response_json(response).await;
        assert_eq!(body["error"]["type"], "conflict_error");
        assert_eq!(body["error"]["message"], "private conflict detail");
    }

    #[tokio::test]
    async fn missing_skill_is_not_found_but_store_failure_is_internal() {
        let (_root, store) = test_store();
        let response = list_skill_versions(
            AxumPath("skill_missing".to_string()),
            HeaderMap::new(),
            RawQuery(None),
            Extension(store),
        )
        .await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let body = response_json(response).await;
        assert_eq!(body["error"]["code"], "skill_not_found");

        let response = skill_error(anyhow::anyhow!("skill store lock poisoned"), false);
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body = response_json(response).await;
        assert_eq!(body["error"]["message"], "Internal server error.");
        assert!(!body.to_string().contains("poison"));
    }

    #[test]
    fn upload_size_errors_keep_their_status() {
        let response = skill_error(skill_upload_too_large("Skill upload is too large."), true);
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }
}
