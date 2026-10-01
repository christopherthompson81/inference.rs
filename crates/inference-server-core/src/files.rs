//! OpenAI-compatible Files routes: HTTP framing over the engine's file store.

use axum::{
    Extension,
    extract::{Multipart, Path, State, multipart::MultipartRejection},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};
use inference_core::FILE_PURPOSE_USER_DATA;

pub use crate::files_api::{
    ContainerFileListObject, ContainerFileMetadata, FileDeleted, FileListObject, FileMetadata,
    SourceMeta,
};
use crate::{
    auth::Owner,
    files_api::{self, FileUpload},
    handler_core::{ApiError, ApiErrorHttp, ApiErrorKind, json_response, openai_error_response},
    types::ExtractedInferenceRsState,
};

/// Whether an open server's `GET /v1/files` lists its shared store; a keyed one always lists each owner's own.
#[derive(Clone, Copy, Debug, Default)]
pub struct FileListing(pub bool);

const FILE_LISTING_DISABLED: &str = "Listing files is off on this server: the store is shared by every client. \
    Fetch a file by the id it was given, list a Responses container's files, or enable listing \
    (`--allow-file-listing`, or `with_file_listing(true)` on the router builder).";
const FILE_LISTING_DISABLED_CODE: &str = "file_listing_disabled";

#[cfg_attr(test, utoipa::path(
    post,
    tag = "inference.rs",
    path = "/v1/files",
    request_body(content_type = "multipart/form-data"),
    responses(
        (status = 200, description = "Uploaded file metadata", body = FileMetadata),
        (status = 400, description = "Invalid upload"),
    )
))]
pub async fn upload_file(
    State(state): ExtractedInferenceRsState,
    Extension(owner): Extension<Owner>,
    payload: Result<Multipart, MultipartRejection>,
) -> Response {
    let multipart = match payload {
        Ok(multipart) => multipart,
        Err(error) => {
            return openai_error_response(ApiError::from_status(error.status(), error.body_text()));
        }
    };
    match parse_upload(multipart).await {
        Ok(upload) => json_response(files_api::upload_file(&state, upload, owner.as_deref())),
        Err(error) => openai_error_response(error),
    }
}

async fn parse_upload(mut multipart: Multipart) -> Result<FileUpload, ApiError> {
    let mut purpose = None;
    let mut file = None;

    while let Some(field) = multipart.next_field().await.map_err(multipart_error)? {
        let field_name = field.name().unwrap_or_default().to_string();
        match field_name.as_str() {
            "purpose" => {
                let value = field.text().await.map_err(multipart_error)?;
                if !value.trim().is_empty() {
                    purpose = Some(value);
                }
            }
            "file" => {
                let filename = field
                    .file_name()
                    .ok_or_else(|| {
                        ApiError::new(
                            ApiErrorKind::InvalidRequest,
                            "Uploaded file is missing a filename.",
                            Some("invalid_file"),
                            Some("file"),
                        )
                    })?
                    .to_string();
                let mime_type = field.content_type().map(ToString::to_string);
                let bytes = field.bytes().await.map_err(multipart_error)?.to_vec();
                file = Some((filename, mime_type, bytes));
            }
            _ => {}
        }
    }

    let purpose = purpose.ok_or_else(|| {
        ApiError::new(
            ApiErrorKind::InvalidRequest,
            format!(
                "File upload requires multipart field `purpose` such as `{}`.",
                FILE_PURPOSE_USER_DATA
            ),
            Some("missing_required_parameter"),
            Some("purpose"),
        )
    })?;
    let (filename, mime_type, bytes) = file.ok_or_else(|| {
        ApiError::new(
            ApiErrorKind::InvalidRequest,
            "File upload requires multipart field `file`.",
            Some("missing_required_parameter"),
            Some("file"),
        )
    })?;

    Ok(FileUpload {
        filename,
        mime_type,
        purpose,
        bytes,
    })
}

fn multipart_error(error: axum::extract::multipart::MultipartError) -> ApiError {
    ApiError::from_status(error.status(), error.body_text())
}

#[cfg_attr(test, utoipa::path(
    get,
    tag = "inference.rs",
    path = "/v1/files/{id}",
    params(("id" = String, Path, description = "File ID")),
    responses(
        (status = 200, description = "File metadata", body = FileMetadata),
        (status = 404, description = "File not found or expired"),
        (status = 500, description = "Internal server error"),
    )
))]
pub async fn get_file(
    State(state): ExtractedInferenceRsState,
    Extension(owner): Extension<Owner>,
    Path(id): Path<String>,
) -> Response {
    json_response(files_api::get_file(&state, &id, owner.as_deref()))
}

#[cfg_attr(test, utoipa::path(
    get,
    tag = "inference.rs",
    path = "/v1/files/{id}/content",
    params(("id" = String, Path, description = "File ID")),
    responses(
        (status = 200, description = "Raw file bytes with the file's MIME type"),
        (status = 404, description = "File not found or expired"),
        (status = 410, description = "File body was elided and is no longer fetchable"),
        (status = 500, description = "Internal server error"),
    )
))]
pub async fn get_file_content(
    State(state): ExtractedInferenceRsState,
    Extension(owner): Extension<Owner>,
    Path(id): Path<String>,
) -> Response {
    serve_bytes(files_api::file_content(&state, &id, owner.as_deref()))
}

#[cfg_attr(test, utoipa::path(
    get,
    tag = "inference.rs",
    path = "/v1/files",
    responses(
        (status = 200, description = "List of file metadata", body = FileListObject),
        (status = 403, description = "File listing is off on this server"),
        (status = 500, description = "Internal server error"),
    )
))]
pub async fn list_files(
    State(state): ExtractedInferenceRsState,
    Extension(owner): Extension<Owner>,
    listing: Option<Extension<FileListing>>,
) -> Response {
    let allowed = owner.0.is_some() || matches!(listing, Some(Extension(FileListing(true))));
    match allowed {
        true => json_response(files_api::list_files(&state, owner.as_deref())),
        false => openai_error_response(ApiError::new(
            ApiErrorKind::Forbidden,
            FILE_LISTING_DISABLED,
            Some(FILE_LISTING_DISABLED_CODE),
            None,
        )),
    }
}

#[cfg_attr(test, utoipa::path(
    delete,
    tag = "inference.rs",
    path = "/v1/files/{id}",
    params(("id" = String, Path, description = "File ID")),
    responses(
        (status = 200, description = "File deleted", body = FileDeleted),
        (status = 404, description = "File not found or expired"),
        (status = 500, description = "Internal server error"),
    )
))]
pub async fn delete_file(
    State(state): ExtractedInferenceRsState,
    Extension(owner): Extension<Owner>,
    Path(id): Path<String>,
) -> Response {
    json_response(files_api::delete_file(&state, &id, owner.as_deref()))
}

#[cfg_attr(test, utoipa::path(
    get,
    tag = "inference.rs",
    path = "/v1/containers/{container_id}/files",
    params(("container_id" = String, Path, description = "Container ID")),
    responses(
        (status = 200, description = "List of container file metadata", body = ContainerFileListObject),
        (status = 500, description = "Internal server error"),
    )
))]
pub async fn list_container_files(
    State(state): ExtractedInferenceRsState,
    Extension(owner): Extension<Owner>,
    Path(container_id): Path<String>,
) -> Response {
    json_response(files_api::list_container_files(
        &state,
        &container_id,
        owner.as_deref(),
    ))
}

#[cfg_attr(test, utoipa::path(
    get,
    tag = "inference.rs",
    path = "/v1/containers/{container_id}/files/{file_id}",
    params(
        ("container_id" = String, Path, description = "Container ID"),
        ("file_id" = String, Path, description = "File ID")
    ),
    responses(
        (status = 200, description = "Container file metadata", body = ContainerFileMetadata),
        (status = 404, description = "File not found or expired"),
        (status = 500, description = "Internal server error"),
    )
))]
pub async fn get_container_file(
    State(state): ExtractedInferenceRsState,
    Extension(owner): Extension<Owner>,
    Path((container_id, file_id)): Path<(String, String)>,
) -> Response {
    json_response(files_api::get_container_file(
        &state,
        &container_id,
        &file_id,
        owner.as_deref(),
    ))
}

#[cfg_attr(test, utoipa::path(
    get,
    tag = "inference.rs",
    path = "/v1/containers/{container_id}/files/{file_id}/content",
    params(
        ("container_id" = String, Path, description = "Container ID"),
        ("file_id" = String, Path, description = "File ID")
    ),
    responses(
        (status = 200, description = "Raw file bytes with the file's MIME type"),
        (status = 404, description = "File not found or expired"),
        (status = 410, description = "File body was elided and is no longer fetchable"),
        (status = 500, description = "Internal server error"),
    )
))]
pub async fn get_container_file_content(
    State(state): ExtractedInferenceRsState,
    Extension(owner): Extension<Owner>,
    Path((container_id, file_id)): Path<(String, String)>,
) -> Response {
    serve_bytes(files_api::container_file_content(
        &state,
        &container_id,
        &file_id,
        owner.as_deref(),
    ))
}

fn serve_bytes(body: Result<files_api::FileBody, ApiError>) -> Response {
    let body = match body {
        Ok(body) => body,
        Err(error) => return openai_error_response(error),
    };
    let disposition = format!(
        "inline; filename=\"{}\"; filename*=UTF-8''{}",
        ascii_safe_filename(&body.filename),
        percent_encode_filename(&body.filename),
    );
    (
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, body.mime_type),
            (header::CONTENT_LENGTH, body.bytes.len().to_string()),
            (header::CONTENT_DISPOSITION, disposition),
        ],
        body.bytes,
    )
        .into_response()
}

fn ascii_safe_filename(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_graphic() && c != '"' && c != '\\' {
                c
            } else if c == ' ' {
                ' '
            } else {
                '_'
            }
        })
        .collect();
    if cleaned.is_empty() {
        "file".to_string()
    } else {
        cleaned
    }
}

/// RFC 5987 attr-char set.
fn percent_encode_filename(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for &b in name.as_bytes() {
        let safe = b.is_ascii_alphanumeric()
            || matches!(
                b,
                b'!' | b'#' | b'$' | b'&' | b'+' | b'-' | b'.' | b'^' | b'_' | b'`' | b'|' | b'~'
            );
        if safe {
            out.push(b as char);
        } else {
            use std::fmt::Write;
            let _ = write!(out, "%{:02X}", b);
        }
    }
    out
}
