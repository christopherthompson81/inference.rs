//! The file store: files uploaded for requests to read and files agentic tools produce.

use std::sync::Arc;

use base64::{engine::general_purpose::STANDARD, Engine};
use inference_core::{
    File as CoreFile, FileContent, FileSource, InferenceRs, InferenceRsError,
    FILE_PURPOSE_USER_DATA,
};
use serde::Serialize;
use utoipa::ToSchema;

use crate::{
    api_error::{ApiError, ApiErrorKind},
    types::SharedInferenceRsState,
};

pub const MAX_FILE_UPLOAD_BYTES: usize = 64 * 1024 * 1024;
const DEFAULT_MIME_TYPE: &str = "application/octet-stream";
const UPLOAD_SOURCE_TOOL: &str = "user_upload";
const FILE_OBJECT: &str = "file";
const CONTAINER_FILE_OBJECT: &str = "container.file";
const LIST_OBJECT: &str = "list";

/// A file to add to the store.
pub struct FileUpload {
    pub filename: String,
    pub mime_type: Option<String>,
    pub purpose: String,
    pub bytes: Vec<u8>,
}

/// OpenAI file metadata + inference.rs extensions (`format`, `mime_type`, `source`, `truncated`).
#[derive(Serialize, ToSchema)]
pub struct FileMetadata {
    pub id: String,
    pub object: &'static str,
    pub bytes: u64,
    pub created_at: u64,
    pub filename: String,
    pub purpose: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    pub mime_type: String,
    pub source: SourceMeta,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub truncated: bool,
}

/// OpenAI-compatible container file metadata backed by the same in-process file store.
#[derive(Serialize, ToSchema)]
pub struct ContainerFileMetadata {
    pub id: String,
    pub object: &'static str,
    pub bytes: u64,
    pub created_at: u64,
    pub filename: String,
    pub container_id: String,
    pub source: SourceMeta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    pub mime_type: String,
}

/// Which agentic tool produced the file, and when in the session.
#[derive(Serialize, ToSchema)]
pub struct SourceMeta {
    pub tool: String,
    pub round: usize,
    pub turn: usize,
}

#[derive(Serialize)]
pub struct FileList<T> {
    pub object: &'static str,
    pub data: Vec<T>,
}

#[derive(Serialize)]
pub struct FileDeleted {
    pub id: String,
    pub object: &'static str,
    pub deleted: bool,
}

/// A file's body with the MIME type and name to serve it under.
pub struct FileBody {
    pub bytes: Vec<u8>,
    pub mime_type: String,
    pub filename: String,
}

fn store_error(state: &SharedInferenceRsState, error: InferenceRsError) -> ApiError {
    InferenceRs::maybe_log_error(state.clone(), &error);
    ApiError::from_error(&error, ApiErrorKind::Internal)
}

pub fn file_too_large() -> ApiError {
    ApiError::new(
        ApiErrorKind::PayloadTooLarge,
        format!("File upload exceeds the {MAX_FILE_UPLOAD_BYTES} byte limit."),
        Some("file_too_large"),
        Some("file"),
    )
}

fn not_found(id: &str) -> ApiError {
    ApiError::new(
        ApiErrorKind::NotFound,
        format!("File '{id}' not found or expired."),
        Some("file_not_found"),
        Some("file_id"),
    )
}

fn content_gone(message: &str) -> ApiError {
    ApiError::new(
        ApiErrorKind::Gone,
        message,
        Some("file_content_unavailable"),
        Some("file_id"),
    )
}

fn find(state: &SharedInferenceRsState, id: &str) -> Result<Arc<CoreFile>, ApiError> {
    state
        .try_find_file(id)
        .map_err(|error| store_error(state, error))?
        .ok_or_else(|| not_found(id))
}

fn source(file: &CoreFile) -> SourceMeta {
    SourceMeta {
        tool: file.source.tool.clone(),
        round: file.source.round,
        turn: file.source.turn,
    }
}

fn mime_type(file: &CoreFile) -> String {
    file.mime_type
        .clone()
        .unwrap_or_else(|| DEFAULT_MIME_TYPE.to_string())
}

fn metadata(file: &CoreFile) -> FileMetadata {
    FileMetadata {
        id: file.id.clone(),
        object: FILE_OBJECT,
        bytes: file.bytes,
        created_at: file.created_at,
        filename: file.name.clone(),
        purpose: file.purpose.clone(),
        format: file.format.clone(),
        mime_type: mime_type(file),
        source: source(file),
        truncated: file.is_truncated(),
    }
}

fn container_metadata(container_id: &str, file: &CoreFile) -> ContainerFileMetadata {
    ContainerFileMetadata {
        id: file.id.clone(),
        object: CONTAINER_FILE_OBJECT,
        bytes: file.bytes,
        created_at: file.created_at,
        filename: file.name.clone(),
        container_id: container_id.to_string(),
        source: source(file),
        format: file.format.clone(),
        mime_type: mime_type(file),
    }
}

/// Stores an uploaded file; requests then name it by the returned id.
pub fn upload_file(
    state: &SharedInferenceRsState,
    upload: FileUpload,
) -> Result<FileMetadata, ApiError> {
    if upload.purpose.trim().is_empty() {
        return Err(ApiError::new(
            ApiErrorKind::InvalidRequest,
            format!("File upload requires a `purpose` such as `{FILE_PURPOSE_USER_DATA}`."),
            Some("missing_required_parameter"),
            Some("purpose"),
        ));
    }
    if upload.bytes.len() > MAX_FILE_UPLOAD_BYTES {
        return Err(file_too_large());
    }
    let file = CoreFile::from_bytes(
        CoreFile::make_upload_id(),
        upload.filename,
        upload.mime_type,
        upload.purpose,
        FileSource {
            tool: UPLOAD_SOURCE_TOOL.to_string(),
            round: 0,
            turn: 0,
        },
        upload.bytes,
    );
    state
        .insert_file(None, file.clone(), None)
        .map_err(|error| store_error(state, error))?;
    Ok(metadata(&file))
}

pub fn get_file(state: &SharedInferenceRsState, id: &str) -> Result<FileMetadata, ApiError> {
    find(state, id).map(|file| metadata(&file))
}

pub fn list_files(state: &SharedInferenceRsState) -> Result<FileList<FileMetadata>, ApiError> {
    let files = state
        .try_list_files()
        .map_err(|error| store_error(state, error))?;
    Ok(FileList {
        object: LIST_OBJECT,
        data: files.iter().map(|file| metadata(file)).collect(),
    })
}

pub fn delete_file(state: &SharedInferenceRsState, id: &str) -> Result<FileDeleted, ApiError> {
    if !state
        .try_remove_file(id)
        .map_err(|error| store_error(state, error))?
    {
        return Err(not_found(id));
    }
    Ok(FileDeleted {
        id: id.to_string(),
        object: FILE_OBJECT,
        deleted: true,
    })
}

/// A file's body; a body the store elided to bound memory is Gone.
pub fn file_content(state: &SharedInferenceRsState, id: &str) -> Result<FileBody, ApiError> {
    let file = find(state, id)?;
    let bytes = match &file.content {
        FileContent::Text {
            text: Some(text), ..
        } => text.as_bytes().to_vec(),
        FileContent::Text { text: None, .. } => {
            return Err(content_gone(
                "Text body was elided and is no longer available.",
            ));
        }
        FileContent::Binary {
            data_base64: Some(data),
        } => STANDARD.decode(data).map_err(|error| {
            tracing::error!(%error, file_id = id, "failed to decode stored file content");
            ApiError::internal()
        })?,
        FileContent::Binary { data_base64: None } => {
            return Err(content_gone(
                "Binary body was elided and is no longer available.",
            ));
        }
        FileContent::Error { message, .. } => {
            return Err(ApiError::new(
                ApiErrorKind::InvalidRequest,
                message,
                Some("file_content_error"),
                Some("file_id"),
            ));
        }
    };
    Ok(FileBody {
        bytes,
        mime_type: mime_type(&file),
        filename: file.name.clone(),
    })
}

pub fn list_container_files(
    state: &SharedInferenceRsState,
    container_id: &str,
) -> Result<FileList<ContainerFileMetadata>, ApiError> {
    let files = state
        .try_list_files()
        .map_err(|error| store_error(state, error))?;
    Ok(FileList {
        object: LIST_OBJECT,
        data: files
            .iter()
            .map(|file| container_metadata(container_id, file))
            .collect(),
    })
}

pub fn get_container_file(
    state: &SharedInferenceRsState,
    container_id: &str,
    file_id: &str,
) -> Result<ContainerFileMetadata, ApiError> {
    find(state, file_id).map(|file| container_metadata(container_id, &file))
}
