//! The file store: files uploaded for requests to read and files agentic tools produce.

use std::sync::Arc;

use base64::{Engine, engine::general_purpose::STANDARD};
pub use inference_core::{FILE_PURPOSE_GENERATED_IMAGE, FILE_PURPOSE_USER_DATA, File};
use inference_core::{FileContent, FileSource, InferenceRs, InferenceRsError};
use serde::Serialize;
use utoipa::ToSchema;

use crate::{
    api_error::{ApiError, ApiErrorKind},
    types::SharedInferenceRsState,
};

pub const MAX_FILE_UPLOAD_BYTES: usize = 64 * 1024 * 1024;
/// Where a file's body is served; generated images' `url`s point here, relative to the inference router.
pub const FILE_CONTENT_PATH: &str = "/v1/files/{id}/content";
const FILE_ID_PARAM: &str = "{id}";
const GENERATED_IMAGE_SOURCE_TOOL: &str = "image_generation";
const PNG_MIME_TYPE: &str = "image/png";
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

#[derive(Serialize, ToSchema)]
pub struct FileListObject {
    pub object: &'static str,
    pub data: Vec<FileMetadata>,
}

#[derive(Serialize, ToSchema)]
pub struct ContainerFileListObject {
    pub object: &'static str,
    pub data: Vec<ContainerFileMetadata>,
}

#[derive(Serialize, ToSchema)]
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

fn find(
    state: &SharedInferenceRsState,
    id: &str,
    owner: Option<&str>,
) -> Result<Arc<File>, ApiError> {
    state
        .try_find_file(id, owner)
        .map_err(|error| store_error(state, error))?
        .ok_or_else(|| not_found(id))
}

fn source(file: &File) -> SourceMeta {
    SourceMeta {
        tool: file.source.tool.clone(),
        round: file.source.round,
        turn: file.source.turn,
    }
}

fn mime_type(file: &File) -> String {
    file.mime_type
        .clone()
        .unwrap_or_else(|| DEFAULT_MIME_TYPE.to_string())
}

fn metadata(file: &File) -> FileMetadata {
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

fn container_metadata(container_id: &str, file: &File) -> ContainerFileMetadata {
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
pub(crate) fn upload_file(
    state: &SharedInferenceRsState,
    upload: FileUpload,
    owner: Option<&str>,
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
    let file = File::from_bytes(
        File::make_upload_id(),
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
        .insert_file(None, file.clone(), None, owner)
        .map_err(|error| store_error(state, error))?;
    Ok(metadata(&file))
}

/// Stores a generated PNG in `model`'s file store and returns the url that serves it.
pub(crate) fn store_generated_image(
    state: &SharedInferenceRsState,
    model: Option<&str>,
    png: Vec<u8>,
    owner: Option<&str>,
) -> Result<String, ApiError> {
    let id = File::make_upload_id();
    let file = File::from_bytes(
        id.clone(),
        format!("{id}.png"),
        Some(PNG_MIME_TYPE.to_string()),
        FILE_PURPOSE_GENERATED_IMAGE.to_string(),
        FileSource {
            tool: GENERATED_IMAGE_SOURCE_TOOL.to_string(),
            round: 0,
            turn: 0,
        },
        png,
    );
    state
        .insert_file(model, file, None, owner)
        .map_err(|error| store_error(state, error))?;
    Ok(FILE_CONTENT_PATH.replace(FILE_ID_PARAM, &id))
}

pub(crate) fn get_file(
    state: &SharedInferenceRsState,
    id: &str,
    owner: Option<&str>,
) -> Result<FileMetadata, ApiError> {
    find(state, id, owner).map(|file| metadata(&file))
}

pub(crate) fn list_files(
    state: &SharedInferenceRsState,
    owner: Option<&str>,
) -> Result<FileListObject, ApiError> {
    let files = state
        .try_list_files(owner)
        .map_err(|error| store_error(state, error))?;
    Ok(FileListObject {
        object: LIST_OBJECT,
        data: files.iter().map(|file| metadata(file)).collect(),
    })
}

pub(crate) fn delete_file(
    state: &SharedInferenceRsState,
    id: &str,
    owner: Option<&str>,
) -> Result<FileDeleted, ApiError> {
    if !state
        .try_remove_file(id, owner)
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
pub(crate) fn file_content(
    state: &SharedInferenceRsState,
    id: &str,
    owner: Option<&str>,
) -> Result<FileBody, ApiError> {
    let file = find(state, id, owner)?;
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

pub(crate) fn list_container_files(
    state: &SharedInferenceRsState,
    container_id: &str,
    owner: Option<&str>,
) -> Result<ContainerFileListObject, ApiError> {
    let files = state
        .try_list_tagged_files(container_id, owner)
        .map_err(|error| store_error(state, error))?;
    Ok(ContainerFileListObject {
        object: LIST_OBJECT,
        data: files
            .iter()
            .map(|file| container_metadata(container_id, file))
            .collect(),
    })
}

pub(crate) fn get_container_file(
    state: &SharedInferenceRsState,
    container_id: &str,
    file_id: &str,
    owner: Option<&str>,
) -> Result<ContainerFileMetadata, ApiError> {
    find_in_container(state, container_id, file_id, owner)
        .map(|file| container_metadata(container_id, &file))
}

/// A container file's body; a file the container's response did not cite is not found there.
pub(crate) fn container_file_content(
    state: &SharedInferenceRsState,
    container_id: &str,
    file_id: &str,
    owner: Option<&str>,
) -> Result<FileBody, ApiError> {
    find_in_container(state, container_id, file_id, owner)?;
    file_content(state, file_id, owner)
}

fn find_in_container(
    state: &SharedInferenceRsState,
    container_id: &str,
    file_id: &str,
    owner: Option<&str>,
) -> Result<Arc<File>, ApiError> {
    state
        .try_list_tagged_files(container_id, owner)
        .map_err(|error| store_error(state, error))?
        .into_iter()
        .find(|file| file.id == file_id)
        .ok_or_else(|| not_found(file_id))
}
