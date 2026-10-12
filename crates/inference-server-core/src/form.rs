//! Multipart forms, as OpenAI's upload endpoints take them: text fields become the request's JSON, files are kept.

use axum::{
    extract::{
        Multipart,
        multipart::{MultipartError, MultipartRejection},
    },
    http::StatusCode,
};
use inference_api::api_error::{ApiError, ApiErrorKind};
use serde_json::{Map, Value};

const STREAM_FIELD: &str = "stream";
const LIST_SUFFIX: &str = "[]";

/// How a text field converts before the request deserializes; unlisted fields stay strings.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Field {
    Number,
    Integer,
    Flag,
    // OpenAI's form repeats an array field, with or without brackets
    List,
}

/// One uploaded file, by the field it came in.
pub(crate) struct Upload {
    pub field: String,
    pub bytes: Vec<u8>,
    pub content_type: Option<String>,
}

pub(crate) fn invalid(message: impl Into<String>, param: &str, code: &str) -> ApiError {
    ApiError::new(
        ApiErrorKind::InvalidRequest,
        message,
        Some(code),
        Some(param),
    )
}

fn multipart_error(error: MultipartError, field: &str, code: &str) -> ApiError {
    let kind = if error.status() == StatusCode::PAYLOAD_TOO_LARGE {
        ApiErrorKind::PayloadTooLarge
    } else {
        ApiErrorKind::InvalidRequest
    };
    ApiError::new(kind, error.body_text(), Some(code), Some(field))
}

pub(crate) fn multipart_payload(
    payload: Result<Multipart, MultipartRejection>,
    code: &str,
) -> Result<Multipart, ApiError> {
    payload.map_err(|rejection| invalid(rejection.body_text(), "body", code))
}

// the form's text fields become the JSON a request deserializes from, so they validate as JSON does; `kinds` names
// the fields that are not strings, `files` the ones whose bytes are kept, in the order they came
pub(crate) async fn read_form<T: serde::de::DeserializeOwned>(
    mut multipart: Multipart,
    kinds: &[(&str, Field)],
    files: &[&str],
    code: &str,
) -> Result<(T, Vec<Upload>), ApiError> {
    let mut fields = Map::new();
    let mut uploads = Vec::new();
    let next = |e| multipart_error(e, "body", code);
    while let Some(field) = multipart.next_field().await.map_err(next)? {
        let raw = field.name().unwrap_or_default().to_string();
        let name = raw.trim_end_matches(LIST_SUFFIX).to_string();
        if files.contains(&name.as_str()) {
            let content_type = field.content_type().map(str::to_string);
            let bytes = field
                .bytes()
                .await
                .map_err(|e| multipart_error(e, &name, code))?;
            uploads.push(Upload {
                field: name,
                bytes: bytes.to_vec(),
                content_type,
            });
            continue;
        }
        let text = field
            .text()
            .await
            .map_err(|e| multipart_error(e, &raw, code))?;
        if raw == STREAM_FIELD {
            if text != "false" {
                let message = "Streamed results are not supported; omit `stream`.";
                return Err(invalid(message, STREAM_FIELD, code));
            }
            continue;
        }
        let kind = kinds.iter().find(|(n, _)| *n == name).map(|(_, k)| *k);
        let value = match kind {
            Some(Field::Number) => Value::from(
                text.parse::<f64>()
                    .map_err(|_| invalid(format!("`{text}` is not a number"), &name, code))?,
            ),
            Some(Field::Integer) => Value::from(
                text.parse::<u64>()
                    .map_err(|_| invalid(format!("`{text}` is not a whole number"), &name, code))?,
            ),
            Some(Field::Flag) => Value::Bool(
                text.parse::<bool>()
                    .map_err(|_| invalid(format!("`{text}` is not true or false"), &name, code))?,
            ),
            Some(Field::List) | None => Value::String(text),
        };
        if kind == Some(Field::List) {
            let list = fields
                .entry(name)
                .or_insert_with(|| Value::Array(Vec::new()));
            if let Value::Array(items) = list {
                items.push(value);
            }
        } else {
            fields.insert(name, value);
        }
    }
    let request = serde_json::from_value(Value::Object(fields))
        .map_err(|e| invalid(e.to_string(), "body", code))?;
    Ok((request, uploads))
}
