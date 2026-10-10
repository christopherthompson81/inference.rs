//! The transcription route: a multipart upload over the engine's speech recognition.

use axum::{
    extract::{
        Multipart,
        multipart::{MultipartError, MultipartRejection},
    },
    http::{HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
};
use inference_api::api_error::{ApiError, ApiErrorKind};
use serde_json::{Map, Value};

use crate::{
    handler_core::openai_error_response, openai::TranscriptionRequest, types::OwnedEngine,
};

const FILE_FIELD: &str = "file";
// OpenAI's form repeats the array field, with or without brackets
const GRANULARITY_FIELDS: [&str; 2] = ["timestamp_granularities[]", "timestamp_granularities"];
const TEMPERATURE_FIELD: &str = "temperature";
const STREAM_FIELD: &str = "stream";
const INVALID_CODE: &str = "invalid_transcription_request";

/// The multipart form `/v1/audio/transcriptions` takes.
#[cfg(test)]
#[derive(utoipa::ToSchema)]
#[allow(dead_code)]
pub struct TranscriptionForm {
    /// The audio to transcribe: WAV, MP3, FLAC, OGG and the other formats symphonia decodes.
    #[schema(value_type = String, format = Binary)]
    file: Vec<u8>,
    #[schema(inline)]
    request: TranscriptionRequest,
}

fn invalid(message: impl Into<String>, param: &str) -> ApiError {
    ApiError::new(
        ApiErrorKind::InvalidRequest,
        message,
        Some(INVALID_CODE),
        Some(param),
    )
}

fn multipart_error(error: MultipartError, field: &str) -> ApiError {
    let kind = if error.status() == StatusCode::PAYLOAD_TOO_LARGE {
        ApiErrorKind::PayloadTooLarge
    } else {
        ApiErrorKind::InvalidRequest
    };
    ApiError::new(kind, error.body_text(), Some(INVALID_CODE), Some(field))
}

// the form's text fields become the JSON a `TranscriptionRequest` deserializes from, so they validate as JSON does
async fn read_form(mut multipart: Multipart) -> Result<(TranscriptionRequest, Vec<u8>), ApiError> {
    let mut fields = Map::new();
    let mut granularities = Vec::new();
    let mut audio = None;
    let next = |e| multipart_error(e, "body");
    while let Some(field) = multipart.next_field().await.map_err(next)? {
        let name = field.name().unwrap_or_default().to_string();
        if name == FILE_FIELD {
            if audio.is_some() {
                return Err(invalid("The form has more than one `file`.", FILE_FIELD));
            }
            let bytes = field
                .bytes()
                .await
                .map_err(|e| multipart_error(e, FILE_FIELD))?;
            audio = Some(bytes.to_vec());
            continue;
        }
        let text = field.text().await.map_err(|e| multipart_error(e, &name))?;
        if name == STREAM_FIELD {
            if text != "false" {
                let message = "Streamed transcription is not supported; omit `stream`.";
                return Err(invalid(message, STREAM_FIELD));
            }
            continue;
        }
        if GRANULARITY_FIELDS.contains(&name.as_str()) {
            granularities.push(Value::String(text));
        } else if name == TEMPERATURE_FIELD {
            let temperature: f64 = text
                .parse()
                .map_err(|_| invalid(format!("`{text}` is not a number"), TEMPERATURE_FIELD))?;
            fields.insert(name, Value::from(temperature));
        } else {
            fields.insert(name, Value::String(text));
        }
    }
    if !granularities.is_empty() {
        fields.insert(
            GRANULARITY_FIELDS[1].to_string(),
            Value::Array(granularities),
        );
    }
    let audio =
        audio.ok_or_else(|| invalid("The form has no `file` to transcribe.", FILE_FIELD))?;
    let request = serde_json::from_value(Value::Object(fields))
        .map_err(|e| invalid(e.to_string(), "body"))?;
    Ok((request, audio))
}

/// Transcription endpoint handler.
#[cfg_attr(test, utoipa::path(
    post,
    tag = "inference.rs",
    path = "/v1/audio/transcriptions",
    request_body(content = inline(TranscriptionForm), content_type = "multipart/form-data"),
    responses((
        status = 200,
        description = "The transcript, in the requested format",
        body = crate::openai::TranscriptionResponse
    ))
))]
pub async fn transcription(
    OwnedEngine(engine): OwnedEngine,
    payload: Result<Multipart, MultipartRejection>,
) -> Response {
    let multipart = match payload {
        Ok(multipart) => multipart,
        Err(rejection) => return openai_error_response(invalid(rejection.body_text(), "body")),
    };
    let (request, audio) = match read_form(multipart).await {
        Ok(form) => form,
        Err(error) => return openai_error_response(error),
    };
    match engine.transcription(request, &audio).await {
        Ok(output) => (
            StatusCode::OK,
            [(
                header::CONTENT_TYPE,
                HeaderValue::from_static(output.content_type),
            )],
            output.body,
        )
            .into_response(),
        Err(error) => openai_error_response(error),
    }
}
