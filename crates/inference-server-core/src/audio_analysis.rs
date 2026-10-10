//! The audio analysis routes: multipart uploads over the engine's speech recognition and voice activity detection.

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
    handler_core::openai_error_response,
    openai::{DiarizationRequest, TranscriptionRequest, VoiceActivityRequest},
    types::OwnedEngine,
};

const FILE_FIELD: &str = "file";
const STREAM_FIELD: &str = "stream";
const INVALID_CODE: &str = "invalid_audio_request";

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

#[derive(Clone, Copy, PartialEq)]
enum Field {
    Number,
    Flag,
    // OpenAI's form repeats an array field, with or without brackets
    List,
}

const TRANSCRIPTION_FIELDS: &[(&str, Field)] = &[
    ("temperature", Field::Number),
    ("timestamp_granularities", Field::List),
];
const VOICE_ACTIVITY_FIELDS: &[(&str, Field)] = &[
    ("threshold", Field::Number),
    ("neg_threshold", Field::Number),
    ("min_speech_duration_ms", Field::Number),
    ("max_speech_duration_s", Field::Number),
    ("min_silence_duration_ms", Field::Number),
    ("speech_pad_ms", Field::Number),
    ("return_probabilities", Field::Flag),
];
const DIARIZATION_FIELDS: &[(&str, Field)] = &[
    ("threshold", Field::Number),
    ("return_probabilities", Field::Flag),
];
const LIST_SUFFIX: &str = "[]";

// the form's text fields become the JSON a request deserializes from, so they validate as JSON does; `kinds` names
// the fields that are not strings
async fn read_form<T: serde::de::DeserializeOwned>(
    mut multipart: Multipart,
    kinds: &[(&str, Field)],
) -> Result<(T, Vec<u8>), ApiError> {
    let mut fields = Map::new();
    let mut audio = None;
    let next = |e| multipart_error(e, "body");
    while let Some(field) = multipart.next_field().await.map_err(next)? {
        let raw = field.name().unwrap_or_default().to_string();
        if raw == FILE_FIELD {
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
        let text = field.text().await.map_err(|e| multipart_error(e, &raw))?;
        if raw == STREAM_FIELD {
            if text != "false" {
                let message = "Streamed results are not supported; omit `stream`.";
                return Err(invalid(message, STREAM_FIELD));
            }
            continue;
        }
        let name = raw.trim_end_matches(LIST_SUFFIX).to_string();
        let kind = kinds.iter().find(|(n, _)| *n == name).map(|(_, k)| *k);
        let value = match kind {
            Some(Field::Number) => Value::from(
                text.parse::<f64>()
                    .map_err(|_| invalid(format!("`{text}` is not a number"), &name))?,
            ),
            Some(Field::Flag) => Value::Bool(
                text.parse::<bool>()
                    .map_err(|_| invalid(format!("`{text}` is not true or false"), &name))?,
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
    let audio = audio.ok_or_else(|| invalid("The form has no `file`.", FILE_FIELD))?;
    let request = serde_json::from_value(Value::Object(fields))
        .map_err(|e| invalid(e.to_string(), "body"))?;
    Ok((request, audio))
}

fn multipart_payload(
    payload: Result<Multipart, MultipartRejection>,
) -> Result<Multipart, ApiError> {
    payload.map_err(|rejection| invalid(rejection.body_text(), "body"))
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
    let multipart = match multipart_payload(payload) {
        Ok(multipart) => multipart,
        Err(error) => return openai_error_response(error),
    };
    let (request, audio) =
        match read_form::<TranscriptionRequest>(multipart, TRANSCRIPTION_FIELDS).await {
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

/// The multipart form `/v1/audio/vad` takes.
#[cfg(test)]
#[derive(utoipa::ToSchema)]
#[allow(dead_code)]
pub struct VoiceActivityForm {
    /// The audio to analyse: WAV, MP3, FLAC, OGG and the other formats symphonia decodes.
    #[schema(value_type = String, format = Binary)]
    file: Vec<u8>,
    #[schema(inline)]
    request: VoiceActivityRequest,
}

/// Voice activity endpoint handler: the speech segments (and optionally the per-chunk probabilities) of an upload.
#[cfg_attr(test, utoipa::path(
    post,
    tag = "inference.rs",
    path = "/v1/audio/vad",
    request_body(content = inline(VoiceActivityForm), content_type = "multipart/form-data"),
    responses((status = 200, description = "Speech segments", body = crate::openai::VoiceActivityResponse))
))]
pub async fn voice_activity(
    OwnedEngine(engine): OwnedEngine,
    payload: Result<Multipart, MultipartRejection>,
) -> Response {
    let multipart = match multipart_payload(payload) {
        Ok(multipart) => multipart,
        Err(error) => return openai_error_response(error),
    };
    let (request, audio) =
        match read_form::<VoiceActivityRequest>(multipart, VOICE_ACTIVITY_FIELDS).await {
            Ok(form) => form,
            Err(error) => return openai_error_response(error),
        };
    match engine.voice_activity(request, &audio).await {
        Ok(activity) => (StatusCode::OK, axum::Json(activity)).into_response(),
        Err(error) => openai_error_response(error),
    }
}

/// The multipart form `/v1/audio/diarization` takes.
#[cfg(test)]
#[derive(utoipa::ToSchema)]
#[allow(dead_code)]
pub struct DiarizationForm {
    /// The audio to diarize: WAV, MP3, FLAC, OGG and the other formats symphonia decodes.
    #[schema(value_type = String, format = Binary)]
    file: Vec<u8>,
    #[schema(inline)]
    request: DiarizationRequest,
}

/// Diarization endpoint handler: who speaks when, as JSON segments or RTTM.
#[cfg_attr(test, utoipa::path(
    post,
    tag = "inference.rs",
    path = "/v1/audio/diarization",
    request_body(content = inline(DiarizationForm), content_type = "multipart/form-data"),
    responses((status = 200, description = "Speaker segments", body = crate::openai::DiarizationResponse))
))]
pub async fn diarization(
    OwnedEngine(engine): OwnedEngine,
    payload: Result<Multipart, MultipartRejection>,
) -> Response {
    let multipart = match multipart_payload(payload) {
        Ok(multipart) => multipart,
        Err(error) => return openai_error_response(error),
    };
    let (request, audio) =
        match read_form::<DiarizationRequest>(multipart, DIARIZATION_FIELDS).await {
            Ok(form) => form,
            Err(error) => return openai_error_response(error),
        };
    match engine.diarization(request, &audio).await {
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
