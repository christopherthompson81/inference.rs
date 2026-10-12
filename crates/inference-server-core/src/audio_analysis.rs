//! The audio analysis routes: multipart uploads over the engine's speech recognition and voice activity detection.

use axum::{
    extract::{Multipart, multipart::MultipartRejection},
    http::{HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
};
use inference_api::api_error::ApiError;

use crate::{
    form::{Field, invalid, multipart_payload, read_form},
    handler_core::openai_error_response,
    openai::{DiarizationRequest, TranscriptionRequest, VoiceActivityRequest},
    types::OwnedEngine,
};

const FILE_FIELD: &str = "file";
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

const TRANSCRIPTION_FIELDS: &[(&str, Field)] = &[
    ("temperature", Field::Number),
    ("timestamp_granularities", Field::List),
    ("beam_size", Field::Integer),
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

// OpenAI's audio forms carry one upload
async fn read_audio_form<T: serde::de::DeserializeOwned>(
    multipart: Multipart,
    kinds: &[(&str, Field)],
) -> Result<(T, Vec<u8>), ApiError> {
    let (request, mut uploads) = read_form(multipart, kinds, &[FILE_FIELD], INVALID_CODE).await?;
    match uploads.len() {
        0 => Err(invalid("The form has no `file`.", FILE_FIELD, INVALID_CODE)),
        1 => Ok((request, uploads.remove(0).bytes)),
        _ => Err(invalid(
            "The form has more than one `file`.",
            FILE_FIELD,
            INVALID_CODE,
        )),
    }
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
    let multipart = match multipart_payload(payload, INVALID_CODE) {
        Ok(multipart) => multipart,
        Err(error) => return openai_error_response(error),
    };
    let (request, audio) =
        match read_audio_form::<TranscriptionRequest>(multipart, TRANSCRIPTION_FIELDS).await {
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
    let multipart = match multipart_payload(payload, INVALID_CODE) {
        Ok(multipart) => multipart,
        Err(error) => return openai_error_response(error),
    };
    let (request, audio) =
        match read_audio_form::<VoiceActivityRequest>(multipart, VOICE_ACTIVITY_FIELDS).await {
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
    let multipart = match multipart_payload(payload, INVALID_CODE) {
        Ok(multipart) => multipart,
        Err(error) => return openai_error_response(error),
    };
    let (request, audio) =
        match read_audio_form::<DiarizationRequest>(multipart, DIARIZATION_FIELDS).await {
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
