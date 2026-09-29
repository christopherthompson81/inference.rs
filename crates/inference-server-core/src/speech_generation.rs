//! The speech generation route: HTTP framing over the engine's speech generation.

use axum::{
    extract::{Json, State, rejection::JsonRejection},
    http::{HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
};

use crate::{
    generation::generate_speech,
    handler_core::{ApiError, ApiErrorHttp, openai_error_response},
    openai::SpeechGenerationRequest,
    types::ExtractedInferenceRsState,
};

/// Speech generation endpoint handler.
#[utoipa::path(
    post,
    tag = "inference.rs",
    path = "/v1/audio/speech",
    request_body = SpeechGenerationRequest,
    responses((status = 200, description = "Speech generation"))
)]
pub async fn speech_generation(
    State(state): ExtractedInferenceRsState,
    payload: Result<Json<SpeechGenerationRequest>, JsonRejection>,
) -> Response {
    let request = match payload {
        Ok(Json(request)) => request,
        Err(error) => return openai_error_response(ApiError::from_json_rejection(error)),
    };
    match generate_speech(&state, request).await {
        Ok(audio) => {
            let content_type =
                HeaderValue::from_str(&audio.content_type).expect("audio content types are ASCII");
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, content_type)],
                audio.bytes,
            )
                .into_response()
        }
        Err(error) => openai_error_response(error),
    }
}
