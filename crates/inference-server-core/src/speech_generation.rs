//! The speech generation route: HTTP framing over the engine's speech generation.

use axum::{
    extract::State,
    http::{HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
};

use crate::handler_core::{ApiJson, ApiJsonRejection};
use crate::{
    generation::generate_speech, handler_core::openai_error_response,
    openai::SpeechGenerationRequest, types::ExtractedInferenceRsState,
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
    payload: Result<ApiJson<SpeechGenerationRequest>, ApiJsonRejection>,
) -> Response {
    let request = match payload {
        Ok(ApiJson(request)) => request,
        Err(ApiJsonRejection(error)) => return openai_error_response(error),
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
