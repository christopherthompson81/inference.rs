//! Request bodies parsed once, here, for both the HTTP server and the C ABI.

use inference_json::FromJson;

use crate::api_error::{ApiError, ApiErrorKind};

pub const MALFORMED_JSON: &str = "malformed_json";
pub const INVALID_REQUEST_BODY: &str = "invalid_request_body";
const SYNTAX_ERROR_PREFIX: &str = "Failed to parse the request body as JSON";
const DATA_ERROR_PREFIX: &str = "Failed to deserialize the JSON body into the target type";

/// A request body type. Each impl is concrete, so its deserializer is compiled in this crate only.
pub trait JsonRequest: Sized {
    fn from_json(body: &[u8]) -> Result<Self, ApiError>;
}

/// Parses `body` as `T` with the failing field's path; call it only from a [`JsonRequest`] impl.
pub fn parse_json<T: FromJson>(body: &[u8]) -> Result<T, ApiError> {
    let value = inference_json::parse_slice(body).map_err(|error| syntax_error(&error))?;
    T::from_json(&value).map_err(|error| {
        ApiError::new(
            ApiErrorKind::InvalidRequest,
            format!("{DATA_ERROR_PREFIX}: {error}"),
            Some(INVALID_REQUEST_BODY),
            None,
        )
    })
}

fn syntax_error(error: &dyn std::fmt::Display) -> ApiError {
    ApiError::new(
        ApiErrorKind::InvalidRequest,
        format!("{SYNTAX_ERROR_PREFIX}: {error}"),
        Some(MALFORMED_JSON),
        None,
    )
}

macro_rules! json_requests {
    ($($request:ty),* $(,)?) => {
        $(impl JsonRequest for $request {
            fn from_json(body: &[u8]) -> Result<Self, ApiError> {
                parse_json(body)
            }
        })*
    };
}

json_requests!(
    inference_protocol::openai::ChatCompletionRequest,
    inference_protocol::openai::CompletionRequest,
    inference_protocol::openai::EmbeddingRequest,
    inference_protocol::openai::RerankRequest,
    inference_protocol::openai::DecisionRequest,
    inference_protocol::openai::ImageGenerationRequest,
    inference_protocol::openai::ImageEditRequest,
    inference_protocol::openai::SpeechGenerationRequest,
    inference_protocol::openai::TranscriptionRequest,
    inference_protocol::openai::TranscriptionSessionRequest,
    inference_protocol::openai::VoiceActivityRequest,
    inference_protocol::openai::DiarizationRequest,
    inference_core::SerializedSession,
    crate::anthropic::AnthropicMessagesRequest,
    crate::responses::OpenResponsesCreateRequest,
    crate::agentic::ApprovalDecisionRequest,
    crate::lora_adapters::ListLoraAdaptersQuery,
    crate::lora_adapters::LoadLoraAdapterRequest,
    crate::lora_adapters::UnloadLoraAdapterRequest,
    crate::models::ModelOperationRequest,
    crate::models::ModelAlias,
    crate::engine::ModelSpec,
    crate::operations::SessionForkRequest,
    crate::engine_logits::PromptLogitsRequest,
    crate::operations::ReIsqRequest,
    crate::operations::CalibrationApplyRequest,
    crate::operations::CalibrationTarget,
    crate::operations::TokenizeRequest,
    crate::operations::DetokenizeRequest,
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::ReIsqRequest;

    #[test]
    fn syntax_and_data_errors_get_distinct_codes() {
        let syntax = <ReIsqRequest as JsonRequest>::from_json(b"{").unwrap_err();
        assert_eq!(syntax.code.as_deref(), Some(MALFORMED_JSON));
        assert!(syntax.message.starts_with(SYNTAX_ERROR_PREFIX));

        let data = <ReIsqRequest as JsonRequest>::from_json(b"{}").unwrap_err();
        assert_eq!(data.code.as_deref(), Some(INVALID_REQUEST_BODY));
        assert!(data.message.starts_with(DATA_ERROR_PREFIX));

        let wrong_type =
            <ReIsqRequest as JsonRequest>::from_json(br#"{"ggml_type": 5}"#).unwrap_err();
        assert!(
            wrong_type.message.contains("ggml_type: invalid type"),
            "{}",
            wrong_type.message
        );

        let trailing =
            <ReIsqRequest as JsonRequest>::from_json(br#"{"ggml_type": "q4k"} x"#).unwrap_err();
        assert_eq!(trailing.code.as_deref(), Some(MALFORMED_JSON));
    }
}
