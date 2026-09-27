//! Engine-level errors: what went wrong and how an OpenAI-style client should see it, independent of transport.

use inference_core::{InferenceRsError, LoraAdapterError, ServiceUnavailableError};
use serde::Serialize;

pub(crate) const INTERNAL_ERROR_MESSAGE: &str = "Internal server error.";
pub(crate) const MODEL_ERROR_MESSAGE: &str = "The model failed to process the request.";
pub(crate) const SERVICE_UNAVAILABLE_MESSAGE: &str = "The service is temporarily unavailable.";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ApiErrorKind {
    InvalidRequest,
    NotFound,
    Conflict,
    PayloadTooLarge,
    UnsupportedMediaType,
    RateLimited,
    Unavailable,
    Overloaded,
    Internal,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ApiError {
    pub(crate) kind: ApiErrorKind,
    pub(crate) message: String,
    pub(crate) code: Option<String>,
    pub(crate) param: Option<String>,
}

impl ApiError {
    /// The OpenAI error `type` for this kind.
    pub(crate) fn openai_type(&self) -> &'static str {
        match self.kind {
            ApiErrorKind::RateLimited => "rate_limit_error",
            ApiErrorKind::Unavailable | ApiErrorKind::Overloaded | ApiErrorKind::Internal => {
                "server_error"
            }
            _ => "invalid_request_error",
        }
    }

    /// The OpenAI error envelope, `{"error": {"message", "type", "param", "code"}}`.
    pub(crate) fn to_openai_body(&self) -> serde_json::Value {
        serde_json::json!({
            "error": {
                "message": self.message,
                "type": self.openai_type(),
                "param": self.param,
                "code": self.code,
            }
        })
    }

    pub(crate) fn new(
        kind: ApiErrorKind,
        message: impl Into<String>,
        code: Option<&str>,
        param: Option<&str>,
    ) -> Self {
        Self {
            kind,
            message: message.into(),
            code: code.map(ToString::to_string),
            param: param.map(ToString::to_string),
        }
    }

    pub(crate) fn invalid_request(message: impl Into<String>) -> Self {
        Self::new(ApiErrorKind::InvalidRequest, message, None, None)
    }

    pub(crate) fn internal() -> Self {
        Self::new(
            ApiErrorKind::Internal,
            INTERNAL_ERROR_MESSAGE,
            Some("internal_error"),
            None,
        )
    }

    pub(crate) fn model_error() -> Self {
        Self::new(
            ApiErrorKind::Internal,
            MODEL_ERROR_MESSAGE,
            Some("model_error"),
            None,
        )
    }

    pub(crate) fn from_error(
        error: &(dyn std::error::Error + 'static),
        fallback: ApiErrorKind,
    ) -> Self {
        if let Some(error) = find_error::<ApiError>(error) {
            return error.clone();
        }
        if let Some(error) = find_error::<InferenceRsError>(error) {
            return Self::from_inference_error(error);
        }
        if find_error::<ServiceUnavailableError>(error).is_some() {
            return Self::new(
                ApiErrorKind::Overloaded,
                SERVICE_UNAVAILABLE_MESSAGE,
                Some("service_unavailable"),
                None,
            );
        }

        match fallback {
            ApiErrorKind::Internal => Self::internal(),
            ApiErrorKind::Unavailable => Self::new(
                ApiErrorKind::Unavailable,
                SERVICE_UNAVAILABLE_MESSAGE,
                Some("service_unavailable"),
                None,
            ),
            ApiErrorKind::Overloaded => Self::new(
                ApiErrorKind::Overloaded,
                SERVICE_UNAVAILABLE_MESSAGE,
                Some("service_unavailable"),
                None,
            ),
            kind => Self::new(kind, error.to_string(), None, None),
        }
    }

    fn from_inference_error(error: &InferenceRsError) -> Self {
        match error {
            InferenceRsError::ModelNotFound(_) => Self::new(
                ApiErrorKind::NotFound,
                error.to_string(),
                Some("model_not_found"),
                Some("model"),
            ),
            InferenceRsError::ModelReloading(_)
            | InferenceRsError::ModelAlreadyLoaded(_)
            | InferenceRsError::ModelAlreadyUnloaded(_) => Self::new(
                ApiErrorKind::Conflict,
                error.to_string(),
                Some("model_state_conflict"),
                Some("model"),
            ),
            InferenceRsError::NoLoaderConfig(_) => Self::new(
                ApiErrorKind::InvalidRequest,
                error.to_string(),
                Some("invalid_model_operation"),
                Some("model"),
            ),
            InferenceRsError::LoraAdapter(error) => Self::from_lora_error(error),
            InferenceRsError::EnginePoisoned
            | InferenceRsError::ReloadFailed(_)
            | InferenceRsError::Other(_) => Self::internal(),
            InferenceRsError::SenderPoisoned => Self::new(
                ApiErrorKind::Unavailable,
                SERVICE_UNAVAILABLE_MESSAGE,
                Some("service_unavailable"),
                None,
            ),
        }
    }

    fn from_lora_error(error: &LoraAdapterError) -> Self {
        let (kind, code) = match error {
            LoraAdapterError::RuntimeUnavailable { .. }
            | LoraAdapterError::TensorParallelUnsupported { .. }
            | LoraAdapterError::RuntimeChanged { .. } => {
                (ApiErrorKind::Conflict, "lora_runtime_unavailable")
            }
            LoraAdapterError::InvalidAlias | LoraAdapterError::AliasTooLong { .. } => {
                (ApiErrorKind::InvalidRequest, "invalid_lora_name")
            }
            LoraAdapterError::LoadBusy => (ApiErrorKind::RateLimited, "lora_load_busy"),
            LoraAdapterError::NotFound { .. } | LoraAdapterError::GenerationNotFound { .. } => {
                (ApiErrorKind::NotFound, "lora_adapter_not_found")
            }
            LoraAdapterError::FileTooLarge { .. } => {
                (ApiErrorKind::PayloadTooLarge, "lora_adapter_file_too_large")
            }
            LoraAdapterError::Io { source, .. }
                if source.kind() == std::io::ErrorKind::NotFound =>
            {
                (ApiErrorKind::NotFound, "adapter_file_not_found")
            }
            LoraAdapterError::Io { source, .. }
                if matches!(
                    source.kind(),
                    std::io::ErrorKind::InvalidData
                        | std::io::ErrorKind::InvalidInput
                        | std::io::ErrorKind::UnexpectedEof
                ) =>
            {
                (ApiErrorKind::InvalidRequest, "invalid_lora_adapter")
            }
            LoraAdapterError::Io { .. } | LoraAdapterError::Load(_) => {
                (ApiErrorKind::Internal, "internal_error")
            }
            LoraAdapterError::Config { .. } | LoraAdapterError::Format(_) => {
                (ApiErrorKind::InvalidRequest, "invalid_lora_adapter")
            }
            LoraAdapterError::AlreadyLoaded { .. }
            | LoraAdapterError::GenerationMismatch { .. }
            | LoraAdapterError::GenerationConflict { .. }
            | LoraAdapterError::AliasLimit { .. }
            | LoraAdapterError::RankLimit { .. }
            | LoraAdapterError::AdapterLimit { .. }
            | LoraAdapterError::ByteLimit { .. }
            | LoraAdapterError::SlotExhausted => (ApiErrorKind::Conflict, "lora_state_conflict"),
            LoraAdapterError::SizeOverflow => {
                (ApiErrorKind::InvalidRequest, "invalid_lora_adapter")
            }
            LoraAdapterError::InvalidRuntimeConfig(_) | LoraAdapterError::Task(_) => {
                (ApiErrorKind::Internal, "internal_error")
            }
            _ => (ApiErrorKind::Internal, "internal_error"),
        };
        let message = if kind == ApiErrorKind::Internal {
            INTERNAL_ERROR_MESSAGE.to_string()
        } else if matches!(kind, ApiErrorKind::Unavailable | ApiErrorKind::Overloaded) {
            SERVICE_UNAVAILABLE_MESSAGE.to_string()
        } else {
            error.to_string()
        };
        Self::new(kind, message, Some(code), Some("adapter"))
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ApiError {}

fn find_error<'a, E: std::error::Error + 'static>(
    mut error: &'a (dyn std::error::Error + 'static),
) -> Option<&'a E> {
    loop {
        if let Some(error) = error.downcast_ref::<E>() {
            return Some(error);
        }
        error = error.source()?;
    }
}

/// Standard JSON error response structure.
#[derive(Serialize, Debug)]
pub(crate) struct JsonError {
    pub(crate) message: String,
}

impl JsonError {
    /// Creates a new JSON error with the specified message.
    pub(crate) fn new(message: String) -> Self {
        Self { message }
    }
}

impl std::fmt::Display for JsonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for JsonError {}

/// Internal error type for model-related errors with a descriptive message.
///
/// This struct wraps error messages from the underlying model and implements
/// the standard error traits for proper error handling and display.
#[derive(Debug)]
pub(crate) struct ModelErrorMessage(pub(crate) String);

impl std::fmt::Display for ModelErrorMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ModelErrorMessage {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_core_errors_without_losing_wrapped_sources() {
        let error = anyhow::Error::new(InferenceRsError::ModelNotFound("missing".to_string()))
            .context("failed to dispatch request");
        let classified = ApiError::from_error(error.as_ref(), ApiErrorKind::Internal);
        assert_eq!(classified.kind, ApiErrorKind::NotFound);
        assert_eq!(classified.code.as_deref(), Some("model_not_found"));
        assert_eq!(classified.param.as_deref(), Some("model"));

        let classified = ApiError::from_error(
            &InferenceRsError::ModelReloading("busy".to_string()),
            ApiErrorKind::Internal,
        );
        assert_eq!(classified.kind, ApiErrorKind::Conflict);

        let classified =
            ApiError::from_error(&InferenceRsError::SenderPoisoned, ApiErrorKind::Internal);
        assert_eq!(classified.kind, ApiErrorKind::Unavailable);

        let error = anyhow::Error::new(ServiceUnavailableError("private detail".to_string()))
            .context("allocation failed");
        let classified = ApiError::from_error(error.as_ref(), ApiErrorKind::Internal);
        assert_eq!(classified.kind, ApiErrorKind::Overloaded);
        assert_eq!(classified.message, SERVICE_UNAVAILABLE_MESSAGE);

        let classified = ApiError::from_error(
            &InferenceRsError::LoraAdapter(LoraAdapterError::SizeOverflow),
            ApiErrorKind::Internal,
        );
        assert_eq!(classified.kind, ApiErrorKind::InvalidRequest);
    }

    #[test]
    fn does_not_expose_internal_core_errors() {
        let error = InferenceRsError::Other("secret backend detail".to_string());
        let classified = ApiError::from_error(&error, ApiErrorKind::InvalidRequest);
        assert_eq!(classified.kind, ApiErrorKind::Internal);
        assert_eq!(classified.message, INTERNAL_ERROR_MESSAGE);
        assert!(!classified.message.contains("secret"));
    }
}
