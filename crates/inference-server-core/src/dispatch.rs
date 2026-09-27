//! Sending requests to the engine and routing their responses.

use inference_core::{InferenceRsError, Request, Response};
use tokio::sync::mpsc::{channel, Receiver, Sender};

use crate::types::SharedInferenceRsState;

/// Default buffer size for the response channel used in streaming operations.
///
/// This constant defines the maximum number of response messages that can be buffered
/// in the channel before backpressure is applied. A larger buffer reduces the likelihood
/// of blocking but uses more memory.
pub const DEFAULT_CHANNEL_BUFFER_SIZE: usize = 10_000;

/// Creates a channel for response communication.
pub fn create_response_channel(
    buffer_size: Option<usize>,
) -> (Sender<Response>, Receiver<Response>) {
    let channel_buffer_size = buffer_size.unwrap_or(DEFAULT_CHANNEL_BUFFER_SIZE);
    channel(channel_buffer_size)
}

/// Sends a request to the model processing pipeline.
pub async fn send_request(
    state: &SharedInferenceRsState,
    request: Request,
) -> Result<(), InferenceRsError> {
    send_request_with_model(state, request, None).await
}

pub async fn send_request_with_model(
    state: &SharedInferenceRsState,
    mut request: Request,
    model_id: Option<&str>,
) -> Result<(), InferenceRsError> {
    if let Some(model_id) = model_id {
        if let Request::Normal(request) = &mut request {
            request.model_id = Some(model_id.to_string());
        } else {
            return state
                .get_sender(Some(model_id))?
                .send(request)
                .await
                .map_err(|_| InferenceRsError::SenderPoisoned);
        }
    }
    state.send_request_async(request).await
}

pub(crate) fn request_model_override(
    requested_model: String,
    routed_model: &str,
) -> Option<String> {
    (requested_model != routed_model).then_some(requested_model)
}

pub(crate) fn apply_model_override(model: &mut String, model_override: Option<&str>) {
    if let Some(model_override) = model_override {
        *model = model_override.to_string();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_model_override_only_preserves_routed_aliases() {
        assert_eq!(
            request_model_override("code".to_string(), "base"),
            Some("code".to_string())
        );
        assert_eq!(request_model_override("base".to_string(), "base"), None);

        let mut response_model = "base".to_string();
        apply_model_override(&mut response_model, Some("code"));
        assert_eq!(response_model, "code");
    }
}
