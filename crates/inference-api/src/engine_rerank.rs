//! Reranking as an engine operation: a cross-encoder scores each (query, document) pair.

use anyhow::Error as AnyhowError;
use futures::future::join_all;
use inference_core::{InferenceRs, RequestMessage};

use crate::{
    api_error::{ApiError, ApiErrorKind},
    dispatch::response_model_id,
    engine_embeddings::{EmbeddingError, EmbeddingWithUsage, fetch, saturating_to_u32},
    openai::{EmbeddingUsage, RerankDocument, RerankRequest, RerankResponse, RerankResult},
    types::SharedInferenceRsState,
    util::validate_model_name,
};

/// Scores every document against the query, most relevant first.
pub(crate) async fn rerank(
    state: SharedInferenceRsState,
    request: RerankRequest,
) -> Result<RerankResponse, EmbeddingError> {
    let repr = serde_json::to_string(&request).expect("Serialization of rerank request failed.");
    InferenceRs::maybe_log_request(state.clone(), repr);

    if let Err(e) = validate_model_name(&request.model, state.clone()) {
        return Err(EmbeddingError::Validation(e.into()));
    }
    if request.documents.is_empty() {
        return Err(EmbeddingError::Validation(AnyhowError::new(ApiError::new(
            ApiErrorKind::InvalidRequest,
            "documents must contain at least one entry.",
            Some("invalid_input"),
            Some("documents"),
        ))));
    }

    let model_override = (request.model != "default").then(|| request.model.clone());
    let truncate_sequence = request.truncate_sequence.unwrap_or(false);
    let futures = request.documents.iter().map(|document| {
        let state = state.clone();
        let model_override = model_override.clone();
        let message = RequestMessage::EmbeddingPair {
            first: request.query.clone(),
            second: document.clone(),
        };
        async move { fetch(state, message, model_override.as_deref(), truncate_sequence).await }
    });

    let mut results = Vec::with_capacity(request.documents.len());
    let (mut prompt_tokens, mut total_tokens) = (0usize, 0usize);
    for (index, result) in join_all(futures).await.into_iter().enumerate() {
        let EmbeddingWithUsage {
            embedding,
            prompt_tokens: item_prompt_tokens,
            total_tokens: item_total_tokens,
        } = result.map_err(|e| {
            InferenceRs::maybe_log_error(state.clone(), e.as_ref());
            EmbeddingError::Internal(e)
        })?;
        let [logit] = embedding[..] else {
            return Err(EmbeddingError::Validation(AnyhowError::new(ApiError::new(
                ApiErrorKind::InvalidRequest,
                format!(
                    "Reranking needs a cross-encoder that scores each pair with one logit; this model returned {} values.",
                    embedding.len()
                ),
                Some("model_not_a_reranker"),
                Some("model"),
            ))));
        };
        let relevance_score = if request.raw_scores {
            logit
        } else {
            1.0 / (1.0 + (-logit).exp())
        };
        results.push(RerankResult {
            index,
            relevance_score,
            document: request.return_documents.then(|| RerankDocument {
                text: request.documents[index].clone(),
            }),
        });
        prompt_tokens = prompt_tokens.saturating_add(item_prompt_tokens);
        total_tokens = total_tokens.saturating_add(item_total_tokens);
    }
    results.sort_by(|a, b| b.relevance_score.total_cmp(&a.relevance_score));
    if let Some(top_n) = request.top_n {
        results.truncate(top_n);
    }

    let response = RerankResponse {
        object: "list",
        model: response_model_id(&state, request.model.clone(), &request.model)
            .unwrap_or(request.model),
        results,
        usage: EmbeddingUsage {
            prompt_tokens: saturating_to_u32(prompt_tokens),
            total_tokens: saturating_to_u32(total_tokens),
        },
    };
    InferenceRs::maybe_log_response(state.clone(), &response);
    Ok(response)
}
