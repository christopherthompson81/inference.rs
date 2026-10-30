//! Decisions as an engine operation: each typed question about a state goes to a decision model, which scores its
//! options in one pass.

use anyhow::Error as AnyhowError;
use futures::future::join_all;
use indexmap::IndexMap;
use inference_core::{InferenceRs, ModelCategory, RequestMessage};
use inference_models_decision::{FormatError, as_text, head};

use crate::{
    api_error::{ApiError, ApiErrorKind},
    dispatch::response_model_id,
    engine_decision_labels,
    engine_embeddings::{EmbeddingError, EmbeddingWithUsage, fetch, saturating_to_u32},
    openai::{DecisionAnswer, DecisionRequest, DecisionResponse, DecisionUsage},
    types::SharedInferenceRsState,
    util::validate_model_name,
};

const NO_DECISIONS: &str = "this model answers no decisions: it is neither a decision model nor a GGUF with a decision profile";

pub(crate) fn invalid(message: impl Into<String>, param: String) -> EmbeddingError {
    EmbeddingError::Validation(AnyhowError::new(ApiError::new(
        ApiErrorKind::InvalidRequest,
        message.into(),
        Some("invalid_question"),
        Some(&param),
    )))
}

impl From<FormatError> for EmbeddingError {
    fn from(error: FormatError) -> Self {
        match error {
            FormatError::Invalid { message, param } => invalid(message, param),
            FormatError::Layout => EmbeddingError::Internal(error.into()),
        }
    }
}

/// Answers each question about the state with a decision model, in the request's order.
pub(crate) async fn decide(
    state: SharedInferenceRsState,
    request: DecisionRequest,
    attachments: crate::media_source::MediaAttachments,
) -> Result<DecisionResponse, EmbeddingError> {
    let repr = serde_json::to_string(&request).expect("Serialization of decision request failed.");
    InferenceRs::maybe_log_request(state.clone(), repr);

    if let Err(e) = validate_model_name(&request.model, state.clone()) {
        return Err(EmbeddingError::Validation(e.into()));
    }
    if request.questions.is_empty() {
        return Err(invalid(
            "questions must hold at least one question",
            "questions".into(),
        ));
    }
    let model_override = (request.model != "default").then(|| request.model.clone());
    if let Ok(Some(profile)) = state.label_decision(model_override.as_deref()) {
        let (answers, input_tokens) = engine_decision_labels::decide(
            &state,
            &profile,
            &request,
            model_override.as_deref(),
            &attachments,
        )
        .await
        .inspect_err(|e| {
            if let EmbeddingError::Internal(e) = e {
                InferenceRs::maybe_log_error(state.clone(), e.as_ref());
            }
        })?;
        return Ok(respond(&state, request, answers, input_tokens));
    }
    if !matches!(
        state.get_model_category(model_override.as_deref()),
        Ok(ModelCategory::Embedding)
    ) {
        return Err(EmbeddingError::Validation(AnyhowError::new(
            ApiError::invalid_request(NO_DECISIONS),
        )));
    }
    let questions = request
        .questions
        .iter()
        .map(|(id, q)| head::question(id, q).map(|question| (id.clone(), question)))
        .collect::<Result<Vec<_>, _>>()?;
    let state_text = as_text(&request.state, false);
    let futures = questions.iter().map(|(_, question)| {
        let state = state.clone();
        let model_override = model_override.clone();
        let message = RequestMessage::Decision {
            state: state_text.clone(),
            question: question.clone(),
        };
        async move { fetch(state, message, model_override.as_deref(), false).await }
    });

    let mut answers = IndexMap::with_capacity(questions.len());
    let mut input_tokens = 0usize;
    for ((id, question), result) in questions.iter().zip(join_all(futures).await) {
        let EmbeddingWithUsage {
            embedding,
            prompt_tokens,
            ..
        } = result.map_err(|e| {
            InferenceRs::maybe_log_error(state.clone(), e.as_ref());
            EmbeddingError::Internal(e)
        })?;
        let Some(logits) = embedding.get(..head::n_options(question)) else {
            return Err(EmbeddingError::Internal(anyhow::anyhow!(
                "the model scored {} options of question `{id}`, which has {}",
                embedding.len(),
                head::n_options(question)
            )));
        };
        answers.insert(id.clone(), head::answer(question, logits));
        input_tokens = input_tokens.saturating_add(prompt_tokens);
    }

    Ok(respond(&state, request, answers, input_tokens))
}

fn respond(
    state: &SharedInferenceRsState,
    request: DecisionRequest,
    answers: IndexMap<String, DecisionAnswer>,
    input_tokens: usize,
) -> DecisionResponse {
    let response = DecisionResponse {
        model: response_model_id(state, request.model.clone(), &request.model)
            .unwrap_or(request.model),
        answers,
        usage: DecisionUsage {
            input_tokens: saturating_to_u32(input_tokens),
            output_tokens: 0,
        },
    };
    InferenceRs::maybe_log_response(state.clone(), &response);
    response
}
