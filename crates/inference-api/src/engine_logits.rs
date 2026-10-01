//! Scoring a prompt: the log-probability of each of its tokens, and on request the raw logits behind them.

use candle_core::{DType, Tensor};
use inference_core::{
    InferenceRs, ModelCategory, NormalRequest, Request, RequestMessage, Response, SamplingParams,
};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::{
    api_error::{ApiError, ApiErrorKind, ModelErrorMessage},
    dispatch::send_request_with_model,
    lora_routing::DEFAULT_MODEL_ID,
    models::loaded_model,
    operations::{self, TokenizeRequest},
    types::SharedInferenceRsState,
};

const SCORING_FAILED: &str = "prompt scoring did not return logits";

/// The prompt to score, as text (tokenized with the model's special tokens) or as token ids.
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
#[serde(untagged)]
pub enum PromptInput {
    Text(String),
    Tokens(Vec<u32>),
}

/// What a scored prompt carries besides the token log-probabilities.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum LogitsOutput {
    #[default]
    Logprobs,
    /// Also every position's full logits, `tokens.len() * vocab_size` floats.
    Logits,
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct PromptLogitsRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub prompt: PromptInput,
    #[serde(default)]
    pub output: LogitsOutput,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct PromptLogits {
    pub tokens: Vec<u32>,
    pub vocab_size: usize,
    /// `log p(tokens[i] | tokens[..i])`; null for the first token, which nothing predicts.
    pub token_logprobs: Vec<Option<f32>>,
    /// Row-major `[tokens.len(), vocab_size]`, present when the request asked for `logits`.
    #[serde(skip)]
    pub logits: Option<Vec<f32>>,
}

pub(crate) async fn prompt_logits(
    state: &SharedInferenceRsState,
    request: PromptLogitsRequest,
) -> Result<PromptLogits, ApiError> {
    let model = request
        .model
        .as_deref()
        .filter(|model| *model != DEFAULT_MODEL_ID);
    if let Some(model) = model {
        loaded_model(state, model)?;
    }
    match state.get_model_category(model) {
        Ok(ModelCategory::Text | ModelCategory::Multimodal { .. }) => {}
        Ok(_) => {
            return Err(ApiError::invalid_request(
                "prompt scoring needs a text or multimodal model",
            ));
        }
        Err(error) => return Err(ApiError::from_error(&error, ApiErrorKind::Internal)),
    }
    let tokens = match request.prompt {
        PromptInput::Tokens(tokens) => tokens,
        PromptInput::Text(text) => {
            let tokenized = TokenizeRequest {
                text,
                add_special_tokens: true,
                model: model.map(str::to_string),
            };
            operations::tokenize(state, tokenized).await?.tokens
        }
    };
    if tokens.len() < 2 {
        return Err(ApiError::invalid_request(
            "a prompt to score needs at least two tokens",
        ));
    }
    // The whole prompt runs in one forward pass, so it must fit the model's context.
    if let Ok(Some(max)) = state.max_sequence_length(model)
        && tokens.len() > max
    {
        return Err(ApiError::invalid_request(format!(
            "the prompt has {} tokens; the model accepts {max}",
            tokens.len()
        )));
    }

    let (tx, mut rx) = tokio::sync::mpsc::channel(1);
    // A raw-logits request ends after its prefill pass; one token only satisfies the generation limit's check.
    let sampling = SamplingParams {
        max_len: Some(1),
        ..SamplingParams::deterministic()
    };
    let mut normal = NormalRequest::new_simple(
        RequestMessage::CompletionTokens(tokens),
        sampling,
        tx,
        state.next_request_id(),
        None,
        None,
    );
    normal.return_raw_logits = true;
    send_request_with_model(state, Request::Normal(Box::new(normal)), model)
        .await
        .map_err(|error| ApiError::from_error(&error, ApiErrorKind::Internal))?;
    let (chunks, tokens) = match rx.recv().await {
        Some(Response::Raw {
            logits_chunks,
            tokens,
        }) => (logits_chunks, tokens),
        Some(Response::ValidationError(error)) => {
            return Err(ApiError::from_error(
                error.as_ref(),
                ApiErrorKind::InvalidRequest,
            ));
        }
        Some(Response::InternalError(error)) => {
            InferenceRs::maybe_log_error(state.clone(), error.as_ref());
            return Err(ApiError::from_error(error.as_ref(), ApiErrorKind::Internal));
        }
        Some(Response::ModelError(message, _)) => {
            InferenceRs::maybe_log_error(state.clone(), &ModelErrorMessage(message));
            return Err(ApiError::model_error());
        }
        _ => {
            return Err(ApiError::new(
                ApiErrorKind::Internal,
                SCORING_FAILED,
                None,
                None,
            ));
        }
    };
    scored(chunks, tokens, request.output)
        .map_err(|error| ApiError::from_error(&error, ApiErrorKind::Internal))
}

fn scored(
    chunks: Vec<Tensor>,
    tokens: Vec<u32>,
    output: LogitsOutput,
) -> candle_core::Result<PromptLogits> {
    let logits = Tensor::cat(&chunks, 0)?.to_dtype(DType::F32)?;
    let (rows, vocab_size) = logits.dims2()?;
    if rows != tokens.len() {
        candle_core::bail!("{rows} rows of logits for {} prompt tokens", tokens.len());
    }
    if let Some(token) = tokens.iter().find(|&&token| token as usize >= vocab_size) {
        candle_core::bail!("token {token} is outside the model's {vocab_size}-entry vocabulary");
    }
    // Row i - 1 predicts token i, so the last row predicts nothing in the prompt.
    let predicting = logits.narrow(0, 0, rows - 1)?;
    let max = predicting.max_keepdim(1)?;
    let log_sum_exp = predicting
        .broadcast_sub(&max)?
        .exp()?
        .sum_keepdim(1)?
        .log()?
        .add(&max)?;
    let targets = Tensor::new(&tokens[1..], predicting.device())?.unsqueeze(1)?;
    let scores = predicting
        .gather(&targets, 1)?
        .sub(&log_sum_exp)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    if scores.iter().any(|score| !score.is_finite()) {
        candle_core::bail!("the model's logits are not finite");
    }
    let logits = match output {
        LogitsOutput::Logits => Some(logits.flatten_all()?.to_vec1::<f32>()?),
        LogitsOutput::Logprobs => None,
    };
    Ok(PromptLogits {
        tokens,
        vocab_size,
        token_logprobs: std::iter::once(None)
            .chain(scores.into_iter().map(Some))
            .collect(),
        logits,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_token_outside_the_vocabulary_is_an_error() -> candle_core::Result<()> {
        let logits = Tensor::new(&[[0.0_f32, 2.0], [5.0, 0.0]], &candle_core::Device::Cpu)?;
        assert!(scored(vec![logits], vec![0, 7], LogitsOutput::Logprobs).is_err());
        Ok(())
    }

    #[test]
    fn each_token_is_scored_by_the_row_before_it() -> candle_core::Result<()> {
        // Two positions over a vocabulary of two; row 0 predicts token 1, which is the second prompt token.
        let logits = Tensor::new(&[[0.0_f32, 2.0], [5.0, 0.0]], &candle_core::Device::Cpu)?;
        let scored = scored(vec![logits], vec![0, 1], LogitsOutput::Logits)?;
        assert_eq!(scored.token_logprobs[0], None);
        let expected = 2.0 - (1.0_f32 + 2.0_f32.exp()).ln();
        assert!((scored.token_logprobs[1].unwrap() - expected).abs() < 1e-6);
        assert_eq!(scored.logits.map(|logits| logits.len()), Some(4));
        Ok(())
    }
}
