//! Scoring a prompt: the log-probability of each of its tokens, and on request the raw logits behind them.

use inference_core::{
    InferenceRs, ModelCategory, NormalRequest, Request, RequestMessage, Response, SamplingParams,
};
use inference_tensor::{DType, Tensor};
use serde::Serialize;
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
#[derive(Debug, Clone, inference_json::FromJson, Serialize, ToSchema)]
#[serde(untagged)]
pub enum PromptInput {
    Text(String),
    Tokens(Vec<u32>),
}

/// What a scored prompt carries besides the token log-probabilities.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, inference_json::FromJson, Serialize, ToSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum LogitsOutput {
    #[default]
    Logprobs,
    /// Also every position's full logits, `tokens.len() * vocab_size` floats.
    Logits,
}

#[derive(Debug, Clone, inference_json::FromJson, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct PromptLogitsRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub prompt: PromptInput,
    #[serde(default)]
    pub output: LogitsOutput,
    /// Audio clips (`data:` URLs, or sources the server's fetcher resolves) for the prompt's audio placeholders, in
    /// order; a multimodal model expands each placeholder as a chat's.
    #[serde(default)]
    pub audio: Vec<String>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct PromptLogits {
    pub tokens: Vec<u32>,
    pub vocab_size: usize,
    /// `log p(tokens[i] | tokens[..i])`; null for the first token, which nothing predicts, and for tokens past the
    /// vocabulary (an audio placeholder the model replaces).
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

    let mut audios = Vec::with_capacity(request.audio.len());
    for source in &request.audio {
        audios.push(
            crate::util::parse_audio_url(source, &crate::media_source::MediaAttachments::default())
                .await
                .map_err(|error| ApiError::invalid_request(format!("audio `{source}`: {error}")))?,
        );
    }
    let message = if audios.is_empty() {
        RequestMessage::CompletionTokens(tokens)
    } else {
        RequestMessage::MultimodalTokens {
            tokens,
            images: Vec::new(),
            audios,
        }
    };
    let (tx, mut rx) = tokio::sync::mpsc::channel(1);
    // A raw-logits request ends after its prefill pass; one token only satisfies the generation limit's check.
    let sampling = SamplingParams {
        max_len: Some(1),
        ..SamplingParams::deterministic()
    };
    let mut normal =
        NormalRequest::new_simple(message, sampling, tx, state.next_request_id(), None, None);
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
        Some(Response::ModelError(message, _) | Response::CompletionModelError(message, _)) => {
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
) -> inference_tensor::Result<PromptLogits> {
    let logits = Tensor::cat(&chunks, 0)?.to_dtype(DType::F32)?;
    let (rows, vocab_size) = logits.dims2()?;
    if rows != tokens.len() {
        inference_tensor::bail!("{rows} rows of logits for {} prompt tokens", tokens.len());
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
    // a placeholder past the vocabulary is gathered as token 0, then scored as null
    let in_vocab = |token: u32| (token as usize) < vocab_size;
    let gathered = tokens[1..]
        .iter()
        .map(|&token| if in_vocab(token) { token } else { 0 })
        .collect::<Vec<_>>();
    let targets = Tensor::new(gathered.as_slice(), predicting.device())?.unsqueeze(1)?;
    let scores = predicting
        .gather(&targets, 1)?
        .sub(&log_sum_exp)?
        .flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .zip(&tokens[1..])
        .map(|(score, &token)| in_vocab(token).then_some(score))
        .collect::<Vec<_>>();
    if scores.iter().flatten().any(|score| !score.is_finite()) {
        inference_tensor::bail!("the model's logits are not finite");
    }
    let logits = match output {
        LogitsOutput::Logits => Some(logits.flatten_all()?.to_vec1::<f32>()?),
        LogitsOutput::Logprobs => None,
    };
    Ok(PromptLogits {
        tokens,
        vocab_size,
        token_logprobs: std::iter::once(None).chain(scores).collect(),
        logits,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_token_outside_the_vocabulary_scores_null() -> inference_tensor::Result<()> {
        let logits = Tensor::new(
            &[[0.0_f32, 2.0], [5.0, 0.0], [1.0, 0.0]],
            &inference_tensor::Device::Cpu,
        )?;
        let scored = scored(vec![logits], vec![0, 7, 1], LogitsOutput::Logprobs)?;
        assert_eq!(scored.token_logprobs[1], None);
        let expected = -(1.0_f32 + 5.0_f32.exp()).ln();
        assert!((scored.token_logprobs[2].unwrap() - expected).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn each_token_is_scored_by_the_row_before_it() -> inference_tensor::Result<()> {
        // Two positions over a vocabulary of two; row 0 predicts token 1, which is the second prompt token.
        let logits = Tensor::new(
            &[[0.0_f32, 2.0], [5.0, 0.0]],
            &inference_tensor::Device::Cpu,
        )?;
        let scored = scored(vec![logits], vec![0, 1], LogitsOutput::Logits)?;
        assert_eq!(scored.token_logprobs[0], None);
        let expected = 2.0 - (1.0_f32 + 2.0_f32.exp()).ln();
        assert!((scored.token_logprobs[1].unwrap() - expected).abs() < 1e-6);
        assert_eq!(scored.logits.map(|logits| logits.len()), Some(4));
        Ok(())
    }
}
