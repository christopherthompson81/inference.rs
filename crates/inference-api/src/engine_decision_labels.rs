//! Decisions on a chat model with a GGUF decision profile, as engine passes: the prompts its format renders, scored
//! by label logits, a pointer head, marker logits or the joint head.

use std::sync::{Arc, Mutex};

use futures::future::join_all;
use indexmap::IndexMap;
use inference_core::decision::{DecisionMedia, DecisionPrompt, LabelDecision};
use inference_core::{
    CustomLogitsProcessor, InferenceRs, JointDecisionRequest, NormalRequest, Request,
    RequestMessage, Response, SamplingParams,
};
use inference_models_decision::labels::{self, LabelQuestion, Prompt};
use inference_tensor::{DType, Tensor};
use serde_json::{Value, json};

use crate::media_source::MediaAttachments;
use crate::{
    api_error::{ApiError, ApiErrorKind, ModelErrorMessage},
    dispatch::send_request_with_model,
    engine_decision::invalid,
    engine_embeddings::{EmbeddingError, fetch},
    openai::{DecisionAnswer, DecisionRequest},
    operations::{self, DetokenizeRequest, TokenizeRequest},
    types::SharedInferenceRsState,
};

const SCORING_FAILED: &str = "the decision pass returned no label logits";
const MAX_MEDIA: usize = 8;
const DATA_URL: &str = "data:";
const DATA_AUDIO: &str = "data:audio/";
const DEFAULT_AUDIO_FORMAT: &str = "wav";
const POINTER_FAILED: &str = "the decision pass returned no hidden states";

/// The logits of the labels at the prompt's last position, kept as the engine samples its one token.
struct LabelCapture {
    groups: Vec<Vec<u32>>,
    scores: Arc<Mutex<Option<Vec<f32>>>>,
}

impl CustomLogitsProcessor for LabelCapture {
    fn apply(&self, logits: &Tensor, _context: &[u32]) -> inference_tensor::Result<Tensor> {
        let scores = labels::label_scores(&logits.to_vec1::<f32>()?, &self.groups);
        *self
            .scores
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(scores);
        Ok(logits.clone())
    }
}

/// The state as h2o-lightning reads it, once per request: compact JSON, cut to its token budget, quoted.
async fn h2o_state(
    state: &SharedInferenceRsState,
    model: Option<&str>,
    profile: &LabelDecision,
    value: &Value,
) -> Result<Value, EmbeddingError> {
    let mut text = labels::h2o_state_text(value);
    // a byte-level BPE never has more tokens than bytes
    if text.len() > labels::H2O_MAX_STATE_TOKENS {
        let tokens = tokenize(state, model, text.clone()).await?;
        if tokens.len() > labels::H2O_MAX_STATE_TOKENS {
            let half = labels::H2O_MAX_STATE_TOKENS / 2;
            let piece = |tokens: &[u32]| {
                operations::detokenize(
                    state,
                    DetokenizeRequest {
                        tokens: tokens.to_vec(),
                        skip_special_tokens: false,
                        model: model.map(str::to_string),
                    },
                )
            };
            let head = piece(&tokens[..half]).await;
            let tail = piece(&tokens[tokens.len() - half..]).await;
            let failed = |e: ApiError| EmbeddingError::Internal(e.into());
            text = labels::h2o_cut(&head.map_err(failed)?.text, &tail.map_err(failed)?.text);
        }
    }
    Ok(json!(labels::h2o_quote(profile, &text)))
}

/// d1-omni's prompt for one question: the format's pieces tokenized one by one, then cut to their budgets.
async fn d1omni_prompt(
    state: &SharedInferenceRsState,
    profile: &LabelDecision,
    request_state: &Value,
    question: &LabelQuestion,
    media: Option<&Arc<DecisionMedia>>,
    model: Option<&str>,
) -> Result<DecisionPrompt, EmbeddingError> {
    let n_media = media.map_or(0, |media| media.images.len() + media.audios.len());
    let is_audio = media.is_some_and(|media| !media.audios.is_empty());
    let (pieces, max_tokens) =
        labels::d1omni_pieces(profile, request_state, question, n_media, is_audio)?;
    let mut tokenized = Vec::with_capacity(pieces.len());
    for (text, role) in pieces {
        tokenized.push((tokenize(state, model, text).await?, role));
    }
    let (ids, markers) =
        labels::d1omni_ids(profile, tokenized, max_tokens, question.options.len())?;
    Ok(DecisionPrompt {
        media: media.cloned(),
        ids,
        markers,
        kind: question.kind,
    })
}

/// d1-omni's marker logits for one question, from the model's decision pass over its prompt.
async fn score_markers(
    state: &SharedInferenceRsState,
    profile: &LabelDecision,
    request_state: &Value,
    question: &LabelQuestion,
    media: Option<&Arc<DecisionMedia>>,
    model: Option<&str>,
) -> Result<(Vec<f32>, usize), EmbeddingError> {
    let prompt = d1omni_prompt(state, profile, request_state, question, media, model).await?;
    let result = fetch(
        state.clone(),
        RequestMessage::DecisionTokens { prompt },
        model,
        false,
    )
    .await
    .map_err(EmbeddingError::Internal)?;
    let logits = result
        .embedding
        .get(..question.options.len())
        .ok_or_else(|| EmbeddingError::Internal(anyhow::anyhow!(SCORING_FAILED)))?
        .to_vec();
    // the media rows are the model's, ahead of the prompt it was sent
    let media_rows = media.map_or(0, |media| media.rows());
    Ok((logits, result.prompt_tokens + media_rows))
}

/// clef: the questions decided together, one prompt and one pass, then the joint head over the prompt's hidden states.
async fn decide_joint(
    state: &SharedInferenceRsState,
    profile: &LabelDecision,
    prompt: &Prompt<'_>,
    questions: &[LabelQuestion],
    media: &Media,
    model: Option<&str>,
) -> Result<(Vec<Vec<f32>>, usize), EmbeddingError> {
    // the model was trained with the pieces tokenized one by one
    let mut pieces = Vec::new();
    for (text, role) in labels::joint_pieces(profile, prompt.state, questions, &prompt.images)? {
        pieces.push((tokenize(state, model, text).await?, role));
    }
    let layout = labels::joint_layout(pieces, questions)?;
    let message = match media.is_empty() {
        true => RequestMessage::CompletionTokens(layout.tokens.clone()),
        false => RequestMessage::MultimodalTokens {
            tokens: layout.tokens.clone(),
            images: media.images.clone(),
            audios: media.audios.clone(),
        },
    };
    let Response::Raw {
        logits_chunks,
        tokens,
    } = run(state, model, message, |request| {
        request.return_raw_logits = true;
    })
    .await?
    else {
        return Err(EmbeddingError::Internal(anyhow::anyhow!(POINTER_FAILED)));
    };
    let layout = labels::shift_layout(layout, tokens)?;
    let n_tokens = layout.tokens.len();
    let hidden = Tensor::cat(&logits_chunks, 0).map_err(|e| EmbeddingError::Internal(e.into()))?;
    let (tx, mut rx) = tokio::sync::mpsc::channel(1);
    let request = Request::JointDecision(JointDecisionRequest {
        hidden,
        layout,
        response: tx,
    });
    send_request_with_model(state, request, model)
        .await
        .map_err(|e| EmbeddingError::Internal(e.into()))?;
    let scores = rx
        .recv()
        .await
        .ok_or_else(|| EmbeddingError::Internal(anyhow::anyhow!(POINTER_FAILED)))?
        .map_err(EmbeddingError::Internal)?;
    Ok((labels::split_scores(&scores, questions), n_tokens))
}

async fn tokenize(
    state: &SharedInferenceRsState,
    model: Option<&str>,
    prompt: String,
) -> Result<Vec<u32>, EmbeddingError> {
    Ok(operations::tokenize(
        state,
        TokenizeRequest {
            text: prompt,
            add_special_tokens: false,
            model: model.map(str::to_string),
        },
    )
    .await
    .map_err(|e| EmbeddingError::Validation(e.into()))?
    .tokens)
}

/// A one-token pass over the prompt, set up by `configure`; its final response, or the error it carried.
async fn run(
    state: &SharedInferenceRsState,
    model: Option<&str>,
    message: RequestMessage,
    configure: impl FnOnce(&mut NormalRequest),
) -> Result<Response, EmbeddingError> {
    let (tx, mut rx) = tokio::sync::mpsc::channel(1);
    let mut request = NormalRequest::new_simple(
        message,
        SamplingParams {
            max_len: Some(1),
            ..SamplingParams::deterministic()
        },
        tx,
        state.next_request_id(),
        None,
        None,
    );
    configure(&mut request);
    send_request_with_model(state, Request::Normal(Box::new(request)), model)
        .await
        .map_err(|e| EmbeddingError::Internal(e.into()))?;
    let failed = |error: ApiError| EmbeddingError::Internal(error.into());
    match rx.recv().await {
        Some(Response::ValidationError(error)) => Err(EmbeddingError::Validation(anyhow::anyhow!(
            error.to_string()
        ))),
        Some(Response::InternalError(error)) => {
            InferenceRs::maybe_log_error(state.clone(), error.as_ref());
            Err(failed(ApiError::from_error(
                error.as_ref(),
                ApiErrorKind::Internal,
            )))
        }
        Some(Response::ModelError(message, _) | Response::CompletionModelError(message, _)) => {
            Err(failed(ApiError::from_error(
                &ModelErrorMessage(message),
                ApiErrorKind::Internal,
            )))
        }
        Some(response) => Ok(response),
        None => Err(EmbeddingError::Internal(anyhow::anyhow!(SCORING_FAILED))),
    }
}

/// One prompt's label logits, from a one-token pass whose sampler reads them.
async fn score(
    state: &SharedInferenceRsState,
    model: Option<&str>,
    prompt: String,
    groups: Vec<Vec<u32>>,
    media: &Media,
) -> Result<(Vec<f32>, usize), EmbeddingError> {
    let tokens = tokenize(state, model, prompt).await?;
    let scores = Arc::new(Mutex::new(None));
    let capture = LabelCapture {
        groups,
        scores: scores.clone(),
    };
    let message = match media.is_empty() {
        true => RequestMessage::CompletionTokens(tokens),
        false => RequestMessage::MultimodalTokens {
            tokens,
            images: media.images.clone(),
            audios: media.audios.clone(),
        },
    };
    let n_tokens = match run(state, model, message, |request| {
        request.logits_processors = Some(vec![Arc::new(capture)]);
    })
    .await?
    {
        // the prompt as the model read it, with each image's tokens in place of its placeholder
        Response::CompletionDone(done) => done.usage.prompt_tokens,
        Response::Done(done) => done.usage.prompt_tokens,
        _ => return Err(EmbeddingError::Internal(anyhow::anyhow!(SCORING_FAILED))),
    };
    let scores = scores
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take()
        .ok_or_else(|| EmbeddingError::Internal(anyhow::anyhow!(SCORING_FAILED)))?;
    Ok((scores, n_tokens))
}

/// One prompt's pointer scores: the last token's query against each option end token's key, over the decision head's
/// `[q | k]` rows, scaled by the key width's square root.
async fn score_pointer(
    state: &SharedInferenceRsState,
    model: Option<&str>,
    prompt: String,
    option_end: u32,
    n_options: usize,
) -> Result<(Vec<f32>, usize), EmbeddingError> {
    let tokens = tokenize(state, model, prompt).await?;
    let markers = labels::pointer_markers(&tokens, option_end, n_options)?;
    let n_tokens = tokens.len();
    let Response::Raw { logits_chunks, .. } = run(
        state,
        model,
        RequestMessage::CompletionTokens(tokens),
        |request| {
            request.return_raw_logits = true;
        },
    )
    .await?
    else {
        return Err(EmbeddingError::Internal(anyhow::anyhow!(POINTER_FAILED)));
    };
    let rows = (|| {
        let rows = Tensor::cat(&logits_chunks, 0)?.to_dtype(DType::F32)?;
        let (n, width) = rows.dims2()?;
        if n != n_tokens || width % 2 != 0 {
            inference_tensor::bail!("{n} rows of width {width} for {n_tokens} prompt tokens");
        }
        Ok((rows.to_vec2::<f32>()?, width / 2))
    })();
    let (rows, half) =
        rows.map_err(|e: inference_tensor::Error| EmbeddingError::Internal(e.into()))?;
    Ok((labels::pointer_scores(&rows, half, &markers), n_tokens))
}

/// A request's images and audio clips, decoded, in the order the prompt places them.
#[derive(Default)]
struct Media {
    images: Vec<image::DynamicImage>,
    audios: Vec<inference_core::AudioInput>,
}

impl Media {
    fn is_empty(&self) -> bool {
        self.images.is_empty() && self.audios.is_empty()
    }

    fn len(&self) -> usize {
        self.images.len() + self.audios.len()
    }
}

/// One data URL as an image, or an audio clip when it says so.
async fn load_media(
    url: &Value,
    media: &mut Media,
    attachments: &MediaAttachments,
) -> Result<(), EmbeddingError> {
    let fail = |message: String| invalid(message, "files".into());
    let Some(url) = url.as_str().filter(|url| url.starts_with(DATA_URL)) else {
        return Err(fail(
            "media must be data URLs (data:image/...;base64,... or data:audio/...;base64,...)"
                .into(),
        ));
    };
    if media.len() >= MAX_MEDIA {
        return Err(fail(format!(
            "too many media files, the maximum is {MAX_MEDIA}"
        )));
    }
    if url.starts_with(DATA_AUDIO) {
        let audio = crate::util::parse_audio_url(url, attachments)
            .await
            .map_err(|e| fail(e.to_string()))?;
        media.audios.push(audio);
    } else {
        let image = crate::util::parse_image_url(url, attachments)
            .await
            .map_err(|e| fail(e.to_string()))?;
        media.images.push(image);
    }
    Ok(())
}

/// The request's media: `files` and `images`, then the image and audio parts of a chat-message state, which leave
/// the state (llama.cpp's `parse_state`).
async fn collect_media(
    request: &DecisionRequest,
    attachments: &MediaAttachments,
) -> Result<(Value, Media), EmbeddingError> {
    let mut media = Media::default();
    for url in request.files.iter().chain(&request.images) {
        load_media(&json!(url), &mut media, attachments).await?;
    }
    let wrapped = request.state.get("messages").is_some_and(Value::is_array);
    let messages = match wrapped {
        true => &request.state["messages"],
        false => &request.state,
    };
    let Value::Array(messages) = messages else {
        return Ok((request.state.clone(), media));
    };
    let mut kept = Vec::with_capacity(messages.len());
    for message in messages {
        let Some(Value::Array(parts)) = message.get("content") else {
            kept.push(message.clone());
            continue;
        };
        let mut content = Vec::with_capacity(parts.len());
        for part in parts {
            match part.get("type").and_then(Value::as_str) {
                Some("image_url") if part.get("image_url").is_some() => {
                    let image_url = &part["image_url"];
                    load_media(
                        image_url.get("url").unwrap_or(image_url),
                        &mut media,
                        attachments,
                    )
                    .await?;
                }
                Some("input_audio") if part.get("input_audio").is_some() => {
                    let audio = &part["input_audio"];
                    let format = audio
                        .get("format")
                        .and_then(Value::as_str)
                        .unwrap_or(DEFAULT_AUDIO_FORMAT);
                    let Some(data) = audio.get("data").and_then(Value::as_str) else {
                        return Err(invalid("audio must be base64 data", "state".into()));
                    };
                    load_media(
                        &json!(format!("{DATA_AUDIO}{format};base64,{data}")),
                        &mut media,
                        attachments,
                    )
                    .await?;
                }
                _ => content.push(part.clone()),
            }
        }
        let mut message = message.clone();
        message["content"] = Value::Array(content);
        kept.push(message);
    }
    let state = match wrapped {
        true => {
            let mut state = request.state.clone();
            state["messages"] = Value::Array(kept);
            state
        }
        false => Value::Array(kept),
    };
    Ok((state, media))
}

/// Answers each question with the label logits of its rendered prompts, in the request's order; with the prompt
/// tokens read.
pub(crate) async fn decide(
    state: &SharedInferenceRsState,
    profile: &LabelDecision,
    request: &DecisionRequest,
    model: Option<&str>,
    attachments: &MediaAttachments,
) -> Result<(IndexMap<String, DecisionAnswer>, usize), EmbeddingError> {
    let questions = request
        .questions
        .iter()
        .map(|(id, q)| labels::parse(profile, id, q))
        .collect::<Result<Vec<_>, _>>()?;
    if request.state.is_null() && !profile.family.accepts_null_state() {
        return Err(invalid("`state` must be provided", "state".into()));
    }
    if profile.family.strict_payloads() && !labels::is_payload(&request.state) {
        return Err(invalid(
            "`state` must be text, an object or an array",
            "state".into(),
        ));
    }
    let (request_state, media) = collect_media(request, attachments).await?;
    if !media.is_empty() {
        labels::check_media(profile, media.images.len(), media.audios.len())?;
    }
    let with_media = !media.is_empty();
    let (request_state, questions) = match with_media {
        true => labels::strip_media_markers(&request_state, questions),
        false => (request_state, questions),
    };
    let input_state = match profile.family.prepares_texts() {
        true => h2o_state(state, model, profile, &request_state).await?,
        false => request_state,
    };
    if profile.family.scores_markers() {
        let Media { images, audios } = media;
        let media = with_media.then(|| Arc::new(DecisionMedia::new(images, audios)));
        let runs = join_all(questions.iter().map(|question| {
            score_markers(
                state,
                profile,
                &input_state,
                question,
                media.as_ref(),
                model,
            )
        }))
        .await;
        let mut answers = IndexMap::with_capacity(questions.len());
        let mut input_tokens = 0usize;
        for (question, run) in questions.iter().zip(runs) {
            let (logits, n) = run?;
            answers.insert(
                question.id.clone(),
                labels::answer(profile, question, &[logits], with_media),
            );
            input_tokens = input_tokens.saturating_add(n);
        }
        return Ok((answers, input_tokens));
    }
    let placeholder = profile.image_placeholder.clone().unwrap_or_default();
    let shared = Prompt {
        state: &input_state,
        questions: &questions,
        images: json!(vec![placeholder; media.images.len()]),
    };
    if profile.family.joint() {
        let (scores, input_tokens) = match profile.family.joint_per_question() {
            // one prompt each, run together
            true => {
                let runs = join_all(questions.iter().map(|question| {
                    decide_joint(
                        state,
                        profile,
                        &shared,
                        std::slice::from_ref(question),
                        &media,
                        model,
                    )
                }))
                .await;
                let mut scores = Vec::with_capacity(questions.len());
                let mut tokens = 0usize;
                for run in runs {
                    let (mut own, n) = run?;
                    scores.append(&mut own);
                    tokens = tokens.saturating_add(n);
                }
                (scores, tokens)
            }
            false => decide_joint(state, profile, &shared, &questions, &media, model).await?,
        };
        let answers = questions
            .iter()
            .zip(scores)
            .map(|(question, scores)| {
                (
                    question.id.clone(),
                    labels::answer(profile, question, &[scores], with_media),
                )
            })
            .collect();
        return Ok((answers, input_tokens));
    }
    let mut prompts = Vec::new();
    for (index, question) in questions.iter().enumerate() {
        for variant in 0..profile
            .family
            .variants(question.kind, question.options.len())
        {
            let prompt = labels::render(profile, &shared, question, variant)?;
            prompts.push((index, prompt));
        }
    }
    let all = &questions;
    let media = &media;
    let results = join_all(prompts.into_iter().map(|(index, prompt)| async move {
        let question = &all[index];
        let result = match profile.option_end {
            Some(option_end) => {
                score_pointer(state, model, prompt, option_end, question.options.len()).await
            }
            None => score(state, model, prompt, question.label_groups.clone(), media).await,
        };
        (index, result)
    }))
    .await;

    let mut scores = vec![Vec::new(); questions.len()];
    let mut input_tokens = 0usize;
    for (index, result) in results {
        let (logits, n_tokens) = result?;
        scores[index].push(logits);
        input_tokens = input_tokens.saturating_add(n_tokens);
    }
    let answers = questions
        .iter()
        .zip(&scores)
        .map(|(question, scores)| {
            (
                question.id.clone(),
                labels::answer(profile, question, scores, with_media),
            )
        })
        .collect();
    Ok((answers, input_tokens))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn png_url() -> String {
        use base64::Engine as _;
        let mut png = Vec::new();
        image::DynamicImage::new_rgb8(2, 2)
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(png)
        )
    }

    fn request(state: Value, images: Vec<String>) -> DecisionRequest {
        serde_json::from_value(json!({
            "state": state,
            "images": images,
            "questions": {"q": {"type": "noul", "instructions": "x"}},
        }))
        .unwrap()
    }

    // `images` and a chat state's image parts both load, and the parts leave the state
    #[tokio::test]
    async fn media_come_from_the_request_and_its_chat_state() {
        let state = json!({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "see"},
            {"type": "image_url", "image_url": {"url": png_url()}},
        ]}]});
        let Ok((state, media)) = collect_media(
            &request(state, vec![png_url()]),
            &MediaAttachments::default(),
        )
        .await
        else {
            panic!("the media load");
        };
        assert_eq!(media.images.len(), 2);
        assert_eq!(
            state,
            json!({"messages": [{"role": "user", "content": [{"type": "text", "text": "see"}]}]})
        );
        let remote = request(json!("s"), vec!["https://example.com/a.png".into()]);
        assert!(
            collect_media(&remote, &MediaAttachments::default())
                .await
                .is_err()
        );
    }
}
