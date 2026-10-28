//! Decisions as an engine operation: each typed question about a state goes to a decision model, which scores its
//! options in one pass.

use anyhow::Error as AnyhowError;
use futures::future::join_all;
use indexmap::IndexMap;
use inference_core::decision::{DecisionOption, DecisionQuestion, QuestionKind};
use inference_core::{InferenceRs, ModelCategory, RequestMessage};
use serde_json::Value;

use crate::{
    api_error::{ApiError, ApiErrorKind},
    dispatch::response_model_id,
    engine_decision_labels,
    engine_embeddings::{EmbeddingError, EmbeddingWithUsage, fetch, saturating_to_u32},
    openai::{
        DecisionAnswer, DecisionQuestionType, DecisionRequest, DecisionRequestQuestion,
        DecisionResponse, DecisionUsage,
    },
    types::SharedInferenceRsState,
    util::validate_model_name,
};

// the option counts llama.cpp serves these models with
const MAX_CHOICE_OPTIONS: usize = 255;
const SCORE_LEVELS: std::ops::RangeInclusive<usize> = 2..=10;
const NOUL_KEYS: [&str; 2] = ["false", "true"];
const NO_DECISIONS: &str = "this model answers no decisions: it is neither a decision model nor a GGUF with a decision profile";
// laya's floor on probabilities in the entropy
const ENTROPY_FLOOR: f64 = 1e-12;

pub(crate) fn invalid(message: impl Into<String>, param: String) -> EmbeddingError {
    EmbeddingError::Validation(AnyhowError::new(ApiError::new(
        ApiErrorKind::InvalidRequest,
        message.into(),
        Some("invalid_question"),
        Some(&param),
    )))
}

/// JSON as Python's `json.dumps` writes it, `", "` and `": "` between items, which the models read their inputs as.
pub(crate) fn python_json(value: &Value, ensure_ascii: bool, out: &mut String) {
    match value {
        Value::String(text) => python_string(text, ensure_ascii, out),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                python_json(item, ensure_ascii, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (i, (key, item)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                python_string(key, ensure_ascii, out);
                out.push_str(": ");
                python_json(item, ensure_ascii, out);
            }
            out.push('}');
        }
        other => out.push_str(&other.to_string()),
    }
}

fn python_string(text: &str, ensure_ascii: bool, out: &mut String) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || (ensure_ascii && !c.is_ascii()) => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Text as the model reads it: a string as it is, any other value as `json.dumps` writes it.
fn as_text(value: &Value, ensure_ascii: bool) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => {
            let mut out = String::new();
            python_json(other, ensure_ascii, &mut out);
            out
        }
    }
}

fn description(value: &Value, param: &str) -> Result<Option<String>, EmbeddingError> {
    match value {
        Value::Null => Ok(None),
        Value::String(text) => Ok(Some(text.clone())),
        _ => Err(invalid(
            "a criterion's description must be text or null",
            param.to_string(),
        )),
    }
}

pub(crate) fn kind_of(kind: DecisionQuestionType) -> QuestionKind {
    match kind {
        DecisionQuestionType::Choice => QuestionKind::Choice,
        DecisionQuestionType::Score => QuestionKind::Score,
        DecisionQuestionType::Noul => QuestionKind::Noul,
    }
}

/// The question as the model reads it, with the keys its answer reports.
fn question(
    id: &str,
    question: &DecisionRequestQuestion,
) -> Result<DecisionQuestion, EmbeddingError> {
    let param = format!("questions.{id}.criteria");
    let options = match (question.kind, &question.criteria) {
        (DecisionQuestionType::Choice, Some(Value::Object(criteria))) => criteria
            .iter()
            .map(|(key, value)| {
                Ok(DecisionOption {
                    key: key.clone(),
                    description: description(value, &param)?,
                })
            })
            .collect::<Result<Vec<_>, _>>()?,
        (DecisionQuestionType::Choice, Some(Value::Array(keys))) => keys
            .iter()
            .map(|key| match key {
                Value::String(key) => Ok(DecisionOption {
                    key: key.clone(),
                    description: None,
                }),
                _ => Err(invalid("choice options must be text", param.clone())),
            })
            .collect::<Result<Vec<_>, _>>()?,
        (DecisionQuestionType::Choice, _) => {
            return Err(invalid(
                "a choice question's criteria map each option to its description",
                param,
            ));
        }
        (DecisionQuestionType::Score, Some(Value::Array(levels))) => levels
            .iter()
            .enumerate()
            .map(|(level, value)| {
                Ok(DecisionOption {
                    key: level.to_string(),
                    description: Some(description(value, &param)?.unwrap_or_default()),
                })
            })
            .collect::<Result<Vec<_>, _>>()?,
        (DecisionQuestionType::Score, _) => {
            return Err(invalid(
                "a score question's criteria list its levels, lowest first",
                param,
            ));
        }
        (DecisionQuestionType::Noul, None) => Vec::new(),
        (DecisionQuestionType::Noul, Some(Value::Object(criteria))) => {
            if let Some(key) = criteria
                .keys()
                .find(|key| !NOUL_KEYS.contains(&key.as_str()))
            {
                return Err(invalid(
                    format!("a noul question's criteria describe `true` and `false`, not `{key}`"),
                    param,
                ));
            }
            NOUL_KEYS
                .iter()
                .filter_map(|key| criteria.get(*key).map(|value| (key, value)))
                .map(|(key, value)| {
                    Ok(DecisionOption {
                        key: key.to_string(),
                        description: description(value, &param)?,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?
        }
        (DecisionQuestionType::Noul, Some(_)) => {
            return Err(invalid(
                "a noul question's criteria describe `true` and `false`",
                param,
            ));
        }
    };
    let kind = kind_of(question.kind);
    match kind {
        QuestionKind::Choice if !(2..=MAX_CHOICE_OPTIONS).contains(&options.len()) => {
            return Err(invalid(
                format!("a choice question has 2 to {MAX_CHOICE_OPTIONS} options"),
                param,
            ));
        }
        QuestionKind::Score if !SCORE_LEVELS.contains(&options.len()) => {
            return Err(invalid(
                format!(
                    "a score question has {} to {} levels",
                    SCORE_LEVELS.start(),
                    SCORE_LEVELS.end()
                ),
                param,
            ));
        }
        _ => {}
    }
    Ok(DecisionQuestion {
        kind,
        instructions: as_text(&question.instructions, true),
        options,
    })
}

/// The softmax of a question's logits, and one minus their normalized entropy.
fn probabilities(logits: &[f32]) -> (Vec<f64>, f64) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let exp = logits
        .iter()
        .map(|&z| (z as f64 - max).exp())
        .collect::<Vec<_>>();
    let total = exp.iter().sum::<f64>();
    let p = exp.iter().map(|e| e / total).collect::<Vec<_>>();
    let k = p.len();
    let confidence = if k < 2 {
        1.0
    } else {
        let entropy = -p
            .iter()
            .map(|&p| p * p.clamp(ENTROPY_FLOOR, 1.0).ln())
            .sum::<f64>();
        1.0 - entropy / (k as f64).ln()
    };
    (p, confidence)
}

fn answer(question: &DecisionQuestion, logits: &[f32]) -> DecisionAnswer {
    let (p, confidence) = probabilities(logits);
    match question.kind {
        QuestionKind::Choice => {
            let best = first_max(&p);
            DecisionAnswer::Choice {
                choice: question.options[best].key.clone(),
                probabilities: question
                    .options
                    .iter()
                    .map(|option| option.key.clone())
                    .zip(p)
                    .collect(),
                confidence,
            }
        }
        QuestionKind::Score => DecisionAnswer::Score {
            score: p.iter().enumerate().map(|(i, p)| i as f64 * p).sum(),
            legend: question
                .options
                .iter()
                .map(|option| {
                    (
                        option.key.clone(),
                        Value::String(option.description.clone().unwrap_or_default()),
                    )
                })
                .collect(),
            probabilities: (0..p.len()).map(|i| i.to_string()).zip(p).collect(),
            confidence,
        },
        // the model reads noul's options as [false, true]
        QuestionKind::Noul => DecisionAnswer::Noul { noul: p[1] },
    }
}

/// The index of the largest probability, the first among equals as numpy's and C++'s argmax take it.
pub(crate) fn first_max(p: &[f64]) -> usize {
    (1..p.len()).fold(0, |best, i| if p[i] > p[best] { i } else { best })
}

/// The number of options the model scores for a question.
fn n_options(question: &DecisionQuestion) -> usize {
    match question.kind {
        QuestionKind::Noul => NOUL_KEYS.len(),
        _ => question.options.len(),
    }
}

/// Answers each question about the state with a decision model, in the request's order.
pub(crate) async fn decide(
    state: SharedInferenceRsState,
    request: DecisionRequest,
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
        let (answers, input_tokens) =
            engine_decision_labels::decide(&state, &profile, &request, model_override.as_deref())
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
        .map(|(id, q)| question(id, q).map(|question| (id.clone(), question)))
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
        let Some(logits) = embedding.get(..n_options(question)) else {
            return Err(EmbeddingError::Internal(anyhow::anyhow!(
                "the model scored {} options of question `{id}`, which has {}",
                embedding.len(),
                n_options(question)
            )));
        };
        answers.insert(id.clone(), answer(question, logits));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_text_values_read_as_python_writes_them() {
        let value = serde_json::json!({"plan": "pro", "n": 2, "tags": ["a", "é"], "ok": null});
        assert_eq!(
            as_text(&value, false),
            r#"{"plan": "pro", "n": 2, "tags": ["a", "é"], "ok": null}"#
        );
        assert_eq!(as_text(&serde_json::json!(["é"]), true), "[\"\\u00e9\"]");
        assert_eq!(as_text(&serde_json::json!("as is"), true), "as is");
    }

    #[test]
    fn noul_reads_its_second_option_and_confidence_falls_with_entropy() {
        let noul = DecisionQuestion {
            kind: QuestionKind::Noul,
            instructions: String::new(),
            options: Vec::new(),
        };
        let DecisionAnswer::Noul { noul } = answer(&noul, &[0.0, 2.0f32.ln()]) else {
            unreachable!()
        };
        assert!((noul - 2.0 / 3.0).abs() < 1e-9);
        let (_, even) = probabilities(&[1.0, 1.0, 1.0]);
        let (_, sure) = probabilities(&[30.0, 0.0, 0.0]);
        assert!(even.abs() < 1e-9 && (sure - 1.0).abs() < 1e-9);
    }
}
