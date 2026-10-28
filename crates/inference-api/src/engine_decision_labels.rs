//! Decisions on a chat model with a GGUF decision profile: its `systemone` template renders each question, and the
//! next-token logits of one label token per option score it, as llama.cpp's label-coded decision types do.

use std::sync::{Arc, Mutex};

use futures::future::join_all;
use indexmap::IndexMap;
use inference_core::decision::{LabelDecision, QuestionKind};
use inference_core::{
    CustomLogitsProcessor, InferenceRs, NormalRequest, Request, RequestMessage, Response,
    SamplingParams,
};
use inference_tensor::Tensor;
use serde_json::{Map, Value, json};

use crate::{
    api_error::{ApiError, ApiErrorKind, ModelErrorMessage},
    dispatch::send_request_with_model,
    engine_decision::{first_max, invalid, kind_of, python_json},
    engine_embeddings::EmbeddingError,
    openai::{DecisionAnswer, DecisionQuestionType, DecisionRequest},
    operations::{self, TokenizeRequest},
    types::SharedInferenceRsState,
};

const TEMPLATE: &str = "systemone";
const SCORE_LEVELS: std::ops::RangeInclusive<usize> = 2..=10;
const NOUL_KEYS: [&str; 2] = ["false", "true"];
const NOUL_TRUE: &str = "true";
const SCORING_FAILED: &str = "the decision pass returned no label logits";

/// A question as the template reads it: its id, type and instructions, and each option's key and description.
struct LabelQuestion {
    id: String,
    kind: QuestionKind,
    instructions: Value,
    options: Vec<(String, Value)>,
}

/// The logits of the labels at the prompt's last position, kept as the engine samples its one token.
struct LabelCapture {
    labels: Vec<u32>,
    scores: Arc<Mutex<Option<Vec<f32>>>>,
}

impl CustomLogitsProcessor for LabelCapture {
    fn apply(&self, logits: &Tensor, _context: &[u32]) -> inference_tensor::Result<Tensor> {
        let all = logits.to_vec1::<f32>()?;
        let scores = self
            .labels
            .iter()
            .map(|&label| {
                all.get(label as usize)
                    .copied()
                    .unwrap_or(f32::NEG_INFINITY)
            })
            .collect();
        *self
            .scores
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(scores);
        Ok(logits.clone())
    }
}

fn parse(
    profile: &LabelDecision,
    id: &str,
    kind: DecisionQuestionType,
    instructions: &Value,
    criteria: Option<&Value>,
) -> Result<LabelQuestion, EmbeddingError> {
    let param = format!("questions.{id}.criteria");
    let kind = kind_of(kind);
    let mut options = match (kind, criteria) {
        (QuestionKind::Choice, Some(Value::Object(criteria))) if !criteria.is_empty() => criteria
            .iter()
            .map(|(key, description)| (key.clone(), description.clone()))
            .collect(),
        (QuestionKind::Choice, _) => {
            return Err(invalid(
                "a choice question's criteria map each option to its description",
                param,
            ));
        }
        (QuestionKind::Score, Some(Value::Array(levels)))
            if SCORE_LEVELS.contains(&levels.len()) =>
        {
            levels
                .iter()
                .enumerate()
                .map(|(level, description)| (level.to_string(), description.clone()))
                .collect()
        }
        (QuestionKind::Score, _) => {
            return Err(invalid(
                format!(
                    "a score question's criteria list {} to {} levels, lowest first",
                    SCORE_LEVELS.start(),
                    SCORE_LEVELS.end()
                ),
                param,
            ));
        }
        (QuestionKind::Noul, None | Some(Value::Null)) => NOUL_KEYS
            .iter()
            .map(|key| (key.to_string(), Value::Null))
            .collect::<Vec<_>>(),
        (QuestionKind::Noul, Some(Value::Object(criteria))) => NOUL_KEYS
            .iter()
            .map(|key| {
                (
                    key.to_string(),
                    criteria.get(*key).cloned().unwrap_or(Value::Null),
                )
            })
            .collect(),
        (QuestionKind::Noul, Some(_)) => {
            return Err(invalid(
                "a noul question's criteria describe `true` and `false`",
                param,
            ));
        }
    };
    if kind == QuestionKind::Noul && profile.family.noul_true_first() {
        options.swap(0, 1);
    }
    if options.len() > profile.labels.len() {
        return Err(invalid(
            format!(
                "too many options ({}), this model reads at most {}",
                options.len(),
                profile.labels.len()
            ),
            param,
        ));
    }
    Ok(LabelQuestion {
        id: id.to_string(),
        kind,
        instructions: instructions.clone(),
        options,
    })
}

fn sort_keys(value: &Value) -> Value {
    match value {
        Value::Array(items) => Value::Array(items.iter().map(sort_keys).collect()),
        Value::Object(map) => {
            let mut keys = map.keys().collect::<Vec<_>>();
            keys.sort();
            Value::Object(
                keys.into_iter()
                    .map(|key| (key.clone(), sort_keys(&map[key])))
                    .collect::<Map<_, _>>(),
            )
        }
        other => other.clone(),
    }
}

/// The options in the order a variant shows them (the second reversed), each with its label's text.
fn render_options(profile: &LabelDecision, question: &LabelQuestion, variant: usize) -> Value {
    let n = question.options.len();
    Value::Array(
        (0..n)
            .map(|i| {
                let (key, description) =
                    &question.options[if variant == 0 { i } else { n - 1 - i }];
                let mut option = json!({"key": key, "description": description});
                if let Some(label) = profile.label_texts.get(i) {
                    option["label"] = json!(label);
                }
                option
            })
            .collect(),
    )
}

fn render(
    profile: &LabelDecision,
    state: &Value,
    questions: &[LabelQuestion],
    question: &LabelQuestion,
    variant: usize,
) -> Result<String, minijinja::Error> {
    let mut input = json!({
        "id": question.id,
        "type": question.kind.name(),
        "instructions": question.instructions,
        "state": state,
        "options": render_options(profile, question, variant),
    });
    if profile.family.lists_questions() {
        input["questions"] = questions
            .iter()
            .map(|q| {
                json!({
                    "id": q.id,
                    "type": q.kind.name(),
                    "instructions": q.instructions,
                    "options": render_options(profile, q, 0),
                })
            })
            .collect();
    }
    if profile.family.sorts_keys() {
        input = sort_keys(&input);
    }
    input["images"] = json!([]);

    let mut env = minijinja::Environment::new();
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    // llama.cpp's tojson, which these templates were converted for: json.dumps's separators, no ASCII escaping
    env.add_filter("tojson", |value: minijinja::Value| {
        let value = serde_json::to_value(&value).map_err(|e| {
            minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, e.to_string())
        })?;
        let mut out = String::new();
        python_json(&value, false, &mut out);
        Ok(minijinja::Value::from_safe_string(out))
    });
    env.add_template(TEMPLATE, &profile.template)?;
    env.get_template(TEMPLATE)?
        .render(minijinja::Value::from_serialize(&input))
}

/// The answer from each variant's label logits: a softmax at the calibrated temperature, averaged over the variants.
fn answer(
    profile: &LabelDecision,
    question: &LabelQuestion,
    scores: &[Vec<f32>],
) -> DecisionAnswer {
    let n = profile
        .family
        .outputs(question.kind, question.options.len());
    let temperature = f64::from(profile.temperature(question.kind, question.options.len()));
    let bias = (question.kind == QuestionKind::Score)
        .then(|| profile.score_bias.get(&n))
        .flatten();
    let mut p = vec![0.0f64; n];
    for (variant, logits) in scores.iter().enumerate() {
        let logits = logits
            .iter()
            .enumerate()
            .map(|(i, &z)| f64::from(z) + bias.map_or(0.0, |bias| f64::from(bias[i])))
            .collect::<Vec<_>>();
        let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exp = logits
            .iter()
            .map(|z| ((z - max) / temperature).exp())
            .collect::<Vec<_>>();
        let total = exp.iter().sum::<f64>();
        for (i, e) in exp.iter().enumerate() {
            // the second variant is in the reverse order
            let at = if variant == 0 { i } else { n - 1 - i };
            p[at] += e / total / scores.len() as f64;
        }
    }

    let keys = || question.options.iter().map(|(key, _)| key.clone());
    match question.kind {
        QuestionKind::Noul if n != question.options.len() => DecisionAnswer::Noul {
            // lev's ratings run from false to true
            noul: p
                .iter()
                .enumerate()
                .map(|(i, p)| p * i as f64 / (n - 1) as f64)
                .sum(),
        },
        QuestionKind::Noul => DecisionAnswer::Noul {
            noul: keys()
                .zip(&p)
                .find(|(key, _)| key == NOUL_TRUE)
                .map_or(0.0, |(_, &p)| p),
        },
        QuestionKind::Choice => {
            let best = first_max(&p);
            DecisionAnswer::Choice {
                choice: question.options[best].0.clone(),
                confidence: confidence(profile, question.kind, &p),
                probabilities: keys().zip(p).collect(),
            }
        }
        QuestionKind::Score => DecisionAnswer::Score {
            score: p.iter().enumerate().map(|(i, p)| i as f64 * p).sum(),
            legend: question
                .options
                .iter()
                .map(|(key, description)| {
                    let description = match description {
                        Value::String(_) => description.clone(),
                        _ if profile.family.canonical_legend() => {
                            Value::String(sort_keys(description).to_string())
                        }
                        _ => description.clone(),
                    };
                    (key.clone(), description)
                })
                .collect(),
            confidence: confidence(profile, question.kind, &p),
            probabilities: keys().zip(p).collect(),
        },
    }
}

/// TypeSafe's published confidences, or one minus the normalized entropy for the families that report that.
fn confidence(profile: &LabelDecision, kind: QuestionKind, p: &[f64]) -> f64 {
    let n = p.len();
    if n < 2 {
        return 1.0;
    }
    if profile.family.entropy_confidence() {
        let entropy = -p
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|p| p * p.ln())
            .sum::<f64>();
        return (1.0 - entropy / (n as f64).ln()).clamp(0.0, 1.0);
    }
    let mode = first_max(p);
    match kind {
        QuestionKind::Score => {
            let center = (n - 1) as f64 / 2.0;
            let dist = p
                .iter()
                .enumerate()
                .map(|(i, p)| p * (i as f64 - mode as f64).abs())
                .sum::<f64>();
            let uniform = (0..n)
                .map(|i| (i as f64 - center).abs() / n as f64)
                .sum::<f64>();
            (1.0 - dist / uniform).max(0.0)
        }
        _ => {
            let uniform = 1.0 / n as f64;
            ((p[mode] - uniform) / (1.0 - uniform)).max(0.0)
        }
    }
}

/// One prompt's label logits, from a one-token pass whose sampler reads them.
async fn score(
    state: &SharedInferenceRsState,
    model: Option<&str>,
    prompt: String,
    labels: Vec<u32>,
) -> Result<(Vec<f32>, usize), EmbeddingError> {
    let tokens = operations::tokenize(
        state,
        TokenizeRequest {
            text: prompt,
            add_special_tokens: false,
            model: model.map(str::to_string),
        },
    )
    .await
    .map_err(|e| EmbeddingError::Validation(e.into()))?
    .tokens;
    let n_tokens = tokens.len();
    let scores = Arc::new(Mutex::new(None));
    let (tx, mut rx) = tokio::sync::mpsc::channel(1);
    let mut request = NormalRequest::new_simple(
        RequestMessage::CompletionTokens(tokens),
        SamplingParams {
            max_len: Some(1),
            ..SamplingParams::deterministic()
        },
        tx,
        state.next_request_id(),
        None,
        None,
    );
    request.logits_processors = Some(vec![Arc::new(LabelCapture {
        labels,
        scores: scores.clone(),
    })]);
    send_request_with_model(state, Request::Normal(Box::new(request)), model)
        .await
        .map_err(|e| EmbeddingError::Internal(e.into()))?;
    let failed = |error: ApiError| EmbeddingError::Internal(error.into());
    match rx.recv().await {
        Some(Response::CompletionDone(_) | Response::Done(_)) => {}
        Some(Response::ValidationError(error)) => {
            return Err(EmbeddingError::Validation(anyhow::anyhow!(
                error.to_string()
            )));
        }
        Some(Response::InternalError(error)) => {
            InferenceRs::maybe_log_error(state.clone(), error.as_ref());
            return Err(failed(ApiError::from_error(
                error.as_ref(),
                ApiErrorKind::Internal,
            )));
        }
        Some(Response::ModelError(message, _) | Response::CompletionModelError(message, _)) => {
            return Err(failed(ApiError::from_error(
                &ModelErrorMessage(message),
                ApiErrorKind::Internal,
            )));
        }
        _ => return Err(EmbeddingError::Internal(anyhow::anyhow!(SCORING_FAILED))),
    }
    let scores = scores
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take()
        .ok_or_else(|| EmbeddingError::Internal(anyhow::anyhow!(SCORING_FAILED)))?;
    Ok((scores, n_tokens))
}

/// Answers each question with the label logits of its rendered prompts, in the request's order; with the prompt
/// tokens read.
pub(crate) async fn decide(
    state: &SharedInferenceRsState,
    profile: &LabelDecision,
    request: &DecisionRequest,
    model: Option<&str>,
) -> Result<(IndexMap<String, DecisionAnswer>, usize), EmbeddingError> {
    let questions = request
        .questions
        .iter()
        .map(|(id, q)| parse(profile, id, q.kind, &q.instructions, q.criteria.as_ref()))
        .collect::<Result<Vec<_>, _>>()?;
    let mut prompts = Vec::new();
    for (index, question) in questions.iter().enumerate() {
        let n = profile
            .family
            .outputs(question.kind, question.options.len());
        for variant in 0..profile
            .family
            .variants(question.kind, question.options.len())
        {
            let prompt =
                render(profile, &request.state, &questions, question, variant).map_err(|e| {
                    invalid(
                        format!("rendering the decision prompt failed: {e}"),
                        "questions".into(),
                    )
                })?;
            prompts.push((index, prompt, profile.labels[..n].to_vec()));
        }
    }
    let results = join_all(
        prompts
            .into_iter()
            .map(|(index, prompt, labels)| async move {
                (index, score(state, model, prompt, labels).await)
            }),
    )
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
        .map(|(question, scores)| (question.id.clone(), answer(profile, question, scores)))
        .collect();
    Ok((answers, input_tokens))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use inference_core::decision::LabelFamily;

    use super::*;

    fn profile(family: LabelFamily, template: &str) -> LabelDecision {
        LabelDecision::new(
            family,
            template.to_string(),
            HashMap::new(),
            HashMap::new(),
            |code: &str| Ok::<_, String>(code.bytes().map(u32::from).collect()),
        )
        .unwrap()
    }

    fn question(
        profile: &LabelDecision,
        kind: DecisionQuestionType,
        criteria: Value,
    ) -> LabelQuestion {
        parse(profile, "q", kind, &json!("Which?"), Some(&criteria))
            .unwrap_or_else(|_| panic!("the question parses"))
    }

    #[test]
    fn templates_read_json_as_llama_cpp_writes_it() {
        let template =
            "{{ state | tojson }}|{% for o in options %}{{ o.label }}={{ o.key }};{% endfor %}";
        let state = json!({"b": 1, "a": "é<\u{1}", "c": [true, null]});
        let nimble = profile(LabelFamily::Nimble, template);
        let q = question(
            &nimble,
            DecisionQuestionType::Choice,
            json!({"x": null, "y": "d"}),
        );
        assert_eq!(
            render(&nimble, &state, &[], &q, 0).unwrap(),
            "{\"b\": 1, \"a\": \"é<\\u0001\", \"c\": [true, null]}|A=x;B=y;"
        );
        // lev sorts the keys, and its second variant shows the options reversed under the same labels
        let lev = profile(LabelFamily::Lev, template);
        let q = question(
            &lev,
            DecisionQuestionType::Choice,
            json!({"x": null, "y": "d"}),
        );
        assert_eq!(
            render(&lev, &state, &[], &q, 1).unwrap(),
            "{\"a\": \"é<\\u0001\", \"b\": 1, \"c\": [true, null]}|A=y;B=x;"
        );
    }

    #[test]
    fn noul_lists_true_first_where_the_family_reads_it_so() {
        let criteria = json!({"true": "holds"});
        let jev = profile(LabelFamily::OpenJev, "");
        let q = question(&jev, DecisionQuestionType::Noul, criteria.clone());
        assert_eq!(q.options[0], ("true".to_string(), json!("holds")));
        let answer = answer(&jev, &q, &[vec![2.0, 0.0]]);
        let DecisionAnswer::Noul { noul } = answer else {
            panic!()
        };
        assert!((noul - 1.0 / (1.0 + (-2.0f64).exp())).abs() < 1e-12);
        let nimble = profile(LabelFamily::Nimble, "");
        let q = question(&nimble, DecisionQuestionType::Noul, criteria);
        assert_eq!(q.options[1].0, "true");
    }

    #[test]
    fn lev_averages_the_reversed_variant_back_and_rates_noul() {
        let lev = profile(LabelFamily::Lev, "");
        let q = question(
            &lev,
            DecisionQuestionType::Choice,
            json!({"x": null, "y": null, "z": null}),
        );
        // the second variant shows z, y, x: its first label scores z
        let DecisionAnswer::Choice {
            probabilities,
            choice,
            ..
        } = answer(&lev, &q, &[vec![1.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]])
        else {
            panic!()
        };
        assert!((probabilities["x"] - probabilities["z"]).abs() < 1e-12);
        assert!(probabilities["x"] > probabilities["y"]);
        assert_eq!(choice, "x");
        let q = question(&lev, DecisionQuestionType::Noul, Value::Null);
        let mut ratings = vec![f32::NEG_INFINITY; 9];
        ratings[6] = 0.0;
        let DecisionAnswer::Noul { noul } = answer(&lev, &q, &[ratings]) else {
            panic!()
        };
        assert!((noul - 0.75).abs() < 1e-12);
    }

    #[test]
    fn legend_and_confidence_follow_the_family() {
        let levels = json!(["low", {"b": 1, "a": 2}]);
        let d3 = profile(LabelFamily::VllmD3, "");
        let q = question(&d3, DecisionQuestionType::Score, levels.clone());
        let DecisionAnswer::Score {
            legend,
            confidence: even,
            ..
        } = answer(&d3, &q, &[vec![0.0, 0.0]])
        else {
            panic!()
        };
        assert_eq!(legend["1"], json!("{\"a\":2,\"b\":1}"));
        assert!(even.abs() < 1e-12);
        let nimble = profile(LabelFamily::Nimble, "");
        let q = question(&nimble, DecisionQuestionType::Score, levels);
        let DecisionAnswer::Score { legend, .. } = answer(&nimble, &q, &[vec![0.0, 0.0]]) else {
            panic!()
        };
        assert_eq!(legend["1"], json!({"b": 1, "a": 2}));
        // TypeSafe's choice confidence: (p_max - 1/k) / (1 - 1/k)
        assert!((confidence(&nimble, QuestionKind::Choice, &[0.7, 0.2, 0.1]) - 0.55).abs() < 1e-12);
        // score: one minus the mean distance to the mode over a uniform's around its center
        let score = confidence(&nimble, QuestionKind::Score, &[0.0, 0.5, 0.5]);
        assert!((score - (1.0 - 0.5 / (2.0 / 3.0))).abs() < 1e-12);
    }
}
