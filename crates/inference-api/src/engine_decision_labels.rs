//! Decisions on a chat model with a GGUF decision profile: its `systemone` template renders each question, and the
//! next-token logits of one label token per option score it, as llama.cpp's label-coded decision types do; kev's
//! pointer head scores it from the hidden states of the option end tokens and the last token instead.

use std::sync::{Arc, Mutex};

use futures::future::join_all;
use indexmap::IndexMap;
use inference_core::decision::{JointLayout, LabelDecision, QuestionKind};
use inference_core::{
    CustomLogitsProcessor, InferenceRs, JointDecisionRequest, NormalRequest, Request,
    RequestMessage, Response, SamplingParams,
};
use inference_tensor::{DType, Tensor};
use serde_json::{Map, Value, json};

use crate::{
    api_error::{ApiError, ApiErrorKind, ModelErrorMessage},
    dispatch::send_request_with_model,
    engine_decision::{
        PythonJson, first_max, invalid, kind_of, required_instructions, required_kind,
    },
    engine_embeddings::EmbeddingError,
    openai::{DecisionAnswer, DecisionQuestionType, DecisionRequest, DecisionRequestQuestion},
    operations::{self, DetokenizeRequest, TokenizeRequest},
    types::SharedInferenceRsState,
};

const TEMPLATE: &str = "systemone";
const SCORE_LEVELS: std::ops::RangeInclusive<usize> = 2..=10;
const NOUL_KEYS: [&str; 2] = ["false", "true"];
const NOUL_TRUE: &str = "true";
const NOUL_YES: &str = "yes";
const NOUL_NO: &str = "no";
// h2o-lightning's state budget, past which its head and tail are kept around an ellipsis line
const H2O_MAX_STATE_TOKENS: usize = 32_000;
const H2O_CUT: &str = "\n...\n";
// the whitespace llama.cpp's jinja `trim` strips (C `isspace`)
const TRIM_CHARS: [char; 6] = [' ', '\t', '\n', '\u{b}', '\u{c}', '\r'];
const H2O_INSTRUCTIONS: PythonJson = PythonJson {
    indent: Some(1),
    item_sep: ",",
    key_sep: ": ",
    ensure_ascii: true,
    python_floats: true,
};
const H2O_DESCRIPTION: PythonJson = PythonJson {
    indent: None,
    item_sep: ", ",
    key_sep: ": ",
    ensure_ascii: false,
    python_floats: true,
};
const H2O_STATE: PythonJson = PythonJson {
    indent: None,
    item_sep: ",",
    key_sep: ":",
    ensure_ascii: false,
    python_floats: true,
};
const SCORING_FAILED: &str = "the decision pass returned no label logits";
const POINTER_FAILED: &str = "the decision pass returned no hidden states";
const KEV_INDENT: &str = "  ";
// clef's template splits its prompt into pieces at `sep` and marks the start of a question or option span
const CLEF_MARKER: &str = "<<clef:";
const CLEF_MARKER_ESCAPED: &str = "<<clef ";
const CLEF_SEP: &str = "<<clef:sep>>";
const CLEF_MARK_QUESTION: &str = "<<clef:question>>";
const CLEF_MARK_OPTION: &str = "<<clef:option>>";

/// A question as the template reads it: its id, type and instructions, and each option's key and description.
struct LabelQuestion {
    id: String,
    kind: QuestionKind,
    instructions: Value,
    options: Vec<(String, Value)>,
    /// the text the template shows for each option's label, and the tokens whose best logit scores each output
    label_texts: Vec<String>,
    label_groups: Vec<Vec<u32>>,
}

/// The logits of the labels at the prompt's last position, kept as the engine samples its one token.
struct LabelCapture {
    groups: Vec<Vec<u32>>,
    scores: Arc<Mutex<Option<Vec<f32>>>>,
}

impl CustomLogitsProcessor for LabelCapture {
    fn apply(&self, logits: &Tensor, _context: &[u32]) -> inference_tensor::Result<Tensor> {
        let all = logits.to_vec1::<f32>()?;
        let scores = self
            .groups
            .iter()
            .map(|group| {
                group
                    .iter()
                    .map(|&label| {
                        all.get(label as usize)
                            .copied()
                            .unwrap_or(f32::NEG_INFINITY)
                    })
                    .fold(f32::NEG_INFINITY, f32::max)
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
    question: &DecisionRequestQuestion,
) -> Result<LabelQuestion, EmbeddingError> {
    let param = format!("questions.{id}.criteria");
    let relaxed = profile.family.prepares_texts();
    let kind = kind_of(match question.kind {
        None if relaxed => DecisionQuestionType::Choice,
        kind => required_kind(id, kind)?,
    });
    let instructions = match &question.instructions {
        Value::Null if relaxed => json!(""),
        instructions if profile.family.default_question() => instructions.clone(),
        instructions => {
            required_instructions(id, instructions)?;
            instructions.clone()
        }
    };
    let levels = match relaxed {
        true => 1..=profile.max_options(),
        false => SCORE_LEVELS,
    };
    let mut options = match (kind, question.criteria.as_ref()) {
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
        (QuestionKind::Score, Some(Value::Array(given))) if levels.contains(&given.len()) => given
            .iter()
            .enumerate()
            .map(|(level, description)| (level.to_string(), description.clone()))
            .collect(),
        (QuestionKind::Score, _) => {
            return Err(invalid(
                format!(
                    "a score question's criteria list {} to {} levels, lowest first",
                    levels.start(),
                    levels.end()
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
                // h2o-lightning also reads the descriptions under `yes` and `no`
                let alias = if *key == NOUL_TRUE { NOUL_YES } else { NOUL_NO };
                let description = criteria
                    .get(*key)
                    .or_else(|| relaxed.then(|| criteria.get(alias)).flatten());
                (key.to_string(), description.cloned().unwrap_or(Value::Null))
            })
            .collect(),
        (QuestionKind::Noul, Some(_)) => {
            return Err(invalid(
                "a noul question's criteria describe `true` and `false`",
                param,
            ));
        }
    };
    if profile.family.strict_payloads() {
        strict_question(
            id,
            kind,
            &instructions,
            question.criteria.as_ref(),
            &options,
        )?;
    }
    if kind == QuestionKind::Choice && profile.family.sorts_choices() {
        options.sort_by(|a, b| a.0.cmp(&b.0));
    }
    let criteria_true_first = matches!(
        question.criteria.as_ref(),
        Some(Value::Object(criteria)) if criteria.len() == 2 && criteria.keys().next().map(String::as_str) == Some(NOUL_TRUE)
    );
    if kind == QuestionKind::Noul
        && (profile.family.noul_true_first()
            || (profile.family.noul_criteria_order() && criteria_true_first))
    {
        options.swap(0, 1);
    }
    if options.len() > profile.max_options() {
        return Err(invalid(
            format!(
                "too many options ({}), this model reads at most {}",
                options.len(),
                profile.max_options()
            ),
            param,
        ));
    }
    let (label_texts, label_groups) = match profile.family.labels_per_question() {
        true => {
            let keys = options
                .iter()
                .map(|(key, _)| key.as_str())
                .collect::<Vec<_>>();
            profile
                .question_labels(kind, &keys)
                .map_err(|message| invalid(message, param))?
        }
        false => {
            let n = profile.family.outputs(kind, options.len());
            (
                profile
                    .label_texts
                    .iter()
                    .take(options.len())
                    .cloned()
                    .collect(),
                profile
                    .labels
                    .iter()
                    .take(n)
                    .map(|&label| vec![label])
                    .collect(),
            )
        }
    };
    Ok(LabelQuestion {
        id: id.to_string(),
        kind,
        instructions,
        options,
        label_texts,
        label_groups,
    })
}

/// Text, an object or an array: what vllm-decision2's reference takes as a state, instructions or description.
fn is_payload(value: &Value) -> bool {
    matches!(value, Value::String(_) | Value::Object(_) | Value::Array(_))
}

/// vllm-decision2's checks on a question, as its reference makes them.
fn strict_question(
    id: &str,
    kind: QuestionKind,
    instructions: &Value,
    criteria: Option<&Value>,
    options: &[(String, Value)],
) -> Result<(), EmbeddingError> {
    let fail = |message: &str| Err(invalid(message, format!("questions.{id}")));
    if !is_payload(instructions) || instructions.as_str() == Some("") {
        return fail("`instructions` must be non-empty text, an object or an array");
    }
    match kind {
        QuestionKind::Choice => {
            if options.iter().any(|(key, description)| {
                key.is_empty() || !(description.is_null() || is_payload(description))
            }) {
                return fail(
                    "option keys must be non-empty, descriptions must be null, text, an object or an array",
                );
            }
            if options.len() < 2 {
                return fail("`criteria` must have at least 2 options");
            }
        }
        QuestionKind::Score => {
            if options
                .iter()
                .any(|(_, description)| !is_payload(description))
            {
                return fail("level descriptions must be text, an object or an array");
            }
        }
        QuestionKind::Noul => {
            if let Some(Value::Object(criteria)) = criteria
                && criteria.iter().any(|(key, description)| {
                    !NOUL_KEYS.contains(&key.as_str()) || !is_payload(description)
                })
            {
                return fail(
                    "`criteria` may only describe `false` and `true` with text, an object or an array",
                );
            }
        }
    }
    Ok(())
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

/// Python's `str.isspace`.
fn python_space(c: char) -> bool {
    matches!(c, '\u{9}'..='\u{d}' | '\u{1c}'..='\u{20}' | '\u{85}' | '\u{a0}' | '\u{1680}')
        || matches!(
            c,
            '\u{2000}'..='\u{200a}' | '\u{2028}' | '\u{2029}' | '\u{202f}' | '\u{205f}'
        )
        || c == '\u{3000}'
}

/// The tokenizer's special spellings in `text` with their angle brackets as JSON escapes, so they stay text.
fn h2o_quote(profile: &LabelDecision, text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    'scan: while let Some(c) = rest.chars().next() {
        for spelling in &profile.quoted_spellings {
            if rest.starts_with(spelling.as_str()) {
                out.push_str(&spelling.replace('<', "\\u003c").replace('>', "\\u003e"));
                rest = &rest[spelling.len()..];
                continue 'scan;
            }
        }
        out.push(c);
        rest = &rest[c.len_utf8()..];
    }
    out
}

/// A string as it is, any other value as `layout` writes it.
fn h2o_text(value: &Value, layout: &PythonJson) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => layout.dumps(other),
    }
}

/// The state as h2o-lightning reads it, once per request: compact JSON, cut to its token budget, quoted.
async fn h2o_state(
    state: &SharedInferenceRsState,
    model: Option<&str>,
    profile: &LabelDecision,
    value: &Value,
) -> Result<Value, EmbeddingError> {
    let mut text = h2o_text(value, &H2O_STATE);
    // a byte-level BPE never has more tokens than bytes
    if text.len() > H2O_MAX_STATE_TOKENS {
        let tokens = tokenize(state, model, text.clone()).await?;
        if tokens.len() > H2O_MAX_STATE_TOKENS {
            let half = H2O_MAX_STATE_TOKENS / 2;
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
            text = format!(
                "{}{H2O_CUT}{}",
                head.map_err(failed)?.text,
                tail.map_err(failed)?.text
            );
        }
    }
    Ok(json!(h2o_quote(profile, &text)))
}

/// h2o-lightning's template input: texts its shim writes, which the template only lays out.
fn h2o_input(profile: &LabelDecision, state: &Value, question: &LabelQuestion) -> Value {
    let instructions = h2o_text(&question.instructions, &H2O_INSTRUCTIONS);
    let n = question.options.len();
    let options = question
        .options
        .iter()
        .enumerate()
        .map(|(i, (key, description))| {
            let mut description = match description {
                // the template has a default, except for a score level, which reads as JSON null
                Value::Null if question.kind == QuestionKind::Score => json!("null"),
                Value::Null => Value::Null,
                other => json!(h2o_quote(profile, &h2o_text(other, &H2O_DESCRIPTION))),
            };
            let key = h2o_quote(profile, key);
            // the reference strips the user turn like `str.strip`, so the text that ends it loses its trailing space
            if i + 1 == n {
                let shown = match (&description, question.kind) {
                    (Value::Null, QuestionKind::Choice) => Some(key.clone()),
                    (Value::String(text), _) => Some(text.clone()),
                    _ => None,
                };
                if let Some(text) = shown {
                    let stripped = text.trim_end_matches(python_space);
                    if stripped != text {
                        description = json!(if stripped.is_empty() { " " } else { stripped });
                    }
                }
            }
            json!({"label": question.label_texts[i], "key": key, "description": description})
        })
        .collect::<Vec<_>>();
    json!({
        "type": question.kind.name(),
        "state": state,
        "instructions": h2o_quote(profile, instructions.trim_start_matches(python_space)),
        "options": options,
    })
}

/// kev's `render`: a JSON value flattened to text, an object's keys kept as labels and a list's items as `- ` lines.
fn kev_render(value: &Value, indent: usize, out: &mut String) {
    let pad = KEV_INDENT.repeat(indent);
    match value {
        Value::Null => {}
        Value::String(text) => out.push_str(text),
        Value::Bool(b) => out.push_str(if *b { "True" } else { "False" }),
        Value::Number(n) => out.push_str(&n.to_string()),
        Value::Array(items) => {
            for (i, item) in items.iter().enumerate() {
                let mut text = String::new();
                kev_render(item, indent + 1, &mut text);
                if i > 0 {
                    out.push('\n');
                }
                out.push_str(&pad);
                out.push_str("- ");
                out.push_str(text.trim_start_matches([' ', '\t', '\n', '\r']));
            }
        }
        Value::Object(map) => {
            for (i, (key, item)) in map.iter().enumerate() {
                let nested = item.is_object() || item.is_array();
                if i > 0 {
                    out.push('\n');
                }
                out.push_str(&pad);
                out.push_str(key);
                out.push_str(if nested { ":\n" } else { ": " });
                kev_render(item, if nested { indent + 1 } else { 0 }, out);
            }
        }
    }
}

/// kev's text input, with `<|name|>` spellings broken (`<\u{a6}name\u{a6}>`) so they stay text.
fn kev_text(value: &Value) -> String {
    let mut text = String::new();
    kev_render(value, 0, &mut text);
    escape_special_tokens(&text)
}

fn escape_special_tokens(text: &str) -> String {
    let is_name = |c: char| c.is_ascii_alphanumeric() || c == '_';
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("<|") {
        let after = &rest[start + 2..];
        let name_len = after.find(|c: char| !is_name(c)).unwrap_or(after.len());
        if name_len > 0 && after[name_len..].starts_with("|>") {
            out.push_str(&rest[..start]);
            out.push_str("<\u{a6}");
            out.push_str(&after[..name_len]);
            out.push_str("\u{a6}>");
            rest = &after[name_len + 2..];
        } else {
            out.push_str(&rest[..start + 1]);
            rest = &rest[start + 1..];
        }
    }
    out.push_str(rest);
    out
}

/// The options in the order a variant shows them (the second reversed), each with its label's text.
fn render_options(profile: &LabelDecision, question: &LabelQuestion, variant: usize) -> Value {
    let n = question.options.len();
    Value::Array(
        (0..n)
            .map(|i| {
                let (key, description) =
                    &question.options[if variant == 0 { i } else { n - 1 - i }];
                let mut option = match profile.family.renders_text() {
                    true => json!({
                        "key": kev_text(&json!(key)),
                        "description": match description {
                            Value::Null => Value::Null,
                            _ => json!(kev_text(description)),
                        },
                    }),
                    false => json!({"key": key, "description": description}),
                };
                if let Some(label) = question.label_texts.get(i) {
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
    let mut input = match profile.family.prepares_texts() {
        true => h2o_input(profile, state, question),
        false => json!({
            "id": question.id,
            "type": question.kind.name(),
            "instructions": question.instructions,
            "state": state,
            "options": render_options(profile, question, variant),
        }),
    };
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
    if profile.family.renders_text() {
        input["state"] = json!(kev_text(state));
        input["instructions"] = json!(kev_text(&question.instructions));
    }
    input["images"] = json!([]);
    render_template(profile, &input)
}

/// The profile's `systemone` template rendered over `input`, with llama.cpp's filters.
fn render_template(profile: &LabelDecision, input: &Value) -> Result<String, minijinja::Error> {
    let mut env = minijinja::Environment::new();
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    // llama.cpp's tojson, which these templates were converted for: json.dumps's layout and float repr, no ASCII
    // escaping unless asked
    env.add_filter(
        "tojson",
        |value: minijinja::Value, kwargs: minijinja::value::Kwargs| {
            let value = serde_json::to_value(&value).map_err(|e| {
                minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, e.to_string())
            })?;
            let indent = kwargs.get::<Option<usize>>("indent")?;
            let separators = kwargs
                .get::<Option<Vec<String>>>("separators")?
                .unwrap_or_default();
            let layout = PythonJson {
                indent,
                item_sep: separators
                    .first()
                    .map_or(if indent.is_some() { "," } else { ", " }, String::as_str),
                key_sep: separators.get(1).map_or(": ", String::as_str),
                ensure_ascii: kwargs.get::<Option<bool>>("ensure_ascii")?.unwrap_or(false),
                python_floats: true,
            };
            kwargs.assert_all_used()?;
            Ok(minijinja::Value::from_safe_string(layout.dumps(&value)))
        },
    );
    env.add_filter("trim", |text: String| {
        text.trim_matches(TRIM_CHARS).to_string()
    });
    env.add_template(TEMPLATE, &profile.template)?;
    env.get_template(TEMPLATE)?
        .render(minijinja::Value::from_serialize(input))
}

/// `value` with `from` replaced by `to` in every string and object key.
fn replace_text(value: &Value, from: &str, to: &str) -> Value {
    match value {
        Value::String(text) => Value::String(text.replace(from, to)),
        Value::Array(items) => {
            Value::Array(items.iter().map(|v| replace_text(v, from, to)).collect())
        }
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(k, v)| (k.replace(from, to), replace_text(v, from, to)))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// What a piece of clef's prompt is: text between spans, a question's span, or an option's span.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Piece {
    Text,
    Question(QuestionKind),
    Option,
}

/// clef's one prompt for every question, split at its piece markers into each piece's text and role.
fn joint_pieces(
    profile: &LabelDecision,
    state: &Value,
    questions: &[LabelQuestion],
) -> Result<Vec<(String, Piece)>, EmbeddingError> {
    let input = json!({
        "state": state,
        "questions": questions
            .iter()
            .map(|q| json!({
                "id": q.id,
                "type": q.kind.name(),
                "instructions": q.instructions,
                "options": q.options
                    .iter()
                    .map(|(key, description)| json!({"key": key, "description": description}))
                    .collect::<Vec<_>>(),
            }))
            .collect::<Vec<_>>(),
    });
    // the template is given sorted keys, and no marker of its own in the input
    let mut input = replace_text(&sort_keys(&input), CLEF_MARKER, CLEF_MARKER_ESCAPED);
    input["images"] = json!([]);
    input["sep"] = json!(CLEF_SEP);
    input["mark_question"] = json!(CLEF_MARK_QUESTION);
    input["mark_option"] = json!(CLEF_MARK_OPTION);
    let prompt = render_template(profile, &input).map_err(|e| {
        invalid(
            format!("rendering the decision prompt failed: {e}"),
            "questions".into(),
        )
    })?;
    let mut kinds = questions.iter().map(|q| q.kind);
    let mut pieces = Vec::new();
    for piece in prompt.split(CLEF_SEP) {
        let (text, role) = if let Some(text) = piece.strip_prefix(CLEF_MARK_QUESTION) {
            let kind = kinds.next().ok_or_else(layout_error)?;
            (text, Piece::Question(kind))
        } else if let Some(text) = piece.strip_prefix(CLEF_MARK_OPTION) {
            (text, Piece::Option)
        } else {
            (piece, Piece::Text)
        };
        pieces.push((text.to_string(), role));
    }
    Ok(pieces)
}

fn layout_error() -> EmbeddingError {
    EmbeddingError::Internal(anyhow::anyhow!("unexpected layout of the decision prompt"))
}

/// clef: the questions decided together, one prompt and one pass, then the joint head over the prompt's hidden states.
async fn decide_joint(
    state: &SharedInferenceRsState,
    profile: &LabelDecision,
    request_state: &Value,
    questions: &[LabelQuestion],
    model: Option<&str>,
) -> Result<(Vec<Vec<f32>>, usize), EmbeddingError> {
    // the model was trained with the pieces tokenized one by one
    let mut layout = JointLayout::default();
    for (text, role) in joint_pieces(profile, request_state, questions)? {
        let tokens = tokenize(state, model, text).await?;
        let span = layout.tokens.len()..layout.tokens.len() + tokens.len();
        if role != Piece::Text && span.is_empty() {
            return Err(invalid(
                "the instructions and the options of a question must not be empty",
                "questions".into(),
            ));
        }
        match role {
            Piece::Question(kind) => layout.questions.push((kind, span)),
            // an option belongs to the question before it; a prompt of one question may list them first
            Piece::Option => {
                let question = match questions.len() {
                    1 => 0,
                    _ => layout
                        .questions
                        .len()
                        .checked_sub(1)
                        .ok_or_else(layout_error)?,
                };
                layout.options.push((question, span));
            }
            Piece::Text => {}
        }
        layout.tokens.extend(tokens);
    }
    let n_options = questions.iter().map(|q| q.options.len()).sum::<usize>();
    if layout.questions.len() != questions.len() || layout.options.len() != n_options {
        return Err(layout_error());
    }
    let n_tokens = layout.tokens.len();
    let Response::Raw { logits_chunks, .. } = run(state, model, layout.tokens.clone(), |request| {
        request.return_raw_logits = true;
    })
    .await?
    else {
        return Err(EmbeddingError::Internal(anyhow::anyhow!(POINTER_FAILED)));
    };
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
    let mut rest = scores.as_slice();
    let mut per_question = Vec::with_capacity(questions.len());
    for question in questions {
        let (own, tail) = rest.split_at(question.options.len());
        per_question.push(own.to_vec());
        rest = tail;
    }
    Ok((per_question, n_tokens))
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
    tokens: Vec<u32>,
    configure: impl FnOnce(&mut NormalRequest),
) -> Result<Response, EmbeddingError> {
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
) -> Result<(Vec<f32>, usize), EmbeddingError> {
    let tokens = tokenize(state, model, prompt).await?;
    let n_tokens = tokens.len();
    let scores = Arc::new(Mutex::new(None));
    let capture = LabelCapture {
        groups,
        scores: scores.clone(),
    };
    match run(state, model, tokens, |request| {
        request.logits_processors = Some(vec![Arc::new(capture)]);
    })
    .await?
    {
        Response::CompletionDone(_) | Response::Done(_) => {}
        _ => return Err(EmbeddingError::Internal(anyhow::anyhow!(SCORING_FAILED))),
    }
    let scores = scores
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take()
        .ok_or_else(|| EmbeddingError::Internal(anyhow::anyhow!(SCORING_FAILED)))?;
    Ok((scores, n_tokens))
}

/// The last row's query (its first `half` columns) against each marker row's key (the rest), over the key width's
/// square root.
fn pointer_scores(rows: &[Vec<f32>], half: usize, markers: &[usize]) -> Vec<f32> {
    let Some(last) = rows.last() else {
        return Vec::new();
    };
    let query = &last[..half];
    let scale = (half as f32).sqrt();
    markers
        .iter()
        .map(|&marker| {
            let key = &rows[marker][half..];
            query.iter().zip(key).map(|(q, k)| q * k).sum::<f32>() / scale
        })
        .collect()
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
    let markers = tokens
        .iter()
        .enumerate()
        .filter(|(_, token)| **token == option_end)
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    if markers.len() != n_options {
        return Err(invalid(
            "unexpected layout of the decision prompt".to_string(),
            "questions".into(),
        ));
    }
    let n_tokens = tokens.len();
    let Response::Raw { logits_chunks, .. } = run(state, model, tokens, |request| {
        request.return_raw_logits = true;
    })
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
    Ok((pointer_scores(&rows, half, &markers), n_tokens))
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
        .map(|(id, q)| parse(profile, id, q))
        .collect::<Result<Vec<_>, _>>()?;
    if request.state.is_null() && !profile.family.accepts_null_state() {
        return Err(invalid("`state` must be provided", "state".into()));
    }
    if profile.family.strict_payloads() && !is_payload(&request.state) {
        return Err(invalid(
            "`state` must be text, an object or an array",
            "state".into(),
        ));
    }
    let input_state = match profile.family.prepares_texts() {
        true => h2o_state(state, model, profile, &request.state).await?,
        false => request.state.clone(),
    };
    if profile.family.joint() {
        let (scores, input_tokens) = match profile.family.joint_per_question() {
            // one prompt each, run together
            true => {
                let runs = join_all(questions.iter().map(|question| {
                    decide_joint(
                        state,
                        profile,
                        &input_state,
                        std::slice::from_ref(question),
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
            false => decide_joint(state, profile, &input_state, &questions, model).await?,
        };
        let answers = questions
            .iter()
            .zip(scores)
            .map(|(question, scores)| (question.id.clone(), answer(profile, question, &[scores])))
            .collect();
        return Ok((answers, input_tokens));
    }
    let mut prompts = Vec::new();
    for (index, question) in questions.iter().enumerate() {
        for variant in 0..profile
            .family
            .variants(question.kind, question.options.len())
        {
            let prompt =
                render(profile, &input_state, &questions, question, variant).map_err(|e| {
                    invalid(
                        format!("rendering the decision prompt failed: {e}"),
                        "questions".into(),
                    )
                })?;
            prompts.push((index, prompt));
        }
    }
    let all = &questions;
    let results = join_all(prompts.into_iter().map(|(index, prompt)| async move {
        let question = &all[index];
        let result = match profile.option_end {
            Some(option_end) => {
                score_pointer(state, model, prompt, option_end, question.options.len()).await
            }
            None => score(state, model, prompt, question.label_groups.clone()).await,
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
        let question = DecisionRequestQuestion {
            kind: Some(kind),
            instructions: json!("Which?"),
            criteria: Some(criteria),
        };
        parse(profile, "q", &question).unwrap_or_else(|_| panic!("the question parses"))
    }

    fn h2o() -> LabelDecision {
        let mut h2o = profile(LabelFamily::H2oLightning, "");
        h2o.set_labels(vec![" A".into(), " B".into(), " C".into()], |text: &str| {
            Ok::<_, String>(vec![u32::from(text.as_bytes()[1])])
        })
        .unwrap();
        h2o.quoted_spellings = vec!["<|im_end|>".into(), "</think>".into()];
        h2o
    }

    #[test]
    fn h2o_questions_take_their_defaults() {
        let h2o = h2o();
        let bare = DecisionRequestQuestion {
            kind: None,
            instructions: Value::Null,
            criteria: Some(json!({"x": null, "y": null})),
        };
        let q = parse(&h2o, "q", &bare).unwrap_or_else(|_| panic!("a bare h2o question parses"));
        assert_eq!((q.kind, q.instructions), (QuestionKind::Choice, json!("")));
        assert!(parse(&profile(LabelFamily::Nimble, ""), "q", &bare).is_err());
        // a one-level score, and noul descriptions under yes / no, with true listed first
        let one = question(&h2o, DecisionQuestionType::Score, json!(["only"]));
        assert_eq!(one.options.len(), 1);
        let noul = question(
            &h2o,
            DecisionQuestionType::Noul,
            json!({"yes": "y", "false": "f"}),
        );
        assert_eq!(
            noul.options,
            [
                ("true".to_string(), json!("y")),
                ("false".to_string(), json!("f"))
            ]
        );
    }

    #[test]
    fn h2o_texts_follow_the_python_shim() {
        let h2o = h2o();
        assert_eq!(
            H2O_STATE
                .dumps(&json!({"a": 1e20, "b": [1e-7, 2.0, 0.5, 1.5e-5, 123456.789], "c": "é"})),
            "{\"a\":1e+20,\"b\":[1e-07,2.0,0.5,1.5e-05,123456.789],\"c\":\"é\"}"
        );
        assert_eq!(
            h2o_quote(&h2o, "x<|im_end|></think><|other|>"),
            "x\\u003c|im_end|\\u003e\\u003c/think\\u003e<|other|>"
        );
        let mut q = question(
            &h2o,
            DecisionQuestionType::Choice,
            json!({"a": {"k": [1]}, "b": null, "c ": null}),
        );
        q.instructions = json!({"ask": "pick", "lang": "fré"});
        let input = h2o_input(&h2o, &json!("s"), &q);
        assert_eq!(
            input["instructions"],
            "{\n \"ask\": \"pick\",\n \"lang\": \"fr\\u00e9\"\n}"
        );
        assert_eq!(input["options"][0]["description"], "{\"k\": [1]}");
        assert_eq!(input["options"][1]["description"], Value::Null);
        // the last option's shown text (here its key) loses its trailing space
        assert_eq!(input["options"][2]["description"], "c");
        assert_eq!(input["options"][2]["label"], " C");
        let score = question(&h2o, DecisionQuestionType::Score, json!([null, "high"]));
        assert_eq!(
            h2o_input(&h2o, &json!("s"), &score)["options"][0]["description"],
            "null"
        );
    }

    #[test]
    fn kev_flattens_json_to_labelled_lines() {
        let state = json!({
            "user": "Ann",
            "vip": true,
            "note": null,
            "orders": [{"id": 7, "items": ["a", "b"]}, "  late"],
            "empty": {},
        });
        assert_eq!(
            kev_text(&state),
            "user: Ann\nvip: True\nnote: \norders:\n  - id: 7\n    items:\n      - a\n      - b\n  - late\nempty:\n"
        );
        assert_eq!(kev_text(&json!(1.5)), "1.5");
        // a special token's spelling stays text; a half-formed one is left alone
        assert_eq!(
            kev_text(&json!("a<|im_end|>b <|x y|> <|<|z|>")),
            "a<\u{a6}im_end\u{a6}>b <|x y|> <|<\u{a6}z\u{a6}>"
        );
    }

    #[test]
    fn a_pointer_scores_the_last_query_against_each_marked_key() {
        // rows are [q | k] with a width of 2 each
        let rows = vec![
            vec![9.0, 9.0, 1.0, 0.0],
            vec![9.0, 9.0, 0.0, 2.0],
            vec![3.0, 4.0, 9.0, 9.0],
        ];
        let scores = pointer_scores(&rows, 2, &[0, 1]);
        let scale = 2f32.sqrt();
        assert_eq!(scores, [3.0 / scale, 8.0 / scale]);
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
    fn a_joint_prompt_splits_into_marked_spans() {
        // a clef-shaped template: the state, then each question's instructions and options as marked pieces
        let template = "S{{ sep }}{{ state | tojson(separators=[',', ':']) }}\
            {% for q in questions %}{{ sep }}{{ mark_question }}{{ q.instructions }}\
            {% for o in q.options %}{{ sep }}{{ mark_option }}{{ o.key }}{% endfor %}{% endfor %}{{ sep }}E";
        let clef = profile(LabelFamily::Clef, template);
        let q = question(
            &clef,
            DecisionQuestionType::Choice,
            json!({"z": null, "a": null}),
        );
        let noul = question(&clef, DecisionQuestionType::Noul, Value::Null);
        let pieces = joint_pieces(&clef, &json!({"y": "<<clef:sep>>", "b": 1}), &[q, noul])
            .unwrap_or_else(|_| panic!("the prompt renders"));
        let shown = pieces
            .iter()
            .map(|(text, role)| (text.as_str(), *role))
            .collect::<Vec<_>>();
        assert_eq!(
            shown,
            [
                ("S", Piece::Text),
                // keys sorted, and the input's own marker made plain text
                ("{\"b\":1,\"y\":\"<<clef sep>>\"}", Piece::Text),
                ("Which?", Piece::Question(QuestionKind::Choice)),
                ("a", Piece::Option),
                ("z", Piece::Option),
                ("Which?", Piece::Question(QuestionKind::Noul)),
                ("true", Piece::Option),
                ("false", Piece::Option),
                ("E", Piece::Text),
            ]
        );
    }

    #[test]
    fn decision2_checks_payloads_and_follows_the_noul_criteria() {
        let d2 = profile(LabelFamily::VllmDecision2, "");
        let parse_with = |kind, instructions: Value, criteria: Value| {
            let question = DecisionRequestQuestion {
                kind: Some(kind),
                instructions,
                criteria: Some(criteria),
            };
            parse(&d2, "q", &question)
        };
        let two = json!({"a": null, "b": null});
        assert!(parse_with(DecisionQuestionType::Choice, json!(5), two.clone()).is_err());
        assert!(parse_with(DecisionQuestionType::Choice, json!(""), two.clone()).is_err());
        assert!(
            parse_with(
                DecisionQuestionType::Choice,
                json!("Which?"),
                json!({"a": null})
            )
            .is_err()
        );
        assert!(
            parse_with(
                DecisionQuestionType::Choice,
                json!("Which?"),
                json!({"a": 1, "b": null})
            )
            .is_err()
        );
        assert!(
            parse_with(
                DecisionQuestionType::Noul,
                json!("Is it?"),
                json!({"maybe": "x"})
            )
            .is_err()
        );
        assert!(parse_with(DecisionQuestionType::Choice, json!({"ask": "x"}), two).is_ok());
        // true first only when the criteria name it first
        let first =
            |criteria| match parse_with(DecisionQuestionType::Noul, json!("Is it?"), criteria) {
                Ok(q) => q.options[0].0.clone(),
                Err(_) => panic!("the noul parses"),
            };
        assert_eq!(first(json!({"true": "yes", "false": "no"})), "true");
        assert_eq!(first(json!({"false": "no", "true": "yes"})), "false");
    }

    #[test]
    fn tojson_takes_python_layouts() {
        let nimble = profile(
            LabelFamily::Nimble,
            "{{ state | tojson(indent=2) }}|{{ state | tojson(separators=[',', ':'], ensure_ascii=true) }}",
        );
        let q = question(&nimble, DecisionQuestionType::Choice, json!({"x": null}));
        assert_eq!(
            render(
                &nimble,
                &json!({"a": [1, 2.0], "b": {}, "c": "é"}),
                &[],
                &q,
                0
            )
            .unwrap(),
            "{\n  \"a\": [\n    1,\n    2.0\n  ],\n  \"b\": {},\n  \"c\": \"é\"\n}|{\"a\":[1,2.0],\"b\":{},\"c\":\"\\u00e9\"}"
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
