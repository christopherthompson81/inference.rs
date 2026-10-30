//! Decisions on a chat model with a GGUF decision profile: its `systemone` template renders each question, and the
//! next-token logits of one label token per option score it, as llama.cpp's label-coded decision types do; kev's
//! pointer head scores it from the hidden states of the option end tokens and the last token instead.

use inference_nn::decision::{JointLayout, LabelDecision, QuestionKind};
use inference_protocol::openai::{DecisionAnswer, DecisionQuestionType, DecisionRequestQuestion};
use serde_json::{Map, Value, json};

use crate::{FormatError, PythonJson, first_max, kind_of, required_instructions, required_kind};

const TEMPLATE: &str = "systemone";
const SCORE_LEVELS: std::ops::RangeInclusive<usize> = 2..=10;
const NOUL_KEYS: [&str; 2] = ["false", "true"];
const NOUL_TRUE: &str = "true";
const NOUL_YES: &str = "yes";
const NOUL_NO: &str = "no";
// h2o-lightning's state budget, past which its head and tail are kept around an ellipsis line
pub const H2O_MAX_STATE_TOKENS: usize = 32_000;
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
// llama.cpp's media marker, which its server makes plain text in a request's own values
pub const MEDIA_MARKER: &str = "<__media__>";
const NO_PROJECTOR: &str = "this server does not take images for decisions; load the model with its projector (`--mmproj`)";
const MEDIA_TEMPERATURE: &str = "image";
const KEV_INDENT: &str = "  ";
// clef's template splits its prompt into pieces at `sep` and marks the start of a question or option span
const CLEF_MARKER: &str = "<<clef:";
const CLEF_MARKER_ESCAPED: &str = "<<clef ";
const CLEF_SEP: &str = "<<clef:sep>>";
const CLEF_MARK_QUESTION: &str = "<<clef:question>>";
const CLEF_MARK_OPTION: &str = "<<clef:option>>";
// d1-omni's template marks its pieces so they are cut to token budgets; the input's own markers are escaped
const D1OMNI_MARKER: &str = "<<d1omni:";
const D1OMNI_MARKER_ESCAPED: &str = "<<d1omni ";
const D1OMNI_SEP: &str = "<<d1omni:sep>>";
const D1OMNI_MARK_STATE: &str = "<<d1omni:state>>";
const D1OMNI_MARK_QUESTION: &str = "<<d1omni:question>>";
const D1OMNI_MARK_OPTION: &str = "<<d1omni:option>>";
// the model's max_length; the options share up to half, at least `D1OMNI_MIN_OPTION_BUDGET`
pub const D1OMNI_MAX_TOKENS: usize = 16384;
// its image_text_length and audio_text_length: the text's room beside images or an audio clip
const D1OMNI_MAX_IMAGE_TOKENS: usize = 896;
const D1OMNI_MAX_AUDIO_TOKENS: usize = 15360;
const D1OMNI_MIN_OPTION_BUDGET: usize = 96;
const D1OMNI_OPTION_TOKENS: usize = 24;
const D1OMNI_OPTION_BASE: usize = 32;
const D1OMNI_OPTION_OVERHEAD: usize = 3;
const D1OMNI_MIN_OPTION_TOKENS: usize = 2;
const D1OMNI_MIN_QUESTION_TOKENS: usize = 16;

/// A question as the template reads it: its id, type and instructions, and each option's key and description.
pub struct LabelQuestion {
    pub id: String,
    pub kind: QuestionKind,
    pub instructions: Value,
    pub options: Vec<(String, Value)>,
    /// the text the template shows for each option's label, and the tokens whose best logit scores each output
    pub label_texts: Vec<String>,
    pub label_groups: Vec<Vec<u32>>,
}

/// Each group's best logit among `logits`, the score of the output it labels.
pub fn label_scores(logits: &[f32], groups: &[Vec<u32>]) -> Vec<f32> {
    groups
        .iter()
        .map(|group| {
            group
                .iter()
                .map(|&label| {
                    logits
                        .get(label as usize)
                        .copied()
                        .unwrap_or(f32::NEG_INFINITY)
                })
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .collect()
}

pub fn parse(
    profile: &LabelDecision,
    id: &str,
    question: &DecisionRequestQuestion,
) -> Result<LabelQuestion, FormatError> {
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
            return Err(FormatError::invalid(
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
            return Err(FormatError::invalid(
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
                let description = criteria.get(*key).or_else(|| {
                    profile
                        .family
                        .noul_aliases()
                        .then(|| criteria.get(alias))
                        .flatten()
                });
                (key.to_string(), description.cloned().unwrap_or(Value::Null))
            })
            .collect(),
        (QuestionKind::Noul, Some(_)) => {
            return Err(FormatError::invalid(
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
        return Err(FormatError::invalid(
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
                .map_err(|message| FormatError::invalid(message, param))?
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
pub fn is_payload(value: &Value) -> bool {
    matches!(value, Value::String(_) | Value::Object(_) | Value::Array(_))
}

/// vllm-decision2's checks on a question, as its reference makes them.
fn strict_question(
    id: &str,
    kind: QuestionKind,
    instructions: &Value,
    criteria: Option<&Value>,
    options: &[(String, Value)],
) -> Result<(), FormatError> {
    let fail = |message: &str| Err(FormatError::invalid(message, format!("questions.{id}")));
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
pub fn h2o_quote(profile: &LabelDecision, text: &str) -> String {
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

/// The state as h2o-lightning reads it before its token budget: compact JSON.
pub fn h2o_state_text(value: &Value) -> String {
    h2o_text(value, &H2O_STATE)
}

/// A state past h2o-lightning's budget: its head and tail around an ellipsis line.
pub fn h2o_cut(head: &str, tail: &str) -> String {
    format!("{head}{H2O_CUT}{tail}")
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

/// What every question's prompt of a request shares: the state, the questions, and the placeholder of each image.
pub struct Prompt<'a> {
    pub state: &'a Value,
    pub questions: &'a [LabelQuestion],
    pub images: Value,
}

pub fn render(
    profile: &LabelDecision,
    prompt: &Prompt<'_>,
    question: &LabelQuestion,
    variant: usize,
) -> Result<String, FormatError> {
    let (state, questions) = (prompt.state, prompt.questions);
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
    input["images"] = prompt.images.clone();
    render_template(profile, &input).map_err(render_failed)
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
pub enum Piece {
    Text,
    Question(QuestionKind),
    Option,
}

/// clef's one prompt for every question, split at its piece markers into each piece's text and role.
pub fn joint_pieces(
    profile: &LabelDecision,
    state: &Value,
    questions: &[LabelQuestion],
    images: &Value,
) -> Result<Vec<(String, Piece)>, FormatError> {
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
    input["images"] = images.clone();
    input["sep"] = json!(CLEF_SEP);
    input["mark_question"] = json!(CLEF_MARK_QUESTION);
    input["mark_option"] = json!(CLEF_MARK_OPTION);
    let prompt = render_template(profile, &input).map_err(render_failed)?;
    let mut kinds = questions.iter().map(|q| q.kind);
    let mut pieces = Vec::new();
    let mut split = prompt.split(CLEF_SEP).peekable();
    // the text before the images is tokenized with them, as llama.cpp's server does
    if let Some(placeholder) = images.get(0).and_then(Value::as_str) {
        let mut head = String::new();
        while !head.contains(placeholder) {
            head.push_str(split.next().ok_or(FormatError::Layout)?);
        }
        if head.contains(CLEF_MARKER) {
            return Err(FormatError::Layout);
        }
        pieces.push((head, Piece::Text));
    }
    for piece in split {
        let (text, role) = if let Some(text) = piece.strip_prefix(CLEF_MARK_QUESTION) {
            let kind = kinds.next().ok_or(FormatError::Layout)?;
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

/// `value` with special token spellings broken in every string and object key (d1-omni's `prompt.py` `escape`).
fn escape_values(value: &Value) -> Value {
    match value {
        Value::String(text) => Value::String(escape_special_tokens(text)),
        Value::Array(items) => Value::Array(items.iter().map(escape_values).collect()),
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(k, v)| (escape_special_tokens(k), escape_values(v)))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// The most tokens an option and the question may keep: the options share `max(96, min(24 n + 32, 16384 / 2))`, less
/// three tokens of framing each.
fn d1omni_budgets(n_options: usize, max_tokens: usize) -> (usize, usize) {
    let budget = D1OMNI_MIN_OPTION_BUDGET
        .max((D1OMNI_OPTION_TOKENS * n_options + D1OMNI_OPTION_BASE).min(max_tokens / 2));
    let option_max = D1OMNI_MIN_OPTION_TOKENS
        .max(budget.saturating_sub(D1OMNI_OPTION_OVERHEAD * n_options) / n_options.max(1));
    (option_max, D1OMNI_MIN_QUESTION_TOKENS.max(budget))
}

/// What a piece of d1-omni's prompt is: the state, cut to the room the rest leaves; text cut to `limit` tokens; or
/// text kept whole.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum D1omniPiece {
    State,
    Limited(usize),
    Whole,
}

/// d1-omni's prompt for one question, as llama.cpp's server builds it: the template's pieces, to be tokenized one by
/// one, and the most tokens the prompt may take.
pub fn d1omni_pieces(
    profile: &LabelDecision,
    request_state: &Value,
    question: &LabelQuestion,
    n_media: usize,
    is_audio: bool,
) -> Result<(Vec<(String, D1omniPiece)>, usize), FormatError> {
    let input = json!({
        "id": question.id,
        "type": question.kind.name(),
        "instructions": question.instructions,
        "state": request_state,
        "options": render_options(profile, question, 0),
    });
    let mut input = replace_text(&escape_values(&input), D1OMNI_MARKER, D1OMNI_MARKER_ESCAPED);
    // one marker a file; the media themselves go ahead of the text
    let media_markers = MEDIA_MARKER.repeat(n_media);
    input["audio"] = json!(is_audio);
    input["images"] = json!(vec![MEDIA_MARKER; n_media]);
    input["sep"] = json!(D1OMNI_SEP);
    input["mark_state"] = json!(D1OMNI_MARK_STATE);
    input["mark_question"] = json!(D1OMNI_MARK_QUESTION);
    input["mark_option"] = json!(D1OMNI_MARK_OPTION);
    let prompt = render_template(profile, &input).map_err(render_failed)?;
    // beside media the text gets the model's image or audio text length
    let max_tokens = match (n_media, is_audio) {
        (0, _) => D1OMNI_MAX_TOKENS,
        (_, true) => D1OMNI_MAX_AUDIO_TOKENS,
        (_, false) => D1OMNI_MAX_IMAGE_TOKENS,
    };
    let (option_max, question_max) = d1omni_budgets(question.options.len(), max_tokens);
    let mut split = prompt.split(D1OMNI_SEP);
    if split.next() != Some(media_markers.as_str()) {
        return Err(FormatError::Layout);
    }
    let pieces = split
        .map(|piece| {
            let (text, role) = if let Some(text) = piece.strip_prefix(D1OMNI_MARK_STATE) {
                (text, D1omniPiece::State)
            } else if let Some(text) = piece.strip_prefix(D1OMNI_MARK_QUESTION) {
                (text, D1omniPiece::Limited(question_max))
            } else if let Some(text) = piece.strip_prefix(D1OMNI_MARK_OPTION) {
                (text, D1omniPiece::Limited(option_max))
            } else {
                (piece, D1omniPiece::Whole)
            };
            (text.to_string(), role)
        })
        .collect();
    Ok((pieces, max_tokens))
}

/// d1-omni's prompt tokens from its tokenized pieces, with the position of each option's marker.
pub fn d1omni_ids(
    profile: &LabelDecision,
    pieces: Vec<(Vec<u32>, D1omniPiece)>,
    max_tokens: usize,
    n_options: usize,
) -> Result<(Vec<u32>, Vec<u32>), FormatError> {
    let (mut head, mut body, mut tail) = (Vec::new(), None, Vec::new());
    for (mut tokens, role) in pieces {
        if let D1omniPiece::Limited(limit) = role {
            tokens.truncate(limit);
        }
        match (role == D1omniPiece::State, &body) {
            (true, None) => body = Some(tokens),
            (true, Some(_)) => return Err(FormatError::Layout),
            (false, None) => head.extend(tokens),
            (false, Some(_)) => tail.extend(tokens),
        }
    }
    let mut body = body.ok_or(FormatError::Layout)?;
    body.truncate(max_tokens.saturating_sub(head.len() + tail.len()));
    let mut ids = head;
    ids.extend(body);
    ids.extend(tail);
    ids.truncate(max_tokens);
    let marker = profile.marker.ok_or(FormatError::Layout)?;
    let markers = ids
        .iter()
        .enumerate()
        .filter(|(_, token)| **token == marker)
        .map(|(i, _)| i as u32)
        .collect::<Vec<_>>();
    if markers.len() != n_options {
        return Err(FormatError::invalid(
            "the options do not fit in the context",
            "questions",
        ));
    }
    Ok((ids, markers))
}

/// clef's layout from its tokenized pieces: the prompt tokens, each question's span and each option's span.
pub fn joint_layout(
    pieces: Vec<(Vec<u32>, Piece)>,
    questions: &[LabelQuestion],
) -> Result<JointLayout, FormatError> {
    let mut layout = JointLayout::default();
    for (tokens, role) in pieces {
        let span = layout.tokens.len()..layout.tokens.len() + tokens.len();
        if role != Piece::Text && span.is_empty() {
            return Err(FormatError::invalid(
                "the instructions and the options of a question must not be empty",
                "questions",
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
                        .ok_or(FormatError::Layout)?,
                };
                layout.options.push((question, span));
            }
            Piece::Text => {}
        }
        layout.tokens.extend(tokens);
    }
    let n_options = questions.iter().map(|q| q.options.len()).sum::<usize>();
    if layout.questions.len() != questions.len() || layout.options.len() != n_options {
        return Err(FormatError::Layout);
    }
    Ok(layout)
}

/// The layout over the tokens the model read: the images sit before every span, so their expansion moves each span by
/// the same count.
pub fn shift_layout(mut layout: JointLayout, tokens: Vec<u32>) -> Result<JointLayout, FormatError> {
    let shift = tokens
        .len()
        .checked_sub(layout.tokens.len())
        .ok_or(FormatError::Layout)?;
    if shift > 0 {
        let moved = |span: &std::ops::Range<usize>| span.start + shift..span.end + shift;
        layout.questions = layout
            .questions
            .iter()
            .map(|(kind, span)| (*kind, moved(span)))
            .collect();
        layout.options = layout
            .options
            .iter()
            .map(|(q, span)| (*q, moved(span)))
            .collect();
        layout.tokens = tokens;
    }
    Ok(layout)
}

/// The joint head's option scores, split per question.
pub fn split_scores(scores: &[f32], questions: &[LabelQuestion]) -> Vec<Vec<f32>> {
    let mut rest = scores;
    let mut per_question = Vec::with_capacity(questions.len());
    for question in questions {
        let (own, tail) = rest.split_at(question.options.len());
        per_question.push(own.to_vec());
        rest = tail;
    }
    per_question
}

/// The answer from each variant's label logits: a softmax at the calibrated temperature, averaged over the variants.
pub fn answer(
    profile: &LabelDecision,
    question: &LabelQuestion,
    scores: &[Vec<f32>],
    with_media: bool,
) -> DecisionAnswer {
    let n = profile
        .family
        .outputs(question.kind, question.options.len());
    // a model may have its own temperature for requests with media
    let temperature = match profile.temperatures.get(MEDIA_TEMPERATURE) {
        // d1-omni has no calibration for media
        _ if with_media && profile.family.scores_markers() => 1.0,
        Some(&media) if with_media => f64::from(media),
        _ => f64::from(profile.temperature(question.kind, question.options.len())),
    };
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

/// The position of each option's end token, where a pointer head reads its key.
pub fn pointer_markers(
    tokens: &[u32],
    option_end: u32,
    n_options: usize,
) -> Result<Vec<usize>, FormatError> {
    let markers = tokens
        .iter()
        .enumerate()
        .filter(|(_, token)| **token == option_end)
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    if markers.len() != n_options {
        return Err(FormatError::invalid(
            "unexpected layout of the decision prompt",
            "questions",
        ));
    }
    Ok(markers)
}

/// The last row's query (its first `half` columns) against each marker row's key (the rest), over the key width's
/// square root.
pub fn pointer_scores(rows: &[Vec<f32>], half: usize, markers: &[usize]) -> Vec<f32> {
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

/// Media the model can read: a type whose prompt has a place for images, loaded with its projector.
pub fn check_media(
    profile: &LabelDecision,
    images: usize,
    audios: usize,
) -> Result<(), FormatError> {
    let fail = |message: &str| Err(FormatError::invalid(message, "files"));
    if !profile.family.takes_images() {
        return fail("this decision model's prompt has no place for images");
    }
    // d1-omni reads images or one audio clip ahead of its prompt
    if profile.family.scores_markers() {
        if audios > 0 && (audios > 1 || images > 0) {
            return fail("a request takes images or one audio clip, not both");
        }
        if profile.image_placeholder.is_none() {
            return fail(NO_PROJECTOR);
        }
        return Ok(());
    }
    if audios > 0 {
        return fail("this decision model does not take audio");
    }
    if profile.image_placeholder.is_none() {
        return fail(NO_PROJECTOR);
    }
    Ok(())
}

/// The request's own media markers made plain text, as llama.cpp's server makes them, in the state and questions.
pub fn strip_media_markers(
    state: &Value,
    questions: Vec<LabelQuestion>,
) -> (Value, Vec<LabelQuestion>) {
    let state = replace_text(state, MEDIA_MARKER, " ");
    let questions = questions
        .into_iter()
        .map(|mut q| {
            q.instructions = replace_text(&q.instructions, MEDIA_MARKER, " ");
            q.options = q
                .options
                .into_iter()
                .map(|(key, description)| {
                    (
                        key.replace(MEDIA_MARKER, " "),
                        replace_text(&description, MEDIA_MARKER, " "),
                    )
                })
                .collect();
            q
        })
        .collect();
    (state, questions)
}

fn render_failed(e: minijinja::Error) -> FormatError {
    FormatError::invalid(
        format!("rendering the decision prompt failed: {e}"),
        "questions",
    )
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use inference_nn::decision::LabelFamily;

    use super::*;

    fn plain(state: &Value) -> Prompt<'_> {
        Prompt {
            state,
            questions: &[],
            images: json!([]),
        }
    }

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
            render(&nimble, &plain(&state), &q, 0).unwrap(),
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
            render(&lev, &plain(&state), &q, 1).unwrap(),
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
        let pieces = joint_pieces(
            &clef,
            &json!({"y": "<<clef:sep>>", "b": 1}),
            &[q, noul],
            &json!([]),
        )
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
    fn d1omni_budgets_follow_llama_cpp() {
        // 3 options: max(96, 104) = 104 shared, (104 - 9) / 3 = 31 an option
        assert_eq!(d1omni_budgets(3, D1OMNI_MAX_TOKENS), (31, 104));
        // 1 option: the 96 floor, (96 - 3) / 1
        assert_eq!(d1omni_budgets(1, D1OMNI_MAX_TOKENS), (93, 96));
        // 40 options: 992 shared, (992 - 120) / 40 = 21
        assert_eq!(d1omni_budgets(40, D1OMNI_MAX_TOKENS), (21, 992));
        // 400 options: capped at half the context, (8192 - 1200) / 400 = 17
        assert_eq!(d1omni_budgets(400, D1OMNI_MAX_TOKENS), (17, 8192));
        // beside images the text has 896 tokens, the options at most half of it: (448 - 120) / 40 = 8
        assert_eq!(d1omni_budgets(40, D1OMNI_MAX_IMAGE_TOKENS), (8, 448));
    }

    #[test]
    fn media_need_a_family_with_a_place_for_them_and_a_projector() {
        assert!(check_media(&profile(LabelFamily::Nimble, ""), 1, 0).is_err());
        let mut d3 = profile(LabelFamily::VllmD3, "");
        assert!(check_media(&d3, 1, 0).is_err());
        d3.image_placeholder = Some("<img>".into());
        assert!(check_media(&d3, 1, 0).is_ok());
        assert!(check_media(&d3, 0, 1).is_err());
    }

    #[test]
    fn each_image_takes_the_model_placeholder() {
        let d3 = profile(
            LabelFamily::VllmD3,
            "{% for image in images %}{{ image }}{% endfor %}|{{ state }}",
        );
        let q = question(&d3, DecisionQuestionType::Noul, Value::Null);
        let prompt = Prompt {
            state: &json!("s"),
            questions: &[],
            images: json!(["<img>", "<img>"]),
        };
        assert_eq!(render(&d3, &prompt, &q, 0).unwrap(), "<img><img>|s");
    }

    // the text up to the images is one piece with them, tokenized together as llama.cpp's server does
    #[test]
    fn a_joint_prompt_keeps_its_images_with_the_text_before_them() {
        let template = "S{{ sep }}{% for image in images %}{{ image }}{% endfor %}\
            |{{ sep }}{{ state }}{% for q in questions %}{{ sep }}{{ mark_question }}{{ q.instructions }}\
            {% for o in q.options %}{{ sep }}{{ mark_option }}{{ o.key }}{% endfor %}{% endfor %}";
        let clef = profile(LabelFamily::Clef, template);
        let q = question(
            &clef,
            DecisionQuestionType::Choice,
            json!({"a": null, "b": null}),
        );
        let pieces = joint_pieces(&clef, &json!("s"), &[q], &json!(["<img>"]))
            .unwrap_or_else(|_| panic!("the prompt renders"));
        assert_eq!(pieces[0], ("S<img>|".to_string(), Piece::Text));
        assert_eq!(pieces[1], ("s".to_string(), Piece::Text));
        assert_eq!(pieces.len(), 5);
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
                &plain(&json!({"a": [1, 2.0], "b": {}, "c": "é"})),
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
        let answer = answer(&jev, &q, &[vec![2.0, 0.0]], false);
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
        } = answer(&lev, &q, &[vec![1.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]], false)
        else {
            panic!()
        };
        assert!((probabilities["x"] - probabilities["z"]).abs() < 1e-12);
        assert!(probabilities["x"] > probabilities["y"]);
        assert_eq!(choice, "x");
        let q = question(&lev, DecisionQuestionType::Noul, Value::Null);
        let mut ratings = vec![f32::NEG_INFINITY; 9];
        ratings[6] = 0.0;
        let DecisionAnswer::Noul { noul } = answer(&lev, &q, &[ratings], false) else {
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
        } = answer(&d3, &q, &[vec![0.0, 0.0]], false)
        else {
            panic!()
        };
        assert_eq!(legend["1"], json!("{\"a\":2,\"b\":1}"));
        assert!(even.abs() < 1e-12);
        let nimble = profile(LabelFamily::Nimble, "");
        let q = question(&nimble, DecisionQuestionType::Score, levels);
        let DecisionAnswer::Score { legend, .. } = answer(&nimble, &q, &[vec![0.0, 0.0]], false)
        else {
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
