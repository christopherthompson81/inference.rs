//! Decision models: a typed question about a state, answered by scoring one `[MASK]` marker per option in a
//! `[CLS] question [SEP] [MASK] option ... [SEP] state [SEP]` sequence (laya, Julia-1), or by a chat model's
//! next-token logits of one label per option (`LabelDecision`), or by a pointer head over a chat model's hidden states
//! (kev).
// positions within a prompt the model's budget keeps far below u32::MAX
#![allow(clippy::cast_possible_truncation)]

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// both reference implementations cut an option's text to this many tokens
const MAX_OPTION_TOKENS: usize = 48;
// the fewest question tokens the head budget keeps, and the option budget it leaves them
const MIN_QUESTION_TOKENS: usize = 8;
const MIN_HEAD_ROOM: usize = 16;
const MIN_OPTION_TOKENS: usize = 4;
const NOUL_FALSE: &str = "false";
const NOUL_TRUE: &str = "true";
// laya's noul options when the question names no criteria
const LAYA_NOUL_FALSE: &str = "no, the statement does not hold";
const LAYA_NOUL_TRUE: &str = "yes, the statement holds";
// llama.cpp reads at most this many label codes, the largest option count its requests allow
const MAX_LABELS: usize = 255;
/// kev reads an option's hidden state at the token that closes it
pub const KEV_OPTION_END: &str = "<|box_end|>";
// d1 reads a noul's answer as any of these spellings
const D1_YES: [&str; 3] = ["yes", "Yes", "YES"];
const D1_NO: [&str; 3] = ["no", "No", "NO"];
// d1's score levels, as digits
const D1_LEVELS: usize = 10;
/// lev answers a noul by rating it on this many levels
pub const LEV_NOUL_RATINGS: usize = 9;
/// Where a decoder's decision head sits in the native weights, beside `lm_head`, whose place it takes.
pub const DECISION_HEAD: &str = "decision_head";

/// The spans a joint decision head reads, as prompt token positions: each question's type and span, each option's
/// question and span, in prompt order; with the prompt's tokens, whose output embeddings the option spans also read.
#[derive(Clone, Debug, Default)]
pub struct JointLayout {
    pub tokens: Vec<u32>,
    pub questions: Vec<(QuestionKind, std::ops::Range<usize>)>,
    pub options: Vec<(usize, std::ops::Range<usize>)>,
}

/// How a checkpoint renders its options; the model and its budgets are otherwise the same.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DecisionFamily {
    /// `key: description`, `level i: description`, `false: ...` / `true: ...`
    Laya,
    /// the description alone (the key without one)
    Julia,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QuestionKind {
    Choice,
    Score,
    Noul,
}

impl QuestionKind {
    /// The row of the model's type embedding.
    pub fn index(self) -> u32 {
        match self {
            Self::Choice => 0,
            Self::Score => 1,
            Self::Noul => 2,
        }
    }

    pub fn from_index(index: u32) -> Option<Self> {
        [Self::Choice, Self::Score, Self::Noul]
            .into_iter()
            .find(|kind| kind.index() == index)
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Choice => "choice",
            Self::Score => "score",
            Self::Noul => "noul",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecisionOption {
    pub key: String,
    pub description: Option<String>,
}

/// A question as the model reads it: noul's options are `[false, true]`, so the second probability is the answer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecisionQuestion {
    pub kind: QuestionKind,
    pub instructions: String,
    pub options: Vec<DecisionOption>,
}

/// The tokenizer's specials the sequence is built from, with the marker's text, which request text must not carry.
#[derive(Clone, Debug)]
pub struct DecisionTokens {
    pub cls: u32,
    pub sep: u32,
    pub mask: u32,
    pub mask_text: String,
}

/// One question's model input: its tokens, where each option's marker sits, and its type embedding row.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionPrompt {
    pub ids: Vec<u32>,
    pub markers: Vec<u32>,
    pub kind: QuestionKind,
}

/// A decision checkpoint's budgets and calibration, from the `decision` section of its merged config.
#[derive(Clone, Debug, Deserialize)]
pub struct DecisionSpec {
    pub family: DecisionFamily,
    /// the tokenizer's texts of the specials the sequence is built from
    pub cls_token: String,
    pub sep_token: String,
    pub mask_token: String,
    pub head_layers: usize,
    pub max_len: usize,
    pub head_max_len: usize,
    /// per question type (choice, score, noul); empty for an uncalibrated model
    #[serde(default)]
    pub temperature: Vec<f32>,
    /// `<type>:<2|3-5|6-10|11+>`, which take precedence over the per-type ones
    #[serde(default)]
    pub temperature_by_options: HashMap<String, f32>,
}

#[derive(Deserialize)]
struct WithDecision {
    #[serde(default)]
    decision: Option<DecisionSpec>,
}

/// Each row's question type and option markers for a decision forward.
#[derive(Clone, Debug)]
pub struct DecisionRows {
    pub kinds: Vec<u32>,
    pub markers: Vec<Vec<u32>>,
}

impl DecisionSpec {
    pub fn from_config(config: &str) -> serde_json::Result<Option<Self>> {
        Ok(serde_json::from_str::<WithDecision>(config)?.decision)
    }

    /// The text after each option's marker, in the order the answer's probabilities take.
    pub fn option_texts(&self, question: &DecisionQuestion) -> Vec<String> {
        match (self.family, question.kind) {
            (DecisionFamily::Laya, QuestionKind::Choice) => question
                .options
                .iter()
                .map(|option| match described(option) {
                    Some(description) => format!("{}: {description}", option.key),
                    None => option.key.clone(),
                })
                .collect(),
            (DecisionFamily::Laya, QuestionKind::Score) => question
                .options
                .iter()
                .enumerate()
                .map(|(level, option)| {
                    format!(
                        "level {level}: {}",
                        option.description.as_deref().unwrap_or_default()
                    )
                })
                .collect(),
            (DecisionFamily::Laya, QuestionKind::Noul) => {
                let side = |key, default| match noul_description(question, key) {
                    Some(description) => format!("{key}: {description}"),
                    None => format!("{key}: {default}"),
                };
                vec![
                    side(NOUL_FALSE, LAYA_NOUL_FALSE),
                    side(NOUL_TRUE, LAYA_NOUL_TRUE),
                ]
            }
            (DecisionFamily::Julia, QuestionKind::Noul) => [NOUL_FALSE, NOUL_TRUE]
                .map(|key| noul_description(question, key).unwrap_or(key).to_string())
                .to_vec(),
            (DecisionFamily::Julia, _) => question
                .options
                .iter()
                .map(|option| described(option).unwrap_or(&option.key).to_string())
                .collect(),
        }
    }

    /// laya's calibration: by question type and option count, then by type alone; 1 without either.
    pub fn temperature(&self, kind: QuestionKind, n_options: usize) -> f32 {
        let bucket = match n_options {
            0..=2 => "2",
            3..=5 => "3-5",
            6..=10 => "6-10",
            _ => "11+",
        };
        self.temperature_by_options
            .get(&format!("{}:{bucket}", kind.name()))
            .or_else(|| self.temperature.get(kind.index() as usize))
            .copied()
            .unwrap_or(1.0)
    }

    /// The question's sequence within the head and total budgets, as laya's `build_sequence` and Julia-1's
    /// `sequence` build it; `None` when an option's marker falls past the total budget.
    pub fn prompt<E>(
        &self,
        question: &DecisionQuestion,
        state: &str,
        tokens: &DecisionTokens,
        mut encode: impl FnMut(&str) -> Result<Vec<u32>, E>,
    ) -> Result<Option<DecisionPrompt>, E> {
        // an empty pattern would match between every character
        let clean = |text: &str| match tokens.mask_text.is_empty() {
            true => text.to_string(),
            false => text.replace(&tokens.mask_text, " "),
        };
        let head = encode(&format!(
            "{} question: {}",
            question.kind.name(),
            clean(&question.instructions)
        ))?;
        let mut options = self
            .option_texts(question)
            .iter()
            .map(|text| {
                let mut ids = vec![tokens.mask];
                ids.extend(
                    encode(&format!(" {}", clean(text)))?
                        .into_iter()
                        .take(MAX_OPTION_TOKENS),
                );
                Ok(ids)
            })
            .collect::<Result<Vec<_>, E>>()?;
        let used = |options: &[Vec<u32>]| options.iter().map(Vec::len).sum::<usize>();
        // too many or too long options: every option text shrinks alike
        if used(&options) + MIN_HEAD_ROOM > self.head_max_len {
            let per = MIN_OPTION_TOKENS
                .max(self.head_max_len.saturating_sub(MIN_HEAD_ROOM) / options.len().max(1));
            for option in &mut options {
                option.truncate(per);
            }
        }
        let question_tokens =
            MIN_QUESTION_TOKENS.max(self.head_max_len.saturating_sub(used(&options)));
        let mut ids = vec![tokens.cls];
        ids.extend(head.into_iter().take(question_tokens));
        ids.push(tokens.sep);
        let mut markers = Vec::with_capacity(options.len());
        for option in options {
            markers.push(ids.len() as u32);
            ids.extend(option);
        }
        ids.push(tokens.sep);
        let room = self.max_len.saturating_sub(ids.len() + 1);
        ids.extend(encode(&clean(state))?.into_iter().take(room));
        ids.push(tokens.sep);
        ids.truncate(self.max_len);
        if markers
            .last()
            .is_some_and(|&marker| marker as usize >= self.max_len)
        {
            return Ok(None);
        }
        Ok(Some(DecisionPrompt {
            ids,
            markers,
            kind: question.kind,
        }))
    }
}

/// How a chat model reads a decision from its next-token logits, one label token per option (llama.cpp's
/// label-coded decision types).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LabelFamily {
    OpenJev,
    Lev,
    Nimble,
    PplxDecider,
    VllmD3,
    /// a pointer head: the last token's query against each option's end-token key
    Kev,
    /// labels from the GGUF (`<arch>.decision.labels`), read after an answer prefill
    H2oLightning,
    /// labels chosen per question (codes by question type), each scored by its best spelling
    Lfm2D1,
    /// one prompt for every question, scored together by a joint head over the question and option spans
    Clef,
}

impl LabelFamily {
    /// The `<arch>.decision.type` a GGUF names the family by.
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "openjev" => Self::OpenJev,
            "lev" => Self::Lev,
            "nimble" => Self::Nimble,
            "pplx-decider" => Self::PplxDecider,
            "vllm-d3" => Self::VllmD3,
            "kev" => Self::Kev,
            "h2o-lightning" => Self::H2oLightning,
            "lfm2-d1" => Self::Lfm2D1,
            "clef" => Self::Clef,
            _ => return None,
        })
    }

    /// The family's label texts in option order; openjev's letters must all be single tokens, the others keep the
    /// codes that are.
    fn label_codes(self) -> (Vec<String>, bool) {
        match self {
            Self::OpenJev => (
                ('A'..='Z')
                    .chain('a'..='z')
                    .map(|c| c.to_string())
                    .collect(),
                true,
            ),
            Self::Kev | Self::H2oLightning | Self::Lfm2D1 | Self::Clef => (Vec::new(), false),
            _ => {
                let letters = || 'A'..='Z';
                let codes = letters()
                    .map(|a| a.to_string())
                    .chain(letters().flat_map(move |a| letters().map(move |b| format!("{a}{b}"))));
                (codes.collect(), false)
            }
        }
    }

    /// noul's options as the template lists them: `[true, false]` for openjev, `[false, true]` otherwise.
    pub fn noul_true_first(self) -> bool {
        matches!(
            self,
            Self::OpenJev | Self::H2oLightning | Self::Lfm2D1 | Self::Clef
        )
    }

    /// All the questions share one prompt, which a joint head scores at once (clef).
    pub fn joint(self) -> bool {
        self == Self::Clef
    }

    /// A choice's options are read sorted by key.
    pub fn sorts_choices(self) -> bool {
        self == Self::Clef
    }

    /// The model's output is a decision head, so it answers decisions only.
    pub fn decision_only(self) -> bool {
        matches!(self, Self::Kev | Self::Clef)
    }

    /// The labels follow each question's type and keys (d1's `prompt.py`), so they are chosen per question.
    pub fn labels_per_question(self) -> bool {
        self == Self::Lfm2D1
    }

    /// The prompt reads without a state, for example to ask about images only.
    pub fn accepts_null_state(self) -> bool {
        matches!(self, Self::VllmD3 | Self::Lfm2D1)
    }

    /// A score's legend reports a non-text description as its key-sorted JSON.
    pub fn canonical_legend(self) -> bool {
        self == Self::VllmD3
    }

    /// The template reads each option's label text (openjev's letters are implied by position).
    pub fn shows_labels(self) -> bool {
        self != Self::OpenJev
    }

    /// The template lists every question of the request, not only the asked one.
    pub fn lists_questions(self) -> bool {
        self == Self::Nimble
    }

    /// The template input's object keys are sorted, as the model was trained.
    pub fn sorts_keys(self) -> bool {
        self == Self::Lev
    }

    /// A choice runs once per option order (lev: as given and reversed), to cancel the preference for the first label.
    pub fn variants(self, kind: QuestionKind, n_options: usize) -> usize {
        match (self, kind) {
            (Self::Lev, QuestionKind::Choice) if n_options > 1 => 2,
            _ => 1,
        }
    }

    /// The labels read for a question: lev rates a noul on a scale at its first labels.
    pub fn outputs(self, kind: QuestionKind, n_options: usize) -> usize {
        match (self, kind) {
            (Self::Lev, QuestionKind::Noul) => LEV_NOUL_RATINGS,
            _ => n_options,
        }
    }

    /// The checkpoint reads its prompt with non-causal full attention (`noncausal_full_attention` in its decision
    /// config), whether or not its GGUF says so.
    pub fn non_causal(self) -> bool {
        matches!(self, Self::PplxDecider | Self::VllmD3)
    }

    /// The model's output rows are a decision head's `[q | k]`, read at the option end tokens and the last token, not
    /// label logits.
    pub fn reads_pointer(self) -> bool {
        self == Self::Kev
    }

    /// The labels are the GGUF's `<arch>.decision.labels`, each one distinct token.
    pub fn labels_from_metadata(self) -> bool {
        self == Self::H2oLightning
    }

    /// The template takes texts as h2o-lightning's shim writes them: the state as compact JSON cut to a token budget,
    /// the other values as `json.dumps` writes them, special token spellings quoted. Its questions may leave out the
    /// type (a choice) and the instructions, a score may have one level, and a noul's descriptions may sit under
    /// `yes` and `no`.
    pub fn prepares_texts(self) -> bool {
        self == Self::H2oLightning
    }

    /// The prompt has a default question, so the instructions may be left out.
    pub fn default_question(self) -> bool {
        matches!(self, Self::VllmD3 | Self::H2oLightning)
    }

    /// The template takes the state, instructions and option texts flattened to text (kev's `render`).
    pub fn renders_text(self) -> bool {
        self == Self::Kev
    }

    /// The confidence is one minus the normalized entropy, not TypeSafe's per-type formulas.
    pub fn entropy_confidence(self) -> bool {
        self == Self::VllmD3
    }
}

/// A label-coded decision model's prompt template and calibration, from its GGUF metadata.
#[derive(Clone, Debug)]
pub struct LabelDecision {
    pub family: LabelFamily,
    /// the `systemone` chat template, rendered with the question, its options and the state
    pub template: String,
    /// by `<type>.<bucket>` or `<type>`
    pub temperatures: HashMap<String, f32>,
    /// by level count, added to a score's logits
    pub score_bias: HashMap<usize, Vec<f32>>,
    /// the token read for each option, in order, and the text the template shows for it
    pub labels: Vec<u32>,
    pub label_texts: Vec<String>,
    /// the token that ends each option, where a pointer head reads it
    pub option_end: Option<u32>,
    /// the texts of the tokenizer's added tokens, longest first, which h2o-lightning quotes in its input
    pub quoted_spellings: Vec<String>,
    /// the codes a per-question family may label with (`d1_codes`) that are one token, with and without a leading
    /// space
    pub single_tokens: HashMap<String, u32>,
}

impl LabelDecision {
    /// The labels from the family's codes; `encode` tokenizes a code without specials.
    pub fn new<E: std::fmt::Display>(
        family: LabelFamily,
        template: String,
        temperatures: HashMap<String, f32>,
        score_bias: HashMap<usize, Vec<f32>>,
        mut encode: impl FnMut(&str) -> Result<Vec<u32>, E>,
    ) -> Result<Self, String> {
        let (codes, all_single) = family.label_codes();
        let mut labels = Vec::new();
        let mut label_texts = Vec::new();
        for code in codes {
            match encode(&code).map_err(|e| e.to_string())?.as_slice() {
                [token] if labels.len() < MAX_LABELS => {
                    labels.push(*token);
                    label_texts.push(code);
                }
                _ if all_single => {
                    return Err(format!("decision label `{code}` is not a single token"));
                }
                _ => {}
            }
        }
        if !family.shows_labels() {
            label_texts.clear();
        }
        Ok(Self {
            family,
            template,
            temperatures,
            score_bias,
            labels,
            label_texts,
            option_end: None,
            quoted_spellings: Vec::new(),
            single_tokens: HashMap::new(),
        })
    }

    /// d1's question labels: each label's text and the tokens that spell it, whose best logit scores it. A score
    /// reads its level digits, a noul `yes` / `no` in three cases; a choice its single-letter keys, else `A`, `B`, ...
    /// (or `00`, `01`, ... past 26), each falling back to the first unused code of the pool when it is not one token.
    pub fn question_labels(
        &self,
        kind: QuestionKind,
        keys: &[&str],
    ) -> Result<(Vec<String>, Vec<Vec<u32>>), String> {
        let single = |forms: &[String]| {
            let mut out = Vec::new();
            for form in forms {
                if let Some(&token) = self.single_tokens.get(form)
                    && !out.contains(&token)
                {
                    out.push(token);
                }
            }
            out
        };
        if kind != QuestionKind::Choice {
            let mut texts = Vec::with_capacity(keys.len());
            let mut groups = Vec::with_capacity(keys.len());
            for &key in keys {
                let forms = match kind {
                    QuestionKind::Score => vec![key.to_string()],
                    _ if key == NOUL_TRUE => D1_YES.map(String::from).to_vec(),
                    _ => D1_NO.map(String::from).to_vec(),
                };
                let group = single(&forms);
                if group.is_empty() {
                    return Err(format!("decision label is not a single token: {key}"));
                }
                texts.push(key.to_string());
                groups.push(group);
            }
            return Ok((texts, groups));
        }
        let letters = keys
            .iter()
            .all(|key| key.len() == 1 && key.as_bytes()[0].is_ascii_alphabetic());
        let pool = d1_codes();
        let mut used = Vec::new();
        let mut texts = Vec::with_capacity(keys.len());
        let mut groups = Vec::with_capacity(keys.len());
        for (i, key) in keys.iter().enumerate() {
            let code = match (letters, keys.len()) {
                (true, _) => key.to_string(),
                (false, 0..=26) => char::from(b'A' + i as u8).to_string(),
                _ => format!("{i:02}"),
            };
            let taken = std::iter::once(&code).chain(&pool).find_map(|code| {
                let &token = self.single_tokens.get(code)?;
                if used.contains(&token) {
                    return None;
                }
                let mut group = vec![token];
                group.extend(
                    single(&[format!(" {code}")])
                        .into_iter()
                        .filter(|&t| t != token),
                );
                Some((code.clone(), token, group))
            });
            let Some((code, token, group)) = taken else {
                return Err(format!(
                    "no single-token label left for {} options",
                    keys.len()
                ));
            };
            used.push(token);
            texts.push(code);
            groups.push(group);
        }
        Ok((texts, groups))
    }

    /// The GGUF's own labels in option order, each of which must be one distinct token.
    pub fn set_labels<E: std::fmt::Display>(
        &mut self,
        texts: Vec<String>,
        mut encode: impl FnMut(&str) -> Result<Vec<u32>, E>,
    ) -> Result<(), String> {
        let mut labels = Vec::with_capacity(texts.len());
        for text in &texts {
            match encode(text).map_err(|e| e.to_string())?.as_slice() {
                [token] if !labels.contains(token) => labels.push(*token),
                _ => {
                    return Err(format!(
                        "decision label `{text}` is not a single distinct token"
                    ));
                }
            }
        }
        if labels.is_empty() {
            return Err("decision model has no labels".to_string());
        }
        self.labels = labels;
        self.label_texts = texts;
        Ok(())
    }

    /// The most options a question can have: one per label, or llama.cpp's request limit for a pointer head.
    pub fn max_options(&self) -> usize {
        match self.family.reads_pointer()
            || self.family.labels_per_question()
            || self.family.joint()
        {
            true => MAX_LABELS,
            false => self.labels.len(),
        }
    }

    /// The temperature for a question type and option count, in the buckets it was fit over; 1 without one.
    pub fn temperature(&self, kind: QuestionKind, n_options: usize) -> f32 {
        let bucket = match (self.family, n_options) {
            (LabelFamily::Lev, 0..=8) => "small",
            (LabelFamily::Lev, 9..=26) => "mid",
            (LabelFamily::Lev, _) => "large",
            (_, 0..=2) => "2",
            (_, 3..=5) => "3_5",
            (_, 6..=10) => "6_10",
            _ => "11",
        };
        self.temperatures
            .get(&format!("{}.{bucket}", kind.name()))
            .or_else(|| self.temperatures.get(kind.name()))
            .copied()
            .unwrap_or(1.0)
    }
}

/// d1's label pool in the order it is tried: `A`..`Z`, `00`..`99`, `a`..`z`, `#0`..`#199`, `AA`..`ZZ`.
fn d1_codes() -> Vec<String> {
    let upper = || ('A'..='Z').map(String::from);
    upper()
        .chain((0..100).map(|i| format!("{i:02}")))
        .chain(('a'..='z').map(String::from))
        .chain((0..200).map(|i| format!("#{i}")))
        .chain(('A'..='Z').flat_map(|a| ('A'..='Z').map(move |b| format!("{a}{b}"))))
        .collect()
}

/// Every text d1's labels may be read as: the pool, the score digits and the noul words, each also after a space.
pub fn d1_label_forms() -> Vec<String> {
    let bare = d1_codes()
        .into_iter()
        .chain((0..D1_LEVELS).map(|i| i.to_string()))
        .chain(D1_YES.iter().chain(&D1_NO).map(|word| word.to_string()))
        .collect::<Vec<_>>();
    let spaced = bare
        .iter()
        .map(|text| format!(" {text}"))
        .collect::<Vec<_>>();
    bare.into_iter().chain(spaced).collect()
}

fn described(option: &DecisionOption) -> Option<&str> {
    option
        .description
        .as_deref()
        .filter(|description| !description.is_empty())
}

fn noul_description<'a>(question: &'a DecisionQuestion, key: &str) -> Option<&'a str> {
    question
        .options
        .iter()
        .find(|option| option.key == key)
        .and_then(|option| option.description.as_deref())
        .filter(|description| !description.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    const MASK: &str = "[MASK]";

    fn tokens() -> DecisionTokens {
        DecisionTokens {
            cls: 1,
            sep: 2,
            mask: 3,
            mask_text: MASK.into(),
        }
    }

    fn spec(family: DecisionFamily, head_max_len: usize, max_len: usize) -> DecisionSpec {
        DecisionSpec {
            family,
            cls_token: "[CLS]".into(),
            sep_token: "[SEP]".into(),
            mask_token: MASK.into(),
            head_layers: 2,
            max_len,
            head_max_len,
            temperature: vec![2.0, 3.0, 4.0],
            temperature_by_options: HashMap::from([("choice:3-5".to_string(), 1.5)]),
        }
    }

    fn option(key: &str, description: Option<&str>) -> DecisionOption {
        DecisionOption {
            key: key.into(),
            description: description.map(Into::into),
        }
    }

    // one token per character, ids from 10, so lengths are easy to follow
    fn encode(text: &str) -> Result<Vec<u32>, ()> {
        Ok(text.chars().map(|c| 10 + c as u32).collect())
    }

    #[test]
    fn each_family_renders_its_options() {
        let choice = DecisionQuestion {
            kind: QuestionKind::Choice,
            instructions: "Which team?".into(),
            options: vec![option("billing", None), option("shipping", Some("parcels"))],
        };
        let noul = DecisionQuestion {
            kind: QuestionKind::Noul,
            instructions: "Angry?".into(),
            options: vec![option("true", Some("yes, clearly"))],
        };
        let laya = spec(DecisionFamily::Laya, 192, 512);
        let julia = spec(DecisionFamily::Julia, 256, 8192);
        assert_eq!(laya.option_texts(&choice), ["billing", "shipping: parcels"]);
        assert_eq!(julia.option_texts(&choice), ["billing", "parcels"]);
        assert_eq!(
            laya.option_texts(&noul),
            [
                "false: no, the statement does not hold",
                "true: yes, clearly"
            ]
        );
        assert_eq!(julia.option_texts(&noul), ["false", "yes, clearly"]);
    }

    #[test]
    fn a_prompt_places_one_marker_per_option_before_the_state() {
        let question = DecisionQuestion {
            kind: QuestionKind::Score,
            instructions: "x".into(),
            options: vec![option("0", Some("lo")), option("1", Some("hi"))],
        };
        let prompt = spec(DecisionFamily::Julia, 64, 128)
            .prompt(&question, "st", &tokens(), encode)
            .unwrap()
            .unwrap();
        let head = encode("score question: x").unwrap();
        let mut expected = vec![tokens().cls];
        expected.extend(&head);
        expected.push(tokens().sep);
        let first = expected.len() as u32;
        expected.push(tokens().mask);
        expected.extend(encode(" lo").unwrap());
        let second = expected.len() as u32;
        expected.push(tokens().mask);
        expected.extend(encode(" hi").unwrap());
        expected.push(tokens().sep);
        expected.extend(encode("st").unwrap());
        expected.push(tokens().sep);
        assert_eq!(prompt.ids, expected);
        assert_eq!(prompt.markers, [first, second]);

        // request text cannot place a marker of its own
        let marked = spec(DecisionFamily::Julia, 64, 128)
            .prompt(&question, &format!("a{MASK}b"), &tokens(), encode)
            .unwrap()
            .unwrap();
        assert!(
            marked
                .ids
                .ends_with(&[encode("a b").unwrap(), vec![tokens().sep]].concat())
        );
    }

    // laya's budgets: the options share what the head leaves after 16 tokens, the question keeps at least 8, and
    // the state fills what the total leaves
    #[test]
    fn long_options_shrink_alike_and_the_state_takes_the_rest() {
        let long = "y".repeat(100);
        let question = DecisionQuestion {
            kind: QuestionKind::Choice,
            instructions: "z".repeat(50),
            options: (0..4)
                .map(|i| option(&format!("{i}{long}"), None))
                .collect(),
        };
        let spec = spec(DecisionFamily::Laya, 96, 160);
        let prompt = spec
            .prompt(&question, &"s".repeat(500), &tokens(), encode)
            .unwrap()
            .unwrap();
        // (96 - 16) / 4 = 20 tokens an option, then 96 - 80 = 16 for the question
        let gaps = prompt
            .markers
            .windows(2)
            .map(|w| w[1] - w[0])
            .collect::<Vec<_>>();
        assert_eq!(gaps, [20, 20, 20]);
        assert_eq!(prompt.markers[0], 1 + 16 + 1);
        assert_eq!(prompt.ids.len(), 160);
        assert_eq!(prompt.ids.last(), Some(&tokens().sep));
    }

    #[test]
    fn calibration_takes_the_option_bucket_then_the_type() {
        let spec = spec(DecisionFamily::Laya, 192, 512);
        assert_eq!(spec.temperature(QuestionKind::Choice, 4), 1.5);
        assert_eq!(spec.temperature(QuestionKind::Choice, 2), 2.0);
        assert_eq!(spec.temperature(QuestionKind::Noul, 2), 4.0);
        let uncalibrated = DecisionSpec {
            temperature: Vec::new(),
            temperature_by_options: HashMap::new(),
            ..spec
        };
        assert_eq!(uncalibrated.temperature(QuestionKind::Score, 5), 1.0);
    }

    // a vocabulary whose single letters are single tokens and whose two-letter codes are not, except "AB"
    fn encode_codes(code: &str) -> Result<Vec<u32>, String> {
        Ok(match code {
            "AB" => vec![1000],
            _ => code.bytes().map(u32::from).collect(),
        })
    }

    fn decision(family: LabelFamily, temperatures: &[(&str, f32)]) -> LabelDecision {
        let temperatures = temperatures
            .iter()
            .map(|(name, t)| (name.to_string(), *t))
            .collect();
        LabelDecision::new(
            family,
            String::new(),
            temperatures,
            HashMap::new(),
            encode_codes,
        )
        .unwrap()
    }

    #[test]
    fn coded_labels_keep_the_single_token_codes_in_order() {
        let nimble = decision(LabelFamily::Nimble, &[]);
        assert_eq!(nimble.labels.len(), 27);
        assert_eq!(nimble.label_texts[..2], ["A", "B"]);
        assert_eq!(nimble.label_texts[26], "AB");
        assert_eq!(nimble.labels[26], 1000);
        let openjev = decision(LabelFamily::OpenJev, &[]);
        assert_eq!(openjev.labels.len(), 52);
        assert_eq!(openjev.labels[26], u32::from(b'a'));
        assert!(openjev.label_texts.is_empty());
    }

    #[test]
    fn openjev_needs_every_letter_as_one_token() {
        let split = |code: &str| -> Result<Vec<u32>, String> {
            Ok(match code {
                "q" => vec![1, 2],
                _ => vec![3],
            })
        };
        let err = LabelDecision::new(
            LabelFamily::OpenJev,
            String::new(),
            HashMap::new(),
            HashMap::new(),
            split,
        )
        .unwrap_err();
        assert!(err.contains("`q`"), "{err}");
    }

    #[test]
    fn metadata_labels_must_be_distinct_tokens() {
        let mut h2o = decision(LabelFamily::H2oLightning, &[]);
        assert!(h2o.labels.is_empty());
        h2o.set_labels(vec!["A".into(), "AB".into()], encode_codes)
            .unwrap();
        assert_eq!(h2o.labels, [u32::from(b'A'), 1000]);
        assert_eq!(h2o.label_texts, ["A", "AB"]);
        let err = h2o
            .set_labels(vec!["A".into(), "A".into()], encode_codes)
            .unwrap_err();
        assert!(err.contains("`A`"), "{err}");
        assert!(h2o.set_labels(vec!["BC".into()], encode_codes).is_err());
    }

    #[test]
    fn d1_labels_follow_the_question_and_fall_back_to_the_pool() {
        let mut d1 = decision(LabelFamily::Lfm2D1, &[]);
        // single letters and digits are one token each, "A" also with a space, "B" is not one token
        let mut id = 0;
        for form in d1_label_forms() {
            let single = match form.as_str() {
                " A" | "yes" | "Yes" | "no" => true,
                "B" => false,
                text => text.len() == 1,
            };
            if single {
                id += 1;
                d1.single_tokens.insert(form, id);
            }
        }
        let (texts, groups) = d1
            .question_labels(QuestionKind::Choice, &["billing", "shipping", "x"])
            .unwrap();
        // B is not one token, so the second option takes the pool's next unused code
        assert_eq!(texts, ["A", "C", "D"]);
        assert_eq!(groups[0].len(), 2);
        let (texts, _) = d1
            .question_labels(QuestionKind::Choice, &["x", "y"])
            .unwrap();
        assert_eq!(texts, ["x", "y"]);
        let (texts, groups) = d1
            .question_labels(QuestionKind::Noul, &["true", "false"])
            .unwrap();
        assert_eq!(texts, ["true", "false"]);
        assert_eq!((groups[0].len(), groups[1].len()), (2, 1));
        assert!(d1.question_labels(QuestionKind::Score, &["0", "1"]).is_ok());
    }

    #[test]
    fn temperature_takes_the_bucket_then_the_type() {
        let nimble = decision(LabelFamily::Nimble, &[("choice.3_5", 2.0), ("choice", 3.0)]);
        assert_eq!(nimble.temperature(QuestionKind::Choice, 4), 2.0);
        assert_eq!(nimble.temperature(QuestionKind::Choice, 6), 3.0);
        assert_eq!(nimble.temperature(QuestionKind::Score, 4), 1.0);
        let lev = decision(LabelFamily::Lev, &[("choice.mid", 5.0)]);
        assert_eq!(lev.temperature(QuestionKind::Choice, 9), 5.0);
        assert_eq!(lev.temperature(QuestionKind::Choice, 8), 1.0);
    }
}
