//! Decision models (laya, Julia-1): a typed question about a state, answered by scoring one `[MASK]` marker per
//! option in a `[CLS] question [SEP] [MASK] option ... [SEP] state [SEP]` sequence.
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
}
