//! Decision models with their own head (laya's family): the question as the model reads it, and its answer from the
//! head's logits over the options.

use inference_nn::decision::{DecisionOption, DecisionQuestion, QuestionKind};
use inference_protocol::openai::{DecisionAnswer, DecisionQuestionType, DecisionRequestQuestion};
use serde_json::Value;

use crate::{FormatError, as_text, first_max, kind_of, required_instructions, required_kind};

// the option counts llama.cpp serves these models with
const MAX_CHOICE_OPTIONS: usize = 255;
const SCORE_LEVELS: std::ops::RangeInclusive<usize> = 2..=10;
const NOUL_KEYS: [&str; 2] = ["false", "true"];
// laya's floor on probabilities in the entropy
const ENTROPY_FLOOR: f64 = 1e-12;

fn description(value: &Value, param: &str) -> Result<Option<String>, FormatError> {
    match value {
        Value::Null => Ok(None),
        Value::String(text) => Ok(Some(text.clone())),
        _ => Err(FormatError::invalid(
            "a criterion's description must be text or null",
            param.to_string(),
        )),
    }
}

/// The question as the model reads it, with the keys its answer reports.
pub fn question(
    id: &str,
    question: &DecisionRequestQuestion,
) -> Result<DecisionQuestion, FormatError> {
    let param = format!("questions.{id}.criteria");
    let question_kind = required_kind(id, question.kind)?;
    required_instructions(id, &question.instructions)?;
    let options = match (question_kind, &question.criteria) {
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
                _ => Err(FormatError::invalid(
                    "choice options must be text",
                    param.clone(),
                )),
            })
            .collect::<Result<Vec<_>, _>>()?,
        (DecisionQuestionType::Choice, _) => {
            return Err(FormatError::invalid(
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
            return Err(FormatError::invalid(
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
                return Err(FormatError::invalid(
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
            return Err(FormatError::invalid(
                "a noul question's criteria describe `true` and `false`",
                param,
            ));
        }
    };
    let kind = kind_of(question_kind);
    match kind {
        QuestionKind::Choice if !(2..=MAX_CHOICE_OPTIONS).contains(&options.len()) => {
            return Err(FormatError::invalid(
                format!("a choice question has 2 to {MAX_CHOICE_OPTIONS} options"),
                param,
            ));
        }
        QuestionKind::Score if !SCORE_LEVELS.contains(&options.len()) => {
            return Err(FormatError::invalid(
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

pub fn answer(question: &DecisionQuestion, logits: &[f32]) -> DecisionAnswer {
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

/// The number of options the model scores for a question.
pub fn n_options(question: &DecisionQuestion) -> usize {
    match question.kind {
        QuestionKind::Noul => NOUL_KEYS.len(),
        _ => question.options.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
