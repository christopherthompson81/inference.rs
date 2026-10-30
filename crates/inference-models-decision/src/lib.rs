//! Decision models' formats: how each family reads a typed question about a state, and how its scores become an
//! answer. The engine runs the passes; this crate knows what goes in and what comes out.

use inference_nn::decision::QuestionKind;
use inference_protocol::openai::DecisionQuestionType;
use serde_json::Value;

pub mod head;
pub mod labels;
mod python_json;

pub use python_json::{PythonJson, as_text, python_json};

/// Why a request's questions cannot be read: the caller's input at `param`, or a prompt the template laid out wrong.
#[derive(Debug)]
pub enum FormatError {
    Invalid { message: String, param: String },
    Layout,
}

impl FormatError {
    pub fn invalid(message: impl Into<String>, param: impl Into<String>) -> Self {
        Self::Invalid {
            message: message.into(),
            param: param.into(),
        }
    }
}

impl std::fmt::Display for FormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid { message, .. } => f.write_str(message),
            Self::Layout => f.write_str("unexpected layout of the decision prompt"),
        }
    }
}

impl std::error::Error for FormatError {}

/// The question's type, which a model without a default needs.
pub fn required_kind(
    id: &str,
    kind: Option<DecisionQuestionType>,
) -> Result<DecisionQuestionType, FormatError> {
    kind.ok_or_else(|| {
        FormatError::invalid(
            "`type` must be one of: choice, score, noul",
            format!("questions.{id}.type"),
        )
    })
}

/// The question's instructions, which a model without a default question needs.
pub fn required_instructions(id: &str, instructions: &Value) -> Result<(), FormatError> {
    match instructions {
        Value::Null => Err(FormatError::invalid(
            "`instructions` must be provided",
            format!("questions.{id}.instructions"),
        )),
        _ => Ok(()),
    }
}

pub fn kind_of(kind: DecisionQuestionType) -> QuestionKind {
    match kind {
        DecisionQuestionType::Choice => QuestionKind::Choice,
        DecisionQuestionType::Score => QuestionKind::Score,
        DecisionQuestionType::Noul => QuestionKind::Noul,
    }
}

/// The index of the largest probability, the first among equals as numpy's and C++'s argmax take it.
pub fn first_max(p: &[f64]) -> usize {
    (1..p.len()).fold(0, |best, i| if p[i] > p[best] { i } else { best })
}
