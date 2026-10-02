//! Embedding requests built in Rust.

use inference_api::openai::{EmbeddingInput, EmbeddingRequest};
use serde_json::json;

use crate::error::{Error, Result};

/// An embedding request with no input and every option at the engine's default.
pub(crate) fn empty_embedding_request() -> EmbeddingRequest {
    serde_json::from_value(json!({"input": ""})).expect("an embedding request needs only its input")
}

/// The texts or token lists to embed, all of one kind, each embedded on its own.
#[derive(Default)]
pub struct EmbeddingRequestBuilder {
    texts: Vec<String>,
    tokens: Vec<Vec<u32>>,
    model: Option<String>,
    dimensions: Option<usize>,
    truncate_sequence: Option<bool>,
}

impl EmbeddingRequestBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.texts.push(prompt.into());
        self
    }

    pub fn add_prompts<I, S>(mut self, prompts: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.texts.extend(prompts.into_iter().map(Into::into));
        self
    }

    pub fn add_tokens(mut self, tokens: impl Into<Vec<u32>>) -> Self {
        self.tokens.push(tokens.into());
        self
    }

    pub fn add_tokens_batch<I>(mut self, batches: I) -> Self
    where
        I: IntoIterator<Item = Vec<u32>>,
    {
        self.tokens.extend(batches);
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Truncates each embedding to `dimensions`, for models trained to allow it.
    pub fn with_dimensions(mut self, dimensions: usize) -> Self {
        self.dimensions = Some(dimensions);
        self
    }

    pub fn with_truncate_sequence(mut self, truncate: bool) -> Self {
        self.truncate_sequence = Some(truncate);
        self
    }

    pub fn build(self) -> Result<EmbeddingRequest> {
        let input = match (self.texts.is_empty(), self.tokens.is_empty()) {
            (false, true) => EmbeddingInput::Multiple(self.texts),
            (true, false) => EmbeddingInput::TokensBatch(self.tokens),
            (true, true) => return Err(Error::Request(NOTHING_TO_EMBED.to_string())),
            (false, false) => return Err(Error::Request(MIXED_INPUTS.to_string())),
        };
        let mut request = empty_embedding_request();
        request.input = input;
        if let Some(model) = self.model {
            request.model = model;
        }
        request.dimensions = self.dimensions;
        request.truncate_sequence = self.truncate_sequence;
        Ok(request)
    }
}

const NOTHING_TO_EMBED: &str = "an embedding request needs a prompt or tokens";
const MIXED_INPUTS: &str = "an embedding request takes prompts or token lists, not both";
