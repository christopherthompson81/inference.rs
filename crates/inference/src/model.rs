//! A loaded engine with the conveniences a Rust caller wants: typed chat, streaming, structured output, embeddings.

use std::{
    ops::Deref,
    pin::Pin,
    task::{Context, Poll},
};

use futures::{Stream, StreamExt};
use inference_api::{
    Engine,
    engine_chat::{ChatStream, ChatStreamEvent},
    openai::{
        EmbeddingInput, EmbeddingRequest, EmbeddingResponse, Grammar, ImageGenerationRequest,
        SpeechGenerationRequest,
    },
    response::ChatCompletionResponse,
};
use serde::de::DeserializeOwned;

use crate::{
    error::{Error, Result},
    request::{ChatRequest, RequestBuilder, TextMessageRole},
};

const SCOPED_PROCESSOR_PREFIX: &str = "sdk-request-";
const STREAMED_APPROVAL: &str = "an approval callback answers non-streaming requests; a stream answers its \
     approval events with resolve_approval";

/// A loaded engine. Every [`Engine`] operation is reachable through it; the methods here add Rust types on top.
#[derive(Clone)]
pub struct Model {
    engine: Engine,
}

impl Deref for Model {
    type Target = Engine;

    fn deref(&self) -> &Engine {
        &self.engine
    }
}

impl From<Engine> for Model {
    fn from(engine: Engine) -> Self {
        Self { engine }
    }
}

// Registered for one request under names of its own, and unregistered when it ends.
struct ScopedProcessors {
    engine: Engine,
    names: Vec<String>,
}

impl Drop for ScopedProcessors {
    fn drop(&mut self) {
        for name in &self.names {
            let _ = self.engine.unregister_logits_processor(name);
        }
    }
}

/// A streaming chat; ending it early (or dropping it) abandons the request.
pub struct ChatEventStream {
    inner: ChatStream,
    _processors: ScopedProcessors,
}

impl Stream for ChatEventStream {
    type Item = ChatStreamEvent;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(cx)
    }
}

impl ChatEventStream {
    pub async fn next(&mut self) -> Option<ChatStreamEvent> {
        StreamExt::next(self).await
    }
}

impl Model {
    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    /// The same engine acting for `owner`, whose sessions and files are that owner's alone.
    pub fn for_owner(&self, owner: impl Into<String>) -> Self {
        self.engine.for_owner(owner).into()
    }

    fn scope(&self, request: &mut ChatRequest) -> Result<ScopedProcessors> {
        let mut scoped = ScopedProcessors {
            engine: self.engine.clone(),
            names: Vec::new(),
        };
        for processor in std::mem::take(&mut request.logits_processors) {
            let name = format!("{SCOPED_PROCESSOR_PREFIX}{}", uuid::Uuid::new_v4().simple());
            self.engine
                .register_logits_processor(name.clone(), processor)?;
            scoped.names.push(name.clone());
            request
                .request
                .logits_processors
                .get_or_insert_with(Vec::new)
                .push(name);
        }
        Ok(scoped)
    }

    /// Runs a chat request to its end, answering approval prompts with the request's callback.
    pub async fn send_chat_request(
        &self,
        request: impl Into<ChatRequest>,
    ) -> Result<ChatCompletionResponse> {
        let mut request = request.into();
        let _processors = self.scope(&mut request)?;
        let response = match request.approval.take() {
            Some(approver) => {
                let chat = self
                    .engine
                    .chat_with_approver(request.request, request.media, approver);
                chat.await?
            }
            None => self.engine.chat(request.request, request.media).await?,
        };
        Ok(response)
    }

    pub async fn stream_chat_request(
        &self,
        request: impl Into<ChatRequest>,
    ) -> Result<ChatEventStream> {
        let mut request = request.into();
        if request.approval.is_some() {
            return Err(Error::Request(STREAMED_APPROVAL.to_string()));
        }
        let processors = self.scope(&mut request)?;
        let inner = self
            .engine
            .chat_stream(request.request, request.media)
            .await?;
        Ok(ChatEventStream {
            inner,
            _processors: processors,
        })
    }

    /// One user message in, the reply's text out.
    pub async fn chat(&self, message: impl ToString) -> Result<String> {
        let request = RequestBuilder::new().add_message(TextMessageRole::User, message);
        let response = self.send_chat_request(request).await?;
        Ok(reply_text(&response))
    }

    /// The reply constrained to `T`'s JSON schema and parsed into it.
    pub async fn generate_structured<T>(&self, request: impl Into<RequestBuilder>) -> Result<T>
    where
        T: DeserializeOwned + schemars::JsonSchema,
    {
        let schema = serde_json::to_value(schemars::schema_for!(T))?;
        let request = request.into().set_grammar(Grammar::JsonSchema(schema));
        let response = self.send_chat_request(request).await?;
        Ok(serde_json::from_str(&reply_text(&response))?)
    }

    /// Requantizes the default model, which must have loaded with ISQ, to `isq`.
    pub async fn re_isq_model(&self, isq: crate::IsqType) -> Result<()> {
        let request = inference_api::operations::ReIsqRequest {
            ggml_type: isq.to_string(),
            model: None,
        };
        self.engine.re_isq(request).await?;
        Ok(())
    }

    pub async fn generate_embeddings(
        &self,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse> {
        Ok(self.engine.embeddings(request).await?)
    }

    /// One text's embedding.
    pub async fn generate_embedding(&self, text: impl Into<String>) -> Result<Vec<f32>> {
        let mut request = crate::embedding::empty_embedding_request();
        request.input = EmbeddingInput::Single(text.into());
        let response = self.engine.embeddings(request).await?;
        response
            .data
            .into_iter()
            .next()
            .and_then(|embedding| match embedding.embedding {
                inference_api::openai::EmbeddingVector::Float(values) => Some(values),
                inference_api::openai::EmbeddingVector::Base64(_) => None,
            })
            .ok_or(Error::Empty)
    }

    pub async fn generate_image(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<inference_api::response::ImageGenerationResponse> {
        Ok(self.engine.image_generation(request).await?)
    }

    pub async fn generate_speech(
        &self,
        request: SpeechGenerationRequest,
    ) -> Result<inference_api::generation::SpeechAudio> {
        Ok(self.engine.speech_generation(request).await?)
    }

    /// Uploads the skill directory at `dir` (its `SKILL.md` at the top) and returns the id requests mount it by.
    pub fn upload_skill(&self, dir: impl AsRef<std::path::Path>) -> Result<String> {
        let dir = dir.as_ref();
        let mut files = inference_api::skill_store::SkillFiles::default();
        let mut pending = vec![dir.to_path_buf()];
        while let Some(current) = pending.pop() {
            for entry in std::fs::read_dir(&current)? {
                let entry = entry?;
                // Links are skipped, so a cycle cannot loop; dotfiles (a `.git`) are not part of a skill.
                let kind = entry.file_type()?;
                if kind.is_symlink() || entry.file_name().to_string_lossy().starts_with('.') {
                    continue;
                }
                let path = entry.path();
                if kind.is_dir() {
                    pending.push(path);
                    continue;
                }
                let relative = path.strip_prefix(dir).unwrap_or(&path);
                let name = relative.to_string_lossy().replace('\\', "/");
                files
                    .push(name, std::fs::read(&path)?)
                    .map_err(|error| Error::Request(format!("{error:#}")))?;
            }
        }
        let uploaded: serde_json::Value =
            serde_json::from_str(&self.engine.upload_skill_json(files)?)?;
        uploaded["id"]
            .as_str()
            .map(str::to_string)
            .ok_or(Error::Empty)
    }
}

fn reply_text(response: &ChatCompletionResponse) -> String {
    response
        .choices
        .first()
        .and_then(|choice| choice.message.content.clone())
        .unwrap_or_default()
}
