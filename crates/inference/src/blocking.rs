//! Blocking wrappers for scripts and programs with no async runtime of their own.
//!
//! A [`BlockingModel`] owns a runtime, so it must not be used from inside one: `block_on` there panics.

use std::sync::Arc;

use inference_api::{engine_chat::ChatStreamEvent, response::ChatCompletionResponse};

use crate::{
    ModelBuilder, TextModelBuilder,
    error::Result,
    model::{ChatEventStream, Model},
    request::{ChatRequest, RequestBuilder},
};

pub struct BlockingModel {
    inner: Model,
    rt: Arc<tokio::runtime::Runtime>,
}

impl BlockingModel {
    /// Runs `load` (a builder's `build()`) on a runtime of its own and keeps that runtime.
    pub fn load(load: impl std::future::Future<Output = Result<Model>>) -> Result<Self> {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;
        let inner = rt.block_on(load)?;
        Ok(Self {
            inner,
            rt: Arc::new(rt),
        })
    }

    pub fn from_builder(builder: TextModelBuilder) -> Result<Self> {
        Self::load(builder.build())
    }

    pub fn from_auto_builder(builder: ModelBuilder) -> Result<Self> {
        Self::load(builder.build())
    }

    pub fn new(model: Model, rt: Arc<tokio::runtime::Runtime>) -> Self {
        Self { inner: model, rt }
    }

    pub fn send_chat_request(
        &self,
        request: impl Into<ChatRequest>,
    ) -> Result<ChatCompletionResponse> {
        self.rt.block_on(self.inner.send_chat_request(request))
    }

    pub fn chat(&self, message: impl ToString) -> Result<String> {
        self.rt.block_on(self.inner.chat(message))
    }

    pub fn stream_chat_request(&self, request: impl Into<ChatRequest>) -> Result<BlockingStream> {
        let stream = self.rt.block_on(self.inner.stream_chat_request(request))?;
        Ok(BlockingStream {
            stream,
            rt: self.rt.clone(),
        })
    }

    pub fn generate_structured<T>(&self, request: impl Into<RequestBuilder>) -> Result<T>
    where
        T: serde::de::DeserializeOwned + schemars::JsonSchema,
    {
        self.rt
            .block_on(self.inner.generate_structured::<T>(request))
    }

    pub fn inner(&self) -> &Model {
        &self.inner
    }
}

/// A streaming chat's events, each waited for on the model's runtime.
pub struct BlockingStream {
    stream: ChatEventStream,
    rt: Arc<tokio::runtime::Runtime>,
}

impl Iterator for BlockingStream {
    type Item = ChatStreamEvent;

    fn next(&mut self) -> Option<Self::Item> {
        self.rt.block_on(self.stream.next())
    }
}
