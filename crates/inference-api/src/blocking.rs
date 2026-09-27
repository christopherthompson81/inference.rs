//! A synchronous face of [`Engine`] for callers without an async runtime, such as the C ABI.

use std::{sync::OnceLock, time::Duration};

use futures::{stream::BoxStream, Stream, StreamExt};
use tokio::runtime::Runtime;

use crate::{
    api_error::ApiError,
    engine::{Engine, EngineLoadError},
    media_source::MediaAttachments,
};

static RUNTIME: OnceLock<Runtime> = OnceLock::new();

// Multi-threaded because engine start-up uses `block_in_place`, which a current-thread runtime does not allow.
fn runtime() -> &'static Runtime {
    RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_name("inference-api")
            .build()
            .expect("failed to start the inference-api runtime")
    })
}

// Work runs on a runtime worker, never on the caller's thread, so `block_in_place` inside the engine is allowed.
fn run<T: Send + 'static>(work: impl std::future::Future<Output = T> + Send + 'static) -> T {
    let rt = runtime();
    rt.block_on(rt.spawn(work))
        .unwrap_or_else(|panic| std::panic::resume_unwind(panic.into_panic()))
}

/// A loaded engine behind blocking calls. Calling from inside a tokio runtime panics, as `Runtime::block_on` does.
#[derive(Clone)]
pub struct BlockingEngine {
    engine: Engine,
}

impl BlockingEngine {
    pub fn load_json(spec: &[u8]) -> Result<Self, EngineLoadError> {
        let spec = spec.to_vec();
        run(async move { Engine::load_json(&spec).await }).map(|engine| Self { engine })
    }

    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    pub fn chat_json(&self, request: &[u8], media: MediaAttachments) -> Result<String, ApiError> {
        let (engine, request) = (self.engine.clone(), request.to_vec());
        run(async move { engine.chat_json(&request, media).await })
    }

    pub fn chat_stream_json(
        &self,
        request: &[u8],
        media: MediaAttachments,
    ) -> Result<BlockingStream, ApiError> {
        let (engine, request) = (self.engine.clone(), request.to_vec());
        run(async move { engine.chat_stream_json(&request, media).await })
            .map(|stream| BlockingStream::new(stream.map(|event| event.to_json())))
    }

    pub fn completion_json(&self, request: &[u8]) -> Result<String, ApiError> {
        let (engine, request) = (self.engine.clone(), request.to_vec());
        run(async move { engine.completion_json(&request).await })
    }

    pub fn completion_stream_json(&self, request: &[u8]) -> Result<BlockingStream, ApiError> {
        let (engine, request) = (self.engine.clone(), request.to_vec());
        run(async move { engine.completion_stream_json(&request).await })
            .map(|stream| BlockingStream::new(stream.map(|event| event.to_json())))
    }

    pub fn anthropic_messages_json(&self, request: &[u8]) -> Result<String, ApiError> {
        let (engine, request) = (self.engine.clone(), request.to_vec());
        run(async move { engine.anthropic_messages_json(&request).await })
    }

    pub fn anthropic_messages_stream_json(
        &self,
        request: &[u8],
    ) -> Result<BlockingStream, ApiError> {
        let (engine, request) = (self.engine.clone(), request.to_vec());
        run(async move { engine.anthropic_messages_stream_json(&request).await })
            .map(|stream| BlockingStream::new(stream.map(|event| event.to_json())))
    }

    pub fn embeddings_json(&self, request: &[u8]) -> Result<String, ApiError> {
        let (engine, request) = (self.engine.clone(), request.to_vec());
        run(async move { engine.embeddings_json(&request).await })
    }
}

/// What one poll of a stream produced.
pub enum StreamPoll {
    /// One event, serialized as its JSON envelope.
    Event(String),
    /// Nothing arrived within the timeout; the stream is still live.
    Timeout,
    /// The stream has finished.
    Done,
}

/// A streaming request behind blocking polls; each event is its JSON envelope. Dropping it abandons the request.
pub struct BlockingStream {
    stream: BoxStream<'static, String>,
}

impl BlockingStream {
    fn new(stream: impl Stream<Item = String> + Send + 'static) -> Self {
        Self {
            stream: stream.boxed(),
        }
    }

    /// Waits up to `timeout` (forever when `None`) for the next event.
    pub fn next(&mut self, timeout: Option<Duration>) -> StreamPoll {
        let event = runtime().block_on(async {
            match timeout {
                Some(timeout) => tokio::time::timeout(timeout, self.stream.next()).await,
                None => Ok(self.stream.next().await),
            }
        });
        match event {
            Ok(Some(event)) => StreamPoll::Event(event),
            Ok(None) => StreamPoll::Done,
            Err(_) => StreamPoll::Timeout,
        }
    }
}
