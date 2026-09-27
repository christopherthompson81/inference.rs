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

    // Runs `op` on a runtime worker with owned copies of the engine and the request.
    fn call<T, Fut>(&self, request: &[u8], op: impl FnOnce(Engine, Vec<u8>) -> Fut) -> T
    where
        T: Send + 'static,
        Fut: std::future::Future<Output = T> + Send + 'static,
    {
        run(op(self.engine.clone(), request.to_vec()))
    }

    fn stream<S, Fut>(
        &self,
        request: &[u8],
        op: impl FnOnce(Engine, Vec<u8>) -> Fut,
    ) -> Result<BlockingStream, ApiError>
    where
        S: Stream<Item = String> + Send + 'static,
        Fut: std::future::Future<Output = Result<S, ApiError>> + Send + 'static,
    {
        self.call(request, op).map(BlockingStream::new)
    }

    pub fn chat_json(&self, request: &[u8], media: MediaAttachments) -> Result<String, ApiError> {
        self.call(request, |engine, request| async move {
            engine.chat_json(&request, media).await
        })
    }

    pub fn chat_stream_json(
        &self,
        request: &[u8],
        media: MediaAttachments,
    ) -> Result<BlockingStream, ApiError> {
        self.stream(request, |engine, request| async move {
            engine
                .chat_stream_json(&request, media)
                .await
                .map(|stream| stream.map(|item| item.to_json()))
        })
    }

    pub fn completion_json(&self, request: &[u8]) -> Result<String, ApiError> {
        self.call(request, |engine, request| async move {
            engine.completion_json(&request).await
        })
    }

    pub fn completion_stream_json(&self, request: &[u8]) -> Result<BlockingStream, ApiError> {
        self.stream(request, |engine, request| async move {
            engine
                .completion_stream_json(&request)
                .await
                .map(|stream| stream.map(|item| item.to_json()))
        })
    }

    pub fn anthropic_messages_json(&self, request: &[u8]) -> Result<String, ApiError> {
        self.call(request, |engine, request| async move {
            engine.anthropic_messages_json(&request).await
        })
    }

    pub fn anthropic_messages_stream_json(
        &self,
        request: &[u8],
    ) -> Result<BlockingStream, ApiError> {
        self.stream(request, |engine, request| async move {
            engine
                .anthropic_messages_stream_json(&request)
                .await
                .map(|stream| stream.map(|item| item.to_json()))
        })
    }

    pub fn responses_json(&self, request: &[u8]) -> Result<String, ApiError> {
        self.call(request, |engine, request| async move {
            engine.responses_json(&request).await
        })
    }

    pub fn responses_stream_json(&self, request: &[u8]) -> Result<BlockingStream, ApiError> {
        self.stream(request, |engine, request| async move {
            engine
                .responses_stream_json(&request)
                .await
                .map(|stream| stream.map(|item| item.to_json()))
        })
    }

    pub fn embeddings_json(&self, request: &[u8]) -> Result<String, ApiError> {
        self.call(request, |engine, request| async move {
            engine.embeddings_json(&request).await
        })
    }

    pub fn reload_model_json(&self, request: &[u8]) -> Result<String, ApiError> {
        self.call(request, |engine, request| async move {
            engine.reload_model_json(&request).await
        })
    }

    pub fn lora_adapters_json(&self, request: &[u8]) -> Result<String, ApiError> {
        self.call(request, |engine, request| async move {
            engine.lora_adapters_json(&request).await
        })
    }

    pub fn load_lora_adapter_json(&self, request: &[u8]) -> Result<String, ApiError> {
        self.call(request, |engine, request| async move {
            engine.load_lora_adapter_json(&request).await
        })
    }

    pub fn unload_lora_adapter_json(&self, request: &[u8]) -> Result<String, ApiError> {
        self.call(request, |engine, request| async move {
            engine.unload_lora_adapter_json(&request).await
        })
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
