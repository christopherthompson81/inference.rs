//! A synchronous face of [`Engine`] for callers without an async runtime, such as the C ABI.

use std::{any::Any, future::Future, pin::Pin, sync::OnceLock, time::Duration};

use futures::{Stream, StreamExt, stream::BoxStream};
use inference_core::RequestCancellation;
use tokio::runtime::Runtime;

use crate::{
    api_error::ApiError,
    engine::{Engine, EngineCallbacks, EngineLoadError},
    generation::SpeechAudio,
    media_source::MediaAttachments,
    openai::TranscriptionOutput,
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
fn run<T: Send + 'static>(work: impl Future<Output = T> + Send + 'static) -> T {
    let output = run_job(Box::pin(async move {
        Box::new(work.await) as Box<dyn Any + Send>
    }));
    *output
        .downcast::<T>()
        .expect("a job returns its own output type")
}

// Every call spawns this one future type, so tokio's task harness is instantiated once, not once per method.
fn run_job(job: Pin<Box<dyn Future<Output = Box<dyn Any + Send>> + Send>>) -> Box<dyn Any + Send> {
    let rt = runtime();
    rt.block_on(rt.spawn(job))
        .unwrap_or_else(|panic| std::panic::resume_unwind(panic.into_panic()))
}

/// A loaded engine behind blocking calls. Calling from inside a tokio runtime panics, as `Runtime::block_on` does.
#[derive(Clone)]
pub struct BlockingEngine {
    engine: Engine,
}

impl BlockingEngine {
    pub fn load_json(spec: &[u8], callbacks: EngineCallbacks) -> Result<Self, EngineLoadError> {
        let spec = spec.to_vec();
        run(async move { Engine::load_json(&spec, callbacks).await }).map(|engine| Self { engine })
    }

    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    /// The same engine acting for `owner`; see [`Engine::for_owner`].
    pub fn for_owner(&self, owner: &str) -> Self {
        Self {
            engine: self.engine.for_owner(owner),
        }
    }

    // Runs `op` on a runtime worker with owned copies of the engine and the request.
    fn call<T, Fut>(&self, request: &[u8], op: impl FnOnce(Engine, Vec<u8>) -> Fut) -> T
    where
        T: Send + 'static,
        Fut: Future<Output = T> + Send + 'static,
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
        Fut: Future<Output = Result<(S, RequestCancellation), ApiError>> + Send + 'static,
    {
        self.call(request, op)
            .map(|(stream, cancellation)| BlockingStream::new(stream, cancellation))
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
                .map(|stream| {
                    let cancellation = stream
                        .cancellation()
                        .expect("engine streams carry their cancellation");
                    (stream.map(|item| item.to_json()), cancellation)
                })
        })
    }

    pub fn completion_json(&self, request: &[u8]) -> Result<String, ApiError> {
        self.call(request, |engine, request| async move {
            engine.completion_json(&request).await
        })
    }

    pub fn completion_stream_json(&self, request: &[u8]) -> Result<BlockingStream, ApiError> {
        self.stream(request, |engine, request| async move {
            engine.completion_stream_json(&request).await.map(|stream| {
                let cancellation = stream
                    .cancellation()
                    .expect("engine streams carry their cancellation");
                (stream.map(|item| item.to_json()), cancellation)
            })
        })
    }

    pub fn anthropic_messages_json(&self, request: &[u8]) -> Result<String, ApiError> {
        self.call(request, |engine, request| async move {
            engine.anthropic_messages_json(&request).await
        })
    }

    pub fn count_tokens_json(&self, request: &[u8]) -> Result<String, ApiError> {
        self.call(request, |engine, request| async move {
            engine.count_tokens_json(&request).await
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
                .map(|stream| {
                    let cancellation = stream.cancellation();
                    (stream.map(|item| item.to_json()), cancellation)
                })
        })
    }

    pub fn responses_json(&self, request: &[u8]) -> Result<String, ApiError> {
        self.call(request, |engine, request| async move {
            engine.responses_json(&request).await
        })
    }

    pub fn responses_stream_json(&self, request: &[u8]) -> Result<BlockingStream, ApiError> {
        self.stream(request, |engine, request| async move {
            engine.responses_stream_json(&request).await.map(|stream| {
                let cancellation = stream.cancellation();
                (stream.map(|item| item.to_json()), cancellation)
            })
        })
    }

    pub fn embeddings_json(&self, request: &[u8]) -> Result<String, ApiError> {
        self.call(request, |engine, request| async move {
            engine.embeddings_json(&request).await
        })
    }

    pub fn image_generation_json(&self, request: &[u8]) -> Result<String, ApiError> {
        self.call(request, |engine, request| async move {
            engine.image_generation_json(&request).await
        })
    }

    pub fn speech_generation_json(&self, request: &[u8]) -> Result<SpeechAudio, ApiError> {
        self.call(request, |engine, request| async move {
            engine.speech_generation_json(&request).await
        })
    }

    pub fn voice_activity_json(&self, request: &[u8], audio: &[u8]) -> Result<String, ApiError> {
        let audio = audio.to_vec();
        self.call(request, |engine, request| async move {
            engine.voice_activity_json(&request, &audio).await
        })
    }

    pub fn transcription_json(
        &self,
        request: &[u8],
        audio: &[u8],
    ) -> Result<TranscriptionOutput, ApiError> {
        let audio = audio.to_vec();
        self.call(request, |engine, request| async move {
            engine.transcription_json(&request, &audio).await
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

    pub fn add_model_json(&self, request: &[u8]) -> Result<String, ApiError> {
        self.call(request, |engine, request| async move {
            engine.add_model_json(&request).await
        })
    }

    pub fn remove_model_json(&self, request: &[u8]) -> Result<String, ApiError> {
        self.call(request, |engine, request| async move {
            engine.remove_model_json(&request).await
        })
    }

    pub fn prompt_logits_json(
        &self,
        request: &[u8],
    ) -> Result<(String, Option<Vec<f32>>), ApiError> {
        self.call(request, |engine, request| async move {
            engine.prompt_logits_json(&request).await
        })
    }

    pub fn re_isq_json(&self, request: &[u8]) -> Result<String, ApiError> {
        self.call(request, |engine, request| async move {
            engine.re_isq_json(&request).await
        })
    }

    pub fn calibration_start_json(&self, request: &[u8]) -> Result<String, ApiError> {
        self.call(request, |engine, request| async move {
            engine.calibration_start_json(&request).await
        })
    }

    pub fn cache_stats_json(&self) -> Result<String, ApiError> {
        self.call(&[], |engine, _| async move { engine.cache_stats_json() })
    }

    pub fn speculative_stats_json(&self) -> Result<String, ApiError> {
        self.call(
            &[],
            |engine, _| async move { engine.speculative_stats_json() },
        )
    }

    pub fn calibration_status_json(&self, request: &[u8]) -> Result<String, ApiError> {
        self.call(request, |engine, request| async move {
            engine.calibration_status_json(&request).await
        })
    }

    pub fn calibration_apply_json(&self, request: &[u8]) -> Result<String, ApiError> {
        self.call(request, |engine, request| async move {
            engine.calibration_apply_json(&request).await
        })
    }

    pub fn tokenize_json(&self, request: &[u8]) -> Result<String, ApiError> {
        self.call(request, |engine, request| async move {
            engine.tokenize_json(&request).await
        })
    }

    pub fn tokenize_chat_json(&self, request: &[u8]) -> Result<String, ApiError> {
        self.call(request, |engine, request| async move {
            engine.tokenize_chat_json(&request).await
        })
    }

    pub fn detokenize_json(&self, request: &[u8]) -> Result<String, ApiError> {
        self.call(request, |engine, request| async move {
            engine.detokenize_json(&request).await
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
    cancellation: RequestCancellation,
}

impl BlockingStream {
    fn new(
        stream: impl Stream<Item = String> + Send + 'static,
        cancellation: RequestCancellation,
    ) -> Self {
        Self {
            stream: stream.boxed(),
            cancellation,
        }
    }

    /// Ends the request on its next sampled token; polling still yields its final event, with usage.
    pub fn cancel(&self) {
        self.cancellation.cancel();
    }

    /// The request's cancellation, for a caller that cancels while another thread polls.
    pub fn cancellation(&self) -> RequestCancellation {
        self.cancellation.clone()
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
