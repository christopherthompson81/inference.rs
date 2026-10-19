//! Where the model runs: the in-process engine or an OpenAI-compatible server, both speaking OpenAI chat JSON.

use std::time::Duration;

use anyhow::{Context, Result};
use futures::{future::BoxFuture, stream::BoxStream};
use serde_json::Value;

const CHAT_PATH: &str = "/v1/chat/completions";
const SSE_DATA: &str = "data:";
const SSE_DONE: &str = "[DONE]";
// a turn's prefill can take minutes at long context; the stream itself has no deadline
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// A chat model that streams OpenAI chat completion chunks for an OpenAI chat request.
pub trait ChatBackend: Send + Sync {
    fn stream(&self, request: Value) -> BoxFuture<'_, Result<BoxStream<'static, Result<Value>>>>;

    /// What the model is, for the session header and the banner.
    fn describe(&self) -> String;
}

/// The engine in this process.
pub struct EngineBackend {
    engine: inference_api::Engine,
    model: String,
}

impl EngineBackend {
    pub fn new(engine: inference_api::Engine, model: impl Into<String>) -> Self {
        Self {
            engine,
            model: model.into(),
        }
    }
}

impl ChatBackend for EngineBackend {
    fn stream(&self, request: Value) -> BoxFuture<'_, Result<BoxStream<'static, Result<Value>>>> {
        Box::pin(async move {
            let request = serde_json::from_value(request).context("building the chat request")?;
            let stream = self
                .engine
                .chat_stream(request, Default::default())
                .await
                .map_err(|error| anyhow::anyhow!("{error}"))?;
            let chunks = futures::stream::unfold(stream, |mut stream| async move {
                loop {
                    match stream.next_event().await? {
                        inference_api::engine_chat::ChatStreamEvent::Chunk(chunk) => {
                            return Some((serde_json::to_value(chunk).map_err(Into::into), stream));
                        }
                        inference_api::engine_chat::ChatStreamEvent::Error(error) => {
                            return Some((Err(anyhow::anyhow!("{error}")), stream));
                        }
                        // the harness runs its own tools, so the engine's agent events never come
                        _ => continue,
                    }
                }
            });
            Ok(Box::pin(chunks) as BoxStream<'static, Result<Value>>)
        })
    }

    fn describe(&self) -> String {
        self.model.clone()
    }
}

/// An OpenAI-compatible server, such as `inference serve`.
pub struct HttpBackend {
    client: reqwest::Client,
    url: String,
    api_key: Option<String>,
    model: String,
}

impl HttpBackend {
    pub fn new(base_url: &str, api_key: Option<String>, model: impl Into<String>) -> Result<Self> {
        inference_mcp::tls::install_provider();
        let client = reqwest::Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .build()
            .context("building the HTTP client")?;
        Ok(Self {
            client,
            url: format!("{}{CHAT_PATH}", base_url.trim_end_matches('/')),
            api_key,
            model: model.into(),
        })
    }
}

impl ChatBackend for HttpBackend {
    fn stream(
        &self,
        mut request: Value,
    ) -> BoxFuture<'_, Result<BoxStream<'static, Result<Value>>>> {
        Box::pin(async move {
            request["model"] = Value::String(self.model.clone());
            request["stream"] = Value::Bool(true);
            let mut post = self.client.post(&self.url).json(&request);
            if let Some(key) = &self.api_key {
                post = post.bearer_auth(key);
            }
            let response = post
                .send()
                .await
                .with_context(|| format!("POST {}", self.url))?;
            let status = response.status();
            if !status.is_success() {
                let body = response.text().await.unwrap_or_default();
                anyhow::bail!("{status} from {}: {body}", self.url);
            }
            // bytes, not text, until an event is whole: a chunk can end inside a multi-byte character
            let chunks = futures::stream::unfold(
                (response, Vec::<u8>::new(), false),
                |(mut response, mut buffer, mut done)| async move {
                    loop {
                        if done {
                            return None;
                        }
                        if let Some(end) = buffer.windows(2).position(|w| w == b"\n\n") {
                            let event = String::from_utf8_lossy(&buffer[..end]).into_owned();
                            buffer.drain(..end + 2);
                            match parse_sse_event(&event) {
                                SseEvent::Chunk(chunk) => {
                                    return Some((chunk, (response, buffer, done)));
                                }
                                SseEvent::Done => done = true,
                                SseEvent::Other => {}
                            }
                            continue;
                        }
                        match response.chunk().await {
                            Ok(Some(bytes)) => buffer.extend(bytes.iter().filter(|&&b| b != b'\r')),
                            Ok(None) => return None,
                            Err(error) => {
                                return Some((Err(error.into()), (response, buffer, true)));
                            }
                        }
                    }
                },
            );
            Ok(Box::pin(chunks) as BoxStream<'static, Result<Value>>)
        })
    }

    fn describe(&self) -> String {
        format!("{} at {}", self.model, self.url.trim_end_matches(CHAT_PATH))
    }
}

enum SseEvent {
    Chunk(Result<Value>),
    Done,
    Other,
}

fn parse_sse_event(event: &str) -> SseEvent {
    let data = event
        .lines()
        .filter_map(|line| line.strip_prefix(SSE_DATA))
        .map(str::trim_start)
        .collect::<Vec<_>>()
        .join("\n");
    if data.is_empty() {
        SseEvent::Other
    } else if data == SSE_DONE {
        SseEvent::Done
    } else {
        SseEvent::Chunk(
            serde_json::from_str::<Value>(&data)
                .map_err(Into::into)
                .and_then(|chunk| match chunk.get("error") {
                    Some(error) => {
                        let message = error
                            .get("message")
                            .and_then(Value::as_str)
                            .map_or_else(|| error.to_string(), str::to_string);
                        Err(anyhow::anyhow!("server error: {message}"))
                    }
                    None => Ok(chunk),
                }),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{SseEvent, parse_sse_event};

    #[test]
    fn sse_events_parse_into_chunks_and_the_end() {
        assert!(matches!(parse_sse_event("data: [DONE]"), SseEvent::Done));
        assert!(matches!(parse_sse_event(": keep-alive"), SseEvent::Other));
        let SseEvent::Chunk(Ok(chunk)) = parse_sse_event("event: message\ndata: {\"choices\":[]}")
        else {
            panic!("a data line is a chunk");
        };
        assert!(chunk["choices"].is_array());
        let SseEvent::Chunk(Err(error)) =
            parse_sse_event("event: error\ndata: {\"error\":{\"message\":\"out of memory\"}}")
        else {
            panic!("an error event is an error");
        };
        assert!(error.to_string().contains("out of memory"));
    }
}
