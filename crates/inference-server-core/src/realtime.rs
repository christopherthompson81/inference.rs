//! OpenAI's Realtime API for transcription sessions: `/v1/realtime?intent=transcription` over a WebSocket, each
//! audio item streamed through an engine transcription session.

use std::time::Duration;

use axum::{
    extract::{
        Query,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    http::HeaderMap,
    response::{IntoResponse, Response},
};
use base64::Engine as _;
use futures::SinkExt as _;
use inference_api::{
    Engine,
    api_error::{ApiError, ApiErrorKind},
    lora_routing::DEFAULT_MODEL_ID,
    openai::{TranscriptionResponseFormat, TranscriptionSessionRequest},
    transcription_session::{PCM16_BYTES, PCM16_SCALE, TranscriptionSession},
};
use serde_json::{Value, json};

use crate::{handler_core::openai_error_response, types::OwnedEngine};

/// Realtime's PCM: 16-bit little-endian mono at 24 kHz unless the session names a rate.
const DEFAULT_RATE: u32 = 24_000;
const MIN_RATE: u32 = 8_000;
const MAX_RATE: u32 = 192_000;
const SUBPROTOCOL: &str = "realtime";
// a beta client says so up front, so its first event is in its own dialect
const BETA_SUBPROTOCOL: &str = "openai-beta.realtime-v1";
const BETA_HEADER: &str = "openai-beta";
const BETA_HEADER_VALUE: &str = "realtime=v1";
const TRANSCRIPTION_INTENT: &str = "transcription";
// OpenAI takes appends up to 15 MiB of base64
const MAX_MESSAGE_BYTES: usize = 16 << 20;
// a client that stops reading, or stops talking, releases its socket and session
const SEND_TIMEOUT: Duration = Duration::from_secs(30);
const IDLE_TIMEOUT: Duration = Duration::from_secs(10 * 60);
// OpenAI's server_vad defaults
const DEFAULT_THRESHOLD: f64 = 0.5;
const DEFAULT_PREFIX_PADDING_MS: u64 = 300;
const DEFAULT_SILENCE_MS: u64 = 500;
// the energy detector's frame, and the loudness a threshold of 0 and of 1 call speech, in dBFS
const VAD_FRAME_MS: u64 = 20;
const QUIETEST_SPEECH_DBFS: f64 = -60.0;
const LOUDEST_THRESHOLD_DBFS: f64 = -20.0;

#[derive(serde::Deserialize)]
pub struct RealtimeQuery {
    intent: Option<String>,
    model: Option<String>,
}

/// The Realtime WebSocket, for transcription sessions only.
pub async fn realtime(
    OwnedEngine(engine): OwnedEngine,
    Query(query): Query<RealtimeQuery>,
    headers: HeaderMap,
    upgrade: WebSocketUpgrade,
) -> Response {
    if query
        .intent
        .as_deref()
        .is_some_and(|i| i != TRANSCRIPTION_INTENT)
    {
        return openai_error_response(json_error(
            "unsupported_intent",
            "only transcription sessions are served: connect with `?intent=transcription`",
            Some("intent"),
        ));
    }
    let beta = beta_requested(&headers);
    upgrade
        .protocols([SUBPROTOCOL])
        .max_message_size(MAX_MESSAGE_BYTES)
        .max_frame_size(MAX_MESSAGE_BYTES)
        .on_upgrade(move |socket| async move {
            let mut connection = Connection::new(engine, socket, query.model, beta);
            connection.run().await;
        })
        .into_response()
}

fn beta_requested(headers: &HeaderMap) -> bool {
    let header = |name: &str| {
        headers
            .get(name)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_string()
    };
    header(BETA_HEADER).contains(BETA_HEADER_VALUE)
        || header(axum::http::header::SEC_WEBSOCKET_PROTOCOL.as_str())
            .split(',')
            .any(|protocol| protocol.trim() == BETA_SUBPROTOCOL)
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ServerVad {
    threshold: f64,
    prefix_padding_ms: u64,
    silence_ms: u64,
}

impl Default for ServerVad {
    fn default() -> Self {
        Self {
            threshold: DEFAULT_THRESHOLD,
            prefix_padding_ms: DEFAULT_PREFIX_PADDING_MS,
            silence_ms: DEFAULT_SILENCE_MS,
        }
    }
}

impl ServerVad {
    // a frame's RMS at or above this is speech
    fn level(&self) -> f64 {
        let dbfs =
            QUIETEST_SPEECH_DBFS + (LOUDEST_THRESHOLD_DBFS - QUIETEST_SPEECH_DBFS) * self.threshold;
        10f64.powf(dbfs / 20.0)
    }
}

struct Config {
    // the model a client named, echoed back; `model` is the one serving it
    requested_model: Option<String>,
    model: String,
    language: Option<String>,
    prompt: Option<String>,
    rate: u32,
    turn: Option<ServerVad>,
    // a client speaking the beta's `transcription_session.*` events hears them back
    beta: bool,
}

// the audio item being transcribed
struct Item {
    id: String,
    session: TranscriptionSession,
    samples: u64,
}

// server VAD's view of the stream: the partial frame waiting for the next append, the audio before speech kept as
// padding, and whether a turn is open
#[derive(Default)]
struct Detector {
    partial: Vec<u8>,
    prefix: Vec<u8>,
    speaking: bool,
    last_loud_ms: f64,
}

struct Connection {
    engine: Engine,
    socket: WebSocket,
    config: Config,
    id: String,
    item: Option<Item>,
    previous_item: Option<String>,
    detector: Detector,
    // audio received over the whole connection, the clock speech times read
    stream_ms: f64,
    events: u64,
}

enum Outcome {
    // the event failed: report it and carry on
    Api(ApiError),
    // the socket is gone, or will not take more
    Socket,
}

type Step = Result<(), Outcome>;

fn json_error(code: &str, message: impl Into<String>, param: Option<&str>) -> ApiError {
    ApiError::new(
        ApiErrorKind::InvalidRequest,
        message.into(),
        Some(code),
        param,
    )
}

fn error_object(error: &ApiError) -> Value {
    json!({
        "type": error.openai_type(),
        "code": error.code,
        "message": error.message,
        "param": error.param,
    })
}

fn rms(frame: &[u8]) -> f64 {
    let samples = frame.as_chunks::<PCM16_BYTES>().0;
    let energy: f64 = samples
        .iter()
        .map(|b| (f64::from(i16::from_le_bytes(*b)) / f64::from(PCM16_SCALE)).powi(2))
        .sum();
    (energy / samples.len().max(1) as f64).sqrt()
}

impl Connection {
    fn new(engine: Engine, socket: WebSocket, model: Option<String>, beta: bool) -> Self {
        let served = served(&engine, model.as_deref());
        Self {
            engine,
            socket,
            config: Config {
                requested_model: model,
                model: served,
                language: None,
                prompt: None,
                rate: DEFAULT_RATE,
                turn: Some(ServerVad::default()),
                beta,
            },
            id: format!("sess_{}", uuid::Uuid::new_v4().simple()),
            item: None,
            previous_item: None,
            detector: Detector::default(),
            stream_ms: 0.0,
            events: 0,
        }
    }

    async fn run(&mut self) {
        let created = self.session_event(if self.config.beta {
            "transcription_session.created"
        } else {
            "session.created"
        });
        if self.send(created).await.is_ok() {
            self.serve().await;
        }
        let _ = self.socket.close().await;
    }

    async fn serve(&mut self) {
        loop {
            let message = match tokio::time::timeout(IDLE_TIMEOUT, self.socket.recv()).await {
                Ok(Some(Ok(message))) => message,
                Ok(Some(Err(e))) => {
                    tracing::debug!("a realtime socket failed: {e}");
                    return;
                }
                Ok(None) | Err(_) => return,
            };
            let outcome = match message {
                Message::Text(text) => self.event(text.as_str()).await,
                Message::Binary(_) => Err(Outcome::Api(json_error(
                    "invalid_message",
                    "events are JSON text messages",
                    None,
                ))),
                Message::Close(_) => return,
                Message::Ping(_) | Message::Pong(_) => Ok(()),
            };
            match outcome {
                Ok(()) => {}
                Err(Outcome::Socket) => return,
                Err(Outcome::Api(error)) => {
                    if self.send_error(&error, None).await.is_err() {
                        return;
                    }
                }
            }
        }
    }

    async fn event(&mut self, text: &str) -> Step {
        let event: Value = serde_json::from_str(text).map_err(|e| {
            Outcome::Api(json_error(
                "invalid_json",
                format!("the event is not JSON: {e}"),
                None,
            ))
        })?;
        let event_id = event["event_id"].as_str().map(str::to_string);
        let outcome = match event["type"].as_str().unwrap_or_default() {
            "session.update" => self.update(&event["session"], false).await,
            "transcription_session.update" => self.update(&event["session"], true).await,
            "input_audio_buffer.append" => self.append(&event).await,
            "input_audio_buffer.commit" => self.commit().await,
            "input_audio_buffer.clear" => {
                self.item = None;
                self.detector = Detector::default();
                let cleared = self.event_json("input_audio_buffer.cleared", json!({}));
                self.send(cleared).await
            }
            other => Err(Outcome::Api(json_error(
                "unknown_event",
                format!("`{other}` is not a client event a transcription session takes"),
                Some("type"),
            ))),
        };
        match outcome {
            Err(Outcome::Api(error)) => self.send_error(&error, event_id.as_deref()).await,
            other => other,
        }
    }

    // the whole update is checked before any of it applies
    async fn update(&mut self, session: &Value, beta: bool) -> Step {
        let (format, transcription, turn, format_param) = if beta {
            (
                session.get("input_audio_format"),
                session.get("input_audio_transcription"),
                session.get("turn_detection"),
                "session.input_audio_format",
            )
        } else {
            if session["type"]
                .as_str()
                .is_some_and(|t| t != TRANSCRIPTION_INTENT)
            {
                return Err(Outcome::Api(json_error(
                    "unsupported_session_type",
                    "only `transcription` sessions are served",
                    Some("session.type"),
                )));
            }
            let input = &session["audio"]["input"];
            (
                input.get("format"),
                input.get("transcription"),
                input.get("turn_detection"),
                "session.audio.input.format",
            )
        };
        let rate = match format {
            None => self.config.rate,
            Some(format) => input_rate(format, format_param).map_err(Outcome::Api)?,
        };
        if self.item.is_some() && rate != self.config.rate {
            return Err(Outcome::Api(json_error(
                "invalid_value",
                "the audio format cannot change with audio in the buffer; commit or clear it first",
                Some(format_param),
            )));
        }
        let turn = match turn {
            None => self.config.turn,
            Some(turn) => turn_detection(turn).map_err(Outcome::Api)?,
        };
        let text = |field: &str| {
            transcription
                .and_then(|t| t.get(field))
                .map(|v| v.as_str().map(str::to_string))
        };
        self.config.beta = beta;
        if rate != self.config.rate {
            self.detector = Detector::default();
        }
        self.config.rate = rate;
        self.config.turn = turn;
        if let Some(Some(model)) = text("model") {
            self.config.model = served(&self.engine, Some(&model));
            self.config.requested_model = Some(model);
        }
        if let Some(language) = text("language") {
            self.config.language = language;
        }
        if let Some(prompt) = text("prompt") {
            self.config.prompt = prompt;
        }
        let updated = self.session_event(if beta {
            "transcription_session.updated"
        } else {
            "session.updated"
        });
        self.send(updated).await
    }

    async fn append(&mut self, event: &Value) -> Step {
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(event["audio"].as_str().unwrap_or_default())
            .map_err(|e| {
                Outcome::Api(json_error(
                    "invalid_value",
                    format!("`audio` is not base64: {e}"),
                    Some("audio"),
                ))
            })?;
        if bytes.is_empty() || !bytes.len().is_multiple_of(PCM16_BYTES) {
            return Err(Outcome::Api(json_error(
                "invalid_value",
                "`audio` is a non-empty run of 16-bit PCM samples",
                Some("audio"),
            )));
        }
        match self.config.turn {
            None => {
                self.stream_ms += self.ms(bytes.len());
                self.push(bytes).await
            }
            Some(vad) => self.detect(bytes, vad).await,
        }
    }

    // scans the audio in frames before any of it is pushed, so a turn ending mid-append keeps its own audio and the
    // next turn its own; a turn opens on speech, with its padding before it
    async fn detect(&mut self, bytes: Vec<u8>, vad: ServerVad) -> Step {
        let frame_bytes =
            (u64::from(self.config.rate) * VAD_FRAME_MS / 1000) as usize * PCM16_BYTES;
        let prefix_bytes =
            (u64::from(self.config.rate) * vad.prefix_padding_ms / 1000) as usize * PCM16_BYTES;
        let mut data = std::mem::take(&mut self.detector.partial);
        data.extend(bytes);
        let whole = data.len() / frame_bytes * frame_bytes;
        self.detector.partial = data.split_off(whole);
        let mut pending = Vec::new();
        for frame in data.chunks_exact(frame_bytes) {
            let start_ms = self.stream_ms;
            self.stream_ms += self.ms(frame.len());
            let loud = rms(frame) >= vad.level();
            if !self.detector.speaking {
                self.detector.prefix.extend_from_slice(frame);
                let over = self
                    .detector
                    .prefix
                    .len()
                    .saturating_sub(prefix_bytes.max(frame.len()));
                self.detector.prefix.drain(..over);
                if loud {
                    self.detector.speaking = true;
                    self.detector.last_loud_ms = self.stream_ms;
                    self.open_item().await?;
                    let start = (start_ms - vad.prefix_padding_ms as f64).max(0.0);
                    let started = self.event_json(
                        "input_audio_buffer.speech_started",
                        json!({"item_id": self.item_id(), "audio_start_ms": start as u64}),
                    );
                    self.send(started).await?;
                    pending = std::mem::take(&mut self.detector.prefix);
                }
                continue;
            }
            pending.extend_from_slice(frame);
            if loud {
                self.detector.last_loud_ms = self.stream_ms;
            } else if self.stream_ms - self.detector.last_loud_ms >= vad.silence_ms as f64 {
                self.push(std::mem::take(&mut pending)).await?;
                let stopped = self.event_json(
                    "input_audio_buffer.speech_stopped",
                    json!({"item_id": self.item_id(), "audio_end_ms": self.stream_ms as u64}),
                );
                self.send(stopped).await?;
                self.commit().await?;
            }
        }
        if !pending.is_empty() {
            self.push(pending).await?;
        }
        Ok(())
    }

    fn ms(&self, bytes: usize) -> f64 {
        (bytes / PCM16_BYTES) as f64 * 1000.0 / f64::from(self.config.rate)
    }

    fn item_id(&self) -> Value {
        json!(self.item.as_ref().map(|item| item.id.clone()))
    }

    async fn open_item(&mut self) -> Step {
        let request = TranscriptionSessionRequest {
            model: self.config.model.clone(),
            language: self.config.language.clone(),
            response_format: TranscriptionResponseFormat::Json,
            ..TranscriptionSessionRequest::new(self.config.rate)
        };
        let session = self
            .engine
            .transcription_session(request)
            .await
            .map_err(Outcome::Api)?;
        let id = format!("item_{}", uuid::Uuid::new_v4().simple());
        let item = json!({
            "id": id,
            "object": "realtime.item",
            "type": "message",
            "status": "in_progress",
            "role": "user",
            "content": [{"type": "input_audio", "transcript": null}],
        });
        let added = self.event_json(
            if self.config.beta {
                "conversation.item.created"
            } else {
                "conversation.item.added"
            },
            json!({"previous_item_id": self.previous_item, "item": item}),
        );
        self.item = Some(Item {
            id,
            session,
            samples: 0,
        });
        self.send(added).await
    }

    // audio into the open item, opening one first when none is; a failed push fails the item
    async fn push(&mut self, bytes: Vec<u8>) -> Step {
        if self.item.is_none() {
            self.open_item().await?;
        }
        let item = self.item.as_mut().expect("opened above");
        item.samples += (bytes.len() / PCM16_BYTES) as u64;
        let id = item.id.clone();
        match item.session.push_pcm16(&bytes).await {
            Ok(delta) if delta.is_empty() => Ok(()),
            Ok(delta) => {
                let event = self.event_json(
                    "conversation.item.input_audio_transcription.delta",
                    json!({"item_id": id, "content_index": 0, "delta": delta}),
                );
                self.send(event).await
            }
            Err(error) => {
                self.item = None;
                self.detector.speaking = false;
                self.failed(&id, &error).await
            }
        }
    }

    async fn commit(&mut self) -> Step {
        // a manual commit takes the partial frame server VAD had not scanned yet
        let partial = std::mem::take(&mut self.detector.partial);
        if self.item.is_some() && !partial.is_empty() {
            self.stream_ms += self.ms(partial.len());
            self.push(partial).await?;
        }
        self.detector.speaking = false;
        self.detector.prefix.clear();
        let Some(item) = self.item.take() else {
            return Err(Outcome::Api(json_error(
                "input_audio_buffer_commit_empty",
                "the input audio buffer is empty",
                None,
            )));
        };
        let committed = self.event_json(
            "input_audio_buffer.committed",
            json!({"item_id": item.id, "previous_item_id": self.previous_item}),
        );
        self.send(committed).await?;
        self.previous_item = Some(item.id.clone());
        let seconds = item.samples as f64 / f64::from(self.config.rate);
        let id = item.id.clone();
        let (delta, output) = match item.session.finish().await {
            Ok(finished) => finished,
            Err(error) => return self.failed(&id, &error).await,
        };
        if !delta.is_empty() {
            let event = self.event_json(
                "conversation.item.input_audio_transcription.delta",
                json!({"item_id": id, "content_index": 0, "delta": delta}),
            );
            self.send(event).await?;
        }
        let transcript = serde_json::from_str::<Value>(&output.body)
            .ok()
            .and_then(|body| body["text"].as_str().map(str::to_string))
            .unwrap_or_default();
        let completed = self.event_json(
            "conversation.item.input_audio_transcription.completed",
            json!({
                "item_id": id,
                "content_index": 0,
                "transcript": transcript,
                "usage": {"type": "duration", "seconds": seconds},
            }),
        );
        self.send(completed).await
    }

    async fn failed(&mut self, item_id: &str, error: &ApiError) -> Step {
        let event = self.event_json(
            "conversation.item.input_audio_transcription.failed",
            json!({"item_id": item_id, "content_index": 0, "error": error_object(error)}),
        );
        self.send(event).await
    }

    fn session_event(&mut self, kind: &str) -> Value {
        let c = &self.config;
        let model = c.requested_model.clone().unwrap_or_else(|| c.model.clone());
        let transcription = json!({"model": model, "language": c.language, "prompt": c.prompt});
        let turn = c.turn.map(|vad| {
            json!({
                "type": "server_vad",
                "threshold": vad.threshold,
                "prefix_padding_ms": vad.prefix_padding_ms,
                "silence_duration_ms": vad.silence_ms,
            })
        });
        let session = if c.beta {
            json!({
                "id": self.id,
                "object": "realtime.transcription_session",
                "input_audio_format": "pcm16",
                "input_audio_transcription": transcription,
                "turn_detection": turn,
                "input_audio_noise_reduction": null,
                "include": null,
            })
        } else {
            json!({
                "id": self.id,
                "object": "realtime.transcription_session",
                "type": TRANSCRIPTION_INTENT,
                "audio": {"input": {
                    "format": {"type": "audio/pcm", "rate": c.rate},
                    "transcription": transcription,
                    "turn_detection": turn,
                    "noise_reduction": null,
                }},
                "include": null,
            })
        };
        self.event_json(kind, json!({"session": session}))
    }

    fn event_json(&mut self, kind: &str, fields: Value) -> Value {
        self.events += 1;
        let mut event = json!({"type": kind, "event_id": format!("event_{}", self.events)});
        if let (Some(event), Value::Object(fields)) = (event.as_object_mut(), fields) {
            event.extend(fields);
        }
        event
    }

    async fn send_error(&mut self, error: &ApiError, event_id: Option<&str>) -> Step {
        let mut object = error_object(error);
        object["event_id"] = json!(event_id);
        let event = self.event_json("error", json!({"error": object}));
        self.send(event).await
    }

    async fn send(&mut self, event: Value) -> Step {
        let sent = self.socket.send(Message::Text(event.to_string().into()));
        match tokio::time::timeout(SEND_TIMEOUT, sent).await {
            Ok(Ok(())) => Ok(()),
            _ => Err(Outcome::Socket),
        }
    }
}

// a client names OpenAI's models; one this server does not serve is answered by its default model
fn served(engine: &Engine, model: Option<&str>) -> String {
    match model {
        Some(model) if engine.model_served(model) => model.to_string(),
        Some(model) => {
            tracing::info!(
                "realtime transcription names `{model}`, which is not served here; the default model serves it"
            );
            DEFAULT_MODEL_ID.to_string()
        }
        None => DEFAULT_MODEL_ID.to_string(),
    }
}

fn input_rate(format: &Value, param: &str) -> Result<u32, ApiError> {
    let rate = match format {
        Value::String(s) if s == "pcm16" => Some(DEFAULT_RATE),
        Value::Object(_) if format["type"] == "audio/pcm" => match format.get("rate") {
            None => Some(DEFAULT_RATE),
            Some(rate) => rate.as_u64().and_then(|r| u32::try_from(r).ok()),
        },
        _ => {
            return Err(json_error(
                "unsupported_audio_format",
                "only 16-bit PCM is taken (`audio/pcm`, or the beta's `pcm16`)",
                Some(param),
            ));
        }
    };
    rate.filter(|r| (MIN_RATE..=MAX_RATE).contains(r))
        .ok_or_else(|| {
            json_error(
                "invalid_value",
                format!("the PCM `rate` is a whole number of hertz, {MIN_RATE} to {MAX_RATE}"),
                Some(param),
            )
        })
}

fn turn_detection(turn: &Value) -> Result<Option<ServerVad>, ApiError> {
    match turn {
        Value::Null => Ok(None),
        Value::Object(_) if turn["type"] == "server_vad" => {
            let defaults = ServerVad::default();
            let vad = ServerVad {
                threshold: turn["threshold"].as_f64().unwrap_or(defaults.threshold),
                prefix_padding_ms: turn["prefix_padding_ms"]
                    .as_u64()
                    .unwrap_or(defaults.prefix_padding_ms),
                silence_ms: turn["silence_duration_ms"]
                    .as_u64()
                    .unwrap_or(defaults.silence_ms),
            };
            if !(0.0..=1.0).contains(&vad.threshold) {
                return Err(json_error(
                    "invalid_value",
                    "`threshold` is 0 to 1",
                    Some("turn_detection.threshold"),
                ));
            }
            Ok(Some(vad))
        }
        _ => Err(json_error(
            "unsupported_turn_detection",
            "turn detection is `server_vad` (an energy detector here) or null for manual commits",
            Some("turn_detection"),
        )),
    }
}
