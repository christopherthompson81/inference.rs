//! Whole recordings transcribed and diarized as streams of events: a local file decoded as it reads, each window's
//! words or each chunk's progress as the model finishes it, stoppable between them.

use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::Stream;
use inference_core::{
    AudioInput, Diarization, DiarizationOptions, DiarizationProgressHook, RequestCancellation,
    RequestMessage, Response, TranscriptionOptions, TranscriptionProgress,
    TranscriptionProgressHook,
};
use serde_json::{Value, json};
use tokio::sync::mpsc;

use crate::{
    api_error::{ApiError, ApiErrorKind},
    generation::{
        audio_error, check_beam, decode_audio, render_diarization, render_transcript, run,
        speaker_turns, unexpected,
    },
    lora_routing::DEFAULT_MODEL_ID,
    model_inventory::{CancelOnDrop, event},
    openai::{
        DiarizationRequest, DiarizationResponseFormat, TimestampGranularity, TranscriptionRequest,
        TranscriptionResponseFormat, TranscriptionTask,
    },
    overlap_pass::{PassContext, noting_windows, overlap_pass},
    request_body::parse_json,
    types::SharedInferenceRsState,
};

/// The rate a file is decoded to: the speech models' own (another model resamples from it).
const STREAM_SAMPLE_RATE: u32 = 16_000;
/// The longest recording a stream reads: 12 hours at [`STREAM_SAMPLE_RATE`], ~2.8 GB decoded.
const MAX_STREAM_SAMPLES: usize = 12 * 60 * 60 * STREAM_SAMPLE_RATE as usize;
// the request fields a stream takes beyond its request's own
const PATH_FIELD: &str = "path";
const FROM_SECONDS_FIELD: &str = "from_seconds";
const MAX_WINDOW_FIELD: &str = "max_window_seconds";
const SPANS_FIELD: &str = "spans";
const TURNS_FIELD: &str = "speaker_turns";
// the shortest window a transcription stream plans: under ~5 s Parakeet, which identifies the language itself, can
// switch language from one window to the next
const MIN_WINDOW_SECONDS: f64 = 5.0;

/// A speech stream's events, each `{"event": <name>, "data": ...}`, and what stops it.
pub struct SpeechStream {
    pub events: Pin<Box<dyn Stream<Item = String> + Send>>,
    pub cancellation: RequestCancellation,
}

// where a stream's audio is, and the second it starts from
struct Source {
    path: Option<PathBuf>,
    from_seconds: f64,
    max_window_seconds: Option<f64>,
    spans: Option<Vec<(f64, f64)>>,
}

// `body`'s `path` and `from_seconds`, and the rest of it; `audio` is the bytes the caller gave, if any
fn split_source(body: &[u8], audio: &Option<Vec<u8>>) -> Result<(Source, Vec<u8>), ApiError> {
    let mut fields: serde_json::Map<String, Value> =
        serde_json::from_slice(body).map_err(|error| {
            ApiError::invalid_request(format!("the request is not a JSON object: {error}"))
        })?;
    let path = match fields.remove(PATH_FIELD) {
        None | Some(Value::Null) => None,
        Some(Value::String(path)) => Some(PathBuf::from(path)),
        Some(_) => return Err(ApiError::invalid_request("`path` must be a string")),
    };
    if path.is_some() == audio.is_some() {
        return Err(ApiError::invalid_request(
            "give the audio as bytes or as a `path`, one of them",
        ));
    }
    let from_seconds = match fields.remove(FROM_SECONDS_FIELD) {
        None | Some(Value::Null) => 0.0,
        Some(value) => value
            .as_f64()
            .filter(|s| s.is_finite() && *s >= 0.0)
            .ok_or_else(|| {
                ApiError::invalid_request("`from_seconds` must be a number of seconds, 0 or more")
            })?,
    };
    let max_window_seconds = match fields.remove(MAX_WINDOW_FIELD) {
        None | Some(Value::Null) => None,
        Some(value) => Some(
            value
                .as_f64()
                .filter(|s| s.is_finite() && *s >= MIN_WINDOW_SECONDS)
                .ok_or_else(|| {
                    ApiError::invalid_request(format!(
                        "`max_window_seconds` must be a number of seconds, {MIN_WINDOW_SECONDS} or more"
                    ))
                })?,
        ),
    };
    let spans = match fields.remove(SPANS_FIELD) {
        None | Some(Value::Null) => None,
        Some(value) => Some(spans(&value)?),
    };
    if spans.is_some() && max_window_seconds.is_some() {
        return Err(ApiError::invalid_request(
            "`spans` are the windows; give them or `max_window_seconds`, not both",
        ));
    }
    let rest = serde_json::to_vec(&fields).map_err(|_| ApiError::internal())?;
    Ok((
        Source {
            path,
            from_seconds,
            max_window_seconds,
            spans,
        },
        rest,
    ))
}

// `spans`: `[start, end]` seconds, each start under its end and none before the last one's
fn spans(value: &Value) -> Result<Vec<(f64, f64)>, ApiError> {
    let invalid = || {
        ApiError::invalid_request(
            "`spans` must be a non-empty list of [start, end] seconds, 0 <= start < end, starts in order",
        )
    };
    let list = value
        .as_array()
        .filter(|l| !l.is_empty())
        .ok_or_else(invalid)?;
    let mut out: Vec<(f64, f64)> = Vec::with_capacity(list.len());
    for span in list {
        let pair = span
            .as_array()
            .filter(|p| p.len() == 2)
            .ok_or_else(invalid)?;
        let (Some(start), Some(end)) = (pair[0].as_f64(), pair[1].as_f64()) else {
            return Err(invalid());
        };
        let after_last = out.last().is_none_or(|&(last, _)| start >= last);
        if !(start.is_finite() && end.is_finite() && 0.0 <= start && start < end && after_last) {
            return Err(invalid());
        }
        out.push((start, end));
    }
    Ok(out)
}

/// Transcribes a whole recording, `audio` bytes or the request's `path` (a local file, decoded as it reads), as a
/// stream: `started` (data: `{"duration", "windows": [{"index", "start", "end"}]}`), then a `window` (data:
/// `{"index", "start", "end", "text", "words": [{"word", "start", "end"}]}`) and a `progress` (data: `{"done",
/// "total", "seconds_done"}`) as each window finishes, and last `done` (data: the verbose_json transcript, words and
/// segments; with `diarization_model` speakers and its `speaker_turns` over the whole recording), `canceled` or
/// `error` (data: the error envelope). Times are seconds into the recording. `from_seconds` resumes at a window's end:
/// the windows are planned over the whole recording as before and those ending by then skipped, so the rest (and
/// `done`, which holds only them) match the uninterrupted run. A model that does not window its audio sends only the
/// last event. Canceling stops it before its next window; dropping the stream cancels it.
pub(crate) fn transcription_events(
    state: SharedInferenceRsState,
    body: &[u8],
    audio: Option<Vec<u8>>,
) -> Result<SpeechStream, ApiError> {
    let (source, rest) = split_source(body, &audio)?;
    let mut request: TranscriptionRequest = parse_json(&rest)?;
    check_beam(&request)?;
    let model = (request.model != DEFAULT_MODEL_ID).then_some(request.model.as_str());
    let decodes = state
        .audio_transcriber(model)
        .map_err(|error| ApiError::from_error(&error, ApiErrorKind::InvalidRequest))?
        .is_some();
    if decodes {
        return Err(ApiError::invalid_request(
            "this model transcribes a recording whole; it does not stream one",
        ));
    }
    request.response_format = TranscriptionResponseFormat::VerboseJson;
    request.timestamp_granularities =
        vec![TimestampGranularity::Word, TimestampGranularity::Segment];
    Ok(stream_of(|events, canceled| async move {
        transcribe(&state, request, source, audio, &events, &canceled).await
    }))
}

/// Diarizes a whole recording, `audio` bytes or the request's `path`, as a stream: `started` (data: `{"duration"}`),
/// a `progress` after each of the model's chunks (data: `{"seconds_done", "duration"}`), and last `done` (data: the
/// json response, `{"duration", "num_speakers", "segments"}`), `canceled` or `error`. One pass over the whole
/// recording, so a speaker keeps its label throughout. `from_seconds` starts there, times staying the recording's.
/// Canceling stops it before its next chunk; dropping the stream cancels it.
pub(crate) fn diarization_events(
    state: SharedInferenceRsState,
    body: &[u8],
    audio: Option<Vec<u8>>,
) -> Result<SpeechStream, ApiError> {
    let (source, rest) = split_source(body, &audio)?;
    let mut request: DiarizationRequest = parse_json(&rest)?;
    if request.threshold.is_some_and(|t| !(0.0..=1.0).contains(&t)) {
        return Err(ApiError::invalid_request(
            "`threshold` must be a probability, from 0 to 1",
        ));
    }
    request.response_format = DiarizationResponseFormat::Json;
    Ok(stream_of(|events, canceled| async move {
        diarize(&state, request, source, audio, &events, &canceled).await
    }))
}

// a stream of what `work` sends, and last `done` with what it returns (`canceled` or `error` if it fails)
fn stream_of<Work, Fut>(work: Work) -> SpeechStream
where
    Work: FnOnce(mpsc::UnboundedSender<String>, RequestCancellation) -> Fut,
    Fut: std::future::Future<Output = Result<Value, ApiError>> + Send + 'static,
{
    let (events, receiver) = mpsc::unbounded_channel();
    let cancellation = RequestCancellation::default();
    let canceled = cancellation.clone();
    let work = work(events.clone(), cancellation.clone());
    tokio::spawn(async move {
        let last = match work.await {
            Ok(done) => event("done", done),
            Err(_) if canceled.is_canceled() => event("canceled", Value::Null),
            Err(error) => event("error", error.to_openai_body()),
        };
        let _ = events.send(last);
    });
    let events = futures::stream::unfold(receiver, |mut receiver| async move {
        receiver.recv().await.map(|event| (event, receiver))
    });
    SpeechStream {
        events: Box::pin(CancelOnDrop {
            events: Box::pin(events),
            cancellation: cancellation.clone(),
        }),
        cancellation,
    }
}

// the stream's audio, decoded off the runtime; cut to `from_seconds` with `trim`, else only checked against it
async fn load_audio(
    source: Source,
    bytes: Option<Vec<u8>>,
    trim: bool,
    canceled: &RequestCancellation,
) -> Result<AudioInput, ApiError> {
    let path = source.path;
    let audio = tokio::task::spawn_blocking(move || match (path, bytes) {
        (Some(path), _) => {
            AudioInput::from_file_mono(&path, STREAM_SAMPLE_RATE, MAX_STREAM_SAMPLES)
                .map_err(audio_error)
        }
        (None, Some(bytes)) => decode_audio(&bytes),
        (None, None) => unreachable!("checked when the stream opened"),
    })
    .await
    .map_err(|_| ApiError::internal())??;
    let audio = skip_seconds(audio, source.from_seconds, trim)?;
    if canceled.is_canceled() {
        return Err(ApiError::invalid_request("canceled"));
    }
    Ok(audio)
}

// the transcription stream's work: its audio loaded, transcribed with its windows reported, rendered
async fn transcribe(
    state: &SharedInferenceRsState,
    request: TranscriptionRequest,
    mut source: Source,
    bytes: Option<Vec<u8>>,
    events: &mpsc::UnboundedSender<String>,
    canceled: &RequestCancellation,
) -> Result<Value, ApiError> {
    let resume_at = (source.from_seconds > 0.0).then_some(source.from_seconds);
    let max_window_seconds = source.max_window_seconds;
    let spans = source.spans.take();
    let audio = load_audio(source, bytes, false, canceled).await?;
    let (progress, windowed) = noting_windows(Some(report(events.clone(), canceled.clone())));
    let repr = serde_json::to_string(&request).map_err(|_| ApiError::internal())?;
    let options = TranscriptionOptions {
        language: request.language.clone(),
        beam_size: request.beam_size,
        translate: request.task == TranscriptionTask::Translate,
        resume_at,
        max_window_seconds,
        spans,
    };
    let message = |audio| RequestMessage::Transcription {
        audio,
        options: options.clone(),
        progress: Some(progress.clone()),
    };
    let Some(diarizer) = request.diarization_model.clone() else {
        let Response::Transcription(transcript) =
            run(state, &request.model, repr, message(audio)).await?
        else {
            return Err(unexpected(state));
        };
        let output = render_transcript(&request, transcript, None)?;
        return serde_json::from_str(&output.body).map_err(|_| ApiError::internal());
    };
    let (transcript, turns) = futures::join!(
        run(state, &request.model, repr.clone(), message(audio.clone())),
        speaker_turns(state, &diarizer, audio.clone()),
    );
    let (Response::Transcription(transcript), Response::Diarization(turns)) = (transcript?, turns?)
    else {
        return Err(unexpected(state));
    };
    let ctx = PassContext {
        state,
        model: &request.model,
        repr,
        options: &options,
        windowed: windowed.load(Ordering::Relaxed),
        // the caller's own spans are its own second pass
        canceled: Some(canceled),
        enabled: request.overlap_pass != Some(false) && options.spans.is_none(),
    };
    let (transcript, speakers) = overlap_pass(ctx, audio, transcript, &turns.segments).await?;
    let output = render_transcript(&request, transcript, Some(&speakers))?;
    let mut done: Value = serde_json::from_str(&output.body).map_err(|_| ApiError::internal())?;
    // the whole recording's turns, so a resumed caller can attribute the words it kept from before
    done[TURNS_FIELD] = serde_json::to_value(&turns.segments).map_err(|_| ApiError::internal())?;
    Ok(done)
}

// the diarization stream's work: its audio loaded, diarized with each chunk reported, rendered
async fn diarize(
    state: &SharedInferenceRsState,
    request: DiarizationRequest,
    source: Source,
    bytes: Option<Vec<u8>>,
    events: &mpsc::UnboundedSender<String>,
    canceled: &RequestCancellation,
) -> Result<Value, ApiError> {
    let offset = source.from_seconds;
    let audio = load_audio(source, bytes, true, canceled).await?;
    let frames = audio.samples.len() / usize::from(audio.channels.max(1));
    let duration = offset + frames as f64 / f64::from(audio.sample_rate);
    let _ = events.send(event("started", json!({ "duration": duration })));
    let (sender, stop) = (events.clone(), canceled.clone());
    let progress = DiarizationProgressHook(Arc::new(move |done: f64, _total: f64| {
        let _ = sender.send(event(
            "progress",
            json!({ "seconds_done": (done + offset).min(duration), "duration": duration }),
        ));
        !stop.is_canceled()
    }));
    let repr = serde_json::to_string(&request).map_err(|_| ApiError::internal())?;
    let message = RequestMessage::Diarization {
        audio,
        options: DiarizationOptions {
            threshold: request.threshold,
        },
        progress: Some(progress),
    };
    let Response::Diarization(diarization) = run(state, &request.model, repr, message).await?
    else {
        return Err(unexpected(state));
    };
    let output = render_diarization(&request, diarization_shifted(diarization, offset))?;
    serde_json::from_str(&output.body).map_err(|_| ApiError::internal())
}

// `audio` from `seconds` in
fn skip_seconds(mut audio: AudioInput, seconds: f64, trim: bool) -> Result<AudioInput, ApiError> {
    let channels = usize::from(audio.channels.max(1));
    let frames = audio.samples.len() / channels;
    let skip = (seconds * f64::from(audio.sample_rate)).round() as usize;
    if skip > 0 && skip >= frames {
        return Err(ApiError::invalid_request(format!(
            "`from_seconds` {seconds} is past the recording's end ({:.2} s)",
            frames as f64 / f64::from(audio.sample_rate)
        )));
    }
    if trim {
        audio.samples.drain(..skip * channels);
    }
    Ok(audio)
}

// a diarization of audio that started `offset` seconds into its recording, timed in the recording
fn diarization_shifted(mut diarization: Diarization, offset: f64) -> Diarization {
    for segment in &mut diarization.segments {
        segment.start += offset;
        segment.end += offset;
    }
    diarization.duration += offset;
    diarization
}

// the model's progress as `started`, `window` and `progress` events; it keeps going while the stream is not canceled
fn report(
    events: mpsc::UnboundedSender<String>,
    canceled: RequestCancellation,
) -> TranscriptionProgressHook {
    let total = AtomicUsize::new(0);
    TranscriptionProgressHook(Arc::new(move |progress: TranscriptionProgress<'_>| {
        match progress {
            TranscriptionProgress::Started { duration, windows } => {
                total.store(windows.len(), Ordering::Relaxed);
                let windows: Vec<Value> = windows
                    .iter()
                    .enumerate()
                    .map(
                        |(index, (start, end))| json!({"index": index, "start": start, "end": end}),
                    )
                    .collect();
                let _ = events.send(event(
                    "started",
                    json!({"duration": duration, "windows": windows}),
                ));
            }
            TranscriptionProgress::Window {
                index,
                start,
                end,
                transcript,
            } => {
                let words: Vec<Value> = transcript
                    .words
                    .iter()
                    .map(|w| json!({"word": w.text, "start": w.start, "end": w.end}))
                    .collect();
                let _ = events.send(event(
                    "window",
                    json!({
                        "index": index,
                        "start": start,
                        "end": end,
                        "text": transcript.text,
                        "words": words,
                    }),
                ));
                let _ = events.send(event(
                    "progress",
                    json!({
                        "done": index + 1,
                        "total": total.load(Ordering::Relaxed),
                        "seconds_done": end,
                    }),
                ));
            }
        }
        !canceled.is_canceled()
    }))
}
