//! Whole recordings transcribed as streams of events: a local file decoded as it reads, each window's words as the
//! model finishes it, stoppable between windows.

use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::Stream;
use inference_core::{
    AudioInput, RequestCancellation, RequestMessage, Response, Transcription, TranscriptionOptions,
    TranscriptionProgress, TranscriptionProgressHook,
};
use serde_json::{Value, json};
use tokio::sync::mpsc;

use crate::{
    api_error::{ApiError, ApiErrorKind},
    generation::{
        audio_error, check_beam, decode_audio, render_transcript, run, speaker_turns, unexpected,
        word_speakers,
    },
    lora_routing::DEFAULT_MODEL_ID,
    model_inventory::{CancelOnDrop, event},
    openai::{
        TimestampGranularity, TranscriptionRequest, TranscriptionResponseFormat, TranscriptionTask,
    },
    request_body::parse_json,
    types::SharedInferenceRsState,
};

/// The rate a file is decoded to: the speech models' own (another model resamples from it).
const STREAM_SAMPLE_RATE: u32 = 16_000;
/// The longest recording a stream reads: 12 hours at [`STREAM_SAMPLE_RATE`], ~2.8 GB decoded.
const MAX_STREAM_SAMPLES: usize = 12 * 60 * 60 * STREAM_SAMPLE_RATE as usize;
// the request fields a stream takes beyond a transcription request's
const PATH_FIELD: &str = "path";
const FROM_SECONDS_FIELD: &str = "from_seconds";

/// A speech stream's events, each `{"event": <name>, "data": ...}`, and what stops it.
pub struct SpeechStream {
    pub events: Pin<Box<dyn Stream<Item = String> + Send>>,
    pub cancellation: RequestCancellation,
}

// where a stream's audio is, and the second it starts from
struct Source {
    path: Option<PathBuf>,
    from_seconds: f64,
}

// `body`'s `path` and `from_seconds`, and the rest of it
fn split_source(body: &[u8]) -> Result<(Source, Vec<u8>), ApiError> {
    let mut fields: serde_json::Map<String, Value> =
        serde_json::from_slice(body).map_err(|error| {
            ApiError::invalid_request(format!("the request is not a JSON object: {error}"))
        })?;
    let path = match fields.remove(PATH_FIELD) {
        None | Some(Value::Null) => None,
        Some(Value::String(path)) => Some(PathBuf::from(path)),
        Some(_) => return Err(ApiError::invalid_request("`path` must be a string")),
    };
    let from_seconds = match fields.remove(FROM_SECONDS_FIELD) {
        None | Some(Value::Null) => 0.0,
        Some(value) => value
            .as_f64()
            .filter(|s| s.is_finite() && *s >= 0.0)
            .ok_or_else(|| {
                ApiError::invalid_request("`from_seconds` must be a number of seconds, 0 or more")
            })?,
    };
    let rest = serde_json::to_vec(&fields).map_err(|_| ApiError::internal())?;
    Ok((Source { path, from_seconds }, rest))
}

/// Transcribes a whole recording, `audio` bytes or the request's `path` (a local file, decoded as it reads), as a
/// stream: `started` (data: `{"duration", "windows": [{"index", "start", "end"}]}`), then a `window` (data:
/// `{"index", "start", "end", "text", "words": [{"word", "start", "end"}]}`) and a `progress` (data: `{"done",
/// "total", "seconds_done"}`) as each window finishes, and last `done` (data: the verbose_json transcript, words and
/// segments, speakers with `diarization_model`), `canceled` or `error` (data: the error envelope). Times are seconds
/// into the recording; `from_seconds` starts there, for resuming. A model that does not window its audio sends only
/// the last event. Canceling stops it before its next window; dropping the stream cancels it.
pub(crate) fn transcription_events(
    state: SharedInferenceRsState,
    body: &[u8],
    audio: Option<Vec<u8>>,
) -> Result<SpeechStream, ApiError> {
    let (source, rest) = split_source(body)?;
    let mut request: TranscriptionRequest = parse_json(&rest)?;
    check_beam(&request)?;
    if source.path.is_some() == audio.is_some() {
        return Err(ApiError::invalid_request(
            "give the audio as bytes or as a `path`, one of them",
        ));
    }
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
    let (events, receiver) = mpsc::unbounded_channel();
    let cancellation = RequestCancellation::default();
    let canceled = cancellation.clone();
    tokio::spawn(async move {
        let last = match transcribe(&state, request, source, audio, &events, &canceled).await {
            Ok(done) => event("done", done),
            Err(_) if canceled.is_canceled() => event("canceled", Value::Null),
            Err(error) => event("error", error.to_openai_body()),
        };
        let _ = events.send(last);
    });
    let events = futures::stream::unfold(receiver, |mut receiver| async move {
        receiver.recv().await.map(|event| (event, receiver))
    });
    Ok(SpeechStream {
        events: Box::pin(CancelOnDrop {
            events: Box::pin(events),
            cancellation: cancellation.clone(),
        }),
        cancellation,
    })
}

// the stream's work: its audio decoded and cut to `from_seconds`, transcribed with its windows reported, rendered
async fn transcribe(
    state: &SharedInferenceRsState,
    request: TranscriptionRequest,
    source: Source,
    bytes: Option<Vec<u8>>,
    events: &mpsc::UnboundedSender<String>,
    canceled: &RequestCancellation,
) -> Result<Value, ApiError> {
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
    let offset = source.from_seconds;
    let audio = skip_seconds(audio, offset)?;
    if canceled.is_canceled() {
        return Err(ApiError::invalid_request("canceled"));
    }
    let progress = report(events.clone(), canceled.clone(), offset);
    let repr = serde_json::to_string(&request).map_err(|_| ApiError::internal())?;
    let message = |audio| RequestMessage::Transcription {
        audio,
        options: TranscriptionOptions {
            language: request.language.clone(),
            beam_size: request.beam_size,
            translate: request.task == TranscriptionTask::Translate,
        },
        progress: Some(progress.clone()),
    };
    let (transcript, speakers) = match request.diarization_model.clone() {
        None => (
            run(state, &request.model, repr, message(audio)).await?,
            None,
        ),
        Some(diarizer) => {
            let (transcript, turns) = futures::join!(
                run(state, &request.model, repr, message(audio.clone())),
                speaker_turns(state, &diarizer, audio),
            );
            let Response::Diarization(turns) = turns? else {
                return Err(unexpected(state));
            };
            (transcript?, Some(turns.segments))
        }
    };
    let Response::Transcription(transcript) = transcript else {
        return Err(unexpected(state));
    };
    let speakers = speakers.map(|turns| word_speakers(&transcript.words, &turns));
    let output = render_transcript(&request, shifted(transcript, offset), speakers.as_deref())?;
    serde_json::from_str(&output.body).map_err(|_| ApiError::internal())
}

// `audio` from `seconds` in
fn skip_seconds(mut audio: AudioInput, seconds: f64) -> Result<AudioInput, ApiError> {
    let channels = usize::from(audio.channels.max(1));
    let frames = audio.samples.len() / channels;
    let skip = (seconds * f64::from(audio.sample_rate)).round() as usize;
    if skip > 0 && skip >= frames {
        return Err(ApiError::invalid_request(format!(
            "`from_seconds` {seconds} is past the recording's end ({:.2} s)",
            frames as f64 / f64::from(audio.sample_rate)
        )));
    }
    audio.samples.drain(..skip * channels);
    Ok(audio)
}

// a transcript of audio that started `offset` seconds into its recording, timed in the recording
fn shifted(mut transcript: Transcription, offset: f64) -> Transcription {
    for timed in transcript
        .words
        .iter_mut()
        .chain(transcript.tokens.iter_mut())
    {
        timed.start += offset;
        timed.end += offset;
    }
    transcript.duration += offset;
    transcript
}

// the model's progress as `started`, `window` and `progress` events, times `offset` into the recording; it keeps
// going while the stream is not canceled
fn report(
    events: mpsc::UnboundedSender<String>,
    canceled: RequestCancellation,
    offset: f64,
) -> TranscriptionProgressHook {
    let total = AtomicUsize::new(0);
    TranscriptionProgressHook(Arc::new(move |progress: TranscriptionProgress<'_>| {
        match progress {
            TranscriptionProgress::Started { duration, windows } => {
                total.store(windows.len(), Ordering::Relaxed);
                let windows: Vec<Value> = windows
                    .iter()
                    .enumerate()
                    .map(|(index, (start, end))| {
                        json!({"index": index, "start": start + offset, "end": end + offset})
                    })
                    .collect();
                let _ = events.send(event(
                    "started",
                    json!({"duration": duration + offset, "windows": windows}),
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
                    .map(|w| json!({"word": w.text, "start": w.start + offset, "end": w.end + offset}))
                    .collect();
                let _ = events.send(event(
                    "window",
                    json!({
                        "index": index,
                        "start": start + offset,
                        "end": end + offset,
                        "text": transcript.text,
                        "words": words,
                    }),
                ));
                let _ = events.send(event(
                    "progress",
                    json!({
                        "done": index + 1,
                        "total": total.load(Ordering::Relaxed),
                        "seconds_done": end + offset,
                    }),
                ));
            }
        }
        !canceled.is_canceled()
    }))
}
