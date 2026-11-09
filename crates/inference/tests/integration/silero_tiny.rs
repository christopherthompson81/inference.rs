//! Silero VAD through the engine on a tiny random-weight GGUF, and Parakeet's long-form transcription through it.

use inference::{
    ApiErrorKind, ModelDType, TranscriptionModelBuilder, TranscriptionRequest,
    TranscriptionResponseFormat, VerboseTranscriptionResponse, VoiceActivityModelBuilder,
    VoiceActivityRequest,
};

#[path = "../support/silero_tiny.rs"]
mod support;
use support::tiny_silero_gguf;

#[path = "../support/parakeet_tiny.rs"]
#[allow(clippy::duplicate_mod)]
mod parakeet_support;

const RATE: u32 = 16_000;
const CHUNK: usize = 512;
const TONE_HZ: f32 = 330.0;
// past Parakeet's 5-minute long-form threshold, so the VAD cuts it into 2-minute windows
const LONG_SECONDS: f64 = 301.0;
const WINDOW_SECONDS: f64 = 120.0;

fn tone_wav(seconds: f64) -> Vec<u8> {
    let pcm: Vec<f32> = (0..(seconds * f64::from(RATE)) as usize)
        .map(|i| (2.0 * std::f32::consts::PI * TONE_HZ * i as f32 / RATE as f32).sin() * 0.3)
        .collect();
    let mut wav = Vec::new();
    inference_models_speech::utils::write_pcm_as_wav(&mut wav, &pcm, RATE, 1).unwrap();
    wav
}

#[tokio::test]
async fn the_vad_scores_every_chunk_and_its_thresholds_cut_the_segments() -> anyhow::Result<()> {
    let dir = tiny_silero_gguf(false)?;
    let model = VoiceActivityModelBuilder::new(dir.path().to_string_lossy())
        .with_force_cpu()
        .build()
        .await?;
    let seconds = 3.0;
    let wav = tone_wav(seconds);
    let mut request = VoiceActivityRequest::new();
    request.return_probabilities = true;
    let activity = model.voice_activity(request.clone(), &wav).await?;
    let probabilities = activity.probabilities.as_deref().unwrap_or_default();
    assert_eq!(
        probabilities.len(),
        (seconds * f64::from(RATE)) as usize / CHUNK + 1
    );
    assert!(probabilities.iter().all(|p| (0.0..=1.0).contains(p)));
    assert!((activity.duration - seconds).abs() < 1e-6);
    assert!((activity.chunk_seconds - CHUNK as f64 / f64::from(RATE)).abs() < 1e-9);

    // every chunk clears a zero threshold, so the whole clip is one segment; none clears one past 1
    request.threshold = Some(0.0);
    request.neg_threshold = Some(-1.0);
    let all = model.voice_activity(request.clone(), &wav).await?;
    assert_eq!(all.segments.len(), 1, "{:?}", all.segments);
    assert_eq!((all.segments[0].start, all.segments[0].end), (0.0, seconds));
    request.threshold = Some(1.1);
    assert!(
        model
            .voice_activity(request.clone(), &wav)
            .await?
            .segments
            .is_empty()
    );

    let err = model
        .voice_activity(VoiceActivityRequest::new(), b"not audio")
        .await
        .expect_err("garbage decoded");
    assert_eq!(err.kind, ApiErrorKind::InvalidRequest, "{err}");
    Ok(())
}

#[tokio::test]
async fn long_audio_transcribes_in_windows_cut_at_the_vads_silences() -> anyhow::Result<()> {
    let vad = tiny_silero_gguf(true)?;
    let checkpoint = parakeet_support::tiny_parakeet_checkpoint(parakeet_support::HEADS[0])?;
    let model = TranscriptionModelBuilder::new(checkpoint.path().to_string_lossy())
        .with_vad_model_id(vad.path().to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .build()
        .await?;
    let mut request = TranscriptionRequest::new(TranscriptionResponseFormat::VerboseJson);
    request.timestamp_granularities = vec![inference::TimestampGranularity::Word];
    let output = model
        .transcription(request, &tone_wav(LONG_SECONDS))
        .await?;
    let verbose: VerboseTranscriptionResponse = serde_json::from_str(&output.body)?;
    assert!((verbose.duration - LONG_SECONDS).abs() < 1e-6);
    let words = verbose.words.unwrap_or_default();
    // the always-speech VAD splits the one long run at the window length, so later windows' words carry their offset
    assert!(
        words.iter().any(|w| w.start > WINDOW_SECONDS),
        "{} words",
        words.len()
    );
    assert!(words.windows(2).all(|w| w[0].start <= w[1].start));
    assert!(words.iter().all(|w| w.end <= verbose.duration));
    Ok(())
}

// the events of a whole-recording transcription stream, read to its end, and their names
async fn stream_events(
    stream: inference_api::speech_streams::SpeechStream,
    cancel_at_window: Option<u64>,
) -> Vec<serde_json::Value> {
    use futures::StreamExt;
    let mut events = stream.events;
    let mut out = Vec::new();
    while let Some(event) = events.next().await {
        let event: serde_json::Value = serde_json::from_str(&event).unwrap();
        if event["event"] == "window" && event["data"]["index"].as_u64() == cancel_at_window {
            stream.cancellation.cancel();
        }
        out.push(event);
    }
    out
}

// a long recording streamed from a file: its windows up front, each window's words and its progress as it finishes,
// and last the transcript the one-shot request gives; canceled at its first window it stops before the last, and
// resumed at a window's end it runs the rest of the same windows as the whole run did
#[tokio::test]
async fn a_long_recording_streams_its_windows_as_they_finish() -> anyhow::Result<()> {
    let vad = tiny_silero_gguf(true)?;
    let checkpoint = parakeet_support::tiny_parakeet_checkpoint(parakeet_support::HEADS[0])?;
    let model = TranscriptionModelBuilder::new(checkpoint.path().to_string_lossy())
        .with_vad_model_id(vad.path().to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .build()
        .await?;
    let wav = tone_wav(LONG_SECONDS);
    let file = tempfile::NamedTempFile::with_suffix(".wav")?;
    std::fs::write(file.path(), &wav)?;
    let path = file.path().to_string_lossy().to_string();
    let mut one_shot = TranscriptionRequest::new(TranscriptionResponseFormat::VerboseJson);
    one_shot.timestamp_granularities = vec![
        inference::TimestampGranularity::Word,
        inference::TimestampGranularity::Segment,
    ];
    let whole: serde_json::Value =
        serde_json::from_str(&model.transcription(one_shot, &wav).await?.body)?;

    let request = serde_json::json!({ "path": path }).to_string();
    let events = stream_events(
        model
            .transcription_stream_json(request.as_bytes(), None)
            .await?,
        None,
    )
    .await;
    let all = events.clone();
    let names: Vec<&str> = events
        .iter()
        .map(|e| e["event"].as_str().unwrap())
        .collect();
    let windows = events[0]["data"]["windows"].as_array().unwrap().len();
    assert!(windows >= 2, "{windows} windows");
    let mut expected = vec!["started"];
    expected.extend(std::iter::repeat_n(["window", "progress"], windows).flatten());
    expected.push("done");
    assert_eq!(names, expected);
    for (i, window) in events.iter().filter(|e| e["event"] == "window").enumerate() {
        assert_eq!(window["data"]["index"], i);
    }
    let last_progress = &events[events.len() - 2]["data"];
    assert_eq!(
        (
            last_progress["done"].as_u64(),
            last_progress["total"].as_u64()
        ),
        (Some(windows as u64), Some(windows as u64))
    );
    let done = &events.last().unwrap()["data"];
    assert_eq!(done["text"], whole["text"]);
    assert_eq!(
        done["words"].as_array().map(Vec::len),
        whole["words"].as_array().map(Vec::len)
    );

    let canceled = stream_events(
        model
            .transcription_stream_json(request.as_bytes(), None)
            .await?,
        Some(0),
    )
    .await;
    assert_eq!(canceled.last().unwrap()["event"], "canceled");
    assert!(canceled.iter().filter(|e| e["event"] == "window").count() < windows);

    // resumed at the first window's end: the same plan, the first window skipped, the rest as the whole run's
    let whole_windows: Vec<&serde_json::Value> =
        all.iter().filter(|e| e["event"] == "window").collect();
    let first_end = whole_windows[0]["data"]["end"].as_f64().unwrap();
    let resumed = serde_json::json!({ "path": path, "from_seconds": first_end }).to_string();
    let events = stream_events(
        model
            .transcription_stream_json(resumed.as_bytes(), None)
            .await?,
        None,
    )
    .await;
    assert_eq!(events[0]["data"]["windows"], all[0]["data"]["windows"]);
    let rest: Vec<&serde_json::Value> = events.iter().filter(|e| e["event"] == "window").collect();
    assert_eq!(rest.len(), whole_windows.len() - 1);
    for (resumed, whole) in rest.iter().zip(&whole_windows[1..]) {
        assert_eq!(resumed["data"], whole["data"]);
    }
    let done = &events.last().unwrap()["data"];
    assert!((done["duration"].as_f64().unwrap() - LONG_SECONDS).abs() < 1e-3);

    let both = model
        .transcription_stream_json(request.as_bytes(), Some(wav))
        .await;
    assert_eq!(
        both.err().map(|e| e.kind),
        Some(ApiErrorKind::InvalidRequest)
    );
    Ok(())
}
