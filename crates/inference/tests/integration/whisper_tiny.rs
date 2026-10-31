//! Whisper through the engine on a tiny random-weight checkpoint: each response format, a recording past one 30 s
//! window, and the language rules.

use inference::{
    ApiErrorKind, ModelDType, TimestampGranularity, TranscriptionModelBuilder,
    TranscriptionRequest, TranscriptionResponseFormat, VerboseTranscriptionResponse,
};

#[path = "../support/whisper_tiny.rs"]
mod support;
use support::tiny_whisper_checkpoint;

// off the model's 16 kHz, so the request is resampled
const RATE: u32 = 22_050;
const SECONDS: f64 = 2.5;
// past one 30 s window
const LONG_SECONDS: f64 = 45.0;
const CHIRP_START_HZ: f64 = 200.0;
const CHIRP_RISE_HZ: f64 = 300.0;
const AMPLITUDE: f64 = 0.4;
// the resampler's filter shortens the audio by a few samples
const DURATION_TOLERANCE: f64 = 0.01;
// the tiny checkpoint's languages, as Whisper names them
const LANGUAGES: [&str; 2] = ["english", "german"];

fn chirp_wav(seconds: f64) -> Vec<u8> {
    let n = (seconds * f64::from(RATE)) as usize;
    let pcm: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f64 / f64::from(RATE);
            let phase =
                2.0 * std::f64::consts::PI * (CHIRP_START_HZ * t + CHIRP_RISE_HZ * t * t / 2.0);
            (AMPLITUDE * phase.sin()) as f32
        })
        .collect();
    let mut wav = Vec::new();
    inference_models_speech::utils::write_pcm_as_wav(&mut wav, &pcm, RATE, 1).unwrap();
    wav
}

fn request(format: TranscriptionResponseFormat, language: Option<&str>) -> TranscriptionRequest {
    let mut request = TranscriptionRequest::new(format);
    request.timestamp_granularities =
        vec![TimestampGranularity::Word, TimestampGranularity::Segment];
    request.language = language.map(str::to_string);
    request
}

async fn model() -> anyhow::Result<(tempfile::TempDir, inference::Model)> {
    let checkpoint = tiny_whisper_checkpoint()?;
    let model = TranscriptionModelBuilder::new(checkpoint.path().to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .build()
        .await?;
    Ok((checkpoint, model))
}

fn check_words(parsed: &VerboseTranscriptionResponse, seconds: f64) {
    assert!(
        (parsed.duration - seconds).abs() < DURATION_TOLERANCE,
        "{}",
        parsed.duration
    );
    let words = parsed.words.as_deref().unwrap_or_default();
    for pair in words.windows(2) {
        assert!(
            pair[0].start <= pair[1].start,
            "{:?} then {:?}",
            pair[0],
            pair[1]
        );
    }
    for word in words {
        assert!(
            word.start <= word.end && word.end <= parsed.duration,
            "{word:?}"
        );
    }
}

// each format renders, the language is one the checkpoint knows, and words keep to the audio
#[tokio::test]
async fn a_clip_transcribes_into_each_format() -> anyhow::Result<()> {
    let (_checkpoint, model) = model().await?;
    let wav = chirp_wav(SECONDS);
    let verbose = model
        .transcription(
            request(TranscriptionResponseFormat::VerboseJson, None),
            &wav,
        )
        .await
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let parsed: VerboseTranscriptionResponse = serde_json::from_str(&verbose.body)?;
    check_words(&parsed, SECONDS);
    assert!(
        LANGUAGES.contains(&parsed.language.as_deref().unwrap_or_default()),
        "{:?}",
        parsed.language
    );
    for format in [
        TranscriptionResponseFormat::Json,
        TranscriptionResponseFormat::Text,
        TranscriptionResponseFormat::Srt,
        TranscriptionResponseFormat::Vtt,
    ] {
        model
            .transcription(request(format, Some("de")), &wav)
            .await
            .map_err(|e| anyhow::anyhow!("{format:?}: {e:?}"))?;
    }
    Ok(())
}

// a recording past one window decodes window by window to its end
#[tokio::test]
async fn a_long_recording_runs_window_by_window() -> anyhow::Result<()> {
    let (_checkpoint, model) = model().await?;
    let output = model
        .transcription(
            request(TranscriptionResponseFormat::VerboseJson, Some("en")),
            &chirp_wav(LONG_SECONDS),
        )
        .await
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let parsed: VerboseTranscriptionResponse = serde_json::from_str(&output.body)?;
    check_words(&parsed, LONG_SECONDS);
    assert_eq!(parsed.language.as_deref(), Some("english"));
    Ok(())
}

// a language the checkpoint has no token for is the request's error, by code or by name
#[tokio::test]
async fn an_unknown_language_is_an_invalid_request() -> anyhow::Result<()> {
    let (_checkpoint, model) = model().await?;
    let wav = chirp_wav(SECONDS);
    for language in ["fr", "klingon"] {
        let error = model
            .transcription(
                request(TranscriptionResponseFormat::Json, Some(language)),
                &wav,
            )
            .await
            .unwrap_err();
        assert_eq!(
            error.kind,
            ApiErrorKind::InvalidRequest,
            "{language}: {error:?}"
        );
    }
    model
        .transcription(
            request(TranscriptionResponseFormat::Json, Some("German")),
            &wav,
        )
        .await
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    Ok(())
}
