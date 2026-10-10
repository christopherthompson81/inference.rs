//! Parakeet and the streaming Nemotron ASR through the engine on tiny random-weight checkpoints: each head, each
//! response format, the language prompt, request errors.

use inference::{
    ApiErrorKind, ModelDType, TimestampGranularity, TranscriptionModelBuilder,
    TranscriptionRequest, TranscriptionResponse, TranscriptionResponseFormat,
    VerboseTranscriptionResponse,
};

#[path = "../support/parakeet_tiny.rs"]
mod support;
use support::{HEADS, LANGUAGES, PROMPTED, tiny_parakeet_checkpoint, tiny_parakeet_nemo};

// off the model's 16 kHz, so the request is resampled
const RATE: u32 = 22_050;
const SECONDS: f64 = 1.5;
const CHIRP_START_HZ: f64 = 200.0;
const CHIRP_RISE_HZ: f64 = 600.0;
const AMPLITUDE: f64 = 0.4;
// past the encoder's 512-frame attention block, 41 s at 80 ms a frame
const LONG_SECONDS: f64 = 45.0;
const REPEATS: usize = 1;
// under the two 10 ms feature frames the model needs
const SHORT_SECONDS: f64 = 0.01;
// the resampler's filter shortens the audio by a few samples
const DURATION_TOLERANCE: f64 = 0.01;

fn chirp_wav() -> Vec<u8> {
    chirp_wav_of(SECONDS)
}

fn chirp_wav_of(seconds: f64) -> Vec<u8> {
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

fn request(format: TranscriptionResponseFormat) -> TranscriptionRequest {
    let mut request = TranscriptionRequest::new(format);
    request.timestamp_granularities =
        vec![TimestampGranularity::Word, TimestampGranularity::Segment];
    request
}

#[tokio::test]
async fn each_head_transcribes_into_each_format() -> anyhow::Result<()> {
    let wav = chirp_wav();
    for head in HEADS {
        let checkpoint = tiny_parakeet_checkpoint(head)?;
        let model = TranscriptionModelBuilder::new(checkpoint.path().to_string_lossy())
            .with_dtype(ModelDType::F32)
            .with_force_cpu()
            .build()
            .await?;

        let verbose = model
            .transcription(request(TranscriptionResponseFormat::VerboseJson), &wav)
            .await?;
        assert_eq!(verbose.content_type, "application/json", "{head}");
        let parsed: VerboseTranscriptionResponse = serde_json::from_str(&verbose.body)?;
        assert!(
            (parsed.duration - SECONDS).abs() < DURATION_TOLERANCE,
            "{head}: {}",
            parsed.duration
        );
        let words = parsed.words.as_deref().unwrap_or_default();
        // the seeded weights emit words under every head, so the timing checks below have something to check
        assert!(!words.is_empty(), "{head}");
        assert!(words.windows(2).all(|w| w[0].start <= w[1].start), "{head}");
        assert!(
            words
                .iter()
                .all(|w| w.start <= w.end && w.end <= parsed.duration + DURATION_TOLERANCE),
            "{head}"
        );
        let joined = words
            .iter()
            .map(|w| w.word.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(joined, parsed.text, "{head}");
        let segmented = parsed
            .segments
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(segmented, parsed.text, "{head}");

        let again = model
            .transcription(request(TranscriptionResponseFormat::VerboseJson), &wav)
            .await?;
        assert_eq!(
            again.body, verbose.body,
            "{head}: a repeat transcribes differently"
        );

        let text = model
            .transcription(request(TranscriptionResponseFormat::Text), &wav)
            .await?;
        assert_eq!(text.body, parsed.text, "{head}");
        let json = model
            .transcription(request(TranscriptionResponseFormat::Json), &wav)
            .await?;
        let json: TranscriptionResponse = serde_json::from_str(&json.body)?;
        assert_eq!(json.text, parsed.text, "{head}");
        let vtt = model
            .transcription(request(TranscriptionResponseFormat::Vtt), &wav)
            .await?;
        assert!(vtt.body.starts_with("WEBVTT"), "{head}");
        let srt = model
            .transcription(request(TranscriptionResponseFormat::Srt), &wav)
            .await?;
        assert_eq!(
            srt.content_type, "application/x-subrip; charset=utf-8",
            "{head}"
        );
        assert_eq!(
            srt.body.matches(" --> ").count(),
            parsed.segments.len(),
            "{head}"
        );
    }
    Ok(())
}

// On the default device (CUDA under `--features cuda`), past one attention block, every head repeats exactly
#[tokio::test]
async fn each_head_repeats_exactly_on_the_default_device() -> anyhow::Result<()> {
    let wav = chirp_wav_of(LONG_SECONDS);
    for head in HEADS {
        let checkpoint = tiny_parakeet_checkpoint(head)?;
        let model = TranscriptionModelBuilder::new(checkpoint.path().to_string_lossy())
            .with_dtype(ModelDType::F32)
            .build()
            .await?;
        let first = model
            .transcription(request(TranscriptionResponseFormat::VerboseJson), &wav)
            .await?
            .body;
        for _ in 0..REPEATS {
            let again = model
                .transcription(request(TranscriptionResponseFormat::VerboseJson), &wav)
                .await?;
            assert_eq!(again.body, first, "{head}");
        }
    }
    Ok(())
}

#[tokio::test]
async fn undecodable_audio_is_an_invalid_request() -> anyhow::Result<()> {
    let checkpoint = tiny_parakeet_checkpoint(HEADS[0])?;
    let model = TranscriptionModelBuilder::new(checkpoint.path().to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .build()
        .await?;
    let err = model
        .transcription(
            request(TranscriptionResponseFormat::Json),
            b"not audio at all",
        )
        .await
        .expect_err("garbage bytes decoded");
    assert_eq!(err.kind, ApiErrorKind::InvalidRequest, "{err}");
    Ok(())
}

// requests that land in one engine step answer each on its own: a clip the model refuses fails only itself
#[tokio::test]
async fn too_short_audio_fails_only_its_own_request() -> anyhow::Result<()> {
    let checkpoint = tiny_parakeet_checkpoint(HEADS[0])?;
    let model = TranscriptionModelBuilder::new(checkpoint.path().to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .build()
        .await?;
    let good = chirp_wav();
    let short = chirp_wav_of(SHORT_SECONDS);
    let (a, b, c) = tokio::join!(
        model.transcription(request(TranscriptionResponseFormat::Text), &good),
        model.transcription(request(TranscriptionResponseFormat::Text), &short),
        model.transcription(request(TranscriptionResponseFormat::Text), &good),
    );
    assert_eq!(a?.body, c?.body);
    let err = b.expect_err("a clip shorter than two feature frames transcribed");
    assert_eq!(err.kind, ApiErrorKind::InvalidRequest, "{err}");
    assert!(err.message.contains("too short"), "{err}");
    Ok(())
}

// Nemotron-3.5 takes the request's language as its prompt: it is reported back, it conditions the encoder, and a
// language outside the checkpoint's dictionary is refused for its own request alone
#[tokio::test]
async fn a_prompted_model_takes_the_requested_language() -> anyhow::Result<()> {
    let checkpoint = tiny_parakeet_checkpoint(PROMPTED)?;
    let model = TranscriptionModelBuilder::new(checkpoint.path().to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .build()
        .await?;
    let wav = chirp_wav();
    let mut transcribed = Vec::new();
    for (language, _) in &LANGUAGES[..2] {
        let mut asked = request(TranscriptionResponseFormat::VerboseJson);
        asked.language = Some(language.to_string());
        let out = model.transcription(asked, &wav).await?;
        let parsed: VerboseTranscriptionResponse = serde_json::from_str(&out.body)?;
        assert_eq!(parsed.language.as_deref(), Some(*language));
        transcribed.push(parsed.text);
    }
    assert_ne!(
        transcribed[0], transcribed[1],
        "the prompt does not reach the encoder"
    );

    let mut unknown = request(TranscriptionResponseFormat::Text);
    unknown.language = Some("tlh".to_string());
    let err = model
        .transcription(unknown, &wav)
        .await
        .expect_err("a language outside the dictionary transcribed");
    assert_eq!(err.kind, ApiErrorKind::InvalidRequest, "{err}");
    assert!(err.message.contains("de-DE"), "{err}");
    Ok(())
}

// a `.nemo` of the same weights transcribes as the transformers layout does, for every head it carries
#[tokio::test]
async fn a_nemo_checkpoint_transcribes_as_its_transformers_layout() -> anyhow::Result<()> {
    let wav = chirp_wav();
    for head in HEADS.iter().filter(|h| **h != PROMPTED) {
        let (dir, nemo) = tiny_parakeet_nemo(head)?;
        let mut bodies = Vec::new();
        for model_id in [dir.path().to_path_buf(), nemo] {
            let model = TranscriptionModelBuilder::new(model_id.to_string_lossy())
                .with_dtype(ModelDType::F32)
                .with_force_cpu()
                .build()
                .await?;
            let out = model
                .transcription(request(TranscriptionResponseFormat::VerboseJson), &wav)
                .await?;
            bodies.push(out.body);
        }
        assert_eq!(bodies[0], bodies[1], "{head}");
    }
    Ok(())
}
