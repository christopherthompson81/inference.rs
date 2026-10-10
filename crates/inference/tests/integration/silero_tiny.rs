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
