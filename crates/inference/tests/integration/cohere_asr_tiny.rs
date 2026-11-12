//! Cohere Transcribe through the engine on a tiny random-weight checkpoint: the language it must be told (or a
//! language identifier names), chunked long audio, and the requests it refuses.

use inference::{
    ModelBuilder, ModelDType, TranscriptionModelBuilder, TranscriptionRequest,
    TranscriptionResponseFormat, VerboseTranscriptionResponse,
};

#[path = "../support/audiocpp_package.rs"]
mod package;
#[path = "../support/cohere_asr_tiny.rs"]
mod support;
use support::tiny_cohere_asr_checkpoint;

#[path = "../support/voxlingua_tiny.rs"]
#[allow(clippy::duplicate_mod)]
mod lid_support;

// off the model's 16 kHz, so the request is resampled
const RATE: u32 = 22_050;
// past the fixture's 4 s chunks, with a silence in each one's last second to cut at
const LONG_SECONDS: f64 = 9.0;
const SILENCE_EVERY_SECONDS: f64 = 3.5;
const SILENCE_SECONDS: f64 = 0.3;
const TONE_HZ: f64 = 300.0;
const AMPLITUDE: f64 = 0.4;
// the resampler's filter shortens the audio by a few samples
const DURATION_TOLERANCE: f64 = 0.01;

fn tone_wav(seconds: f64) -> Vec<u8> {
    let pcm: Vec<f32> = (0..(seconds * f64::from(RATE)) as usize)
        .map(|i| {
            let t = i as f64 / f64::from(RATE);
            let silent = t % SILENCE_EVERY_SECONDS > SILENCE_EVERY_SECONDS - SILENCE_SECONDS;
            let amplitude = if silent { 0.0 } else { AMPLITUDE };
            (amplitude * (2.0 * std::f64::consts::PI * TONE_HZ * t).sin()) as f32
        })
        .collect();
    let mut wav = Vec::new();
    inference_models_speech::utils::write_pcm_as_wav(&mut wav, &pcm, RATE, 1).unwrap();
    wav
}

fn request(language: Option<&str>) -> TranscriptionRequest {
    TranscriptionRequest {
        language: language.map(str::to_string),
        ..TranscriptionRequest::new(TranscriptionResponseFormat::VerboseJson)
    }
}

// told a language, a recording past one chunk transcribes (the same twice), its language echoed; no language, one it
// does not take, a translation or a beam is refused
#[tokio::test]
async fn cohere_transcribe_needs_its_language() -> anyhow::Result<()> {
    let checkpoint = tiny_cohere_asr_checkpoint()?;
    let model = TranscriptionModelBuilder::new(checkpoint.path().to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .build()
        .await?;
    let wav = tone_wav(LONG_SECONDS);
    let parse = |body: &str| -> anyhow::Result<VerboseTranscriptionResponse> {
        Ok(serde_json::from_str(body)?)
    };
    let first = parse(&model.transcription(request(Some("de")), &wav).await?.body)?;
    assert_eq!(first.language.as_deref(), Some("de"));
    assert!(
        (first.duration - LONG_SECONDS).abs() < DURATION_TOLERANCE,
        "{}",
        first.duration
    );
    let again = parse(
        &model
            .transcription(request(Some("de-DE")), &wav)
            .await?
            .body,
    )?;
    assert_eq!(first.text, again.text);

    let refused = |request: TranscriptionRequest| {
        let model = &model;
        let wav = &wav;
        async move { model.transcription(request, wav).await.unwrap_err().message }
    };
    assert!(refused(request(None)).await.contains("language_model"));
    assert!(
        refused(request(Some("fr")))
            .await
            .contains("does not transcribe")
    );
    let mut beam = request(Some("en"));
    beam.beam_size = Some(2);
    assert!(refused(beam).await.contains("greedily"));
    let mut translate = request(Some("en"));
    translate.task = inference::TranscriptionTask::Translate;
    assert!(refused(translate).await.contains("translate"));
    Ok(())
}

// a language identifier beside it names the language it was not told
#[tokio::test]
async fn a_language_model_tells_cohere_transcribe_its_language() -> anyhow::Result<()> {
    let checkpoint = tiny_cohere_asr_checkpoint()?;
    let lid = lid_support::tiny_language_id(&[("ja", "Japanese"), ("en", "English")])?;
    let model = inference::MultiModelBuilder::new()
        .add_model_with_alias(
            "asr",
            ModelBuilder::new(checkpoint.path().to_string_lossy())
                .with_dtype(ModelDType::F32)
                .with_force_cpu(),
        )
        .add_model_with_alias(
            "lid",
            ModelBuilder::new(lid.path().to_string_lossy()).with_force_cpu(),
        )
        .with_default_model("asr")
        .build()
        .await?;
    let mut identified = request(None);
    identified.language_model = Some("lid".to_string());
    let body = model.transcription(identified, &tone_wav(2.0)).await?.body;
    let parsed: VerboseTranscriptionResponse = serde_json::from_str(&body)?;
    assert_eq!(parsed.language.as_deref(), Some("ja"));
    Ok(())
}

// repacked as an audio.cpp GGUF, its encoder read under Parakeet's names, it transcribes as its safetensors
#[tokio::test]
async fn an_audiocpp_package_transcribes_as_its_checkpoint() -> anyhow::Result<()> {
    let checkpoint = tiny_cohere_asr_checkpoint()?;
    let dir = tempfile::tempdir()?;
    let gguf = package::repack_all(
        checkpoint.path(),
        "cohere_asr",
        &[("", "model.safetensors")],
        &dir.path().join("tiny-f32.gguf"),
    )?;
    let wav = tone_wav(LONG_SECONDS);
    let mut bodies = Vec::new();
    for model_id in [checkpoint.path(), gguf.as_path()] {
        let model = TranscriptionModelBuilder::new(model_id.to_string_lossy())
            .with_dtype(ModelDType::F32)
            .with_force_cpu()
            .build()
            .await?;
        bodies.push(model.transcription(request(Some("de")), &wav).await?.body);
    }
    assert_eq!(bodies[1], bodies[0]);
    Ok(())
}
