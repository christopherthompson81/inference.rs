//! A tiny random-weight SpeechBrain language identifier through the engine, alone and naming a Qwen3-ASR
//! transcription's language.

use inference::{
    ApiErrorKind, LanguageIdModelBuilder, LanguageIdRequest, ModelBuilder, ModelDType,
    MultimodalModelBuilder, TranscriptionRequest, TranscriptionResponseFormat,
};

#[path = "../support/voxlingua_tiny.rs"]
mod support;
use support::{EMBEDDING_DIM, tiny_language_id};

#[path = "../support/qwen3_asr_tiny.rs"]
#[allow(clippy::duplicate_mod)]
mod asr_support;

const RATE: u32 = 16_000;
const SECONDS: f64 = 3.0;
const TONE_HZ: f32 = 220.0;
const DEFAULT_TOP_K: usize = 5;
const LANGUAGES: [(&str, &str); 6] = [
    ("de", "German"),
    ("en", "English"),
    ("fr", "French"),
    ("es", "Spanish"),
    ("it", "Italian"),
    ("nl", "Dutch"),
];
// a code no recognizer takes, first so it wins
const UNKNOWN: [(&str, &str); 2] = [("zz", "Unlisted"), ("en", "English")];
const PROBABILITY_SUM_TOLERANCE: f32 = 1e-4;

fn tone_wav() -> Vec<u8> {
    let pcm: Vec<f32> = (0..(SECONDS * f64::from(RATE)) as usize)
        .map(|i| (2.0 * std::f32::consts::PI * TONE_HZ * i as f32 / RATE as f32).sin() * 0.3)
        .collect();
    let mut wav = Vec::new();
    inference_models_speech::utils::write_pcm_as_wav(&mut wav, &pcm, RATE, 1).unwrap();
    wav
}

// the likeliest five by default, all with a zero top_k, the embedding on request, and audio that does not decode refused
#[tokio::test]
async fn the_identifier_ranks_the_languages() -> anyhow::Result<()> {
    let dir = tiny_language_id(&LANGUAGES)?;
    let model = LanguageIdModelBuilder::new(dir.path().to_string_lossy())
        .with_force_cpu()
        .build()
        .await?;
    let wav = tone_wav();
    let identified = model.language_id(LanguageIdRequest::new(), &wav).await?;
    assert_eq!(identified.language, LANGUAGES[0].0);
    assert_eq!(identified.languages.len(), DEFAULT_TOP_K);
    assert_eq!(identified.languages[0].name, LANGUAGES[0].1);
    assert!(!identified.ambiguous, "{identified:?}");
    assert!(identified.embedding.is_none());
    assert!((identified.duration - SECONDS).abs() < 1e-6);
    assert!(
        identified
            .languages
            .windows(2)
            .all(|pair| pair[0].probability >= pair[1].probability)
    );

    let request = LanguageIdRequest {
        top_k: Some(0),
        return_embedding: true,
        ..LanguageIdRequest::new()
    };
    let all = model.language_id(request, &wav).await?;
    assert_eq!(all.languages.len(), LANGUAGES.len());
    let sum: f32 = all.languages.iter().map(|l| l.probability).sum();
    assert!((sum - 1.0).abs() < PROBABILITY_SUM_TOLERANCE, "{sum}");
    assert_eq!(all.embedding.map(|e| e.len()), Some(EMBEDDING_DIM));

    let error = model
        .language_id(LanguageIdRequest::new(), b"not audio")
        .await
        .unwrap_err();
    assert_eq!(error.kind, ApiErrorKind::InvalidRequest, "{error:?}");
    Ok(())
}

fn verbose(language_model: Option<&str>, language: Option<&str>) -> TranscriptionRequest {
    TranscriptionRequest {
        language: language.map(str::to_string),
        language_model: language_model.map(str::to_string),
        ..TranscriptionRequest::new(TranscriptionResponseFormat::VerboseJson)
    }
}

// an unset language comes from the identifier and is echoed; one the recognizer does not take transcribes as unset; a
// language the request names wins; a model that does not identify languages is the request's error
#[tokio::test]
async fn a_language_model_names_the_transcription_language() -> anyhow::Result<()> {
    let transcriber = asr_support::tiny_checkpoint()?;
    let known = tiny_language_id(&LANGUAGES)?;
    let unknown = tiny_language_id(&UNKNOWN)?;
    let model = inference::MultiModelBuilder::new()
        .add_model_with_alias(
            "asr",
            MultimodalModelBuilder::new(transcriber.path().to_string_lossy())
                .with_dtype(ModelDType::F32)
                .with_force_cpu(),
        )
        .add_model_with_alias(
            "lid",
            ModelBuilder::new(known.path().to_string_lossy()).with_force_cpu(),
        )
        .add_model_with_alias(
            "unknown",
            ModelBuilder::new(unknown.path().to_string_lossy()).with_force_cpu(),
        )
        .with_default_model("asr")
        .build()
        .await?;
    let wav = tone_wav();
    let language = |body: &str| -> anyhow::Result<Option<String>> {
        let body: serde_json::Value = serde_json::from_str(body)?;
        Ok(body["language"].as_str().map(str::to_string))
    };
    let identified = model
        .transcription(verbose(Some("lid"), None), &wav)
        .await?;
    assert_eq!(language(&identified.body)?.as_deref(), Some(LANGUAGES[0].0));

    let fallback = model
        .transcription(verbose(Some("unknown"), None), &wav)
        .await?;
    assert_ne!(language(&fallback.body)?.as_deref(), Some(UNKNOWN[0].0));

    let named = model
        .transcription(verbose(Some("lid"), Some("en")), &wav)
        .await?;
    assert_eq!(language(&named.body)?.as_deref(), Some("en"));

    let error = model
        .transcription(verbose(Some("asr"), None), &wav)
        .await
        .unwrap_err();
    assert_eq!(error.param.as_deref(), Some("language_model"), "{error:?}");
    Ok(())
}
