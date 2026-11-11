//! Qwen3-TTS through the engine on a tiny random-weight checkpoint: preset voices, languages, instructions, seeds,
//! detection from the config, and request errors.

use inference::{
    AudioResponseFormat, ModelBuilder, ModelDType, SpeechGenerationRequest, SpeechGenerationSpec,
    SpeechLoaderType, SpeechModelBuilder,
};

#[path = "../support/qwen3_tts_tiny.rs"]
mod support;
use support::{LANGUAGES, SAMPLES_PER_FRAME, SPEAKERS, tiny_qwen3_tts_checkpoint};

const TEXT: &str = "Hello world, this is a test.";
const SEED: u64 = 7;
const MAX_FRAMES: usize = 12;
const PCM_BYTES: usize = 2;
const CUSTOM_VOICE: &str = "custom_voice";
const VOICE_DESIGN: &str = "voice_design";
const SMALL: &str = "0b6";
const LARGE: &str = "1b7";
const INSTRUCTIONS: &str = "A calm, low voice, speaking slowly.";

fn request(voice: Option<&str>) -> SpeechGenerationRequest {
    SpeechGenerationRequest {
        voice: voice.map(str::to_string),
        language: Some(LANGUAGES[0].to_string()),
        seed: Some(SEED),
        ..SpeechGenerationRequest::new(TEXT, AudioResponseFormat::Pcm)
    }
}

async fn load(path: &std::path::Path) -> anyhow::Result<inference::Model> {
    Ok(
        SpeechModelBuilder::new(path.to_string_lossy(), SpeechLoaderType::Qwen3Tts)
            .with_dtype(ModelDType::F32)
            .with_force_cpu()
            .with_generation(SpeechGenerationSpec {
                max_tokens: Some(MAX_FRAMES),
                ..SpeechGenerationSpec::default()
            })
            .build()
            .await?,
    )
}

fn error_text<T, E: std::fmt::Display>(result: Result<T, E>) -> String {
    match result {
        Ok(_) => panic!("the request should have failed"),
        Err(err) => err.to_string(),
    }
}

// whole frames of audio, at most the configured cap
fn assert_frames(bytes: &[u8]) {
    let samples = bytes.len() / PCM_BYTES;
    assert!(samples > 0, "no audio");
    assert_eq!(samples % SAMPLES_PER_FRAME, 0, "{samples} samples");
    assert!(
        samples <= MAX_FRAMES * SAMPLES_PER_FRAME,
        "{samples} samples"
    );
}

#[tokio::test]
async fn qwen3_tts_speaks_preset_voices_with_seeds() -> anyhow::Result<()> {
    let checkpoint = tiny_qwen3_tts_checkpoint(CUSTOM_VOICE, SMALL)?;
    let model = load(checkpoint.path()).await?;
    let base = model.generate_speech(request(Some(SPEAKERS[0]))).await?;
    assert!(
        base.content_type.contains("rate=24000"),
        "{}",
        base.content_type
    );
    assert_frames(&base.bytes);
    assert_eq!(
        model
            .generate_speech(request(Some(SPEAKERS[0])))
            .await?
            .bytes,
        base.bytes
    );

    let reseeded = SpeechGenerationRequest {
        seed: Some(SEED + 1),
        ..request(Some(SPEAKERS[0]))
    };
    assert_ne!(model.generate_speech(reseeded).await?.bytes, base.bytes);
    // voice names are case-insensitive, as the reference's are
    let upper = request(Some(&SPEAKERS[0].to_uppercase()));
    assert_eq!(model.generate_speech(upper).await?.bytes, base.bytes);
    let other = model.generate_speech(request(Some(SPEAKERS[1]))).await?;
    assert_frames(&other.bytes);
    assert_ne!(other.bytes, base.bytes);
    // no language lets the model pick
    let auto = SpeechGenerationRequest {
        language: None,
        ..request(Some(SPEAKERS[0]))
    };
    assert_ne!(model.generate_speech(auto).await?.bytes, base.bytes);
    Ok(())
}

#[tokio::test]
async fn qwen3_tts_refuses_what_the_checkpoint_cannot_do() -> anyhow::Result<()> {
    let checkpoint = tiny_qwen3_tts_checkpoint(CUSTOM_VOICE, SMALL)?;
    let model = load(checkpoint.path()).await?;
    let unknown = error_text(model.generate_speech(request(Some("nobody"))).await);
    assert!(
        unknown.contains("unknown voice `nobody`") && unknown.contains(SPEAKERS[1]),
        "{unknown}"
    );
    let missing = error_text(model.generate_speech(request(None)).await);
    assert!(missing.contains("needs a `voice`"), "{missing}");
    let language = SpeechGenerationRequest {
        language: Some("klingon".to_string()),
        ..request(Some(SPEAKERS[0]))
    };
    let language = error_text(model.generate_speech(language).await);
    assert!(
        language.contains("unknown language `klingon`"),
        "{language}"
    );
    let instructed = SpeechGenerationRequest {
        instructions: Some(INSTRUCTIONS.to_string()),
        ..request(Some(SPEAKERS[0]))
    };
    let instructed = error_text(model.generate_speech(instructed).await);
    assert!(
        instructed.contains("take no `instructions`"),
        "{instructed}"
    );
    let phonemes = SpeechGenerationRequest {
        phonemes: Some("hɛloʊ".to_string()),
        ..request(Some(SPEAKERS[0]))
    };
    let phonemes = error_text(model.generate_speech(phonemes).await);
    assert!(phonemes.contains("does not take `phonemes`"), "{phonemes}");
    Ok(())
}

// A 1.7B-style checkpoint takes instructions; a voice design one needs them and ignores the voice
#[tokio::test]
async fn qwen3_tts_follows_instructions() -> anyhow::Result<()> {
    let checkpoint = tiny_qwen3_tts_checkpoint(CUSTOM_VOICE, LARGE)?;
    let model = load(checkpoint.path()).await?;
    let plain = model.generate_speech(request(Some(SPEAKERS[0]))).await?;
    let instructed = SpeechGenerationRequest {
        instructions: Some(INSTRUCTIONS.to_string()),
        ..request(Some(SPEAKERS[0]))
    };
    let instructed = model.generate_speech(instructed).await?;
    assert_frames(&instructed.bytes);
    assert_ne!(instructed.bytes, plain.bytes);

    let checkpoint = tiny_qwen3_tts_checkpoint(VOICE_DESIGN, LARGE)?;
    let model = load(checkpoint.path()).await?;
    let missing = error_text(model.generate_speech(request(Some(SPEAKERS[0]))).await);
    assert!(missing.contains("needs `instructions`"), "{missing}");
    let designed = |voice: Option<&str>| SpeechGenerationRequest {
        instructions: Some(INSTRUCTIONS.to_string()),
        ..request(voice)
    };
    let designed_speech = model.generate_speech(designed(Some("alloy"))).await?;
    assert_frames(&designed_speech.bytes);
    assert_eq!(
        model.generate_speech(designed(None)).await?.bytes,
        designed_speech.bytes
    );
    Ok(())
}

// No architecture named: the auto loader reads config.json and picks Qwen3-TTS
#[tokio::test]
async fn qwen3_tts_is_detected_from_its_config() -> anyhow::Result<()> {
    let checkpoint = tiny_qwen3_tts_checkpoint(CUSTOM_VOICE, SMALL)?;
    let detected = ModelBuilder::new(checkpoint.path().to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .build()
        .await?;
    let speech = detected.generate_speech(request(Some(SPEAKERS[0]))).await?;
    assert!(!speech.bytes.is_empty());
    Ok(())
}
