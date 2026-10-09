//! Kokoro through the engine on a tiny random-weight checkpoint: voices, speed, seeds, chunking and request errors.

use inference::{
    AudioResponseFormat, ModelBuilder, ModelDType, SpeechGenerationRequest, SpeechLoaderType,
    SpeechModelBuilder,
};

#[path = "../support/kokoro_tiny.rs"]
mod support;
use support::{VOICES, tiny_kokoro_checkpoint};

// invented phoneme strings over Kokoro's vocabulary
const PHONEMES: &str = "həlˈoʊ wˈɜːld.";
const SEED: u64 = 7;
const WAV_HEADER: usize = 44;
const SAMPLE_RATE: u32 = 24_000;
// the tiny config's context is 64 tokens, so this splits into several chunks
const LONG_WORDS: usize = 30;

fn request(phonemes: Option<&str>) -> SpeechGenerationRequest {
    SpeechGenerationRequest {
        phonemes: phonemes.map(str::to_string),
        seed: Some(SEED),
        ..SpeechGenerationRequest::new("hello world", AudioResponseFormat::Pcm)
    }
}

fn error_text<T, E: std::fmt::Display>(result: Result<T, E>) -> String {
    match result {
        Ok(_) => panic!("the request should have failed"),
        Err(err) => err.to_string(),
    }
}

fn samples(bytes: &[u8]) -> usize {
    bytes.len() / 2
}

#[tokio::test]
async fn kokoro_speaks_phonemes_with_voices_speeds_and_seeds() -> anyhow::Result<()> {
    let checkpoint = tiny_kokoro_checkpoint()?;
    let model = SpeechModelBuilder::new(
        checkpoint.path().to_string_lossy(),
        SpeechLoaderType::Kokoro,
    )
    .with_dtype(ModelDType::F32)
    .with_force_cpu()
    .build()
    .await?;

    let base = model.generate_speech(request(Some(PHONEMES))).await?;
    assert!(
        base.content_type.contains(&format!("rate={SAMPLE_RATE}")),
        "{}",
        base.content_type
    );
    assert!(samples(&base.bytes) > 0);
    assert_eq!(
        model.generate_speech(request(Some(PHONEMES))).await?.bytes,
        base.bytes
    );

    let reseeded = SpeechGenerationRequest {
        seed: Some(SEED + 1),
        ..request(Some(PHONEMES))
    };
    assert_ne!(model.generate_speech(reseeded).await?.bytes, base.bytes);

    // the default is the first voice, so naming it changes nothing and naming the other does
    let first = SpeechGenerationRequest {
        voice: Some(VOICES[0].into()),
        ..request(Some(PHONEMES))
    };
    assert_eq!(model.generate_speech(first).await?.bytes, base.bytes);
    let second = SpeechGenerationRequest {
        voice: Some(VOICES[1].into()),
        ..request(Some(PHONEMES))
    };
    assert_ne!(model.generate_speech(second).await?.bytes, base.bytes);
    let blend = SpeechGenerationRequest {
        voice: Some(VOICES.join(",")),
        ..request(Some(PHONEMES))
    };
    assert!(samples(&model.generate_speech(blend).await?.bytes) > 0);

    let slow = SpeechGenerationRequest {
        speed: Some(0.5),
        ..request(Some(PHONEMES))
    };
    assert!(samples(&model.generate_speech(slow).await?.bytes) >= samples(&base.bytes));

    let long = vec![PHONEMES; LONG_WORDS].join(" ");
    assert!(
        samples(&model.generate_speech(request(Some(&long))).await?.bytes) > samples(&base.bytes)
    );

    let wav = SpeechGenerationRequest {
        response_format: AudioResponseFormat::Wav,
        ..request(Some(PHONEMES))
    };
    assert_eq!(
        model.generate_speech(wav).await?.bytes.len(),
        base.bytes.len() + WAV_HEADER
    );

    let unknown = SpeechGenerationRequest {
        voice: Some("no_such_voice".into()),
        ..request(Some(PHONEMES))
    };
    let err = error_text(model.generate_speech(unknown).await);
    assert!(err.contains("no_such_voice"), "{err}");
    let err = error_text(model.generate_speech(request(None)).await);
    assert!(err.contains("phonemes"), "{err}");
    let bad_speed = SpeechGenerationRequest {
        speed: Some(0.),
        ..request(Some(PHONEMES))
    };
    assert!(model.generate_speech(bad_speed).await.is_err());
    Ok(())
}

// No architecture named: the auto loader reads config.json and picks Kokoro, giving the same audio
#[tokio::test]
async fn kokoro_is_detected_from_its_config() -> anyhow::Result<()> {
    let checkpoint = tiny_kokoro_checkpoint()?;
    let path = checkpoint.path().to_string_lossy();
    let detected = ModelBuilder::new(&path)
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .build()
        .await?;
    let named = SpeechModelBuilder::new(&path, SpeechLoaderType::Kokoro)
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .build()
        .await?;
    assert_eq!(
        detected
            .generate_speech(request(Some(PHONEMES)))
            .await?
            .bytes,
        named.generate_speech(request(Some(PHONEMES))).await?.bytes
    );
    Ok(())
}
